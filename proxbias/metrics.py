from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from efaar_benchmarking.constants import BENCHMARK_SOURCES, N_NULL_SAMPLES, RANDOM_SEED
from efaar_benchmarking.utils import (
    generate_null_cossims,
    generate_query_cossims,
    get_benchmark_data,
    get_feats_w_indices,
)
from numba import jit  # type: ignore[attr-defined]
from numba.typed import List as NumbaList
from scipy.stats import combine_pvalues, spearmanr
from sklearn.metrics.pairwise import cosine_similarity as sk_cossim
from sklearn.utils import Bunch
from statsmodels.stats.nonparametric import rank_compare_2indep

from proxbias.utils.chromosome_info import get_chromosome_info_as_dfs, get_chromosome_info_as_dicts
from proxbias.utils.constants import ARMS_ORD
from proxbias.utils.cosine_similarity import cosine_similarity


def _monte_carlo_brunner_munzel(
    population_a_samples: np.ndarray,
    population_b_samples: np.ndarray,
    combined: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.float32, np.float32]]:
    n_trials = population_a_samples.shape[0]
    probability_a_greater = []
    pvalues = []

    for i in range(n_trials):
        trial_samples_a = population_a_samples[i, :]
        trial_samples_b = population_b_samples[i, :]

        rank_compare_result = rank_compare_2indep(trial_samples_a, trial_samples_b, use_t=False)
        probability_a_greater.append(rank_compare_result.prob1.copy())
        pvalues.append(rank_compare_result.pvalue.copy())

    if combined:
        combined_prob = np.mean(probability_a_greater)
        _, combined_pvalue = combine_pvalues(pvalues, method="fisher")
        return combined_prob, combined_pvalue
    return np.array(probability_a_greater), np.array(pvalues)


def _prep_data(
    gene_df: pd.DataFrame,
    min_samples_in_arm: Optional[int] = None,
) -> Tuple[np.ndarray, pd.Series, List[np.ndarray]]:
    gene_df = gene_df.copy()
    gene_info, _, _ = get_chromosome_info_as_dfs()
    gene_info = gene_info.loc[gene_info.index.intersection(gene_df.index)].sort_values(  # type: ignore
        "chrom_arm_name", ascending=True
    )
    gene_info["chrom_arm_code"] = gene_info.chrom_arm_name.astype("category").cat.codes
    if min_samples_in_arm:
        seen_genes = gene_info.loc[gene_info.index.intersection(gene_df.index)]  # type: ignore
        gene_counts_by_arm = seen_genes.groupby(["chrom_arm_code"]).size()
        allowed_arms = gene_counts_by_arm.loc[gene_counts_by_arm > min_samples_in_arm].index
        allowed_genes = seen_genes.loc[seen_genes["chrom_arm_code"].isin(allowed_arms)].index
    else:
        allowed_genes = gene_df.index.intersection(gene_info.index)  # type: ignore
    gene_info = gene_info.loc[allowed_genes]
    gene_df = gene_df.loc[allowed_genes]

    gene_lookup = {gene_df.index[i]: i for i in range(len(gene_df.index))}
    gene_info.index.name = "gene_name"
    gene_info["gene_code"] = gene_info.index.map(lambda x: gene_lookup[x])
    gene_info = gene_info.reset_index().set_index("gene_code").sort_index()

    gigb = gene_info.groupby("chrom_arm_code")
    gene_codes_by_arm = gigb.apply(lambda x: x.index.to_numpy(dtype=np.int32)).to_list()  # type: ignore
    typed_gene_codes_by_arm = NumbaList()
    [typed_gene_codes_by_arm.append(x) for x in gene_codes_by_arm]
    cossims = sk_cossim(gene_df)
    gene_to_arm = gene_info.chrom_arm_code

    return cossims, gene_to_arm, typed_gene_codes_by_arm


@jit(fastmath=True, nopython=True)
def _get_intra_samples(
    cossims: np.ndarray,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    gene_to_arm: np.ndarray,
    genes_by_arm: List[np.ndarray],
) -> np.ndarray:
    """
    cossims - index / columns are actual gene name
    i_indices - ints
    j_indices - ints
    gene_info - index is gene_code, `chrom_arm_code` and `gene_name` are columns
    genes_by_arm - row is chrom_arm_code, columns correspond to `gene_codes` and `count`
    """
    original_shape = i_indices.shape
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()

    total_samples = len(i_flat)

    samples = np.empty(shape=total_samples, dtype=np.float64)
    for index in range(total_samples):
        i_index = i_flat[index]
        j_lookup = j_flat[index]
        arm_genes = genes_by_arm[gene_to_arm[i_index]]
        potentials = np.take(
            arm_genes,
            indices=np.asarray([j_lookup, j_lookup + 1], dtype=np.int32) % len(arm_genes),
        )

        samples[index] = cossims[i_index, potentials[0] if potentials[0] != i_index else potentials[1]]

    return samples.reshape(original_shape)


@jit(fastmath=True, nopython=True)
def _get_inter_samples(
    cossims: np.ndarray,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    gene_to_arm: np.ndarray,
    genes_by_arm: List[np.ndarray],
):
    original_shape = i_indices.shape
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()

    total_samples = len(i_flat)

    gene_set = set(np.arange(cossims.shape[0], dtype=np.int32))
    allowed_genes_by_arm = genes_by_arm.copy()
    for arm_idx in range(len(genes_by_arm)):
        allowed_genes_by_arm[arm_idx] = np.asarray(list(gene_set - set(genes_by_arm[arm_idx])))

    samples = np.empty(shape=total_samples, dtype=np.float64)
    for index in range(total_samples):
        i_index = i_flat[index]
        j_lookup = j_flat[index]
        i_arm = gene_to_arm[i_index]
        allowed_genes = allowed_genes_by_arm[i_arm]
        samples[index] = cossims[i_index, allowed_genes[j_lookup % len(allowed_genes)]]

    return samples.reshape(original_shape)


def genome_proximity_bias_score(
    gene_df: pd.DataFrame,
    n_trials: int = 200,
    n_samples: int = 500,
    seed: Optional[int] = None,
    return_samples: bool = True,
    combined: bool = True,
    min_samples_in_arm: int = 5,
) -> Union[
    Tuple[Union[np.ndarray, np.float32], Union[np.ndarray, np.float32], np.ndarray, np.ndarray],
    Tuple[Union[np.ndarray, np.float32], Union[np.ndarray, np.float32]],
]:
    cossims, gene_to_arm, genes_by_arm = _prep_data(gene_df, min_samples_in_arm=min_samples_in_arm)
    num_genes = cossims.shape[0]
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(num_genes, size=(4, n_trials, n_samples), dtype=np.int32)

    intra_arm_samples = _get_intra_samples(
        cossims,
        sample_indices[0].copy(),
        sample_indices[1].copy(),
        gene_to_arm.to_numpy(dtype=np.dtype(np.int32)),  # type: ignore
        genes_by_arm,
    )
    inter_arm_samples = _get_inter_samples(
        cossims,
        sample_indices[2].copy(),
        sample_indices[3].copy(),
        gene_to_arm.to_numpy(dtype=np.dtype(np.int32)),  # type: ignore
        genes_by_arm,
    )
    prob_intra_greater, pvalue = _monte_carlo_brunner_munzel(
        intra_arm_samples,
        inter_arm_samples,
        combined=combined,
    )
    if return_samples:
        return prob_intra_greater, pvalue, intra_arm_samples, inter_arm_samples
    return prob_intra_greater, pvalue


def bm_metrics(
    df: pd.DataFrame,
    arms_ord: list = ARMS_ORD,
    verbose: bool = False,
    sample_frac: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate Brunner-Munzel statistics for the whole genome and each chromosome arm
    Arms with less than 20 within-arm pairs are skipped.

    Inputs:
    -------
    - df: square dataframe with full-genome cosine similarities and matching index and columns.
          Index/columns should contain `chromosome_arm`
    - arms_ord: list or chromosome arm names in order. These should match names in the index/columns of df
    - verbose: whether to print progress
    - sample_frac: factor to downsample between-arm relationships in bigger datasets

    Outputs:
    --------
    - bm_all_df: dataframe of statistics for the whole genome tests
    - bm_per_arm_df: dataframe of statistics for each chromosome arm
    """
    bm_per_arm = {}
    all_within = []
    all_between = []

    for arm in arms_ord:
        arm_mat = df.query(f'chromosome_arm=="{arm}"')
        ind = np.array([x in arm_mat.index for x in arm_mat.columns])
        within = arm_mat.values[:, ind][np.triu_indices(arm_mat.shape[0], 1)]
        between = arm_mat.values[:, ~ind].flatten()
        if verbose:
            print(arm, within.shape, between.shape)
        within_l = len(within)
        between_l = len(between)
        if sample_frac < 1 and between_l > 10000:
            # Sample the between relationships to save memory
            between = np.random.choice(between, int(between_l * sample_frac))
        all_within.append(within)
        all_between.append(between)
        if within_l > 20 and between_l > 20:
            bm_result = rank_compare_2indep(within, between, use_t=False)
            bm_per_arm[arm] = (
                bm_result.statistic,
                bm_result.prob1,
                bm_result.test_prob_superior(alternative="larger").pvalue,
                within_l,
                between_l,
            )

    bm_per_arm_df = pd.DataFrame(bm_per_arm).T
    bm_per_arm_df.columns = ["stat", "prob", "pval", "n_within", "n_between"]  # type: ignore
    bm_per_arm_df = bm_per_arm_df.assign(bonf_p=bm_per_arm_df.pval * bm_per_arm_df.shape[0])

    all_w = np.concatenate(all_within)
    all_b = np.concatenate(all_between)
    bm_result = rank_compare_2indep(all_w, all_b, use_t=False)
    bm_all = {
        "stat": bm_result.statistic,
        "prob": bm_result.prob1,
        "pval": bm_result.test_prob_superior(alternative="larger").pvalue,
        "n_within": len(all_w),
        "n_between": len(all_b),
    }
    bm_all_df = pd.DataFrame(bm_all, index=["all"])

    return bm_all_df, bm_per_arm_df


def compute_gene_bm_metrics(
    df: pd.DataFrame,
    min_n_genes: int = 20,
) -> pd.DataFrame:
    """
    Compute the Brunner-Munzel statistic of intra- vs. inter-arm cosine similarities
      for each row in the dataframe, which should correspond to a gene.

    Inputs:
    -------
    - df: Embeddings for genes. Index should include the level `chromosome_arm`.
    - min_n_genes: Minimum number of genes on a given chromosome arm. Genes on
        arms with fewer genes will not be included in the results.

    Outputs:
    --------
    - bm_per_gene_df : DataFrame of Brunner-Munzel test results per gene.
    """
    bm_per_row = []
    index_level_names = df.index.names
    for chrom_arm, chrom_arm_df in df.groupby("chromosome_arm"):
        if chrom_arm_df.shape[0] >= min_n_genes:
            other_arms_df = df.query(f'chromosome_arm != "{chrom_arm!r}"')
            inter_cos_df = cosine_similarity(chrom_arm_df, other_arms_df)
            intra_cos_df = cosine_similarity(chrom_arm_df)
            for idx, row in chrom_arm_df.iterrows():  # type: ignore
                inter_cos = inter_cos_df.loc[idx].values  # type: ignore
                intra_cos = intra_cos_df.loc[idx].drop(idx).values  # type: ignore
                bm_result = rank_compare_2indep(intra_cos, inter_cos, use_t=False)
                bm_dict = {
                    "stat": bm_result.statistic,
                    "prob": bm_result.prob1,
                    "pval": bm_result.test_prob_superior(alternative="larger").pvalue,
                    "n_within": intra_cos.shape[0],
                    "n_between": inter_cos.shape[0],
                }
                for idx_level_name, idx_val in zip(index_level_names, idx):  # type: ignore
                    bm_dict[idx_level_name] = idx_val
                row_bm = pd.Series(bm_dict)
                bm_per_row.append(row_bm)
    bm_per_gene_df = pd.concat(bm_per_row, axis=1).T.set_index(index_level_names)
    return bm_per_gene_df


def compute_bm_centro_telo_rank_correlations(
    bm_per_gene_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman correlation between per-gene Brunner-Munzel statistics and
      centromere-to-telomere rank for each chromosome arm. Nominal and Bonferroni-
      corrected p-values are also computed, along with the negative correlation
      for plotting convenience.

    Inputs:
    -------
    - bm_per_gene_df : Dataframe with Brunner-Munzel statistics for each gene.
        Must also contain the index levels `chromosome_arm` and `gene_bp`.

    Outputs:
    --------
    - arm_corr_df : DataFrame of Spearman correlation results per arm
    - sample_sizes_table : Numbers of genes used for each arm
    """
    arm2corr = {}
    arm2sample_size = {}
    for chrom_arm, arm_df in bm_per_gene_df.groupby("chromosome_arm"):
        bm_stats = arm_df.sort_index(level="gene_bp", ascending=chrom_arm[-1] == "q").prob  # type: ignore
        n_genes = len(bm_stats)
        norm_ranks = np.arange(n_genes) / n_genes
        arm2sample_size[chrom_arm] = n_genes
        corr, p = spearmanr(norm_ranks, bm_stats)
        arm2corr[chrom_arm] = corr, p
    arm_corr_df = pd.DataFrame(arm2corr, index=["corr", "p"]).T
    arm_corr_df.index.name = "chromosome_arm"
    corr_sample_sizes = pd.Series(arm2sample_size)  # type: ignore
    sample_sizes_table = pd.DataFrame(corr_sample_sizes, columns=["sample size"])
    sample_sizes_table.index = [c.replace("chr", "") for c in sample_sizes_table.index]  # type: ignore
    sample_sizes_table["chromosome"] = [c.replace("p", "").replace("q", "") for c in sample_sizes_table.index]
    sample_sizes_table["arm"] = [c[-1] for c in sample_sizes_table.index]
    order = list(map(str, range(1, 23))) + ["X"]
    sample_sizes_table = (
        sample_sizes_table.reset_index(drop=True)
        .pivot(index="arm", columns="chromosome", values="sample size")
        .loc[:, order]
        .fillna(0)
        .astype(int)
        .astype(str)
        .replace("0", "-")
        .T
    )
    sorted_idx = sorted(arm_corr_df.index, key=lambda x: (int(x[3:-1].replace("X", "23").replace("Y", "24")), x[-1]))
    arm_corr_df = arm_corr_df.loc[sorted_idx]
    bonf_factor = arm_corr_df.shape[0]
    arm_corr_df["bonf_p"] = arm_corr_df.p * bonf_factor
    arm_corr_df["neg_corr"] = -1 * arm_corr_df["corr"]
    return arm_corr_df, sample_sizes_table


def _compute_recall(null_cossims, query_cossims, pct_thresholds) -> dict:
    null_sorted = np.sort(null_cossims)
    percentiles = np.searchsorted(null_sorted, query_cossims) / len(null_sorted)
    return sum((percentiles <= np.min(pct_thresholds)) | (percentiles >= np.max(pct_thresholds))) / len(percentiles)


def compute_within_cross_arm_pairwise_metrics(
    data: Bunch,
    pert_label_col: str = "gene",
    pct_thresholds: list = [0.05, 0.95],
) -> tuple:
    """Compute known biology benchmarks stratified by whether the pairs of genes
    are on the same chromosome arm or not.

    Parameters
    ----------
    data : Bunch
        Metadata-features bunch
    pert_label_col : str, optional
        Column in the metadata that defines the perturbation, by default "gene"
    pct_thresholds : list, optional
        Percentile thresholds for the recall computation, by default [0.05, 0.95]

    Returns
    -------
    tuple
        Results for within-arm and cross-arm pairs, respectively.
    """
    np.random.seed(RANDOM_SEED)
    within = {}
    between = {}
    for source in BENCHMARK_SOURCES:
        random_seed_pair = np.random.randint(2**32, size=2)
        gt_data = get_benchmark_data(source)
        gene_dict, _, _ = get_chromosome_info_as_dicts()

        feats = get_feats_w_indices(data, pert_label_col)

        gt_data["entity1_chrom"] = gt_data.entity1.apply(lambda x: gene_dict[x]["arm"] if x in gene_dict else "no info")
        gt_data["entity2_chrom"] = gt_data.entity2.apply(lambda x: gene_dict[x]["arm"] if x in gene_dict else "no info")
        gt_data = gt_data.query("entity1_chrom != 'no info' and entity2_chrom != 'no info'")
        df_gg_null = generate_null_cossims(
            feats,
            feats,
            rseed_entity1=random_seed_pair[0],
            rseed_entity2=random_seed_pair[1],
            n_entity1=N_NULL_SAMPLES,
            n_entity2=N_NULL_SAMPLES,
        )

        within_gt_subset = gt_data.query("entity1_chrom == entity2_chrom")
        between_gt_subset = gt_data.query("entity1_chrom != entity2_chrom")

        df_gg_within = generate_query_cossims(feats, feats, within_gt_subset)
        df_gg_between = generate_query_cossims(feats, feats, between_gt_subset)

        within[source] = _compute_recall(df_gg_null, df_gg_within, pct_thresholds)

        between[source] = _compute_recall(df_gg_null, df_gg_between, pct_thresholds)
    return within, between
