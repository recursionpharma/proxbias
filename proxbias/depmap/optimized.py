from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List as NumbaList
from scipy.stats import combine_pvalues
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.nonparametric import rank_compare_2indep

from proxbias.utils.chromosome_info import get_chromosome_info_as_dfs


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
    gene_info = gene_info.loc[gene_info.index.intersection(gene_df.index)].sort_values("chrom_arm_name", ascending=True)
    gene_info["chrom_arm_code"] = gene_info.chrom_arm_name.astype("category").cat.codes
    if min_samples_in_arm:
        seen_genes = gene_info.loc[gene_info.index.intersection(gene_df.index)]
        gene_counts_by_arm = seen_genes.groupby(["chrom_arm_code"]).size()
        allowed_arms = gene_counts_by_arm.loc[gene_counts_by_arm > min_samples_in_arm].index
        allowed_genes = seen_genes.loc[seen_genes["chrom_arm_code"].isin(allowed_arms)].index
    else:
        allowed_genes = gene_df.index.intersection(gene_info.index)
    gene_info = gene_info.loc[allowed_genes]
    gene_df = gene_df.loc[allowed_genes]

    gene_lookup = {gene_df.index[i]: i for i in range(len(gene_df.index))}
    gene_info.index.name = "gene_name"
    gene_info["gene_code"] = gene_info.index.map(lambda x: gene_lookup[x])
    gene_info = gene_info.reset_index().set_index("gene_code").sort_index()

    gigb = gene_info.groupby("chrom_arm_code")
    gene_codes_by_arm = gigb.apply(lambda x: x.index.to_numpy(dtype=np.int32)).to_list()
    typed_gene_codes_by_arm = NumbaList()
    [typed_gene_codes_by_arm.append(x) for x in gene_codes_by_arm]
    cossims = cosine_similarity(gene_df)

    gene_to_arm = gene_info.chrom_arm_code

    return cossims, gene_to_arm, typed_gene_codes_by_arm


@njit(fastmath=True)
def _get_intra_samples_jit(
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
    for index in prange(total_samples):
        i_index = i_flat[index]
        j_lookup = j_flat[index]
        arm_genes = genes_by_arm[gene_to_arm[i_index]]
        potentials = np.take(
            arm_genes,
            indices=np.asarray([j_lookup, j_lookup + 1], dtype=np.int32) % len(arm_genes),
        )

        samples[index] = cossims[i_index, potentials[0] if potentials[0] != i_index else potentials[1]]

    return samples.reshape(original_shape)


@njit(fastmath=True)
def _get_inter_samples_jit(
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
    for arm_idx in prange(len(genes_by_arm)):
        allowed_genes_by_arm[arm_idx] = np.asarray(list(gene_set - set(genes_by_arm[arm_idx])))

    samples = np.empty(shape=total_samples, dtype=np.float64)
    for index in range(total_samples):
        i_index = i_flat[index]
        j_lookup = j_flat[index]
        i_arm = gene_to_arm[i_index]
        allowed_genes = allowed_genes_by_arm[i_arm]
        samples[index] = cossims[i_index, allowed_genes[j_lookup % len(allowed_genes)]]

    return samples.reshape(original_shape)


def genome_proximity_bias_score_fast(
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

    intra_arm_samples = _get_intra_samples_jit(
        cossims,
        sample_indices[0].copy(),
        sample_indices[1].copy(),
        gene_to_arm.to_numpy(dtype=np.int32),
        genes_by_arm,
    )
    inter_arm_samples = _get_inter_samples_jit(
        cossims,
        sample_indices[2].copy(),
        sample_indices[3].copy(),
        gene_to_arm.to_numpy(dtype=np.int32),
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
