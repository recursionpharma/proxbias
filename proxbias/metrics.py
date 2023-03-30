from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import combine_pvalues
from statsmodels.stats.nonparametric import rank_compare_2indep

from proxbias.utils.chromosome_info import get_chromosome_info_as_dfs
from proxbias.utils.cosine_similarity import cosine_similarity


def _monte_carlo_brunner_munzel(
    population_a_samples: npt.NDArray,
    population_b_samples: npt.NDArray,
    combined: bool = True,
) -> Union[Tuple[npt.NDArray, npt.NDArray], Tuple[np.float32, np.float32],]:
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
    gene_df: pd.DataFrame, min_samples_in_arm: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_df = gene_df.copy()
    gene_info, _, _ = get_chromosome_info_as_dfs()
    seen_genes = gene_info.loc[gene_info.index.intersection(gene_df.index)]
    if min_samples_in_arm:
        gene_counts_by_arm = seen_genes.groupby(["chrom_arm_name"]).size()
        allowed_arms = gene_counts_by_arm.loc[gene_counts_by_arm > min_samples_in_arm].index
        allowed_genes = seen_genes.loc[seen_genes["chrom_arm_name"].isin(allowed_arms)].index
    else:
        allowed_genes = gene_df.index.intersection(gene_info.index)
    gene_info = gene_info.loc[allowed_genes]
    gene_df = gene_df.loc[allowed_genes]

    gigb = gene_info.groupby("chrom_arm_name")
    genes_by_arm = pd.concat([gigb.apply(lambda x: list(x.index)), gigb.size()], axis="columns").rename(
        columns={0: "genes", 1: "count"}
    )
    cossims = cosine_similarity(gene_df, index_names=("gene_A", "gene_B"))

    return cossims, gene_info, genes_by_arm


def _get_intra_samples(cossims, i_indices, j_indices, gene_info, genes_by_arm):
    original_shape = i_indices.shape
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()
    samples = np.zeros(shape=i_flat.shape)
    geneset_by_arm = {}
    count_by_arm = {}
    for arm in genes_by_arm.index:
        geneset_by_arm[arm] = genes_by_arm.loc[arm]["genes"]
        count_by_arm[arm] = genes_by_arm.loc[arm]["count"]

    for index in range(len(i_flat)):
        i_index = i_flat[index]
        j_lookup = j_flat[index]
        i_gene = gene_info.index[i_index]
        arm = gene_info["chrom_arm_name"].iloc[i_index]
        j_gene = geneset_by_arm[arm][j_lookup % count_by_arm[arm]]
        if i_gene == j_gene:
            # Don't allow same gene
            # TODO: this is the slowest part of the code
            j_gene = geneset_by_arm[arm][(j_lookup + 1) % count_by_arm[arm]]
        samples[index] = cossims.loc[i_gene, j_gene]
    return samples.reshape(original_shape)


def _get_inter_samples(cossims, i_indices, j_indices, gene_info, genes_by_arm):
    original_shape = i_indices.shape
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()
    samples = np.zeros(shape=i_flat.shape)
    allowed_genes_by_arm = {}
    for arm in genes_by_arm.index:
        allowed_genes_by_arm[arm] = gene_info.index.difference(genes_by_arm.loc[arm]["genes"])

    for index in range(len(i_flat)):
        i_index = i_flat[index]
        j_lookup = j_flat[index]
        i_gene = gene_info.index[i_index]
        prohibited_arm = gene_info["chrom_arm_name"].iloc[i_index]
        allowed_genes = allowed_genes_by_arm[prohibited_arm]
        j_gene = allowed_genes[j_lookup % len(allowed_genes)]
        samples[index] = cossims.loc[i_gene, j_gene]
    return samples.reshape(original_shape)


def genome_proximity_bias_score(
    gene_df: pd.DataFrame,
    n_trials: int = 200,
    n_samples: int = 500,
    seed: Optional[int] = None,
    return_samples: bool = True,
    combined: bool = True,
) -> Union[
    Tuple[Union[npt.NDArray, np.float32], Union[npt.NDArray, np.float32], npt.NDArray, npt.NDArray],
    Tuple[Union[npt.NDArray, np.float32], Union[npt.NDArray, np.float32]],
]:
    cossims, gene_info, genes_by_arm = _prep_data(gene_df)
    num_genes = len(cossims)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(num_genes, size=(4, n_trials, n_samples))

    # TODO: almost all of the time is spent in these sampling functions - make them faster
    intra_arm_samples = _get_intra_samples(
        cossims,
        sample_indices[0].copy(),
        sample_indices[1].copy(),
        gene_info,
        genes_by_arm,
    )
    inter_arm_samples = _get_inter_samples(
        cossims,
        sample_indices[2].copy(),
        sample_indices[3].copy(),
        gene_info,
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
