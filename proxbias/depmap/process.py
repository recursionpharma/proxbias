import concurrent.futures as cf
import multiprocessing as mp
import os
import time
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from proxbias.depmap.constants import CN_GAIN_CUTOFF, CN_LOSS_CUTOFF, COMPLETE_LOF_MUTATION_TYPES
from proxbias.depmap.load import center_gene_effects
from proxbias.metrics import genome_proximity_bias_score


def split_models(
    gene_symbol: str,
    candidate_models: List[str],
    cnv_data: pd.DataFrame,
    mutation_data: pd.DataFrame,
    cutoffs: Tuple[float, float] = (CN_LOSS_CUTOFF, CN_GAIN_CUTOFF),
    complete_only: bool = False,
    filter_amp: bool = False,
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Split models (cell lines) into 4 groups:
    - lof: models with a loss of function mutation for the gene of interest
    - wt_low_change: models with no mutation and normal number
    - amp: models with a copy number gain
    - mutant_low_change: models with a mutation and normal number

    Inputs:
    -------
    - gene_symbol: gene of interest
    - candidate_models: list of models to consider
    - cnv_data: copy number data from depmap
    - mutation_data: data on which models have which mutations on a per gene basis
    - cutoffs: tuple of cutoffs for copy number gain and loss
    - complete_only: whether to only consider models with a complete loss of function mutation
    - filter_amp: whether to remove models with a complete loss of function mutation from the amp group

    Returns:
    --------
    - lof: set of models with a loss of function mutation
    - wt_low_change: set of models with no mutation and normal number
    - amp: set of models with a copy number gain
    - mutant_low_change: set of models with a mutation and normal number
    """
    cnv_subset = pd.Series((np.power(2, cnv_data.loc[gene_symbol]) - 1) * 2)
    all_cnv = cnv_subset.loc[cnv_subset.index.intersection(candidate_models)]
    lof = set(candidate_models).intersection(all_cnv.loc[all_cnv < cutoffs[0]].index.unique())
    amp = set(candidate_models).intersection(all_cnv.loc[all_cnv >= cutoffs[1]].index.unique())
    mutants_for_gene = mutation_data.loc[mutation_data["HugoSymbol"] == gene_symbol]
    if complete_only:
        mutant_lines = (
            mutants_for_gene.loc[mutants_for_gene["VariantInfo"].isin(COMPLETE_LOF_MUTATION_TYPES), "ModelID"]
            .unique()
            .tolist()
        )
        mutants = set(candidate_models).intersection(mutant_lines)
        wt_low_change = set(candidate_models).difference(lof | amp | set(mutants))
        return mutants, wt_low_change, set(), set()
    if filter_amp:
        complete_lof_lines = (
            mutants_for_gene.loc[mutants_for_gene["VariantInfo"].isin(COMPLETE_LOF_MUTATION_TYPES), "ModelID"]
            .unique()
            .tolist()
        )
        amp = amp.difference(set(complete_lof_lines))

    mutant_lines = mutants_for_gene.loc[~mutants_for_gene["VariantInfo"].isna(), "ModelID"].unique().tolist()
    wild_type = list(set(candidate_models).difference(mutant_lines))

    mutant_low_change = set(candidate_models).difference(lof | amp | set(wild_type))
    wt_low_change = set(wild_type).difference(lof | amp)
    return lof, wt_low_change, amp, mutant_low_change


def _compute_stats_for_gene(
    gene_of_interest: str,
    dep_data: pd.DataFrame,
    cnv_data: pd.DataFrame,
    mutation_data: pd.DataFrame,
    candidate_models: List[str],
    model_sample_rate: float,
    search_mode: str,
    n_min_cell_lines: int,
    n_iterations: int,
    seed: int,
    cnv_cutoffs: Tuple[float, float],
    eval_function: Callable,
    eval_kwargs: Dict[str, Any],
    complete_lof: bool,
    filter_amp: bool,
    verbose: bool,
    fixed_cell_line_sampling: bool,
):
    start_gene_time = time.time()
    rng = np.random.default_rng(seed)
    lof, wt, amp, _ = split_models(
        gene_symbol=gene_of_interest,
        candidate_models=candidate_models,
        cnv_data=cnv_data,
        mutation_data=mutation_data,
        cutoffs=cnv_cutoffs,
        complete_only=complete_lof,
        filter_amp=filter_amp,
    )
    wt_columns = dep_data.columns.intersection(list(wt))
    test_columns = dep_data.columns.intersection(list(lof if search_mode == "lof" else amp))
    n_test = len(test_columns)
    n_wt = len(wt_columns)
    available_samples = min(n_test, n_wt)

    if available_samples < n_min_cell_lines:
        if verbose:
            print(f"Insufficient samples for {gene_of_interest}")
        return {}
    if fixed_cell_line_sampling:
        choose_n = int(n_min_cell_lines * model_sample_rate)
    else:
        choose_n = int(available_samples * model_sample_rate)
    test_stats = []
    wt_stats = []
    for _ in range(n_iterations):
        wt_deps = list(rng.choice(wt_columns, size=choose_n, replace=False))
        test_deps = list(rng.choice(test_columns, size=choose_n, replace=False))
        wt_df = dep_data.loc[:, wt_deps].copy()
        test_df = dep_data.loc[:, test_deps].copy()
        wt, _ = eval_function(wt_df, seed=rng.integers(low=0, high=9001, size=1)[0], **eval_kwargs)
        test, _ = eval_function(test_df, seed=rng.integers(low=0, high=9001, size=1)[0], **eval_kwargs)
        wt_stats.append(wt)
        test_stats.append(test)

    duration = time.time() - start_gene_time
    diff = np.array(test_stats).mean() - np.array(wt_stats).mean()
    print(f"Stats for {gene_of_interest} computed in {duration} - diff is {diff}, {n_wt} wt and {n_test} {search_mode}")
    return {
        "test_stats": test_stats,
        "test_mean": np.array(test_stats).mean(),
        "wt_stats": wt_stats,
        "wt_mean": np.array(wt_stats).mean(),
        "diff": diff,
        "search_mode": search_mode,
        "n_models": choose_n,
        "n_test": len(test_columns),
        "n_wt": len(wt_columns),
    }


def compute_monte_carlo_stats(
    genes_of_interest: List[str],
    dependency_data: pd.DataFrame,
    cnv_data: pd.DataFrame,
    mutation_data: pd.DataFrame,
    candidate_models: List[str],
    model_sample_rate: float = 0.8,
    search_mode: str = "lof",
    n_min_cell_lines: int = 25,
    n_iterations: int = 100,
    seed: int = 42,
    center_genes: bool = True,
    cnv_cutoffs: Tuple[float, float] = (CN_LOSS_CUTOFF, CN_GAIN_CUTOFF),
    eval_function: Callable = genome_proximity_bias_score,
    eval_kwargs: Dict[str, Any] = {"n_samples": 100, "n_trials": 50, "return_samples": False},
    complete_lof: bool = False,
    filter_amp: bool = False,
    verbose: bool = False,
    n_workers: int = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", 1)),
    fixed_cell_line_sampling: bool = False,
) -> pd.DataFrame:
    """
    Compute proximity bias scores for a list of genes of interest using a monte carlo approach

    Inputs:
    -------
    - genes_of_interest: list of genes to compute metrics over
    - dependency_data: dependency data from depmap
    - cnv_data: copy number data from depmap
    - mutation_data: data on which models have which mutations on a per gene basis
    - candidate_models: list of models to consider
    - model_sample_rate: fraction of models to sample for each sample
    - search_mode: whether to search for models with a loss of function mutation
        and decreased copy number (lof) or a copy number gain (amp)
    - n_min_cell_lines: minimum number of cell lines to use for each sample
    - n_iterations: number of iterations to perform
    - seed: random seed
    - center_genes: whether to center gene effects
    - cnv_cutoffs: tuple of cutoffs for copy number gain and loss
    - eval_function: function to use for evaluating proximity bias
    - eval_kwargs: keyword arguments to pass to `eval_function`
    - complete_lof: whether to only consider models with a complete loss of function mutation for the lof group
    - filter_amp: whether to remove models with a complete loss of function mutation from the amp group
    - verbose: whether to print progress
    - n_workers: number of workers to use for multiprocessing
    - fixed_cell_line_sampling: whether to sample the same number of cell lines for each iteration

    Returns:
    --------
    - df: dataframe with results
    """
    dep_data = dependency_data.loc[:, dependency_data.columns.intersection(candidate_models)].copy()  # type: ignore
    genes_of_interest_index = pd.Index(genes_of_interest, dtype=object)  # type: ignore
    # TODO: test if it's okay to do this here or if I need to do it for wt/test specifically
    if center_genes:
        dep_data = center_gene_effects(dep_data)

    available_genes = (
        dep_data.index.intersection(mutation_data.HugoSymbol.values)  # type: ignore
        .intersection(cnv_data.index)
        .intersection(genes_of_interest_index)
    )
    invalid_genes = genes_of_interest_index.difference(available_genes)
    if not invalid_genes.empty:  # type: ignore
        print(f"{invalid_genes} not found in data.")

    results = {}
    future_results = {}
    with cf.ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as executor:
        for gene_of_interest in available_genes:
            fut = executor.submit(
                _compute_stats_for_gene,
                gene_of_interest=gene_of_interest,
                candidate_models=candidate_models,
                dep_data=dep_data,
                cnv_data=cnv_data,
                mutation_data=mutation_data,
                cnv_cutoffs=cnv_cutoffs,
                complete_lof=complete_lof,
                filter_amp=filter_amp,
                verbose=verbose,
                search_mode=search_mode,
                model_sample_rate=model_sample_rate,
                n_min_cell_lines=n_min_cell_lines,
                n_iterations=n_iterations,
                seed=seed,
                eval_function=eval_function,
                eval_kwargs=eval_kwargs,
                fixed_cell_line_sampling=fixed_cell_line_sampling,
            )
            future_results[fut] = gene_of_interest
        for fut in cf.as_completed(future_results):
            gene_of_interest = future_results[fut]
            results[gene_of_interest] = fut.result()
    return pd.DataFrame.from_dict(results, orient="index")
