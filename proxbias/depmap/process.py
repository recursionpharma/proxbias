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
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    cnv_subset = pd.Series((np.power(2, cnv_data.loc[gene_symbol]) - 1) * 2)
    all_cnv = cnv_subset.loc[cnv_subset.index.intersection(candidate_models)]
    lof = set(candidate_models).intersection(all_cnv.loc[all_cnv < cutoffs[0]].index.unique())
    gof = set(candidate_models).intersection(all_cnv.loc[all_cnv >= cutoffs[1]].index.unique())
    mutants_for_gene = mutation_data.loc[mutation_data["HugoSymbol"] == gene_symbol]
    if complete_only:
        mutant_lines = (
            mutants_for_gene.loc[mutants_for_gene["VariantInfo"].isin(COMPLETE_LOF_MUTATION_TYPES), "ModelID"]
            .unique()
            .tolist()
        )
        mutants = set(candidate_models).intersection(mutant_lines)
        wt_low_change = set(candidate_models).difference(lof | gof | set(mutants))
        return mutants, wt_low_change, set(), set()

    mutant_lines = mutants_for_gene.loc[~mutants_for_gene["VariantInfo"].isna(), "ModelID"].unique().tolist()
    wild_type = list(set(candidate_models).difference(mutant_lines))

    mutant_low_change = set(candidate_models).difference(lof | gof | set(wild_type))
    wt_low_change = set(wild_type).difference(lof | gof)
    return lof, wt_low_change, gof, mutant_low_change


def _bootstrap_gene(
    gene_of_interest: str,
    dep_data: pd.DataFrame,
    cnv_data: pd.DataFrame,
    mutation_data: pd.DataFrame,
    candidate_models: List[str],
    model_sample_rate: float,
    search_mode: str,
    n_min_samples: int,
    n_bootstrap: int,
    seed: int,
    cnv_cutoffs: Tuple[float, float],
    eval_function: Callable,
    eval_kwargs: Dict[str, Any],
    complete_lof: bool,
    verbose: bool,
    n_workers: int = 1,
):
    start_gene_time = time.time()
    rng = np.random.default_rng(seed)
    lof, wt, gof, _ = split_models(
        gene_symbol=gene_of_interest,
        candidate_models=candidate_models,
        cnv_data=cnv_data,
        mutation_data=mutation_data,
        cutoffs=cnv_cutoffs,
        complete_only=complete_lof,
    )
    wt_columns = dep_data.columns.intersection(list(wt))
    test_columns = dep_data.columns.intersection(list(lof if search_mode == "lof" else gof))
    n_test = len(test_columns)
    n_wt = len(wt_columns)
    choose_n = int(min(n_test, n_wt) * model_sample_rate)

    if choose_n < n_min_samples:
        if verbose:
            print(f"Insufficient samples for {gene_of_interest}")
        return {}
    test_stats = []
    wt_stats = []
    futures_dict = {}
    with cf.ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as executor:
        for _ in range(n_bootstrap):
            wt_deps = rng.choice(wt_columns, size=choose_n, replace=False)
            test_deps = rng.choice(test_columns, size=choose_n, replace=False)
            wt_df = dep_data.loc[:, wt_deps]  # type: ignore
            test_df = dep_data.loc[:, test_deps]  # type: ignore
            fut_wt: cf.Future = executor.submit(
                eval_function, wt_df, seed=rng.integers(low=0, high=9001, size=1)[0], **eval_kwargs
            )
            fut_test: cf.Future = executor.submit(
                eval_function, test_df, seed=rng.integers(low=0, high=9001, size=1)[0], **eval_kwargs
            )
            futures_dict[fut_wt] = "wt"
            futures_dict[fut_test] = "test"
        for fut in cf.as_completed(futures_dict):
            fut_type = futures_dict[fut]
            stats, _ = fut.result()
            if fut_type == "wt":
                wt_stats.append(stats)
            else:
                test_stats.append(stats)

    duration = time.time() - start_gene_time
    diff = np.array(test_stats).mean() - np.array(wt_stats).mean()
    print(f"Stats for {gene_of_interest} computed in {duration} - diff is {diff}, {n_wt} wt and {n_test} test")
    return {
        "test_stats": test_stats,
        "wt_stats": wt_stats,
        "search_mode": search_mode,
        "n_sample_bootstrap": choose_n,
        "n_test": len(test_columns),
        "n_wt": len(wt_columns),
    }


def bootstrap_stats(
    genes_of_interest: List[str],
    dependency_data: pd.DataFrame,
    cnv_data: pd.DataFrame,
    mutation_data: pd.DataFrame,
    candidate_models: List[str],
    model_sample_rate: float = 0.8,
    search_mode: str = "lof",
    n_min_samples: int = 10,
    n_bootstrap: int = 100,
    seed: int = 42,
    center_genes: bool = True,
    cnv_cutoffs: Tuple[float, float] = (CN_LOSS_CUTOFF, CN_GAIN_CUTOFF),
    eval_function: Callable = genome_proximity_bias_score,
    eval_kwargs: Dict[str, Any] = {"n_samples": 100, "n_trials": 50, "return_samples": False},
    complete_lof: bool = False,
    verbose: bool = False,
    n_workers: int = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", 1)),
) -> pd.DataFrame:
    """
    gene of interest
    dependency data
    cnv data
    mutation data
    search mode. lof=loss of function, gof=gain of function
    min number of samples to start bootstraping, otherwise return empty dict
    number of bootstrap steps
    evaluation function
    evaluation function keyword arguments
    """
    dep_data = dependency_data.loc[:, dependency_data.columns.intersection(candidate_models)].copy()  # type: ignore
    genes_of_interest_index = pd.Index(genes_of_interest, dtype=object)
    # TODO: test if it's okay to do this here or if I need to do it for wt/test specifically
    if center_genes:
        dep_data = center_gene_effects(dep_data)

    available_genes = (
        dep_data.index.intersection(mutation_data.HugoSymbol.values)
        .intersection(cnv_data.index)
        .intersection(genes_of_interest_index)
    )
    invalid_genes = genes_of_interest_index.difference(available_genes)
    if not invalid_genes.empty:
        print(f"{invalid_genes} not found in data.")

    results = {}
    for gene_of_interest in available_genes:
        results[gene_of_interest] = _bootstrap_gene(
            gene_of_interest=gene_of_interest,
            candidate_models=candidate_models,
            dep_data=dep_data,
            cnv_data=cnv_data,
            mutation_data=mutation_data,
            cnv_cutoffs=cnv_cutoffs,
            complete_lof=complete_lof,
            verbose=verbose,
            search_mode=search_mode,
            model_sample_rate=model_sample_rate,
            n_min_samples=n_min_samples,
            n_bootstrap=n_bootstrap,
            seed=seed,
            eval_function=eval_function,
            eval_kwargs=eval_kwargs,
            n_workers=n_workers,
        )
    return pd.DataFrame(results)
