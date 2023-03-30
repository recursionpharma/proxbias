from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.stats.nonparametric import RankCompareResult, rank_compare_2indep

from proxbias.utils.cosine_similarity import add_gene_info, cosine_similarity


def monte_carlo_brunner_munzel(
    population_a: pd.DataFrame,
    population_b: pd.DataFrame,
    num_trials: int,
    num_samples: int,
) -> Tuple[List[np.float]]:
    pass


def _prep_data(gene_df):
    cossims = cosine_similarity(gene_df, as_long=True, triu=True, index_names=("gene_A", "gene_B"))
    cossims = add_gene_info(cossims)
    return cossims.loc[cossims["same_chromosome_arm"]], cossims.loc[~cossims["same_chromosome_arm"]]


def genome_proximity_bias_score(
    gene_df: pd.DataFrame,
    n_trials: int = 100,
    n_samples: int = 200,
    seed: int = 42,
    return_samples: bool = True,
) -> Union[np.float, Tuple[np.float32, np.float32, npt.ArrayLike, npt.ArrayLike]]:
    intra_arm, inter_arm = _prep_data(gene_df)


def chromosome_arm_proximity_bias_score(
    gene_df: pd.DataFrame,
    n_trials: int = 200,
    n_samples: int = 50,
    seed: int = 42,
    return_samples: bool = True,
) -> Union[np.float, Tuple[np.float32, np.float32, npt.ArrayLike, npt.ArrayLike]]:
    pass
