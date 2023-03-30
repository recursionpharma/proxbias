from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim

from proxbias.utils.chromosome_info import get_chromosome_info_as_dfs


def cosine_similarity(
    a: pd.DataFrame,
    b: Optional[pd.DataFrame] = None,
    index_names: Tuple[str, str] = ("perturbation_A", "perturbation_B"),
    as_long: bool = False,
    triu: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    index = a.index.copy().rename(name=index_names[0])
    if isinstance(b, pd.DataFrame) and not b.empty:
        if triu:
            raise ValueError("`triu` is only supported when getting pairwise cosine similarity from one dataframe, A.")
        columns = b.index.copy().rename(name=index_names[1])
        cossim_matrix = pd.DataFrame(sk_cosine_sim(a, b), index=index, columns=columns)
    else:
        cos = sk_cosine_sim(a, a)
        if triu:
            cos[np.tril_indices_from(cos, 1)] = np.NaN

        cossim_matrix = pd.DataFrame(cos, index=index, columns=index)

    if as_long:
        cossims_long = cossim_matrix.melt(ignore_index=False, var_name=index_names[1], value_name="cosine_similarity")
        cossims_long = cossims_long.reset_index().set_index(list(index_names))["cosine_similarity"]
        if triu:
            cossims_long = cossims_long.dropna()
        return cossims_long
    return cossim_matrix


def add_gene_info(cossims: pd.DataFrame) -> pd.DataFrame:
    cossims = cossims.copy()
    gene_info, _, _ = get_chromosome_info_as_dfs()
    gene_info_A = gene_info.copy().rename(mapper=lambda x: x + "_A", axis="columns")
    gene_info_B = gene_info.copy().rename(mapper=lambda x: x + "_B", axis="columns")
    cossims = cossims.merge(gene_info_A, left_index=True, right_index=True, how="inner")
    cossims = cossims.merge(gene_info_B, left_index=True, right_index=True, how="inner")
    cossims["same_chromosome_arm"] = (cossims["chrom_A"] == cossims["chrom_B"]) & (
        cossims["chrom_arm_A"] == cossims["chrom_arm_B"]
    )
    return cossims
