from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim


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
    else:
        b = a
    columns = b.index.copy().rename(name=index_names[1])
    cos = sk_cosine_sim(a.values, b.values)
    cossim_matrix = pd.DataFrame(cos, index=index, columns=columns)
    if triu:
        cos[np.tril_indices_from(cos, 1)] = np.NaN
    if as_long:
        cossim_series = pd.Series(cossim_matrix.values.flatten(), index=pd.MultiIndex.from_product([index, columns]))
        if triu:
            cossim_series = cossim_series.dropna()
        return cossim_series

    return cossim_matrix
