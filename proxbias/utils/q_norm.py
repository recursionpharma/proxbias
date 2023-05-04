import numpy as np
import pandas as pd
from scipy import stats as ss

def _get_percentiles(null: pd.Series, x: pd.Series) -> pd.Series:
    percentile_data = np.searchsorted(null, x) / len(null)
    return pd.Series(data=percentile_data, index=x.index, name="percentiles")

def _transform(percentiles: pd.Series, norm: ss.norm, x: pd.Series) -> pd.Series:
    rescaled_norm = norm.ppf(percentiles)
    return pd.Series(data=rescaled_norm, index=x.index, name="norm")

def add_transforms(
    distance_df: pd.DataFrame,
    null_distribution: pd.Series,
    *,
    distance_col: str = "cosine_sim",
    loc: int = 0.0,
    scale: float = 0.2,
) -> pd.DataFrame:
    """Obtain the percentile ranking transformed values for a dataframe.
    This function will map the distance metric into the null distribution to
    obtain each measurement's percentile ranking, and then rescale that
    percentile according to a normal function to mimic a normal distribution of
    transformed values.
    The resultant dataframe will include both values.
    The provided null_distribution is expected to be a pandas series that
    represents a well-selected representative sample of the distances from the
    corresponing map's underlying distribution of like distances.
    Args:
        distance_df (pd.DataFrame): The output from pairwise_data_processing.
        null_distribution (pd.Series): A series of values that represents a null
        distribution of distances from the underlying map to percentile rank from.
        distance_col (str, optional): The column in the distance_df to map into
        the transformed values. Defaults to 'cosine_sim'.
        loc (int, optional): Used for the center of the normal distribution. Defaults to 0.0
        loc (int, optional): Used for the scale of the normal distribution. Defaults to 0.2
    Returns:
        pd.DataFrame: The input distance_df with added columns 'percentile' and
        'norm' as computed against the provided null_distribution.
    """
    norm = ss.norm(loc=loc, scale=scale)
    null = null_distribution.sort_values().values
    percs = _get_percentiles(null, distance_df[distance_col])
    norms = _transform(percs, norm, distance_df[distance_col])
    percs[distance_df[distance_col].isna()] = np.nan
    norms[distance_df[distance_col].isna()] = np.nan
    return pd.concat([distance_df, percs, norms], axis=1)

def q_norm(mat):
    X = np.zeros(mat.shape)
    ind_u = np.triu_indices(mat.shape[0], 1)
    tmp_u = pd.DataFrame(mat.values[ind_u], columns=['cosine_sim'])
    tmp_u = add_transforms(tmp_u, tmp_u.cosine_sim.dropna())
    X[ind_u] = tmp_u.norm.values.clip(-1,1)
    mat_norm = pd.DataFrame(X + X.T, index=mat.index, columns=mat.columns)
    np.fill_diagonal(mat_norm.values, 1)
    return mat_norm