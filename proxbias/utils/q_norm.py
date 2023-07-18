import numpy as np
import pandas as pd
from scipy import stats as ss


def _get_percentiles(
    null: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Convert values in `x` to percentiles relative to the `null`
    """
    return np.searchsorted(null, x) / len(null)


def _transform(
    percentiles: np.ndarray,
    norm: ss.norm,
) -> np.ndarray:
    """
    transform `percentiles` into corresponding values in `norm`
    """
    return norm.ppf(percentiles)


def get_transforms(
    x: np.ndarray,
    null: np.ndarray,
    *,
    loc: float = 0.0,
    scale: float = 0.2,
) -> np.ndarray:
    """Obtain the percentile ranking transformed values for an array.
    This function will map the distance metric into the null distribution to
    obtain each measurement's percentile ranking, and then rescale that
    percentile according to a normal function to mimic a normal distribution of
    transformed values.
    Inputs:
    -------
    - x (np.ndarray): array of values
    - null_distribution (np.ndarray): values that represents a null
    - loc (float, optional): Used for the center of the normal distribution. Defaults to 0.0
    - scale (int, optional): Used for the scale of the normal distribution. Defaults to 0.2
    Returns:
    --------
    - np.ndarray
    """
    norm = ss.norm(loc=loc, scale=scale)
    null = null.copy()
    null.sort()
    percs = _get_percentiles(null, x)
    trans = _transform(percs, norm)
    return trans


def q_norm(
    df: pd.DataFrame,
    trans_args: dict = {},
) -> pd.DataFrame:
    """
    Quantile normalizes a square symmetrical dataframe to a normal distrbution
    """
    X = np.zeros(df.shape)
    ind_u = np.triu_indices(df.shape[0], 1)
    tmp_u = get_transforms(df.values[ind_u], df.values[ind_u], **trans_args)
    X[ind_u] = tmp_u.clip(-1, 1)
    df_norm = pd.DataFrame(X + X.T, index=df.index, columns=df.columns)
    np.fill_diagonal(df_norm.values, 1)
    return df_norm
