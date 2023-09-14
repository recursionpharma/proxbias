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
    return np.searchsorted(null, x, side="right") / len(null)


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
    """
    null.sort()
    percs = _get_percentiles(x=x, null=null)
    return ss.norm(loc=loc, scale=scale).ppf(percs)


def q_norm(
    df: pd.DataFrame,
    trans_args: dict = {},
) -> pd.DataFrame:
    """
    Quantile normalizes a square symmetrical dataframe to a normal distrbution
    """
    ind_u = np.triu_indices(df.shape[0], 1)
    X = np.zeros(df.shape)
    X[ind_u] = get_transforms(x=df.values[ind_u], null=df.values[ind_u], **trans_args)
    X += X.T
    df_norm = pd.DataFrame(X, index=df.index, columns=df.columns)
    np.fill_diagonal(df_norm.values, 1)
    return df_norm
