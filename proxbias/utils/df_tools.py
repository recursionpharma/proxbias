import numpy as np
import pandas as pd
import scipy as sp
from typing import Tuple
from sklearn.utils import Bunch
import scipy.cluster.hierarchy as scipy_hierarchy
import scipy.spatial.distance as scipy_distance


def harmonize_data(
    data1: Bunch,
    data2: Bunch,
    cols: list = ["gene", "chromosome", "chromosome_arm", "chr_idx", "gene_bp"],
    kind: str = "intersection",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make dataframes with index of `cols` and features as values.
    Genes present in either dataset should be present in both with NA feature values if they were not
    originally present

    Inputs:
    -------
    - data1 = Bunch object with `metadata` pd.DataFrame and `features` pd.DataFrame.
              `metadata` must contain `cols` that will be used as the index for the resulting pd.DataFrame
              `metadata` and `features` must have matching indices.
    - data2 = Bunch object with `metadata` pd.DataFrame and `features` pd.DataFrame.
              `metadata must contain `cols` that will be used as the index for the resulting pd.DataFrame
    - cols = List of column names from the `metadata` in `data1` and `data2`.
             The first value should be the name of the column containing gene identifiers.
    - kind = 'intersection' or 'union'. Whether to return dataframes containing rows present only both
             or either of the input bunches
    """

    for x in ["metadata", "features"]:
        if x not in data1:
            raise KeyError(f"data1 must contain {x}")
        if x not in data2:
            raise KeyError(f"data2 must contain {x}")
    for x in cols:
        if x not in data1.metadata.columns:
            raise KeyError(f"{x} is not in data1.metadata")
        if x not in data2.metadata.columns:
            raise KeyError(f"{x} is not in data2.metadata")
    assert kind in ("union", "intersection"), 'Kind should be "union" or "intersection"'

    g1 = data1.metadata[cols[0]].unique()
    g2 = data2.metadata[cols[0]].unique()

    d1 = data1.features.copy()
    d2 = data2.features.copy()
    d1 = d1.set_index(pd.MultiIndex.from_frame(data1.metadata.loc[:, cols]))
    d2 = d2.set_index(pd.MultiIndex.from_frame(data2.metadata.loc[:, cols]))

    if kind == "union":
        all_genes = np.union1d(g1, g2)

        # Add rows of NaNs to each for genes that aren't in the other
        d2notd1 = np.setdiff1d(g2, g1)
        tmp = np.empty((len(d2notd1), d1.shape[1]))
        tmp[:] = np.nan
        d1 = pd.concat([d1, pd.DataFrame(tmp, index=d2.query("gene in @d2notd1").index)])

        d1notd2 = np.setdiff1d(g1, g2)
        tmp = np.empty((len(d1notd2), d2.shape[1]))
        tmp[:] = np.nan
        d2 = pd.concat([d2, pd.DataFrame(tmp, index=d1.query("gene in @d1notd2").index)])

    elif kind == "intersection":
        all_genes = np.intersect1d(g1, g2)

        # Subset to the shared genes
        d1 = d1.query("gene in @all_genes")
        d2 = d2.query("gene in @all_genes")

    print(f"{len(g1)} genes in dataset1 {len(g2)} genes in dataset2, {len(all_genes)} in the {kind}")

    # Sort by chrom index and bp
    d1 = d1.sort_values(["chr_idx", "gene_bp", "gene"])
    d2 = d2.sort_values(["chr_idx", "gene_bp", "gene"])
    return d1, d2


def make_pairwise_cos(
    df: pd.DataFrame,
    convert: bool = True,
    dtype: type = np.float16,
) -> pd.DataFrame:
    """
    Converts a dataframe of samples X features into a square dataframe of samples X samples
    of cosine similarities between rows.

    Inputs
    ------
    - df = pd.DataFrame
    - convert = bool. Whether to convert the results to a smaller data type
    - dtype = type. Data type to convert to
    """
    mat = (1 - sp.spatial.distance.cdist(df.values, df.values, metric="cosine")).clip(-1, 1)
    if convert:
        mat = mat.astype(dtype)
    return pd.DataFrame(mat, index=df.index, columns=df.index)


def make_split_cosmat(
    d1_cos: pd.DataFrame,
    d2_cos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Makes an array of cosine similarities where above the diagonal comes from the first
    data frame and below comes from the second.
    Note: dataframes must be the same number of rows
    """
    assert d1_cos.shape[0] == d1_cos.shape[1], "df1 must be square"
    assert d1_cos.index.to_frame().equals(d1_cos.columns.to_frame()), "Indices must match columns for df1"
    assert d2_cos.shape[0] == d2_cos.shape[1], "Dataframe two must be square"
    assert d2_cos.index.to_frame().equals(d2_cos.columns.to_frame()), "Indices must match columns for df2"
    assert d1_cos.shape[0] == d2_cos.shape[0], f"Number of rows must match, got {d1_cos.shape[0]} and {d2_cos.shape[0]}"
    assert d1_cos.index.to_frame().equals(d2_cos.index.to_frame()), "Indices must match"
    assert d1_cos.columns.to_frame().equals(d2_cos.columns.to_frame()), "Columns must match"

    split_mat = np.ones(d1_cos.shape)
    ind_u = np.triu_indices(d1_cos.shape[0], 1)
    ind_l = np.tril_indices(d1_cos.shape[0], -1)

    split_mat[ind_u] = d1_cos.values[ind_u]
    split_mat[ind_l] = d2_cos.values[ind_l]

    return pd.DataFrame(split_mat, index=d1_cos.index, columns=d1_cos.columns)


def mk_gene_mats(
    genes: list,
    split_df: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subset square cosim dataframes to `gene` and cluster them by one dataset or the other

    Inputs:
    -------
    - genes: list of genes to select
    - split_mat: square dataframe with cosims for one dataset above the diagonal and the other below the diag
    - df1: dataframe with cosims for the first datset (above diag in `split_df`)
    - df2: dataframe with cosims for the second datset (below diag in `split_df`)

    Returns:
    --------
    - df: split dataframe selected to the desired genes (not clustered)
    - df1_sub: df1 selected to the desired genes (not clustered)
    - df2_sub: df2 selected to the desired genes (not clustered)
    - clust_df1: split dataframe selected to the desired genes and clustered by df1
    - clust_df2: split dataframe selected to the desired genes and clustered by df2
    """
    ind = [x in genes for x in split_df.index.get_level_values("gene")]
    df = split_df.loc[ind, ind]  # type: ignore

    ind = [x in genes for x in df1.index.get_level_values("gene")]
    df1_sub = df1.loc[ind, ind]  # type: ignore

    ind = [x in genes for x in df2.index.get_level_values("gene")]
    df2_sub = df2.loc[ind, ind]  # type: ignore

    try:
        idx = scipy_hierarchy.dendrogram(scipy_hierarchy.linkage(scipy_distance.pdist(df1_sub)), no_plot=True)["ivl"]
        idx = [int(i) for i in idx]
        clust_df1 = make_split_cosmat(df1_sub.iloc[idx, idx], df2_sub.iloc[idx, idx])
    except AssertionError:
        print("Missing values in df1, couldnt cluster")
        clust_df1 = make_split_cosmat(df1_sub, df2_sub)
    try:
        idx = scipy_hierarchy.dendrogram(scipy_hierarchy.linkage(scipy_distance.pdist(df2_sub)), no_plot=True)["ivl"]
        idx = [int(i) for i in idx]
        clust_df2 = make_split_cosmat(df1_sub.iloc[idx, idx], df2_sub.iloc[idx, idx])
    except AssertionError:
        clust_df2 = make_split_cosmat(df1_sub, df2_sub)
        print("Missing values in df2, couldnt cluster")

    return df, df1_sub, df2_sub, clust_df1, clust_df2
