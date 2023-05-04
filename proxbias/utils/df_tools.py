import numpy as np
import pandas as pd

def harmonize_data(data1, data2, kind='intersection') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make dataframes with index of `[gene, chrom, chrom_arm, start_coord]` and features as values. 
    Genes present in either dataset should be present in both with NA feature values if they were not 
    originally present
    """
    g1 = data1.metadata.display_label.unique()
    g2 = data2.metadata.display_label.unique()

    cols = ['display_label', 'chromosome', 'chromosome_arm', 'chr_idx', 'gene_bp']
    d1 = data1.features.copy()
    d2 = data2.features.copy()
    d1 = d1.set_index(pd.MultiIndex.from_frame(data1.metadata.loc[:,cols]))
    d2 = d2.set_index(pd.MultiIndex.from_frame(data2.metadata.loc[:,cols]))

    assert kind in ('union', 'intersection'), 'Kind should be "union" or "intersection"'

    if kind == 'union':
        all_genes = np.union1d(g1, g2)

        # Add rows of NaNs to each for genes that aren't in the other
        d2notd1 = np.setdiff1d(g2, g1)
        tmp = np.empty((len(d2notd1), d1.shape[1]))
        tmp[:] = np.nan
        d1 = pd.concat([d1, pd.DataFrame(tmp, index=d2.query('display_label in @d2notd1').index)])

        d1notd2 = np.setdiff1d(g1, g2)
        tmp = np.empty((len(d1notd2), d2.shape[1]))
        tmp[:] = np.nan
        d2 = pd.concat([d2, pd.DataFrame(tmp, index=d1.query('display_label in @d1notd2').index)])

    elif kind == 'intersection':
        all_genes = np.intersect1d(g1, g2)

        # Subset to the shared genes
        d1 = d1.query('display_label in @all_genes')
        d2 = d2.query('display_label in @all_genes')

    print(f'{len(g1)} genes in dataset1 {len(g2)} genes in dataset2, {len(all_genes)} in the {kind}')

    # Sort by chrom index and bp 
    d1 = d1.sort_values(['chr_idx', 'gene_bp', 'display_label'])
    d2 = d2.sort_values(['chr_idx', 'gene_bp', 'display_label'])
    return d1, d2

def make_pairwise(df, convert=True, dtype=np.float16) -> pd.DataFrame:
    mat = (1-scipy.spatial.distance.cdist(df.values, df.values, metric='cosine')).clip(-1, 1)
    if convert:
        mat = mat.astype(dtype)
    return pd.DataFrame(mat, index=df.index, columns=df.index)

def make_split_cosmat(d1_cos, d2_cos) -> pd.DataFrame:
    """
    Makes an array of cosine similarities where above the diagonal comes from the first
    data frame and below comes from the second. 
    Note: dataframes must be the same number of rows
    """
    assert d1_cos.shape[0] == d1_cos.shape[1], "df1 must be square"
    assert d1_cos.index.to_frame().equals(d1_cos.columns.to_frame()), f"Indices must match columns for df1"
    assert d2_cos.shape[0] == d2_cos.shape[1], "Dataframe two must be square"
    assert d2_cos.index.to_frame().equals(d2_cos.columns.to_frame()), f"Indices must match columns for df2"
    assert d1_cos.shape[0] == d2_cos.shape[0], f'Number of rows must match, got {d1_cos.shape[0]} and {d2_cos.shape[0]}'
    assert d1_cos.index.to_frame().equals(d2_cos.index.to_frame()), f"Indices must match"
    assert d1_cos.columns.to_frame().equals(d2_cos.columns.to_frame()), f"Columns must match"

    split_mat = np.ones(d1_cos.shape)
    ind_u = np.triu_indices(d1_cos.shape[0], 1)
    ind_l = np.tril_indices(d1_cos.shape[0], -1)

    split_mat[ind_u] = d1_cos.values[ind_u]
    split_mat[ind_l] = d2_cos.values[ind_l]

    return pd.DataFrame(split_mat, index=d1_cos.index, columns=d1_cos.columns)