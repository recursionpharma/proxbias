from typing import List

import pandas as pd


def build_arm_centering_df(
    data: pd.DataFrame,
    metadata_cols: List[str],
    arm_column="chromosome_arm",
    subset_query="zfpkm <-3",
    min_num_gene=20,
) -> pd.DataFrame:
    """Build a dataframe with the mean feature values for each chromosome arm

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with metadata and features
    metadata_cols : List[str]
        Metadata columns
    arm_column : str, optional
        Metadata column with arm identifier, by default "chromosome_arm"
    subset_query : str, optional
        Query to subset genes, by default "zfpkm <-3"
    min_num_gene : int, optional
        Minimum number of genes required. Dataframe returned will only
        include arms that meet this threshold, by default 20

    Returns
    -------
    pd.DataFrame
        DataFrame with mean feature values for each chromosome arm
    """
    subset = data.query(subset_query)
    if arm_column not in metadata_cols:
        metadata_cols = metadata_cols + [arm_column]
    features = subset.drop(metadata_cols, axis="columns")
    return features.groupby(subset[arm_column]).mean()[
        subset.groupby(arm_column)[metadata_cols[0]].size() > min_num_gene
    ]


def perform_arm_centering(
    data: pd.DataFrame,
    metadata_cols: List[str],
    arm_centering_df: pd.DataFrame,
    arm_column: str = "chromosome_arm",
) -> pd.DataFrame:
    """Apply arm centering to data

    Parameters
    ----------
    data : pd.DataFrame
        Data DataFrame
    metadata_cols : List[str]
        List of metadata columns
    arm_centering_df : pd.DataFrame
        Arm centering dataframe
    arm_column : str, optional
        Column that identifies chromosome arm, by default "chromosome_arm"

    Returns
    -------
    pd.DataFrame
        Arm centered data
    """
    metadata = data[metadata_cols]
    features = data.drop(metadata_cols, axis="columns")
    for chromosome_arm in arm_centering_df.index:
        arm_features = features[metadata[arm_column] == chromosome_arm]
        arm_features = arm_features - arm_centering_df.loc[chromosome_arm]
        features[metadata[arm_column] == chromosome_arm] = arm_features
    return metadata.join(features)
