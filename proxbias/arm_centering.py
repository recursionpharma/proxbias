from typing import List

import pandas as pd


def build_arm_centering_df(
    data: pd.DataFrame, metadata_cols: List[str], arm_column="chromosome_arm", subset_query="zfpkm <-3", min_num_gene=20
) -> pd.DataFrame:
    subset = data.query(subset_query)
    if arm_column not in metadata_cols:
        metadata_cols = metadata_cols + [arm_column]
    features = subset.drop(metadata_cols, axis="columns")
    return features.groupby(subset[arm_column]).mean()[
        subset.groupby(arm_column)[metadata_cols[0]].size() > min_num_gene
    ]


def perform_arm_centering(data, metadata_cols, arm_centering_df, arm_column="chromosome_arm") -> pd.DataFrame:
    metadata = data[metadata_cols]
    features = data.drop(metadata_cols, axis="columns")
    for chromosome_arm in arm_centering_df.index:
        arm_features = features[metadata[arm_column] == chromosome_arm]
        arm_features = arm_features - arm_centering_df.loc[chromosome_arm]
        features[metadata[arm_column] == chromosome_arm] = arm_features
    return metadata.join(features)
