import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CPG_METADATA_COLS = [
    "Metadata_Source",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_JCP2022",
    "Metadata_Batch",
    "Metadata_PlateType",
    "Metadata_NCBI_Gene_ID",
    "Metadata_Symbol",
]


def preprocess_data(
    data: pd.DataFrame, metadata_cols: list = CPG_METADATA_COLS, drop_image_cols: bool = True
) -> pd.DataFrame:
    """Preprocess data by dropping feature columns with nan values, dropping the
    Number_Object columns, and optionaly dropping the Image columns

    Parameters
    ----------
    data : pd.DataFrame
        Data to preprocess
    metadata_cols : List[str], optional
        Metadata columns, by default CPG_METADATA_COLS
    drop_image_cols : bool, optional
        Whether to drop Image columns, by default True

    Returns
    -------
    pd.DataFrame
        Processed dataframe
    """
    metadata = data[metadata_cols]
    features = data[[col for col in data.columns if col not in metadata_cols]]
    features = features.dropna(axis=1)
    features = features.drop(columns=[col for col in features.columns if col.endswith("Object_Number")])
    if drop_image_cols:
        image_cols = [col for col in features.columns if col.startswith("Image_")]
        features = features.drop(columns=image_cols)
    return metadata.join(features)


def transform_data(
    data: pd.DataFrame,
    metadata_cols: list = CPG_METADATA_COLS,
    variance=0.98,
) -> pd.DataFrame:
    """Transform data by scaling and applying PCA. Data is scaled by plate
    before and after PCA is applied. The experimental replicates are averaged
    together by taking the mean.

    Parameters
    ----------
    data : pd.DataFrame
        Data to transform
    metadata_cols : list, optional
        Metadata columns, by default CPG_METADATA_COLS
    variance : float, optional
        Variance to keep after PCA, by default 0.98

    Returns
    -------
    pd.DataFrame
        Transformed data
    """
    metadata = data[metadata_cols]
    features = data[[col for col in data.columns if col not in metadata_cols]]

    for plate in metadata.Metadata_Plate.unique():
        scaler = StandardScaler()
        features.loc[metadata.Metadata_Plate == plate, :] = scaler.fit_transform(
            features.loc[metadata.Metadata_Plate == plate, :]
        )

    features = pd.DataFrame(PCA(variance).fit_transform(features))

    for plate in metadata.Metadata_Plate.unique():
        scaler = StandardScaler()
        features.loc[metadata.Metadata_Plate == plate, :] = scaler.fit_transform(
            features.loc[metadata.Metadata_Plate == plate, :]
        )

    return features.groupby(metadata["Metadata_Symbol"], as_index=True).mean().reset_index()
