import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from proxbias.utils.data_utils import _get_data_path

CP_FEATURE_FORMATTER = (
    "s3://cellpainting-gallery/cpg0016-jump/"
    "{Metadata_Source}/workspace/profiles/"
    "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
)


def load_cpg_crispr_well_metadata():
    """Load well metadata for CRISPR plates from Cell Painting Gallery."""
    plates = pd.read_csv(_get_data_path("plate.csv.gz"))
    crispr_plates = plates.query("Metadata_PlateType=='CRISPR'")
    wells = pd.read_csv(_get_data_path("well.csv.gz"))
    crispr = pd.read_csv(_get_data_path("crispr.csv.gz"))

    well_plate = wells.merge(crispr_plates, on=["Metadata_Source", "Metadata_Plate"])
    crispr_well_metadata = well_plate.merge(crispr, on="Metadata_JCP2022")
    return crispr_well_metadata


def _load_plate_features(path: str):
    return pd.read_parquet(path, storage_options={"anon": True})


def load_feature_data(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Load feature data from Cell Painting Gallery from metadata dataframe.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Well metadata dataframe

    Returns
    -------
    pd.DataFrame
        Well features
    """
    cripsr_plates = metadata_df[
        ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_PlateType"]
    ].drop_duplicates()
    data = []
    with ThreadPoolExecutor(max_workers=10) as executer:
        future_to_plate = {
            executer.submit(
                _load_plate_features, CP_FEATURE_FORMATTER.format(**row.to_dict())
            ): CP_FEATURE_FORMATTER.format(**row.to_dict())
            for _, row in cripsr_plates.iterrows()
        }
        for future in concurrent.futures.as_completed(future_to_plate):
            data.append(future.result())
    return pd.concat(data)


def build_combined_data(metadata: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join well metadata and well features

    Parameters
    ----------
    metadata : pd.DataFrame
        Well metadata
    features : pd.DataFrame
        Well features

    Returns
    -------
    pd.DataFrame
        Combined dataframe
    """
    return metadata.merge(features, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"])
