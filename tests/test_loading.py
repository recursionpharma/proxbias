import pandas as pd
from proxbias.cpg_processing.loading import load_cpg_crispr_well_metadata


def test_load_cpg_crispr_well_metadata():
    """Test loading well metadata for CRISPR plates from Cell Painting Gallery."""
    crispr_well_metadata = load_cpg_crispr_well_metadata()
    assert isinstance(crispr_well_metadata, pd.DataFrame)
    assert {
        "Metadata_Source",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_JCP2022",
        "Metadata_Batch",
        "Metadata_PlateType",
        "Metadata_NCBI_Gene_ID",
        "Metadata_Symbol",
    } == set(crispr_well_metadata.columns)
