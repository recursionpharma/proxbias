import pandas as pd
import numpy as np

from proxbias.cpg_processing.transform import preprocess_data, transform_data


def test_preprocess_data():
    data = pd.DataFrame(
        {
            "Metadata_Plate": ["plate1", "plate1", "plate1", "plate2", "plate2", "plate2"],
            "Metadata_Symbol": ["gene1", "gene1", "gene2", "gene1", "gene1", "gene2"],
            "Image_feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [1, 2, 3, 4, 5, 6],
            "feature3": [1, 2, np.nan, 4, 5, 6],
            "feature4": [1, 2, 3, 4, 5, 6],
        }
    )
    preprocessed_data = preprocess_data(data, metadata_cols=["Metadata_Plate", "Metadata_Symbol"], drop_image_cols=True)
    assert "Image_feature1" not in preprocessed_data.columns
    assert "feature3" not in preprocessed_data.columns
    assert preprocessed_data.shape[0] == 6


def test_transform_data():
    """Test transform_data function."""
    data = pd.DataFrame(
        {
            "Metadata_Plate": ["plate1", "plate1", "plate1", "plate2", "plate2", "plate2"],
            "Metadata_Symbol": ["gene1", "gene1", "gene2", "gene1", "gene1", "gene2"],
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [1, 2, 3, 4, 5, 6],
            "feature3": [1, 2, 3, 4, 5, 6],
            "feature4": [1, 2, 3, 4, 5, 6],
        }
    )

    transformed_data = transform_data(data, metadata_cols=["Metadata_Plate", "Metadata_Symbol"], variance=0.98)
    assert transformed_data.shape[0] == 2
    assert transformed_data.Metadata_Symbol.tolist() == ["gene1", "gene2"]
