import pandas as pd

from proxbias.arm_centering import build_arm_centering_df, perform_arm_centering


def test_build_arm_centering_df():
    data = pd.DataFrame(
        {
            "chromosome_arm": ["p", "p", "q", "q", "q", "q"],
            "gene": ["a", "b", "c", "d", "e", "f"],
            "zfpkm": [1, 2, 3, 4, 5, 6],
            "feature": [0, 2, 4, 6, 8, 10],
        }
    )
    metadata_cols = ["chromosome_arm", "gene", "zfpkm"]
    arm_centering_df = build_arm_centering_df(data, metadata_cols, subset_query="zfpkm < 6", min_num_gene=1)
    print(arm_centering_df)

    assert arm_centering_df.loc["p", "feature"] == 1
    assert arm_centering_df.loc["q", "feature"] == 6


def test_perform_arm_centering():
    data = pd.DataFrame(
        {
            "chromosome_arm": ["p", "p", "q", "q", "q", "q"],
            "gene": ["a", "b", "c", "d", "e", "f"],
            "zfpkm": [1, 2, 3, 4, 5, 6],
            "feature": [0, 2, 4, 6, 8, 10],
        }
    )
    metadata_cols = ["chromosome_arm", "gene", "zfpkm"]
    arm_centering_df = build_arm_centering_df(data, metadata_cols, subset_query="zfpkm < 6", min_num_gene=1)
    centered_data = perform_arm_centering(data, metadata_cols, arm_centering_df)
    assert centered_data.loc[0, "feature"] == -1
    assert centered_data.loc[1, "feature"] == 1
    assert centered_data.loc[2, "feature"] == -2
    assert centered_data.loc[3, "feature"] == 0
    assert centered_data.loc[4, "feature"] == 2
    assert centered_data.loc[5, "feature"] == 4
