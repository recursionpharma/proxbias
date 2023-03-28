import warnings

import numpy as np
import pandas as pd

from proxbias.utils import get_chromosome_info_dataframes, get_chromosome_info_dicts


def test_keys_equal():
    gene_df, chrom_df, band_df = get_chromosome_info_dataframes()
    gene_dict, chrom_dict, _, band_dict = get_chromosome_info_dicts()
    assert set(gene_df.index) == set(gene_dict.keys())
    assert set(chrom_df.index) == set(chrom_dict.keys())
    assert len(band_df.index) == len(band_dict.keys())


def test_coordinates_equal():
    gene_df, chrom_df, band_df = get_chromosome_info_dataframes()
    gene_dict, chrom_dict, _, band_dict = get_chromosome_info_dicts()

    gene_df2 = pd.DataFrame.from_dict(gene_dict).T.sort_index()
    gene_df2[["start", "end"]] = gene_df2[["start", "end"]].astype(np.int64)
    gene_df["chrom"] = gene_df["chrom_int"].apply(lambda x: f"chr{x}" if x < 23 else ("chrX" if x == 24 else "chrY"))
    gene_df["arm"] = gene_df["chrom_arm_int"].apply(lambda x: "p" if x == 0 else "q")
    gene_df["arm"] = gene_df.apply(lambda x: f"{x.chrom}{x.arm}", axis=1)
    gene_df = gene_df[["chrom", "start", "end", "arm"]].sort_index()

    pd.testing.assert_frame_equal(
        gene_df,
        gene_df2,
        check_names=False,
        check_like=True,
    )

    chrom_df2 = pd.DataFrame.from_dict(chrom_dict).T
    pd.testing.assert_frame_equal(
        chrom_df.drop(columns="chrom_int"),
        chrom_df2,
        check_names=False,
        check_like=True,
    )

    band_df2 = pd.DataFrame.from_dict(band_dict).T
    band_df2.columns = ["chrom", "band_start", "band_end"]
    band_df2[["band_start", "band_end"]] = band_df2[["band_start", "band_end"]].astype(np.int64)
    band_df["combined"] = band_df.chrom + band_df.name
    band_df = band_df.set_index("combined").drop(columns=["band_chrom_arm", "chrom_int", "name"])
    pd.testing.assert_frame_equal(
        band_df,
        band_df2,
        check_names=False,
        check_like=True,
    )
