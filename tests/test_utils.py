import numpy as np
import pandas as pd

from proxbias.utils.chromosome_info import get_chromosome_info_as_dfs, get_chromosome_info_as_dicts


def test_keys_equal(legacy_chromosome_info_dicts):
    gene_df, chrom_df, band_df = get_chromosome_info_as_dfs()
    gene_dict, chrom_dict, _, band_dict = legacy_chromosome_info_dicts
    assert set(gene_df.index) == set(gene_dict.keys())
    assert set(chrom_df.index) == set(chrom_dict.keys())
    assert len(band_df.index) == len(band_dict.keys())


def test_df_parity(legacy_chromosome_info_dicts):
    gene_df, chrom_df, band_df = get_chromosome_info_as_dfs()
    gene_dict, chrom_dict, _, band_dict = legacy_chromosome_info_dicts

    gene_df2 = pd.DataFrame.from_dict(gene_dict).T.sort_index()
    gene_df2[["start", "end"]] = gene_df2[["start", "end"]].astype(np.int64)
    gene_df["arm"] = gene_df.chrom + gene_df.chrom_arm
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
    band_df["region"] = band_df.chrom + band_df.name
    band_df = band_df.set_index("region").drop(columns=["band_chrom_arm", "chrom_int", "name"])
    pd.testing.assert_frame_equal(
        band_df,
        band_df2,
        check_names=False,
        check_like=True,
    )


def test_dicts_parity(legacy_chromosome_info_dicts):
    legacy_gene_dict, legacy_chrom_dict, _, legacy_band_dict = legacy_chromosome_info_dicts
    gene_dict, chrom_dict, band_dict = get_chromosome_info_as_dicts(legacy_bands=True)

    gene_keys = ["chrom", "start", "end", "arm"]
    for gene in legacy_gene_dict.keys():
        for key in gene_keys:
            leg = legacy_gene_dict[gene].get(key)
            new = gene_dict[gene].get(key)
            if isinstance(leg, str):
                assert leg == new
            else:
                assert np.isclose(leg, new)

    chrom_keys = ["start", "end", "centromere_start", "centromere_end"]
    for chrom in legacy_chrom_dict.keys():
        for key in chrom_keys:
            leg = legacy_chrom_dict[chrom].get(key)
            new = chrom_dict[chrom].get(key)
            if isinstance(leg, str):
                assert leg == new
            else:
                assert np.isclose(leg, new)
    for band in legacy_band_dict.keys():
        leg = legacy_band_dict[band]
        new = band_dict[band]
        assert leg == new
