from proxbias.utils import get_chromosome_info_dataframe, get_chromosome_info_dict


def test_keys_equal():
    gene_df, chrom_df, band_df = get_chromosome_info_dataframe()
    gene_dict, chrom_dict, _, band_dict = get_chromosome_info_dict()
    print(chrom_df)
    assert set(gene_df.index) == set(gene_dict.keys())
    assert set(chrom_df.index) == set(chrom_dict.keys())
    assert len(band_df.index) == len(band_dict.keys())
