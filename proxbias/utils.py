import functools
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from proxbias.constants import DATA_DIR, VALID_CHROMS


def _get_data_path(name):
    return DATA_DIR.joinpath(name)


def _chr_to_int(chr):
    if chr == "x":
        return 24
    elif chr == "y":
        return 25
    return int(chr)


def _chrom_int(chrom):
    return chrom.copy().str.split("chr").str[1].str.lower().map(_chr_to_int)


def _load_centromeres() -> pd.DataFrame:
    centros = pd.read_csv(_get_data_path("centromeres_hg38.tsv"), sep="\t", usecols=["chrom", "chromStart", "chromEnd"])
    centros[["chromStart", "chromEnd"]] = centros[["chromStart", "chromEnd"]].astype(int)
    centros = centros.groupby("chrom", as_index=False).agg(
        centromere_start=("chromStart", "min"),
        centromere_end=("chromEnd", "max"),
    )
    return centros


def _load_chromosomes(centromeres: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    chroms = pd.read_csv(_get_data_path("hg38_scaffolds.tsv"), sep="\t", usecols=["chrom", "chromStart", "chromEnd"])
    chroms[["chromStart", "chromEnd"]] = chroms[["chromStart", "chromEnd"]].astype(int)
    chroms = chroms.loc[chroms.chrom.isin(VALID_CHROMS)].rename(columns={"chromStart": "start", "chromEnd": "end"})
    chroms["chrom_int"] = _chrom_int(chroms.chrom)

    # Merge in centromere data if available
    if isinstance(centromeres, pd.DataFrame) and not centromeres.empty:
        chroms = chroms.merge(centromeres, on="chrom", how="left")

    chroms = chroms.set_index("chrom").sort_values("chrom_int", ascending=True)
    return chroms


def _load_bands() -> pd.DataFrame:
    bands = pd.read_csv(
        _get_data_path("hg38_cytoband.tsv.gz"), sep="\t", usecols=["name", "#chrom", "chromStart", "chromEnd"]
    )
    bands = bands.rename(columns={"#chrom": "chrom"})
    bands = bands.groupby(["chrom", "name"], as_index=False).agg(
        band_start=("chromStart", "min"),
        band_end=("chromEnd", "max"),
        band_chrom_arm=("name", lambda x: x.str[:1].min()),
    )
    bands["chrom_int"] = _chrom_int(bands.chrom)
    return bands


def _load_genes(chromosomes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    genes = pd.read_csv(
        _get_data_path("ncbirefseq_hg38.tsv.gz"), sep="\t", usecols=["name2", "chrom", "txStart", "txEnd"]
    ).rename(columns={"name2": "gene"})

    genes = genes.loc[genes.chrom.isin(VALID_CHROMS)]
    genes["chrom_int"] = _chrom_int(genes.chrom)
    genes = genes.groupby("gene", as_index=False).agg(
        start=("txStart", "min"),
        end=("txEnd", "max"),
        chrom_int=("chrom_int", "min"),
        chrom_count=("chrom", "nunique"),
        chrom=("chrom", "first"),
    )

    # Filter out (psuedo-)genes of unknown function and genes that show up on multiple chromosomes
    genes = genes.loc[~genes.gene.str.contains("^LOC*", regex=True)]
    genes = genes.loc[genes.chrom_count == 1].drop(columns="chrom_count").set_index("gene")
    genes = genes.sort_values(["chrom_int", "start", "end"], ascending=True)

    # Use the middle of the centromere as the way to determine if a gene is on the 0/1 chromosome
    if isinstance(chromosomes, pd.DataFrame) and not chromosomes.empty:
        chroms_centromere_mid = chromosomes.copy().set_index("chrom_int")
        chrom_centromere_mid = (
            (chroms_centromere_mid.centromere_start + chroms_centromere_mid.centromere_end) / 2
        ).to_dict()
        genes["chrom_arm_int"] = genes.apply(lambda x: x.end > chrom_centromere_mid[x.chrom_int], axis=1).astype(int)

        # NOTE: Assumes that p is the first chromosome
        genes["chrom_arm"] = genes["chrom_arm_int"].apply(lambda x: "p" if x == 0 else "q")
    return genes


@functools.cache  # type: ignore[attr-defined]
def get_chromosome_info_as_dfs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get structured information about the chromosomes that genes lie on as three dataframes:
        - Genes, including start and end and which chromosome arm they are on
        - Chromosomes, including the centromere start and end genomic coordinates
        - Cytogenic bands, including the name and start and end genomic

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Genes, chromosomes, cytogenic bands
    """

    # TODO: refactor into more composable functions
    bands = _load_bands()
    chroms = _load_chromosomes(centromeres=_load_centromeres())
    genes = _load_genes(chromosomes=chroms)

    return genes, chroms, bands


@functools.cache  # type: ignore[attr-defined]
def get_chromosome_info_as_dicts(legacy_bands: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convert the output of `get_chromosome_info_as_dfs` to dictionary form for compatibility
    with legacy notebooks.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        Dict corresponding to genes, chromosomes, and cytogenic bands
    """
    gene_df, chrom_df, band_df = get_chromosome_info_as_dfs()

    # Extra composite key for convenience
    gene_df["arm"] = gene_df.chrom + gene_df.chrom_arm
    gene_dict = gene_df.to_dict(orient="index")

    chrom_dict = chrom_df.to_dict(orient="index")

    # Extra composite key for convenience
    band_df["region"] = band_df.chrom + band_df.name
    if legacy_bands:
        band_df = band_df[["region", "chrom", "band_start", "band_end"]].set_index("region")
        band_dict = {k: tuple(v) for k, v in band_df.iterrows()}
    else:
        band_dict = band_df.to_dict(orient="index")
    return gene_dict, chrom_dict, band_dict
