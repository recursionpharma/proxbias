import functools
from typing import Tuple

import pandas as pd

from proxbias.constants import DATA_DIR, VALID_CHROMS


@functools.cache  # type: ignore[attr-defined]
def get_chromosome_info_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    def _chr_to_int(chr):
        if chr == "x":
            return 24
        if chr == "y":
            return 25
        return int(chr)

    def _chrom_int(chrom):
        return chrom.copy().str.split("chr").str[1].str.lower().map(_chr_to_int)

    # TODO: refactor into more composable functions

    chroms = pd.read_csv(DATA_DIR + "/hg38_scaffolds.tsv", sep="\t", usecols=["chrom", "chromStart", "chromEnd"])
    centros = pd.read_csv(DATA_DIR + "/centromeres_hg38.tsv", sep="\t", usecols=["chrom", "chromStart", "chromEnd"])
    bands = pd.read_csv(
        DATA_DIR + "/hg38_cytoband.tsv.gz", sep="\t", usecols=["name", "#chrom", "chromStart", "chromEnd"]
    )

    centros[["chromStart", "chromEnd"]] = centros[["chromStart", "chromEnd"]].astype(int)
    chroms[["chromStart", "chromEnd"]] = chroms[["chromStart", "chromEnd"]].astype(int)
    centros = centros.groupby("chrom", as_index=False).agg(
        centromere_start=("chromStart", "min"),
        centromere_end=("chromEnd", "max"),
    )

    chroms = chroms.loc[chroms.chrom.isin(VALID_CHROMS)].rename(columns={"chromStart": "start", "chromEnd": "end"})
    chroms = chroms.merge(centros, on="chrom", how="left")
    chroms["chrom_int"] = _chrom_int(chroms.chrom)
    chroms = chroms.set_index("chrom").sort_values("chrom_int", ascending=True)

    bands = (
        bands.rename(columns={"#chrom": "chrom"})
        .groupby(["chrom", "name"], as_index=False)
        .agg(
            band_start=("chromStart", "min"),
            band_end=("chromEnd", "max"),
            band_chrom_arm=("name", lambda x: x.str[:1].min()),
        )
    )
    bands["chrom_int"] = _chrom_int(bands.chrom)

    genes = pd.read_csv(
        DATA_DIR + "/ncbirefseq_hg38.tsv.gz", sep="\t", usecols=["name2", "chrom", "txStart", "txEnd"]
    ).rename(columns={"name2": "gene"})

    genes = genes.loc[genes.chrom.isin(VALID_CHROMS)]
    genes["chrom_int"] = _chrom_int(genes.chrom)
    genes = genes.groupby("gene", as_index=False).agg(
        start=("txStart", "min"),
        end=("txEnd", "max"),
        chrom_int=("chrom_int", "min"),
        chrom_count=("chrom", "nunique"),
    )
    genes = genes.loc[~genes.gene.str.contains("^LOC*", regex=True)]
    genes = genes.loc[genes.chrom_count == 1].drop(columns="chrom_count").set_index("gene")
    genes = genes.sort_values(["chrom_int", "start", "end"], ascending=True)

    chroms_centromere_mid = chroms.copy().set_index("chrom_int")
    chrom_centromere_mid = (
        (chroms_centromere_mid.centromere_start + chroms_centromere_mid.centromere_end) / 2
    ).to_dict()
    genes["chrom_arm_int"] = genes.apply(lambda x: x.end > chrom_centromere_mid[x.chrom_int], axis=1).astype(int)
    return genes, chroms, bands
