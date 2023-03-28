# Load reference genome info: genes and coordinates
import csv
import functools
import gzip
import os
from typing import Any, Dict, Tuple

import pandas as pd

VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

DATA_DIR = os.path.realpath(os.path.join("/", os.path.dirname(__file__), "..", "data"))


@functools.cache  # type: ignore[attr-defined]
def get_chromosome_information() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    def _chrom(chrom):
        return chrom.str.split("chr").str[1].str.lower().map(_chr_to_int)

    # Load chromosome information

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
    bands = (
        bands.rename(columns={"#chrom": "chrom"})
        .groupby(["chrom", "name"], as_index=False)
        .agg(
            band_start=("chromStart", "min"),
            band_end=("chromEnd", "max"),
        )
    )
    bands.chrom = _chrom(bands.chrom)

    chroms = chroms.loc[chroms.chrom.isin(VALID_CHROMS)].rename(columns={"chromStart": "start", "chromEnd": "end"})
    chroms = chroms.merge(centros, on="chrom", how="left")
    chroms.chrom = _chrom(chroms.chrom)
    chroms = chroms.assign(chrom_int=chroms.chrom).sort_values("chrom_int", ascending=True)

    genes = pd.read_csv(
        DATA_DIR + "/ncbirefseq_hg38.tsv.gz", sep="\t", usecols=["name2", "chrom", "txStart", "txEnd"]
    ).rename(columns={"name2": "gene"})

    genes = genes.loc[genes.chrom.isin(VALID_CHROMS)]
    genes.chrom = _chrom(genes.chrom)
    genes = genes.groupby("gene", as_index=False).agg(
        start=("txStart", "min"),
        end=("txEnd", "max"),
        chrom=("chrom", "min"),
    )
    genes = genes.drop_duplicates(subset="gene").set_index("gene")
    genes = genes.assign(chrom_int=genes.chrom).sort_values(["chrom_int", "start", "end"], ascending=True)

    chrom_centromere = chroms.set_index("chrom_int").centromere_start.to_dict()
    genes = genes.assign(chrom_arm=genes.apply(lambda x: x.start > chrom_centromere[x.chrom_int], axis=1).astype(int))
    return genes, chroms, bands


@functools.cache  # type: ignore[attr-defined]
def get_chromosome_info_legacy() -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Returns chromosome information as a 4-tuple of dictionaries. The first is the genes, the second is the
    chromosomes, the third are the boundaries for chromosome arms, and the fourth are cytogenic bands.

    Returns
    -------
    Tuple[Dict, Dict, Dict, Dict]
        genes, chromosomes, arms, and cytogenic bands
    Raises
    ------
    ValueError
        Raised when there is an unexpected chromosome name in the data
    """
    CHROMS: Dict[str, Any] = {}
    CHROM_BANDS: Dict[str, Any] = {}
    CHROM_ARMS: Dict[str, Any] = {}
    GENES: Dict[str, Any] = {}

    def _find_chrom_arm(chrom, pos, chrom_arms):
        for arm in "pq":
            _, start, end = chrom_arms[chrom + arm]
            if pos >= start and pos < end:
                return chrom + arm
        else:
            return None

    # Load chromosome information
    with open(DATA_DIR + "/hg38_scaffolds.tsv", "rt") as stf:
        for row in csv.DictReader(stf, delimiter="\t"):
            chrom = row["chrom"]
            if chrom in VALID_CHROMS:
                CHROMS[chrom] = {"start": int(row["chromStart"]), "end": int(row["chromEnd"])}
    reordered_chroms = {ch: CHROMS[ch] for ch in VALID_CHROMS}
    CHROMS = reordered_chroms

    with open(DATA_DIR + "/centromeres_hg38.tsv", "rt") as ctf:
        for row in csv.DictReader(ctf, delimiter="\t"):
            chrom = row["chrom"]
            start, end = (int(row["chromStart"]), int(row["chromEnd"]))
            CHROMS[chrom]["centromere_start"] = min(start, CHROMS[chrom].get("centromere_start", 3e9))
            CHROMS[chrom]["centromere_end"] = max(end, CHROMS[chrom].get("centromere_end", 0))

    # Load chromosome cytoband/arm mapping
    with gzip.open(DATA_DIR + "/hg38_cytoband.tsv.gz", "rt") as btf:
        for row in csv.DictReader(btf, delimiter="\t"):
            if row["#chrom"] not in CHROMS:
                continue
            band_name = row["#chrom"] + row["name"]
            arm_name = row["#chrom"] + row["name"][0]
            if arm_name[-1] not in "pq":
                raise ValueError(("Unexpected chromosome arm name: ", arm_name))
            coords = (row["#chrom"], int(row["chromStart"]), int(row["chromEnd"]))
            CHROM_BANDS[band_name] = coords
            if arm_name in CHROM_ARMS:
                start = min(coords[1], CHROM_ARMS[arm_name][1])
                end = max(coords[2], CHROM_ARMS[arm_name][2])
                CHROM_ARMS[arm_name] = (coords[0], start, end)
            else:
                CHROM_ARMS[arm_name] = coords

    # Load gene information
    flagged_genes = set()

    with gzip.open("data/ncbirefseq_hg38.tsv.gz", "rt") as gtf:
        for row in csv.DictReader(gtf, delimiter="\t"):
            gene_name = row["name2"]
            chrom = row["chrom"]
            start = int(row["txStart"])
            end = int(row["txEnd"])
            if gene_name in flagged_genes or chrom not in CHROMS:
                continue
            if gene_name not in GENES:
                GENES[gene_name] = {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "arm": _find_chrom_arm(chrom, start, CHROM_ARMS),
                }
            else:
                if chrom != GENES[gene_name]["chrom"]:
                    print(
                        f"Flagged multi-chrom gene {gene_name} previously found at "
                        f"{GENES[gene_name]} now found at {(chrom, start, end)}"
                    )
                    del GENES[gene_name]
                    flagged_genes.add(gene_name)
                    continue
                GENES[gene_name]["start"] = min(start, GENES[gene_name]["start"])
                GENES[gene_name]["end"] = max(start, GENES[gene_name]["end"])
    return GENES, CHROMS, CHROM_ARMS, CHROM_BANDS
