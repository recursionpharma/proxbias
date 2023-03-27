# Load reference genome info: genes and coordinates
import csv
import gzip
from functools import cache
from typing import Dict, Union

import pandas as pd

VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def _chr_to_int(ch):
    if ch.lower() == "x":
        return 24
    if ch.lower() == "y":
        return 25
    return int(ch)


def _chrom(chrom_agg):
    return chrom_agg.str.split("chr").str[1].min()


@cache
def get_chromosome_bands() -> pd.DataFrame:
    CHROMS = {}
    CHROM_BANDS = {}
    CHROM_ARMS = {}
    with gzip.open("hg38_cytoband.tsv.gz", "rt") as tf:
        for row in csv.DictReader(tf, delimiter="\t"):
            if row["#chrom"] not in CHROMS:
                continue
            band_name = row["#chrom"] + row["name"]
            arm_name = row["#chrom"] + row["name"][0]
            assert arm_name[-1] in "pq"
            coords = (row["#chrom"], int(row["chromStart"]), int(row["chromEnd"]))
            CHROM_BANDS[band_name] = coords
            if arm_name in CHROM_ARMS:
                start = min(coords[1], CHROM_ARMS[arm_name][1])
                end = max(coords[2], CHROM_ARMS[arm_name][2])
                CHROM_ARMS[arm_name] = (coords[0], start, end)
            else:
                CHROM_ARMS[arm_name] = coords


@cache
def get_chromosome_information() -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load chromosome information

    chroms = pd.read_csv("data/hg38_scaffolds.tsv", sep="\t", usecols=["chrom", "chromStart", "chromEnd"])
    centros = pd.read_csv("data/centromeres_hg38.tsv", sep="\t", usecols=["chrom", "chromStart", "chromEnd"])
    bands = pd.read_csv("data/hg38_ctyoband.tsv.gz", sep="\t", usecols=["chrom", "chromStart", "chromEnd"])

    centros[["chromStart", "chromEnd"]] = centros[["chromStart", "chromEnd"]].astype(int)
    chroms[["chromStart", "chromEnd"]] = chroms[["chromStart", "chromEnd"]].astype(int)
    centros = centros.groupby("chrom", as_index=False).agg(
        centromere_start=("chromStart", "min"),
        centromere_end=("chromEnd", "max"),
    )
    bands = bands.groupby(["chrom", "name"], as_index=False).agg(
        band_start=("chromStart", "min"),
        band_end=("chromEnd", "max"),
    )

    chroms = chroms.loc[chroms.chrom.isin(VALID_CHROMS)].rename(columns={"chromStart": "start", "chromEnd": "end"})
    chroms = chroms.merge(centros, on="chrom", how="left")
    chroms.chrom = chroms.chrom.str.split("chr").str[1]
    chroms = chroms.assign(chrom_int=chroms.chrom.apply(_chr_to_int)).sort_values("chrom_int", ascending=True)

    genes = pd.read_csv("data/ncbirefseq_hg38.tsv.gz", sep="\t", usecols=["name2", "chrom", "txStart", "txEnd"]).rename(
        columns={"name2": "gene"}
    )
    genes = genes.loc[genes.chrom.isin(VALID_CHROMS)]
    genes = genes.groupby("gene", as_index=False).agg(
        start=("txStart", "min"),
        end=("txEnd", "max"),
        chrom=("chrom", _chrom),
    )
    genes = genes.drop_duplicates(subset="gene").set_index("gene")
    genes = genes.assign(chrom_int=genes.chrom.apply(_chr_to_int)).sort_values(
        ["chrom_int", "start", "end"], ascending=True
    )

    chrom_centromere = chroms.set_index("chrom_int").centromere_start.to_dict()
    genes = genes.assign(chrom_arm=genes.apply(lambda x: x.start > chrom_centromere[x.chrom_int], axis=1).astype(int))
    return genes, chroms, bands


@cache
def get_chromosome_info_legacy() -> Union[Dict, Dict, Dict, Dict]:
    # Load reference genome info: genes and coordinates

    # Load chromosome information
    CHROMS = {}
    with open("hg38_scaffolds.tsv", "rt") as tf:
        for row in csv.DictReader(tf, delimiter="\t"):
            chrom = row["chrom"]
            if chrom in VALID_CHROMS:
                CHROMS[chrom] = {"start": int(row["chromStart"]), "end": int(row["chromEnd"])}
    reordered_chroms = {ch: CHROMS[ch] for ch in VALID_CHROMS}
    CHROMS = reordered_chroms

    with open("centromeres_hg38.tsv", "rt") as tf:
        for row in csv.DictReader(tf, delimiter="\t"):
            chrom = row["chrom"]
            start, end = (int(row["chromStart"]), int(row["chromEnd"]))
            CHROMS[chrom]["centromere_start"] = min(start, CHROMS[chrom].get("centromere_start", 3e9))
            CHROMS[chrom]["centromere_end"] = max(end, CHROMS[chrom].get("centromere_end", 0))

    # Load chromosome cytoband/arm mapping
    CHROM_BANDS = {}
    CHROM_ARMS = {}
    with gzip.open("hg38_cytoband.tsv.gz", "rt") as tf:
        for row in csv.DictReader(tf, delimiter="\t"):
            if row["#chrom"] not in CHROMS:
                continue
            band_name = row["#chrom"] + row["name"]
            arm_name = row["#chrom"] + row["name"][0]
            assert arm_name[-1] in "pq"
            coords = (row["#chrom"], int(row["chromStart"]), int(row["chromEnd"]))
            CHROM_BANDS[band_name] = coords
            if arm_name in CHROM_ARMS:
                start = min(coords[1], CHROM_ARMS[arm_name][1])
                end = max(coords[2], CHROM_ARMS[arm_name][2])
                CHROM_ARMS[arm_name] = (coords[0], start, end)
            else:
                CHROM_ARMS[arm_name] = coords

    def find_chrom_arm(chrom, pos):
        for arm in "pq":
            _, start, end = CHROM_ARMS[chrom + arm]
            if pos >= start and pos < end:
                return chrom + arm
        else:
            return None

    # Load gene information
    flagged_genes = set()
    GENES = {}
    with gzip.open("ncbirefseq_hg38.tsv.gz", "rt") as tf:
        for row in csv.DictReader(tf, delimiter="\t"):
            gene_name = row["name2"]
            chrom = row["chrom"]
            start = int(row["txStart"])
            end = int(row["txEnd"])
            if gene_name in flagged_genes or chrom not in CHROMS:
                continue
            if gene_name not in GENES:
                GENES[gene_name] = {"chrom": chrom, "start": start, "end": end, "arm": find_chrom_arm(chrom, start)}
            else:
                if chrom != GENES[gene_name]["chrom"]:
                    print(
                        f"Flagged multi-chrom gene {gene_name} previously found at {GENES[gene_name]} now found at {(chrom, start, end)}"
                    )
                    del GENES[gene_name]
                    flagged_genes.add(gene_name)
                    continue
                GENES[gene_name]["start"] = min(start, GENES[gene_name]["start"])
                GENES[gene_name]["end"] = max(start, GENES[gene_name]["end"])
    return GENES, CHROMS, CHROM_ARMS, CHROM_BANDS
