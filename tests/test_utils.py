import csv
import gzip
import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from proxbias.constants import DATA_DIR, VALID_CHROMS
from proxbias.utils import get_chromosome_info_dataframes


@pytest.fixture
def legacy_chromosome_info_dicts() -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Legacy method for creating chromosome info

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
            if gene_name in flagged_genes or chrom not in CHROMS or re.match("^LOC*", gene_name):
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
                GENES[gene_name]["end"] = max(end, GENES[gene_name]["end"])
    return GENES, CHROMS, CHROM_ARMS, CHROM_BANDS


def test_keys_equal(legacy_chromosome_info_dicts):
    gene_df, chrom_df, band_df = get_chromosome_info_dataframes()
    gene_dict, chrom_dict, _, band_dict = legacy_chromosome_info_dicts
    assert set(gene_df.index) == set(gene_dict.keys())
    assert set(chrom_df.index) == set(chrom_dict.keys())
    assert len(band_df.index) == len(band_dict.keys())


def test_coordinates_equal(legacy_chromosome_info_dicts):
    gene_df, chrom_df, band_df = get_chromosome_info_dataframes()
    gene_dict, chrom_dict, _, band_dict = legacy_chromosome_info_dicts

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
