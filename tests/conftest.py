import csv
import gzip
import re
from typing import Any, Dict, Tuple

import pytest

from proxbias.utils.data_utils import _get_data_path
from proxbias.utils.constants import VALID_CHROMS


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
    with open(_get_data_path("hg38_scaffolds.tsv"), "rt") as stf:
        for row in csv.DictReader(stf, delimiter="\t"):
            chrom = row["chrom"]
            if chrom in VALID_CHROMS:
                CHROMS[chrom] = {"start": int(row["chromStart"]), "end": int(row["chromEnd"])}
    reordered_chroms = {ch: CHROMS[ch] for ch in VALID_CHROMS}
    CHROMS = reordered_chroms

    with open(_get_data_path("centromeres_hg38.tsv"), "rt") as ctf:
        for row in csv.DictReader(ctf, delimiter="\t"):
            chrom = row["chrom"]
            start, end = (int(row["chromStart"]), int(row["chromEnd"]))
            CHROMS[chrom]["centromere_start"] = min(start, CHROMS[chrom].get("centromere_start", 3e9))
            CHROMS[chrom]["centromere_end"] = max(end, CHROMS[chrom].get("centromere_end", 0))

    # Load chromosome cytoband/arm mapping
    with gzip.open(_get_data_path("hg38_cytoband.tsv.gz"), "rt") as btf:
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

    with gzip.open(_get_data_path("ncbirefseq_hg38.tsv.gz"), "rt") as gtf:
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
