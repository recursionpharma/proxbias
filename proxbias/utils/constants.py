from importlib_resources import files

VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

DATA_DIR = files("proxbias").joinpath("data")  # type:ignore[attr-defined]

ARMS_ORD = (
    "chr1p,chr1q,chr2p,chr2q,chr3p,chr3q,chr4p,chr4q,chr5p,chr5q,chr6p,"
    "chr6q,chr7p,chr7q,chr8p,chr8q,chr9p,chr9q,chr10p,chr10q,"
    "chr11p,chr11q,chr12p,chr12q,chr13p,chr13q,chr14p,chr14q,chr15p,chr15q,chr16p,chr16q,chr17p,chr17q,chr18p,chr18q,"
    "chr19p,chr19q,chr20p,chr20q,chr21p,chr21q,chr22p,chr22q,chrXp,chrXq,chrYp,chrYq"
).split(",")

CANCER_GENES_FILENAME = "cancerGeneList.tsv"
