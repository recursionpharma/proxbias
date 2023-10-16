DEPMAP_API_URL = "https://depmap.org/portal/api/download/files"
DEPMAP_RELEASE_DEFAULT = "DepMap Public 22Q4"

CNV_FILENAME = "OmicsCNGene.csv"
CRISPR_DEPENDENCY_EFFECT_FILENAME = "CRISPRGeneEffect.csv"
MUTATION_FILENAME = "OmicsSomaticMutations.csv"

DEMETER2_RELEASE_DEFAULT = "DEMETER2 Data v6"
RNAI_DEPENDENCY_EFFECT_FILENAME = "D2_combined_gene_dep_scores.csv"


CN_LOSS_CUTOFF = 1.5
CN_GAIN_CUTOFF = 2.5


LOF_MUTATION_TYPES = [
    "MISSENSE",
    "NONSENSE",
    "IN_FRAME_DEL",
    "SPLICE_SITE",
    "FRAME_SHIFT_INS",
    "FRAME_SHIFT_DEL",
    "IN_FRAME_INS",
]

COMPLETE_LOF_MUTATION_TYPES = [
    "NONSENSE",
    "FRAME_SHIFT_INS",
    "FRAME_SHIFT_DEL",
]

# gain of function mutation types
AMP_MUTATION_TYPES = [
    "MISSENSE",
    "IN_FRAME_DEL",
    "SPLICE_SITE",
    "IN_FRAME_INS",
]
