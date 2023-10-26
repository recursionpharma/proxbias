from typing import List, Tuple

import pandas as pd

from proxbias.utils.constants import CANCER_GENES_FILENAME, DATA_DIR


def _get_data_path(name):
    return DATA_DIR.joinpath(name)


def get_cancer_gene_lists(valid_genes: List[str]) -> Tuple[List[str], List[str]]:
    # Assumes data file is present in data folder
    oncokb = pd.read_csv(f"proxbias/data/{CANCER_GENES_FILENAME}", delimiter="\t")
    oncokb = oncokb.loc[oncokb["Hugo Symbol"].isin(valid_genes)]
    tsg_genes = oncokb.loc[oncokb["Is Tumor Suppressor Gene"] == "Yes", "Hugo Symbol"].tolist()
    oncogenes = oncokb.loc[oncokb["Is Oncogene"] == "Yes", "Hugo Symbol"].tolist()
    return tsg_genes, oncogenes
