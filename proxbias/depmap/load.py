import os
from functools import partial
from typing import Tuple

import pandas as pd

from proxbias.depmap.constants import (
    CNV_FILENAME,
    CRISPR_DEPENDENCY_EFFECT_FILENAME,
    DEMETER2_RELEASE_DEFAULT,
    DEPMAP_API_URL,
    DEPMAP_RELEASE_DEFAULT,
    MUTATION_FILENAME,
    RNAI_DEPENDENCY_EFFECT_FILENAME,
)


def _download_file(_release_files: pd.DataFrame, _filename: str, **read_kwargs):
    url = _release_files.loc[_release_files.filename == _filename].iloc[0].url
    return pd.read_csv(url, **read_kwargs)


def _cache_file(_df: pd.DataFrame, _prefix: str, _filename: str):
    if not os.path.exists(_prefix):
        os.makedirs(_prefix)
    _df.to_csv(f"{_prefix}/{_filename}")


def get_depmap_data(
    depmap_release: str = DEPMAP_RELEASE_DEFAULT,
    rnai_release: str = DEMETER2_RELEASE_DEFAULT,
    cache: bool = True,
    cache_base_dir: str = "depmap",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Obtain and format DepMap data.

    Parameters
    ----------
    depmap_release : str, optional
        a depmap release string, by default DEPMAP_RELEASE_DEFAULT
    rnai_release : str, optional
        a rnai release string, by default DEMETER2_RELEASE_DEFAULT. If an empty string,
        do not check cache and return an empty dataframe
    cache : bool, optional
        whether cache the data as csv or not, by default True
    cache_base_dir : str, optional
        path to cache the data, by default "depmap"

    Returns
    -------
    crispr_effect_data, rnai_effect_data, cnv_data, mutation_data pd.DataFrames
    index = gene symbol
    columns = cell line models

    Examples
    --------
    >>> crispr_effect_data, rnai_effect_data, cnv_data, mutation_data = get_depmap_data(
    >>>     depmap_release="DepMap Public 22Q4",
    >>>     rnai_release="DEMETER2 Data v6",
    >>>     cache=True,
    >>>     cache_base_dir="depmap",
    >>> )
    """

    def _read_and_cache(release_name, release_files, file_prefix, filename, **read_kwargs):
        target_file = f"{file_prefix}/{filename}"
        if os.path.isfile(target_file):
            print(f"{filename} from {release_name} is found. Reading dataframe from cache.")
            target_data = pd.read_csv(target_file, **read_kwargs)
            print("Done!")
        else:
            print(f"Cached file {filename} is not found. Downloading from {release_name}...")
            target_data = _download_file(release_files, filename, **read_kwargs)
            print("Done!")
            if cache:
                _cache_file(target_data, file_prefix, filename)
        return target_data

    release_files = pd.read_csv(DEPMAP_API_URL)

    depmap_file_prefix = f"{cache_base_dir}/{depmap_release}"
    depmap_release_files = release_files.loc[release_files.release == depmap_release]
    rnai_file_prefix = f"{cache_base_dir}/{rnai_release}"
    rnai_release_files = release_files.loc[release_files.release == rnai_release]

    _read_cache_depmap = partial(
        _read_and_cache,
        depmap_release,
        depmap_release_files,
        depmap_file_prefix,
    )
    _read_cache_rnai = partial(
        _read_and_cache,
        rnai_release,
        rnai_release_files,
        rnai_file_prefix,
    )

    # CRISPR Dependency Effect
    crispr_effect_data = _read_cache_depmap(CRISPR_DEPENDENCY_EFFECT_FILENAME, index_col=0)
    crispr_effect_data.index.name = "ModelID"
    crispr_effect_data = crispr_effect_data.T
    crispr_effect_data.index = [g.split(" ")[0] for g in crispr_effect_data.index]

    # RNAi Dependency Effect
    if rnai_release:
        rnai_effect_data = _read_cache_rnai(RNAI_DEPENDENCY_EFFECT_FILENAME, index_col=0)
        rnai_effect_data.columns.name = "ModelID"
        rnai_effect_data.index = [g.split(" ")[0] for g in rnai_effect_data.index]
        # remove multi-mapping oligos
        rnai_effect_data = rnai_effect_data.query("~index.str.contains('&')")
        # some genes are not available for a majority of cell models.
        # most are deprecated or low confidence genes. remove them.
        n_missed_models = rnai_effect_data.isna().sum(axis=1)
        rnai_models = n_missed_models[n_missed_models < 200].index
        rnai_effect_data = rnai_effect_data.loc[rnai_models].dropna(axis=1)
    else:
        rnai_effect_data = pd.DataFrame()

    # Copy Number Variation
    cnv_data = _read_cache_depmap(CNV_FILENAME, index_col=0)
    cnv_data.index.name = "ModelID"
    cnv_data = cnv_data.T
    cnv_data.index = [g.split(" ")[0] for g in cnv_data.index]

    # Mutations
    mutation_usecols = ["DepMap_ID", "HugoSymbol", "VariantInfo"]
    mutation_data = _read_cache_depmap(MUTATION_FILENAME, usecols=mutation_usecols)
    mutation_data = mutation_data.rename(columns={"DepMap_ID": "ModelID"})

    return crispr_effect_data, rnai_effect_data, cnv_data, mutation_data


def center_gene_effects(embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract the mean effect for each row.

    Parameters
    ----------
    embeddings : pd.DataFrame
        a dataframe with index as genes and column as features

    Returns
    -------
    pd.DataFrame after row centering
    """
    embeddings = embeddings.sub(embeddings.mean(axis=1), axis=0)
    return embeddings
