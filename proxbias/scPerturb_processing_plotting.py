import os
import itertools
import wget
from re import findall
import scanpy
import numpy as np
import pandas as pd
from proxbias import utils
import infercnvpy
from scanpy import AnnData
from scipy.stats import zscore
from typing import List, Optional
from ast import literal_eval


def _get_telo_centro(arm: str, direction: str) -> Optional[str]:
    """
    Determines the location of a genomic arm within a chromosome based on the chromosome number and direction.

    Args:
        arm (str): The genomic arm represented by the chromosome number followed by 'p' or 'q' (e.g., '1p', '2q', 'Xp').
        direction (str): The direction of the genomic arm (e.g., '5prime', '3prime').

    Returns:
        str: The location of the genomic arm within the chromosome, which can be either 'centromeric' or 'telomeric'.

    Raises:
        None

    Examples:
        >>> _get_telo_centro('1p', '3prime')
        'centromeric'
        >>> _get_telo_centro('2q', '5prime')
        'centromeric'
        >>> _get_telo_centro('1p', '5prime')
        'telomeric'
        >>> _get_telo_centro('Xq', '3prime')
        'telomeric'
    """
    if arm[-1] == "p":
        return "centromeric" if "3" in direction else "telomeric"
    if arm[-1] == "q":
        return "telomeric" if "3" in direction else "centromeric"
    return None


def _compute_loss_w_specificity(
    anndat: AnnData, blocksize: int, neighborhood_cnt: int = 150, frac_cutoff: float = 0.7, cnv_cutoff: float = -0.05
) -> pd.DataFrame:
    """
    Compute chromosomal loss with specificity based on the provided AnnData object.

    Args:
        anndat (AnnData): AnnData object containing the data.
        blocksize (int): Block size for computing the loss.
        neighborhood_cnt (int): Number of neighboring blocks to consider. Default is 150.
        frac_cutoff (float): Cutoff fraction for low CNV. Default is 0.7.
        cnv_cutoff (float): CNV cutoff value. Default is -0.05.

    Returns:
        pd.DataFrame: DataFrame containing the computed loss values.

    """
    avar = anndat.var
    cnvarr = anndat.obsm["X_cnv"].toarray() <= cnv_cutoff
    perturbed_genes = list(set(anndat.obs.gene).intersection(avar.index))
    ko_genes_to_look_at = perturbed_genes

    list_aff, list_ko = zip(*itertools.product(perturbed_genes, ko_genes_to_look_at))
    loss = pd.DataFrame({"ko_gene": list_ko, "aff_gene": list_aff})

    loss_5p_cells: List[List[str]] = [[]] * len(loss)
    loss_5p_ko_cell_count = np.empty(len(loss))
    loss_5p_ko_cell_frac = np.empty(len(loss))
    loss_3p_cells: List[List[str]] = [[]] * len(loss)
    loss_3p_ko_cell_count = np.empty(len(loss))
    loss_3p_ko_cell_frac = np.empty(len(loss))
    i5p = 0
    i3p = 0
    for aff_gene in perturbed_genes:
        aff_chr = avar.loc[aff_gene].chromosome
        if pd.isna(aff_chr):
            continue
        # get all genes in the same chromosome with the KO gene, sorted
        avari = avar[avar.chromosome == aff_chr].sort_values("start")
        # get position of the gene in the chromosome
        aff_gene_ordpos = avari.index.get_loc(aff_gene)
        # get block position of the gene in the chromosome
        aff_gene_blocknum_ = aff_gene_ordpos // blocksize
        # get total number of blocks in the chromosome
        total_chr_blocks = avari.shape[0] // blocksize
        # get start block of the chromosome gene is in
        aff_chr_startblocknum = anndat.uns["cnv"]["chr_pos"][aff_chr]
        # get end block of the chromosome gene is in
        aff_chr_endblocknum = aff_chr_startblocknum + total_chr_blocks
        # get block position of the gene in the genome
        aff_gene_blocknum = aff_chr_startblocknum + aff_gene_blocknum_

        block_count_5p = min(int(neighborhood_cnt / blocksize), aff_gene_blocknum - aff_chr_startblocknum)
        block_count_3p = min(int(neighborhood_cnt / blocksize), aff_chr_endblocknum - aff_gene_blocknum)

        blocks_5p = list(np.arange((aff_gene_blocknum - block_count_5p), aff_gene_blocknum))
        blocks_3p = list(np.arange(aff_gene_blocknum, (aff_gene_blocknum + block_count_3p)))

        for t, blocks in {"5p": blocks_5p, "3p": blocks_3p}.items():
            low_frac = np.sum(cnvarr[:, blocks], axis=1) / len(blocks)
            for ko_gene in ko_genes_to_look_at:
                ko_gene_ixs = anndat.obs.gene == ko_gene
                cells = list(anndat.obs.index[(low_frac >= frac_cutoff) & ko_gene_ixs])
                if t == "5p":
                    loss_5p_cells[i5p] = cells
                    loss_5p_ko_cell_count[i5p] = len(cells)
                    loss_5p_ko_cell_frac[i5p] = len(cells) / sum(ko_gene_ixs)
                    i5p += 1
                elif t == "3p":
                    loss_3p_cells[i3p] = cells
                    loss_3p_ko_cell_count[i3p] = len(cells)
                    loss_3p_ko_cell_frac[i3p] = len(cells) / sum(ko_gene_ixs)
                    i3p += 1

    loss["loss5p_cells"] = loss_5p_cells
    loss["loss3p_cells"] = loss_3p_cells
    loss["loss5p_cellcount"] = loss_5p_ko_cell_count
    loss["loss3p_cellcount"] = loss_3p_ko_cell_count
    loss["loss5p_cellfrac"] = loss_5p_ko_cell_frac
    loss["loss3p_cellfrac"] = loss_3p_ko_cell_frac

    loss["ko_chr"] = loss.ko_gene.apply(lambda x: avar.loc[x].chromosome)
    loss["ko_arm"] = loss.ko_gene.apply(lambda x: avar.loc[x].arm)
    loss["aff_chr"] = loss.aff_gene.apply(lambda x: avar.loc[x].chromosome)
    loss["aff_arm"] = loss.aff_gene.apply(lambda x: avar.loc[x].arm)

    return loss


def _get_chromosome_info() -> pd.DataFrame:
    """
    Retrieve chromosome information for genes and return it as a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing chromosome information for genes,
            with gene names as the index and "chromosome" as the column name.

    """
    gene_dict, _, _ = utils.chromosome_info.get_chromosome_info_as_dicts()
    return pd.DataFrame.from_dict(gene_dict, orient="index").rename(columns={"chrom": "chromosome"})


def apply_infercnv_and_save_loss(filename: str, blocksize: int = 5, window: int = 100, neigh: int = 150) -> str:
    """
    Apply infercnv and compute loss with specificity on the given data file, and save the results.
    The function loads and processes the data file using the _load_and_process_data() function,
    applies infercnv analysis with the specified parameters, computes loss with specificity,
    and saves the results to a CSV file.
    The result file is saved in the directory specified by utils.constants.DATA_DIR with a name
    generated using the _make_infercnv_result_file_name() function, which incorporates the original
    filename, blocksize, window, and neigh values. If the result file already exists, the function
    skips the infercnv and loss computation steps.

    Args:
        filename (str): Name of the file to process.
        blocksize (int, optional): Block size for infercnv analysis. Default is 5.
        window (int, optional): Window size for infercnv analysis. Default is 100.
        neigh (int, optional): Number of nearest neighbors for specificity computation. Default is 150.

    Returns:
        Result file path.
    """
    res_path = os.path.join(
        str(utils.constants.DATA_DIR), _make_infercnv_result_file_name(filename, blocksize, window, neigh)
    )
    if not os.path.exists(res_path):
        anndat = _load_and_process_data(filename)
        infercnvpy.tl.infercnv(
            anndat,
            reference_key="perturbation_label",
            reference_cat="control",
            window_size=window,
            step=blocksize,
            exclude_chromosomes=None,
        )
        res = _compute_loss_w_specificity(anndat, blocksize, neigh)
        res.to_csv(res_path)
    return res_path


def _load_and_process_data(filename: str, chromosome_info: pd.DataFrame = None) -> AnnData:
    """
    Load and process the specified file prior to applying `infercnv()`
    The result of the processing is an AnnData object with a 'perturbation_label'
    key specifying the reference category for infercnv analysis.

    Args:
        filename (str): Name of the file to load. Available options: "FrangiehIzar2021_RNA",
            "PapalexiSatija2021_eccite_RNA", "ReplogleWeissman2022_rpe1", "TianKampmann2021_CRISPRi",
            "AdamsonWeissman2016_GSM2406681_10X010".
        chromosome_info (pd.DataFrame, optional): DataFrame containing gene chromosome information.
            Default is None, in which case we use get_chromosome_info() to pull the requried information.

    Returns:
        AnnData: Processed data.
    """
    if chromosome_info is None:
        chromosome_info = _get_chromosome_info()
    print(filename)
    destination_path = os.path.join(str(utils.constants.DATA_DIR), f"{filename}.h5ad")
    if not os.path.exists(destination_path):
        source_path = f"https://zenodo.org/record/7416068/files/{filename}.h5ad?download=1"
        wget.download(source_path, destination_path)
    ad = scanpy.read_h5ad(destination_path)
    ad.var = ad.var.rename(columns={"start": "st", "end": "en"}).join(chromosome_info, how="left")
    if filename.startswith("Adamson"):
        ad.obs["gene"] = ad.obs.perturbation.apply(lambda x: x.split("_")[0]).fillna("")
        ad.obs["chromosome"] = ad.obs.gene.apply(lambda x: chromosome_info.chromosome.get(x, "")).fillna("")
        ad.obs["perturbation_label"] = ad.obs["chromosome"]
        ad.obs["perturbation_label"].loc[pd.isna(ad.obs.perturbation)] = "control"
    elif filename.startswith("Papalexi"):
        ad.obs["gene"] = ad.obs.perturbation.apply(lambda x: x.split("g")[0] if x != "control" else "").fillna("")
        ad.obs["chromosome"] = ad.obs.gene.apply(lambda x: chromosome_info.chromosome.get(x, "")).fillna("")
        ad.obs["perturbation_label"] = ad.obs["chromosome"]
        ad.obs.loc[ad.obs.perturbation == "control", "perturbation_label"] = "control"
    elif filename.startswith("Replogle"):
        ad.obs["gene"] = ad.obs["gene"].apply(lambda x: x if x != "non-targeting" else "")
        ad.obs["chromosome"] = ad.obs.gene.apply(lambda x: ad.var.chromosome.get(x, "")).fillna("")
        ad.obs["perturbation_label"] = ad.obs["chromosome"]
        ad.obs.loc[ad.obs.perturbation == "control", "perturbation_label"] = "control"
    elif filename.startswith("Frangieh") or filename.startswith("Tian"):
        ad.obs["gene"] = ad.obs.perturbation.apply(lambda x: x if x != "control" else "").fillna("")
        ad.obs["chromosome"] = ad.obs.gene.apply(lambda x: chromosome_info.chromosome.get(x, "")).fillna("")
        ad.obs["perturbation_label"] = ad.obs["chromosome"]
        ad.obs.loc[ad.obs.perturbation == "control", "perturbation_label"] = "control"
    return ad[ad.obs.perturbation_label != ""]


def _make_infercnv_result_file_name(filename: str, blocksize: int, window: int, neigh: int) -> str:
    """
    Constructs the filename for the infercnv result file using the given filename, blocksize, window,
    and neigh values. The resulting filename follows the format "{filename}_b{blocksize}_w{window}_n{neigh}.csv".

    Args:
        filename (str): The base filename.
        blocksize (int): The blocksize value.
        window (int): The window value.
        neigh (int): The neigh value.

    Returns:
        str: The generated infercnv result filename.
    """
    return f"{filename}_b{blocksize}_w{window}_n{neigh}.csv"


def generate_save_summary_results(filenames: List[str], zscore_cutoff: float = 3.0) -> None:
    """
    Generates and saves summary chromosomal loss results from a list of multiple infercnv result files.
    The function reads CSV files based on the filenames and performs data processing and aggregation to generate
    summary results. The results are saved in CSV format.

    Args:
        filenames (List[str]): A list of filenames to process and generate summary results for.
        zscore_cutoff (float, optional): The z-score cutoff value for filtering specific loss. Defaults to 3.0.

    Returns:
        None

    Raises:
        FileNotFoundError: If the CSV file for a given filename is not found.

    Example usage:
        >>> filenames = ["PapalexiSatija2021_eccite_RNA", "TianKampmann2021_CRISPRi"]
        >>> generate_save_summary_results(filenames)
    """

    spec_genes_dict = {}
    for filename in filenames:
        res = pd.read_csv(apply_infercnv_and_save_loss(filename), index_col=0)
        for c in ["3p", "5p"]:
            col = f"loss{c}_cellfrac"
            col2 = f"loss{c}_cellcount"
            res_c = res[["aff_gene", "ko_gene", col, col2]]
            res_trans = res_c.copy()
            res_trans[col] = res_trans.groupby("aff_gene")[col].transform(lambda x: zscore(x))
            res_trans = (
                res_trans[(res_trans.aff_gene == res_trans.ko_gene) & (res_trans[col2] >= 1)]
                .sort_values(by=col)
                .reset_index(drop=True)
            )
            res_trans = res_trans[res_trans[col] >= zscore_cutoff]
            spec_genes = list(res_trans.ko_gene)
            spec_genes_dict[(filename, c)] = spec_genes

    tested_gene_count_dict = {}
    allres = []
    for filename in filenames:
        res = pd.read_csv(apply_infercnv_and_save_loss(filename), index_col=0)
        filename_short = _get_short_filename(filename)
        tested_gene_count_dict[filename_short] = len(res.aff_gene.unique())
        for c in ["3p", "5p"]:
            col = f"loss{c}_cellfrac"
            col2 = f"loss{c}_cellcount"
            res_c = res[["aff_gene", "aff_arm", "ko_gene", "ko_arm", col, col2]]
            specific_loss = res_c[
                (res_c.aff_gene == res_c.ko_gene) & res_c.aff_gene.isin(spec_genes_dict[(filename, c)])
            ]
            tmp = specific_loss[["ko_gene", "ko_arm", col, col2]].rename(
                columns={
                    col: "% affected cells",
                    col2: "# affected cells",
                    "ko_gene": "Perturbed gene",
                    "ko_arm": "Chr arm",
                }
            )
            tmp["% affected cells"] = tmp["% affected cells"].apply(lambda x: round(x * 100, 2))
            tmp["Dataset"] = filename_short
            tmp["Perturbation type"] = "CRISPR-cas9" if filename_short in ["Frangieh", "Papalexi"] else "CRISPRi"
            tmp["Tested loss direction"] = c.replace("p", "'")
            allres.append(tmp.sort_values("% affected cells", ascending=False))

    allres_df = pd.concat(allres)
    allres_df["Total # cells"] = allres_df.apply(
        lambda x: int(x["# affected cells"] / x["% affected cells"] * 100), axis=1
    )
    allres_df["Telomeric or centromeric"] = allres_df.apply(
        lambda x: _get_telo_centro(x["Chr arm"], x["Tested loss direction"]), axis=1
    )
    allres_df = allres_df[
        [
            "Perturbed gene",
            "Perturbation type",
            "Dataset",
            "Chr arm",
            "Tested loss direction",
            "Total # cells",
            "# affected cells",
            "% affected cells",
            "Telomeric or centromeric",
        ]
    ]
    allres_df.to_csv(os.path.join(str(utils.constants.DATA_DIR), "allres.csv"), index=None)

    gr_cols = ["Perturbation type", "Dataset", "Tested loss direction"]
    summaryres_df = (
        allres_df.groupby(gr_cols)
        .agg({"Telomeric or centromeric": [len, lambda x: sum(x == "telomeric"), lambda x: sum(x == "centromeric")]})
        .reset_index()
    )
    add_cols = [
        "# targets w/ specific loss",
        "# targets w/ loss towards telomere",
        "# targets w/ loss towards centromere",
    ]
    summaryres_df.columns = gr_cols + add_cols
    summaryres_df["Total # tested targets"] = summaryres_df["Dataset"].apply(lambda x: tested_gene_count_dict[x])
    summaryres_df["% targets w/ specific loss"] = summaryres_df.apply(
        lambda r: round((r["# targets w/ specific loss"] / r["Total # tested targets"]) * 100, 1), axis=1
    )
    summaryres_df = summaryres_df[gr_cols + ["% targets w/ specific loss", "Total # tested targets"] + add_cols]
    summaryres_df.to_csv(os.path.join(str(utils.constants.DATA_DIR), "summaryres.csv"), index=None)


def _get_short_filename(filename: str) -> str:
    """
    Retrieves the short filename from a given filename by extracting the sequence starting with a capital letter
    until the next capital letter is encountered.

    Args:
        filename (str): The original filename from which the short filename will be extracted.

    Returns:
        str: The short filename containing the first sequence of uppercase letters found in the given filename,
             or an empty string if no uppercase letters are present.

    Raises:
        IndexError: If no uppercase letters are found in the given filename.

    Examples:
        >>> _get_short_filename("PapalexiSatija2021_eccite_RNA")
        'Papalexi'
        >>> _get_short_filename("TianKampmann2021_CRISPRi")
        'Tian'
        >>> _get_short_filename("data_file.csv")
        Raises IndexError since there are no capital letters
    """
    return findall("[A-Z][^A-Z]*", filename)[0]


def _get_mid_ticks(lst: List[int]) -> List[float]:
    """
    Takes a list of values and calculates the middle ticks by averaging each value with its preceding
    value and returns a list of the calculated middle ticks.
    Assumes that `lst` includes 0 as the first item.

    Args:
        lst (List[int]): A list of values.

    Returns:
        List[int]: A list of calculated middle ticks.

    Examples:
        >>> ticks = _get_mid_ticks([0, 10, 20, 30, 40])
        >>> print(ticks)

    """
    return [(lst[i] - lst[i - 1]) / 2 + lst[i - 1] for i in range(1, len(lst))]


def _get_cell_count_threshold(filename_short: str) -> int:
    """
    Retrieves the cell count threshold based on the given filename.

    The function returns the cell count threshold based on the provided short filename. The threshold values are defined
    in a dictionary where the short filename serves as the key and the corresponding threshold as the value.

    Args:
        filename_short (str): The short filename to retrieve the cell count threshold for.

    Returns:
        int: The cell count threshold.

    Raises:
        KeyError: If the provided short filename is not found in the dictionary.

    Examples:
        >>> threshold = _get_cell_count_threshold("Frangieh")
        >>> print(threshold)
    """

    return {"Frangieh": 20, "Papalexi": 10}[filename_short]


def generate_plot_args(
    filename: str,
    chromosome_info: pd.DataFrame = None,
    blocksize: int = 5,
    window: int = 100,
    neigh: int = 150,
) -> dict:
    """
    Generates plot arguments for visualization based on a given filename and processing parameters.
    The function reads an H5AD file based on the filename, performs data processing, and extracts plot arguments
    for visualization of the chromosomal loss. The resulting plot arguments are returned as a dictionary.
    Note that although this functions reads the chromosomal loss from the `allres` parameter, we rerun infercnvpy
    for the subset of genes that are in allres.csv in order to retrieve the actual infercnv values for the visualization.
    Also note that this function requires _apply_infercnv_and_save_loss(filename) to be called prior so that we have summary result files from which we
    pull the genes to rerun infercnv on and plot.

    Args:
        filename (str): The name of the file to process.
        chromosome_info (pd.DataFrame, optional): DataFrame containing gene chromosome information.
        blocksize (int, optional): The block size parameter for processing. Defaults to 5.
        window (int, optional): The window size parameter for processing. Defaults to 100.
        neigh (int, optional): The neighborhood count parameter for processing. Defaults to 150.

    Returns:
        dict: A dictionary containing the plot arguments for visualization.

    Raises:
        FileNotFoundError: If the H5AD file for the given filename or allres.csv is not found.
    """
    if chromosome_info is None:
        chromosome_info = _get_chromosome_info()
    allres = pd.read_csv(os.path.join(str(utils.constants.DATA_DIR), "allres.csv"))
    plot_args = {}
    ad = scanpy.read_h5ad(os.path.join(str(utils.constants.DATA_DIR), f"{filename}.h5ad"))
    filename_short = _get_short_filename(filename)
    perts2check_df = allres[
        (allres["Dataset"] == filename_short)
        & (allres["# affected cells"] >= _get_cell_count_threshold(filename_short))
    ]
    perts2check = sorted(set(perts2check_df["Perturbed gene"]))

    ad.var = ad.var.rename(columns={"start": "st", "end": "en"}).join(chromosome_info, how="left")
    if filename_short == "Papalexi":
        ad.obs["gene"] = (
            ad.obs.perturbation.apply(lambda x: x.split("g")[0] if x != "control" else "").fillna("").astype(str)
        )
    elif filename_short == "Frangieh":
        ad.obs["gene"] = ad.obs.perturbation.apply(lambda x: x if x != "control" else "").fillna("").astype(str)
    ad.obs.loc[ad.obs.perturbation == "control", "gene"] = "control"
    ad = ad[ad.obs.gene.isin(perts2check + ["control"])]
    infercnvpy.tl.infercnv(
        ad, reference_key="gene", reference_cat="control", window_size=window, step=blocksize, exclude_chromosomes=None
    )

    res = pd.read_csv(_make_infercnv_result_file_name(filename, blocksize, window, neigh), index_col=0)
    loss_arrs: List[float] = []
    other_arrs: List[float] = []
    loss_seps: List[int] = []
    other_seps: List[int] = []
    blocknums = []
    for p in perts2check:
        res_p = res[(res.ko_gene == p) & (res.aff_gene == p)]
        direcs = list(perts2check_df[perts2check_df["Perturbed gene"] == p]["Tested loss direction"])
        loss_cell_inds: List[str] = sum(
            [[ad.obs.index.get_loc(x) for x in literal_eval(res_p[f"loss{d[0]}p_cells"].iloc[0])] for d in direcs],
            [],
        )
        other_cell_inds = list(
            set(ad.obs.index.get_loc(x) for x in ad.obs.loc[ad.obs.gene == p].index).difference(loss_cell_inds)
        )
        arr1 = ad.obsm["X_cnv"].toarray()[loss_cell_inds]
        arr2 = ad.obsm["X_cnv"].toarray()[other_cell_inds]
        aff_chr = ad.var.loc[p].chromosome
        aff_chr_startblocknum = ad.uns["cnv"]["chr_pos"][aff_chr]
        blocknum_p = (
            aff_chr_startblocknum
            + ad.var[ad.var.chromosome == aff_chr].sort_values("start").index.get_loc(p) // blocksize
        )
        blocknums.append(blocknum_p)
        loss_arrs = arr1 if len(loss_arrs) == 0 else np.concatenate((loss_arrs, arr1), axis=0)
        other_arrs = arr2 if len(other_arrs) == 0 else np.concatenate((other_arrs, arr2), axis=0)
        loss_seps.append(len(loss_cell_inds) if len(loss_seps) == 0 else len(loss_cell_inds) + loss_seps[-1])
        other_seps.append(len(other_cell_inds) if len(other_seps) == 0 else len(other_cell_inds) + other_seps[-1])

    plot_args["x_tick_lab"] = list(ad.uns["cnv"]["chr_pos"].keys())
    plot_args["chr_pos"] = ad.uns["cnv"]["chr_pos"].values()
    plot_args["loss_arrs"] = loss_arrs
    plot_args["other_arrs"] = other_arrs
    plot_args["loss_seps"] = loss_seps
    plot_args["other_seps"] = other_seps
    plot_args["blocknums"] = blocknums
    return plot_args


def plot_losses(filenames):
    sns.set(font_scale=1.7)
    plt.rcParams["svg.fonttype"] = "none"
    for filename in filenames:
        p_args = generate_plot_args(filename)
        crunch = 10 if filename == "PapalexiSatija2021_eccite_RNA" else 30  # Set to get the file size below 25Mb
        plt.figure(figsize=[20, int(p_args["loss_seps"][-1] / 15)])
        tmp = block_reduce(p_args["loss_arrs"], (1, crunch), np.mean)
        ax = sns.heatmap(
            tmp, cmap="seismic", center=0, cbar_kws=dict(use_gridspec=False, location="top", shrink=0.5, pad=0.01)
        )
        ax.set_xticks([x / crunch for x in p_args["x_tick_loc"]])
        loss_seps_tmp = [0] + p_args["loss_seps"]
        ax.set_yticks(_get_mid_ticks(loss_seps_tmp))
        ax.set_xticklabels(p_args["x_tick_lab"])
        ax.set_yticklabels(perts2check_d[filename])
        ax.hlines(loss_seps_tmp, *ax.get_xlim())
        for i in range(len(p_args["blocknums"])):
            ax.vlines(
                p_args["blocknums"][i] / crunch, loss_seps_tmp[i], loss_seps_tmp[i + 1], color="lime", linewidth=3
            )
        for j in p_args["chr_pos"]:
            ax.vlines(j / crunch, *ax.get_ylim())
        plt.gcf().set_facecolor("white")
        plt.savefig(f"{filename}.svg", format="svg", bbox_inches="tight")
        plt.show()
