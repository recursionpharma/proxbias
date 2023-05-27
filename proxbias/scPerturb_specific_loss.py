import os
import itertools
import wget
import re
import scanpy
import numpy as np
import pandas as pd
from proxbias import utils
import infercnvpy
from scanpy import AnnData


def get_telo_centro(arm: str, direction: str) -> str:
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
        >>> get_telo_centro('1p', '3prime')
        'centromeric'
        >>> get_telo_centro('2q', '5prime')
        'centromeric'
        >>> get_telo_centro('1p', '5prime')
        'telomeric'
        >>> get_telo_centro('Xq', '3prime')
        'telomeric'
    """
    if arm[-1] == "p":
        return "centromeric" if "3" in direction else "telomeric"
    if arm[-1] == "q":
        return "telomeric" if "3" in direction else "centromeric"


def compute_loss_w_specificity(
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

    loss_5p_cells = [[]] * len(loss)
    loss_5p_ko_cell_count = np.empty(len(loss))
    loss_5p_ko_cell_frac = np.empty(len(loss))
    loss_3p_cells = [[]] * len(loss)
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


def get_chromosome_info() -> pd.DataFrame:
    """
    Retrieve chromosome information for genes and return it as a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing chromosome information for genes,
            with gene names as the index and "chromosome" as the column name.

    """

    gene_dict, _, _ = utils.chromosome_info.get_chromosome_info_as_dicts()
    return pd.DataFrame.from_dict(gene_dict, orient="index").rename(columns={"chrom": "chromosome"})


def apply_infercnv_and_save_loss(
    anndat, filename: str, blocksize: int = 5, window: int = 100, neigh: int = 150
) -> None:
    """
    Apply infercnv to the provided AnnData object, compute loss with specificity, and save the results to a CSV file.

    Args:
        anndat (AnnData): AnnData object containing the data.
        filename (str): Name of the file to use in the result csv filename.
        blocksize (int): Block size for infercnv. Default is 5.
        window (int): Window size for infercnv. Default is 100.
        neigh (int): Number of neighboring blocks to consider. Default is 150.

    Returns:
        None

    """

    infercnvpy.tl.infercnv(
        anndat,
        reference_key="perturbation_label",
        reference_cat="control",
        window_size=window,
        step=blocksize,
        exclude_chromosomes=None,
    )
    res = compute_loss_w_specificity(anndat, blocksize, neigh)
    res_filename = f"infercnv_{filename}_b{blocksize}_w{window}_n{neigh}.csv"
    res.to_csv(os.path.join(utils.constants.DATA_DIR, res_filename))


def load_and_process_data(filename: str, chromosome_info: pd.DataFrame) -> AnnData:
    """
    Load and process AnnData object from the specified file prior to applying `infercnv()`

    Args:
        filename (str): Name of the file to load. Available options: "FrangiehIzar2021_RNA",
            "PapalexiSatija2021_eccite_RNA", "ReplogleWeissman2022_rpe1", "TianKampmann2021_CRISPRi",
            "AdamsonWeissman2016_GSM2406681_10X010".
        chromosome_info (pd.DataFrame): DataFrame containing gene chromosome information.

    Returns:
        AnnData: Processed data.
    """

    print(filename)
    source_path = f"https://zenodo.org/record/7416068/files/{filename}.h5ad?download=1"
    destination_path = f"{filename}.h5ad"
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


def generate_save_summary_results(
    filenames, blocksize: int = 5, window: int = 100, neigh: int = 150, zscore_cutoff: float = 3.0
):
    spec_genes_dict = {}
    for filename in filenames:
        res = pd.read_csv(f"f{filename}_b{blocksize}_w{window}_n{neigh}.csv", index_col=0)
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
        res = pd.read_csv(f"f{filename}_b{blocksize}_w{window}_n{neigh}.csv", index_col=0)
        filename_short = re.findall("[A-Z][^A-Z]*", filename)[0]
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
            tmp["Perturbation type"] = (
                "CRISPR-cas9" if filename_short in ["Frangieh", "Papalexi", "Dixit"] else "CRISPRi"
            )
            tmp["Tested loss direction"] = c.replace("p", "'")
            allres.append(tmp.sort_values("% affected cells", ascending=False))

    allres = pd.concat(allres)
    allres["Total # cells"] = allres.apply(lambda x: int(x["# affected cells"] / x["% affected cells"] * 100), axis=1)
    allres["Telomeric or centromeric"] = allres.apply(
        lambda x: get_telo_centro(x["Chr arm"], x["Tested loss direction"]), axis=1
    )
    allres = allres[
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
    allres.to_csv("allres.csv", index=None)

    gr_cols = ["Perturbation type", "Dataset", "Tested loss direction"]
    summaryres = (
        allres.groupby(gr_cols)
        .agg({"Telomeric or centromeric": [len, lambda x: sum(x == "telomeric"), lambda x: sum(x == "centromeric")]})
        .reset_index()
    )
    add_cols = [
        "# targets w/ specific loss",
        "# targets w/ loss towards telomere",
        "# targets w/ loss towards centromere",
    ]
    summaryres.columns = gr_cols + add_cols
    summaryres["Total # tested targets"] = summaryres["Dataset"].apply(lambda x: tested_gene_count_dict[x])
    summaryres["% targets w/ specific loss"] = summaryres.apply(
        lambda r: round((r["# targets w/ specific loss"] / r["Total # tested targets"]) * 100, 1), axis=1
    )
    summaryres = summaryres[gr_cols + ["% targets w/ specific loss", "Total # tested targets"] + add_cols]
    summaryres.to_csv("summaryres.csv", index=None)


def generate_plot_args():
    perts2check_d = {}
    plot_args = {}

    plot_args[filename] = {}
    ad = scanpy.read_h5ad(f"{filename}.h5ad")
    filename_short = re.findall("[A-Z][^A-Z]*", filename)[0]
    perts2check_df = allres[
        (allres["Dataset"] == filename_short) & (allres["# affected cells"] >= cell_count_thrs[filename_short])
    ]
    perts2check = sorted(set(perts2check_df["Perturbed gene"]))
    perts2check_d[filename] = perts2check

    ad.var = ad.var.rename(columns={"start": "st", "end": "en"}).join(all_genes_bed_HGNC, how="left")
    if filename.startswith("Papalexi"):
        ad.obs["gene"] = (
            ad.obs.perturbation.apply(lambda x: x.split("g")[0] if x != "control" else "").fillna("").astype(str)
        )
        print(ad.obs["gene"].unique())
    elif filename.startswith("Dixit"):
        ad = ad[
            (ad.obs.perturbation == "control")
            | ((ad.obs.perturbation.str.count("-") > 0) & ~ad.obs.perturbation.str.contains("INTERGENIC"))
        ].copy()
        ad.obs["gene"] = (
            ad.obs.perturbation.apply(lambda x: x.split("-")[1].split("sg")[1] if x != "control" else "")
            .fillna("")
            .astype(str)
        )
    elif filename.startswith("Frangieh"):
        ad.obs["gene"] = ad.obs.perturbation.apply(lambda x: x if x != "control" else "").fillna("").astype(str)
    ad.obs.loc[ad.obs.perturbation == "control", "gene"] = "control"
    ad = ad[ad.obs.gene.isin(perts2check + ["control"])]
    infercnvpy.tl.infercnv(
        ad, reference_key="gene", reference_cat="control", window_size=window, step=blocksize, exclude_chromosomes=None
    )

    res = pd.read_csv(f"f{filename}_b{blocksize}_w{window}_n{neigh}.csv", index_col=0)
    loss_arrs = []
    other_arrs = []
    loss_seps = []
    other_seps = []
    blocknums = []
    for p in perts2check:
        res_p = res[(res.ko_gene == p) & (res.aff_gene == p)]
        direcs = list(perts2check_df[perts2check_df["Perturbed gene"] == p]["Tested loss direction"])
        loss_cell_inds = sum(
            [[ad.obs.index.get_loc(x) for x in ast.literal_eval(res_p[f"loss{d[0]}p_cells"].iloc[0])] for d in direcs],
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

    plot_args[filename]["mid_tick"] = list(ad.uns["cnv"]["chr_pos"].values()) + [ad.obsm["X_cnv"].shape[1]]
    plot_args[filename]["x_tick_lab"] = list(ad.uns["cnv"]["chr_pos"].keys())
    plot_args[filename]["x_tick_loc"] = get_mid_ticks(
        list(ad.uns["cnv"]["chr_pos"].values()) + [ad.obsm["X_cnv"].shape[1]]
    )
    plot_args[filename]["chr_pos"] = ad.uns["cnv"]["chr_pos"].values()
    plot_args[filename]["loss_arrs"] = loss_arrs
    plot_args[filename]["other_arrs"] = other_arrs
    plot_args[filename]["loss_seps"] = loss_seps
    plot_args[filename]["other_seps"] = other_seps
    plot_args[filename]["blocknums"] = blocknums
    return plot_args