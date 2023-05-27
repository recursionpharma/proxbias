import os
import itertools
import wget
import scanpy
import numpy as np
import pandas as pd
import infercnvpy
from proxbias import utils


def compute_loss_w_specificity(anndat, blocksize, neighborhood_cnt=150, frac_cutoff=0.7, cnv_cutoff=-0.05):
    """
    Compute loss with specificity based on the provided AnnData object.

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


def get_chromosome_info():
    """
    Retrieve chromosome information for genes and return it as a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing chromosome information for genes,
            with gene names as the index and "chromosome" as the column name.

    """

    gene_dict, _, _ = utils.chromosome_info.get_chromosome_info_as_dicts()
    return pd.DataFrame.from_dict(gene_dict, orient="index").rename(columns={"chrom": "chromosome"})


def apply_infercnv_and_save_loss(anndat, filename, blocksize=5, window=100, neigh=150):
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


def load_and_process_data(filename: AnnData, chromosome_info):
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
