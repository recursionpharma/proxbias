from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import block_reduce


def crunch_square_df(
    sims: pd.DataFrame,
    crunch_factor: int,
) -> pd.DataFrame:
    """
    Compress `sims` dataframe by `crunch_factor` to make visualizations reasonable. This takes averages of squares of
    size `crunch_factor` X `crunch_factor`. Indices are replaced by the first value in the crunch block.

    Inputs:
    -------
    - sims: pd.DataFrame() with matching row and column indices
    - crunch_fctor: int to compress the data by.
    """

    idx = [(i % crunch_factor == 0) for i, x in enumerate(sims.index)]
    new_index = sims.index[idx]  # type: ignore
    crunched = block_reduce(sims.values, (crunch_factor, crunch_factor), np.mean)

    return pd.DataFrame(crunched, index=new_index, columns=new_index)


def plot_heatmap(
    sims: pd.DataFrame,
    f_name: Optional[str] = None,
    format: str = "png",
    crunch_factor: int = 1,
    show_chr_lines: bool = True,
    show_cent_lines: bool = True,
    show_chroms: bool = True,
    show_chrom_arms: bool = False,
    figsize: tuple = (20, 20),
    title: Optional[str] = None,
    label_locy: Optional[float] = None,
    lab_s: int = 12,
    drop_chry: bool = True,
    show_cbar: bool = True,
    lw: float = 0.5,
    lab_rot: int = 0,
):
    """
    Plotting function for heatmaps (full-genome or subsets) can be sorted/clustered or ordered by chromosome

    Inputs:
    - sims: Square dataframe with matching row and column indices and similarity values. This can be "split" along
            the diagonal to show different datasets. Index should include `chromosome` and `chromosome_arm` if
            ordering by genomic position. Each row/column should represent one gene ordered by genomic position
    - f_name: file name
    - format: file format for saving a file
    - crunch_factor: if > 1 will apply a average smoothing to reduce the size of the output file
    - show_chr_lines: whether to show lines at chromosome boundaries
    - show_cent_lines: whether to show lines at centromeres
    - show_chroms: Whether to label chromosomes on top and right
    - show_chrom_arms: whether to label chromosome arms on top and right
    - figsize: size of the resulting figure
    - title: plot title
    - label_locy: location of labels in y
    - lab_s: Font size of labels
    - drop_chry: Whether to remove Chromosome Y values
    - show_cbar: whether to show the color bar
    - lw: line width
    - lab_rot: rotation of labels
    """
    color_norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = mpl.colormaps["RdBu_r"]
    # This sets nan values to white
    # cmap.set_bad(color='white')

    if crunch_factor > 1:
        # Downsample the data to make the file size less crazy
        # Every `sample_factor`th row/column will be kept
        sims = crunch_square_df(sims, crunch_factor=crunch_factor)

    image_data = cmap(color_norm(sims.values))

    plt.figure(figsize=figsize)  # type: ignore[arg-type]
    plt.imshow(image_data)
    if show_cbar:
        plt.colorbar(mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap), shrink=0.5)

    if drop_chry and "chromosome" in sims.index.names:
        noy_idx = sims.index.get_level_values("chromosome") != "chrY"
        sims = sims.copy().loc[noy_idx, noy_idx]  # type: ignore

    if show_chr_lines or show_chroms or show_cent_lines or show_chrom_arms:
        # Get the position of all the chromosomes and centromeres
        index_df = sims.index.to_frame(index=False).reset_index().rename({"index": "pos"}, axis=1)
        chr_pos = index_df.groupby("chromosome").pos.max().sort_values()
        cent_pos = index_df.groupby("chromosome_arm").pos.max().sort_values()
        # Filter to just p-arms
        cent_pos_p = cent_pos[[x[-1] == "p" for x in cent_pos.index]]
        # Get midpoints for annotations
        chr_mids = pd.DataFrame(
            (np.insert(chr_pos.values[:-1], 0, 0) + chr_pos.values) / 2, index=chr_pos.index  # type: ignore
        ).to_dict()[
            0  # type: ignore
        ]
        cent_mids = pd.DataFrame(
            (np.insert(cent_pos.values[:-1], 0, 0) + cent_pos.values) / 2, index=cent_pos.index  # type: ignore
        ).to_dict()[
            0  # type: ignore
        ]

    # Hide X and Y axes label marks
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    xm, xM = ax.get_xlim()
    ym, yM = ax.get_ylim()

    if show_chr_lines:
        for x in chr_pos.values:
            plt.plot([x + 0.5, x + 0.5], [ym, yM], color="k", ls="-", lw=lw)
            plt.plot([xm, xM], [x + 0.5, x + 0.5], color="k", ls="-", lw=lw)
    if show_cent_lines:
        for x in cent_pos_p.values:  # type: ignore[attr-defined]
            plt.plot([x + 0.5, x + 0.5], [ym, yM], color="k", ls=":", lw=lw)
            plt.plot([xm, xM], [x + 0.5, x + 0.5], color="k", ls=":", lw=lw)

    if show_chroms:
        # Label chromosomes on top/right to not clash with coords
        ax = plt.gca()
        s = sims.shape[0]
        for ch in chr_mids:
            # Labels across the top
            if label_locy is None:
                label_locy = -0.008 * s
            # Labels on top
            ax.text(
                chr_mids[ch], label_locy, ch.replace("chr", ""), ha="center", va="bottom", rotation=lab_rot, size=lab_s
            )
            # Labels on the right
            ax.text(sims.shape[0] + 0.008 * s, chr_mids[ch], ch.replace("chr", ""), ha="left", va="center", size=lab_s)

    if show_chrom_arms:
        # Label chromosome arms on top/right
        ax = plt.gca()
        s = sims.shape[0]
        for cent in cent_mids:
            # Labels across the top
            if label_locy is None:
                label_locy = -0.008 * s
            ax.text(cent_mids[cent], label_locy, cent, ha="left", rotation=lab_rot, size=lab_s)
            # Labels on the right
            ax.text(sims.shape[0] + 0.001 * s, cent_mids[cent], cent, ha="left", size=lab_s)
    if title:
        plt.title(title, size="xx-large", y=1.1)
    plt.gcf().set_facecolor("white")
    if f_name is not None:
        plt.savefig(f_name, dpi=600, format=format, bbox_inches="tight")
