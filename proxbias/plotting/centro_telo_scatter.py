from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_centromere_telomere_scatterplots(
    bm_per_gene_df: pd.DataFrame,
    arms_to_plot: List[str],
    f_name: Optional[str] = None,
    fmt: str = "png",
):
    """
    Create and optionally save a scatterplot of per-gene Brunner-Munzel statistics
     for one or more chromosome arms as a function of normalized centromere-to-
     telomere rank.

    Inputs:
    -------
    - bm_per_gene_df: Brunner-Munzel statistics and locations for each gene.
        Index levels or columns should contain
        - `chromosome_arm`
        - `gene_bp`
    - arms_to plot: Which chromosome arms to include in the plot.
    - f_name: optional file name to save figure. If None, the figure is not saved.
    - fmt: format of file to save figure, e.g. 'png', 'svg', etc.
    """
    palette = iter(sns.color_palette())
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=1.75):
        fig, ax = plt.subplots()
        fig.set_size_inches((10, 8))
        for arm in arms_to_plot:
            arm_df = bm_per_gene_df.query(f'chromosome_arm == "chr{arm}"')
            sorted_arm_df = arm_df.sort_index(
                level="gene_bp",
                ascending=arm[-1] == "q",
            )
            color = next(palette)
            bm_stats = sorted_arm_df.prob
            x = np.arange(len(bm_stats)) / len(bm_stats)
            y = bm_stats
            sns.regplot(
                x=x,
                y=y,
                color=color,
                label=f"chr{arm}",
                lowess=True,
                line_kws={"linewidth": 4},
                ax=ax,
            )
        ax.set_xlabel("Normalized centromere-to-telomere rank")
        ax.set_ylabel("P(intra arm cosine > inter)")
        ax.legend()
        ax.set_title("Gene-level proximity bias scores")
        fig.set_facecolor("white")
        if f_name is not None:
            plt.savefig(f_name, dpi=600, format=fmt, bbox_inches="tight")
