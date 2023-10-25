from functools import partial
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_stat_arm_bars(
    df: pd.DataFrame,
    stat_col: str,
    pval_col: str,
    ylabel: str,
    title: Optional[str] = None,
    figsize: tuple = (15, 5),
    palette: str = "YlGnBu_r",
    legend: bool = True,
    ylim: Optional[Tuple[int, int]] = None,
    ref_line_yval: Optional[float] = None,
    ref_line_label: str = "_nolegend_",
    f_name: Optional[str] = None,
    fmt: str = "png",
):
    """
    Plotting function for barplot of a statistical test result per chromosome arm.

    Inputs:
    -------
    - df: DataFrame with values to be plotted. Index should contain chromosome
        arms in the desired order for plotting. Should contain a column with
        test statistics and a column with p-values.
    - stat_col: column of `df` containing the statistic to plot as bars
    - pval_col: column of `df` containing p-values to be used for significance
    - ylabel: y-axis label for the plot, describing the statistic in `stat_col`
    - title: optional title for figure
    - figsize: tuple for size of the figure
    - palette: Name of Seaborn palette to use
    - legend: whether to show a legend
    - ylim: optional specified y-axis limits
    - ref_line_yval: value for optional horizontal reference line
    - ref_line_label: optional label for horizontal reference line
    - f_name: optional file name to save figure. If None, the figure is not saved.
    - fmt: format of file to save figure, e.g. 'png', 'svg', etc.
    """
    colors = [sns.color_palette(palette)[i] for i in [1, 3, 4, 5]]
    if palette == "YlGnBu_r":
        colors[0] = sns.color_palette()[0]
    if palette == "YlGn_r":
        colors[0] = sns.color_palette()[2]
    patch1 = mpatches.Patch(color=colors[0], label="p<0.001")
    patch2 = mpatches.Patch(color=colors[1], label="p<0.01")
    patch3 = mpatches.Patch(color=colors[2], label="p<0.05")
    patch4 = mpatches.Patch(color=colors[3], label="p≥0.05")
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=1.5):
        bar_colors = []
        for arm in df.index:
            if df.loc[arm, pval_col] < 0.001:
                bar_colors.append(colors[0])
            elif df.loc[arm, pval_col] < 0.01:
                bar_colors.append(colors[1])
            elif df.loc[arm, pval_col] < 0.05:
                bar_colors.append(colors[2])
            else:
                bar_colors.append(colors[3])
        fig, bar_ax = plt.subplots()
        df[stat_col].plot(kind="bar", color=bar_colors, ax=bar_ax)
        bar_ax.set_ylabel(ylabel)
        bar_ax.set_title(title)
        bar_ax.set_xticklabels([xtl.get_text().replace("chr", "") for xtl in bar_ax.get_xticklabels()])
        bar_ax.set_xticklabels(bar_ax.get_xticklabels(), rotation=0)
        if ylim is not None:
            bar_ax.set_ylim(ylim)
        if ref_line_yval is not None:
            bar_ax.plot(bar_ax.get_xlim(), [ref_line_yval, ref_line_yval], linestyle=":", c="k", label=ref_line_label)
        if legend:
            bar_ax.legend(handles=[patch1, patch2, patch3, patch4], loc=(1.01, 0.3))
        for tick in bar_ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(25)
        fig.set_size_inches(figsize)  # type: ignore[arg-type]
        fig.set_facecolor("white")
        if f_name is not None:
            plt.savefig(f_name, dpi=600, format=fmt, bbox_inches="tight")


plot_bm_arm_bars = partial(
    plot_stat_arm_bars,
    stat_col="prob",
    pval_col="bonf_p",
    ylabel="P(intra-arm cos > inter)",
    ylim=(0.4, 1),
    ref_line_yval=0.5,
    ref_line_label="intra ≈ inter",
)


def plot_bm_bar_pairs(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    lab1: str,
    lab2: str,
    figsize: tuple = (15, 5),
    legend_loc: str = "upper center",
    f_name: str = "test.svg",
    fmt: str = "svg",
    i: int = 0,
):
    """
    Plotting function for Brunner-Munzel probabilities stratified by `hue` annotations

    Inputs:
    -------
    - df: dataframe with columns 'prob', and hue column
    - x: x column name
    - y: y column name
    - hue: hue column name
    - lab1: label for the first group
    - lab2: label for the second group
    - figsize: tuple for size of the figure
    - legend_loc: location of the legend
    - f_name: filename of output file
    - i: integer to offset color selection
    """
    palette = sns.color_palette("Paired")
    fig, ax = plt.subplots(nrows=1, ncols=1)
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=2):
        pal = [palette[i * 2], palette[i * 2 + 1]]
        barplot = sns.barplot(data=df, x=x, y=y, hue=hue, palette=pal, ax=ax)
        barplot.axhline(0.5, linestyle="--", color="grey")
        barplot.set_xticklabels([xtl.get_text().replace("chr", "") for xtl in barplot.get_xticklabels()])
        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=0)
        for tick in barplot.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(30)
        barplot.set_xlabel("")
        patch1 = mpatches.Patch(color=pal[0], label=lab1)
        patch2 = mpatches.Patch(color=pal[1], label=lab2)
        line1 = mpl.lines.Line2D([0], [0], color="grey", lw=3, label="intra ≈ inter", linestyle="--")
        barplot.legend(handles=[patch1, patch2, line1], loc=legend_loc, fontsize=15)
        ax.set_ylim((0.4, 1))
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
    plt.subplots_adjust(hspace=0.4)
    plt.gcf().set_facecolor("white")
    ax.set_facecolor("white")
    fig.set_size_inches(figsize)  # type: ignore[arg-type]
    plt.savefig(f_name, format=fmt, bbox_inches="tight")
