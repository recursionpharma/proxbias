import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def plot_bm_arm_bars(
    df: pd.DataFrame,
    f_name: str,
    palette: str='YlGnBu_r',
    format: str='png',
    legend: bool=True
):
    """
    Plotting function for barplot of Brunner-Munzel statistics per chromosome arm
    """
    colors = [sns.color_palette(palette)[i] for i in [1, 3, 4, 5]]
    if palette == 'YlGnBu_r':
        colors[0] = sns.color_palette()[0]
    if palette == 'YlGn_r':
        colors[0] = sns.color_palette()[2]
    patch1 = mpatches.Patch(color=colors[0], label='p<0.001')
    patch2 = mpatches.Patch(color=colors[1], label='p<0.01')
    patch3 = mpatches.Patch(color=colors[2], label='p<0.05')
    patch4 = mpatches.Patch(color=colors[3], label='p≥0.05')
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=1.8):
        bar_colors = []
        for arm in df.index:
            if df.loc[arm, 'bonf_p'] < 0.001:
                bar_colors.append(colors[0])
            elif df.loc[arm, 'bonf_p'] < 0.01:
                bar_colors.append(colors[1])
            elif df.loc[arm, 'bonf_p'] < 0.05:
                bar_colors.append(colors[2])
            else:
                bar_colors.append(colors[3])
        df.stat.plot(kind='bar', color=bar_colors)
        bar_ax = plt.gca()
        fig = plt.gcf()
        bar_ax.set_ylabel('P(intra-arm cos > inter)')
        bar_ax.set_xticklabels([xtl.get_text().replace('chr', '') for xtl in bar_ax.get_xticklabels()]);
        bar_ax.set_xticklabels(bar_ax.get_xticklabels(), rotation=0)
        bar_ax.set_ylim((0.4, 1))
        plt.plot(plt.gca().get_xlim(), [.5,.5], ':', c='k')
        if legend:
            bar_ax.legend(handles=[patch1, patch2, patch3, patch4], loc=(1.01, 0.3))
        for tick in bar_ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(25)
        fig.set_size_inches((15, 5))
        plt.gcf().set_facecolor('white')
        if f_name is not None:
            plt.savefig(f_name, dpi=600, format=format, bbox_inches = "tight")
            
def plot_bm_bar_pairs(
    df: pd.DataFrame, 
    x: str,
    y: str,
    hue: str,
    lab1: str,
    lab2: str,
    legend_loc: str='upper center',
    f_name: str='test.svg', 
    format: str='svg',
    i: int=0,
):
    """
    Plotting function for Brunner-Munzel statistics stratified by `hue` annotations

    Inputs:
    -------
    - df: dataframe with columns 'stat', and hue column
    - x: x column name
    - y: y column name
    - hue: hue column name
    - lab1: label for the first group
    - lab2: label for the second group
    - legend_loc: location of the legend
    - f_name: filename of output file
    - i: integer to offset color selection
    """
    palette = sns.color_palette('Paired')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=2):
        pal = [palette[i * 2], palette[i * 2 + 1]]
        barplot = sns.barplot(data=df, x=x, y=y, hue=hue, palette = pal, ax=ax);
        barplot.axhline(0.5, linestyle='--', color='grey');
        barplot.set_xticklabels([xtl.get_text().replace('chr', '') for xtl in barplot.get_xticklabels()]);
        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=0)
        for tick in barplot.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(30)
        barplot.set_xlabel('')
        patch1 = mpatches.Patch(color=pal[0], label=lab1)
        patch2 = mpatches.Patch(color=pal[1], label=lab2)
        line1 = mpl.lines.Line2D([0], [0], color='grey', lw=3, label='intra ≈ inter', linestyle='--')
        barplot.legend(handles=[patch1, patch2, line1], loc=legend_loc, fontsize=30)
        ax.set_ylim((0.4, 1))
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_ylabel(ax.get_ylabel(), fontsize=35)
    plt.subplots_adjust(hspace=0.4)
    plt.gcf().set_facecolor('white')
    ax.set_facecolor('white')
    fig.set_size_inches((26, 12))
    plt.savefig(f_name, format=format, bbox_inches = "tight")