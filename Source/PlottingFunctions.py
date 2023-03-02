"""
Functions used for plotting of microphone calibration and measurement DataFrames.
"""
import inspect
import functools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from matplotlib.ticker import MultipleLocator
from typing import Union, Optional, Tuple, Sequence, Any, Callable, Dict
from seaborn.axisgrid import PairGrid
import Source.ProcessingFunctions as proc_f
import Source.CalibrationMeasurement as cal_c


# *** DATA PLOTTING ***
def pi_scale(min_val: Union[int, float], max_val: Union[int, float], ax, pi_minor_spacing: Optional[float] = None):
    """
    Change y-axis of plot to have labels as multiples of pi within the provided range.

    :param min_val: Minimum value of y-axis range.
    :param max_val: Maximum value of y-axis range.
    :param ax: Matplotlib axis object.
    :param pi_minor_spacing: Spacing between minor ticks. Provided value multiplied with pi.
        Default None, no minor ticks.

    :return: None.
    """
    maj_val = np.pi
    format_str = r"{:n}$\pi$"  # Formatting of major tick labels.
    min_val_root = np.ceil(min_val/maj_val)  # Minimum value of axis.
    max_val_root = np.floor(max_val/maj_val)  # Maximum value of axis.
    # Range of label values, as multiples of pi.
    label_val_arr = np.arange(min_val_root, max_val_root + 1, 1)
    arr = label_val_arr * np.pi  # Full values of major tick values.
    ax.set_ylim(min_val, max_val)  # Limit the axis.
    ax.set_yticks([])  # Remove old ticks.
    ax.set_yticks(arr)  # Use new values for ticks.
    if pi_minor_spacing is not None and type(pi_minor_spacing) in (int, float):  # Optional minor ticks.
        # For the minor ticks, use no labels; default NullFormatter.
        ax.yaxis.set_minor_locator(MultipleLocator(pi_minor_spacing*np.pi))
    label_arr = [format_str.format(elem) for elem in label_val_arr]  # Make major tick labels.
    # Fix special cases of tick labels.
    if 0 in label_val_arr:  # "0pi" -> "0".
        try:
            label_arr[label_arr.index(r"0$\pi$")] = "0"
        except ValueError:
            try:
                label_arr[label_arr.index(r"-0$\pi$")] = "0"
            except ValueError:
                print(label_arr)
    if -1 in label_val_arr:  # "-1pi" -> "-pi".
        try:
            label_arr[label_arr.index(r"-1$\pi$")] = r"-$\pi$"
        except ValueError:
            print(label_arr)
    if 1 in label_val_arr:  # "1pi" -> "pi".
        try:
            label_arr[label_arr.index(r"1$\pi$")] = r"$\pi$"
        except ValueError:
            print(label_arr)
    ax.set_yticklabels(label_arr)  # Set new major tick labels.


def freq_resp_style(ax: plt.Axes, db_bool: bool = False, phase_deg_bool: bool = False, phase_lim: bool = False,
                    pi_phase_sub: Optional[float] = None):
    """
    Sets a default styling for frequency response plots.

    :param ax: List of two matplotlib axis objects.
    :param db_bool: Amplitude in dB.
    :param phase_deg_bool: Phase in degrees.
    :param phase_lim: Tuple of lower and upper phase limit.
    :param pi_phase_sub: Subdivisions of pi used as minor axes.

    :return: Nothing. Applies style to ax objects.
    """
    ax[1].set_xlabel('f [Hz]')
    if db_bool:
        ax[0].set_ylabel('|TF| [dB]')
    else:
        ax[0].set_ylabel('|TF| [-]')
    if phase_deg_bool:
        ax[1].set_ylabel(r'$\angle$TF [deg]')
    else:
        ax[1].set_ylabel(r'$\angle$TF [rad]')
    for i in range(2):
        ax[i].grid(True, which='major')
        ax[i].grid(True, which='minor', linewidth=0.5)
        ax[i].set_xscale('log')

    if type(phase_lim) is bool:
        if phase_lim:  # Set -2 pi to 2 pi scale
            pi_scale(min_val=-2 * np.pi, max_val=2 * np.pi, ax=ax[1])
    elif type(phase_lim) is tuple or type(phase_lim) is list:
        pi_scale(min_val=phase_lim[0], max_val=phase_lim[1], ax=ax[1], pi_minor_spacing=pi_phase_sub)


def plot_single_df(df: DataFrame, ax: Optional[plt.Axes] = None, fig_dim: Optional[Sequence] = None,
                   x_channel: Optional[Tuple[str, str]] = None, x_channel_scale: float = 1.,
                   x_lim: Optional[Sequence] = None, y_lim: Optional[Sequence] = None,
                   legend_loc: Optional[Union[str, Tuple[float, float]]] = None, title: Optional[str] = None,
                   alpha: float = 1., color: Optional[Union[str, Sequence]] = None,
                   linestyle: Optional[Union[str, Sequence]] = '-', linewidth: float = 1.5, prefix: str = '',
                   label_format: str = '%(prefix)s%(channel_i)s', x_scale: str = 'linear', y_scale: str = 'linear',
                   x_str: Optional[str] = None, y_str: str = 'E, V') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot all columns of a Pandas DataFrame in a Matplotlib figure.

    :param df: DataFrame containing the data to be plotted. Index is used as x-axis.
    :param ax: Matplotlib axis used to plot. Default: None, create own Matplotlib figure with single axis.
    :param fig_dim: Dimensions in (width, height) in inches when ax is None and figure is created.
    :param x_channel: Name of DataFrame channel to use for x-axis of plot. Default: None; index of DataFrame is used.
    :param x_channel_scale: Scaling of the x-axis array of the plot. Default: 1.
    :param x_lim: List of x-axis minimum and maximum. Default: None.
    :param y_lim: List of y-axis minimum and maximum. Default: None.
    :param legend_loc: Location of legend for Matplotlib figure. Default: None, no legend.
    :param title: Title for figure. Default: ''.
    :param alpha: Opacity for data lines in plot.
    :param color: String or iterable of strings for color of plotting lines.
        Default: None, color is chosen by Matplotlib and iterated per channel.
    :param linestyle: String or iterable of strings for line-style of plotting lines.
        Default: '-'.
    :param linewidth: Width of plotting lines. Default: 1.5.
    :param prefix: String prefix for all labels in of legend that is pasted in front of column names of DataFrame df,
        following label_format.
    :param label_format: Format for legend strings, accepts variables: ('prefix', 'channel_i').
        Default: '%(prefix)s%(channel_i)s'.
    :param x_scale: String for scaling of x-axis, see matplotlib ax.set_xscale for options. Default: 'linear'.
    :param y_scale: String for scaling of y-axis, see matplotlib ax.set_yscale for options. Default: 'linear'.
    :param x_str: String label for x-axis.
        Default: None, either uses the name of the index column of the DataFrame, or if not available uses 'f, Hz'.
    :param y_str: String label for y-axis. Default: 'E, V'.

    :return: Matplotlib fig and ax object.
    """
    # Create new or use provided plotting objects.
    if ax is None:
        fig_t, ax_t = plt.subplots(1, 1, figsize=fig_dim)
    else:
        fig_t, ax_t = None, ax

    if title is not None:  # Set figure title.
        ax_t.set_title(title)

    if x_str is None:
        df_index_name = df.index.name
        if df_index_name is not None and type(df_index_name) is str:
            x_str = df_index_name
        else:
            x_str = 'f, Hz'  # The default.

    # Set the style of the figure.
    ax_t.set_xlabel(x_str)  # Axis labels.
    ax_t.set_ylabel(y_str)
    ax_t.grid(True, which='both')  # Grid, for both major and minor ticks.
    ax_t.set_xscale(x_scale)  # Axis scaling.
    ax_t.set_yscale(y_scale)
    if x_lim is not None:  # Axis limits.
        ax_t.set_xlim(*x_lim)
    if y_lim is not None:
        ax_t.set_ylim(*y_lim)

    if type(linestyle) == str:  # Linestyle. Set same style for all lines if only single string provided.
        linestyle = len(df.columns) * [linestyle]

    if type(color) == str or color is None:  # Color. Set same color for all lines if only single string provided.
        color = len(df.columns) * [color]

    if label_format is None:
        str_i = None

    if x_channel is None:
        x_arr = df.index  # x-array.
    else:
        x_arr = df.loc[:, x_channel]  # x-array.
    x_arr *= x_channel_scale  # Scale x-array.
    for i, channel_i in enumerate(df.columns):  # For each channel in DataFrame:
        str_i = label_format % {'prefix': prefix, 'channel_i': channel_i}  # Define label for figure legend.
        ax_t.plot(x_arr, df[channel_i], alpha=alpha, linewidth=linewidth, linestyle=linestyle[i], color=color[i],
                  label=str_i)  # Plot the data.

    if legend_loc is not None:  # If the legend location is None, no legend is plotted.
        ax_t.legend(loc=legend_loc)

    if ax is None:  # If axes are not user-defined, plot the figure.
        fig_t.show()
    return fig_t, ax_t  # Return both figure and axis used for plotting. If axis user-provided, figure is None.


def plot_transfer_function_df(df: DataFrame, ax: Optional[plt.Axes] = None, fig_dim: Optional[Sequence] = None,
                              x_lim: Optional[Sequence] = None, y_lim_amp: Optional[Sequence] = None,
                              y_lim_phase: Optional[Sequence] = None,
                              legend_loc: Optional[Union[str, Tuple[float, float]]] = None,
                              title: Optional[str] = None, alpha: float = 1.,
                              color: Optional[Union[str, Sequence]] = None,
                              linestyle: Optional[Union[str, Sequence]] = '-', linewidth: float = 1.5, prefix: str = '',
                              minor_phase: Optional[float] = None, label_format: str = '%(prefix)s%(channel_i)s',
                              y_scale: str = 'linear', x_str: str = 'f, Hz', amp_str: str = '|TF|, -',
                              phase_str: str = r'$\angle$TF, rad') \
        -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot all columns of a Pandas DataFrame containing complex-valued TFs, in a Matplotlib figure.

    :param df: DataFrame containing the complex-valued TFs to be plotted. Index is used as the frequency axis.
    :param ax: Matplotlib axis used to plot. Default: None, create own Matplotlib figure with single axis.
    :param fig_dim: Dimensions in (width, height) in inches when ax is None and figure is created.
    :param x_lim: List of x-axis minimum and maximum. Default: None.
    :param y_lim_amp: List of y-axis minimum and maximum for the amplitude of the TF. Default: None.
    :param y_lim_phase: List of y-axis minimum and maximum for the phase of the TF. Default: None.
    :param legend_loc: Location of legend for Matplotlib figure. Default: None, no legend.
    :param title: Title for figure. Default: ''.
    :param alpha: Opacity for data lines in plot.
    :param color: String or iterable of strings for color of plotting lines.
        Default: None, color is chosen by Matplotlib and iterated per channel.
    :param linestyle: String or iterable of strings for line-style of plotting lines.
        Default: '-'.
    :param linewidth: Width of plotting lines. Default: 1.5.
    :param prefix: String prefix for all labels in of legend that is pasted in front of column names of DataFrame df,
        following label_format.
    :param minor_phase: If y_lim_phase is provided, the major ticks are set at multiples of pi.
        This parameter sets the minor ticks at minor_phase multiples of pi,
        e.g., minor_phase=0.5 set them at pi/2 multiples. Default: None, no minor ticks.
    :param label_format: Format for legend strings, accepts variables: ('prefix', 'channel_i').
        Default: '%(prefix)s%(channel_i)s'.
    :param y_scale: String for scaling of y-axis of the TF amplitude, see matplotlib ax.set_yscale for options.
        Default: 'linear'.
    :param x_str: String label for x-axis. Default: 'f, Hz'.
    :param amp_str: String label for TF amplitude y-axis. Default: '|TF|, -'.
    :param phase_str: String label for TF phase y-axis. Default: r'$\angle$TF, rad'.

    :return: Matplotlib fig and ax object.
    """
    if ax is None:
        fig_t, ax_t = plt.subplots(2, 1, figsize=fig_dim, sharex='col')
    else:
        fig_t, ax_t = None, ax

    for i in range(2):
        ax_t[i].grid(True, which='major')
        ax_t[i].grid(True, which='minor', linewidth=0.5)
    if x_lim is not None:
        ax_t[1].set_xlim(*x_lim)
    if y_lim_amp is not None:
        ax_t[0].set_ylim(*y_lim_amp)
    if y_lim_phase is not None:
        pi_scale(min_val=min(y_lim_phase), max_val=max(y_lim_phase), ax=ax_t[1],
                 pi_minor_spacing=minor_phase)
    if title is not None:
        ax_t[0].set_title(title)
    ax_t[0].set_ylabel(amp_str)
    ax_t[1].set_ylabel(phase_str)
    ax_t[1].set_xlabel(x_str)
    ax_t[1].set_xscale('log')
    ax_t[0].set_yscale(y_scale)

    df_amp, df_phase = proc_f.frequency_response(complex_pressure_ratio=df, phase_deg_bool=False, axis=0,
                                                 unwrap_phase=True)
    f_arr = df.index

    if type(linestyle) == str:
        linestyle = len(df.columns) * [linestyle]

    if type(color) == str or color is None:
        color = len(df.columns) * [color]

    if label_format is None:
        str_i = None

    for i, channel_i in enumerate(df.columns):
        if label_format is not None:
            str_i = label_format % {'prefix': prefix, 'channel_i': channel_i}
        ax_t[0].plot(f_arr, df_amp[channel_i], alpha=alpha, color=color[i], linestyle=linestyle[i],
                     linewidth=linewidth, label=str_i)
        ax_t[1].plot(f_arr, df_phase[channel_i], alpha=alpha, color=color[i], linestyle=linestyle[i],
                     linewidth=linewidth, label=str_i)

    if legend_loc is None or legend_loc is True:
        ax_t[1].legend()
    elif legend_loc:  # If it is a string.
        ax_t[1].legend(loc=legend_loc)
    if ax is None:
        fig_t.show()
    return fig_t, ax_t


def plot_kde_df(df: DataFrame, kwargs_for_pair_grid: Optional[Dict[str, Any]] = None,
                kwargs_for_seaborn_theme: Optional[Dict[str, Any]] = None) -> PairGrid:
    """
    Plot the Kernel Density Estimate (KDE) of a Markov-chain Monte Carlo (McMC) sample DataFrame.

    :param df: DataFrame of McMC samples, where the indices are the samples and the columns the variables
    :param kwargs_for_pair_grid: Dictionary of arguments for the Seaborn PairGrid function. Default: diag_sharey=False.
    :param kwargs_for_seaborn_theme: Dictionary of arguments for the Seaborn set_theme function.

    :return: The Seaborn PairGrid object.
    """
    if kwargs_for_pair_grid is None:
        kwargs_for_pair_grid = {}
    kwargs_for_pair_grid = cal_c.var_kwargs(var_str='diag_sharey', default_val=False, kwargs=kwargs_for_pair_grid)
    if kwargs_for_seaborn_theme is None:
        kwargs_for_seaborn_theme = {}
    sns.set_theme(**kwargs_for_seaborn_theme)
    g = sns.PairGrid(df, **kwargs_for_pair_grid)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)

    '''
    f_kde = sns.PairGrid(df_alpha.loc[N_BURN_IN:, :])
    f_kde.map_upper(sns.scatterplot)
    f_kde.map_lower(sns.kdeplot)
    f_kde.map_diag(sns.kdeplot, lw=3, legend=False)
    '''
    '''
    f_kde2 = sns.PairGrid(df_alpha.loc[N_BURN_IN:, :], corner=True)  # , hue="species")
    f_kde2.map_lower(sns.kdeplot, hue=None, levels=5, color="r")
    f_kde2.map_lower(sns.scatterplot, marker="+", color='k')
    f_kde2.map_diag(sns.histplot, element="step", color='k', linewidth=0, kde=True)
    # f_kde2.add_legend(frameon=True)
    # f_kde2.legend.set_bbox_to_anchor((.61, .6))
    '''
    '''
    # Draw a combo histogram and scatterplot with density contours
    f3, ax3 = plt.subplots(figsize=(6, 6))
    x, y = df_alpha.loc[N_BURN_IN:, r'R $\nu^{-1/2}$, s$^{1/2}$'], df_alpha.loc[N_BURN_IN:, r'V$_v$/V$_t$, -']
    sns.scatterplot(x=x, y=y, s=5, color=".15")
    sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
    '''
    return g


def plot_chain_df(df: DataFrame, n_burn_in: int = 0,
                  log_mode: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the parameter value chain, per iteration, of a Markov-chain Monte Carlo (McMC) sample DataFrame.

    :param df: DataFrame of McMC samples, where the indices are the samples and the columns the variables
    :param n_burn_in: Number of burn-in samples of the chain. Default: 0.
    :param log_mode: Bool to have a logarithmic iteration-axis (for the complete chain, i.e. with the burn-in included).
        Default: False.

    :return: Matplotlib fig and ax object.
    """
    par_str = list(df.columns)

    n_samples, n_param = df.shape  # Number of samples in chain, number of parameters used in BI.

    if n_burn_in > 0:  # If burn-in samples considered, add second column of chain, only showing non-burn-in samples.
        fig_chain, ax_chain = plt.subplots(n_param, 2, sharex='col', figsize=(5.5, 2 * n_param))
    else:
        fig_chain, ax_chain = plt.subplots(n_param, 1, sharex='col', figsize=(3, 2 * n_param))
        # So that one can use column indices below for both cases.
        ax_chain = np.tile(ax_chain[:, np.newaxis], (1, 2))
    ax_chain = ax_chain.reshape((n_param, -1))  # Make 2D array, even if there is only one parameter.

    iter_arr = np.arange(1, n_samples+1)  # Array of sample iteration number.
    for i in range(n_param):  # Parameter per row of figure.
        y_label_str = par_str[i]  # Label string used for y-axis of plot.
        # Plot the chain.
        ax_chain[i, 0].plot(iter_arr, df.iloc[:, i], color='k', linewidth=0.8, label=y_label_str)
        ax_chain[i, 0].grid()  # Grid in figure.
        ax_chain[i, 0].set_ylabel(y_label_str)  # y-axis label.
    ax_chain[-1, 0].set_xlabel('Iteration, -')  # x-axis label.
    ax_chain[-1, 0].set_xlim(1, n_samples + 1)  # x-axis limits.
    if n_burn_in > 0:  # If burn-in samples considered, then plot second column of only non-burn-in samples.
        iter_arr_burn = np.arange(n_burn_in, n_samples+1)  # Shortened non-burn-in sample iterations.
        for i in range(n_param):
            y_label_str = par_str[i]  # Get appropriate plotting label.
            # Plot the chain, without burn-in samples.
            ax_chain[i, 1].plot(iter_arr_burn, df.iloc[n_burn_in - 1:, i], color='k', linewidth=0.8, label=y_label_str)
            ax_chain[i, 1].grid()
            ax_chain[i, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # Show which samples are considered burn-in in full chain column.
            ax_chain[i, 0].fill_betweenx(y=ax_chain[i, 0].get_ylim(), x1=2 * [1], x2=2 * [n_burn_in - 1], alpha=0.15,
                                         color='r', zorder=-3, edgecolor=None)
        ax_chain[-1, 1].set_xlabel('Iteration, -')
        ax_chain[-1, 1].set_xlim(n_burn_in, n_samples + 1)
    if log_mode:
        ax_chain[-1, 0].set_xscale('log')

    fig_chain.tight_layout(pad=0.1)  # Tries to fill white-space in figure. Can sometimes ruin the plot.
    fig_chain.subplots_adjust(wspace=0.236)  # , hspace=0.2)
    fig_chain.show()  # Show the figure.
    return fig_chain, ax_chain


def all_plotting_decorator(
        plot_func: Callable[[DataFrame, Optional[Any]], Tuple[plt.Figure, plt.Axes]] = plot_single_df):
    args_plotting_function_i = inspect.getfullargspec(plot_func).args

    def plotting_decorator(func):
        @functools.wraps(func)
        def inner1(*args, **kwargs):
            try:
                vis_bool = kwargs.pop('visualise')
            except KeyError:
                vis_bool = False
            kwargs_plotting = {key_i: kwargs[key_i] for key_i in kwargs if key_i in args_plotting_function_i}
            kwargs = {key_i: kwargs[key_i] for key_i in kwargs if key_i not in args_plotting_function_i}
            out_i = func(*args, **kwargs)

            if vis_bool:
                fig_i, ax_i = plot_func(out_i, **kwargs_plotting)
                return out_i, (fig_i, ax_i)
            else:
                return out_i

        return inner1
    return plotting_decorator

