"""
Helper functions for Bayesian inference.
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Callable, Union, Optional, Tuple, Dict, Sequence
import numpy.typing as npt


# --- HELPER FUNCTIONS ---
def mask_f_data(frequency: npt.NDArray[float], f_mask_arr: Sequence[Sequence[float]]) -> \
        Tuple[npt.NDArray[bool], npt.NDArray[float]]:
    """
    Mask frequency array and get boolean mask for given frequency band limits.

    :param frequency: Array of frequencies.
    :param f_mask_arr: Iterable of minimum {f_mask_arr[i][0]} and maximum {f_mask_arr[i][1]} frequencies of pass-bands.

    :return: Boolean mask array, Band-passed frequency array.
    """
    mask_arr = np.zeros(frequency.shape, dtype=bool)
    for lim in f_mask_arr:
        mask_arr[np.argwhere(frequency >= lim[0])] = True
        mask_arr[np.argwhere(frequency > lim[1])] = False
    return mask_arr, frequency[mask_arr]


def back_frequency_response(amplitude_arr: npt.NDArray[float], phase_arr: npt.NDArray[float]) -> npt.NDArray[complex]:
    """
    Convert amplitude and phase of TF back to complex-valued TF.
    ! Doesn't always give great results !

    :param amplitude_arr: Array of TF amplitude, -.
    :param phase_arr: Array of TF phase, rad.

    :return: Array of TF complex-valued, -.
    """
    phase_arr = np.arctan2(np.sin(phase_arr), np.cos(phase_arr))  # Tried to make this function more robust.
    re = amplitude_arr / (1 + np.tan(phase_arr)**2)**0.5
    im = re * np.tan(phase_arr)
    z = re + 1j * im
    return z


# --- BAYESIAN INFERENCE ---
# Class used for prior of Bayesian inversion. You could easily make something similar, but customized to your needs.
class PriorArray:
    def __init__(self, mean_array: npt.NDArray[Union[float, complex]], sd_array: npt.NDArray[Union[float, complex]]):
        """
        Prior for array of parameters. Uses multivariate Gaussian.

        :param mean_array: Array of parameter means.
        :param sd_array: Array of parameter standard deviations.
        """
        self.prior_arr = stats.norm(loc=mean_array, scale=sd_array)

    def log_pdf(self, array_i: npt.NDArray[Union[float, complex]]) -> float:
        """
        Compute logarithmic PDF of sample 'array_i' for prior.

        :param array_i: Proposal sample array.

        :return: Log PDF value for array_i.
        """
        return float(np.sum(self.prior_arr.logpdf(x=array_i)))


# Likelihood of Bayesian inversion.
def likelihood_log_pdf(u: npt.NDArray[Union[float, complex]], d: npt.NDArray[Union[float, complex]],
                       sigma_d: Union[float, npt.NDArray[Union[float, complex]]]) -> float:
    """
    Likelihood of model state with data, given measurement variance.

    :param u: Model state vector.
    :param d: Measurement vector.
    :param sigma_d: Measurement variance.

    :return: Value of log likelihood PDF.
    """
    d_hu = d - u
    log_rho = float(np.sum(stats.norm.logpdf(x=d_hu, scale=sigma_d)))
    return log_rho


# Markov chain Monte Carlo.
def mcmc_update(n_params: int, posterior_log_pdf: Callable[[npt.NDArray[Union[float, complex]]], float],
                alpha_0: npt.NDArray[Union[float, complex]], n_samples: int,
                sigma_p: Union[int, float, npt.NDArray[float]], n_updates: int, seed: Optional[int] = None) \
        -> Tuple[npt.NDArray[Union[float, complex]], npt.NDArray[float], Dict[str, int]]:
    """
    Metropolis-Hastings (Markov chain Monte Carlo, McMC).
    Adapted from: https://aerodynamics.lr.tudelft.nl/~rdwight/cfdiv/index.html (Accessed 12/07/2021).

    Try to keep the acceptance ratio to ca. 30%.
    (rule-of-thumb for balance between accepting many samples and keeping the correlation between samples low)

    Low acceptance means a lot of wasted samples, the chain will not evolve in any direction.
    High acceptance means that the samples have a high correlation between them. So using these samples for the KDE
    will result in a PDF estimate that doesn't represent the pure posterior from which it samples, but also the sampling
    Gaussian statistics will come through.
    Can also later re-sample samples to lower correlation.
    Tune this parameter with the variance of the sampling Gaussian, sigma_p.

    More parameters mean that the acceptance rate could dip too, requiring smaller sigma_p
    to get the same acceptance ratio.

    Remove the burn-in samples (samples needed to reach region of high-probability,
    where the chain will seem to converge to) before using this to estimate the posterior PDF with.

    :param n_params: Number of parameters to fit.
    :param posterior_log_pdf: Callable that returns the posterior probability density for input array of parameters.
        Parameter input array contains parameters to fit, must have length of n_params.
    :param alpha_0: Initial guess of parameters to fit, must have length of n_params.
    :param n_samples: Number of samples to be taken from posterior using McMC method.
    :param sigma_p: Variance of parameter sampling multivariate Gaussian.
        If float provided then value used for all parameters.
        If array not of length n_params, then average used as float input.
        If array of length n_params,
        the values are used in the main diagonal of the covariance array of the multivariate Gaussian.
    :param n_updates: Amount of iterations for print updates.
    :param seed: Seed number of sampler. Optional: Default None, seed not set.

    :return: Tuple containing:
        - Chain: Array of parameter samples (n_samples, n_params).
        - PDF: Posterior probability for the chain of samples (n_samples).
        - SEED: Dictionary of seed for both the multi-variate normal G and uniform Y.
    """
    alpha_chain = np.zeros((n_samples, n_params))
    log_pdf_chain = np.zeros(n_samples)
    alpha_chain[0] = alpha_0
    log_pdf_chain[0] = posterior_log_pdf(alpha_0)
    acceptance = 0
    local_acceptance = 0

    # Covariance matrix of multivariate Gaussian used for parameter samples.
    if type(sigma_p) != int and type(sigma_p) != float:
        if len(sigma_p) == n_params:
            p = np.diag(sigma_p ** 2)
        else:
            p = np.average(sigma_p) ** 2 * np.eye(n_params)
    else:
        p = sigma_p ** 2 * np.eye(n_params)

    g = stats.multivariate_normal(mean=np.zeros(n_params), cov=p)  # Sampling Gaussian.
    y = stats.uniform(0, 1)  # Sieve for acceptance/rejection of samples.

    if seed is not None:  # Optional set seed.
        g_seed, y_seed = seed, seed
        g.random_state = np.random.Generator(np.random.PCG64(seed))
        y.random_state = np.random.Generator(np.random.PCG64(seed))
    else:
        g_seed, y_seed = g.random_state, y.random_state

    for i in range(1, n_samples):
        alpha_p = np.abs(g.rvs() + alpha_chain[i-1])  # Take sample from G, sampled around chain[i-1].
        log_rho_p = posterior_log_pdf(alpha_p)  # Run model to compute posterior probability.
        # Pi parameter used for acceptance/rejection of sample. Ratio of posterior values.
        post_frac = log_rho_p - log_pdf_chain[i-1]

        y_i = np.log(y.rvs())  # Sample of sieve.

        if y_i < post_frac:     # Accept sample.
            acceptance += 1
            local_acceptance += 1
            alpha_chain[i] = alpha_p
            log_pdf_chain[i] = log_rho_p
        else:                   # Reject sample. (Just repeat last sample as new one).
            alpha_chain[i] = alpha_chain[i-1]
            log_pdf_chain[i] = log_pdf_chain[i-1]
        if i % n_updates == 0:  # Visual update on acceptance ratio. Can be removed in code for improved performance.
            print(f'({i//n_updates}/{n_samples//n_updates}) Acceptance Ratio: '
                  f'{100. * local_acceptance / (n_updates - 1):.2f}%')
            local_acceptance = 0

    print(f'Acceptance Ratio: {100. * acceptance / (n_samples - 1):.2f}%')
    return alpha_chain, log_pdf_chain, {'g': g_seed, 'y': y_seed}


# Visualize McMC posterior PDF using KDE (contours).
def kde_2d(sample_arr: npt.NDArray[float], n: int = 100,
           limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) \
        -> Tuple[npt.NDArray[float], Tuple[npt.NDArray[float], npt.NDArray[float]],
        Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Kernel Density Estimate.
    ! THERE EXIST PYTHON PACKAGES THAT CAN DO THIS (FASTER), e.g., Arviz.
    Source: https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python

    :param sample_arr: Array of parameter samples.
        Shape: (n_samples, n_parameters=2!).
    :param n: Number of points in both x and y to compute the KDE on, n x n grid.
        Default: 100.
    :param limits: List for each parameter extrema, in order of sample_arr.
        e.g., [(x_min, x_max), (y_min, y_max)],
        where x_arr is sample_arr[:, 0] and y_arr is sample_arr[:, 1].
        Default: None. Takes extrema from sample_arr.

    :return: KDE array (n, n), [x_meshgrid (n, n), y_meshgrid (n, n)], limits (2, 2).
    """
    # Get arrays for each separate parameter.
    x_arr, y_arr = sample_arr[:, 0], sample_arr[:, 1]
    if limits is None:  # If no limits is provided.
        x_min, x_max, y_min, y_max = np.min(x_arr), np.max(x_arr), np.min(y_arr), np.max(y_arr)
        limits = ((x_min, x_max), (y_min, y_max))
    else:  # Limits are provided.
        x_min, x_max = limits[0]
        y_min, y_max = limits[1]
    # Create mesh-grids on which KDE is computed.
    xx, yy = np.mgrid[x_min:x_max:n*1j, y_min:y_max:n*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_arr, y_arr])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return f, (xx, yy), limits


def find_hdi_contours(density: npt.NDArray[float], hdi_probs: Sequence[float]) -> npt.NDArray[float]:
    """
    ! CODE FROM ARVIZ PACKAGE !
    Find contours enclosing regions of the highest posterior density.

    Parameters
    ----------
    density : array-like
        A 2D KDE on a grid with cells of equal area.
    hdi_probs : array-like
        An array of the highest density interval confidence probabilities.

    Returns
    -------
    contour_levels : array
        The contour levels corresponding to the given HDI probabilities.
    """
    # Using the algorithm from corner.py
    sorted_density = np.sort(density, axis=None)[::-1]  # Sort the PDF values from high to low.
    sm = sorted_density.cumsum()
    sm /= sm[-1]  # Cumulative probability function, starting from max PDF to min PDF.
    # Iterate though desired hdi_probs, find region of inverse CDF sm that contains hdi probability.
    contours = np.empty_like(hdi_probs)
    for idx, hdi_prob in enumerate(hdi_probs):
        try:
            contours[idx] = sorted_density[sm <= hdi_prob][-1]
        except IndexError:
            contours[idx] = sorted_density[0]

    return contours


def kde_hdi_plots(chain: npt.NDArray[float], var_lst: Sequence[str], n_grid: int = 128,
                  limits_tuple_list: Optional[Sequence[Tuple[float, float]]] = None,
                  hdi_probs: Sequence[float] = (0.68, 0.95, 0.99), ax: Optional[plt.Axes] = None,
                  sub_plot: bool = False, sub_plot_center: Optional[Sequence[float]] = None,
                  sub_plot_wh_list: Optional[Sequence[float]] = None, remove_upper: bool = False,
                  dct_var_label: Optional[Dict[str, str]] = None, chain_plt: bool = False, x_label_loc: str = 'center',
                  y_label_loc: str = 'center', sci_ticks: bool = True):
    """
    Plot KDE highest density region contour for n_parameters > 2.

    :param chain: Parameter chain array, shape: (n_samples, n_parameters).
    :param var_lst: List of variable name strings, shape: n_parameters.
    :param n_grid: Number of points in both x and y to compute the KDE on, n x n grid.
        Default: 2*128.
    :param limits_tuple_list: limits: List for each parameter extrema, in order of sample_arr.
        e.g., [(par_0_min, par_0_max), ..., (par_i_min, par_i_max)..., (par_n_min, par_n_max)].
        Default: None. Takes extrema from sample_arr.
    :param hdi_probs: List of HDI probabilities. Should be increasing confidence level, e.g.: (0.68, 0.95).
        Default: (0.68, 0.95, 0.99).
    :param ax: Matplotlib axis array used to plot. Should be of shape (n_parameters-1, n_parameters-1).
        Default: None, own axes created but not returned.
    :param sub_plot: Make small zoomed in subplot in contour plots. Default: False.
    :param sub_plot_center: List of parameter values on which the sub_plots are centered.
        Default: None, the mean for each parameter is used as the center.
    :param sub_plot_wh_list: List of all parameters, containing list of width and height for each sub_plot.
        Default: None, use 0.5 in both width and height for all the parameters.
    :param remove_upper: Remove upper triangle of axes of figure. Default: False.
    :param dct_var_label: Dictionary with var_lst as keys, and the elements are the strings used
        as plotting axis labels. Default: None, use the var_lst as plotting labels.
    :param chain_plt: Plot the actual chain samples in the contours. Default: False.
    :param x_label_loc: Where to align the x-axis labels. Default: 'center'.
    :param y_label_loc: Where to align the y-axis labels. Default: 'center'.
    :param sci_ticks: Plot the axes using matplotlib scientific notation for the numbers.
        Default: True.

    :return: Nothing.
    """
    # Take care of default input values.
    n_var = len(var_lst)
    if limits_tuple_list is None:
        limits_tuple_list = n_var * [None]

    if sub_plot_center is None:
        sub_plot_center = np.mean(chain, axis=0)

    if sub_plot_wh_list is None:
        sub_plot_wh_list = (n_var * [0.5], n_var * [0.5])

    if dct_var_label is None:
        dct_var_label = dict(zip(var_lst, var_lst))

    initial_arr = chain[0, :]  # Initial parameter guess.

    if ax is None:  # No plotting axes given.
        fig_plt, ax_plt = plt.subplots(n_var - 1, n_var - 1, sharex='col', sharey='row')
    else:  # Use provided plotting axes.
        fig_plt, ax_plt = None, ax

    for i, var_x in enumerate(var_lst[:-1]):
        ax_plt[-1, i].set_xlabel(dct_var_label[var_x], loc=x_label_loc)
        for j, var_y in enumerate(var_lst[1:]):
            if i <= j:  # Lower triangle of plotting axes array.
                j1 = j + 1
                xy_lim = (limits_tuple_list[i], limits_tuple_list[j1])
                if None in xy_lim:
                    xy_lim = None
                x_arr = chain[:, i]
                y_arr = chain[:, j1]
                density_2d, kde_grid, limits = kde_2d(sample_arr=chain[:, [i, j1]], n=n_grid, limits=xy_lim)
                # density_2d, kde_grid, bandwidth_kde = kde2d(x=x_arr, y=y_arr, n=n_grid, limits=xy_lim)
                contour_levels_2d = find_hdi_contours(density=density_2d, hdi_probs=hdi_probs)

                x_centre, y_centre = sub_plot_center[i], sub_plot_center[j1]

                if sci_ticks:
                    ax_plt[j, i].ticklabel_format(axis=('y', 'both')[1], style='sci', scilimits=(0, 0))
                ax_plt[j, i].scatter(initial_arr[i], initial_arr[j1], color='b', marker='.', label='Initial guess')
                ax_plt[j, i].scatter(x_centre, y_centre, color='r', marker='+', label=r'Best guess')

                if chain_plt:
                    ax_plt[j, i].plot(x_arr, y_arr, color='k', marker='.', linewidth=0.3, markersize=1, alpha=0.1)

                if sub_plot:
                    w_ij, h_ij = sub_plot_wh_list[0][i], sub_plot_wh_list[1][j1]
                    # 0.03 lower x0 when no tick labels.
                    ax_ins = ax_plt[j, i].inset_axes(bounds=[0.2, 0.4, 0.47, 0.47])
                    cs = ax_ins.contour(kde_grid[0], kde_grid[1], density_2d,
                                        levels=contour_levels_2d[::-1], linewidths=0.5, colors='k')
                    ax_ins.scatter(x_centre, y_centre, color='r', marker='+')  # , label=r'Best guess')
                    ax_ins.set_xlim(x_centre - 0.5 * w_ij, x_centre + 0.5 * w_ij)
                    ax_ins.set_ylim(y_centre - 0.5 * h_ij, y_centre + 0.5 * h_ij)
                    # ax_ins_kde.set_xticklabels([])
                    # ax_ins_kde.set_yticklabels([])
                    ax_plt[j, i].indicate_inset_zoom(ax_ins, edgecolor="black")
                else:
                    cs = ax_plt[j, i].contour(kde_grid[0], kde_grid[1], density_2d, levels=contour_levels_2d[::-1],
                                              linewidths=0.5, colors='k')
                    ax_ins = None

                fmt_p = {}
                for k, s in zip(cs.levels, ['99%', '95%', '68%']):
                    fmt_p[k] = s
                if sub_plot:
                    ax_ins.clabel(cs, cs.levels, inline=True, fmt=fmt_p, fontsize=9)
                else:
                    ax_plt[j, i].clabel(cs, cs.levels, inline=True, fmt=fmt_p, fontsize=9)

                if i == 0:
                    ax_plt[j, 0].set_ylabel(dct_var_label[var_y], loc=y_label_loc)
                ax_plt[j, i].grid(True)

    if remove_upper and ax is None:
        remove_upper_triangle(fig=fig_plt, ax=ax_plt)

    if ax is None:
        if remove_upper:
            fig_plt.tight_layout(pad=0.1)
            ax_plt[0, 0].legend(loc='lower left', bbox_to_anchor=[1, 0])
        else:
            ax_plt[-1, 0].legend()
            fig_plt.tight_layout(pad=0.1)
        fig_plt.show()


def remove_upper_triangle(fig: plt.Figure, ax: plt.Axes):
    """
    Remove the upper the matplotlib axes in the upper right triangle of the figure.

    :param fig: Matplotlib figure.
    :param ax: 2D-array of Matplotlib axes.

    :return: Nothing.
    """
    n_dim = ax.shape[0]
    for i in range(n_dim):
        for j in range(n_dim):
            if i > j:
                fig.delaxes(ax[j, i])  # Remove axis from figure.
