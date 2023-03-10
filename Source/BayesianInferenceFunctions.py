"""
Helper functions for Bayesian inference.
"""
import numpy as np
import scipy.stats as stats
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
