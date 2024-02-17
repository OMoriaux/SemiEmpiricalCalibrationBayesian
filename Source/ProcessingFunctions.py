"""
Functions used for processing of microphone calibration and measurement time-series data DataFrames.
"""
import numpy as np
import scipy as sp
import pandas as pd
from scipy.signal.windows import hann
from pandas.core.frame import DataFrame
import numpy.typing as npt
from typing import Union, Iterable, Tuple

# Some typing.
type_arr_etc = Union[DataFrame, npt.ArrayLike, Iterable, int, float]
type_tuple_of_arrays = Tuple[Union[DataFrame, npt.NDArray[float]], Union[DataFrame, npt.NDArray[float]]]


# *** DATA PROCESSING ***
def frequency_response(complex_pressure_ratio: Union[DataFrame, npt.NDArray[complex], Iterable, int, float, complex],
                       phase_deg_bool: bool = False, unwrap_phase: bool = True, **kwargs) -> type_tuple_of_arrays:
    """
    Get amplitude and phase arrays from complex pressure ratio array.

    :param complex_pressure_ratio: Complex pressure ratio array.
    :param phase_deg_bool: Bool for if the phase is returned in degrees. Otherwise, phase given in radians.
    :param unwrap_phase: Unwrap the phase of the transfer function.

    :return: Amplitude array, phase array.
    """
    # If nan in ratio, messes up the unwrap function for the phase.
    complex_pressure_ratio[np.isnan(complex_pressure_ratio)] = 1 + 0j
    # Amplitude.
    pr_mag = np.abs(complex_pressure_ratio)
    # Phase.
    pr_phase = np.angle(complex_pressure_ratio, deg=phase_deg_bool)
    if unwrap_phase:
        pr_phase = np.unwrap(pr_phase, **kwargs)
    if type(complex_pressure_ratio) is DataFrame:
        pr_phase = pd.DataFrame(data=pr_phase, index=complex_pressure_ratio.index,
                                columns=complex_pressure_ratio.columns)
    return pr_mag, pr_phase


def tf_estimate(x: type_arr_etc, y: type_arr_etc, fs: float, **kwargs) \
        -> Tuple[Union[DataFrame, npt.NDArray[float]], Union[DataFrame, npt.NDArray[complex]]]:
    """
    'Transfer function estimate' function from MATLAB. Not all features are implemented.
    For documentation see: https://nl.mathworks.com/help/signal/ref/tfestimate.html
    Python code from: https://stackoverflow.com/questions/28462144/python-version-of-matlab-signal-toolboxs-tfestimate

    :param x: Input signal.
    :param y: Output signal.
    :param fs: Sample rate.
    :param kwargs: See the input parameters of scipy.signal.csd and scipy.signal.welch.
        Example extra input parameters: window=hann(self._w_size, sym=True), noverlap=None, nfft=2*window_size,
        return_onesided=True, axis=0. With window_size=2**15.

    :return: Transfer function estimate.
    """
    # Just let the scipy functions take care of the non-user-defined settings.
    freq, csd = sp.signal.csd(x=x, y=y, fs=fs, **kwargs)
    freq, psd = sp.signal.welch(x=x, fs=fs, **kwargs)
    tfe = csd / psd
    return freq, tfe


def f_psd(data: type_arr_etc, fs: float = 51200., **kwargs) -> type_tuple_of_arrays:
    """
    Computes power spectral density of signal.

    :param data: Array of data.
    :param fs: Sampling frequency [Hz]. Default: 51200.
    :param kwargs: See the input parameters of scipy.signal.welch.
        Example extra input parameters: window=hann(window_size, sym=True), noverlap=None, nfft=2*window_size,
        detrend=False, axis=0. With window_size=2**15.

    :return: (Frequency array, PSD array).
    """
    return sp.signal.welch(x=data, fs=fs, **kwargs)  # f, psd


def f_spectra(data_psd: type_arr_etc, p_ref: float = 2E-5) -> Union[DataFrame, npt.NDArray]:
    """
    Computes spectrum of data, based on PSD data.

    :param data_psd: Power spectral density array.
    :param p_ref: Reference pressure to compute. Default: 2E-5 Pa.

    :return: PSD array.
    """
    return 10 * np.log10(data_psd*p_ref**-2)  # [dB/Hz] or [dB/St] or whatever units of f_tf and fs.


def f_coherence(x: type_arr_etc, y: type_arr_etc, fs: float = 51200, **kwargs) -> type_tuple_of_arrays:
    """
    Computes cross-coherence between x - and y data arrays.

    :param x: Array of data x.
    :param y: Array of data y.
    :param fs: Sampling frequency [Hz].
    :param kwargs: See the input parameters of scipy.signal.welch.
        Example extra input parameters: window=hann(window_size, sym=True), noverlap=None, nfft=2*window_size,
        detrend=False, axis=0. With window_size=2**15.

    :return: (Frequency array, Cross-coherence array).
    """
    return sp.signal.coherence(x=x, y=y, fs=fs, **kwargs)  # f, coh
