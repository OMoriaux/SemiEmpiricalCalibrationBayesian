"""
Follows the method for computing the sensitivity of a microphone found in code from the TU Delft.

Could also estimate the sensitivity by computing the RMS of the pistonphone measurement, in voltage.
Then know what the RMP pressure is from the specification of the pistonphone.
From comparing both, know microphone sensitivity in mV/Pa. But this is a nice alternative with some filtering.

Original author of much better, more advanced MATLAB code from which I stole only a small fragment ;):
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MICROPHONE CALIBRATION CODE
%
% OWNER: Tercio
% DATE : 15/06/2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os
import numpy as np
import pandas as pd
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, find_peaks
import numpy.typing as npt
from typing import Tuple
# - Other code files (either own code, or from source listed in doc-string of function).
import Source.ProcessingFunctions as proc_f
import Source.PlottingFunctions as plot_f


# HELPER FUNCTIONS
def bandpass_signal(data_unfiltered: npt.NDArray[float], f_cal: float, delta_f: float = 50., fs: float = 51200.) -> \
        npt.NDArray[float]:
    """
    Band-pass filter a signal.

    :param data_unfiltered: Data time-array to be filtered.
    :param f_cal: Center of frequency range to band-pass.
    :param delta_f: Half-width of band-pass frequency range, [f_cal-delta_f, f_cal+delta_f].
    :param fs: Sampling frequency of signal to be filtered.

    :return: Filtered data time-array.
    """
    flt = butter(N=10, Wn=(f_cal - delta_f, f_cal + delta_f), fs=fs, btype='bandpass', output='sos')
    data_filt = np.copy(data_unfiltered)
    data_filt = sosfilt(flt, data_filt, axis=0)
    return data_filt


def find_peaks_outliers(data: npt.NDArray[float], cl: float = 1) -> Tuple[float, float]:
    """
    Find the maxima of the time-data of a sine-wave. Compute the mean value of the maxima, and their variance.
    Removes maxima if they fall outside of confidence level.

    :param data: Time data array of sine-wave.
    :param cl: Confidence level inside which the maxima must fall to be considered for the mean and variance.

    :return: Mean value, and variance of the maxima.
    """
    pks, _ = find_peaks(data)
    y_pks = data[pks]
    mu_pks0, sd_pks0 = np.mean(y_pks), np.std(y_pks)

    pks = pks[np.abs(y_pks - mu_pks0) <= cl*sd_pks0]
    y_pks = data[pks]
    mu_pks, var_pks = float(np.mean(y_pks)), float(np.var(y_pks))
    return mu_pks, var_pks


def calc_mean_pks_outlier(data: npt.NDArray[float], cl: float = 1, debug: bool = False) -> Tuple[float, float]:
    """
    Find the extrema (minima and maxima) of the time-data of a sine-wave.
    Compute the mean value of the extrema, and their variance.
    Removes extrema if they fall outside of confidence level.

    :param data: Time data array of sine-wave.
    :param cl: Confidence level inside which the extrema must fall to be considered for the mean and variance.
    :param debug: Print the mean and variance of both minima and maxima, as well as their combined mean and variance.

    :return: Mean value, and variance of the extrema.
    """
    # CALC +
    pks_m_p, var_m_p = find_peaks_outliers(data, cl=cl)
    if debug:
        print('+', pks_m_p, var_m_p)

    # CALC -
    pks_m_m, var_m_m = find_peaks_outliers(-data, cl=cl)
    if debug:
        print('-', pks_m_m, var_m_m)

    # END
    mean_pk = 0.5*(pks_m_p + pks_m_m)
    var_pk = var_m_p + var_m_m
    if debug:
        print('+/-', mean_pk, var_pk)
    return mean_pk, var_pk


def calc_mean_pks_default(data: npt.NDArray[float]) -> Tuple[float, float]:
    """
    Find the extrema (minima and maxima) of the time-data of a sine-wave.
    Compute the mean value of the extrema, and their variance.

    :param data: Time data array of sine-wave.

    :return: Mean value, and variance of the extrema.
    """
    # NUMBER OF EXCLUDED PEAKS
    # CALC +
    pks_p, _ = find_peaks(data)

    nex = np.floor(len(pks_p)/2)

    pks_m_p = np.mean(data[pks_p[nex:len(pks_p)]])
    var_m_p = np.var(data[pks_p[nex:len(pks_p)]])

    # CALC -
    pks_m, _ = find_peaks(-data)
    pks_m_m = np.mean(data[pks_m[nex:len(pks_m)]])
    var_m_m = np.var(data[pks_m[nex:len(pks_m)]])

    # END
    mean_pk = 0.5*(pks_m_p + pks_m_m)
    var_pk = var_m_p + var_m_m
    return mean_pk, var_pk


def calc_sensitivity(mean_pk: float, var_pk: float, cal_spl: float, p_ref: float = 2E-5, debug: bool = False) -> \
        Tuple[float, float]:
    """
    Compute the microphone sensitivity from the mean value of the time-signal extrema (Voltage)
    and the known pistonphone amplitude (dB).

    :param mean_pk: Mean of the acquired pistonphone sine time-signal in Volt.
    :param var_pk: Variance of the acquired pistonphone sine time-signal in Volt.
    :param cal_spl: Pistonphone amplitude in dB.
    :param p_ref: Reference pressure used to define the dB of cal_spl.
    :param debug: Print the mean and variance of both minima and maxima, as well as their combined mean and variance.

    :return: Microphone sensitivity, variance of the sensitivity estimation.
    """
    cal_pa = p_ref * 10**(cal_spl/20.)  # Convert from dB to Pa.

    sens = 2**-0.5 * mean_pk / cal_pa
    sens_var = 2**-0.5 * var_pk / cal_pa
    if debug:
        print(f'MIC Sensitivity estimated to: {sens} mV/Pa\nEstimation Variance: {sens_var}')
    return sens, sens_var


# MAIN FUNCTIONS TO USE
def main_mic_sensitivity(file_in: str, cal_spl: float = (94, 114)[0], f_cal: float = 1E3,
                         channel: Tuple[str, str] = ('Data', 'MIC'), fs: float = 51200, delta_f: float = 50,
                         debug: bool = False) -> Tuple[float, float]:
    """
    Compute microphone sensitivity from pistonphone calibration tdms file, using code of Tercio.

    :param file_in: TDMS file containing pistonphone calibration data.
    :param cal_spl: Pistonphone calibration amplitude, dB.
    :param f_cal: Pistonphone tone frequency, Hz.
    :param channel: Tuple containing the data 'group' and 'channel' in which the data is saved in the tdms file,
        e.g., tuple = (group, channel) = ('Data', 'MIC').
    :param fs: Sampling frequency of data, Hz.
    :param delta_f: Half-width of band-pass frequency range centered on pistonphone frequency,
        i.e., [f_cal-delta_f, f_cal+delta_f], Hz.
    :param debug: Print the mean and variance of both minima and maxima, as well as their combined mean and variance.

    :return: Microphone sensitivity, variance of the sensitivity estimation.
    """
    with TdmsFile.read(file_in) as f:
        _data_arr = f[channel[0]][channel[1]][:].to_numpy(float) * 1E3  # V -> mV
    # _df_data = cal_c.tdms_safe_read(file_in, key_lst=[channel])
    # _data_arr = _df_data[channel].to_numpy(float) * 1E3  # V -> mV
    _data_f = bandpass_signal(data_unfiltered=_data_arr, f_cal=f_cal, delta_f=delta_f, fs=fs)
    _mean_peaks, _var_peaks = calc_mean_pks_outlier(data=_data_f, debug=debug)
    _sens_val, _sens_var = calc_sensitivity(mean_pk=_mean_peaks, var_pk=_var_peaks, cal_spl=cal_spl, debug=debug)
    return _sens_val, _sens_var


def full_signal_sensitivity(file_in: str, cal_spl: float = (94, 114)[0], f_cal: float = 1E3,
                            channel: Tuple[str, str] = ('Data', 'MIC'), fs: float = 51200, delta_f: float = 50,
                            debug: bool = False, pre_amp: float = 1) -> Tuple[float, float]:
    """
    Compute microphone sensitivity from pistonphone calibration tdms file, using code of Tercio.
    Includes some visualization, otherwise similar to the main_mic_sensitivity function.

    :param file_in: TDMS file containing pistonphone calibration data.
    :param cal_spl: Pistonphone calibration amplitude, dB.
    :param f_cal: Pistonphone tone frequency, Hz.
    :param channel: Tuple containing the data 'group' and 'channel' in which the data is saved in the tdms file,
        e.g., tuple = (group, channel) = ('Data', 'MIC').
    :param fs: Sampling frequency of data, Hz.
    :param delta_f: Half-width of band-pass frequency range centered on pistonphone frequency,
        i.e., [f_cal-delta_f, f_cal+delta_f], Hz.
    :param debug: Print the mean and variance of both minima and maxima, as well as their combined mean and variance.
    :param pre_amp: Pre-amplification of the time-series data. Used for testing.

    :return: Microphone sensitivity, variance of the sensitivity estimation.
    """
    with TdmsFile.read(file_in) as f:
        _data_arr = f[channel[0]][channel[1]][:].to_numpy(float) * 1E3  # V -> mV
    # _df_data = p_hf.tdms_safer_reader(file_in, key_lst=[channel])
    # _data_arr = _df_data[channel].to_numpy(float) * 1E3  # V -> mV
    _data_arr *= pre_amp  # Pre-amplification for data.
    _data_f = bandpass_signal(data_unfiltered=_data_arr, f_cal=f_cal, delta_f=delta_f, fs=fs)
    _mean_peaks, _var_peaks = calc_mean_pks_outlier(data=_data_f, debug=debug)
    _sens_val, _sens_var = calc_sensitivity(mean_pk=_mean_peaks, var_pk=_var_peaks, cal_spl=cal_spl, debug=debug)

    f_raw, psd_raw = proc_f.f_psd(_data_arr, axis=0)
    f_flt, psd_flt = proc_f.f_psd(_data_f, axis=0)
    sp_raw = proc_f.f_spectra(psd_raw, p_ref=2E-5)
    sp_flt = proc_f.f_spectra(psd_flt, p_ref=2E-5)
    df_sp_raw = pd.DataFrame(data=sp_raw, index=f_raw, columns=['Unfiltered'])
    df_sp_flt = pd.DataFrame(data=sp_flt, index=f_flt, columns=['Filtered'])

    t_arr = np.arange(_data_arr.size) / fs
    fig_t, ax_t = plt.subplots(1, 2, figsize=(7, 3))
    c_u = ax_t[0].plot(t_arr, _data_arr, alpha=0.5, color='k', label='Unfiltered')
    c_f = ax_t[0].plot(t_arr, _data_f, alpha=0.5, color='b', label='Filtered')
    ax_t[0].set_xlim(0, t_arr[-1])
    ax_t[0].set_xlabel('t, s')
    ax_t[0].set_ylabel('E, V')
    ax_t[0].grid(True)

    plot_f.plot_single_df(df=df_sp_raw, ax=ax_t[1], alpha=0.7, color=c_u[-1].get_color(), linestyle='-', linewidth=1.5,
                          prefix='')
    plot_f.plot_single_df(df=df_sp_flt, ax=ax_t[1], x_lim=(1E0, 2E4), y_lim=(0, None), alpha=1,
                          color=c_f[-1].get_color(), linestyle='-', linewidth=1.5, prefix='', x_scale='log',
                          y_scale='linear', x_str='f, Hz', y_str=r"$\Phi_{p'p'}$, dB/Hz")
    ax_t[1].axvline(f_cal - delta_f, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax_t[1].axvline(f_cal + delta_f, color='k', alpha=0.5, linestyle='--', linewidth=1)
    ax_t[1].grid(True)

    fig_t.tight_layout(pad=0.3)
    fig_t.show()

    return _sens_val, _sens_var


def main():
    # Example of how the above functions are used.
    file = os.path.join('.', 'Calibration_Data', 'Sensitivity', 'Low_155868.tdms')
    group_channel_name_tuple = ('Data', 'MIC')  # Tuple of ('group name', 'channel name') in tdms file.

    # Read and process data.
    s_mean, s_var = full_signal_sensitivity(file_in=file, cal_spl=94, f_cal=1E3, fs=51200, delta_f=50,
                                            channel=group_channel_name_tuple, debug=True, pre_amp=1)


# If you run this code directly, then the example code (main) will execute.
if __name__ == '__main__':
    main()
