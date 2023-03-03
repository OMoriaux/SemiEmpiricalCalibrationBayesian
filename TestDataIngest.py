"""
Example code of how the Source.CalibrationMeasurement.PressureAcquisition class can be used to estimate the transfer
function of the remote microphone probe using empirical calibration data, and how this class can be used to correct
unsteady pressure measurements with these transfer functions.
"""
import os
import numpy as np
import Source.CalibrationMeasurement as cal_c

# --- INPUT ---
FILE_FLUSH = os.path.join('.', 'TestData', 'BK_Pinhole', 'Flush_1.tdms')
FILE_MIC = os.path.join('.', 'TestData', 'BK_Pinhole', 'Pinhole_1.tdms')
KEY_FLUSH_IN_OUT = ([('Untitled', 'Channel 1')], [('Untitled', 'Channel 2')])
KEY_MIC_IN_OUT = ([('Untitled', 'Channel 2')], [('Untitled', 'Channel 1')])
DCT_TF2TF = {'Untitled_Channel 1>Untitled_Channel 2': 'Untitled_Channel 2>Untitled_Channel 1'}

CALIBRATE_MEASUREMENTS = False
FILE_MEAS = os.path.join('.', 'TestData', 'BK_Pinhole', 'Flush_1.tdms')
KEY_MEAS_PLOT = [('Untitled', 'Channel 3')]
DCT_M2TF = {('Untitled', 'Channel 3'): 'Untitled_Channel 1>Untitled_Channel 2'}

# *** end of input ***

# --- MAIN CODE ---
# Load 'flush' (flush-mounted reference microphone) calibration step.
obj_flush = cal_c.PressureAcquisition(file_path=FILE_FLUSH, safe_read=True, fs=51200, window_size=2**15)

# Visualise the data in several example ways:
# - The raw, unprocessed time-signals. If the microphone sensitivities are applied before this, then the plotted data
#    will no longer be voltage but pascale. Currently, it is up to the user to change the axis label accordingly,
#    using 'y_str=...'.
df_raw, (fig_raw, ax_raw) = obj_flush.raw(which='all', visualise=True, alpha=0.5, legend_loc='lower left',
                                          x_lim=(0, None))  # , y_lim=(-1.2, 1.2))
# - Power spectral density of the signals. Same as above, signal shown depends on whether sensitivities have been set.
df_psd, (fig_psd, ax_psd) = obj_flush.psd(which='all', visualise=True, alpha=0.5, legend_loc='lower left',
                                          y_str=r"$\Phi_{EE}$, V$^2$/Hz", x_scale='log', y_scale='log',
                                          x_lim=(1E-1, 3E4))
# - Normalised power spectral density (by default with p_ref=2E-5 Pa), dB/Hz.
#    Default p_ref doesn't really make sense without the microphone sensitivities applied.
#    This is just shown as an example.
df_psd_n, (fig_psd_n, ax_psd_n) = obj_flush.psd_norm(which='all', visualise=True, alpha=0.5, legend_loc='lower left',
                                                     y_str=r"$\Phi_{EE}$, dB/Hz", x_scale='log',
                                                     x_lim=(1E1, 13.5E3), y_lim=(-10, 60), title='Flush')

# - Cross-coherence between input and output signals.
df_coh, (fig_coh, ax_coh) = obj_flush.cross_coherence(in_channel=KEY_FLUSH_IN_OUT[0], out_channel=KEY_FLUSH_IN_OUT[1],
                                                      visualise=True, alpha=0.5, legend_loc='lower left',
                                                      y_str=r"$\gamma^2_{xy}$, -", x_scale='log',
                                                      x_lim=(1E1, 13.5E3), y_lim=(0.75, 1.01))

# - Transfer function between input and output signals.
df_tf, (fig_tf, ax_tf) = obj_flush.transfer_function(in_channel=KEY_FLUSH_IN_OUT[0], out_channel=KEY_FLUSH_IN_OUT[1],
                                                     visualise=True, alpha=0.5, legend_loc='lower left',
                                                     x_lim=(1E1, 13.5E3), minor_phase=0.25, prefix='Flush: ',
                                                     y_lim_amp=(0, 6), y_lim_phase=(-0.3, 1.1*np.pi+0.1),
                                                     linestyle='-.', color='b', fig_dim=(4, 5))

# Load 'mic' (remote microphone probe mounted to calibrator) calibration step.
obj_mic = cal_c.PressureAcquisition(file_path=FILE_MIC, safe_read=True, fs=51200, window_size=2**15)

# - 'Normalised' power spectral density.
df_psd_n_mic, (fig_psd_n_mic, ax_psd_n_mic) = \
    obj_mic.psd_norm(which='all', visualise=True, alpha=0.5, legend_loc='lower left', y_str=r"$\Phi_{EE}$, dB/Hz",
                     x_scale='log', x_lim=(1E1, 13.5E3), y_lim=(-15, 60), title='Mic')
# - Transfer function of the 'mic' calibration step.
df_tf_mic, _ = obj_mic.transfer_function(in_channel=KEY_MIC_IN_OUT[0], out_channel=KEY_MIC_IN_OUT[1], ax=ax_tf,
                                         visualise=True, alpha=0.5, legend_loc='lower left', prefix='Mic: ',
                                         linestyle='--', color='r', minor_phase=0.25, y_lim_amp=(1E-2, 3E1),
                                         y_lim_phase=(-2.1*np.pi-0.1, 1.1*np.pi+0.1))

# Combining the two (or more) calibration steps together into the total transfer function.
df_tf_full, _ = obj_flush.add_transfer_function_step(
    df_tf_new=df_tf_mic, axis=0, dct_old_tf_to_new_tf_channels=DCT_TF2TF, visualise=True, ax=ax_tf,
    legend_loc='lower left', prefix='Total: ', y_scale='log', linestyle='-', color='k')
ax_tf[1].legend(['Flush', 'Mic', 'Total'], loc='lower left')
fig_tf.tight_layout(pad=0.1)
fig_tf.subplots_adjust(left=0.16)
fig_tf.show()

if CALIBRATE_MEASUREMENTS:
    # Load unsteady pressure measurements.
    obj_meas = cal_c.PressureAcquisition(file_path=FILE_MEAS, safe_read=True, fs=51200, window_size=2**15)
    # Apply the previous transfer function to the measurement data.
    # ! One should also apply the sensitivities of the microphones.
    obj_meas.set_transfer_function(
        df_tf=1/df_tf_full, dct_channel_transfer_function=DCT_M2TF, axis=0)

    # Show the spectrum of the unsteady pressure measurements.
    df_psd_n_meas, (fig_psd_n_meas, ax_psd_n_meas) = \
        obj_meas.psd_norm(which=KEY_MEAS_PLOT, visualise=True, alpha=1., legend_loc='lower left',
                          y_str=r"$\Phi_{EE}$, dB/Hz", x_scale='log', x_lim=(1E1, 13.5E3), y_lim=(0, 50))
