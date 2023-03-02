import os
import numpy as np
import Source.CalibrationMeasurement as cal_c

FILE_FLUSH = os.path.join('.', 'TestData', 'RMP_ExtraElement', 'FLUSH_02.tdms')
FILE_MIC = os.path.join('.', 'TestData', 'RMP_ExtraElement', 'RMP_E_02.tdms')
FILE_MEAS = os.path.join('.', 'TestData', 'RMP_ExtraElement', 'D00_V20_Set2.tdms')
KEY_FLUSH_IN_OUT = ([('Data', 'MIC'), ('Data', 'MIC')][:-1], [('Data', 'CAL_UP'), ('Data', 'SIGNAL')][:-1])
KEY_MIC_IN_OUT = ([('Data', 'CAL_UP'), ('Data', 'SIGNAL')][:-1], [('Data', 'MIC'), ('Data', 'MIC')][:-1])
KEY_MEAS_PLOT = [('Data', 'E')]
DCT_TF2TF = {'Data_MIC>Data_CAL_UP': 'Data_CAL_UP>Data_MIC'}
DCT_M2TF = {('Data', 'E'): 'Data_MIC>Data_CAL_UP'}

obj_flush = cal_c.PressureAcquisition(file_path=FILE_FLUSH, safe_read=True, fs=51200, window_size=2**15)


df_raw, (fig_raw, ax_raw) = obj_flush.raw(which='all', visualise=True, alpha=0.5, legend_loc='lower left',
                                          x_lim=(0, 60), y_lim=(-1.2, 1.2))
df_psd, (fig_psd, ax_psd) = obj_flush.psd(which='all', visualise=True, alpha=0.5, legend_loc='lower left',
                                          y_str=r"$\Phi_{EE}$, V$^2$/Hz", x_scale='log', y_scale='log',
                                          x_lim=(1E-1, 3E4))
df_psd_n, (fig_psd_n, ax_psd_n) = obj_flush.psd_norm(which='all', visualise=True, alpha=0.5, legend_loc='lower left',
                                                     y_str=r"$\Phi_{EE}$, dB/Hz", x_scale='log',
                                                     x_lim=(1E1, 13.5E3), y_lim=(-30, 50), title='Flush')
df_coh, (fig_coh, ax_coh) = obj_flush.cross_coherence(in_channel=KEY_FLUSH_IN_OUT[0], out_channel=KEY_FLUSH_IN_OUT[1],
                                                      visualise=True, alpha=0.5, legend_loc='lower left',
                                                      y_str=r"$\gamma^2_{xy}$, -", x_scale='log',
                                                      x_lim=(1E1, 13.5E3), y_lim=(0.95, 1.01))

df_tf, (fig_tf, ax_tf) = obj_flush.transfer_function(in_channel=KEY_FLUSH_IN_OUT[0], out_channel=KEY_FLUSH_IN_OUT[1],
                                                     visualise=True, alpha=0.5, legend_loc='lower left',
                                                     x_lim=(1E1, 13.5E3), minor_phase=0.25, prefix='Flush: ',
                                                     y_lim_amp=(0, 2), y_lim_phase=(-0.3, 1.1*np.pi+0.1),
                                                     linestyle='-.', color='b', fig_dim=(4, 5))

obj_mic = cal_c.PressureAcquisition(file_path=FILE_MIC, safe_read=True, fs=51200, window_size=2**15)

df_psd_n_mic, (fig_psd_n_mic, ax_psd_n_mic) = \
    obj_mic.psd_norm(which='all', visualise=True, alpha=0.5, legend_loc='lower left', y_str=r"$\Phi_{EE}$, dB/Hz",
                     x_scale='log', x_lim=(1E1, 13.5E3), y_lim=(-30, 50), title='Mic')
df_tf_mic, _ = obj_mic.transfer_function(in_channel=KEY_MIC_IN_OUT[0], out_channel=KEY_MIC_IN_OUT[1], ax=ax_tf,
                                         visualise=True, alpha=0.5, legend_loc='lower left', prefix='Mic: ',
                                         minor_phase=1, y_lim_amp=(6E-3, 6), y_lim_phase=(-8*np.pi, 1.1*np.pi+0.1),
                                         linestyle='--', color='r')

df_tf_full, _ = obj_flush.add_transfer_function_step(
    df_tf_new=df_tf_mic, axis=0, dct_old_tf_to_new_tf_channels=DCT_TF2TF, visualise=True, ax=ax_tf,
    legend_loc='lower left', prefix='Total: ', y_scale='log', linestyle='-', color='k')
fig_tf.tight_layout(pad=0.1)
fig_tf.subplots_adjust(left=0.16)
fig_tf.show()

obj_meas = cal_c.PressureAcquisition(file_path=FILE_MEAS, safe_read=True, fs=51200, window_size=2**15)
obj_meas.set_transfer_function(
    df_tf=1/df_tf_full, dct_channel_transfer_function=DCT_M2TF, axis=0)

df_psd_n_meas, (fig_psd_n_meas, ax_psd_n_meas) = \
    obj_meas.psd_norm(which=KEY_MEAS_PLOT, visualise=True, alpha=1., legend_loc='lower left',
                      y_str=r"$\Phi_{EE}$, dB/Hz", x_scale='log', x_lim=(1E1, 13.5E3), y_lim=(0, 50))
