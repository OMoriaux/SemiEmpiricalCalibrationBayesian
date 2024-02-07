"""
Example code for reading and plotting the FEM plane-wave tube data.
"""
import os
import numpy as np
import pandas as pd
import Source.PlottingFunctions as plot_f

# --- INPUT ---
# File of FEM simulation data.
FILE_NAME = os.path.join('.', 'TestData', 'FEM_PlaneWaveTube', 'p_3_sensors.csv')
# Names of pressure measurement points in simulation, in same order as FEM data columns.
MICS = ('reference', 'calibrator', 'junction')
# Indices for TF_{x->y}. Index of: (..., (x_mic, y_mic), ...). Uses order of elements in "MICS".
TF_IN_OUT_KEYS = ((2, 1), (0, 1))  # i.e.: (('junction'->'calibrator'), ('reference'->'calibrator')).
# --- END OF INPUT ---
# --------------------

# --- MAIN CODE ---
df_fem_raw_data = pd.read_csv(FILE_NAME, header=1, index_col=0)  # Load data file.
df_fem_raw_data.index.name = 'f, Hz'  # Set index name.

# Convert data to numpy array.
# Columns: [..., col_i-1 imag, col_i-1 real, col_i imag, col_i real, ...]; Rows: frequency array.
fem_raw_data_arr = df_fem_raw_data.to_numpy(float)
# Convert to complex array. Columns: [..., col_i-1 complex, col_i complex, ...]; Rows: frequency array.
fem_complex_data_arr = fem_raw_data_arr[:, ::2]*1j + fem_raw_data_arr[:, 1::2]

# Create pandas DataFrame with complex data.
df_fem_complex = pd.DataFrame(index=df_fem_raw_data.index, columns=MICS, data=fem_complex_data_arr, dtype=complex)

# Plot frequency data at measurement points.
fig_d, ax_d = plot_f.plot_transfer_function_df(
    df=df_fem_complex, fig_dim=(3, 4), x_lim=(1E2, 1.5E4), y_lim_amp=(1E-3, 1E1), y_lim_phase=(-10*np.pi, 0*np.pi),
    legend_loc='lower left', alpha=0.9, color=('k', 'r', 'b'), linestyle=('-', '--', ':'), linewidth=1.,
    minor_phase=None, y_scale='log')
y_ticks_lst = np.concatenate(([0], np.arange(-2, -12, -2)*np.pi))
ax_d[1].set_yticks(y_ticks_lst, ['0']+[rf'${elem/np.pi:.0f}\pi$' for elem in y_ticks_lst[1:]])
[ax_i.grid(False, which='both') for ax_i in ax_d]
fig_d.tight_layout(pad=0.1)
fig_d.show()

# Create transfer functions (TFs) between defined input and output indices. TF = p'_out / p'_in.
tf_keys_arr = np.array(TF_IN_OUT_KEYS)
tf_arr = fem_complex_data_arr[:, tf_keys_arr[:, 1]] / fem_complex_data_arr[:, tf_keys_arr[:, 0]]
# Header for DataFrame, used for plotting.
tf_header_lst = [rf'{MICS[key_in]}$\rightarrow${MICS[key_out]}' for key_in, key_out in TF_IN_OUT_KEYS]
df_fem_tfs = pd.DataFrame(index=df_fem_raw_data.index, columns=tf_header_lst, data=tf_arr, dtype=complex)

# This is used to shift the phase for the TF "reference -> calibrator" from -pi to +pi (+2pi shift in phase),
# to match the W model and how the data is presented in the paper.
n_tf = len(TF_IN_OUT_KEYS)  # Number of TFs
rmp_bool_case_lst = np.zeros(n_tf, dtype=bool)  # List of bools, where only True if matches "reference -> calibrator".
f_arr, amp_rmp, phase_rmp = 3*[None]  # Define variables.
if r'reference$\rightarrow$calibrator' in df_fem_tfs.columns:  # If that case (reference -> calibrator) is present:
    from Source.ProcessingFunctions import frequency_response
    f_arr = df_fem_tfs.index.to_numpy(float)  # Frequency array, for plotting this case separately later.
    # Compute amplitude and phase of TF_{reference -> calibrator}.
    amp_rmp, phase_rmp = frequency_response(df_fem_tfs.loc[:, [r'reference$\rightarrow$calibrator']], axis=0)
    phase_rmp[phase_rmp < -np.pi/2] += 2*np.pi  # Shift phase by +2pi or -2pi, as desired.
    # Set the specific item in 'rmp_bool_case_lst' to True for the specific case "reference -> calibrator".
    rmp_bool_case_lst[np.argwhere(df_fem_tfs.columns == r'reference$\rightarrow$calibrator')] = True

# Plot chosen TFs.
c_tf_arr = np.array(['k', 'r', 'b', 'g'])[:n_tf]  # List of line colours.
ls_tf_arr = np.array(['-', '--', ':', '-.'])[:n_tf]  # List of line styles.
fig_tf, ax_tf = plot_f.plot_transfer_function_df(
    df=df_fem_tfs.loc[:, ~rmp_bool_case_lst], fig_dim=(3, 5), legend_loc=False,
    x_lim=(1E2, 1.5E4), y_lim_amp=(1E-2, 1E2), y_lim_phase=(-1*np.pi-0.1, 1*np.pi+0.1),
    alpha=0.9, color=c_tf_arr[~rmp_bool_case_lst], linestyle=ls_tf_arr[~rmp_bool_case_lst], linewidth=1.,
    minor_phase=None, y_scale='log')  # Plot all cases, except "reference -> calibrator". '~' inverses the booleans.
if np.any(rmp_bool_case_lst):  # Plot the "reference -> calibrator" TF separately.
    kwargs_rmp = {'color': c_tf_arr[rmp_bool_case_lst][0], 'linestyle': ls_tf_arr[rmp_bool_case_lst][0], 'linewidth': 1,
                  'alpha': 0.9, 'label': r'reference$\rightarrow$calibrator'}  # Plotting style arguments.
    ax_tf[0].plot(f_arr, amp_rmp, **kwargs_rmp)
    ax_tf[1].plot(f_arr, phase_rmp, **kwargs_rmp)
[ax_i.grid(False, which='both') for ax_i in ax_tf]  # Remove grids from axes.
ax_tf[0].legend(loc='lower left', bbox_to_anchor=(0, 1))  # Place legend in figure.
# Set figure margins.
fig_tf.tight_layout(pad=0.1)
fig_tf.subplots_adjust(hspace=0.1)
fig_tf.show()
