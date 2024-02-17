"""
Visualisation for MCMC output data, as exported by MCMC_BK_Pinhole.py.

Load Markov chain, shows:
 - chain for each parameter.
 - Kernel Density Estimate using samples of chain (considering number of burn-in samples).
 - TF (data, model initial guess, model best guess).

Kernel Density Estimate (KDE) estimates probability density function (PDF) that was used to get parameter samples of
chain. Uses sample density over parameter space. KDE code not mine, see source in code.
"""
# --- IMPORT PACKAGES ---
# - Default Python.
import os  # Used to define file paths. Uses correct separators for whatever OS it is run on.
import pickle  # MCMC output data exported as pickle file, to be loaded by processing code.
# - Might require installation from web.
import matplotlib
import numpy as np
import pandas as pd
# - Other code files (either own code, or from source listed in doc-string of function).
import Source.PlottingFunctions as PlotF
import Source.CalibrationMeasurement as CalM
# import Processing_Helper_Functions as p_hf
# --- CASE TO PROCESS ---
from BK_Pinhole_MCMC import model_bt, model_w

# --- INPUT ---
F_IN_CASE = 'BK_Pinhole'
F_MCMC = os.path.join('.', 'MCMC_Output', f'{F_IN_CASE}_01.pickle')  # File containing MCMC samples.
SAVE_FIG = True  # Save output figures to PNG.
LOG_MODE = True  # Set x-axis to logarithmic for the chain evolution figure.

# It is recommended to start without the KDE, figure out the N_BURN_IN parameter from the chain figure.
# After that, with the appropriate burn-in, the KDE can be run, which can take quite some time.
PLOT_CHAIN = True  # Plot the parameter chain.
PLOT_KDE = True  # Compute and plot the Kernel Density Estimate (KDE).
PLOT_TF = True  # Estimate and plot the Transfer Function (TF).

N_BURN_IN = 2230  # Number of samples at the start thrown away for KDE. To consider all samples, set to 0.
# Credible region (Highest Density Interval, HDI) probability value list. From largest to smallest.
HDI_LST = [0.99, 0.95, 0.68]
N_GRID = 128  # Number of points (in both x and y) used to compute KDE on.

# Path and name of PNG-files to save. Will add, e.g., '_TF.png' to the end of 'NAME_OUT'.
NAME_OUT = os.path.join('.', 'MCMC_Figures', F_IN_CASE, F_MCMC.split(os.sep)[-1][:-7])

# For plotting. Converts parameter labels strings defined in MCMC sample file into strings used for figures.
dct_par_labels = {'L_c': 'L/c, s', 'R_nu': r'R $\nu^{-1/2}$, s$^{1/2}$', 'Vv_Vt': r'V$_v$/V$_t$, -',
                  'square_amp': r'c$_{2, |TF|}$, s$^2$', 'slope_amp': r'c$_{1, |TF|}$, s',
                  'intercept_amp': r'c$_{0, |TF|}$, -', 'slope_phase': r'c$_{1, \angle TF}$, rad s',
                  'intercept_phase': r'c$_{0, \angle TF}$, rad',
                  'Lc_rmp_up': '(L/c)$_{upper}$, s', 'Rnu_rmp_up': r'(R $\nu^{-1/2}$)$_{upper}$, s$^{1/2}$',
                  'VtVv_rmp_up': r'$\left(V_v/V_t\right)_{upper}$, -', 'Lc_rmp_s': '(L/c)$_{side}$, s',
                  'Rnu_rmp_s': r'(R $\nu^{-1/2}$)$_{side}$, s$^{1/2}$',
                  'VtVv_rmp_s': r'$\left(V_v/V_t\right)_{side}$, -', 'Lc_leak': '(L/c)$_{leak}$, s',
                  'Rnu_leak': r'(R $\nu^{-1/2}$)$_{leak}$, s$^{1/2}$', 'VtVv_leak': r'$\left(V_v/V_t\right)_{leak}$, -',
                  'Vt_leak__Vv_rmp_s': r'$V_{v, leak}/V_{t, side}$, -',
                  'Vt_rmp_s__Vv_rmp_up': r'$V_{v, side}/V_{t, upper}$, -',
                  'Vt_rmp_do__Vv_rmp_up': r'$V_{v, lower}/V_{t, upper}$, -'}

# --- END OF INPUT ---
# --------------------

# --- LOAD MCMC SAMPLES AND INFORMATION ---
with open(F_MCMC, 'rb') as handle:  # Open pickle file.
    dct_mcmc = pickle.load(handle)
theta_arr = dct_mcmc.pop('theta_chain')  # Parameter chain.
p_arr = dct_mcmc.pop('p_chain')  # Posterior PDF at chain parameter value sets.
idx_select = dct_mcmc['MCMC_SETTINGS']['PAR_SELECT']  # Indices of parameters used for BI.
par_str = dct_mcmc['MCMC_SETTINGS']['PAR_STR'][idx_select]  # Parameter label strings.
bt_bool = dct_mcmc['MCMC_SETTINGS']['BT_MODE']  # Which model is used: Whitmore (False) or Bergh & Tijdeman (True).

# Convert parameter chain values to an easily plottable DataFrame.
df_chain = pd.DataFrame(theta_arr, columns=par_str)
df_chain.columns = [dct_par_labels[key_i] for key_i in df_chain.columns]

n_samples, n_param = theta_arr.shape  # Number of samples in chain, number of parameters used in BI.
theta_0 = theta_arr[0, :]  # Initial guess parameter values.
theta_map = theta_arr[np.argmax(p_arr), :]  # Optimal parameter values (MAP: Maximum A Posteriori).
print(f'The chain has {n_param} parameters, and {n_samples} samples.')

# Define font-size and font for plotting.
matplotlib.rcParams.update({'font.size': 12, "font.family": 'sans-serif', "font.sans-serif": "Arial"})

# --- CHAIN ---
# Parameter value chain (in the y-axis), x-axis is the iteration number.
if PLOT_CHAIN:
    fig_chain, ax_chain = PlotF.plot_chain_df(df=df_chain, n_burn_in=N_BURN_IN, log_mode=LOG_MODE)
    if SAVE_FIG:
        fig_chain.savefig(NAME_OUT + '_Chain.png', dpi=600, transparent=False)

# --- KDE ---
# Kernel Density Estimate. Estimates the posterior PDF from the MCMC samples (of the posterior).
if PLOT_KDE:
    g_kde = PlotF.plot_kde_df(
        df=df_chain.iloc[N_BURN_IN:, :],
        kwargs_for_pair_grid=None,
        kwargs_for_seaborn_theme={'style': 'whitegrid'})

    g_kde.tight_layout()
    if SAVE_FIG:
        g_kde.fig.savefig(NAME_OUT + '_KDE.png', dpi=600, transparent=False)
    g_kde.fig.show()

# --- TF ---
# Transfer function. The main objective of the method, so it's good to visualise.
if PLOT_TF:
    # Calibration data.
    dct_data = dct_mcmc['DATA']
    cal_flush = CalM.PressureAcquisition(dct_data['F_FLUSH'], safe_read=False)
    cal_mic = CalM.PressureAcquisition(dct_data['F_MIC'], safe_read=False)
    cal_mic.set_sensitivities(dct_channel_sensitivity=dct_data['S_DCT'], strict_mode=True, dct_prop_str_new_val=None)
    df_tf_flush = cal_flush.transfer_function(
        in_channel=dct_data['KEY_FLUSH_IN'], out_channel=dct_data['KEY_FLUSH_OUT'], set_property=True)
    df_tf_mic = cal_mic.transfer_function(
        in_channel=dct_data['KEY_MIC_IN'], out_channel=dct_data['KEY_MIC_OUT'], set_property=True)
    df_tf_full = cal_flush.add_transfer_function_step(
        df_tf_new=df_tf_mic, dct_old_tf_to_new_tf_channels=dict(zip(df_tf_flush.columns, df_tf_mic.columns)))
    df_tf_full.columns = ['Data']
    f_arr = df_tf_full.index.to_numpy(float)
    w_arr = 2 * np.pi * f_arr

    # Band-passing of data.
    # Used to transform band-passed frequencies of calibration data to band-removed frequencies of data.
    f_min_range, f_max_range = 0., np.max(f_arr)
    f_msk = dct_mcmc['MCMC_SETTINGS']['F_MASK_LST']
    f_mask = [[f_msk[i][-1], f_msk[i + 1][0]] for i in range(len(f_msk) - 1) if f_msk[i + 1][0] >= f_msk[i][-1]]
    if len(f_mask) == 0:
        f_mask = [[f_min_range, f_msk[0][0]], [f_msk[-1][-1], f_max_range]]
    else:
        f_mask = [[f_min_range, f_msk[0][0]]] + f_mask + [[f_msk[-1][-1], f_max_range]]

    # Model TF.
    if bt_bool:
        model = model_bt
    else:
        model = model_w

    theta_full = dct_mcmc['MCMC_SETTINGS']['THETA_FULL']
    amp_m_0, phase_m_0 = model(theta_i=theta_0, w_arr=w_arr, theta_long=theta_full, par_idx=idx_select)
    amp_m_map, phase_m_map = model(theta_i=theta_map, w_arr=w_arr, theta_long=theta_full, par_idx=idx_select)

    # Plotting.
    fig_tf, ax_tf = PlotF.plot_transfer_function_df(
        df=df_tf_full, fig_dim=(4, 5), color='k', linestyle='--', alpha=0.8, minor_phase=0.25,
        x_lim=(1E2, 13.5E3), y_lim_amp=(0, 4), y_lim_phase=(-1*np.pi-0.1, 0.1))
    # Initial guess.
    ax_tf[0].plot(f_arr, amp_m_0, color='b', linestyle=':', label=r'$\theta_0$')
    ax_tf[1].plot(f_arr, phase_m_0, color='b', linestyle=':', label=r'$\theta_0$')
    # MAP: Best guess.
    ax_tf[0].plot(f_arr, amp_m_map, color='r', linestyle='-', label=r'$\theta_{MAP}$')
    ax_tf[1].plot(f_arr, phase_m_map, color='r', linestyle='-', label=r'$\theta_{MAP}$')

    # Band-removed data.
    for ax_i in ax_tf.flatten():
        y_lim_i = ax_i.get_ylim()
        for (x_lower_limit_i, x_upper_limit_i) in f_mask:
            ax_i.fill_betweenx(y=y_lim_i, x1=2*[x_lower_limit_i], x2=2*[x_upper_limit_i],
                               color='r', alpha=0.2)

    ax_tf[1].legend(loc='lower left')
    fig_tf.tight_layout(pad=0.1)
    fig_tf.show()
    if SAVE_FIG:
        fig_tf.savefig(NAME_OUT + '_TF.png', dpi=600, transparent=False)
