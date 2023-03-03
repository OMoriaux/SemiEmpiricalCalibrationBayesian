import os
import pickle
import numpy as np
import Source.PlottingFunctions as plot_f
import Source.ProcessingFunctions as proc_f
import Source.CalibrationMeasurement as cal_c
import Source.BayesianInferenceFunctions as bi_f
import Source.BerghTijdemanWhitmoreModels as model_f

# --- INPUT ---
RUN_MCMC = True  # To actually run the McMC, run this file, or run main(). If False, only shows data and initial guess.
CASE_NAME = 'BK2_Pinhole_02'  # Name used for export file of McMC output.
# - Calibration data.
FILE_FLUSH = os.path.join('.', 'TestData', 'BK_Pinhole_2', 'Flush_1.tdms')
FILE_MIC = os.path.join('.', 'TestData', 'BK_Pinhole_2', 'Pinhole_1.tdms')
IN_FLUSH, OUT_FLUSH = [('Untitled', 'Channel 1')], [('Untitled', 'Channel 2')]
IN_MIC, OUT_MIC = [('Untitled', 'Channel 2')], [('Untitled', 'Channel 1')]

# - Model.
LRV_PIN = (2E-3, 0.5E-3/2, 10E-9)  # Length, radius, volume of pinhole probe used as initial guess. Also used for prior.
# LRV_PIN = (2E-3, 1.58662804e-04, 3.00340462e-09)
GAMMA, PR, C0, NU = 1.4, 0.7, 340.26, 1.46E-5  # Conditions used in model.

# - Bayesian Inference.
# Frequency mask list. Band-passes the TF data used for BI. List of lists,
# i.e. [[start frequency 0, end frequency 0], [start frequency 1, end frequency 1], ...].
F_MASK_LST = [[1E2, 2.3E3], [4.5E3, 8E3]]
# Standard deviation of prior probability density function (PDF). Considers normalised parameters.
ALPHA_SD = np.array([1.5E-6, 1.8E-2, 6.4E1])
# Standard deviation of Gaussian used for sampling of McMC. Considers normalised parameters.
G_SD = np.array([4.5E-9, 1E-3, 1.92E-1]) * 1E0
SIG_M_AMP = 0.5  # Measurement error standard deviation of TF amplitude. Used in likelihood PDF.
SIG_M_PHASE = 0.2  # Measurement error standard deviation of TF phase. Used in likelihood PDF.
N_SAMPLES = 20000  # Total number of samples in McMC. Includes rejected samples.
N_UPDATES = N_SAMPLES//10  # Number of samples in McMC after which each time the current acceptance rate is printed.
SEED = 3751  # Seed used for Gaussian sampling distribution and Uniform sieve in McMC.
BT_MODE = False  # Boolean to select Bergh & Tijdeman model istead of Whitmore model. Need to provide models for both.
PAR_SELECT = np.array([1, 2])  # Indices used to select fitting parameters of BI.
PAR_STR = np.array(['L_c', 'R_nu', 'Vv_Vt'])  # Parameter label strings.
NOTES = ''  # Optional notes to save in McMC output file.

F_MCMC_OUT = os.path.join('.', 'McMC_Output', f'{CASE_NAME}.pickle')  # Output file.

# --- END OF INPUT ---
# --------------------

# --- CALIBRATION DATA ---
cal_flush = cal_c.PressureAcquisition(FILE_FLUSH, safe_read=False)
cal_mic = cal_c.PressureAcquisition(FILE_MIC, safe_read=False)

s_dct_mic = dict.fromkeys([cal_mic.channel_float_type_dct, cal_mic.data_channel_names][1], 1E3)
# Don't know the actual sensitivity. Just set flat part of TF amplitude to 1.
s_dct_mic[('Untitled', 'Channel 3')] = 1.076 * 1E3
cal_mic.set_sensitivities(dct_channel_sensitivity=s_dct_mic, strict_mode=True, dct_prop_str_new_val=None)

df_tf_flush = cal_flush.transfer_function(in_channel=IN_FLUSH, out_channel=OUT_FLUSH, set_property=True)
df_tf_mic = cal_mic.transfer_function(in_channel=IN_MIC, out_channel=OUT_MIC, set_property=True)
df_tf_full = cal_flush.add_transfer_function_step(
    df_tf_new=df_tf_mic, dct_old_tf_to_new_tf_channels=dict(zip(df_tf_flush.columns, df_tf_mic.columns)))

f_arr = df_tf_full.index.to_numpy(float)  # Frequency array (at which TF is estimated from measurement files).
# Create masking array for band-passing calibration data to BI.
mask_arr, f_masked = bi_f.mask_f_data(frequency=f_arr, f_mask_arr=F_MASK_LST)
w_masked = 2*np.pi*f_masked  # Band-passed angular frequency array.
# Calibration TF amplitude and phase.
amp_d, phase_d = proc_f.frequency_response(df_tf_full.iloc[:, 0].to_numpy(complex), axis=0)
amp_d_masked, phase_d_masked = amp_d[mask_arr], phase_d[mask_arr]  # Band-pass calibration data.

# --- BAYESIAN INFERENCE PREPARATION ---
alpha_full = model_f.dim_to_norm(*LRV_PIN, c0=C0, nu=NU, alpha_complex=False)  # All normalised parameters of pinhole.
alpha_0 = alpha_full[PAR_SELECT]  # Only parameters used for BI.
prior_obj = bi_f.PriorArray(mean_array=alpha_0, sd_array=ALPHA_SD[PAR_SELECT])  # Prior PDF object.


# Models (both B&T and W) used for BI to calibration data.
def model_bt(alpha_i, w_arr=w_masked, alpha_long=alpha_full, par_idx=PAR_SELECT):
    """
    Bergh & Tijdeman model that ought to fit calibration data.

    :param alpha_i: Sample fitting parameters for model.
    :param w_arr: Angular frequency array at which TF is modelled.
    :param alpha_long: Full array of model parameters (not only the ones used for fitting).
    :param par_idx: Index array indicating where alpha_i parameters are in the alpha_long array.

    :return: Tuple of amplitude array and phase array of model at all frequencies of w_arr.
    """
    alpha_long[par_idx] = alpha_i  # Insert sample alpha_i parameters into long parameter array.
    # Build here whatever kind of model best represents the probe from which calibration data is available. Example:
    pr = model_f.bt_pinhole(w_arr=w_arr, k_l_pre=alpha_long[0], alpha_pre=alpha_long[1]*1j**1.5, vv_vt=alpha_long[2],
                            sigma=0, gamma=GAMMA, k_n=1., pr=PR, p1_p0=0)  # Complex-valued TF array.
    return proc_f.frequency_response(pr)  # Convert complex-valued TF into tuple of amplitude and phase of TF.


def model_w(alpha_i, w_arr=w_masked, alpha_long=alpha_full, par_idx=PAR_SELECT):
    """
    Whitmore model that ought to fit calibration data.

    :param alpha_i: Sample fitting parameters for model.
    :param w_arr: Angular frequency array at which TF is modelled.
    :param alpha_long: Full array of model parameters (not only the ones used for fitting).
    :param par_idx: Index array indicating where alpha_i parameters are in the alpha_long array.

    :return: Tuple of amplitude array and phase array of model at all frequencies of w_arr.
    """
    alpha_long[par_idx] = alpha_i  # Insert sample alpha_i parameters into long parameter array.
    # Build here whatever kind of model best represents the probe from which calibration data is available. Example:
    pr = model_f.whitmore_pinhole(k_l_pre=alpha_long[0], alpha_pre=alpha_long[1]*1j**1.5, vv_vt=alpha_long[2],
                                  w_arr=w_arr, gamma=GAMMA, pr=PR, sum_i_phi_ij=0)  # Complex-valued TF array.
    return proc_f.frequency_response(pr)  # Convert complex-valued TF into tuple of amplitude and phase of TF.


if BT_MODE:  # Select which model is used for BI.
    model = model_bt
else:
    model = model_w


def posterior(alpha_i):
    """
    Logarithmic posterior PDF used for BI. Computes log rho(alpha_i|d) = log rho(d|alpha_i) + log rho_0 (alpha_i).
    Likelihood contains model evaluation, i.e. computationally most expensive part of the BI method.

    :param alpha_i: Sample fitting parameters.

    :return: Log posterior PDF computed at alpha_i.
    """
    amp_m, phase_m = model(alpha_i=alpha_i)  # Model TF computed at band-passed angular frequencies.
    a = 0  # log posterior PDF. Written like this so three contributions below can be commented out, if desired.
    # Commenting-out prior can show whether prior is constricting fit too much.
    a += prior_obj.log_pdf(array_i=alpha_i)  # Add log prior PDF.
    a += bi_f.likelihood_log_pdf(u=amp_m, d=amp_d_masked, sigma_d=SIG_M_AMP)  # Add log likelihood of TF amplitude.
    a += bi_f.likelihood_log_pdf(u=phase_m, d=phase_d_masked, sigma_d=SIG_M_PHASE)  # Add log likelihood of TF phase.
    return a  # Return log posterior PDF.


# --- RUN MARKOV-CHAIN MONTE CARLO (mcmc_run=True) OR SIMPLY PLOT DATA AND INITIAL GUESS ---
def main(mcmc_run=RUN_MCMC):
    """
    Actual code that runs McMC (optional) and plots calibration data with initial guess model TF
    (and best fit model TF; optional). Encompassed in function, such that the model functions can be imported by McMC
    visualisation code without running the code in this function.

    :param mcmc_run: Boolean to run McMC or not. Otherwise, only plots calibration data and initial guess model TF.

    :return: None.
    """
    if mcmc_run:
        # Run McMC.
        alpha_chain, rho_chain, dct_seed = bi_f.mcmc_update(n_params=len(alpha_0), posterior_log_pdf=posterior,
                                                            alpha_0=alpha_0, n_samples=N_SAMPLES, seed=SEED,
                                                            sigma_p=G_SD[PAR_SELECT], n_updates=N_UPDATES)
        # Make dictionaries to write to output file for McMC.
        dct_conditions = {'C0': C0, 'NU': NU, 'GAMMA': GAMMA, 'PR': PR}
        dct_data = {'F_FLUSH': FILE_FLUSH, 'F_MIC': FILE_MIC, 'KEY_FLUSH_IN': IN_FLUSH, 'KEY_FLUSH_OUT': OUT_FLUSH,
                    'KEY_MIC_IN': IN_MIC, 'KEY_MIC_OUT': OUT_MIC, 'S_DCT': s_dct_mic}
        dct_mcmc = {'F_MASK_LST': F_MASK_LST, 'ALPHA_SD': ALPHA_SD, 'G_SD': G_SD, 'SIG_M_AMP': SIG_M_AMP,
                    'NOTES': NOTES, 'SIG_M_PHASE': SIG_M_PHASE, 'N_SAMPLES': N_SAMPLES, 'BT_MODE': BT_MODE,
                    'PAR_SELECT': PAR_SELECT, 'LRV': LRV_PIN, 'SEED': dct_seed, 'PAR_STR': PAR_STR,
                    'ALPHA_FULL': alpha_full}
        dct_out = {'DATA': dct_data, 'CONDITIONS': dct_conditions, 'MCMC_SETTINGS': dct_mcmc,
                   'alpha_chain': alpha_chain, 'rho_chain': rho_chain}

        # Write output file to pickle file. Can be loaded by McMC processing code:
        # Don't want to run McMC again each time you want to process the McMC results.
        with open(F_MCMC_OUT, 'wb') as handle:
            pickle.dump(dct_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # For plotting: Best fitting parameters (MAP: Maximum A Posteriori).
        alpha_map = alpha_chain[np.argmax(rho_chain), :]  # Parameter array.
        amp_m_map, phase_m_map = model(alpha_i=alpha_map, w_arr=2*np.pi*f_arr)  # Amplitude and phase of TF.
    else:
        # Otherwise, Pycharm complains about variables not existing in all circumstances.
        amp_m_map, phase_m_map = None, None
        dct_out = None

    amp_m_0, phase_m_0 = model(alpha_i=alpha_0, w_arr=2*np.pi*f_arr)  # Initial guess model TF.

    # --- VISUALISATION ---
    fig_tf, ax_tf = plot_f.plot_transfer_function_df(df=df_tf_flush, fig_dim=(4, 5), color='b', linestyle=':',
                                                     alpha=1., prefix='Flush: ')
    plot_f.plot_transfer_function_df(df=df_tf_mic, ax=ax_tf, color='r', linestyle='-.', alpha=0.5,
                                     prefix='Mic: ')
    plot_f.plot_transfer_function_df(df=df_tf_full, ax=ax_tf, color='k', linestyle='--', alpha=0.8,
                                     prefix='Full: ', legend_loc='lower left', minor_phase=0.5, x_lim=(1E2, 13.5E3),
                                     y_lim_amp=(0, 4), y_lim_phase=(-2*np.pi-0.1, 1*np.pi+0.3))

    ax_tf[0].plot(f_arr, amp_m_0, color='g', linestyle=':', label=r'$\overline{\alpha}_0$')  # Initial guess.
    ax_tf[1].plot(f_arr, phase_m_0, color='g', linestyle=':', label=r'$\overline{\alpha}_0$')
    ax_tf[1].legend(['d: Flush', 'd: Mic', 'd: Total', r'$\overline{\alpha}_0$'], loc='lower left')
    if mcmc_run:  # MAP.
        ax_tf[0].plot(f_arr, amp_m_map, color='m', linestyle='-', label=r'$\overline{\alpha}_{MAP}$')
        ax_tf[1].plot(f_arr, phase_m_map, color='m', linestyle='-', label=r'$\overline{\alpha}_{MAP}$')
        ax_tf[1].legend(['d: Flush', 'd: Mic', 'd: Total', r'$\overline{\alpha}_0$', r'$\overline{\alpha}_{MAP}$'],
                        loc='lower left')
    fig_tf.tight_layout(pad=0.1)  # Tries to fill white-space in figure.
    fig_tf.show()  # Shows the figure.
    return dct_out


# If this code is made to run, then the main function will run by itself.
# If the code is imported in another piece of code, then main will not run by itself.
# Alternatively, when importing the code into console to run the code,
# one should still call the main function to actually run the code.
if __name__ == '__main__':
    main()
