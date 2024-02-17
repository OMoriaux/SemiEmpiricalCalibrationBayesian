"""
ParaDRAM version of BK_Pinhole_MCMC.py, uses ParaDRAM MCMC method instead of Metropolis-Hastings MCMC method.
Apply the semi-empirical calibration method on empirical pinhole calibration data.
This version imports most of the setup from BK_Pinhole_MCMC.py, as only the MCMC differs.
This ParaDRAM is used throughout the thesis document:
 http://resolver.tudelft.nl/uuid:7cfd53ce-c443-43af-a0d3-205fa5468e8c.

Exported data can be processed using ParaDRAM_Process_MCMC.py.
"""
import os
import pickle
import numpy as np
import paramonte as pm
import BK_Pinhole_MCMC as BKpMCMC
import Source.BerghTijdemanWhitmoreModels as BtWM

# !!! MOST GLOBAL PARAMETERS ARE STILL DEFINED IN BK_Pinhole_MCMC.py !!!

# --- ADDITIONAL INPUTS ---
# Minimum and maximum allowable values for the full parameter vector.
# print('LRV_0: ', bk_p.LRV_PIN)  # Can be uncommented to see current initial guess values for all the parameters.
LRV_MIN = np.array([1E-3, 0.2E-3/2, 7E-9])
LRV_MAX = np.array([3E-3, 1E-3/2, 20E-9])

# Parameters for tweaking the Gaussian proposal distribution.
COV_MULT = 1E0  # Multiplication of the entire covariance matrix.
OWN_COV = True  # Define the covariance matrix, either using G_SD, or by centring a Gaussian between LRV_MIN and LRV_MAX
SD_COV_BOOL = False  # Use own input G_SD for the variance of the parameters, instead of based on LRV_MIN and LRV_MAX.

# Parameters for the adaptive Metropolis.
ACCEPTANCE_RATE = [0.10, 0.31]
ADAPTIVE_UPDATE_PERIOD = 50

TEST = False  # Test the posterior.

# --- MAIN ---
# Minimum and maximum normalised parameters of pinhole.
theta_min = np.array([LRV_MIN[0] / BKpMCMC.C0, LRV_MIN[1] * BKpMCMC.NU ** -0.5,
                      BtWM.f_vv_vt(length=LRV_MAX[0], radius=LRV_MAX[1], volume=LRV_MIN[2])])[BKpMCMC.PAR_SELECT]
theta_max = np.array([LRV_MAX[0] / BKpMCMC.C0, LRV_MAX[1] * BKpMCMC.NU ** -0.5,
                      BtWM.f_vv_vt(length=LRV_MIN[0], radius=LRV_MIN[1], volume=LRV_MAX[2])])[BKpMCMC.PAR_SELECT]
# Check if the initial guess lies between the extrema.
assert np.all(BKpMCMC.theta_0 >= theta_min)
assert np.all(BKpMCMC.theta_0 <= theta_max)

# Define the Gaussian proposal distribution covariance matrix.
if SD_COV_BOOL:
    s = COV_MULT * BKpMCMC.G_SD
else:
    p_str = 2.38 / BKpMCMC.theta_0.size ** 0.5 * np.ones(BKpMCMC.theta_0.size)
    s = COV_MULT * np.min(np.abs(np.concatenate(((theta_max - BKpMCMC.theta_0)[np.newaxis, :],
                                                 (theta_min - BKpMCMC.theta_0)[np.newaxis, :],
                                                 p_str[np.newaxis, :]), axis=0)), axis=0)
cov_mat = np.diag(s**2)

# Define the output file name.
dir_para_dram = os.path.join(os.sep.join(BKpMCMC.F_MCMC_OUT.split(os.sep)[:-1]), f"ParaDRAM_{BKpMCMC.CASE_NAME}")
file_para_dram = os.path.join(dir_para_dram, f"{BKpMCMC.CASE_NAME}")
if not os.path.exists(dir_para_dram):
    os.makedirs(dir_para_dram)


def main():
    # Make dictionaries to write to output file for MCMC.
    dct_conditions = {'C0': BKpMCMC.C0, 'NU': BKpMCMC.NU, 'GAMMA': BKpMCMC.GAMMA, 'PR': BKpMCMC.PR}
    dct_data = {'F_FLUSH': BKpMCMC.FILE_FLUSH, 'F_MIC': BKpMCMC.FILE_MIC, 'KEY_FLUSH_IN': BKpMCMC.IN_FLUSH,
                'KEY_FLUSH_OUT': BKpMCMC.OUT_FLUSH, 'KEY_MIC_IN': BKpMCMC.IN_MIC, 'KEY_MIC_OUT': BKpMCMC.OUT_MIC,
                'S_DCT': BKpMCMC.s_dct_mic}
    dct_mcmc = {'F_MASK_LST': BKpMCMC.F_MASK_LST, 'THETA_SD': BKpMCMC.THETA_SD, 'G_SD': BKpMCMC.G_SD,
                'SIG_M_AMP': BKpMCMC.SIG_M_AMP, 'NOTES': BKpMCMC.NOTES, 'SIG_M_PHASE': BKpMCMC.SIG_M_PHASE,
                'N_SAMPLES': BKpMCMC.N_SAMPLES, 'BT_MODE': BKpMCMC.BT_MODE, 'PAR_SELECT': BKpMCMC.PAR_SELECT,
                'LRV': BKpMCMC.LRV_PIN, 'SEED': BKpMCMC.SEED, 'PAR_STR': BKpMCMC.PAR_STR,
                'THETA_FULL': BKpMCMC.theta_full}
    dct_out = {'DATA': dct_data, 'CONDITIONS': dct_conditions, 'MCMC_SETTINGS': dct_mcmc}

    # Write output file to pickle file. Can be loaded by MCMC processing code:
    # Don't want to run MCMC again each time you want to process the MCMC results.
    with open(file_para_dram + '_parameters.pickle', 'wb') as handle:
        pickle.dump(dct_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # FITTING
    # !!! THIS SERVES AS AN EXAMPLE OF THE SETTINGS USED IN THE REPORT,
    # PLAYING WITH THESE PARAMETERS CAN YIELD IMPROVED RESULTS !!!
    # For example the delayed rejection functionality of this sampler has not been explored here.
    # pm.verify()
    pmpd = pm.ParaDRAM()  # define a ParaMonte sampler instance

    pmpd.mpiEnabled = True  # This is essential as it enables the invocation of the MPI-parallelized ParaDRAM routines.

    pmpd.spec.overwriteRequested = True  # overwrite existing output files if needed
    pmpd.spec.randomSeed = BKpMCMC.SEED  # initialize the random seed to generate reproducible results.
    pmpd.spec.outputFileName = file_para_dram
    # pmpd.spec.chainFileFormat = 'binary'
    pmpd.spec.restartFileFormat = ('binary', 'ascii')[0]
    pmpd.spec.progressReportPeriod = 1000
    pmpd.spec.chainSize = BKpMCMC.N_SAMPLES  # the default 100,000 unique points is too large for this simple example.
    # pmpd.spec.sampleSize = 10000
    # pmpd.spec.sampleRefinementCount = 1
    pmpd.spec.startPointVec = BKpMCMC.theta_0
    pmpd.spec.variableNameList = list(BKpMCMC.PAR_STR[BKpMCMC.PAR_SELECT])
    pmpd.spec.parallelizationModel = 'single chain'
    # pmpd.spec.adaptiveUpdateCount = ...  # larger easier for targetAcceptanceRate.
    # pmpd.spec.scaleFactor = '0.5'  # str(2.38/len(lrv0)**0.5)
    # pmpd.spec.sampleRefinementCount = 0

    pmpd.spec.domainLowerLimitVec = theta_min
    pmpd.spec.domainUpperLimitVec = theta_max
    pmpd.spec.targetAcceptanceRate = ACCEPTANCE_RATE  # Upper and lower value range.
    pmpd.spec.adaptiveUpdatePeriod = ADAPTIVE_UPDATE_PERIOD  # larger easier for targetAcceptanceRate.
    if OWN_COV:
        pmpd.spec.proposalStartCovMat = cov_mat

    # pmpd.spec.burninAdaptationMeasure = 1E-3
    # pmpd.spec.delayedRejectionCount = 5

    pmpd.runSampler(ndim=len(BKpMCMC.theta_0), getLogFunc=BKpMCMC.posterior)  # call the ParaDRAM sampler


if __name__ == '__main__':
    main()
elif TEST:
    print(BKpMCMC.posterior(theta_i=BKpMCMC.theta_0))  # Test if the code works.
# Can run the code by typing in terminal, e.g.: mpiexec -localonly -n 6 python .\ParaDRAM_BK_Pinhole_MCMC.py
