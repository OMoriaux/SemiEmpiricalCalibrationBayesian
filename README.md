# SemiEmpiricalCalibrationBayesian
A novel post-processing method for empirical calibration data of remote microphone probes (RMP), 
using Bayesian inference (BI) and pre-existing analytical model for the frequency response of the considered probe.

Contains code to process empirical calibration data of RMPs, and unsteady pressure measurements with these RMPs.
Also includes the code implementation of the Bergh & Tijdeman [1965] model and Whitmore [2006] model for the transfer function of pneumatic systems.
As for the BI, a simple Metropolis-Hastings Markov-chain Monte Carlo is provided, as well as helper functions to pre-process the data etc.

Sources:
Bergh, H., and Tijdeman, H., “Theoretical and experimental results for the dynamic response
of pressure measuring systems,” Tech. Rep. NLR-TR F. 238, National Aerospace Laboratory NLR,
Amsterdam, The Netherlands, Jan. 1965. http://resolver.tudelft.nl/uuid:e88af84e-120f-4c27-8123-3225c2acd4ad.

Whitmore, S. A., “Frequency response model for branched pneumatic sensing systems,” Journal of Aircraft,
Vol. 43, No. 6, 2006, pp. 1845–1853. https://doi.org/10.2514/1.20759.



Required Python install and packages:
- Python 3.6 or newer
- Numpy
- Scipy
- Matplotlib
- Pandas (tdms files are read into Pandas DataFrames, arrays with named columns)
- npTDMS (reads LabVIEW VI output files, i.e. tdms files)
