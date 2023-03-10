# Semi-Empirical Calibration Bayesian
A novel post-processing method for empirical calibration data of remote microphone probes (RMP), 
using Bayesian inference (BI) and pre-existing analytical model for the frequency response of the considered probe.

Contains code to process empirical calibration data of RMPs, and unsteady pressure measurements with these RMPs.
Also includes the code implementation of the Bergh & Tijdeman [1965] model and Whitmore [2006] model for the transfer function of pneumatic systems.
As for the BI, a simple Metropolis-Hastings Markov-chain Monte Carlo is provided, as well as helper functions to pre-process the data etc.


## 1. Project structure
The code is meant to explain the presented method, and highlight its strengths. 

### a. ./Source
The core functions of the method are written in the Source directory. 
- **BayesianInferenceFunctions**: Code used to couple data with models using Bayesian Inference (BI); Metropolis-Hastings (MH), Markov-chain Monte Carlo (McMC).
- **BerghTijdemanWhitmoreModels**: Implementations of the Bergh & Tijdeman [1965] and Whitmore [2006] models.
- **CalibrationMeasurement**: Code used to ingest calibration and unsteady pressure measurement data from TDMS files. ! REQUIRES npTDMS
- **PlottingFunctions**: Functions used to plot the results of the TDMS data and McMC fitting results.
- **ProcessingFunctions**: Functions to process the TDMS data into power spectral density, transfer functions etc.

### b. Main (SemiEmpiricalCalibrationBayesian)
Example code, using these functions, are provided in the main directory (SemiEmpiricalCalibrationBayesian):
- **TestDataIngest**: Example of how to use Source/CalibrationMeasurement to process calibration (and measurement) data.
- **BK_Pinhole_McMC**: The semi-empirical calibration method using the Whitmore model, applied to the example calibration data. Outputs the results to *./McMC_Output*.
- **Process_McMC**: Further processing from the McMC results from BK_Pinhole_McMC. Outputs figures to *./McMC_Figures*.

### c. TutorialsInJupyter
These are Jupyter notebooks that aim to explain in a more step-by-step method the problem with idea behind the proposed method, and how to apply the method, and process its results.
- **Step 1**: Similar to TestDataIngest, where one reads the calibration data and observes spurious resonance.
- **Step 2**: A short overview of how the Whitmore model works, e.g., its parameters.
- **Step 3**: Setting up the semi-empirical calibration method for the example dataset.
- **Step 4**: Processing the results of the semi-empirical calibration method.
- **Extra - Basics of Bayesian Inference** (**WORK IN PROGRESS**): Meant to explain basics of MH McMC, e.g., the step size of the sampling Gaussian.


## 2. Sources
- Bergh, H., and Tijdeman, H., “Theoretical and experimental results for the dynamic response of pressure measuring systems,” Tech. Rep. NLR-TR F. 238, National Aerospace Laboratory NLR, Amsterdam, The Netherlands, Jan. 1965. http://resolver.tudelft.nl/uuid:e88af84e-120f-4c27-8123-3225c2acd4ad.
- Whitmore, S. A., “Frequency response model for branched pneumatic sensing systems,” Journal of Aircraft, Vol. 43, No. 6, 2006, pp. 1845–1853. https://doi.org/10.2514/1.20759.


## 3. Required Python install and packages
- Python 3.9 or newer (still testing older versions)
- Numpy
- Scipy
- Matplotlib
- Seaborn
- Pandas (tdms files are read into Pandas DataFrames, arrays with named columns)
- npTDMS (reads LabVIEW VI output files, i.e. tdms files)

! npTDMS will need to be installed in most cases. Either install the package in the user's prefered manner or simply run ">>> pip install npTDMS" in the Python console.
