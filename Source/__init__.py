"""
Helper functions ranging from tube-cavity models (Whitmore and Bergh & Tijdeman), the functions for those models,
the functions for Bayesian inference including pre-processing calibration with band-passing, Markov-chain Monte-Carlo,
and some processing functions for calibration data.

__________________________________________________________________________________
Code used for the Bayesian Inference lecture in the lecture series:
Remote microphone techniques for the characterization of aeroacoustic sources.

__________________________________________________________________________________
Required Python install and packages:
- Python 3.9 or newer (still testing older verions)
- Numpy
- Scipy
- Matplotlib
- Pandas (tdms files are read into Pandas DataFrames, arrays with named columns)
- npTDMS (reads LabVIEW VI output files, i.e. tdms files)

__________________________________________________________________________________
Functions and classes:
- ...
"""
import sys
import warnings
import importlib
try:
    from packaging.version import parse as parse_version
except ModuleNotFoundError:
    warnings.warn("packaging package is not found. "
                  "This is used to test if the required packages are installed. Code works without."
                  "Can install package, e.g., using in console >>> pip install packaging", ImportWarning)
# Import the various HelperFunction files.
from . import BerghTijdemanWhitmoreModels
from . import BayesianInferenceFunctions
from . import CalibrationMeasurement
from . import ProcessingFunctions
from . import PlottingFunctions

__version__ = '0.5'
__authors__ = u'Olivier Moriaux'
__copyright__ = "Copyright 2023, VKI"
__credits__ = u'Olivier Moriaux, Riccardo Zamponi, Christophe Schram'  # Not sure if latter two want to be included?
__license__ = "CC BY"
__maintainer__ = "Olivier Moriaux"
__email__ = "olivier dot moriaux at vki dot ac dot be"
__status__ = "Development"  # "Prototype", "Development", or "Production"

req_version = (3, 9)
cur_version = sys.version_info

if cur_version < req_version:
    raise Exception("Your Python interpreter %s.%s.%s is too old. Please consider upgrading." % cur_version[:3])


if 'parse_version' in globals():
    def _check_versions():
        for modname, minver in [
                ("numpy", "1.17"),  # 1.20
                ("scipy", "1.7.3"),  # 1.6.0
                ("pandas", "1.3.3"),
                ("nptdms", "1.4.0"),
                ("matplotlib", "3.5.1"),
                ("seaborn", "0.12.2")
        ]:
            module = importlib.import_module(modname)
            if parse_version(module.__version__) < parse_version(minver):
                '''
                raise ImportError(f"HelperFunctions requires {modname}>={minver}; "
                                  f"you have {module.__version__}")
                '''
                # TODO: Check the absolute minimum specs, so can use Error instead of Warning.
                raise ImportWarning(f"HelperFunctions tested with {modname}>={minver}; "
                                    f"you have {module.__version__}. Code might still work for lower versions.")


    _check_versions()


__all__ = ["CalibrationMeasurement", "PlottingFunctions", "BayesianInferenceFunctions", "BerghTijdemanWhitmoreModels"]
