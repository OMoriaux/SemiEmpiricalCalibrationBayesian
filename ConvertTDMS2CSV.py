"""
File to convert TDMS data file to CSV file.
Can construct CSV files from provided TDMS files of the repository, which can be used to test both reader modes of
'PressureAcquisition' and 'sensitivity_calculation' in Source.CalibrationMeasurement.
"""
import os
import numpy as np
import pandas as pd
import Source.CalibrationMeasurement as cal_m

# --- INPUT ---
WRITE = False  # Write read DataFrame from TDMS file to CSV file.
TEST_READ = False  # Read the CSV file again, and compare to the original DataFrame from TDMS file.
# Filename to convert (input).
FILE_IN = os.path.join(
    '.', 'TestData', 'BK_Pinhole',
    ('Flush_1.tdms', 'Pinhole_1.tdms', 'Fake_example_unsteady_wall_pressure_data.tdms')[0])
# Output filename.
FILE_OUT = FILE_IN[:-5] + '.csv'

SAFE_READ = False  # TDMS reader mode.
# --- END OF INPUT ---
# --------------------

# --- MAIN CODE ---
# Read the TDMS file and save into pandas DataFrame.
if SAFE_READ:  # Save (but possibly slower) mode.
    df_data, df_prop_data = cal_m.tdms_safe_read(f_name=FILE_IN, return_properties=True)
else:  # Default mode.
    df_data = cal_m.tdms_to_dataframe(f_name=FILE_IN)
    df_prop_data = None

if WRITE:  # Write DataFrame to CSV file.
    df_data.to_csv(FILE_OUT)

if TEST_READ:  # Test written CSV file.
    df_data_read = pd.read_csv(FILE_OUT, index_col=0, header=[0, 1])  # Read CSV file.
    bool_write_equals_read = np.all(df_data == df_data_read)  # Compare all the data in the original and 'new' DataFrame
    print(f'Read DataFrame == Written DataFrame: {bool_write_equals_read}')  # Print result of comparison.
