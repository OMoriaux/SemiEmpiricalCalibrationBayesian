"""
Main object 'PressureAcquisition' and helper functions used to process microphone calibration and measurement TDMS
files.
"""
import copy
import warnings
import numpy as np
import pandas as pd
try:
    from nptdms import TdmsFile
except ModuleNotFoundError:
    # TODO: Implement as a warning, allowing data to be ingested through pickle- or other files.
    # Only warn, such that one might still use the code (in Source) for processing of data,
    # importing the data some other way.
    '''
    warnings.warn("npTDMS package is not found. "
                  "Features STRONGLY limited.\n"
                  "Install package, e.g., using in console >>> pip install npTDMS", ImportWarning)
    '''
    raise ModuleNotFoundError("npTDMS package is not found. Features STRONGLY limited.\n"
                              "Install package, e.g., using in console >>> pip install npTDMS")
from scipy.signal.windows import hann
from scipy.signal import butter, sosfilt
from scipy.interpolate import PchipInterpolator
import Source.ProcessingFunctions as proc_f
import Source.PlottingFunctions as plot_f
from pandas.core.frame import DataFrame
from typing import Union, Tuple, Any, Optional, Dict, List, Sequence


def tdms_to_dataframe(f_name: str) -> DataFrame:
    """
    Load TDMS file into Pandas dataframe. Header is multi-level: level 1 is the group name, level 2 is the channel name.
    Seems to be rather universal for TDMS files created by NI VIs.
    Loads all data at once. Will be problematic for very large files.
    Function working seems strongly dependent on version of npTDMS package. Written for v1.3(.1).

    :param f_name: TDMS file name.

    :return: Pandas dataframe.
    """
    try:
        with TdmsFile.read(f_name) as f:  # Open file.
            # Get list of all groups, with all the channels for each group in the TDMS file.
            lst_group_chan = [(group_i.name, chan_j.name)  # (chan_j.group_name, chan_j.name)
                              for group_i in f.groups()
                              for chan_j in group_i.channels()]
            # Load entire file as DataFrame, with ugly column names.
            df = f.as_dataframe()

        # Build multi-level (group and channel) header.
        header = pd.MultiIndex.from_tuples(lst_group_chan, names=('Group', 'Channel'))
        df.columns = header  # Rename ugly dataframe header to multi-level header.
    except NameError:
        raise NameError("TdmsFile not imported. npTDMS package not installed.\n"
                        "Install package, e.g., by typing in console: >>> pip install npTDMS")
    # Return dataframe with improved header.
    return df


def tdms_safe_read(f_name: str, sample_number_str: str = 'wf_samples', return_properties: bool = False) -> \
        Union[None, DataFrame, Tuple[Union[None, DataFrame], Union[None, DataFrame]]]:
    """
    Load TDMS file into Pandas dataframe, similarly to 'tdms_to_dataframe' but in a slower, albeit safer way.

    :param f_name: TDMS file name.
    :param sample_number_str: TDMS channel property name for the number of samples in each channel.
        If the key cannot be found in the channel properties, then it is computed by reading all the channels.
    :param return_properties: If True, returns not only the DataFrame of the TDMS file contents,
        but also a DataFrame of the channel properties. Else, only the DataFrame of the TDMS data is returned.

    :return: If return_properties: DataFrame of data, DataFrame of properties. Else: DataFrame of data.
    """
    try:
        df_prop_file = get_all_properties(file_path=f_name)  # Read the properties of the TDMS data channels.
        group_chan_lst = list(df_prop_file.index)  # Get list of tuples: [..., (group_i, channel_j), ...].
        # Get the maximum amount of samples for all the channels in TDMS file.
        try:  # If the amount of samples is provided as a property of the TDMS channels.
            n_samp_max = df_prop_file[sample_number_str].max()
        except KeyError:  # Otherwise, open all channels of the file, and compute their length (number of samples).
            n_samp_max = 0
            with TdmsFile.read(f_name) as f:
                for group_chan_i in group_chan_lst:
                    n_samp_max = max(n_samp_max, len(f[group_chan_i[0]][group_chan_i[1]][:]))
        # Build the DataFrame with the sample number as index,
        # and the column names as the (group, channel) names of each channel.
        df_data = pd.DataFrame(index=np.arange(n_samp_max),
                               columns=pd.MultiIndex.from_tuples(group_chan_lst, names=('Group', 'Channel')))
        with TdmsFile.read(f_name) as f:  # Open file.
            for group_chan_i in group_chan_lst:  # Go through all the (group_i, channel_j).
                chan_data_i = f[group_chan_i[0]][group_chan_i[1]][:]  # Read data from file.
                n_chan_len = chan_data_i.size  # Get the amount of samples of the specific channel.
                try:  # Try to assign data to DataFrame.
                    df_data.loc[:n_chan_len-1, group_chan_i] = chan_data_i
                except KeyError:  # If the key is not part of the DataFrame, fill it with NaN.
                    df_data.loc[:, group_chan_i] = np.nan

        # Convert the columns of the DataFrame to the correct data types.
        dct_types = get_type_dict_channels(df=df_data)  # Get the data types of the first element of each column.
        try:  # Try to assign all data types for all channels.
            df_data = df_data.astype(dct_types)
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:  # If there is an issue with the time channels:
            # Only convert channels of the float type, i.e. not time.
            dct_types = {key: dct_types[key] for key in dct_types if dct_types[key] is float}
            df_data = df_data.astype(dct_types)

        if return_properties:  # Not only return the data DataFrame, but also the property DataFrame.
            df_prop_file.loc[list(dct_types.keys()), 'dtypes'] = list(dct_types.values())
            return df_data, df_prop_file
        else:  # Only return the data DataFrame.
            return df_data
    except NameError:
        raise NameError("TdmsFile not imported. npTDMS package not installed.\n"
                        "Install package, e.g., by typing in console: >>> pip install npTDMS")


def get_all_properties(file_path: str) -> DataFrame:
    """
    Return a DataFrame with the properties of each channel in the provided TDMS file.

    :param file_path: TDMS file name.

    :return: DataFrame containing properties.
    """
    prop_dct = {}
    with TdmsFile.read(file_path) as f:  # Open the file.
        for group_i in f.groups():
            for channel_j in group_i.channels():
                # Assign to key (group_i, channel_j), all the properties of the specific channel.
                prop_dct[(group_i.name, channel_j.name)] = channel_j.properties
    # Convert to a DataFrame.
    df_prop = pd.DataFrame.from_dict(prop_dct)  # , orient='index')
    df_prop = df_prop.transpose()  # Transpose the DataFrame.
    return df_prop


def var_kwargs(var_str: str, default_val: Any, kwargs: dict) -> dict:
    """
    See if a specific 'var_str' key is present in the 'kwargs' dictionary,
    otherwise assign the 'default_val' value to it.

    :param var_str: Key in the 'kwargs' dictionary.
    :param default_val: Value to assign to the 'var_str' key if it is not already present in the 'kwargs' dictionary.
    :param kwargs: Dictionary of parameters and their assigned values.

    :return: Updated 'kwargs' dictionary.
    """
    if var_str not in kwargs.keys():  # If the key is not already present in kwargs:
        kwargs[var_str] = default_val  # Assign a default value to that key.
    return kwargs  # Return (updated) dictionary.


def get_type_dict_channels(df: DataFrame) -> dict:
    """
    Get the data types of each column of a DataFrame, based on the first row of each column?

    :param df: DataFrame.

    :return: Dictionary with as keys the column names, and as assigned values the data types.
    """
    chan_lst = df.columns  # Get column names of DataFrame, e.g. (group_i, channel_j).
    ds_first_index = df.iloc[0, :]  # Only keep first row of DataFrame.
    type_lst = [type(elem) for elem in ds_first_index]  # Get type of first row element for all columns.
    return dict(zip(chan_lst, type_lst))  # Make dictionary: {..., column_name: data type, ...}.


class PressureAcquisition:
    def __init__(self, file_path: str, safe_read: bool = False, fs: float = 51200, window_size: int = 2**15,
                 label_format: str = '%(file)s: %(channel)s', axis_format: str = '%(var)s, %(unit)s',
                 key_process: str = '%(file_in)s_%(channel_in)s>%(file_out)s_%(channel_out)s', **kwargs):
        """
        Object used for reading and processing TDMS files, for microphone calibration and measurements.

        Used to:
            1. Estimate TFs from empirical frequency-domain microphone calibrations,
                combine various calibration steps together.
            2. Assign said TF to a microphone measurement, and compute the power-spectral density.

        :param file_path: File path of the data TDMS file, from either calibration or measurement.
        :param safe_read: Safe (but slower) mode for reading the TDMS file. Default: False.
        :param fs: Sampling frequency of the data, Hz.
            TODO: Get from file properties instead of having to define, also channel specific.
        :param window_size: Default window size used for the operations of this object, e.g. psd, coherence,
            and transfer function methods.
        :param label_format: Format used to convert file and channel information into a string
            that can be used as label in a plot. Can contain a 'file' and 'channel' variable.
        :param axis_format: Format used for axes of plots, e.g., 'E, V' or 'E [V]' etc.
            Can contain a 'var' and 'unit' variable.
        :param key_process: Format used to assign column names to DataFrames output from the methods
            that compare one signal to another, e.g. coherence and all the frequency response/transfer function methods.
            Can contain a 'file_in', 'file_out', 'channel_in', 'channel_out' variable.
        :param kwargs: Optional input parameters:
            - 'sample_number_str' for the 'tdms_safe_read' function. Default: 'wf_samples'.
        """
        # Set default parameters of kwargs.
        kwargs = var_kwargs(var_str='sample_number_str', default_val='wf_samples', kwargs=kwargs)
        # Read the TDMS file.
        if safe_read:  # Save (but slower) mode.
            self.df_data, self.df_prop_data = tdms_safe_read(f_name=file_path, return_properties=True, **kwargs)
        else:  # Default mode.
            self.df_data = tdms_to_dataframe(f_name=file_path)
            if 'TdmsFile' in globals():
                self.df_prop_data = get_all_properties(file_path=file_path)
            else:
                self.df_prop_data = pd.DataFrame(index=list(self.df_data.columns), columns=['dtypes'])

        dct_types = get_type_dict_channels(df=self.df_data)  # Data types of each data column.
        # Write data types to property DataFrame.
        self.df_prop_data.loc[list(dct_types.keys()), 'dtypes'] = list(dct_types.values())
        # List of all data columns that have as type float.
        # TODO: Might need a smarter way to catch all relevant channels.
        # Only these will be processed when 'which="all"' for the methods of this class.
        self.channel_float_type_dct = np.unique(
            list(self.df_prop_data.loc[self.df_prop_data['dtypes'] == np.float64].index) +
            list(self.df_prop_data.loc[self.df_prop_data['dtypes'] == float].index)
            , axis=0)
        self.channel_float_type_dct = [tuple(elem) for elem in self.channel_float_type_dct]

        self.data_channel_names = list(self.df_prop_data.index)  # Get all the data DataFrame columns.

        self.fs = fs  # Sampling frequency.
        # Define default processing settings for any time- to frequency-domain methods, e.g. Welch.
        # For the transfer function, by default 'return_onesided' is True.
        # Can further change in the kwargs of the transfer function method.
        self.default_proc_settings = {'fs': fs, 'noverlap': None, 'window': hann(window_size, sym=True),
                                      'nfft': 2*window_size, 'detrend': False, 'axis': 0}  # , 'return_onesided': True}

        # Plotting parameters.
        self.label_format = label_format.lower()  # TODO: Currently not implemented.
        # - Define the key for columns of DataFrames from a 'process' method, i.e. from channel x to channel y, e.g. TF.
        self.key_process = key_process.lower()
        self.axis_format = axis_format.lower()

        # TODO: Currently not implemented. Want to change plotting based on if sensitivity set or not.
        self.data_var_and_unit = ["E", 'V']
        self.time_var_and_unit = ["t", 's']
        self.frequency_var_and_unit = ["f", 'Hz']

        # Properties of the object that are set by the methods.
        self.sensitivities = None  # Sensitivities of the data DataFrame.
        self.df_tf = None  # Transfer function DataFrame.

    def set_sensitivities(self, dct_channel_sensitivity: Optional[Dict[Union[str, Tuple[str, str]], float]] = None,
                          strict_mode: bool = True, dct_prop_str_new_val: Optional[Dict[str, Any]] = None):
        """
        Assign microphone sensitivities (in mV/Pa) to the various channels.

        :param dct_channel_sensitivity: Dictionary containing the sensitivity values, [mV/Pa].
            This dictionary is rejected with an attached print statement in case all the channels of data are not
            found in the keys of the dictionary AND strict_mode is True.
            Check the {self.channels_data} to see all the necessary keys.
            Default None: The default is a way to assign a sensitivity, without altering the data, i.e. 1 V/Pa.
        :param strict_mode: Use this method is strict or non-strict mode. For strict mode all the channels need to have
            a defined sensitivity value in dct_channel_sensitivity.
        :param dct_prop_str_new_val: For the channels where the sensitivity is set, assign or change in the property
            DataFrame a new value to a certain property in the DataFrame. This is a dictionary,
            where the key is the property string in the property DataFrame and the assigned value is the value to assign
            to said property string.

        :return: Nothing. Simply applies microphone sensitivities to the {self.df_data} DataFrame used for processing.
        """
        # Lazy default. There is a cleaner way to write this, I simply lost that code.
        if dct_channel_sensitivity is None:
            dct_channel_sensitivity = dict(zip(self.data_channel_names, len(self.data_channel_names)*[1E3]))
        # If strict mode: Check for each data channel if a sensitivity is provided in the dictionary.
        if all(k in dct_channel_sensitivity.keys() for k in self.data_channel_names) or not strict_mode:
            for channel_i in dct_channel_sensitivity.keys():  # V -> V / (milliV/Pa 1/milli) = Pa.
                self.df_data.loc[:, channel_i] /= dct_channel_sensitivity[channel_i] * 1E-3
                if dct_prop_str_new_val is not None:
                    for key in dct_prop_str_new_val:
                        self.df_prop_data.loc[channel_i, key] = dct_prop_str_new_val[key]
            self.sensitivities = dct_channel_sensitivity  # Save the microphone sensitivities.
            self.data_var_and_unit = ["p'", 'Pa']  # TODO: Need to implement this for plotting.
        else:  # If strict more and there are missing keys.
            print(f'Missing keys:',
                  [key for key in self.data_channel_names if key not in dct_channel_sensitivity.keys()])

    def set_transfer_function(self, df_tf: DataFrame,
                              dct_channel_transfer_function: Dict[Union[str, Tuple[str, str]], str], axis: int = 0):
        """
        Assign microphone probe TFs to the various channels.

        :param df_tf: DataFrame of TFs for all the relevant data channels to be corrected with the TFs.
        :param dct_channel_transfer_function: Dictionary that translates the channels of the measurement data
            to the channels of the TF DataFrame. So {'measurement column key': 'TF column key'}.
        :param axis: Axis along which to apply the TF, in case set_property is True.

        :return: Nothing. Simply applies microphone sensitivities to the {self.df_data} DataFrame used for processing.
        """
        if dct_channel_transfer_function is None:  # Same names assumed, so no translation dictionary.
            dct_channel_transfer_function = dict(zip(self.channel_float_type_dct, self.channel_float_type_dct))

        # Create DataFrame of TFs for each channel of the data DataFrame.
        df_tf_copy = pd.DataFrame(index=df_tf.index, columns=pd.MultiIndex.from_tuples(self.channel_float_type_dct),
                                  dtype=complex)
        for channel_i in self.channel_float_type_dct:
            # If the channel is available in dct_channel_transfer_function, assign to df_tf_copy.
            # Otherwise, assign 1+0j TF to said channel, i.e. flat TF.
            if channel_i in dct_channel_transfer_function.keys():
                df_tf_copy.loc[:, channel_i] = df_tf.loc[:, dct_channel_transfer_function[channel_i]]
            else:
                df_tf_copy.loc[:, channel_i] = 1 + 0j

        # Apply TF.
        f_int = np.fft.rfftfreq(n=self.df_data.loc[:, self.channel_float_type_dct].shape[axis], d=1 / self.fs)
        tf_int = PchipInterpolator(df_tf_copy.index.to_numpy(), df_tf_copy.to_numpy(), axis=axis)(f_int)

        # FFT computation in frequency domain.
        # - Single-sided fourier transform
        fft_data = np.fft.rfft(self.df_data.loc[:, self.channel_float_type_dct], axis=axis)

        pre = fft_data * tf_int  # Multiply (R)FFT[Data] with (single-sided) frequency-domain TF.
        # Inverse (R)FFT and assign to channels of data DataFrame.
        self.df_data.loc[:, self.channel_float_type_dct] = np.fft.irfft(pre, axis=axis)

    # @plot_f.all_plotting_decorator(plot_func=functools.partial(plot_f.plot_single_df, x_str='t, s'))
    @plot_f.all_plotting_decorator(plot_func=plot_f.plot_single_df)
    def raw(self, which: Union[str, List[Tuple[str, str]]] = 'all') -> DataFrame:
        """
        Output the 'raw' time-signal (or with the sensitivities applied, depending on the order of operation).

        If visualise set to True, then will make a plot with optional parameters of plot_single_df.

        :param which: List of strings or singular string of channels to provide.
            Either from the {self.channel_float_type_dct}, or 'all',
            which simply export all channels with the data type float.

        :return: DataFrame, with as index a time array made from the sample number divided with the sampling frequency.
        """
        channel_lst = self._which(which=which)  # Get channels to process.
        df_out = self.df_data.loc[:, channel_lst].copy(True)  # Get a copy of the data DataFrame.
        # Set index of the DataFrame.
        df_out.index = np.arange(df_out.index.size) / self.fs
        df_out.index.name = self.axis_format % {'var': self.time_var_and_unit[0], 'unit': self.time_var_and_unit[1]}
        return df_out  # Return only channels.

    # @plot_f.all_plotting_decorator(plot_func=functools.partial(plot_f.plot_single_df, y_str=r"$\Phi_{EE}$, V$^2$/Hz",
    #                                                            x_scale='log', y_scale='log'))
    @plot_f.all_plotting_decorator(plot_func=plot_f.plot_single_df)
    def psd(self, which: Union[str, List[Tuple[str, str]]] = 'all', **kwargs) -> DataFrame:
        """
        Output the power spectral densities of the chosen channels.

        If visualise set to True, then will make a plot with optional parameters of plot_single_df.

        :param which: List of strings or singular string of channels to provide.
            Either from the {self.channel_float_type_dct}, or 'all',
            which simply export all channels with the data type float.
        :param kwargs: Welch settings. Default: 'default_proc_settings' property of object.

        :return: DataFrame.
        """
        channel_lst = self._which(which=which)  # Get channels to process.
        kwargs_new = self._default_settings(kwargs)
        # Compute spectrum for all the channels.
        f_arr, psd_arr = proc_f.f_psd(data=self.df_data.loc[:, channel_lst].dropna(), **kwargs_new)
        df_psd = pd.DataFrame(data=psd_arr, index=f_arr,
                              columns=pd.MultiIndex.from_tuples(channel_lst, names=self.df_data.columns.names))
        df_psd.index.name = self.axis_format % {'var': self.frequency_var_and_unit[0],
                                                'unit': self.frequency_var_and_unit[1]}
        return df_psd

    # @plot_f.all_plotting_decorator(plot_func=functools.partial(plot_f.plot_single_df, y_str=r"$\Phi_{EE}$, dB/Hz",
    #                                                            x_scale='log'))
    @plot_f.all_plotting_decorator(plot_func=plot_f.plot_single_df)
    def psd_norm(self, which: Union[str, List[Tuple[str, str]]] = 'all', p_ref: float = 2E-5) -> DataFrame:
        """
        Output the normalised PSD, with reference pressure, [dB/Hz] of the chosen channels.

        If visualise set to True, then will make a plot with optional parameters of plot_single_df.

        :param which: List of strings or singular string of channels to provide.
            Either from the {self.channel_float_type_dct}, or 'all',
            which simply export all channels with the data type float.
        :param p_ref: Reference pressure used to compute the spectrum.

        :return: DataFrame.
        """
        df_psd = self.psd(which=which)  # Get non-normalised PSD.
        return proc_f.f_spectra(data_psd=df_psd, p_ref=p_ref)  # Normalise.

    # @plot_f.all_plotting_decorator(plot_func=functools.partial(plot_f.plot_single_df, y_str=r"$\gamma^2_{xy}$, -",
    #                                                            x_scale='log'))
    @plot_f.all_plotting_decorator(plot_func=plot_f.plot_single_df)
    def cross_coherence(self, in_channel: List[Tuple[str, str]], out_channel: List[Tuple[str, str]], **kwargs) -> \
            DataFrame:
        """
        Output the cross-coherence between the elements of input (x) and output (y) lists.
        So [('group', 'channel')] or [('group_a', 'channel_c'), ('group_b', 'channel_d')] as input/output channels.
        Will compute the cross-coherence the following way:
        in_lst=[a, b, c]. out_lst=[d, e, f]. output=[r_ad, r_be, r_cf]

        If visualise set to True, then will make a plot with optional parameters of plot_single_df.

        :param in_channel: List (!) of strings of channels from the {self.channel_float_type_dct}
            that serves as the x signal.
        :param out_channel: List (!) of strings of channels from the {self.channel_float_type_dct}
            that serves as the y signal.
        :param kwargs: Scipy coherence function settings. Default: 'default_proc_settings' property of object.

        :return: DataFrame.
        """
        # Compute coherence between the chosen channels.
        # data_in_out = self.df_data.loc[:, np.concatenate(([in_channel], [out_channel])).flatten()].dropna()
        data_in_out = self.df_data.loc[:, in_channel + out_channel].dropna()  # Keep non-nan rows.
        data_in_out = data_in_out.loc[:, ~data_in_out.columns.duplicated()]  # Remove duplicate columns.

        kwargs_new = self._default_settings(kwargs)

        # Compute coherence.
        f_arr, coh_arr = proc_f.f_coherence(x=data_in_out.loc[:, in_channel], y=data_in_out.loc[:, out_channel],
                                            **kwargs_new)
        # Column names for the output DataFrame.
        process_columns = [self._tuple_in_out_to_key(in_tuple=in_channel[i], out_tuple=out_channel[i])
                           for i in range(len(in_channel))]
        df_coh = pd.DataFrame(data=coh_arr, index=f_arr, columns=process_columns)  # Make DataFrame.
        # Write index name for DataFrame.
        df_coh.index.name = self.axis_format % {'var': self.frequency_var_and_unit[0],
                                                'unit': self.frequency_var_and_unit[1]}
        return df_coh

    @plot_f.all_plotting_decorator(plot_func=plot_f.plot_transfer_function_df)
    def transfer_function(self, in_channel: List[Tuple[str, str]], out_channel: List[Tuple[str, str]],
                          set_property: bool = True, **kwargs) -> DataFrame:
        """
        Output the complex-valued transfer function between the elements of input (x) and output (y) lists.
        Channel inputs provided as lists, so:
        [('group', 'channel')] or [('group_a', 'channel_c'), ('group_b', 'channel_d')] as input/output channels.
        Will compute the TF the following way:
        in_lst=[a, b, c]. out_lst=[d, e, f]. output=[TF_ad, TF_be, TF_cf]

        If visualise set to True, then will make a plot with optional parameters of plot_transfer_function_df.

        :param in_channel: List (!) of strings of channels from the {self.channel_float_type_dct}
            that serves as the x signal.
        :param out_channel: List (!) of strings of channels from the {self.channel_float_type_dct}
            that serves as the y signal.
        :param set_property: If True, sets the computed DataFrame as the {self.df_tf} variable of the object.
            Default: True.
        :param kwargs: Cross-power spectral density and Welch function settings.
            Default: 'default_proc_settings' property of object.

        :return: DataFrame.
        """
        # Compute transfer function between the chosen channels.
        data_in_out = self.df_data.loc[:, in_channel + out_channel].dropna()  # Keep non-nan rows.
        data_in_out = data_in_out.loc[:, ~data_in_out.columns.duplicated()]  # Remove duplicate columns.

        # Check if specific processing functions specified, otherwise use defaults.
        kwargs_new = self._default_settings(kwargs)
        kwargs_new = var_kwargs(var_str='return_onesided', default_val=True, kwargs=kwargs_new)

        # Compute transfer function.
        f_arr, tf_arr = proc_f.tf_estimate(x=data_in_out.loc[:, in_channel], y=data_in_out.loc[:, out_channel],
                                           **kwargs_new)
        # Column names for the output DataFrame.
        process_columns = [self._tuple_in_out_to_key(in_tuple=in_channel[i], out_tuple=out_channel[i])
                           for i in range(len(in_channel))]
        df_tf = pd.DataFrame(data=tf_arr, index=f_arr, columns=process_columns)
        df_tf.index.name = self.axis_format % {'var': self.frequency_var_and_unit[0],
                                               'unit': self.frequency_var_and_unit[1]}
        if set_property:  # Set TF DataFrame as property of object.
            self.df_tf = df_tf
        return df_tf.copy(True)

    @plot_f.all_plotting_decorator(plot_func=plot_f.plot_transfer_function_df)
    def add_transfer_function_step(self, df_tf_new: DataFrame,
                                   dct_old_tf_to_new_tf_channels: Optional[Dict[str, str]] = None, axis: int = 0) -> \
            DataFrame:
        """
        Combine a provided new transfer function DataFrame 'df_tf_new' with the object property transfer function
        DataFrame 'object.df_tf'. Return the combined total transfer function DataFrame, and assign to the object
        property transfer function DataFrame 'object.df_tf'. If object df_tf is None, i.e. not assigned, then assign
        df_tf_new to df_tf.

        If visualise set to True, then will make a plot with optional parameters of plot_transfer_function_df.

        :param df_tf_new: New transfer function DataFrame to combine with current object df_tf.
        :param dct_old_tf_to_new_tf_channels: Dictionary linking column names of both transfer function DataFrames.
            {..., df_tf column: df_tf_new column, ...}.
            Default: None; Assume that all columns in df_tf are present in df_tf_new, and have the same names.
        :param axis: DataFrame axis of the frequency axis.

        :return: Total, combined, transfer function DataFrame.
        """
        if self.df_tf is None:  # If no TF DataFrame is assigned to this object yet, assign new one.
            self.df_tf = df_tf_new
        else:
            # If no dictionary provided to link column names of both TF DataFrames.
            if dct_old_tf_to_new_tf_channels is None:  # Same names assumed, so no translation dictionary.
                dct_old_tf_to_new_tf_channels = dict(zip(self.df_tf.columns, self.df_tf.columns))

            # Frequency arrays used for interpolation.
            f_current_arr = self.df_tf.index.to_numpy(float)  # Frequency array of current TF DataFrame.
            f_new_arr = df_tf_new.index.to_numpy(float)  # Frequency array of new TF DataFrame.

            for channel_i in self.df_tf.columns:
                tf_old_i = self.df_tf.loc[:, channel_i].to_numpy(complex)  # Current TF channel.
                # New TF channel.
                tf_new_i = df_tf_new.loc[:, dct_old_tf_to_new_tf_channels[channel_i]].to_numpy(complex)
                # Interpolate new TF channel on old TF channel frequencies.
                tf_new_int = PchipInterpolator(f_new_arr, tf_new_i, axis=axis)(f_current_arr)
                self.df_tf.loc[:, channel_i] = tf_old_i * tf_new_int  # Multiply both and assign to (old) df_tf channel.
        return self.df_tf.copy(True)  # Return combined TF DataFrame.

    # HELPER
    def _which(self, which: Union[str, List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
        """
        Used to check for 'special' inputs for which input variable of methods of the object.

        :param which: List of strings or singular string of channels to provide.
            Either from the {self.channel_float_type_dct}, or 'all',
            which simply export all channels with the data type float.

        :return: The correct list of channel strings.
        """
        if which == 'all':  # All channels.
            return self.channel_float_type_dct  # self.data_channel_names
        else:  # Could do further type checking.
            return which

    def _label_from_tuple(self, channel_tuple: Tuple[str, str]) -> str:
        """
        Converts the tuple column name of the channel to a single string following the {self.label_format}
        provided upon creation of the object.

        :param channel_tuple: Tuple of the calibration step name and the channel name.

        :return: String.
        """
        return self.label_format % {'file': channel_tuple[0], 'channel': channel_tuple[1]}

    def _tuple_in_out_to_key(self, in_tuple: Tuple[str, str], out_tuple: Tuple[str, str]) -> str:
        """
        Converts tuples of column names of two channels to a single string following the {self.key_process}
        provided upon creation of the object.
        Used to assign a column name of DataFrames created by functions that use an x and y signal as input.

       :param in_tuple: Tuple of the calibration step name and the channel name that serves as input signal.
       :param out_tuple: Tuple of the calibration step name and the channel name that serves as output signal.

       :return: String.
       """
        return self.key_process % {'file_in': in_tuple[0], 'channel_in': in_tuple[1],
                                   'file_out': out_tuple[0], 'channel_out': out_tuple[1]}

    def _default_settings(self, kwargs: dict) -> dict:
        """
        Define default processing settings for functions such as Welch etc. If a new user-defined parameter is provided
        in kwargs, it overwrites the parameter in 'self.default_proc_settings'.

        :param kwargs: Parameter dictionary provided by the user as input for processing functions.

        :return: Updated processing function parameter dictionary.
        """
        temp_dct = copy.deepcopy(self.default_proc_settings)  # Copy current defaults.
        for key in kwargs:  # Go through user-provided parameters, and either assign or overwrite.
            temp_dct[key] = kwargs[key]
        return temp_dct  # Return updated parameters.


def multi_step_calibration(calibration_file_path_list: Sequence[str],
                           calibration_input_channel_list: Sequence[List[Tuple[str, str]]],
                           calibration_output_channel_list: Sequence[List[Tuple[str, str]]]) -> \
        Tuple['PressureAcquisition', DataFrame]:
    """
    Simplified function to combine multiple empirical frequency-domain calibration TDMS files into a final TF and a
    calibration 'PressureAcquisition' object.

    Assumes that each input and output list for each file is the same length,
    and that the order of the keys is the same.
    The first input and output are multiplied with that of the next step etc.

    :param calibration_file_path_list: Sequence of file paths of all calibration steps. Example: [..., file_i, ...].
    :param calibration_input_channel_list: Sequence that contains sequence of input keys of the TF for all files in
        calibration_file_path_list. Example: [..., [input_0, ..., input_j, ..., input_m]_for_file_i, ...].
    :param calibration_output_channel_list: Sequence that contains sequence of output keys of the TF for all files in
        calibration_file_path_list. Example: [..., [output_0, ..., output_j, ..., output_m]_for_file_i, ...].

    :return: Calibration object 'PressureAcquisition' based on first element of the input list,
        combined transfer function DataFrame.
    """
    # Check if all inputs have the same length.
    assert len(calibration_file_path_list) == len(calibration_input_channel_list)
    assert len(calibration_input_channel_list) == len(calibration_output_channel_list)

    # Create base object and TF for first file.
    obj_base = PressureAcquisition(file_path=calibration_file_path_list[0])
    df_tf_total = obj_base.transfer_function(in_channel=calibration_input_channel_list[0],
                                             out_channel=calibration_output_channel_list[0], set_property=True)
    df_tf_base_columns = df_tf_total.columns  # All column names of TF DataFrame.

    # Go through all subsequent calibration files.
    for i, file_i in enumerate(calibration_file_path_list[1:]):
        j = i + 1  # The 0th element is the base defined before the for-loop, so add 1.
        # Create new calibration object, and compute the transfer function of that calibration step alone.
        obj_i = PressureAcquisition(file_path=file_i)
        df_tf_i = obj_i.transfer_function(in_channel=calibration_input_channel_list[j],
                                          out_channel=calibration_output_channel_list[j], set_property=True)
        # Get dictionary to link both TF DataFrames.
        df_tf_i_columns = df_tf_i.columns
        dct_tf_keys = dict(zip(df_tf_base_columns, df_tf_i_columns))
        # Combined TFs together.
        df_tf_total = obj_base.add_transfer_function_step(
            df_tf_new=df_tf_i, axis=0, dct_old_tf_to_new_tf_channels=dct_tf_keys)

    return obj_base, df_tf_total


def sensitivity_calculation(file_in: str, cal_spl: float = 94, f_cal: float = 1E3,
                            channel: Tuple[str, str] = ('Data', 'MIC'), fs: float = 51200, delta_f: float = 50,
                            debug: bool = False, pre_amp: float = 1, p_ref: float = 2E-5) -> float:
    """
    Compute microphone sensitivity from pistonphone calibration TDMS file,
    using the root-mean-square value of the band-pass filtered microphone voltage data.

    :param file_in: TDMS file containing pistonphone calibration data.
    :param cal_spl: Pistonphone calibration amplitude, dB.
    :param f_cal: Pistonphone tone frequency, Hz.
    :param channel: Tuple containing the data 'group' and 'channel' in which the data is saved in the tdms file,
        e.g., tuple = (group, channel) = ('Data', 'MIC').
    :param fs: Sampling frequency of data, Hz.
    :param delta_f: Half-width of band-pass frequency range centered on pistonphone frequency,
        i.e., [f_cal-delta_f, f_cal+delta_f], Hz.
    :param debug: Print the mean and variance of both minima and maxima, as well as their combined mean and variance.
    :param pre_amp: Pre-amplification of the time-series data. Used for testing.
    :param p_ref: Reference pressure used to define the dB of cal_spl.

    :return: Microphone sensitivity estimation.
    """
    pressure_rms = p_ref * 10**(cal_spl/20.)  # Convert from dB to Pa.

    with TdmsFile.read(file_in) as f:
        data_arr = f[channel[0]][channel[1]][:]  # Read microphone voltage data, V.
    data_arr *= 1E3  # Convert microphone voltage units, V -> mV.
    data_arr *= pre_amp  # Pre-amplification for data.

    # Band-pass raw voltage data around the pistonphone frequency +/- delta_f.
    flt = butter(N=10, Wn=(f_cal - delta_f, f_cal + delta_f), fs=fs, btype='bandpass', output='sos')
    data_filtered_arr = sosfilt(flt, np.copy(data_arr), axis=0)  # Filtered voltage data.
    voltage_rms = np.std(data_filtered_arr, axis=0)  # Root-mean-square of filtered voltage data, mV.

    sensitivity = voltage_rms / pressure_rms  # Sensitivity, mV/Pa.
    if debug:  # Optional print of sensitivity in console.
        print(f'Sensitivity estimated to: {sensitivity} mV/Pa')
    return sensitivity
