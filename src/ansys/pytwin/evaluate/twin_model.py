import json
import os
import time

import numpy as np
import pandas as pd
from pytwin.evaluate.model import Model
from pytwin.evaluate.saved_state_registry import SavedState, SavedStateRegistry
from pytwin.settings import PyTwinLogLevel, get_pytwin_log_level, pytwin_logging_is_enabled
from pytwin.twin_runtime.log_level import LogLevel
from pytwin.twin_runtime.twin_runtime_core import TwinRuntime


class TwinModel(Model):
    """
    The public class to evaluate a twin model given a twin model file (with .twin extension) created with Ansys Twin
    Builder. After being initialized, a twin model object can be evaluated with two modes (step-by-step or batch mode)
    to make predictions. Parametric workflows are also supported.

    Parameters
    ----------
    model_filepath : str
        File path to the twin model (with .twin) extension.

    Examples
    --------
    Create a twin model object given the file path to the twin model file. Initialize two parameters and two inputs of
    the twin model. Then evaluate two steps and retrieve results in a dictionary.

    >>> from pytwin import TwinModel
    >>>
    >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
    >>>
    >>> twin_model.initialize_evaluation(parameters={'param1': 1., 'param2': 2.}, inputs={'input1': 1., 'input2': 2.})
    >>> outputs = dict()
    >>> outputs['Time'] = [twin_model.evaluation_time]
    >>> outputs['output1'] = [twin_model.outputs['output1']]
    >>> outputs['output2'] = [twin_model.outputs['output2']]
    >>>
    >>> twin_model.evaluate_step_by_step(step_size=0.1, inputs={'input1': 10., 'input2': 20.})
    >>> outputs['Time'].append(twin_model.evaluation_time)
    >>> outputs['output1'].append(twin_model.outputs['output1'])
    >>> outputs['output2'].append(twin_model.outputs['output2'])
    >>>
    >>> twin_model.evaluate_step_by_step(step_size=0.1, inputs={'input1': 20., 'input2': 30.})
    >>> outputs['Time'].append(twin_model.evaluation_time)
    >>> outputs['output1'].append(twin_model.outputs['output1'])
    >>> outputs['output2'].append(twin_model.outputs['output2'])
    """

    TBROM_FILENAME_TIME_FORMAT = ".6f"
    TBROM_FOLDER_NAME = "ROM_files"
    TBROM_IMAGE_EXT = ".png"
    TBROM_VIEWS_KEY = "views"
    TBROM_SNAPSHOT_FILE_PREFIX = "snapshot_"
    TBROM_SNAPSHOT_EXT = ".bin"

    def __init__(self, model_filepath: str):
        super().__init__()
        self._evaluation_time = None
        self._initialization_time = None
        self._instantiation_time = None
        self._inputs = None
        self._model_filepath = None
        self._outputs = None
        self._parameters = None
        self._ss_registry = None
        self._twin_runtime = None
        self._tbrom_info = None

        if self._check_model_filepath_is_valid(model_filepath):
            self._model_filepath = model_filepath
        self._instantiate_twin_model()

    def __del__(self):
        """
        Close twin runtime when object is garbage collected.
        """
        if self._twin_runtime is not None:
            self._twin_runtime.twin_close()

    def _check_model_filepath_is_valid(self, model_filepath):
        """
        Check provided twin model filepath is valid. Raise a TwinModelError if not.
        """
        if model_filepath is None:
            msg = f"TwinModel cannot be called with {model_filepath} as model_filepath!"
            msg += "\nPlease provide valid filepath to initialize the TwinModel object."
            raise self._raise_error(msg)
        if not os.path.exists(model_filepath):
            msg = f"Provided twin model filepath: {model_filepath} does not exist!"
            msg += "\nPlease provide existing filepath to initialize the TwinModel object."
            raise self._raise_error(msg)
        return True

    def _create_dataframe_inputs(self, inputs_df: pd.DataFrame):
        """
        Create a dataframe inputs that satisfies the conventions of the runtime SDK batch mode evaluation, that are:
        (1) 'Time' as first column (2) one column per twin model input (3) columns order is the same as twin model
        input names list return by SDK.

        If an input is not found in the given inputs_df, then initialization value is used to keep associated input
        constant over Time.
        """
        self._warns_if_input_key_not_found(inputs_df.to_dict())
        _inputs_df = pd.DataFrame()
        _inputs_df["Time"] = inputs_df["Time"]
        for name, value in self._inputs.items():
            if name in inputs_df:
                _inputs_df[name] = inputs_df[name]
            else:
                _inputs_df[name] = np.full(shape=(_inputs_df.shape[0], 1), fill_value=value)
        return _inputs_df

    @staticmethod
    def _get_runtime_log_level():
        pytwin_level = get_pytwin_log_level()
        if pytwin_level == PyTwinLogLevel.PYTWIN_LOG_DEBUG:
            return LogLevel.TWIN_LOG_ALL
        if pytwin_level == PyTwinLogLevel.PYTWIN_LOG_INFO:
            return LogLevel.TWIN_LOG_ALL
        if pytwin_level == PyTwinLogLevel.PYTWIN_LOG_WARNING:
            return LogLevel.TWIN_LOG_WARNING
        if pytwin_level == PyTwinLogLevel.PYTWIN_LOG_ERROR:
            return LogLevel.TWIN_LOG_ERROR
        if pytwin_level == PyTwinLogLevel.PYTWIN_LOG_CRITICAL:
            return LogLevel.TWIN_LOG_FATAL
        if not pytwin_logging_is_enabled():
            return LogLevel.TWIN_NO_LOG

    def _initialize_evaluation(self, parameters: dict = None, inputs: dict = None):
        """
        Initialize the twin model evaluation with dictionaries:
        (1) Initialize parameters and/or inputs values to their start values (default value found in the twin file),
        (2) Update parameters and/or inputs values with provided dictionaries. Ignore value whose name is not found
        into the list of parameters/inputs names of the twin model (value is kept to default one in that case).
        (3) Initialize evaluation time to 0.
        (4) Save universal time (time since epoch) at which the method is called
        (5) Evaluation twin model at time instant 0. and store its results into outputs dictionary.
        Twin runtime is reset in case of already initialized twin model.
        """
        if self._twin_runtime is None:
            self._raise_error("Twin model has not been successfully instantiated!")

        if self._twin_runtime.is_model_initialized:
            self._twin_runtime.twin_reset()

        self._initialize_parameters_with_start_values()
        if parameters is not None:
            self._update_parameters(parameters)

        self._initialize_inputs_with_start_values()
        if inputs is not None:
            self._update_inputs(inputs)

        self._warns_if_parameter_key_not_found(parameters)
        self._warns_if_input_key_not_found(inputs)

        self._evaluation_time = 0.0
        self._initialization_time = time.time()

        try:
            tbrom_info = self._twin_runtime.twin_get_visualization_resources()
            if tbrom_info:
                self._log_key += "WithTBROM : {}".format(tbrom_info)
                self._tbrom_info = tbrom_info
                directory_path = os.path.join(self.model_dir, self.TBROM_FOLDER_NAME)
                for model_name, data in tbrom_info.items():
                    self._twin_runtime.twin_set_rom_image_directory(model_name, directory_path)

            self._twin_runtime.twin_initialize()
        except Exception as e:
            msg = f"Something went wrong during model initialization!"
            msg += f"\n{str(e)}"
            msg += f"\nYou will find more details in model log (see {self.model_log} file)"
            self._raise_error(msg)

        self._update_outputs()

    def _initialize_inputs_with_start_values(self):
        """
        Initialize inputs dictionary {name:value} with starting input values found in twin model.
        """
        self._inputs = dict()
        for name in self._twin_runtime.twin_get_input_names():
            self._inputs[name] = self._twin_runtime.twin_get_var_start(var_name=name)

    def _initialize_parameters_with_start_values(self):
        """
        Initialize parameters dictionary {name:value} with starting parameter values found in twin model.
        """
        self._parameters = dict()
        for name in self._twin_runtime.twin_get_param_names():
            if "solver." not in name:
                self._parameters[name] = self._twin_runtime.twin_get_var_start(var_name=name)

    def _initialize_outputs_with_none_values(self):
        """
        Initialize outputs dictionary {name:value} with None values.
        """
        output_names = self._twin_runtime.twin_get_output_names()
        output_values = [None] * len(output_names)
        self._outputs = dict(zip(output_names, output_values))

    def _instantiate_twin_model(self):
        """
        Connect TwinModel with TwinRuntime and load twin model.
        """
        try:
            # Create temp dir if needed
            if not os.path.exists(self.model_temp):
                os.mkdir(self.model_temp)

            # Instantiate twin runtime
            self._twin_runtime = TwinRuntime(
                model_path=self._model_filepath,
                load_model=True,
                log_path=self.model_log,
                log_level=self._get_runtime_log_level(),
            )
            self._twin_runtime.twin_instantiate()

            # Create subfolder
            self._model_name = self._twin_runtime.twin_get_model_name()
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
            os.link(self.model_log, self.model_log_link)

            # Update TwinModel variables
            self._instantiation_time = time.time()
            self._initialize_inputs_with_start_values()
            self._initialize_parameters_with_start_values()
            self._initialize_outputs_with_none_values()

        except Exception as e:
            msg = "Twin model failed during instantiation!"
            msg += f"\n{str(e)}"
            self._raise_error(msg)

    def _raise_model_error(self, msg):
        """
        Raise a TwinModelError with formatted message.
        """
        raise TwinModelError(msg)

    def _read_eval_init_config(self, json_filepath: str):
        """
        Deserialize a json object into a dictionary that is used to store twin model inputs and parameters values
        to be passed to the internal evaluation initialization method.
        """
        if not os.path.exists(json_filepath):
            msg = "Provided config filepath (for evaluation initialization) does not exist!"
            msg += f"\nProvided filepath is: {json_filepath}"
            msg += "\nPlease provide an existing filepath to initialize the twin model evaluation."
            raise self._raise_error(msg)
        try:
            with open(json_filepath) as file:
                cfg = json.load(file)
                return cfg
        except Exception as e:
            msg = "Something went wrong while reading config file!"
            msg += f"n{str(e)}"
            self._raise_error(msg)

    def _update_inputs(self, inputs: dict):
        """Update input values with given dictionary."""
        for name, value in inputs.items():
            if name in self._inputs:
                self._inputs[name] = value
                self._twin_runtime.twin_set_input_by_name(input_name=name, value=value)

    def _update_outputs(self):
        """Update output values with twin model results at current evaluation time."""
        self._outputs = dict(zip(self._twin_runtime.twin_get_output_names(), self._twin_runtime.twin_get_outputs()))

    def _update_parameters(self, parameters: dict):
        """Update parameters values with given dictionary."""
        for name, value in parameters.items():
            if name in self._parameters:
                self._parameters[name] = value
                self._twin_runtime.twin_set_param_by_name(param_name=name, value=value)

    def _tbrom_resource_directory(self, rom_name: str):
        """
        Return the path of the resource directory associated with rom_name.
        """
        if self._twin_runtime is None:
            self._raise_error("Twin model has not been successfully instantiated!")

        if not self.evaluation_is_initialized:
            self._raise_error("Twin model evaluation has not been initialized! Please initialize evaluation.")

        if self.tbrom_info is None:
            self._raise_error("Twin model does not include any TBROM!")

        if rom_name not in self.tbrom_info:
            self._raise_error(f"Twin model does not include any TBROM named {rom_name}!")

        return self._twin_runtime.twin_get_rom_resource_directory(rom_name)

    def _warns_if_input_key_not_found(self, inputs: dict):
        if inputs is not None:
            for _input in inputs:
                if _input not in self.inputs:
                    if _input != "Time":
                        msg = f"Provided input ({_input}) has not been found in model inputs!"
                        self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_WARNING)

    def _warns_if_parameter_key_not_found(self, parameters: dict):
        if parameters is not None:
            for param in parameters:
                if param not in self.parameters:
                    msg = f"Provided parameter ({param}) has not been found in model parameters!"
                    self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_WARNING)

    @property
    def evaluation_is_initialized(self):
        """Return true if evaluation has been initialized."""
        if self._twin_runtime is None:
            self._raise_error("Twin model has not been successfully instantiated!")
        return self._twin_runtime.is_model_initialized

    @property
    def evaluation_time(self):
        """
        Return a floating point number that is the current twin model evaluation time (in second).
        """
        return self._evaluation_time

    @property
    def inputs(self):
        """
        Return a dictionary with input values at current evaluation time.
        """
        return self._inputs

    @property
    def initialization_time(self):
        """
        Return a floating point number that is the time at which the twin model has been initialized.
        It is given in seconds since the epoch."""
        return self._initialization_time

    @property
    def instantiation_time(self):
        """
        Return a floating point number that is the time at which the twin model has been instantiated.
        It is given in seconds since the epoch."""
        return self._instantiation_time

    @property
    def outputs(self):
        """
        Return a dictionary with output values at current evaluation time.
        """
        return self._outputs

    @property
    def parameters(self):
        """
        Return a dictionary with parameter values at current evaluation time.
        """
        return self._parameters

    @property
    def model_filepath(self):
        """
        Return a string that is the twin model filepath that has been verified and loaded.
        Return None if model filepath is not valid.
        """
        return self._model_filepath

    @property
    def tbrom_info(self):
        """
        Return a dictionary with TBROM model names included in the Twin and their corresponding 3D visualization
        capabilities available (e.g. snapshots, and optionnally images generation). If no TBROM is included in the
        Twin, it returns None
        """
        return self._tbrom_info

    @property
    def tbrom_names(self):
        """
        Return available TBROM model names. If no TBROM is included in the Twin, it returns an empty list.
        """
        if self._tbrom_info is not None:
            return list(self._tbrom_info)
        return []

    @property
    def tbrom_directory_path(self):
        """
        Return TBROM directory path. This is the directory where temporary TBROM files are stored.
        """
        return os.path.join(self.model_dir, self.TBROM_FOLDER_NAME)

    def initialize_evaluation(self, parameters: dict = None, inputs: dict = None, json_config_filepath: str = None):
        """
        Initialize the twin model evaluation with: (1) a dictionary of parameters values and/or inputs (start) values
        OR (2) a json configuration file (see example below).

        Option (2) overrides option (1).

        If no inputs is given (rather in the arguments or in the config file), then inputs are reset to their default
        values when calling this method. Same thing happens for parameters.

        Parameters and inputs that are not found into the provided dictionaries or config file, keep their default
        values (i.e. the start value of the twin model).

        Evaluation time is reset to zero after calling this method and Initialization time is updated.

        This method must be called:
        (1) before to evaluate the twin model,
        (2) if you want to update parameters values between multiple twin evaluations
        (in such case the twin model is reset).

        Parameters
        ----------
        parameters : dict, optional
            The parameter values (i.e. {"name": value}) to be used for the next evaluation.
        inputs : dict, optional
            The input values (i.e. {"name": value}) to be used for twin model initialization.
        json_config_filepath : str, optional
            A file path to a json config file (with .json extension) to be used to initialize the evaluation

        Examples
        --------
        >>> import json
        >>> from pytwin import TwinModel
        >>>
        >>> config = {"version": "0.1.0", "model": {"inputs": {"input-name-1": 1., "input-name-2": 2.}, \
        >>> "parameters": {"param-name-1": 1.,"param-name-2": 2.}}}
        >>> with open('path_to_your_config.json', 'w') as f:
        >>>     f.write(json.dumps(config))
        >>>
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation(json_config_filepath='path_to_your_config.json')
        >>> outputs = twin_model.outputs
        """
        self._log_key = "InitializeEvaluation"

        if json_config_filepath is None:
            self._log_key += "WithDictionary"
            self._initialize_evaluation(parameters=parameters, inputs=inputs)
        else:
            self._log_key += "WithConfigFile"
            cfg = self._read_eval_init_config(json_config_filepath)
            _parameters = None
            _inputs = None
            if "model" in cfg:
                if "parameters" in cfg["model"]:
                    _parameters = cfg["model"]["parameters"]
                if "inputs" in cfg["model"]:
                    _inputs = cfg["model"]["inputs"]
            self._initialize_evaluation(parameters=_parameters, inputs=_inputs)

    def evaluate_step_by_step(self, step_size: float, inputs: dict = None):
        """
        Evaluate the twin model at time instant t + step_size given inputs at time instant t. Return list of
        outputs values at time instant t + step_size (ordered by output_names).

        Twin model evaluation must have been initialized before calling this method
        (see `initialize_evaluation` method).

        Parameters
        ----------
        step_size : float
            The step size (in second) to reach next time step. It must be strictly positive.

        inputs : dict (optional)
            The input values (i.e. {"name": value}) at time instant t. An input is not updated if associated key is
            not found in twin model input_names property. Other inputs keep current value if not provided.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation()
        >>> twin_model.evaluate_step_by_step(step_size=0.1, inputs={'input1': 1., 'input2': 2.})
        >>> results = {'Time': twin_model.evaluation_time, 'Outputs': twin_model.outputs}
        """
        self._log_key = "EvaluateStepByStep"

        if self._twin_runtime is None:
            self._raise_error("Twin model has not been successfully instantiated!")

        if not self.evaluation_is_initialized:
            self._raise_error("Twin model evaluation has not been initialized! Please initialize evaluation.")

        if step_size <= 0.0:
            msg = f"Step size must be strictly bigger than zero ({step_size} was provided)!"
            self._raise_error(msg)

        self._warns_if_input_key_not_found(inputs)
        if inputs is not None:
            self._update_inputs(inputs)

        try:
            self._twin_runtime.twin_simulate(self._evaluation_time + step_size)
            self._evaluation_time += step_size
            self._update_outputs()
        except Exception as e:
            msg = f"Something went wrong during evaluation at time step {self._evaluation_time}:"
            msg += f"\n{str(e)}"
            msg += f"\nPlease reinitialize the model evaluation and restart evaluation."
            msg += f"\nYou will find more details in model log (see {self.model_log} file)"
            self._raise_error(msg)

    def evaluate_batch(self, inputs_df: pd.DataFrame):
        """
        Evaluate the twin model with historical input values given with a data frame.

        Parameters
        ----------
        inputs_df: pandas.DataFrame
            The historical input values stored in a pandas dataframe. It must have a 'Time' column and all twin
            model inputs history you want to simulate (one input per column),starting at time instant t=0.(s). If a
            twin model input is not found in the dataframe columns then this input is kept constant to its
            initialization value. The column header must match with a twin model input name.

        Returns
        -------
        output_df: pandas.DataFrame
            The twin output values associated to the input values, stored in a pandas.DataFrame.

        Raises
        ------
        TwinModelError:
            if initialize_evaluation(...) has not been called before, if there is no 'Time' column in the inputs
            dataframe, if there is no time instant t=0.s in the inputs dataframe.

        Examples
        --------
        >>> import pandas as pd
        >>> from pytwin import TwinModel
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> inputs_df = pd.DataFrame({'Time': [0., 1., 2.], 'input1': [1., 2., 3.], 'input2': [1., 2., 3.]})
        >>> twin_model.initialize_evaluation(inputs={'input1': 1., 'input2': 1.})
        >>> outputs_df = twin_model.evaluate_batch(inputs_df=inputs_df)
        """
        self._log_key = "EvaluateBatch"

        if self._twin_runtime is None:
            self._raise_error("Twin model has not been successfully instantiated!")

        if not self.evaluation_is_initialized:
            self._raise_error("Twin model evaluation has not been initialized! Please initialize evaluation.")

        if "Time" not in inputs_df:
            msg = "Given inputs dataframe has no 'Time' column!"
            msg += f"\nExisting column labels are :{[s for s in inputs_df.columns]}"
            msg += f"\nPlease provide a dataframe with a 'Time' column to use batch mode evaluation."
            self._raise_error(msg)

        t0 = inputs_df["Time"][0]
        if not np.isclose(t0, 0.0, atol=np.spacing(0.0)):
            msg = "Given inputs dataframe has no time instant t=0.s!"
            msg += f" (first provided time instant is : {t0})."
            msg += "\nPlease provide inputs at time instant t=0.s"
            self._raise_error(msg)

        # Ensure SDK conventions are fulfilled
        _inputs_df = self._create_dataframe_inputs(inputs_df)
        _output_col_names = ["Time"] + list(self._outputs.keys())

        try:
            return self._twin_runtime.twin_simulate_batch_mode(
                input_df=_inputs_df, output_column_names=_output_col_names
            )
        except Exception as e:
            msg = f"Something went wrong during batch evaluation:"
            msg += f"\n{str(e)}"
            msg += f"\nPlease reinitialize the model evaluation and restart evaluation."
            msg += f"\nYou will find more details in model log (see {self.model_log} file)"
            self._raise_error(msg)

    def get_available_view_names(self, rom_name: str):
        """
        Get a list of available view names for a given Reduced Order Model (ROM) available in the TwinModel.

        Parameters
        ----------
        rom_name : str
            This is the name of a ROM model that is available in the TwinModel. See TwinModel.tbrom_names property to
            get a list of available ROM model.

        Raises
        ------
        TwinModelError:
            It raises an error if TwinModel has not been initialized.
            It raises an error if TwinModel does not include any TBROM.
            It raises an error if rom_name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.initialize_evaluation()
        >>> model.get_available_view_names(rom_name=model.tbrom_names[0])
        """
        self._log_key = "GetImageViewNames"

        if not self.evaluation_is_initialized:
            msg = "TwinModel has not been initialized! "
            msg += "Please initialize evaluation before to call this method!"
            self._raise_error(msg)

        if self.tbrom_info is None:
            self._raise_error("Twin model does not include any TBROM!")

        if rom_name not in self.tbrom_names:
            msg = f"The provided rom_name {rom_name} has not been found in the available TBROM names. "
            msg += f"Please call this method with a valid TBROM name."
            msg += f"\n Available TBROM name are: {self.tbrom_names}"
            self._raise_error(msg)

        view_names = list(self._tbrom_info[rom_name][self.TBROM_VIEWS_KEY])

        if len(view_names) == 0:
            msg = f"No views are available for given rom_name: {rom_name}."
            self._log_message(msg, level=PyTwinLogLevel.PYTWIN_LOG_WARNING)

        return view_names

    def get_image_filepath(self, rom_name: str, view_name: str, evaluation_time: float = 0.0):
        """
        Get the image file associated to a Reduced Order Model (ROM) available in the TwinModel and evaluated at the
        given time instant. The image file shows the field results of the ROM in the given predefined view.

        Parameters
        ----------
        rom_name : str
            This is the name of a ROM model that is available in the TwinModel. See TwinModel.tbrom_names property to
            get a list of available ROM model.
        view_name : str
            The view name associated to the rendering view with which the ROM results are displayed. See
            TwinModel.get_available_view_names method to get a list of available rendering view names for a given ROM.
        evaluation_time: float
            This is the evaluation time at which you want to get the snapshot file. The method returns None if no
            snapshot file is available at this evaluation_time. Two evaluation times can be distinguished up to 6 digits
            after the comma.

        Raises
        ------
        TwinModelError:
            It raises an error if TwinModel has not been initialized.
            It raises an error if TwinModel does not include any TBROM.
            It raises an error if rom_name is not available.
            It raises an error if view_name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.initialize_evaluation()
        >>> rom_name = model.tbrom_names[0]
        >>> view_name = model.get_available_view_names(rom_name)[0]
        >>> geometry_filepath = TwinModel.get_image_filepath(rom_name, view_name)
        """
        self._log_key = "GetImageFilePath"

        if not self.evaluation_is_initialized:
            msg = "TwinModel has not been initialized! "
            msg += "Please initialize evaluation before to call this method!"
            self._raise_error(msg)

        if self.tbrom_info is None:
            self._raise_error("Twin model does not include any TBROM!")

        if rom_name not in self.tbrom_names:
            msg = f"The provided rom_name {rom_name} has not been found in the available TBROM names. "
            msg += f"Please call this method with a valid TBROM name."
            msg += f"\n Available TBROM name are: {self.tbrom_names}"
            self._raise_error(msg)

        view_names = self.get_available_view_names(rom_name)
        if view_name not in view_names:
            msg = f"The provided view_name {view_name} is not available for rom_name {rom_name}."
            msg += f"Please call this method with a valid view name."
            msg += f'\n Available view name for "{rom_name}" are: {view_names}'
            self._raise_error(msg)

        filename = f"{view_name}_"
        filename += f"{format(evaluation_time, self.TBROM_FILENAME_TIME_FORMAT)}"
        filename += f"{self.TBROM_IMAGE_EXT}"
        filepath = os.path.join(self.tbrom_directory_path, rom_name, filename)

        if not os.path.exists(filepath):
            msg = f"Could not find the image file for given available rom_name: {rom_name}, "
            msg += f"available view_name: {view_name} "
            msg += f"and evaluation_time: {evaluation_time}."
            msg += f"Image filepath you are looking for is: {filepath}"
            self._log_message(msg, level=PyTwinLogLevel.PYTWIN_LOG_WARNING)

        return filepath

    def get_geometry_filepath(self, rom_name: str):
        """
        Get the geometry file associated to a Reduced Order Model (ROM) available in the TwinModel. The geometry file
        contains the coordinates of the points that are used to define the geometrical support of the ROM field output.

        Parameters
        ----------
        rom_name : str
            This is the name of a ROM model that is available in the TwinModel. See TwinModel.tbrom_names property to
            get a list of available ROM model.

        Raises
        ------
        TwinModelError:
            It raises an error if TwinModel has not been initialized.
            It raises an error if TwinModel does not include any TBROM.
            It raises an error if rom_name is not available.
            It raises an error if geometry file cannot be found for an available ROM.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.initialize_evaluation()
        >>> available_rom_names = model.tbrom_names
        >>> geometry_filepath = TwinModel.get_geometry_filepath(rom_name=available_rom_names[0])
        """
        self._log_key = "GetGeometryFilePath"

        if not self.evaluation_is_initialized:
            msg = "TwinModel has not been initialized! "
            msg += "Please initialize evaluation before to call the geometry file getter!"
            self._raise_error(msg)

        if self.tbrom_info is None:
            self._raise_error("Twin model does not include any TBROM!")

        if rom_name not in self.tbrom_names:
            msg = f"The provided rom_name {rom_name} has not been found in the available TBROM names. "
            msg += f"Please call the geometry file getter with a valid TBROM name."
            msg += f"\n Available TBROM name are: {self.tbrom_names}"
            self._raise_error(msg)

        filepath = os.path.join(self._tbrom_resource_directory(rom_name), "binaryOutputField", "points.bin")

        if not os.path.exists(filepath):
            msg = f"Could not find the geometry file for given available rom_name: {rom_name}. "
            msg += f"Geometry filepath you are looking for is: {filepath}"
            self._raise_error(msg)

        return filepath

    def get_rom_directory(self, rom_name):
        """
        Get working directory path for a given Reduced Order Model (ROM) available in the TwinModel.

        Parameters
        ----------
        rom_name : str
            This is the name of a ROM model that is available in the TwinModel. See TwinModel.tbrom_names property to
            get a list of available ROM model.

        Raises
        ------
        TwinModelError:
            It raises an error if TwinModel has not been initialized.
            It raises an error if TwinModel does not include any TBROM.
            It raises an error if rom_name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.initialize_evaluation()
        >>> model.get_rom_directory(model.tbrom_names[0])
        """
        self._log_key = "GetRomDirectory"

        if not self.evaluation_is_initialized:
            msg = "TwinModel has not been initialized! "
            msg += "Please initialize evaluation before to call this method!"
            self._raise_error(msg)

        if self.tbrom_info is None:
            self._raise_error("Twin model does not include any TBROM!")

        if rom_name not in self.tbrom_names:
            msg = f"The provided rom_name {rom_name} has not been found in the available TBROM names. "
            msg += f"Please call this method with a valid TBROM name."
            msg += f"\n Available TBROM name are: {self.tbrom_names}"
            self._raise_error(msg)

        return os.path.join(self.tbrom_directory_path, rom_name)

    def get_snapshot_filepath(self, rom_name: str, evaluation_time: float = 0.0):
        """
        Get the snapshot file associated to a Reduced Order Model (ROM) available in the TwinModel and evaluated at the
        given time instant. The snapshot file contains the field results of the ROM.

        Parameters
        ----------
        rom_name : str
            This is the name of a ROM model that is available in the TwinModel. See TwinModel.tbrom_names property to
            get a list of available ROM model.
        evaluation_time: float
            This is the evaluation time at which you want to get the snapshot file. The method returns None if no
            snapshot file is available at this evaluation_time. Two evaluation times can be distinguished up to 6 digits
            after the comma.

        Raises
        ------
        TwinModelError:
            It raises an error if TwinModel has not been initialized.
            It raises an error if TwinModel does not include any TBROM.
            It raises an error if rom_name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.initialize_evaluation()
        >>> available_rom_names = model.tbrom_names
        >>> geometry_filepath = TwinModel.get_snapshot_filepath(rom_name=available_rom_names[0])
        """
        self._log_key = "GetSnapshotFilePath"

        if not self.evaluation_is_initialized:
            msg = "TwinModel has not been initialized! "
            msg += "Please initialize evaluation before to call the snapshot file getter!"
            self._raise_error(msg)

        if self.tbrom_info is None:
            self._raise_error("Twin model does not include any TBROM!")

        if rom_name not in self.tbrom_names:
            msg = f"The provided rom_name {rom_name} has not been found in the available TBROM names. "
            msg += f"Please call the snapshot file getter with a valid TBROM name."
            msg += f"\n Available TBROM name are: {self.tbrom_names}"
            self._raise_error(msg)

        filename = f"{self.TBROM_SNAPSHOT_FILE_PREFIX}"
        filename += f"{format(evaluation_time, self.TBROM_FILENAME_TIME_FORMAT)}"
        filename += f"{self.TBROM_SNAPSHOT_EXT}"
        filepath = os.path.join(self.tbrom_directory_path, rom_name, filename)

        if not os.path.exists(filepath):
            msg = f"Could not find the snapshot file for given available rom_name: {rom_name} "
            msg += f"and evaluation_time: {evaluation_time}."
            msg += f"Snapshot filepath you are looking for is: {filepath}"
            self._log_message(msg, level=PyTwinLogLevel.PYTWIN_LOG_WARNING)

        return filepath

    def load_state(self, model_id: str, evaluation_time: float, epsilon: float = 1e-8):
        """
        Load a state that has been saved by a TwinModel instantiated with same .twin file. Calling this method replaces
        evaluation initialization.

        Parameters
        ----------
        model_id: str
            This is the id of the model that saved the state.
        evaluation_time: float
            Evaluation time at which the state was saved.
        epsilon: float
            Absolute period that is added before and after evaluation time to account for round off error while
            searching the saved state. Search is performed in the interval [t-epsilon, t+epsilon]
            with t the evaluation time. First found saved state in this interval is loaded.

        Raises
        ------
        TwinModelError:
            If no state has been saved by model with given model_id and same model name as the one calling this method.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> # Instantiate a TwinModel, initialize it and evaluate it step by step until you want to save its state
        >>> model1 = TwinModel('model.twin')
        >>> model1.initialize_evaluation()
        >>> model1.evaluate_step_by_step(step_size=0.1)
        >>> model1.save_state()
        >>> # Instantiate a new TwinModel with same twin file and load the saved state
        >>> model2 = TwinModel('model.twin')
        >>> model2.load_state(model_id=model1.id, evaluation_time=model1.evaluation_time)
        >>> model2.evaluate_step_by_step(step_size=0.1)
        """
        self._log_key = "LoadState"

        try:
            # Search for existing state in registry
            ss_registry = SavedStateRegistry(model_id=model_id, model_name=self.name)
            ss = ss_registry.extract_saved_state(evaluation_time, epsilon)
            ss_filepath = ss_registry.return_saved_state_filepath(ss)

            # Initialize model accordingly and load existing state
            self._initialize_evaluation(parameters=ss.parameters, inputs=ss.inputs)
            self._twin_runtime.twin_load_state(ss_filepath)
            self._evaluation_time = ss.time

            BU732106_WORKAROUND = True
            if BU732106_WORKAROUND:
                # Rather we call a step by step evaluation with a small time-step OR we use the registry outputs
                # self.evaluate_step_by_step(step_size=ss.time * 1e-12, inputs=ss.inputs)
                self._outputs = ss.outputs
            else:
                self._update_outputs()

        except Exception as e:
            msg = f"Something went wrong while loading state:"
            msg += f"\n{str(e)}"
            self._raise_error(msg)

    def save_state(self):
        """
        Save the state of a TwinModel. This method will save the state of the twin model after its initialization and/or
        step by step evaluation.

        It should be used in conjunction with the `load_state` method.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> # Instantiate a TwinModel, initialize it and evaluate it step by step until you want to save its state
        >>> model1 = TwinModel('model.twin')
        >>> model1.initialize_evaluation()
        >>> model1.evaluate_step_by_step(step_size=0.1)
        >>> model1.save_state()
        >>> # Instantiate a new TwinModel with same twin file and load the saved state
        >>> model2 = TwinModel('model.twin')
        >>> model2.load_state(model_id=model1.id, evaluation_time=model1.evaluation_time)
        >>> model2.evaluate_step_by_step(step_size=0.1)
        """
        self._log_key = "SaveState"

        try:
            # Lazy init saved state registry for this TwinModel
            if self._ss_registry is None:
                self._ss_registry = SavedStateRegistry(model_id=self.id, model_name=self.name)

            # Store saved state meta-data
            ss = SavedState()
            ss.time = self.evaluation_time
            ss.parameters = self.parameters
            ss.outputs = self.outputs
            ss.inputs = self.inputs
            ss_filepath = self._ss_registry.return_saved_state_filepath(ss)

            # Create actual saved state and register it
            self._twin_runtime.twin_save_state(save_to=ss_filepath)
            self._ss_registry.append_saved_state(ss)
        except Exception as e:
            msg = f"Something went wrong while saving state:"
            msg += f"\n{str(e)}"
            self._raise_error(msg)


class TwinModelError(Exception):
    def __str__(self):
        return f"[TwinModelError] {self.args[0]}"
