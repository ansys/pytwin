import os
import time
import json
import pandas as pd
import numpy as np

from src.ansys.twin.twin_runtime.twin_runtime_core import TwinRuntime


class TwinModel:
    """
    Class to run Twin model as evaluation model.
    This class takes twin model filepath. The model can be run in
    batch mode or step-by-step mode.
    """
    def __init__(self, model_filepath: str):
        self._evaluation_time = None
        self._initialization_time = None
        self._instantiation_time = None
        self._inputs = None
        self._model_filepath = None
        self._outputs = None
        self._parameters = None
        self._twin_runtime = None
        self._init_evaluation_has_been_done = None

        if self._check_model_filepath_is_valid(model_filepath):
            self._model_filepath = model_filepath
        self._instantiate_twin_model()

    def __del__(self):
        if self._twin_runtime is not None:
            self._twin_runtime.twin_close()

    def _check_model_filepath_is_valid(self, model_filepath):
        """
        (internal) Check provided twin model filepath is valid. Raise a TwinModelError if not.
        """
        if model_filepath is None:
            msg = f'TwinModel cannot be called with {model_filepath} as model_filepath!'
            msg += '\nPlease provide valid filepath to initialize the TwinModel object.'
            raise self._raise_error(msg)
        if not os.path.exists(model_filepath):
            msg = f'Provided twin model filepath: {model_filepath} does not exist!'
            msg += '\nPlease provide existing filepath to initialize the TwinModel object.'
            raise self._raise_error(msg)
        return True

    def _create_dataframe_inputs(self, inputs_df: pd.DataFrame):
        """Create a dataframe inputs that satisfies the conventions of the runtime SDK batch mode evaluation, that are:
        (1) 'Time' as first column (2) one column per twin model input (3) columns order is the same as twin model
        input names list return by SDK.

        If an input is not found in the given inputs_df, then initialization value is used to keep associated input
        constant over Time."""
        if 'Time' not in inputs_df:
            msg = 'Given inputs dataframe has no \'Time\' column!'
            msg += f'\nExisting column labels are :{[s for s in inputs_df.columns]}'
            msg += f'\nPlease provide a dataframe with a \'Time\' column to use batch mode evaluation.'
            self._raise_error(msg)

        _inputs_df = pd.DataFrame()
        _inputs_df['Time'] = inputs_df['Time']
        for name, value in self._inputs.items():
            if name in inputs_df:
                _inputs_df[name] = inputs_df[name]
            else:
                _inputs_df[name] = np.full(shape=(_inputs_df.shape[0], 1), fill_value=value)
        return _inputs_df

    def _initialize_evaluation(self, parameters: dict = None, inputs: dict = None):
        """
        (internal) Initialize the twin model evaluation with dictionaries:
        (1) Initialize parameters and/or inputs values to their start values (default value found in the twin file),
        (2) Update parameters and/or inputs values with provided dictionaries. Ignore value whose name is not found
        into the list of parameters/inputs names of the twin model (value is kept to default one in that case).
        (3) Initialize evaluation time to 0.
        (4) Save universal time (time since epoch) at which the method is called
        (5) Evaluation twin model at time instant 0. and store its results into outputs dictionary.
        Twin runtime is reset in case of already initialized twin model.
        """
        if self._twin_runtime is None:
            self._raise_error('Twin model has not been successfully instantiated!')

        if self._init_evaluation_has_been_done:
            if self._twin_runtime.is_model_initialized:
                self._twin_runtime.twin_reset()

        self._initialize_parameters_with_start_values()
        if parameters is not None:
            self._update_parameters(parameters)

        self._initialize_inputs_with_start_values()
        if inputs is not None:
            self._update_inputs(inputs)

        self._evaluation_time = 0.0
        self._initialization_time = time.time()
        self._twin_runtime.twin_initialize()
        self._update_outputs()

        self._init_evaluation_has_been_done = True

    def _initialize_inputs_with_start_values(self):
        """
        (internal) Initialize inputs dictionary {name:value} with starting input values found in twin model.
        """
        self._inputs = dict()
        for name in self._twin_runtime.twin_get_input_names():
            self._inputs[name] = self._twin_runtime.twin_get_var_start(var_name=name)

    def _initialize_parameters_with_start_values(self):
        """
        (internal) Initialize parameters dictionary {name:value} with starting parameter values found in twin model.
        """
        self._parameters = dict()
        for name in self._twin_runtime.twin_get_param_names():
            if 'solver.' not in name:
                self._parameters[name] = self._twin_runtime.twin_get_var_start(var_name=name)

    def _initialize_outputs_with_none_values(self):
        """
        (internal) Initialize outputs dictionary {name:value} with None values.
        """
        output_names = self._twin_runtime.twin_get_output_names()
        output_values = [None]*len(output_names)
        self._outputs = dict(zip(output_names, output_values))

    def _instantiate_twin_model(self):
        """
        (internal) Connect TwinModel with TwinRuntime and load twin model.
        """
        try:
            self._twin_runtime = TwinRuntime(model_path=self._model_filepath, load_model=True)
            self._twin_runtime.twin_instantiate()
            self._instantiation_time = time.time()
            self._initialize_inputs_with_start_values()
            self._initialize_parameters_with_start_values()
            self._initialize_outputs_with_none_values()
            self._init_evaluation_has_been_done = False
        except Exception as e:
            msg = 'Twin model failed during instantiation!'
            msg += f'\n{str(e)}'
            self._raise_error(msg)

    def _raise_error(self, msg):
        """
        (internal) Raise a TwinModelError with formatted message.
        """
        raise TwinModelError(msg)

    def _read_eval_init_config(self, json_filepath: str):
        """
        (internal) Deserialize a json object into a dictionary that is used to store twin model inputs and parameters values
        to be passed to the internal evaluation initialization method.
        """
        if not os.path.exists(json_filepath):
            msg = 'Provided config filepath (for evaluation initialization) does not exist!'
            msg += f'\nProvided filepath is: {json_filepath}'
            msg += '\nPlease provide an existing filepath to initialize the twin model evaluation.'
            raise self._raise_error(msg)
        try:
            with open(json_filepath) as file:
                cfg = json.load(file)
                return cfg
        except Exception as e:
            msg = 'Something went wrong while reading config file!'
            msg += f'n{str(e)}'
            self._raise_error(msg)

    def _update_inputs(self, inputs: dict):
        """(internal) Update input values with given dictionary."""
        for name, value in inputs.items():
            if name in self._inputs:
                self._inputs[name] = value
                self._twin_runtime.twin_set_input_by_name(input_name=name, value=value)

    def _update_outputs(self):
        """(internal) Update output values with twin model results at current evaluation time."""
        self._outputs = dict(zip(self._twin_runtime.twin_get_output_names(), self._twin_runtime.twin_get_outputs()))

    def _update_parameters(self, parameters: dict):
        """(internal) Update parameters values with given dictionary."""
        for name, value in parameters.items():
            if name in self._parameters:
                self._parameters[name] = value
                self._twin_runtime.twin_set_param_by_name(param_name=name, value=value)

    @property
    def evaluation_is_initialized(self):
        """Return true if evaluation has been initialized."""
        if self._twin_runtime is None:
            self._raise_error('Twin model has not been successfully instantiated!')
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

    def initialize_evaluation(self, parameters: dict = None, inputs: dict = None, json_config_filepath: str = None):
        """
        Initialize the twin model evaluation with:
        (1) a dictionary of parameters values and/or inputs (start) values
        OR
        (2) a json configuration file with below format:
        {
            'version': '0.1.0',
            'model': {
                'inputs': {
                    'input-name-1': 'some-number',
                    'input-name-2': 'some-number'
                },
                'parameters': {
                    'param-name-1': 'some-number',
                    'param-name-2': 'some-number'
                }
        }

        Parameters and inputs that are not found into the provided dictionaries or config file, keep their default
        values (i.e. the start value of the twin model).

        Evaluation time is reset to zero after calling this method and Initialization time is updated.

        This method must be called:
        (1) before to evaluate the twin model,
        (2) if you want to update parameters values between multiple twin evaluations
        (in such case the twin model is reset).
        """
        if json_config_filepath is None:
            self._initialize_evaluation(parameters=parameters, inputs=inputs)
        else:
            cfg = self._read_eval_init_config(json_config_filepath)
            self._initialize_evaluation(parameters=cfg['model']['parameters'], inputs=cfg['model']['inputs'])

    def evaluate_step_by_step(self, step_size: float, inputs: dict = None):
        """
        Evaluate the twin model a next time step (equals current time step plus step_size) and return list of
        outputs values at next time step (ordered by output_names).

        Twin model evaluation must have been initialized before calling this method
        (see `initialize_evaluation` method).

        **step_size:** it is the step size (in second) to reach next time step. It must be strictly positive.

        **inputs:** dictionary of input values (name:value) at next time step. Input is not updated if
        associated key is not found in twin model input_names. Other inputs keep current value at next time step if not
        provided in the inputs dict.
        """
        if self._twin_runtime is None:
            self._raise_error('Twin model has not been successfully instantiated!')

        if not self.evaluation_is_initialized:
            self._raise_error('Twin model evaluation has not been initialized! Please initialize evaluation.')

        if step_size <= 0.:
            msg = f'Step size must be strictly bigger than zero ({step_size} was provided)!'
            self._raise_error(msg)

        if inputs is not None:
            self._update_inputs(inputs)

        self._evaluation_time += step_size
        try:
            self._twin_runtime.twin_simulate(self._evaluation_time)
            self._update_outputs()
        except Exception as e:
            msg = f'Something went wrong during evaluation at time step {self._evaluation_time}:'
            msg += f'\n{str(e)}'
            msg += f'\nPlease reinitialize the model evaluation and restart it.'
            self._raise_error(msg)

    def evaluate_batch(self, inputs_df: pd.DataFrame):
        """
        Evaluate the twin model with the historical inputs' data given with a Pandas.DataFrame().

        Twin model evaluation must be initialized before calling this method, which solve time instant 0.(s).

        Historical inputs must not include time instant 0.

        **inputs_df:** is a pandas.DataFrame with historical inputs data. It must have a 'Time' column and all twin
        model inputs history you want to simulate (one input per column). If a twin model input is not found in the
        dataframe columns then this input is kept constant to its initialization value. The column header must match
        with a twin model input name.

        **return:** a pandas.DataFrame with all twin model outputs associated to the historical inputs' data.
        """
        if self._twin_runtime is None:
            self._raise_error('Twin model has not been successfully instantiated!')

        if not self.evaluation_is_initialized:
            self._raise_error('Twin model evaluation has not been initialized! Please initialize evaluation.')

        _inputs_df = self._create_dataframe_inputs(inputs_df)
        _output_col_names = ['Time'] + list(self._outputs.keys())

        return self._twin_runtime.twin_simulate_batch_mode(input_df=_inputs_df, output_column_names=_output_col_names)


class TwinModelError(Exception):
    def __str__(self):
        return f'[pyAnsys][pyTwin][TwinModelError] {self.args[0]}'
