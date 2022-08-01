import logging
import pathlib
import platform
from enum import Enum

# Local application imports
from src.ansys.twin.twin_runtime.twin_runtime_core import TwinRuntime


class TwinState(Enum):
    TWIN_CLOSED = 0
    TWIN_OPENED = 1
    TWIN_INSTANTIATED = 2
    TWIN_INITIALIZED = 3


# TODO : reformat class/method documentation (example explicitly given in docstrings or implementation example for
#  more complex object) 

class TwinModel:
    """Class to run Twin model as evaluation model.
    This class takes twin model filepath. The model can be run in
    batch mode or step-by-step mode."""

    def __init__(self, model_path):
        if model_path is None:
            raise ValueError('TwinModel cannot be called with None as model_path!')
        self.model_path = model_path
        self.twin_runtime = None
        self.parameters = None
        self.inputs = None
        self.outputs = None
        self.twin_time = None
        self.state = TwinState.TWIN_CLOSED.value
        self.instantiate_twin_model()

    def get_twin_model_path(self):
        """method to get twin model
        filepath"""
        return self.model_path

    def get_twin_model_state(self):
        """method to get twin model
        status"""
        return self.state

    @staticmethod
    def get_runtime_path():
        """method to get required paths
        to initialize twin model instance"""
        lib_file = ''
        if platform.system() == 'Windows':
            lib_file = 'TwinRuntimeSDK.dll'
        elif platform.system() == 'Linux':
            lib_file = 'TwinRuntimeSDK.so'
        else:
            logging.critical(f'Platform ({platform.system()}) is not supported by twin runtime!')
        cwd_address = str(pathlib.Path(__file__).parent.parent.absolute())
        final_address = str(pathlib.Path(cwd_address, 'twin_runtime', lib_file))
        return final_address

    def get_twin_inputs_names(self):
        """returns a list of inputs from twin model"""
        return self.twin_runtime.twin_get_input_names()

    def get_twin_outputs_names(self):
        """returns a list of outputs from twin model"""
        return self.twin_runtime.twin_get_output_names()

    def get_twin_parameters_names(self):
        """returns a list of parameters from twin model"""
        return self.twin_runtime.twin_get_param_names()

    def instantiate_twin_model(self):
        """method to instantiate twin model"""
        self.twin_runtime = TwinRuntime(model_path=self.get_twin_model_path(),
                                        twin_runtime_library_path=self.get_runtime_path(),
                                        load_model=True)
        self.state = TwinState.TWIN_OPENED.value
        self.twin_runtime.twin_instantiate()
        self.parameters = self.get_twin_parameters_names()
        self.inputs = self.get_twin_inputs_names()
        self.outputs = self.get_twin_outputs_names()
        self.state = TwinState.TWIN_INSTANTIATED.value

    def initialize_twin_model(self, parameters: dict = None, inputs: dict = None):
        # TO DO TEST : what happens if fewer param/inputs provided, what happens if wrong parameters/inputs provided,...
        """method to initialize the twin model given the parameters values and inputs (start) values, arguments are
        provided as dictionaries {name,value} """
        # if twin already initialized, it needs to be reset
        if self.get_twin_model_state() == TwinState.TWIN_INITIALIZED.value:
            self.twin_runtime.twin_reset()

        # setting parameters and input start values
        if parameters is not None:
            for name, value in parameters:
                if name in self.parameters:
                    self.twin_runtime.twin_set_param_by_name(name, value)
        if inputs is not None:
            for name, value in inputs:
                if name in self.inputs:
                    self.twin_runtime.twin_set_input_by_name(name, value)

        self.twin_time = 0.0
        self.twin_runtime.twin_initialize()
        self.state = TwinState.TWIN_INITIALIZED.value
        return self.twin_runtime.twin_get_outputs()

    def compute_twin_model_step(self, step_size, inputs=None):
        """method to simulate the Twin with step_size, inputs argument are
        provided as dictionaries {name,value} """
        if inputs is not None:
            for name, value in inputs:
                if name in self.inputs:
                    self.twin_runtime.twin_set_input_by_name(name, value)

        self.twin_runtime.twin_simulate(self.twin_time + step_size)
        self.twin_time = self.twin_time + step_size
        return self.twin_runtime.twin_get_outputs()

    def compute_twin_model_batch(self, inputs_df):
        """method to simulate the Twin in batch mode, inputs argument are
        provided as pandas dataframes """
        step_size = 0
        interpolate = 0
        inputs_df.set_index('Time', inplace=True)
        output_columns = ['Time'] + list(self.outputs)
        outputs_df = self.twin_runtime.twin_simulate_batch_mode(inputs_df, output_columns, step_size, interpolate,
                                                                time_as_index=True)
        return outputs_df

    def close(self):
        """method to close the twin model"""
        self.twin_runtime.twin_close()
        self.state = TwinState.TWIN_CLOSED.value
