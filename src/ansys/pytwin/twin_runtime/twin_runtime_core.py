import pandas as pd
import numpy as np
import os
import json
import sys
import math
import platform
from pathlib import Path
from ctypes import*

if platform.system() == 'Windows':
    import win32api

from .twin_runtime_error import *
from .twin_runtime_error import TwinRuntimeError
from .log_level import LogLevel

CUR_DIR = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
os.environ['TWIN_RUNTIME_SDK'] = CUR_DIR
default_log_name = "model.log"


class TwinStatus(Enum):
    TWIN_STATUS_OK = 0
    TWIN_STATUS_WARNING = 1
    TWIN_STATUS_ERROR = 2
    TWIN_STATUS_FATAL = 3


class TwinRuntime:

    debug_mode = False
    twin_status = None
    is_model_opened = False
    is_model_initialized = False
    is_model_instantiated = False
    last_time_stop = 0

    model_name = None
    number_parameters = None
    number_inputs = None
    number_outputs = None

    has_default_settings = False
    p_end_time = None
    p_step_size = None
    p_tolerance = None

    output_names = None
    input_names = None
    parameter_names = None
    os_version = None

    if platform.system() == 'Windows':
        twin_runtime_library = 'TwinRuntimeSDK.dll'
        os_version = 'win64'
    else:
        twin_runtime_library = 'libTwinRuntimeSDK.so'
        os_version = 'linux64'

    @staticmethod    
    def load_dll(twin_runtime_library_path=None):

        def _setup_env(sdk_folder_path):
            if platform.system() == 'Windows':
                sep = ';'
            else:
                sep = ':'
            if sdk_folder_path not in os.environ['PATH']: 
                os.environ['PATH'] = '{}{}{}'.format(sdk_folder_path, sep, os.environ['PATH'])

        if twin_runtime_library_path is None:
            folder = os.path.join(CUR_DIR, TwinRuntime.os_version)
            _setup_env(sdk_folder_path=folder)
            return cdll.LoadLibrary(os.path.join(folder, TwinRuntime.twin_runtime_library))
        else:
            _setup_env(sdk_folder_path=os.path.dirname(twin_runtime_library_path))
            return cdll.LoadLibrary(twin_runtime_library_path)

    @staticmethod
    def twin_is_cross_platform(file_path):
        twin_runtime_library = TwinRuntime.load_dll()
        TwinIsCrossPlatform = twin_runtime_library.IsTwinCrossPlatform
        
        if type(file_path) is not bytes:
            file_path = file_path.encode()
        
        cross_platform = c_bool()
        TwinIsCrossPlatform(c_char_p(file_path), byref(cross_platform))
        return cross_platform.value

    @staticmethod
    def get_twin_version(file_path):
        twin_runtime_library = TwinRuntime.load_dll()
        TwinGetVersion = twin_runtime_library.TwinGetVersion

        if type(file_path) is not bytes:
            file_path = file_path.encode()

        valid_model = c_bool()
        twin_version = c_char_p()
        TwinGetVersion(c_char_p(file_path), byref(valid_model), byref(twin_version))
        return valid_model.value, twin_version.value.decode()

    @staticmethod
    def twin_get_model_dependencies(file_path):
        twin_runtime_library = TwinRuntime.load_dll()
        
        TwinGetModelDependencies = twin_runtime_library.TwinGetModelDependencies

        if type(file_path) is not bytes:
            file_path = file_path.encode()

        twin_dependencies = c_char_p()
        TwinGetModelDependencies(c_char_p(file_path), byref(twin_dependencies))
        twin_dependencies_dict = json.loads(twin_dependencies.value.decode())
        return twin_dependencies_dict

    @staticmethod
    def evaluate_twin_status(twin_status, twin_runtime, method_name):
        if twin_status == 1:
            message = "The method " + method_name + " caused a warning! \n"
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            print(message)

        elif twin_status == 2:
            message = "The method " + method_name + " caused a error!\n"
            message += "TwinRuntime error message: " + twin_runtime.twin_get_status_string()
            raise TwinRuntimeError(message, twin_runtime)

        elif twin_status == 3:
            message = "The method " + method_name + " caused a fatal error!\n"
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise TwinRuntimeError(message, twin_runtime)

    @staticmethod
    def evaluate_twin_prop_status(prop_status, twin_runtime, method_name, var):
        if prop_status == 4:
            message = "The method {} with the variable {} caused an error!\n".format(method_name.encode(), var)
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise PropertyError(message, twin_runtime, prop_status)

        elif prop_status == 3:
            message = "The method {} with the variable {} is invalid (i.e., variable does not exist)!\n".format(method_name.encode(), var)
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise PropertyInvalidError(message, twin_runtime, prop_status)

        elif prop_status == 2:
            message = "The method {} with the variable {} is not applicable!\n".format(method_name.encode(), var)
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise PropertyNotApplicableError(message, twin_runtime, prop_status)

        elif prop_status == 1:
            message = "The method {} with the variable {} is not defined!\n".format(method_name, var)
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise PropertyNotDefinedError(message, twin_runtime, prop_status)

    def __init__(self, model_path, log_path=None, twin_runtime_library_path=None, log_level=LogLevel.TWIN_LOG_WARNING,
                 load_model=True):

        local_path = os.path.dirname(__file__)
        model_path = Path(model_path)
        self.log_level = log_level

        if model_path.is_file() is False:
            raise FileNotFoundError("File is not found at {}".format(model_path.absolute()))

        self._twin_runtime_library = TwinRuntime.load_dll(twin_runtime_library_path)

        if log_path is None:
            # Getting parent directory
            file_name = os.path.splitext(model_path)[0]
            log_path = file_name + ".log"

        self.model_path = model_path.absolute().as_posix().encode()
        self.log_path = log_path.encode()

        # ---------------- Mapping sdk functions as class methods --------------------
        self._modelPointer = c_void_p()

        self._TwinOpen = self._twin_runtime_library.TwinOpen
        self._TwinOpen.restype = c_int

        self._TwinClose = self._twin_runtime_library.TwinClose

        self._TwinReset = self._twin_runtime_library.TwinReset
        self._TwinReset.restype = c_int

        self.TwinGetStatusString = self._twin_runtime_library.TwinGetStatusString
        self.TwinGetStatusString.argtypes = [c_void_p]
        self.TwinGetStatusString.restype = c_char_p

        self._TwinGetModelName = self._twin_runtime_library.TwinGetModelName
        self._TwinGetModelName.restype = c_char_p

        self._TwinGetAPIVersion = self._twin_runtime_library.TwinGetAPIVersion
        self._TwinGetAPIVersion.restype = c_char_p

        self._TwinGetNumParameters = self._twin_runtime_library.TwinGetNumParameters
        self._TwinGetNumParameters.restype = c_int

        self._TwinGetNumInputs = self._twin_runtime_library.TwinGetNumInputs
        self._TwinGetNumInputs.restype = c_int

        self._TwinGetNumOutputs = self._twin_runtime_library.TwinGetNumOutputs
        self._TwinGetNumOutputs.restype = c_int

        self._TwinGetParamNames = self._twin_runtime_library.TwinGetParamNames
        self._TwinGetParamNames.restype = c_int

        self._TwinGetInputNames = self._twin_runtime_library.TwinGetInputNames
        self._TwinGetInputNames.restype = c_int

        self._TwinGetOutputNames = self._twin_runtime_library.TwinGetOutputNames
        self._TwinGetOutputNames.restype = c_int

        self._TwinInstantiate = self._twin_runtime_library.TwinInstantiate
        self._TwinInstantiate.argtypes = [c_void_p]
        self._TwinInstantiate.restype = c_int

        self._TwinInitialize = self._twin_runtime_library.TwinInitialize
        self._TwinInitialize.argtypes = [c_void_p]
        self._TwinInitialize.restype = c_int

        self._TwinSetParamByName = self._twin_runtime_library.TwinSetParamByName
        self._TwinSetParamByName.argtypes = [c_void_p, c_char_p, c_double]
        self._TwinSetParamByName.restype = c_int

        self._TwinSetStrParamByName = self._twin_runtime_library.TwinSetStrParamByName
        self._TwinSetStrParamByName.argtypes = [c_void_p, c_char_p, c_char_p]
        self._TwinSetStrParamByName.restype = c_int

        self._TwinSetParamByIndex = self._twin_runtime_library.TwinSetParamByIndex
        self._TwinSetParamByIndex.argtypes = [c_void_p, c_int, c_double]
        self._TwinSetParamByIndex.restype = c_int

        self._TwinGetOutputs = self._twin_runtime_library.TwinGetOutputs
        self._TwinGetOutputs.restype = c_int

        self._TwinSimulate = self._twin_runtime_library.TwinSimulate
        self._TwinSimulate.restype = c_int

        self._TwinSimulateBatchMode = self._twin_runtime_library.TwinSimulateBatchMode
        self._TwinSimulateBatchMode.restype = c_int

        self._TwinSimulateBatchModeCSV = self._twin_runtime_library.TwinSimulateBatchModeCSV
        self._TwinSimulateBatchModeCSV.restype = c_int

        self._TwinSetInputs = self._twin_runtime_library.TwinSetInputs
        self._TwinSetInputs.restype = c_int

        self._TwinSetInputByName = self._twin_runtime_library.TwinSetInputByName
        self._TwinSetInputByName.argtypes = [c_void_p, c_char_p, c_double]
        self._TwinSetInputByName.restype = c_int

        self._TwinSetInputByIndex = self._twin_runtime_library.TwinSetInputByIndex
        self._TwinSetInputByIndex.argtypes = [c_void_p, c_int, c_double]
        self._TwinSetInputByIndex.restype = c_int

        self._TwinGetOutputByName = self._twin_runtime_library.TwinGetOutputByName
        self._TwinGetOutputByName.restype = c_int

        self._TwinGetOutputByIndex = self._twin_runtime_library.TwinGetOutputByIndex
        self._TwinGetOutputByIndex.restype = c_int

        self._TwinGetDefaultSimulationSettings = self._twin_runtime_library.TwinGetDefaultSimulationSettings
        self._TwinGetDefaultSimulationSettings.restype = c_int

        self._TwinGetVarDataType = self._twin_runtime_library.TwinGetVarDataType
        self._TwinGetVarDataType.restype = c_int

        self._TwinGetVarUnit = self._twin_runtime_library.TwinGetVarUnit
        self._TwinGetVarUnit.restype = c_int

        self._TwinGetVarStart = self._twin_runtime_library.TwinGetVarStart
        self._TwinGetVarStart.restype = c_int

        self._TwinGetStrVarStart = self._twin_runtime_library.TwinGetStrVarStart
        self._TwinGetStrVarStart.restype = c_int

        self._TwinGetVarMin = self._twin_runtime_library.TwinGetVarMin
        self._TwinGetVarMin.restype = c_int

        self._TwinGetVarMax = self._twin_runtime_library.TwinGetVarMax
        self._TwinGetVarMax.restype = c_int

        self._TwinGetVarNominal = self._twin_runtime_library.TwinGetVarNominal
        self._TwinGetVarNominal.restype = c_int

        self._TwinGetVarQuantityType = self._twin_runtime_library.TwinGetVarQuantityType
        self._TwinGetVarQuantityType.restype = c_int

        self._TwinGetVarDescription = self._twin_runtime_library.TwinGetVarDescription
        self._TwinGetVarDescription.restype = c_int

        self._TwinGetVisualizationResources = self._twin_runtime_library.TwinGetVisualizationResources
        self._TwinGetVisualizationResources.restype = c_int

        self._TwinEnableROMImages = self._twin_runtime_library.TwinEnableROMImages
        self._TwinEnableROMImages.restype = c_int

        self._TwinDisableROMImages = self._twin_runtime_library.TwinDisableROMImages
        self._TwinDisableROMImages.restype = c_int

        self._TwinEnable3DROMData = self._twin_runtime_library.TwinEnable3DROMData
        self._TwinEnable3DROMData.restype = c_int

        self._TwinDisable3DROMData = self._twin_runtime_library.TwinDisable3DROMData
        self._TwinDisable3DROMData.restype = c_int

        self._TwinGetRomImageFiles = self._twin_runtime_library.TwinGetRomImageFiles
        self._TwinGetRomImageFiles.restype = c_int

        self._TwinGetNumRomImageFiles = self._twin_runtime_library.TwinGetNumRomImageFiles
        self._TwinGetNumRomImageFiles.restype = c_int

        self._TwinGetRomModeCoefFiles = self._twin_runtime_library.TwinGetRomModeCoefFiles
        self._TwinGetRomModeCoefFiles.restype = c_int

        self._TwinGetNumRomModeCoefFiles = self._twin_runtime_library.TwinGetNumRomModeCoefFiles
        self._TwinGetNumRomModeCoefFiles.restype = c_int

        self._TwinGetRomSnapshotFiles = self._twin_runtime_library.TwinGetRomSnapshotFiles
        self._TwinGetRomSnapshotFiles.restype = c_int

        self._TwinGetNumRomSnapshotFiles = self._twin_runtime_library.TwinGetNumRomSnapshotFiles
        self._TwinGetNumRomSnapshotFiles.restype = c_int

        self._TwinGetDefaultROMImageDirectory = self._twin_runtime_library.TwinGetDefaultROMImageDirectory
        self._TwinGetDefaultROMImageDirectory.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
        self._TwinGetDefaultROMImageDirectory.restype = c_int

        self._TwinGetRomResourcePath = self._twin_runtime_library.TwinGetRomResourcePath
        self._TwinGetRomResourcePath.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
        self._TwinGetRomResourcePath.restype = c_int

        self._TwinSetROMImageDirectory = self._twin_runtime_library.TwinSetROMImageDirectory
        self._TwinSetROMImageDirectory.argtypes = [c_void_p, c_char_p, c_char_p]
        self._TwinSetROMImageDirectory.restype = c_int

        self._TwinSaveState = self._twin_runtime_library.TwinSaveState
        self._TwinSaveState.argtypes = [c_void_p]
        self._TwinSaveState.restype = c_int

        self._TwinLoadState = self._twin_runtime_library.TwinLoadState
        self._TwinLoadState.argtypes = [c_void_p]
        self._TwinLoadState.restype = c_int

        model_path = Path(model_path)
        if model_path.is_file() is False:
            raise FileNotFoundError("File is not found at {}".format(model_path.absolute()))

        if log_path is None:
            # Getting parent directory
            file_name = os.path.splitext(model_path)[0]
            log_path = file_name + ".log"

        self.model_path = model_path.absolute().as_posix().encode()
        self.log_path = log_path.encode()

        if load_model:
            self.twin_load(log_level)

    """
    Model opening/closing
    Functions for opening and closing a Twin model. Opening models is hidden within the constructor.
    """
    def twin_load(self, log_level):
        # This ensures that DLL loading mechanism gets reset to its default behavior, which is altered when the
        #  SDK launches in Twin Deployer built with PyInstaller.
        # If this is not reset, optislang FMUs don't load because their dependent DLLs (from the binaries/win64)
        #  are not found.
        # See https://github.com/pyinstaller/pyinstaller/issues/3795
        if platform.system() == 'Windows':
            win32api.SetDllDirectory(None)
        file_buf = create_string_buffer(self.model_path)
        log_buf = create_string_buffer(self.log_path)

        self.twin_status = self._TwinOpen(file_buf, byref(self._modelPointer), log_buf, c_int(log_level.value))

        self.evaluate_twin_status(self.twin_status, self, "twin_load")
        self.is_model_opened = True

        self.model_name = self._TwinGetModelName(self._modelPointer)

        self.twin_get_number_inputs()
        self.twin_get_number_outputs()
        self.twin_get_number_params()

        self.twin_get_param_names()
        self.twin_get_input_names()
        self.twin_get_output_names()
        self.load_twin_default_sim_settings()

    def twin_close(self):
        if not self.is_model_opened:
            print('[Warning]: twin_close() will not execute since model is not loaded. Maybe it was already closed?')
            return
        self._TwinClose(self._modelPointer)
        self.is_model_opened = False
        self.is_model_initialized = False
        self.is_model_instantiated = False
        self.last_time_stop = 0

        self.model_name = None
        self.number_parameters = None
        self.number_inputs = None
        self.number_outputs = None

        self.has_default_settings = None
        self.p_end_time = None
        self.p_step_size = None
        self.p_tolerance = None

        self.output_names = None
        self.input_names = None
        self.parameter_names = None

    """
    Model properties
    Functions for getting model properties
    """
    def twin_get_model_name(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning its name")

        return self._TwinGetModelName(self._modelPointer).decode()

    def twin_get_number_params(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning the number of parameters!")

        if self.number_parameters is None:
            c_number_params = c_int(0)
            self.twin_status = self._TwinGetNumParameters(self._modelPointer, byref(c_number_params))
            self.evaluate_twin_status(self.twin_status, self, "twin_get_number_params")
            self.number_parameters = c_number_params.value
            return self.number_parameters
        else:
            return self.number_parameters

    def twin_get_number_inputs(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning the number of inputs!")

        if self.number_inputs is None:
            c_number_inputs = c_int(0)
            self.twin_status = self._TwinGetNumInputs(self._modelPointer, byref(c_number_inputs))
            self.evaluate_twin_status(self.twin_status, self, "twin_get_number_inputs")
            self.number_inputs = c_number_inputs.value
            return self.number_inputs
        else:
            return self.number_inputs

    def twin_get_number_outputs(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning the number of outputs!")

        if self.number_outputs is None:
            c_number_outputs = c_int(0)
            self.twin_status = self._TwinGetNumOutputs(self._modelPointer, byref(c_number_outputs))
            self.evaluate_twin_status(self.twin_status, self, "twin_get_number_outputs")
            self.number_outputs = c_number_outputs.value
            return self.number_outputs
        else:
            return self.number_outputs

    def twin_get_param_names(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning parameter names!")

        if self.parameter_names is None:
            self._TwinGetParamNames.argtypes = [c_void_p, POINTER(c_char_p * self.number_parameters), c_int]

            parameter_names_c = (c_char_p * self.number_parameters)()

            self.twin_status = self._TwinGetParamNames(self._modelPointer,parameter_names_c, self.number_parameters)
            self.evaluate_twin_status(self.twin_status, self, "twin_get_param_names")

            self.parameter_names = to_np_array(parameter_names_c)

            return self.parameter_names
        else:
            return self.parameter_names

    def twin_get_input_names(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning input names!")

        if self.input_names is None:
            self._TwinGetInputNames.argtypes = [c_void_p, POINTER(c_char_p * self.number_inputs), c_int]

            input_names_c = (c_char_p * self.number_inputs)()

            self.twin_status = self._TwinGetInputNames(self._modelPointer, input_names_c, self.number_inputs)
            self.evaluate_twin_status(self.twin_status, self, "twin_get_input_names")

            self.input_names = to_np_array(input_names_c)

            return self.input_names
        else:
            return self.input_names

    def twin_get_output_names(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning output names!")

        if self.output_names is None:
            self._TwinGetInputNames.argtypes = [c_void_p, POINTER(c_char_p * self.number_outputs), c_int]

            output_names_c = (c_char_p * self.number_outputs)()

            self.twin_status = self._TwinGetOutputNames(self._modelPointer, output_names_c, self.number_outputs)
            self.evaluate_twin_status(self.twin_status, self, "twin_get_output_names")

            self.output_names = to_np_array(output_names_c)

            return self.output_names
        else:
            return self.output_names

    def twin_get_default_simulation_settings(self):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning default settings!")

        c_end_time = c_double(0)
        c_step_size = c_double(0)
        c_tolerance = c_double(0)

        self.twin_status = self._TwinGetDefaultSimulationSettings(self._modelPointer, byref(c_end_time),
                                                                  byref(c_step_size), byref(c_tolerance))
        self.evaluate_twin_status(self.twin_status, self, "twin_get_default_sim_settings")

        return c_end_time.value, c_step_size.value, c_tolerance.value

    """
    Model variable properties
    Functions for getting specific property for a given model variable. 
    """
    def twin_get_var_data_type(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable data type!")

        var_type = c_char_p()

        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarDataType(self._modelPointer, c_char_p(var_name), byref(var_type))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_type", var_name)

        if var_type.value is None:
            return None
        else:
            return var_type.value.decode()

    def twin_get_var_quantity_type(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable quantity type!")

        quantity_type = c_char_p()

        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarQuantityType(self._modelPointer, c_char_p(var_name), byref(quantity_type))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_quantity_type", var_name)

        if quantity_type.value is None:
            return None
        else:
            return quantity_type.value.decode()

    def twin_get_var_description(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable description!")

        var_description = c_char_p()

        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarDescription(self._modelPointer, c_char_p(var_name), byref(var_description))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_description", var_name)

        if var_description.value is None:
            return None
        else:
            return var_description.value.decode()

    def twin_get_var_unit(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable unit type!")

        var_unit = c_char_p()

        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarUnit(self._modelPointer, c_char_p(var_name), byref(var_unit))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_unit", var_name)

        if var_unit.value is None:
            return None
        else:
            return var_unit.value.decode()

    def twin_get_var_start(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable start value!")

        start_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarStart(self._modelPointer, c_char_p(var_name), byref(start_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_start", var_name)

        return start_value.value

    def twin_get_str_var_start(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable start value!")

        start_value = c_char_p()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetStrVarStart(self._modelPointer, c_char_p(var_name), byref(start_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_str_var_start", var_name)

        return start_value.value.decode()

    def twin_get_var_min(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable minimum value!")

        min_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarMin(self._modelPointer, c_char_p(var_name), byref(min_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_min", var_name)

        return min_value.value

    def twin_get_var_max(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable maximum value!")

        max_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarMax(self._modelPointer, c_char_p(var_name), byref(max_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_max", var_name)

        return max_value.value

    def twin_get_var_nominal(self, var_name):
        if self.is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning variable nominal value!")

        nominal_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarNominal(self._modelPointer, c_char_p(var_name), byref(nominal_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_nominal", var_name)

        return nominal_value.value

    # def twin_get_str_var_nominal(self, var_name):
    #     if self.is_model_opened is False:
    #         raise TwinRuntimeError("The model has to be opened before returning variable nominal value!")
    #
    #     nominal_value = c_char_p()
    #     if type(var_name) is not bytes:
    #         var_name = var_name.encode()
    #
    #     property_status = self._TwinGetStrVarNominal(self._modelPointer, c_char_p(var_name), byref(nominal_value))
    #     self.evaluate_twin_prop_status(property_status, self, "twin_get_str_var_nominal", var_name)
    #
    #     return nominal_value.value

    """
    Simulation operations
    Functions for simulating the Twin model.
    """
    def twin_instantiate(self):
        self.twin_status = self._TwinInstantiate(self._modelPointer)
        self.evaluate_twin_status(self.twin_status, self, "twin_instantiate")
        self.is_model_instantiated = True

    def twin_initialize(self):
        if self.is_model_instantiated is False:
            raise TwinRuntimeError("The model has to be instantiated before initialization!")

        try:
            self.twin_status = self._TwinInitialize(self._modelPointer)
        except OSError:
            message = "Error while initializing the model. This model may need start values or has other dependencies."
            raise TwinRuntimeError(message)

        self.evaluate_twin_status(self.twin_status, self, "twin_initialize")
        self.is_model_initialized = True

    def twin_simulate(self, time_stop, time_step=0):
        if self.is_model_initialized is False:
            raise TwinRuntimeError("The Model has to be initialized before simulation!")

        self.twin_status = self._TwinSimulate(self._modelPointer, c_double(time_stop), c_double(time_step))
        self.evaluate_twin_status(self.twin_status, self, "twin_simulate")

    def twin_simulate_batch_mode(self, input_df, output_column_names, step_size=0, interpolate=0, time_as_index=False):
        output_number_of_columns = self.number_outputs + 1

        if self.is_model_initialized is False:
            raise TwinRuntimeError("The Model has to be initialized before simulation!")

        local_df = input_df  # Creates a local copy so that the source DF does not get modified outside this scope
        num_input_rows = local_df.shape[0]
        if time_as_index:
            local_df = local_df.reset_index()

        end_time = local_df.iloc[-1, 0]
        if step_size != 0:
            max_output_rows = int(math.ceil(end_time / step_size) + 1)
        else:
            if local_df.iloc[0, 0] > 0:
                max_output_rows = num_input_rows + 1  # + 1 to account for t=0 that's not on the input DF
            else:
                max_output_rows = num_input_rows

        # Preallocate Dataframe rows
        local_df = local_df.astype(np.float64)
        input_data = build_ctype_2d_array(num_input_rows, local_df)

        # Pandas float to Python equivalent
        out_data = build_empty_ctype_2d_array(max_output_rows, output_number_of_columns)

        self.twin_status = self._TwinSimulateBatchMode(self._modelPointer, byref(input_data), c_int(num_input_rows),
                                                       byref(out_data), c_int(max_output_rows),
                                                       c_double(step_size),
                                                       c_int(interpolate))
        data = [np.ctypeslib.as_array(out_data[i], shape=(output_number_of_columns,)) for i in range(max_output_rows)]
        output_df = pd.DataFrame(data=data, index=np.arange(0, max_output_rows), columns=output_column_names)
        self.evaluate_twin_status(self.twin_status, self, "twin_simulate_batch_mode")

        return output_df

    # This method will generate the response also as a csv
    def twin_simulate_batch_mode_csv(self, input_csv, output_csv, step_size=0, interpolate=0):
        if self.is_model_initialized is False:
            raise TwinRuntimeError("The Model has to be initialized before simulation!")

        if type(input_csv) is not bytes:
            input_csv = input_csv.encode()
        if type(output_csv) is not bytes:
            output_csv = output_csv.encode()

        input_csv = create_string_buffer(input_csv)
        output_csv = create_string_buffer(output_csv)

        self.twin_status = self._TwinSimulateBatchModeCSV(self._modelPointer, input_csv, output_csv,
                                                          c_double(step_size),
                                                          c_int(interpolate))

        self.evaluate_twin_status(self.twin_status, self, "twin_simulate_batch_mode_csv")

    def twin_reset(self):
        self.twin_status = self._TwinReset(self._modelPointer)
        self.evaluate_twin_status(self.twin_status, self, "twin_reset")

    """
    Input/output handling
    Functions for setting parameters/inputs and getting outputs.
    """
    def twin_set_inputs(self, input_array):
        if self.is_model_instantiated is False:
            raise TwinRuntimeError("The model has to be instantiated before setting inputs!")

        if len(input_array) != self.number_inputs:
            raise TwinRuntimeError("The input array size must match the the models number of inputs!")

        array_np = np.array(input_array)
        array_ctypes = array_np.ctypes.data_as(POINTER(c_double * self.number_inputs))

        self._TwinSetInputs.argtypes = [c_void_p, POINTER(c_double * self.number_inputs), c_int]
        self.twin_status = self._TwinSetInputs(self._modelPointer, array_ctypes, self.number_inputs)
        self.evaluate_twin_status(self.twin_status, self, "twin_get_outputs")

    def twin_get_outputs(self):
        if self.is_model_initialized is False:
            raise TwinRuntimeError("The Model has to be initialized before it can return outputs!")

        self._TwinGetOutputs.argtypes = [c_void_p, POINTER(c_double * self.number_outputs), c_int]
        outputs = (c_double * self.number_outputs)()

        self.twin_status = self._TwinGetOutputs(self._modelPointer, outputs, self.number_outputs)
        self.evaluate_twin_status(self.twin_status, self, "twin_get_outputs")

        outputs_list = np.array(outputs).tolist()
        return outputs_list

    def twin_set_param_by_name(self, param_name, value):
        if self.is_model_instantiated is False:
            raise TwinRuntimeError("The model has to be instantiated before setting parameters!")

        if isinstance(param_name, str):
            param_name = param_name.encode()

        self.twin_status = self._TwinSetParamByName(self._modelPointer, c_char_p(param_name), c_double(value))
        self.evaluate_twin_status(self.twin_status, self, "twin_set_param_by_name")

    def twin_set_str_param_by_name(self, param_name, value):
        if self.is_model_instantiated is False:
            raise TwinRuntimeError("The model has to be instantiated before setting parameters!")

        if isinstance(param_name, str):
            param_name = param_name.encode()

        if isinstance(value, str):
            value = value.encode()

        self.twin_status = self._TwinSetStrParamByName(self._modelPointer, c_char_p(param_name), c_char_p(value))
        self.evaluate_twin_status(self.twin_status, self, "twin_set_str_param_by_name")


    def twin_set_param_by_index(self, index, value):

        if self.is_model_instantiated is False:
            raise TwinRuntimeError("The model has to be instantiated before setting parameters!")

        self.twin_status = self._TwinSetParamByIndex(self._modelPointer, c_int(index), c_double(value))
        self.evaluate_twin_status(self.twin_status, self, "twin_set_param_by_index")

    def twin_set_input_by_name(self, input_name, value):
        if self.is_model_instantiated is False:
            raise TwinRuntimeError("The model has to be instantiated before setting inputs!")

        if isinstance(input_name, str):
            input_name = input_name.encode()

        self.twin_status = self._TwinSetInputByName(self._modelPointer, c_char_p(input_name), c_double(value))
        self.evaluate_twin_status(self.twin_status, self, "twin_set_input_by_name")

    def twin_set_input_by_index(self, index, value):
        if self.is_model_instantiated is False:
            raise TwinRuntimeError("The model has to be instantiated before setting inputs!")

        self.twin_status = self._TwinSetInputByIndex(self._modelPointer, c_int(index), c_double(value))
        self.evaluate_twin_status(self.twin_status, self, "twin_set_input_by_index")

    def twin_get_output_by_name(self, output_name):
        if self.is_model_initialized is False:
            raise TwinRuntimeError("The Model has to be initialized before it can return outputs!")

        value = c_double(0)
        self.twin_status = self._TwinGetOutputByName(self._modelPointer, c_char_p(output_name.encode()), byref(value))
        self.evaluate_twin_status(self.twin_status, self, "twin_get_output_by_name")
        return value

    def twin_get_output_by_index(self, index):
        if self.is_model_initialized is False:
            raise TwinRuntimeError("The Model has to be initialized before it can return outputs!")

        value = c_double(0)
        self.twin_status = self._TwinGetOutputByIndex(self._modelPointer, c_int(index), byref(value))
        self.evaluate_twin_status(self.twin_status, self, "twin_get_output_by_index")
        return value

    def twin_get_visualization_resources(self):
        visualization_info = c_char_p()
        self.twin_status = self._TwinGetVisualizationResources(self._modelPointer, byref(visualization_info))
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_visualization_resources')
        try:
            data = json.loads(visualization_info.value.decode().replace('\n', ''))
            return data
        except json.decoder.JSONDecodeError:
            return None

    def twin_get_default_rom_image_directory(self, model_name):
        if type(model_name) != bytes:
            model_name = model_name.encode()
        default_location = c_char_p()
        self.twin_status = self._TwinGetDefaultROMImageDirectory(self._modelPointer, c_char_p(model_name), byref(default_location))
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_default_rom_image_location')
        if default_location:
            return default_location.value.decode()

    def twin_set_rom_image_directory(self, model_name, directory_path):
        if type(model_name) != bytes:
            model_name = model_name.encode()
        if type(directory_path) != bytes:
            directory_path = directory_path.encode()
        self.twin_status = self._TwinSetROMImageDirectory(self._modelPointer, c_char_p(model_name), c_char_p(directory_path))
        self.evaluate_twin_status(self.twin_status, self, 'twin_set_rom_image_directory')

    def twin_enable_rom_model_images(self, model_name, views):
        n_views_c = c_int(len(views))
        array_ctypes = (c_char_p * len(views))()
        for ind, view_name in enumerate(views):
            array_ctypes[ind] = view_name.encode()

        self.twin_status = self._TwinEnableROMImages(self._modelPointer, c_char_p(model_name.encode()), array_ctypes,
                                                     n_views_c)
        self.evaluate_twin_status(self.twin_status, self, 'twin_enable_rom_model_image')

    def twin_disable_rom_model_images(self, model_name, views):
        n_views_c = c_int(len(views))
        array_ctypes = (c_char_p * len(views))()
        for ind, view_name in enumerate(views):
            array_ctypes[ind] = view_name.encode()

        self.twin_status = self._TwinDisableROMImages(self._modelPointer, c_char_p(model_name.encode()), array_ctypes,
                                                      n_views_c)
        self.evaluate_twin_status(self.twin_status, self, 'twin_disable_rom_model_images')

    def twin_get_rom_resource_directory(self, model_name):
        if type(model_name) != bytes:
            model_name = model_name.encode()
        ret = c_char_p()
        self.twin_status = self._TwinGetRomResourcePath(self._modelPointer, c_char_p(model_name), byref(ret))
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_rom_resource_directory')
        if ret:
            return ret.value.decode()

    def twin_enable_3d_rom_model_data(self, model_name):
        self.twin_status = self._TwinEnable3DROMData(self._modelPointer, c_char_p(model_name.encode()))
        self.evaluate_twin_status(self.twin_status, self, 'twin_enable_3d_rom_model_data')

    def twin_disable_3d_rom_model_data(self, model_name):
        self.twin_status = self._TwinDisable3DROMData(self._modelPointer, c_char_p(model_name.encode()))
        self.evaluate_twin_status(self.twin_status, self, 'twin_disable_3d_rom_model_data')

    def twin_get_rom_images_files(self, model_name, views, time_from=-1, time_to=-1):
        n_views_c = c_int(len(views))
        array_ctypes = (c_char_p * len(views))()
        for ind, view_name in enumerate(views):
            array_ctypes[ind] = view_name.encode()

        num_files_c = c_size_t()
        self.twin_status = self._TwinGetNumRomImageFiles(self._modelPointer,
                                                         c_char_p(model_name.encode()),
                                                         array_ctypes,
                                                         n_views_c,
                                                         byref(num_files_c),
                                                         c_double(time_from),
                                                         c_double(time_to)
                                                         )
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_rom_images_files')

        image_files_c = (c_char_p * num_files_c.value)()
        self.twin_status = self._TwinGetRomImageFiles(self._modelPointer,
                                                      c_char_p(model_name.encode()),
                                                      array_ctypes,
                                                      n_views_c,
                                                      byref(image_files_c),
                                                      c_double(time_from),
                                                      c_double(time_to)
                                                      )
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_rom_images_files')
        return to_np_array(image_files_c)

    def twin_get_rom_mode_coef_files(self, model_name, time_from=-1, time_to=-1):
        num_files_c = c_size_t()
        self.twin_status = self._TwinGetNumRomModeCoefFiles(self._modelPointer,
                                                            c_char_p(model_name.encode()),
                                                            byref(num_files_c),
                                                            c_double(time_from),
                                                            c_double(time_to)
                                                            )
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_rom_mode_coef_data')

        bin_files_c = (c_char_p * num_files_c.value)()
        self.twin_status = self._TwinGetRomModeCoefFiles(self._modelPointer,
                                                         c_char_p(model_name.encode()),
                                                         byref(bin_files_c),
                                                         c_double(time_from),
                                                         c_double(time_to)
                                                         )
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_rom_mode_coef_files')
        return to_np_array(bin_files_c)

    def twin_get_rom_snapshot_files(self, model_name, time_from=-1, time_to=-1):
        num_files_c = c_size_t()
        self.twin_status = self._TwinGetNumRomSnapshotFiles(self._modelPointer,
                                                            c_char_p(model_name.encode()),
                                                            byref(num_files_c),
                                                            c_double(time_from),
                                                            c_double(time_to)
                                                            )
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_rom_mode_coef_data')

        bin_files_c = (c_char_p * num_files_c.value)()
        self.twin_status = self._TwinGetRomSnapshotFiles(self._modelPointer,
                                                         c_char_p(model_name.encode()),
                                                         byref(bin_files_c),
                                                         c_double(time_from),
                                                         c_double(time_to)
                                                         )
        self.evaluate_twin_status(self.twin_status, self, 'twin_get_rom_snapshot_files')
        return to_np_array(bin_files_c)

    def twin_save_state(self, save_to):
        save_to = save_to.encode()
        self.twin_status = self._TwinSaveState(self._modelPointer, c_char_p(save_to))
        self.evaluate_twin_status(self.twin_status, self, 'twin_save_state')

    def twin_load_state(self, load_from):
        load_from = load_from.encode()
        try:
            self.twin_status = self._TwinLoadState(self._modelPointer, c_char_p(load_from))
            self.evaluate_twin_status(self.twin_status, self, 'twin_load_state')
        except OSError as err:
            msg = 'Fatal error when loading the model state'
            raise TwinRuntimeError(msg, self, TwinStatus.TWIN_STATUS_FATAL.value)


    """
    Status message retrieval
    Function for getting status of the last operation if the result is not TWIN_STATUS_OK.
    """
    def twin_get_status_string(self):
        return self.TwinGetStatusString(self._modelPointer).decode()

    # Returns runtime version
    def twin_get_api_version(self):
        return self._TwinGetAPIVersion(self._modelPointer).decode()

    """
    TwinRuntime Wrapper Helper functions 
    """
    def load_twin_default_sim_settings(self):
        if self.has_default_settings is False:
            self.p_end_time, self.p_step_size, self.p_tolerance = self.twin_get_default_simulation_settings()
            self.has_default_settings = True

    # pragma: no cover
    def print_model_info(self, max_var_to_print=np.inf):

        print("------------------------------------- Model Info -------------------------------------")
        print("Twin Runtime Version: {}".format(self.twin_get_api_version()))
        print("Model Name: {}".format(self.model_name.decode()))
        print("Number of outputs: {}".format(self.number_outputs))
        print("Number of Inputs: {}".format(self.number_inputs))
        print("Number of parameters: {}".format(self.number_parameters))
        print("Default time end: {}".format(self.p_end_time))
        print("Default step size: {}".format(self.p_step_size))
        print("Default tolerance(Integration Accuracy): {}".format(self.p_tolerance))
        print()
        print("Output names: ")
        self.print_var_info(self.twin_get_output_names(), max_var_to_print)

        print("Input names: ")
        self.print_var_info(self.twin_get_input_names(), max_var_to_print)

        print("Parameter names: ")
        self.print_var_info(self.twin_get_param_names(), max_var_to_print)

    def print_var_info(self, var_names, max_var_to_print):
        if max_var_to_print > len(var_names):
            max_var_to_print = None

        print(self.model_properties_info_df(var_names, max_var_to_print))

        if max_var_to_print == 0:
            print('{} items not shown.'.format(len(var_names)))
        elif max_var_to_print is not None:
            print('and {} more...'.format(len(var_names) - max_var_to_print))
        print("\n")

    def full_model_properties_info_df(self):

        prop_matrix_list = []

        input_vars = self.twin_get_input_names()
        prop_matrix_list += self.build_prop_info_df(input_vars)

        output_vars = self.twin_get_output_names()
        prop_matrix_list += self.build_prop_info_df(output_vars)

        param_vars = self.twin_get_param_names()
        prop_matrix_list += self.build_prop_info_df(param_vars)

        var_inf_columns = ['Name', 'Unit', 'Type', 'Start', 'Min', 'Max', 'Description']
        variable_info_df = pd.DataFrame(prop_matrix_list, columns=var_inf_columns)

        return variable_info_df

    def model_properties_info_df(self, vars_names, max_var_to_print):

        prop_matrix_list = self.build_prop_info_df(vars_names[:max_var_to_print])

        var_inf_columns = ['Name', 'Unit', 'Type', 'Start', 'Min', 'Max', 'Description']

        variable_info_df = pd.DataFrame(data=prop_matrix_list, columns=var_inf_columns)
        return variable_info_df

    def build_prop_info_df(self, var_names):
        prop_matrix_list = []
        for value in var_names:
            o_name = value
            try:
                o_unit = self.twin_get_var_unit(value)
            except (PropertyNotDefinedError, PropertyNotApplicableError, PropertyInvalidError, PropertyError) as e:
                o_unit = e.property_status_flag.name
            try:
                o_quantity_type = self.twin_get_var_quantity_type(value)
            except (PropertyNotDefinedError, PropertyNotApplicableError, PropertyInvalidError, PropertyError) as e:
                o_quantity_type = e.property_status_flag.name
            try:
                o_var_description = self.twin_get_var_description(value)
            except (PropertyNotDefinedError, PropertyNotApplicableError, PropertyInvalidError, PropertyError) as e:
                o_var_description = e.property_status_flag.name
            try:
                o_start = self.twin_get_var_start(value)
            except (PropertyNotDefinedError, PropertyNotApplicableError, PropertyInvalidError, PropertyError) as e:
                o_start = None
            try:
                o_min = self.twin_get_var_min(value)
            except (PropertyNotDefinedError, PropertyNotApplicableError, PropertyInvalidError, PropertyError) as e:
                o_min = None
            try:
                o_max = self.twin_get_var_max(value)
            except (PropertyNotDefinedError, PropertyNotApplicableError, PropertyInvalidError, PropertyError) as e:
                o_max = None

            prop_row = [o_name, o_unit, o_quantity_type, o_start, o_min, o_max, o_var_description]
            prop_matrix_list.append(prop_row)
        return prop_matrix_list


def build_empty_ctype_2d_array(num_input_rows, number_of_columns):
    row_elements = c_double * number_of_columns

    # The one liner below initializes each row if 'input_data' with an array of 'number_of_column' elements
    # 'row_elements()' creates one ctypes array for each input row
    input_data = (POINTER(c_double) * num_input_rows)(*[row_elements() for _ in range(num_input_rows)])

    return input_data


def build_ctype_2d_array(num_input_rows, source_df):
    input_data = (POINTER(c_double) * num_input_rows)()
    number_of_columns = source_df.shape[1]

    row_size = c_double * number_of_columns
    for i in range(num_input_rows):
        # Allocate arrays of double
        input_data[i] = row_size()
        for j in range(number_of_columns):
            input_data[i][j] = source_df.iat[i, j]

    return input_data


def to_np_array(ctypes_array):

    array_np = np.array([x.decode() for x in ctypes_array])

    return array_np



