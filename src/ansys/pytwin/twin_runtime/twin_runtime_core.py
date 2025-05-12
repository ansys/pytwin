# Copyright (C) 2022 - 2025 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ctypes import (
    POINTER,
    byref,
    c_bool,
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    c_void_p,
    cdll,
    create_string_buffer,
)
from enum import Enum
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Set, Tuple
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

if platform.system() == "Windows":
    import win32api

from .log_level import LogLevel
from .twin_runtime_error import (
    PropertyError,
    PropertyInvalidError,
    PropertyNotApplicableError,
    PropertyNotDefinedError,
    TwinRuntimeError,
)

CUR_DIR = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
default_log_name = "model.log"


class TwinStatus(Enum):
    TWIN_STATUS_OK = 0
    TWIN_STATUS_WARNING = 1
    TWIN_STATUS_ERROR = 2
    TWIN_STATUS_FATAL = 3


class FmiType(Enum):
    CS = 0
    ME = 1
    UNDEFINED = 2


class TwinRuntime:
    """
    Instantiate a TwinRuntime wrapper object based on a TWIN file created by
    Ansys Twin Builder.

    After a TwinRuntime object is instantiated, it can be used to call
    different APIs to manipulate and process the TWIN Runtime execution
    (e.g. initialization, setting up inputs, evaluating a time step,
    getting the outputs,...).

    Parameters
    ----------
    model_path : str
        File path to the TWIN file for the twin model.
    log_path : str (optional)
        File path to the log file associated to the TwinRuntime. By default,
        the log is written at the same location as the TWIN file.
    twin_runtime_library_path : str (optional)
        File path to the TWIN Runtime library. By default, it is located in a
        subfolder of the current working directory based on the OS
        (TwinRuntimeSDK.dll or libTwinRuntimeSDK.so).
    log_level : LogLevel (optional)
        Level option associated to the TWIN Runtime logging. By default, it is
        set to LogLevel.TWIN_LOG_WARNING.
    load_model : bool (optional)
        Whether the TWIN model is loaded (True) or not (False) during the
        TwinRuntime object instantiation. Default value is True.

    Examples
    --------
    Create the TwinRuntime given the file path to the TWIN file. Print the
    general information related to the TWIN model, then instantiate the model,
    initialize it, evaluate for a step and close the TwinRuntime.

    >>> from pytwin import TwinRuntime
    >>> twin_file = "model.twin"
    >>> twin_runtime = TwinRuntime(twin_file)
    >>> twin_runtime.twin_instantiate()
    >>> twin_runtime.twin_initialize()
    >>> twin_runtime.twin_simulate(0.001)
    >>> twin_runtime.twin_close()

    """

    _debug_mode = False
    _twin_status = None
    _is_model_opened = False
    _is_model_initialized = False
    _is_model_instantiated = False
    _last_time_stop = 0

    _model_name = None
    _number_parameters = None
    _number_inputs = None
    _number_outputs = None

    _has_default_settings = False
    _p_end_time = None
    _p_step_size = None
    _p_tolerance = None

    _output_names = None
    _input_names = None
    _parameter_names = None

    if platform.system() == "Windows":
        _twin_runtime_library = "TwinRuntimeSDK.dll"
    else:
        _twin_runtime_library = "libTwinRuntimeSDK.so"

    # def __getattribute__(self, name):
    #     attr = super().__getattribute__(name)
    #     if inspect.ismethod(attr):
    #         def hooked(*args, **kwargs):
    #             print(f"Calling method: {name}")
    #             result = attr(*args, **kwargs)
    #             print(f"Method {name} returned")
    #             return result
    #
    #         return hooked
    #     return attr

    @staticmethod
    def load_dll(twin_runtime_library_path=None):
        """
        Load the TwinRuntime library.

        Parameters
        ----------
        twin_runtime_library_path : str (optional)
            File path to the TWIN Runtime library. By default,
            it is located in a subfolder of the current working
            directory based on the OS
            (TwinRuntimeSDK.dll or libTwinRuntimeSDK.so).

        Returns
        -------
        ctypes.cdll
            The TwinRuntime loaded library in Python
        """

        def _setup_env(sdk_folder_path):
            if platform.system() == "Windows":
                sep = ";"
            else:
                sep = ":"
            if sdk_folder_path not in os.environ["PATH"]:
                os.environ["PATH"] = "{}{}{}".format(sdk_folder_path, sep, os.environ["PATH"])

        if twin_runtime_library_path is None:
            _setup_env(str(CUR_DIR))
            return cdll.LoadLibrary(os.path.join(str(CUR_DIR), TwinRuntime._twin_runtime_library))
        else:
            _setup_env(os.path.dirname(twin_runtime_library_path))
            return cdll.LoadLibrary(twin_runtime_library_path)

    @staticmethod
    def twin_get_api_version():
        """
        Returns the version of the Twin Runtime SDK being used.
        """
        twin_runtime_library = TwinRuntime.load_dll()
        get_api_version = twin_runtime_library.TwinGetAPIVersion
        get_api_version.restype = c_char_p
        return get_api_version().decode()

    @staticmethod
    def twin_is_cross_platform(file_path):
        """
        Returns whether the loaded TWIN model is cross-platform
        (Windows and Linux) compiled or not.

        Note that "zip_handler.namelist()" might return different contents
        depending on how the model archive was created. For example,
        FMUs and Twin models return the following list. Note that there
        are entries for binaries/linux64/ and binaries/win64/ folders.
          [
          'binaries/', 'documentation/', 'resources/', 'modelDescription.xml',
          'binaries/linux64/', 'binaries/win64/',
          'binaries/linux64/ModelWith_Min50_Max150.so',
          'binaries/win64/ModelWith_Min50_Max150.dll'
          ]

        For a .tbrom model, the following list is returned. Note the absence
        of binaries/linux64/ and binaries/win64/ folders.
        [
        'model.png', 'binaries/win64/rom24LP.dll',
        'binaries/win64/RomViewerSharedLib.dll', 'binaries/linux64/rom24LP.so',
        'binaries/linux64/RomViewerSharedLib.so', 'resources/properties.json',
        'resources/binaryOutputField/basis.svd',
        'resources/binaryOutputField/points.bin',
        'resources/binaryOutputField/settings.json',
        'resources/binaryOutputField/views.json',
        'resources/binaryOutputField/operationsDefinition.json',
        'resources/model.coreRom', 'modelDescription.xml']
        ]

        Parameters
        ----------
        file_path : str
            File path to the TWIN file for the twin model.

        Returns
        -------
        bool
            True if the TWIN model has binaries for Windows and Linux
        """
        with zipfile.ZipFile(file_path) as zip_handler:
            zip_contents = zip_handler.namelist()
            has_windows = any([name.startswith("binaries/win64/") for name in zip_contents])
            has_linux = any([name.startswith("binaries/linux64/") for name in zip_contents])

        return has_windows and has_linux

    @staticmethod
    def twin_number_of_deployments(file_path):
        """
        Returns the expected number of deployments for the given TWIN model
        as defined at the export time.

        Parameters
        ----------
        file_path : str
            File path to the TWIN model.

        Returns
        -------
        int
            Number of expected deployments for the TWIN model.
        """
        runtime_library = TwinRuntime.load_dll()
        TwinNumberOfDeployments = runtime_library.TwinGetNumberOfDeployments

        if type(file_path) is not bytes:
            file_path = file_path.encode()

        number_of_deployments = c_size_t()
        TwinNumberOfDeployments(c_char_p(file_path), byref(number_of_deployments))
        return number_of_deployments.value

    @staticmethod
    def get_model_fmi_type(file_path: str) -> Set[str]:
        """
        Searches the description file of the source model to discover if it
        contains Model Exchange and/or CoSimulation types of model.

        Parameters
        ----------
        file_path : str
            File path to the source model file (it could be a .twin, .fmu,
            or modelDescription.xml).

        Returns
        -------
        Set[str]
            'me' (Model Exchange) or 'cs' (Co Simulation) model or both.
        """

        def _parse_xml(model_description) -> Set[str]:
            tree = ET.parse(model_description)
            root = tree.getroot()

            available_fmi_types = set()
            co_simulation_tag = root.find("CoSimulation")
            if co_simulation_tag is not None:
                available_fmi_types.add("cs")
            model_exchange_tag = root.find("ModelExchange")
            if model_exchange_tag is not None:
                available_fmi_types.add("me")
            return available_fmi_types

        file_path = Path(file_path)
        if file_path.suffix in [".twin", ".fmu"]:
            with zipfile.ZipFile(file_path) as zip_handler:
                if "TwinDescription.xml" in zip_handler.namelist():
                    fmi_types = {"cs"}  # Twin models are all CS
                else:
                    with zip_handler.open("modelDescription.xml") as xml_file:
                        fmi_types = _parse_xml(xml_file)

        elif file_path.suffix == ".xml":
            fmi_types = _parse_xml(str(file_path))
        elif file_path.suffix == ".tbrom":
            raise TwinRuntimeError("Cannot read encrypted modelDescription.xml from .tbrom models")
        else:
            raise TwinRuntimeError("Unsupported file extension: " f"{file_path.suffix}")
        return fmi_types

    @staticmethod
    def get_model_name(file_path: str) -> str:
        """
        Reads the description file of the source model to discover
        the model name.

        Parameters
        ----------
        file_path : str
            File path to the source model file (it could be a .fmu
            or modelDescription.xml).

        Returns
        -------
        str
            Name of the model.
        """

        def _parse_xml(model_description) -> str:
            tree = ET.parse(model_description)
            root = tree.getroot()
            name = root.get("modelName")
            if name is None:
                raise TwinRuntimeError("Failed to find model name!")
            return name

        file_path = Path(file_path)
        if file_path.suffix in [".fmu"]:
            with zipfile.ZipFile(file_path) as zip_handler:
                with zip_handler.open("modelDescription.xml") as xml_file:
                    model_name = _parse_xml(xml_file)

        elif file_path.suffix == ".xml":
            model_name = _parse_xml(str(file_path))
        elif file_path.suffix in [".tbrom", ".twin"]:
            raise TwinRuntimeError("Cannot read encrypted description XML files from .tbrom or .twin models")
        else:
            raise TwinRuntimeError("Unsupported file extension: " f"{file_path.suffix}")
        return model_name

    @staticmethod
    def get_fmi_version(file_path: str) -> str:
        """
        Returns the FMI version described in the given model or XML file.

        Parameters
        ----------
        file_path : str
            File path to the model description XML file for the twin model.

        Returns
        -------
        str
            FMI version of the model
        """

        def _parse_xml(model_description) -> str:
            tree = ET.parse(model_description)
            root = tree.getroot()
            version = root.get("fmiVersion")
            if version is None:
                raise TwinRuntimeError("Failed to find model version!")
            return version

        file_path = Path(file_path)
        if file_path.suffix in [".fmu"]:
            with zipfile.ZipFile(file_path) as zip_handler:
                with zip_handler.open("modelDescription.xml") as xml_file:
                    model_fmi_version = _parse_xml(xml_file)

        elif file_path.suffix == ".xml":
            model_fmi_version = _parse_xml(str(file_path))
        elif file_path.suffix in [".tbrom", ".twin"]:
            raise TwinRuntimeError("Cannot read encrypted description XML files from .tbrom or .twin models")
        else:
            raise TwinRuntimeError("Unsupported file extension: " f"{file_path.suffix}")
        return model_fmi_version

    @staticmethod
    def is_fmu_supported(file_path) -> Tuple[bool, str]:
        """
        Returns whether the given FMI-based model is supported by the Twin SDK.

        Twin SDK currently only supports FMI 2.0 models

        Parameters
        ----------
        file_path : str
            File path to the model or model description XML file.

        Returns
        -------
        Tuple[bool, str]
            True if the FMU model is supported
            False with a message if the FMU model is not supported
        """

        version = TwinRuntime.get_fmi_version(file_path)
        if version == "2.0":
            return True, ""
        return False, f"FMI {version} models are not supported"

    @staticmethod
    def twin_platform_support(file_path):
        """
        Determines whether the TWIN model has Windows and/or Linux binaries.

        Note that "zip_handler.namelist()" might return different contents
        depending on how the model archive was created. For example,
        FMUs and Twin models return the following list. Note that there
        are entries for binaries/linux64/ and binaries/win64/ folders.
          [
          'binaries/', 'documentation/', 'resources/', 'modelDescription.xml',
          'binaries/linux64/', 'binaries/win64/',
          'binaries/linux64/ModelWith_Min50_Max150.so',
          'binaries/win64/ModelWith_Min50_Max150.dll'
          ]

        For a .tbrom model, the following list is returned. Note the absence
        of binaries/linux64/ and binaries/win64/ folders.
        [
        'model.png', 'binaries/win64/rom24LP.dll',
        'binaries/win64/RomViewerSharedLib.dll', 'binaries/linux64/rom24LP.so',
        'binaries/linux64/RomViewerSharedLib.so', 'resources/properties.json',
        'resources/binaryOutputField/basis.svd',
        'resources/binaryOutputField/points.bin',
        'resources/binaryOutputField/settings.json',
        'resources/binaryOutputField/views.json',
        'resources/binaryOutputField/operationsDefinition.json',
        'resources/model.coreRom', 'modelDescription.xml']
        ]

        Parameters
        ----------
        file_path : str
            File path to the TWIN model.

        Returns
        -------
        dict
            Dictionary indicating if Windows binaries are included (True) or
            not (False), and Linux binaries are included (True) or not (False).
        """
        with zipfile.ZipFile(file_path) as zip_handler:
            zip_contents = zip_handler.namelist()
            has_windows = any([name.startswith("binaries/win64/") for name in zip_contents])
            has_linux = any([name.startswith("binaries/linux64/") for name in zip_contents])

        return {"has_windows": has_windows, "has_linux": has_linux}

    @staticmethod
    def get_twin_version(file_path):
        """
        Returns whether the loaded TWIN model is a valid model or not, as well
        as the Twin Builder version used to compile it.

        Parameters
        ----------
        file_path : str
            File path to the TWIN model.

        Returns
        -------
        (bool, str)
            True if the TWIN model is a valid model, False otherwise.
            Twin Builder version used to compile it.
        """
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
        """
        Returns the list of associated dependencies to the TWIN model and the
        corresponding binaries found on the current environment where the TWIN
        is loaded. This method is supported only in a Linux OS.

        Parameters
        ----------
        file_path : str
            File path to the TWIN file for the twin model.

        Returns
        -------
        dict
            Dictionary of TWIN model's dependencies and the corresponding
            binaries found.
        """
        runtime_library = TwinRuntime.load_dll()

        TwinGetModelDependencies = runtime_library.TwinGetModelDependencies

        if type(file_path) is not bytes:
            file_path = file_path.encode()

        twin_dependencies = c_char_p()
        TwinGetModelDependencies(c_char_p(file_path), byref(twin_dependencies))
        twin_dependencies_dict = json.loads(twin_dependencies.value.decode())
        return twin_dependencies_dict

    @staticmethod
    def evaluate_twin_status(twin_status, twin_runtime, method_name):
        """
        Returns the current status message associated to the TWIN if its
        status is TWIN_STATUS_WARNING. Raises a TwinRuntimeError if the TWIN's
        status is TWIN_STATUS_ERROR or TWIN_STATUS_FATAL.

        Parameters
        ----------
        twin_status : int
            Current status of the TWIN model
        twin_runtime : TwinRuntime
            TwinRuntime instance associated to the TWIN model
        method_name : str
            Method executed when the TWIN's status is evaluated
        """
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
        """
        Returns the appropriate error message depending on prop_status when
        executing the function method_name with the variable var.

        Parameters
        ----------
        prop_status : int
            Status of the variable for which its properties are
            being evaluated.
        twin_runtime : TwinRuntime
            TwinRuntime instance associated to the TWIN model.
        method_name : str
            Method executed when the TWIN's status is evaluated.
        var : str
            TWIN model's variable name.
        """
        if prop_status == 4:
            message = f"The method {method_name.encode()}" f"with the variable {var} " "caused an error!\n"
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise PropertyError(message, twin_runtime, prop_status)

        elif prop_status == 3:
            message = (
                f"The method {method_name.encode()} "
                f"with the variable {var} "
                "is invalid (i.e., variable does not exist)!\n"
            )
            message += "TwinRuntime error message: " + twin_runtime.twin_get_status_string()
            raise PropertyInvalidError(message, twin_runtime, prop_status)

        elif prop_status == 2:
            message = f"The method {method_name.encode()} " f"with the variable {var} is not applicable!\n"
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise PropertyNotApplicableError(message, twin_runtime, prop_status)

        elif prop_status == 1:
            message = "The method {} with the variable {} is not defined!\n".format(method_name, var)
            message += "TwinRuntime error message :" + twin_runtime.twin_get_status_string()
            raise PropertyNotDefinedError(message, twin_runtime, prop_status)

    def __init__(
        self,
        model_path,
        log_path=None,
        twin_runtime_library_path=None,
        log_level=LogLevel.TWIN_LOG_WARNING,
        load_model=True,
        fmi_type=FmiType.UNDEFINED,
    ):
        model_path = Path(model_path)
        self.log_level = log_level

        # if model_path.is_file() is False:
        #     raise FileNotFoundError("File is not found at {}".format(model_path.absolute()))

        self._twin_runtime_library = TwinRuntime.load_dll(twin_runtime_library_path)

        if log_path is None:
            # Getting parent directory
            file_name = os.path.splitext(model_path)[0]
            log_path = file_name + ".log"

        self.model_path = str(model_path.absolute()).encode()
        self.log_path = str(Path(log_path)).encode()

        # Mapping sdk functions as class methods
        self._modelPointer = c_void_p()

        self._TwinOpen = self._twin_runtime_library.TwinOpen
        self._TwinOpen.argtypes = [c_char_p, c_void_p, c_char_p, c_int]
        self._TwinOpen.restype = c_int

        self._TwinOpenWithFmiType = self._twin_runtime_library.TwinOpenWithFmiType
        self._TwinOpenWithFmiType.argtypes = [c_char_p, c_void_p, c_char_p, c_int, c_int]
        self._TwinOpenWithFmiType.restype = c_int

        self._TwinClose = self._twin_runtime_library.TwinClose
        self._TwinClose.argtypes = [c_void_p]
        self._TwinClose.restype = None

        self._TwinReset = self._twin_runtime_library.TwinReset
        self._TwinReset.argtypes = [c_void_p]
        self._TwinReset.restype = c_int

        self.TwinGetStatusString = self._twin_runtime_library.TwinGetStatusString
        self.TwinGetStatusString.argtypes = [c_void_p]
        self.TwinGetStatusString.restype = c_char_p

        self._TwinGetModelName = self._twin_runtime_library.TwinGetModelName
        self._TwinGetModelName.argtypes = [c_void_p]
        self._TwinGetModelName.restype = c_char_p

        self._TwinGetNumParameters = self._twin_runtime_library.TwinGetNumParameters
        self._TwinGetNumParameters.argtypes = [c_void_p, POINTER(c_size_t)]
        self._TwinGetNumParameters.restype = c_int

        self._TwinGetNumInputs = self._twin_runtime_library.TwinGetNumInputs
        self._TwinGetNumInputs.argtypes = [c_void_p, POINTER(c_size_t)]
        self._TwinGetNumInputs.restype = c_int

        self._TwinGetNumOutputs = self._twin_runtime_library.TwinGetNumOutputs
        self._TwinGetNumOutputs.argtypes = [c_void_p, POINTER(c_size_t)]
        self._TwinGetNumOutputs.restype = c_int

        self._TwinGetParamNames = self._twin_runtime_library.TwinGetParamNames
        self._TwinGetParamNames.argtypes = [c_void_p, POINTER(c_char_p), c_size_t]
        self._TwinGetParamNames.restype = c_int

        self._TwinGetInputNames = self._twin_runtime_library.TwinGetInputNames
        self._TwinGetInputNames.argtypes = [c_void_p, POINTER(c_char_p), c_size_t]
        self._TwinGetInputNames.restype = c_int

        self._TwinGetOutputNames = self._twin_runtime_library.TwinGetOutputNames
        self._TwinGetOutputNames.argtypes = [c_void_p, POINTER(c_char_p), c_size_t]
        self._TwinGetOutputNames.restype = c_int

        self._TwinGetNumberOfDeployments = self._twin_runtime_library.TwinGetNumberOfDeploymentsFromInstance
        self._TwinGetNumberOfDeployments.argtypes = [c_void_p, POINTER(c_size_t)]
        self._TwinGetNumberOfDeployments.restype = c_int

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
        self._TwinGetOutputs.argtypes = [c_void_p, POINTER(c_double), c_size_t]
        self._TwinGetOutputs.restype = c_int

        self._TwinSimulate = self._twin_runtime_library.TwinSimulate
        self._TwinSimulate.argtypes = [c_void_p, c_double, c_double]
        self._TwinSimulate.restype = c_int

        self._TwinSimulateBatchMode = self._twin_runtime_library.TwinSimulateBatchMode
        self._TwinSimulateBatchMode.argtypes = [
            c_void_p,
            POINTER(POINTER(c_double)),
            c_size_t,
            POINTER(POINTER(c_double)),
            c_size_t,
            c_double,
            c_int,
        ]
        self._TwinSimulateBatchMode.restype = c_int

        self._TwinSimulateBatchModeCSV = self._twin_runtime_library.TwinSimulateBatchModeCSV
        self._TwinSimulateBatchModeCSV.argtypes = [c_void_p, c_char_p, c_char_p, c_double, c_int]
        self._TwinSimulateBatchModeCSV.restype = c_int

        self._TwinSetInputs = self._twin_runtime_library.TwinSetInputs
        self._TwinSetInputs.argtypes = [c_void_p, POINTER(c_double), c_size_t]
        self._TwinSetInputs.restype = c_int

        self._TwinSetInputByName = self._twin_runtime_library.TwinSetInputByName
        self._TwinSetInputByName.argtypes = [c_void_p, c_char_p, c_double]
        self._TwinSetInputByName.restype = c_int

        self._TwinSetInputByIndex = self._twin_runtime_library.TwinSetInputByIndex
        self._TwinSetInputByIndex.argtypes = [c_void_p, c_int, c_double]
        self._TwinSetInputByIndex.restype = c_int

        self._TwinGetOutputByName = self._twin_runtime_library.TwinGetOutputByName
        self._TwinGetOutputByName.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
        self._TwinGetOutputByName.restype = c_int

        self._TwinGetOutputByIndex = self._twin_runtime_library.TwinGetOutputByIndex
        self._TwinGetOutputByIndex.argtypes = [c_void_p, c_size_t, POINTER(c_double)]
        self._TwinGetOutputByIndex.restype = c_int

        self._TwinGetDefaultSimulationSettings = self._twin_runtime_library.TwinGetDefaultSimulationSettings
        self._TwinGetDefaultSimulationSettings.argtypes = [
            c_void_p,
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
        ]
        self._TwinGetDefaultSimulationSettings.restype = c_int

        self._TwinGetVarDataType = self._twin_runtime_library.TwinGetVarDataType
        self._TwinGetVarDataType.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
        self._TwinGetVarDataType.restype = c_int

        self._TwinGetVarUnit = self._twin_runtime_library.TwinGetVarUnit
        self._TwinGetVarUnit.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
        self._TwinGetVarUnit.restype = c_int

        self._TwinGetVarStart = self._twin_runtime_library.TwinGetVarStart
        self._TwinGetVarStart.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
        self._TwinGetVarStart.restype = c_int

        self._TwinGetStrVarStart = self._twin_runtime_library.TwinGetStrVarStart
        self._TwinGetStrVarStart.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
        self._TwinGetStrVarStart.restype = c_int

        self._TwinGetVarMin = self._twin_runtime_library.TwinGetVarMin
        self._TwinGetVarMin.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
        self._TwinGetVarMin.restype = c_int

        self._TwinGetVarMax = self._twin_runtime_library.TwinGetVarMax
        self._TwinGetVarMax.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
        self._TwinGetVarMax.restype = c_int

        self._TwinGetVarNominal = self._twin_runtime_library.TwinGetVarNominal
        self._TwinGetVarNominal.argtypes = [c_void_p, c_char_p, POINTER(c_double)]
        self._TwinGetVarNominal.restype = c_int

        self._TwinGetVarQuantityType = self._twin_runtime_library.TwinGetVarQuantityType
        self._TwinGetVarQuantityType.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
        self._TwinGetVarQuantityType.restype = c_int

        self._TwinGetVarDescription = self._twin_runtime_library.TwinGetVarDescription
        self._TwinGetVarDescription.argtypes = [c_void_p, c_char_p, POINTER(c_char_p)]
        self._TwinGetVarDescription.restype = c_int

        self._TwinGetVisualizationResources = self._twin_runtime_library.TwinGetVisualizationResources
        self._TwinGetVisualizationResources.argtypes = [c_void_p, POINTER(c_char_p)]
        self._TwinGetVisualizationResources.restype = c_int

        self._TwinEnableROMImages = self._twin_runtime_library.TwinEnableROMImages
        self._TwinEnableROMImages.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t]
        self._TwinEnableROMImages.restype = c_int

        self._TwinDisableROMImages = self._twin_runtime_library.TwinDisableROMImages
        self._TwinDisableROMImages.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t]
        self._TwinDisableROMImages.restype = c_int

        self._TwinEnable3DROMData = self._twin_runtime_library.TwinEnable3DROMData
        self._TwinEnable3DROMData.argtypes = [c_void_p, c_char_p]
        self._TwinEnable3DROMData.restype = c_int

        self._TwinDisable3DROMData = self._twin_runtime_library.TwinDisable3DROMData
        self._TwinDisable3DROMData.argtypes = [c_void_p, c_char_p]
        self._TwinDisable3DROMData.restype = c_int

        self._TwinGetRomImageFiles = self._twin_runtime_library.TwinGetRomImageFiles
        self._TwinGetRomImageFiles.argtypes = [
            c_void_p,
            c_char_p,
            POINTER(c_char_p),
            c_size_t,
            POINTER(c_char_p),
            c_double,
            c_double,
        ]
        self._TwinGetRomImageFiles.restype = c_int

        self._TwinGetNumRomImageFiles = self._twin_runtime_library.TwinGetNumRomImageFiles
        self._TwinGetNumRomImageFiles.argtypes = [
            c_void_p,
            c_char_p,
            POINTER(c_char_p),
            c_size_t,
            POINTER(c_size_t),
            c_double,
            c_double,
        ]
        self._TwinGetNumRomImageFiles.restype = c_int

        self._TwinGetRomModeCoefFiles = self._twin_runtime_library.TwinGetRomModeCoefFiles
        self._TwinGetRomModeCoefFiles.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_double, c_double]
        self._TwinGetRomModeCoefFiles.restype = c_int

        self._TwinGetNumRomModeCoefFiles = self._twin_runtime_library.TwinGetNumRomModeCoefFiles
        self._TwinGetNumRomModeCoefFiles.argtypes = [c_void_p, c_char_p, POINTER(c_size_t), c_double, c_double]
        self._TwinGetNumRomModeCoefFiles.restype = c_int

        self._TwinGetRomSnapshotFiles = self._twin_runtime_library.TwinGetRomSnapshotFiles
        self._TwinGetRomSnapshotFiles.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_double, c_double]
        self._TwinGetRomSnapshotFiles.restype = c_int

        self._TwinGetNumRomSnapshotFiles = self._twin_runtime_library.TwinGetNumRomSnapshotFiles
        self._TwinGetNumRomSnapshotFiles.argtypes = [c_void_p, c_char_p, POINTER(c_size_t), c_double, c_double]
        self._TwinGetNumRomSnapshotFiles.restype = c_int

        self._TwinGetDefaultROMImageDirectory = self._twin_runtime_library.TwinGetDefaultROMImageDirectory
        self._TwinGetDefaultROMImageDirectory.argtypes = [
            c_void_p,
            c_char_p,
            POINTER(c_char_p),
        ]
        self._TwinGetDefaultROMImageDirectory.restype = c_int

        self._TwinGetRomResourcePath = self._twin_runtime_library.TwinGetRomResourcePath
        self._TwinGetRomResourcePath.argtypes = [
            c_void_p,
            c_char_p,
            POINTER(c_char_p),
        ]
        self._TwinGetRomResourcePath.restype = c_int

        self._TwinGetRomOutputBasisSize = self._twin_runtime_library.TwinGetRomOutputBasisSize
        self._TwinGetRomOutputBasisSize.argtypes = [
            c_void_p,
            c_char_p,
            POINTER(c_size_t),
        ]
        self._TwinGetRomOutputBasisSize.restype = c_int

        self._TwinGetRomOutputBasis = self._twin_runtime_library.TwinGetRomOutputBasis
        self._TwinGetRomOutputBasis.argtypes = [
            c_void_p,
            c_char_p,
            POINTER(c_double),
            POINTER(c_size_t),
            POINTER(c_size_t),
        ]
        self._TwinGetRomOutputBasis.restype = c_int

        self._TwinGetRomInputBasisSize = self._twin_runtime_library.TwinGetRomInputBasisSize
        self._TwinGetRomInputBasisSize.argtypes = [
            c_void_p,
            c_char_p,
            c_char_p,
            POINTER(c_size_t),
        ]
        self._TwinGetRomInputBasisSize.restype = c_int

        self._TwinGetRomInputBasis = self._twin_runtime_library.TwinGetRomInputBasis
        self._TwinGetRomInputBasis.argtypes = [
            c_void_p,
            c_char_p,
            c_char_p,
            POINTER(c_double),
            POINTER(c_size_t),
            POINTER(c_size_t),
        ]
        self._TwinGetRomInputBasis.restype = c_int

        self._TwinSetROMImageDirectory = self._twin_runtime_library.TwinSetROMImageDirectory
        self._TwinSetROMImageDirectory.argtypes = [
            c_void_p,
            c_char_p,
            c_char_p,
        ]
        self._TwinSetROMImageDirectory.restype = c_int

        self._TwinSaveState = self._twin_runtime_library.TwinSaveState
        self._TwinSaveState.argtypes = [c_void_p, c_char_p]
        self._TwinSaveState.restype = c_int

        self._TwinLoadState = self._twin_runtime_library.TwinLoadState
        self._TwinLoadState.argtypes = [c_void_p, c_char_p, c_bool]
        self._TwinLoadState.restype = c_int

        self.model_path = Path(model_path).resolve()
        # if model_path.is_file() is False:
        #     raise FileNotFoundError("File is not found at {}".format(model_path.absolute()))

        if log_path is None:
            # Getting parent directory
            file_name = os.path.splitext(model_path)[0]
            log_path = file_name + ".log"

        self.log_path = Path(log_path).resolve()

        if load_model:
            self.twin_load(log_level, fmi_type)

    """
    Model opening/closing
    Functions for opening and closing a Twin model. Opening models is hidden
    within the constructor.
    """

    def twin_load(self, log_level, fmi_type: FmiType = FmiType.UNDEFINED):
        """
        Opens and loads a TWIN model, with a given log level for the log file.
        Client code can also specify the FMI type of the model. If the loaded
        model does not support the specified FMI type, an error is raised.

        Parameters
        ----------
        log_level : LogLevel
            Log level selected for the log file
            (LogLevel.TWIN_LOG_ALL, LogLevel.TWIN_LOG_WARNING,
             LogLevel.TWIN_LOG_ERROR, LogLevel.TWIN_LOG_FATAL,
             LogLevel.TWIN_NO_LOG).

        fmi_type : FmiType
            FMI type of the model (FmiType.CS, FmiType.ME, FmiType.UNDEFINED).
        """
        # This ensures that DLL loading mechanism gets reset to its default
        # behavior, which is altered when the SDK launches in Twin Deployer.
        # If this is not reset, some FMUs won't load because their dependent
        # DLLs (from the binaries/win64)  are not found.
        if platform.system() == "Windows":
            win32api.SetDllDirectory(None)
        file_buf = create_string_buffer(str(self.model_path).encode())
        log_buf = create_string_buffer(str(self.log_path).encode())
        self._twin_status = self._TwinOpenWithFmiType(
            file_buf, byref(self._modelPointer), log_buf, c_int(log_level.value), c_int(fmi_type.value)
        )

        self.evaluate_twin_status(self._twin_status, self, "twin_load")
        self._is_model_opened = True

        self._model_name = self._TwinGetModelName(self._modelPointer)

        self.twin_get_number_inputs()
        self.twin_get_number_outputs()
        self.twin_get_number_params()

        self.twin_get_param_names()
        self.twin_get_input_names()
        self.twin_get_output_names()
        self.load_twin_default_sim_settings()

    def twin_close(self):
        """
        Closes the TWIN model. After this call, the TwinRuntime instance is
        no longer valid. Only a new model instance of TwinRuntime can be
        created before using other function calls.
        """
        if not self._is_model_opened:
            print("[Warning]: twin_close() will not execute since model is not" "loaded. Maybe it was already closed?")
            return
        self._TwinClose(self._modelPointer)
        self._is_model_opened = False
        self._is_model_initialized = False
        self._is_model_instantiated = False
        self._last_time_stop = 0

        self._model_name = None
        self._number_parameters = None
        self._number_inputs = None
        self._number_outputs = None

        self._has_default_settings = None
        self._p_end_time = None
        self._p_step_size = None
        self._p_tolerance = None

        self._output_names = None
        self._input_names = None
        self._parameter_names = None

    """
    Model properties
    Functions for getting model properties
    """

    def twin_number_of_deployments_from_instance(self):
        """
        Returns the expected number of deployments for the current TWIN model
        instance as defined at the export time.

        Returns
        -------
        int
            Number of expected number of deployments for the TWIN model.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("The model has to be opened before returning " "the number of deployments!")
        c_number_deployments = c_size_t(0)
        self.twin_status = self._TwinGetNumberOfDeployments(self._modelPointer, byref(c_number_deployments))
        self.evaluate_twin_status(self.twin_status, self, "twin_get_number_of_deployments")
        self.number_outputs = c_number_deployments.value
        return c_number_deployments.value

    def twin_get_model_name(self):
        """
        Retrieves the name of the TWIN model.

        Returns
        -------
        str
            Name of the TWIN model
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning its name")

        return self._TwinGetModelName(self._modelPointer).decode()

    def twin_get_number_params(self):
        """
        Retrieves the number of parameters of the TWIN model.

        Returns
        -------
        int
            Number of parameters of the TWIN model
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning " "the number of parameters!")

        if self._number_parameters is None:
            c_number_params = c_size_t(0)
            self._twin_status = self._TwinGetNumParameters(self._modelPointer, byref(c_number_params))
            self.evaluate_twin_status(self._twin_status, self, "twin_get_number_params")
            self._number_parameters = c_number_params.value
            return self._number_parameters
        else:
            return self._number_parameters

    def twin_get_number_inputs(self):
        """
        Retrieves the number of inputs of the TWIN model.

        Returns
        -------
        int
            Number of inputs of the TWIN model
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning " "the number of inputs!")

        if self._number_inputs is None:
            c_number_inputs = c_size_t(0)
            self._twin_status = self._TwinGetNumInputs(self._modelPointer, byref(c_number_inputs))
            self.evaluate_twin_status(self._twin_status, self, "twin_get_number_inputs")
            self._number_inputs = c_number_inputs.value
            return self._number_inputs
        else:
            return self._number_inputs

    def twin_get_number_outputs(self):
        """
        Retrieves the number of outputs of the TWIN model.

        Returns
        -------
        int
            Number of outputs of the TWIN model
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning " "the number of outputs!")

        if self._number_outputs is None:
            c_number_outputs = c_size_t(0)
            self._twin_status = self._TwinGetNumOutputs(self._modelPointer, byref(c_number_outputs))
            self.evaluate_twin_status(self._twin_status, self, "twin_get_number_outputs")
            self._number_outputs = c_number_outputs.value
            return self._number_outputs
        else:
            return self._number_outputs

    def twin_get_param_names(self):
        """
        Retrieves the names of parameters of the TWIN model.

        Returns
        -------
        list
            List of names of TWIN parameters
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning parameter names!")

        if self._parameter_names is None:
            self._TwinGetParamNames.argtypes = [
                c_void_p,
                POINTER(c_char_p * self._number_parameters),
                c_int,
            ]

            parameter_names_c = (c_char_p * self._number_parameters)()

            self._twin_status = self._TwinGetParamNames(self._modelPointer, parameter_names_c, self._number_parameters)
            self.evaluate_twin_status(self._twin_status, self, "twin_get_param_names")

            self._parameter_names = to_np_array(parameter_names_c)

            return self._parameter_names
        else:
            return self._parameter_names

    def twin_get_input_names(self):
        """
        Retrieves the names of inputs of the TWIN model.

        Returns
        -------
        list
            List of names of TWIN inputs
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning input names!")

        if self._input_names is None:
            self._TwinGetInputNames.argtypes = [
                c_void_p,
                POINTER(c_char_p * self._number_inputs),
                c_int,
            ]

            input_names_c = (c_char_p * self._number_inputs)()

            self._twin_status = self._TwinGetInputNames(self._modelPointer, input_names_c, self._number_inputs)
            self.evaluate_twin_status(self._twin_status, self, "twin_get_input_names")

            self._input_names = to_np_array(input_names_c)

            return self._input_names
        else:
            return self._input_names

    def twin_get_output_names(self):
        """
        Retrieves the names of outputs of the TWIN model.

        Returns
        -------
        list
            List of names of TWIN outputs
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning output names!")

        if self._output_names is None:
            self._TwinGetInputNames.argtypes = [
                c_void_p,
                POINTER(c_char_p * self._number_outputs),
                c_int,
            ]

            output_names_c = (c_char_p * self._number_outputs)()

            self._twin_status = self._TwinGetOutputNames(self._modelPointer, output_names_c, self._number_outputs)
            self.evaluate_twin_status(self._twin_status, self, "twin_get_output_names")

            self._output_names = to_np_array(output_names_c)

            return self._output_names
        else:
            return self._output_names

    def twin_get_default_simulation_settings(self):
        """
        Retrieves the default simulation settings
        (end time, step size and tolerance) associated with the TWIN.

        Returns
        -------
        list
            A list of float representing the end time, step size
            and tolerance values
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning default settings!")

        c_end_time = c_double(0)
        c_step_size = c_double(0)
        c_tolerance = c_double(0)

        self._twin_status = self._TwinGetDefaultSimulationSettings(
            self._modelPointer,
            byref(c_end_time),
            byref(c_step_size),
            byref(c_tolerance),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_default_sim_settings")

        return c_end_time.value, c_step_size.value, c_tolerance.value

    """
    Model variable properties
    Functions for getting specific property for a given model variable.
    """

    def twin_get_var_data_type(self, var_name):
        """
        Retrieves the data type of a given variable
        ("Real", "Integer", "Boolean", "Enumeration", or String) by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        str
            Data type of the given variable returned as string.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable data type!")

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
        """
        Retrieves the physical quantity type such as pressure,
        temperature, etc. for a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        str
            Physical quantity of the given variable returned as string.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable quantity type!")

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
        """
        Retrieves the description string for a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        str
            Description of the given variable returned as string.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable description!")

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
        """
        Retrieves the unit string of a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        str
            Unit of the given variable returned as string.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable unit type!")

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
        """
        Retrieves the start value of a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        float
            Start value of the given variable.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable start value!")

        start_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarStart(self._modelPointer, c_char_p(var_name), byref(start_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_start", var_name)

        return start_value.value

    def twin_get_str_var_start(self, var_name):
        """
        Retrieves the start value of a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        str
            Start value of the given variable returned as string.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable start value!")

        start_value = c_char_p()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetStrVarStart(self._modelPointer, c_char_p(var_name), byref(start_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_str_var_start", var_name)

        return start_value.value.decode()

    def twin_get_var_min(self, var_name):
        """
        Retrieves the minimum value of a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        float
            Minimum value of the given variable.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable minimum value!")

        min_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarMin(self._modelPointer, c_char_p(var_name), byref(min_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_min", var_name)

        return min_value.value

    def twin_get_var_max(self, var_name):
        """
        Retrieves the maximum value of a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        float
            Maximum value of the given variable.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable maximum value!")

        max_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarMax(self._modelPointer, c_char_p(var_name), byref(max_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_max", var_name)

        return max_value.value

    def twin_get_var_nominal(self, var_name):
        """
        Retrieves the nominal value of a given variable by name.

        Parameters
        ----------
        var_name: str
            Name of the variable.

        Returns
        -------
        float
            Nominal value of the given variable.
        """
        if self._is_model_opened is False:
            raise TwinRuntimeError("Model must be opened before returning variable nominal value!")

        nominal_value = c_double()
        if type(var_name) is not bytes:
            var_name = var_name.encode()

        property_status = self._TwinGetVarNominal(self._modelPointer, c_char_p(var_name), byref(nominal_value))
        self.evaluate_twin_prop_status(property_status, self, "twin_get_var_nominal", var_name)

        return nominal_value.value

    """
    Simulation operations
    Functions for simulating the Twin model.
    """

    def twin_instantiate(self):
        """
        Instantiates the TWIN model.

        """
        self._twin_status = self._TwinInstantiate(self._modelPointer)
        self.evaluate_twin_status(self._twin_status, self, "twin_instantiate")
        self._is_model_instantiated = True

    def twin_initialize(self):
        """
        Initializes the TWIN model. Must be called after twin_instantiate().

        """
        if self._is_model_instantiated is False:
            raise TwinRuntimeError("Model must be instantiated before initialization!")

        try:
            self._twin_status = self._TwinInitialize(self._modelPointer)
        except OSError:
            message = (
                "Error while initializing the model. " "This model may need start values or has other dependencies."
            )
            raise TwinRuntimeError(message)

        self.evaluate_twin_status(self._twin_status, self, "twin_initialize")
        self._is_model_initialized = True

    def twin_simulate(self, time_stop, time_step=0):
        """
        Simulates the TWIN model from previous time point to the
        stop point given by time_stop.

        Parameters
        ----------
        time_stop : float
            Stop time.
        time_step : float (optional)
            Step size. If the value is 0, only one stepping call will be
            performed such that the model will be stepped from previous stop
            point to the given point in one shot (internally the model can
            take smaller time steps for numerical integration); otherwise,
            it will perform multiple steps with the step size of h.
            Default is 0.
        """
        if self._is_model_initialized is False:
            raise TwinRuntimeError("Model must be initialized before simulation!")

        self._twin_status = self._TwinSimulate(self._modelPointer, c_double(time_stop), c_double(time_step))
        self.evaluate_twin_status(self._twin_status, self, "twin_simulate")

    def twin_simulate_batch_mode(
        self,
        input_df,
        output_column_names,
        step_size=0,
        interpolate=0,
        time_as_index=False,
    ):
        """
        Simulates the TWIN model in batch mode using given input dataframe and
        returns the results in an output dataframe using output column names.

        Parameters
        ----------
        input_df : pandas.DataFrame
            Pandas dataframe storing all the TWIN inputs to be evaluated of
            the batch simulation.
        output_column_names : list
            List of string describing the different output columns name
            (including 'Time' as first column).
        step_size : float (optional)
            Step size. If 0, time points in the input table will be used as
            the output points; otherwise it will produce
            output at an equal spacing of h. Default is 0.
        interpolate : int (optional)
            Flag to interpolate real continuous variables if step size > 0.
        time_as_index : bool (optional)
            Flag to reset the input_df index if set to True. Default is False.

        Returns
        -------
        pandas.DataFrame
            Pandas dataframe storing all the TWIN outputs evaluated over
            the batch simulation.
        """
        output_number_of_columns = self._number_outputs + 1

        if self._is_model_initialized is False:
            raise TwinRuntimeError("Model must be initialized before simulation!")

        # Creates a local copy so that the source DF does not
        # get modified outside this scope
        local_df = input_df
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

        self._twin_status = self._TwinSimulateBatchMode(
            self._modelPointer,
            input_data,
            c_size_t(num_input_rows),
            out_data,
            c_size_t(max_output_rows),
            c_double(step_size),
            c_int(interpolate),
        )
        data = [np.ctypeslib.as_array(out_data[i], shape=(output_number_of_columns,)) for i in range(max_output_rows)]
        output_df = pd.DataFrame(
            data=data,
            index=np.arange(0, max_output_rows),
            columns=output_column_names,
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_simulate_batch_mode")

        return output_df

    # This method will generate the response also as a csv
    def twin_simulate_batch_mode_csv(self, input_csv, output_csv, step_size=0, interpolate=0):
        """
        Simulates the TWIN model in batch mode using given input CSV file and
        write the results in the output CSV file.

        Parameters
        ----------
        input_csv : str
            Input CSV file. First column represents time and the next ones
            represent inputs. Header is optional.
        output_csv : str
            Output CSV file. If empty or NULL no output will be generated.
            First column represents time and the next ones represent outputs.
        step_size : float (optional)
            Step size. If 0, time points in the input table will be used as
            the output points; otherwise it will produce output at an equal
            spacing of h. Default is 0.
        interpolate : int (optional)
            Flag to interpolate real continuous variables if step size > 0.

        """
        if self._is_model_initialized is False:
            raise TwinRuntimeError("Model must be initialized before simulation!")

        if type(input_csv) is not bytes:
            input_csv = input_csv.encode()
        if type(output_csv) is not bytes:
            output_csv = output_csv.encode()

        input_csv = create_string_buffer(input_csv)
        output_csv = create_string_buffer(output_csv)

        self._twin_status = self._TwinSimulateBatchModeCSV(
            self._modelPointer,
            input_csv,
            output_csv,
            c_double(step_size),
            c_int(interpolate),
        )

        self.evaluate_twin_status(self._twin_status, self, "twin_simulate_batch_mode_csv")

    def twin_reset(self):
        """
        Resets the state of the TWIN model back to the instantiated state.

        """
        self._twin_status = self._TwinReset(self._modelPointer)
        self.evaluate_twin_status(self._twin_status, self, "twin_reset")

    """
    Input/output handling
    Functions for setting parameters/inputs and getting outputs.
    """

    def twin_set_inputs(self, input_array):
        """
        Set the current value of all the TWIN inputs.

        Parameters
        ----------
        input_array : list
            List of inputs value.
        """
        if self._is_model_instantiated is False:
            raise TwinRuntimeError("Model must be instantiated before setting inputs!")

        if len(input_array) != self._number_inputs:
            raise TwinRuntimeError("Input array size must match the the models number of inputs!")

        array_np = np.array(input_array, dtype=float)
        array_ctypes = array_np.ctypes.data_as(POINTER(c_double * self._number_inputs))

        self._TwinSetInputs.argtypes = [
            c_void_p,
            POINTER(c_double * self._number_inputs),
            c_int,
        ]
        self._twin_status = self._TwinSetInputs(self._modelPointer, array_ctypes, self._number_inputs)
        self.evaluate_twin_status(self._twin_status, self, "twin_get_outputs")

    def twin_get_outputs(self):
        """
        Retrieves the current value of all the TWIN outputs.

        Returns
        -------
        list
            List of outputs value.
        """
        if self._is_model_initialized is False:
            raise TwinRuntimeError("Model must be initialized before it can return outputs!")

        self._TwinGetOutputs.argtypes = [
            c_void_p,
            POINTER(c_double * self._number_outputs),
            c_int,
        ]
        outputs = (c_double * self._number_outputs)()

        self._twin_status = self._TwinGetOutputs(self._modelPointer, outputs, self._number_outputs)
        self.evaluate_twin_status(self._twin_status, self, "twin_get_outputs")

        outputs_list = np.array(outputs).tolist()
        return outputs_list

    def twin_set_param_by_name(self, param_name, value):
        """
        Set the current value of a single TWIN parameter specified by name.

        Parameters
        ----------
        param_name : str
            Parameter name.
        value : float
            Parameter value.
        """
        if self._is_model_instantiated is False:
            raise TwinRuntimeError("Model must be instantiated before setting parameters!")

        if isinstance(param_name, str):
            param_name = param_name.encode()

        self._twin_status = self._TwinSetParamByName(self._modelPointer, c_char_p(param_name), c_double(value))
        self.evaluate_twin_status(self._twin_status, self, "twin_set_param_by_name")

    def twin_set_str_param_by_name(self, param_name, value):
        """
        Set the value of a single TWIN string parameter specified by name.

        Parameters
        ----------
        param_name : str
            Parameter name.
        value : float
            Parameter value.
        """
        if self._is_model_instantiated is False:
            raise TwinRuntimeError("Model must be instantiated before setting parameters!")

        if isinstance(param_name, str):
            param_name = param_name.encode()

        if isinstance(value, str):
            value = value.encode()

        self._twin_status = self._TwinSetStrParamByName(self._modelPointer, c_char_p(param_name), c_char_p(value))
        self.evaluate_twin_status(self._twin_status, self, "twin_set_str_param_by_name")

    def twin_set_param_by_index(self, index, value):
        """
        Set the current value of a single TWIN parameter specified by index.

        Parameters
        ----------
        index : int
            Parameter index.
        value : float
            Parameter value.
        """

        if self._is_model_instantiated is False:
            raise TwinRuntimeError("Model must be instantiated before setting parameters!")

        self._twin_status = self._TwinSetParamByIndex(self._modelPointer, c_int(index), c_double(value))
        self.evaluate_twin_status(self._twin_status, self, "twin_set_param_by_index")

    def twin_set_input_by_name(self, input_name, value):
        """
        Set the current value of a single TWIN input specified by name.

        Parameters
        ----------
        input_name : str
            Input name.
        value : float
            Input value.
        """
        if self._is_model_instantiated is False:
            raise TwinRuntimeError("Model must be instantiated before setting inputs!")

        if isinstance(input_name, str):
            input_name = input_name.encode()

        self._twin_status = self._TwinSetInputByName(self._modelPointer, c_char_p(input_name), c_double(value))
        self.evaluate_twin_status(self._twin_status, self, "twin_set_input_by_name")

    def twin_set_input_by_index(self, index, value):
        """
        Set the current value of a single TWIN input specified by index.

        Parameters
        ----------
        index : int
            Input index.
        value : float
            Input value.
        """
        if self._is_model_instantiated is False:
            raise TwinRuntimeError("Model must be instantiated before setting inputs!")

        self._twin_status = self._TwinSetInputByIndex(self._modelPointer, c_int(index), c_double(value))
        self.evaluate_twin_status(self._twin_status, self, "twin_set_input_by_index")

    def twin_get_output_by_name(self, output_name):
        """
        Retrieves the current value of a single TWIN output specified by name.

        Returns
        -------
        float
            Output value.
        """
        if self._is_model_initialized is False:
            raise TwinRuntimeError("Model must be initialized before it can return outputs!")

        value = c_double(0)
        self._twin_status = self._TwinGetOutputByName(self._modelPointer, c_char_p(output_name.encode()), byref(value))
        self.evaluate_twin_status(self._twin_status, self, "twin_get_output_by_name")
        return value

    def twin_get_output_by_index(self, index):
        """
        Retrieves the current value of a single TWIN output specified by index.

        Returns
        -------
        float
            Output value.
        """
        if self._is_model_initialized is False:
            raise TwinRuntimeError("Model must be initialized before it can return outputs!")

        value = c_double(0)
        self._twin_status = self._TwinGetOutputByIndex(self._modelPointer, c_int(index), byref(value))
        self.evaluate_twin_status(self._twin_status, self, "twin_get_output_by_index")
        return value

    def twin_get_visualization_resources(self):
        """
        Retrieves a JSON-like data structure in string format with the
        information about model visualization resources available in the
        TWIN model. This method is only supported for Twin models created
        from one or more TBROM components.

        Returns
        -------
        str
            Information about TBROM models visualization resources included
            in the TWIN. Example of output:

            .. code-block:: python

                {
                    "myTBROM_1": {
                        "type": "image,3D",
                        "modelname": "myTBROM",
                        "views": {"View1": "View1"},
                        "trigger": {"field_data_storage": "field_data_storage"},
                        "inputfields": ["inputPressure", "inputTemperature"],
                    }
                }
        """
        visualization_info = c_char_p()
        self._twin_status = self._TwinGetVisualizationResources(self._modelPointer, byref(visualization_info))
        self.evaluate_twin_status(self._twin_status, self, "twin_get_visualization_resources")
        try:
            data = json.loads(visualization_info.value.decode().replace("\n", ""))
            return data
        except json.decoder.JSONDecodeError:
            return None

    def twin_get_default_rom_image_directory(self, model_name):
        """
        Retrieves the default directory in the local filesystem where ROM
        images will be saved for the TBROM model name. This method is only
        supported for Twin models created from one or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM.

        Returns
        -------
        str
            Absolute path to the resources directory.
        """
        if type(model_name) is not bytes:
            model_name = model_name.encode()
        default_location = c_char_p()
        self._twin_status = self._TwinGetDefaultROMImageDirectory(
            self._modelPointer, c_char_p(model_name), byref(default_location)
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_default_rom_image_location")
        if default_location:
            return default_location.value.decode()

    def twin_set_rom_image_directory(self, model_name, directory_path):
        """
        Set the directory in the local filesystem where ROM images will be
        saved for the TBROM model name. This method is only supported for Twin
        models created from one or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM.
        directory_path : str
            Absolute path of the directory where to store the images.
        """
        if type(model_name) is not bytes:
            model_name = model_name.encode()
        if type(directory_path) is not bytes:
            directory_path = directory_path.encode()
        self._twin_status = self._TwinSetROMImageDirectory(
            self._modelPointer, c_char_p(model_name), c_char_p(directory_path)
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_set_rom_image_directory")

    def twin_enable_rom_model_images(self, model_name, views):
        """
        Enables the ROM image generation for the given model name and views in
        the next time steps (until disabled). If the image generation is
        already enabled, behavior remains unchanged. This method is only
        supported for Twin models created from one or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the image generation needs
            to be enabled.
        views : list
            View names for which the image generation needs to be enabled.
        """
        n_views_c = c_size_t(len(views))
        array_ctypes = (c_char_p * len(views))()
        for ind, view_name in enumerate(views):
            array_ctypes[ind] = view_name.encode()

        self._twin_status = self._TwinEnableROMImages(
            self._modelPointer,
            c_char_p(model_name.encode()),
            array_ctypes,
            n_views_c,
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_enable_rom_model_image")

    def twin_disable_rom_model_images(self, model_name, views):
        """
        Disables the ROM image generation for the given model name and views
        in the next time steps. If the image generation is already disabled,
        behavior remains unchanged. This method is only supported for Twin
        models created from one or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the image generation needs
            to be disabled.
        views : list
            View names for which the image generation needs to be disabled.
        """
        n_views_c = c_size_t(len(views))
        array_ctypes = (c_char_p * len(views))()
        for ind, view_name in enumerate(views):
            array_ctypes[ind] = view_name.encode()

        self._twin_status = self._TwinDisableROMImages(
            self._modelPointer,
            c_char_p(model_name.encode()),
            array_ctypes,
            n_views_c,
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_disable_rom_model_images")

    def twin_get_rom_resource_directory(self, model_name):
        """
        Retrieves the absolute path of the resource directory for the
        given TBROM model name. This method is only supported for Twin
        models created from one or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the resource directory needs
            to be retrieved.

        Returns
        -------
        str
            Absolute path to the resources' directory of the TBROM model.
        """
        if type(model_name) is not bytes:
            model_name = model_name.encode()
        ret = c_char_p()
        self._twin_status = self._TwinGetRomResourcePath(self._modelPointer, c_char_p(model_name), byref(ret))
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_resource_directory")
        if ret:
            return ret.value.decode()

    def twin_get_rom_output_basis(self, model_name):
        """
        Retrieve the output field basis for the given TBROM model name.
        This method is only supported for Twin models created from one or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the basis needs to be retrieved.

        Returns
        -------
        basis : np.ndarray
            SVD basis
        modes : int
            number of modes
        size : int
            field size
        """
        if type(model_name) != bytes:
            model_name = model_name.encode()
        c_basis_size = c_size_t()
        self._twin_status = self._TwinGetRomOutputBasisSize(
            self._modelPointer, c_char_p(model_name), byref(c_basis_size)
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_output_basis")

        basis = (c_double * c_basis_size.value)()
        c_nb_modes = c_size_t()
        c_field_size = c_size_t()
        self._twin_status = self._TwinGetRomOutputBasis(
            self._modelPointer, c_char_p(model_name), byref(basis), byref(c_nb_modes), byref(c_field_size)
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_output_basis")
        np_basis = np.array([x for x in basis])

        if basis:
            return np_basis, c_nb_modes.value, c_field_size.value

    def twin_get_rom_input_basis(self, model_name, field_name):
        """
        Retrieve the input field basis for the given TBROM model name and field name.
        This method is only supported for Twin models created from one or more TBROM components,
        and having input fields.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the basis needs to be retrieved.
        field_name : str
            Input field name of the TBROM for which the basis needs to be retrieved.

        Returns
        -------
        basis : np.ndarray
            SVD basis
        modes : int
            number of modes
        size : int
            field size
        """
        if type(model_name) != bytes:
            model_name = model_name.encode()
        if type(field_name) != bytes:
            field_name = field_name.encode()
        c_basis_size = c_size_t()
        self._twin_status = self._TwinGetRomInputBasisSize(
            self._modelPointer, c_char_p(model_name), c_char_p(field_name), byref(c_basis_size)
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_input_basis")

        basis = (c_double * c_basis_size.value)()
        c_nb_modes = c_size_t()
        c_field_size = c_size_t()
        self._twin_status = self._TwinGetRomInputBasis(
            self._modelPointer,
            c_char_p(model_name),
            c_char_p(field_name),
            byref(basis),
            byref(c_nb_modes),
            byref(c_field_size),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_input_basis")
        np_basis = np.array([x for x in basis])

        if basis:
            return np_basis, c_nb_modes.value, c_field_size.value

    def twin_enable_3d_rom_model_data(self, model_name):
        """
        Enables the generation of 3D data (mode coefficients and optionally
        snapshots files) for the given model name in the next time steps.
        If the 3D generation is already enabled, behavior remains unchanged.
        This method is only supported for Twin models created from one or more
        TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which 3D data generation needs
            to be enabled.
        """
        self._twin_status = self._TwinEnable3DROMData(self._modelPointer, c_char_p(model_name.encode()))
        self.evaluate_twin_status(self._twin_status, self, "twin_enable_3d_rom_model_data")

    def twin_disable_3d_rom_model_data(self, model_name):
        """
        Disables the generation of 3D data (mode coefficients and optionally
        snapshots files) for the given model name in the next time steps.
        If the 3D generation is already disabled, behavior remains unchanged.
        This method is only supported for Twin models created from one or
        more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which 3D data generation needs
            to be disabled.
        """
        self._twin_status = self._TwinDisable3DROMData(self._modelPointer, c_char_p(model_name.encode()))
        self.evaluate_twin_status(self._twin_status, self, "twin_disable_3d_rom_model_data")

    def twin_get_rom_images_files(self, model_name, views, time_from=-1, time_to=-1):
        """
        Retrieves the model images from 'time_from' up to 'time_to' for the
        given views from the given TBROM model name. By default, it returns
        the images for the current simulation step (for step-by-step
        simulation) or for all previous steps (for batch model simulation).
        This method is only supported for Twin models created from one
        or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the images need to be retrieved.
        views : list
            View names for which the images need to be retrieved.
        time_from : float (optional)
            Time stamp from which images need to be retrieved.
        time_to : float (optional)
            Time stamp up to which images need to be retrieved.

        Returns
        -------
        list
            List of path of all the images retrieved
        """
        n_views_c = c_size_t(len(views))
        array_ctypes = (c_char_p * len(views))()
        for ind, view_name in enumerate(views):
            array_ctypes[ind] = view_name.encode()

        num_files_c = c_size_t()
        self._twin_status = self._TwinGetNumRomImageFiles(
            self._modelPointer,
            c_char_p(model_name.encode()),
            array_ctypes,
            n_views_c,
            byref(num_files_c),
            c_double(time_from),
            c_double(time_to),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_images_files")

        image_files_c = (c_char_p * num_files_c.value)()
        self._twin_status = self._TwinGetRomImageFiles(
            self._modelPointer,
            c_char_p(model_name.encode()),
            array_ctypes,
            n_views_c,
            image_files_c,
            c_double(time_from),
            c_double(time_to),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_images_files")
        return to_np_array(image_files_c)

    def twin_get_rom_mode_coef_files(self, model_name, time_from=-1, time_to=-1):
        """
        Retrieves the model mode coefficients files from 'time_from' up to
        'time_to' for the given TBROM model name. By default, it returns the
        mode coefficients files for the current simulation step (for
        step-by-step simulation) or for all previous steps (for batch
        model simulation). This method is only supported for Twin models
        created from one or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the mode coefficient files
            need to be retrieved.
        time_from : float (optional)
            Time stamp from which the mode coefficient files need
            to be retrieved.
        time_to : float (optional)
            Time stamp up to which the mode coefficient files need
            to be retrieved.

        Returns
        -------
        list
            List of path of all the mode coefficients files retrieved
        """
        num_files_c = c_size_t()
        self._twin_status = self._TwinGetNumRomModeCoefFiles(
            self._modelPointer,
            c_char_p(model_name.encode()),
            byref(num_files_c),
            c_double(time_from),
            c_double(time_to),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_mode_coef_data")

        bin_files_c = (c_char_p * num_files_c.value)()
        self._twin_status = self._TwinGetRomModeCoefFiles(
            self._modelPointer,
            c_char_p(model_name.encode()),
            bin_files_c,
            c_double(time_from),
            c_double(time_to),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_mode_coef_files")
        return to_np_array(bin_files_c)

    def twin_get_rom_snapshot_files(self, model_name, time_from=-1, time_to=-1):
        """
        Retrieves the model snapshots files from 'time_from' up to 'time_to'
        for the given TBROM model name. By default, it returns the snapshots
        files for the current simulation step (for step-by-step simulation)
        or for all previous steps (for batch model simulation).
        This method is only supported for Twin models created from one
        or more TBROM components.

        Parameters
        ----------
        model_name : str
            Model name of the TBROM for which the snapshot files
            need to be retrieved.
        time_from : float (optional)
            Time stamp from which the snapshot files need to be retrieved.
        time_to : float (optional)
            Time stamp up to which the snapshot files need to be retrieved.

        Returns
        -------
        list
            List of path of all the snapshots files retrieved
        """
        num_files_c = c_size_t()
        self._twin_status = self._TwinGetNumRomSnapshotFiles(
            self._modelPointer,
            c_char_p(model_name.encode()),
            byref(num_files_c),
            c_double(time_from),
            c_double(time_to),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_mode_coef_data")

        bin_files_c = (c_char_p * num_files_c.value)()
        self._twin_status = self._TwinGetRomSnapshotFiles(
            self._modelPointer,
            c_char_p(model_name.encode()),
            bin_files_c,
            c_double(time_from),
            c_double(time_to),
        )
        self.evaluate_twin_status(self._twin_status, self, "twin_get_rom_snapshot_files")
        return to_np_array(bin_files_c)

    def twin_save_state(self, save_to):
        """
        Save the TWIN states (including model parameters) in the file
        indicated by the 'save_to' argument.

        Parameters
        ----------
        save_to : str
            Path of the file used to save the TWIN states.
        """
        save_to = save_to.encode()
        self._twin_status = self._TwinSaveState(self._modelPointer, c_char_p(save_to))
        self.evaluate_twin_status(self._twin_status, self, "twin_save_state")

    def twin_load_state(self, load_from, do_fmi_init=True):
        """
        Load and set the TWIN states with the ones stored in the file
        'load_from' (including model values used in the TWIN when saving
        the states).

        Parameters
        ----------
        load_from : str
            Path of the file used to load the TWIN states.
        do_fmi_init : bool (optional)
            Whether to initialize the TWIN underlying models (True) or
            not (False) before loading the states. Default value is True.
        """
        load_from = load_from.encode()
        try:
            self._twin_status = self._TwinLoadState(self._modelPointer, c_char_p(load_from), c_bool(do_fmi_init))
            self.evaluate_twin_status(self._twin_status, self, "twin_load_state")
        except OSError:
            msg = "Fatal error when loading the model state"
            raise TwinRuntimeError(msg, self, TwinStatus.TWIN_STATUS_FATAL.value)

        # The TwinRuntimeSDK always puts the model at least in INITIALIZED
        # state when loading a state
        self._is_model_initialized = True

    """
    Status message retrieval
    Function for getting status of the last operation if the result
    is not TWIN_STATUS_OK.
    """

    def twin_get_status_string(self):
        """
        Retrieves the status of the last operation if the result
        is not TWIN_STATUS_OK.

        Returns
        -------
        str
            String of the status retrieved.
        """
        return self.TwinGetStatusString(self._modelPointer).decode()

    """
    TwinRuntime Wrapper Helper functions
    """

    def load_twin_default_sim_settings(self):
        """
        Set the default simulation settings (end time, step size, tolerance)
        stored within the model.
        """
        if self._has_default_settings is False:
            (
                self._p_end_time,
                self._p_step_size,
                self._p_tolerance,
            ) = self.twin_get_default_simulation_settings()
            self._has_default_settings = True

    # pragma: no cover
    def print_model_info(self, max_var_to_print=np.inf):
        """
        Print all the model information including Twin Runtime version,
        model name, number of outputs, inputs, parameters, default simulation
        settings, output names, input names and parameter names.

        Parameters
        ----------
        max_var_to_print : int (optional)
            Maximum number of variables for which the properties need to be
            evaluated, default value is numpy.inf.
        """

        print("------------------------------------- Model Info" " -------------------------------------")
        print("Twin Runtime Version: {}".format(self.twin_get_api_version()))
        print("Model Name: {}".format(self._model_name.decode()))
        print("Number of outputs: {}".format(self._number_outputs))
        print("Number of Inputs: {}".format(self._number_inputs))
        print("Number of parameters: {}".format(self._number_parameters))
        print("Default time end: {}".format(self._p_end_time))
        print("Default step size: {}".format(self._p_step_size))
        print("Default tolerance(Integration Accuracy): {}".format(self._p_tolerance))
        print()
        print("Output names: ")
        self.print_var_info(self.twin_get_output_names(), max_var_to_print)

        print("Input names: ")
        self.print_var_info(self.twin_get_input_names(), max_var_to_print)

        print("Parameter names: ")
        self.print_var_info(self.twin_get_param_names(), max_var_to_print)

    def print_var_info(self, var_names, max_var_to_print):
        """
        Print all the properties (name, unit, quantity type, start value,
        minimum value, maximum values, description) of the given variables,
        with a maximum number of variables to consider.

        Parameters
        ----------
        var_names : list
            List of variables names for which the variable properties need to
            be evaluated.
        max_var_to_print : int (optional)
            Maximum number of variables for which the properties need to be
            evaluated, default value is numpy.inf.
        """
        if max_var_to_print > len(var_names):
            max_var_to_print = None

        print(self.model_properties_info_df(var_names, max_var_to_print))

        if max_var_to_print == 0:
            print("{} items not shown.".format(len(var_names)))
        elif max_var_to_print is not None:
            print("and {} more...".format(len(var_names) - max_var_to_print))
        print("\n")

    def full_model_properties_info_df(self):
        """
        Evaluate the properties (name, unit, quantity type, start value,
        minimum value, maximum values, description) of all the model's
        variables (inputs, outputs, parameters).

        Returns
        -------
        pandas.DataFrame
            Pandas dataframe storing all the properties evaluated for all
            the model's variables.
        """

        prop_matrix_list = []

        input_vars = self.twin_get_input_names()
        prop_matrix_list += self.build_prop_info_df(input_vars)

        output_vars = self.twin_get_output_names()
        prop_matrix_list += self.build_prop_info_df(output_vars)

        param_vars = self.twin_get_param_names()
        prop_matrix_list += self.build_prop_info_df(param_vars)

        var_inf_columns = [
            "Name",
            "Unit",
            "Type",
            "Start",
            "Min",
            "Max",
            "Description",
        ]
        variable_info_df = pd.DataFrame(prop_matrix_list, columns=var_inf_columns)

        return variable_info_df

    def model_properties_info_df(self, var_names, max_var_to_print):
        """
        Evaluate the properties (name, unit, data type, start value,
        minimum value, maximum values, description) of the given variables,
        with a maximum number of variables to consider.

        Parameters
        ----------
        var_names : list
            List of variables names for which the variable properties need
            to be evaluated.
        max_var_to_print : int
            Maximum number of variables for which the properties need
            to be evaluated.

        Returns
        -------
        pandas.DataFrame
            Pandas dataframe storing the properties evaluated for the
            given variables and maximum number to consider.
        """

        prop_matrix_list = self.build_prop_info_df(var_names[:max_var_to_print])

        var_inf_columns = [
            "Name",
            "Unit",
            "Type",
            "Start",
            "Min",
            "Max",
            "Description",
        ]

        variable_info_df = pd.DataFrame(data=prop_matrix_list, columns=var_inf_columns)
        return variable_info_df

    def build_prop_info_df(self, var_names):
        """
        Evaluate the properties (name, unit, data type, start value,
        minimum value, maximum values, description) of the given variables.

        Parameters
        ----------
        var_names : list
            List of variables names for which the variable properties
            need to be evaluated.

        Returns
        -------
        list
            List of the variables properties evaluated.
        """
        prop_matrix_list = []
        for value in var_names:
            o_name = value
            try:
                o_unit = self.twin_get_var_unit(value)
            except (
                PropertyNotDefinedError,
                PropertyNotApplicableError,
                PropertyInvalidError,
                PropertyError,
            ) as e:
                o_unit = e.property_status_flag.name
            try:
                o_data_type = self.twin_get_var_data_type(value)
            except (
                PropertyNotDefinedError,
                PropertyNotApplicableError,
                PropertyInvalidError,
                PropertyError,
            ) as e:
                o_data_type = e.property_status_flag.name
            try:
                o_var_description = self.twin_get_var_description(value)
            except (
                PropertyNotDefinedError,
                PropertyNotApplicableError,
                PropertyInvalidError,
                PropertyError,
            ) as e:
                o_var_description = e.property_status_flag.name
            try:
                if o_data_type == "String":
                    o_start = self.twin_get_str_var_start(value)
                else:
                    o_start = self.twin_get_var_start(value)
            except (
                PropertyNotDefinedError,
                PropertyNotApplicableError,
                PropertyInvalidError,
                PropertyError,
            ):
                o_start = None
            try:
                o_min = self.twin_get_var_min(value)
            except (
                PropertyNotDefinedError,
                PropertyNotApplicableError,
                PropertyInvalidError,
                PropertyError,
            ):
                o_min = None
            try:
                o_max = self.twin_get_var_max(value)
            except (
                PropertyNotDefinedError,
                PropertyNotApplicableError,
                PropertyInvalidError,
                PropertyError,
            ):
                o_max = None

            prop_row = [
                o_name,
                o_unit,
                o_data_type,
                o_start,
                o_min,
                o_max,
                o_var_description,
            ]
            prop_matrix_list.append(prop_row)
        return prop_matrix_list


def build_empty_ctype_2d_array(num_input_rows, number_of_columns):
    row_elements = c_double * number_of_columns

    # The one liner below initializes each row if 'input_data' with an array
    # of 'number_of_column' elements 'row_elements()' creates one ctypes
    # array for each input row
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
