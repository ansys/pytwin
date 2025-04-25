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

import json
import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Union
import weakref

import numpy as np
import pandas as pd
from pytwin.evaluate.model import Model
from pytwin.evaluate.saved_state_registry import SavedState, SavedStateRegistry
from pytwin.evaluate.tbrom import TbRom, read_snapshot_size
from pytwin.settings import PyTwinLogLevel, get_pytwin_log_level, pytwin_logging_is_enabled
from pytwin.twin_runtime.log_level import LogLevel
from pytwin.twin_runtime.twin_runtime_core import TwinRuntime

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv


class TwinModel(Model):
    """
    Evaluates a twin model in a TWIN file created by Ansys Twin Builder.

    After a twin model is initialized, it can be evaluated with two modes (step-by-step or batch).
    to make predictions. Parametric workflows are also supported.

    Parameters
    ----------
    model_filepath : str
        File path to the TWIN file for the twin model.

    Examples
    --------
    Create the twin model given the file path to the TWIN file. Initialize two parameters and two inputs of
    the twin model. Then, evaluate the two steps and retrieve the results in a dictionary.

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
        self._tbroms = None
        if self._check_model_filepath_is_valid(model_filepath):
            self._model_filepath = model_filepath
        self._instantiate_twin_model()
        self._finalizer = weakref.finalize(self, self._cleanup, self._twin_runtime, self.model_dir)

    @staticmethod
    def _cleanup(twin_runtime, model_dir):
        """
        Close twin runtime and remove model temporary folder.
        """
        if twin_runtime is not None:
            if twin_runtime._is_model_opened:
                twin_runtime.twin_close()
            # Delete model directory
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

    def close(self):
        """
        Cleanup object when user asks to close it.
        """
        self._cleanup(self._twin_runtime, self.model_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _check_model_filepath_is_valid(self, model_filepath):
        """
        Check if the filepath provided for the twin model is valid. Raise a ``TwinModelError`` message if not.
        """
        if model_filepath is None:
            msg = f"Twin model cannot be called with {model_filepath} as the model's filepath."
            msg += "\nProvide a valid filepath to initialize the ``TwinModel`` object."
            raise self._raise_error(msg)

        if not os.path.exists(model_filepath):
            msg = f"The provided filepath does not exist: {model_filepath}."
            msg += "\nProvide the correct filepath to initialize the ``TwinModel`` object."
            raise self._raise_error(msg)
        return True

    def _check_tbrom_model_filepath_is_valid(self, model_filepath):
        """
        Check if the resource directory path provided for the TBROM instantiation is valid.

        If the path is invalid, a ``TwinModelError`` error message is raised.
        """
        if model_filepath is None:
            msg = f"TbRom model cannot be instantiated with {model_filepath} as the resource directory path."
            msg += "\nProvide a valid resource directory path to instantiate the ``TbRom`` object."
            raise self._raise_error(msg)

        if not os.path.exists(model_filepath):
            msg = f"The provided resource directory does not exist: {model_filepath}."
            msg += "\nProvide the correct filepath to instantiate the ``TbRom`` object."
            raise self._raise_error(msg)
        return True

    def _check_rom_name_is_valid(self, rom_name):
        """
        Check if the rom name provided is part of the Twin's list of TbRom models. Raise a ``TwinModelError``
        message if not.
        """
        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            raise self._raise_error(msg)
        return True

    def _check_snapshot_detail(self, snapshot_detail: Union[str, Path, np.ndarray], tbrom: TbRom, fieldname: str):
        if snapshot_detail is None:
            msg = self._error_msg_input_snapshot_none()
            raise self._raise_error(msg)
        elif isinstance(snapshot_detail, np.ndarray):
            if len(snapshot_detail.shape) > 1:
                msg = self._error_msg_input_snapshot_array_wrong_shape(snapshot_detail)
                raise self._raise_error(msg)
            snapshotsize = len(snapshot_detail)
        elif isinstance(snapshot_detail, (str, Path)):
            if not os.path.exists(snapshot_detail):
                msg = self._error_msg_input_snapshot_path_does_not_exist(snapshot_detail)
                raise self._raise_error(msg)
            snapshotsize = read_snapshot_size(snapshot_detail)
        else:
            msg = self._error_msg_input_snapshot_detail_wrong_type(snapshot_detail)
            raise self._raise_error(msg)

        inputfieldsize = tbrom._input_field_size(fieldname)
        if snapshotsize != inputfieldsize:
            msg = self._error_msg_input_snapshot_size(snapshotsize, inputfieldsize)
            raise self._raise_error(msg)

    def _check_tbrom_input_field_dict_is_valid(self, tbrom_name: str, field_inputs: dict, t_count: int = None):
        """
        Check if the dictionary describing the input field snapshots is valid and consistent with the Twin's
        TBROM.
        1st : check if input field name provided is valid
        2nd : check if tbrom/twin have common inputs
        3rd : check if the provided snapshot array is 1D (for array snapshot)
        4th : check if the provided snapshot path exists (for string snapshot path)
        5th : check if the provided snapshot is a string or array
        6th : check if the provided snapshot size is consistent with input field basis
        7th : check the number of provided snapshots equal t_count (used in batch mode only)
        Raise a ``TwinModelError`` message if not.
        """
        tbrom = None
        if self._check_rom_name_is_valid(tbrom_name):
            tbrom = self._tbroms[tbrom_name]

        for fieldname, snapshot_details in field_inputs.items():
            if fieldname not in tbrom.field_input_names:
                msg = self._error_msg_for_unknown_field_name(fieldname, tbrom)
                raise self._raise_error(msg)

            if not tbrom._hasinfmcs[fieldname]:
                msg = self._error_msg_for_rom_input_connection(tbrom_name)
                raise self._raise_error(msg)

            # Add additional condition to this check, since accidentally passing a single snapshot
            # as a list, rather than array should be checked first
            if t_count is None:
                self._check_snapshot_detail(snapshot_details, tbrom, fieldname)
            else:
                # Snapshot details should be list of paths or list of numpy arrays
                if snapshot_details is None:
                    msg = self._error_msg_input_snapshot_list_none()
                    raise self._raise_error(msg)
                elif type(snapshot_details) is list:
                    if len(snapshot_details) != t_count:
                        msg = self._error_msg_input_snapshot_count(
                            found_count=len(snapshot_details), expected_count=t_count
                        )
                        raise self._raise_error(msg)

                    for i in snapshot_details:
                        self._check_snapshot_detail(i, tbrom, fieldname)
                else:
                    msg = self._error_msg_input_snaphshot_detail_list(snapshot_details)
                    raise self._raise_error(msg)

        return True

    def _check_tbrom_snapshot_generation_args(self, rom_name: str, namedselection: str = None):
        """
        Check if the arguments of the snapshot generation method are valid.

        If the arguments are not valid, a ``TwinModelError`` error message is raised.
        """
        tbrom = None
        if self._check_rom_name_is_valid(rom_name):
            tbrom = self._tbroms[rom_name]

        if not tbrom._hasoutmcs:
            msg = self._error_msg_for_rom_output_connection(rom_name)
            raise self._raise_error(msg)

        if namedselection is not None:
            if namedselection not in tbrom.named_selections:
                msg = self._error_msg_for_unknown_named_selection(namedselection, tbrom)
                raise self._raise_error(msg)
        return True

    def _check_tbrom_points_file(self, rom_name: str):
        """
        Check if the points file is available for the given ROM. Raise a ``TwinModelError`` message if not.
        """
        filepath = os.path.join(self._tbrom_resource_directory(rom_name), "binaryOutputField", "points.bin")
        if not os.path.exists(filepath):
            msg = self._error_msg_for_geometry_file_not_found(rom_name, filepath)
            raise self._raise_error(msg)
        return filepath

    def _check_tbrom_points_generation_args(self, rom_name: str, namedselection: str = None):
        """
        Check if the arguments of points generation method are valid. Raise a ``TwinModelError`` message if not.
        """
        tbrom = None
        if self._check_rom_name_is_valid(rom_name):
            tbrom = self._tbroms[rom_name]

        self._check_tbrom_points_file(rom_name)

        if namedselection is not None:
            if namedselection not in tbrom.named_selections:
                msg = self._error_msg_for_unknown_named_selection(namedselection, tbrom)
                raise self._raise_error(msg)
        return True

    def _check_tbrom_mesh_projection_args(self, rom_name: str, mesh: "pv.DataSet", namedselection: str = None):
        """
        Check if the arguments of mesh projection method are valid. Raise a ``TwinModelError`` message if not.
        """
        tbrom = None
        if self._check_rom_name_is_valid(rom_name):
            tbrom = self._tbroms[rom_name]

        if not tbrom._hasoutmcs:
            msg = self._error_msg_for_rom_output_connection(rom_name)
            raise self._raise_error(msg)

        if mesh.n_cells == 0 and mesh.n_points == 0:
            msg = self._error_msg_for_pv_mesh_empty(rom_name, mesh)
            raise self._raise_error(msg)

        if namedselection is not None:
            if namedselection not in tbrom.named_selections:
                msg = self._error_msg_for_unknown_named_selection(namedselection, tbrom)
                raise self._raise_error(msg)
        return True

    def _error_msg_for_unknown_field_name(self, fieldname, tbrom):
        msg = f"[FieldName]The field name provided {fieldname} is not part of the list of input field names "
        msg += f"{tbrom.field_input_names}."
        msg += "\nProvide a valid field name to use this method."
        return msg

    def _error_msg_for_geometry_file_not_found(self, rom_name, filepath):
        msg = f"[GeometryFile]Could not find the geometry file for the given ROM name: {rom_name}. "
        msg += (
            f"The geometry filepath that you are looking for is: {filepath}. Make sure to embed the geometry as part "
            f"of the TBROM."
        )
        return msg

    def _error_msg_for_pv_mesh_empty(self, rom_name, mesh):
        msg = f"[PyVistaMesh]Mesh {mesh} "
        msg += f"associated to the ROM name : {rom_name}"
        msg += f"seems to be empty : number of points = {mesh.n_points}, number of cells = {mesh.n_cells}"
        return msg

    def _error_msg_for_rom_name(self, rom_name):
        msg = f"[RomName]The provided ROM name ({rom_name}) has not been found in the available TBROM names. "
        msg += f"Call this method with a valid TBROM name."
        msg += f"\n Available TBROM names are: {self.tbrom_names}."
        return msg

    def _error_msg_for_not_initialized(self):
        msg = "[Initialization]Twin model has not been initialized. "
        msg += "Initialize the evaluation before using this method."
        return msg

    def _error_msg_for_rom_input_connection(self, tbrom_name):
        msg = f"[RomInputConnection]The TBROM {tbrom_name} has no common inputs with the Twin {self._model_name}."
        msg += "\nMake sure the TBROM has its mode coefficients inputs properly connected to the twin's inputs."
        return msg

    def _error_msg_for_rom_output_connection(self, rom_name):
        msg = f"[RomOutputConnection]The TBROM {rom_name} has no common outputs with the Twin {self._model_name}."
        msg += "\nMake sure the TBROM has its mode coefficients outputs properly connected to the twin's outputs."
        return msg

    def _error_msg_for_unknown_named_selection(self, namedselection, tbrom):
        msg = f"[NamedSelection]The provided named selection ({namedselection})"
        msg += " is not part of TBROM's list of named selection:"
        msg += f" {tbrom.named_selections}."
        msg += "\nProvide a valid named selection to use this method."
        return msg

    def _error_msg_not_instantiated(self):
        msg = "[Instantiation]Twin model has not been successfully instantiated."
        return msg

    def _error_msg_input_snapshot_none(self):
        msg = f"[InputSnapshotNone]The snapshot argument is missing."
        msg += "\nProvide a input field snapshot path or Numpy array to use this method."
        return msg

    def _error_msg_input_snapshot_detail_wrong_type(self, snapshot):
        # Remove snapshot from message, since don't want to dump a list or other long value
        msg = f"[InputSnapshotType]The snapshot is a {type(snapshot)}, not a string, Path or Numpy array."
        msg += "\nProvide a input field snapshot path or Numpy array to use this method."
        return msg

    def _error_msg_input_snapshot_array_wrong_shape(self, snapshot: np.ndarray):
        msg = f"[InputSnapshotShape]The snapshot array is not a 1D array, but rather {snapshot.shape}."
        msg += "\nProvide a 1D input field snapshot array to use this method with array input."
        return msg

    def _error_msg_input_snapshot_path_does_not_exist(self, snapshot: str):
        msg = f"[InputSnapshotPath]The snapshot path does not exist: {snapshot}."
        msg += "\nProvide an existing snapshot path as input field to use this method with file input."
        return msg

    def _error_msg_input_snapshot_size(self, snapshotsize, inputfieldsize):
        msg = f"[InputSnapshotSize]The provided snapshot size {snapshotsize} is not consistent with the input field"
        msg += f" size {inputfieldsize}."
        msg += "\nProvide a valid input field snapshot to use this method."
        return msg

    def _error_msg_input_snapshot_list_none(self):
        msg = f"[InputSnapshotListNone]The snapshot list argument for batch mode is missing."
        msg += "\nProvide a list of input field snapshot paths or Numpy arrays to use this method."
        return msg

    def _error_msg_input_snaphshot_detail_list(self, snapshot_details):
        msg = "[InputSnapshotList]When used in batch mode, the snapshot file paths or arrays must given in a list!"
        msg += f"\nProvided snapshot file details are: {snapshot_details}"
        return msg

    def _error_msg_input_snapshot_count(self, found_count: int, expected_count: int):
        msg = f"[InputSnapshotCount]The provided number of snapshots ({found_count}) to be used as input field "
        msg += f"must equal the number of time instants given in the batch data frame ({expected_count})."
        return msg

    def _error_msg_no_tbrom(self):
        msg = "[NoRom]Twin model does not include any TBROMs."
        return msg

    def _error_msg_view_names(self, view_name, rom_name, view_names):
        msg = f"[ViewName]The provided view name {view_name} is not available for this ROM name {rom_name}."
        msg += f"Call this method with a valid view name."
        msg += f'\n Available view names for "{rom_name}" are: {view_names}.'
        return msg

    def _error_msg_view_filepath_does_not_exist(self, rom_name, view_name, evaluation_time, filepath):
        msg = f"[ViewFilePath]Could not find the image file for the given ROM name: {rom_name}, "
        msg += f"view_name: {view_name}, "
        msg += f"and evaluation_time: {evaluation_time}."
        msg += f"The image filepath you are looking for is: {filepath}."
        return msg

    def _error_msg_output_snapshot_filepath_does_not_exist(self, rom_name, evaluation_time, filepath):
        msg = f"[OutputSnapshotPath]Could not find the snapshot file for the given ROM name ({rom_name}) "
        msg += f"and evaluation time ({evaluation_time})."
        msg += f"The snapshot filepath that you are looking for is: {filepath}."
        return msg

    def _error_msg_no_time_column_in_batch(self, columns):
        msg = "[TimeColumn]Dataframe given for inputs has no 'Time' column."
        msg += f"\nExisting column labels are :{[s for s in columns]}."
        msg += f"\nProvide a dataframe with a 'Time' column to use batch mode evaluation."
        return msg

    def _error_msg_no_time_zero_in_batch(self, t0):
        msg = "[TimeZero]Dataframe given for inputs has no time instant 't=0.s'."
        msg += f" The first provided time instant is: {t0})."
        msg += "\nProvide inputs at time instant 't=0.s'."
        return msg

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
        if not pytwin_logging_is_enabled():
            return LogLevel.TWIN_NO_LOG
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

    def _initialize_evaluation(
        self, parameters: dict = None, inputs: dict = None, field_inputs: dict = None, runtime_init: bool = True
    ):
        """
        Initialize the twin model evaluation with dictionaries:
        (1) Initialize parameters and/or inputs values to their start values (default values found in the twin file).
        (2) Update parameters and/or inputs values with provided dictionaries (including input field snapshot if
        applicable. Ignore values whose names are not found in the list of parameters/inputs names of the twin model.
        (Value is kept to the default value in this case.)
        (3) Initialize evaluation time to 0.
        (4) Save universal time (time since epoch) at which the method is called.
        (5) Evaluate twin model at time instance 0. Store its results into an outputs dictionary.
        Twin runtime is reset in case of already initialized twin model.
        Twin runtime is not initialized in case runtime_init is False.
        """
        if self._twin_runtime is None:
            self._raise_error("Twin model has not been successfully instantiated.")
        try:
            if self._twin_runtime._is_model_initialized:
                self._twin_runtime.twin_reset()

            self._initialize_parameters_with_start_values()
            if parameters is not None:
                self._update_parameters(parameters)

            self._initialize_inputs_with_start_values()
            if inputs is not None:
                self._update_inputs(inputs)

            if field_inputs is not None:
                self._update_field_inputs(field_inputs)

            self._warns_if_parameter_key_not_found(parameters)
            self._warns_if_input_key_not_found(inputs)

            self._evaluation_time = 0.0
            self._initialization_time = time.time()

            if runtime_init:
                self._twin_runtime.twin_initialize()
                self._update_outputs()

        except Exception as e:
            msg = f"Something went wrong during model initialization."
            msg += f"\n{str(e)}"
            msg += f"\nFor more information, see the model log file: {self.model_log}."
            self._raise_error(msg)

    def _initialize_inputs_with_start_values(self):
        """
        Initialize inputs dictionary {name:value} with starting input values found in the twin model.
        """
        self._inputs = dict()
        for name in self._twin_runtime.twin_get_input_names():
            self._inputs[name] = self._twin_runtime.twin_get_var_start(var_name=name)

    def _initialize_parameters_with_start_values(self):
        """
        Initialize parameters dictionary {name:value} with starting parameter values found in the twin model.
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
        self._log_key = "InstantiateTwinModel"
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

            # Create link to log file if any
            if os.path.exists(self.model_log):
                os.link(self.model_log, self.model_log_link)

            # Retrieve inputs, outputs and parameters meta-data
            self._inputs = dict()
            for name in self._twin_runtime.twin_get_input_names():
                self._inputs[name] = None
            self._parameters = dict()
            for name in self._twin_runtime.twin_get_param_names():
                if "solver." not in name:
                    self._parameters[name] = None
            self._outputs = dict()
            for name in self._twin_runtime.twin_get_output_names():
                self._outputs[name] = None

            # Retrieve tbrom_info
            tbrom_info = self._twin_runtime.twin_get_visualization_resources()
            if tbrom_info:
                self._log_key += "WithTBROM : {}".format(tbrom_info)
                self._tbrom_info = tbrom_info
                # directory_path = os.path.join(self.model_dir, self.TBROM_FOLDER_NAME)
                directory_path = self.tbrom_directory_path
                for tbrom_name, data in tbrom_info.items():
                    self._twin_runtime.twin_set_rom_image_directory(tbrom_name, directory_path)

            # Instantiate tbroms
            if self.tbrom_count > 0:
                self._tbroms = dict()
                for tbrom_name in self.tbrom_names:
                    tbrom_resdir = self._tbrom_resource_directory(tbrom_name)
                    if self._check_tbrom_model_filepath_is_valid(tbrom_resdir):
                        tbrom = TbRom(tbrom_name, tbrom_resdir)
                        self._tbrom_init(tbrom)
                        self._tbroms[tbrom_name] = tbrom

            # Update TwinModel variables
            self._instantiation_time = time.time()
            self._initialize_inputs_with_start_values()
            self._initialize_parameters_with_start_values()
            self._initialize_outputs_with_none_values()

        except Exception as e:
            msg = "Twin model failed during instantiation."
            msg += f"\n{str(e)}"
            self._raise_error(msg)

    def _raise_model_error(self, msg):
        """
        Raise a formatted ``TwinModelError`` message.
        """
        raise TwinModelError(msg)

    def _read_eval_init_config(self, json_filepath: str):
        """
        Deserialize a JSON object into a dictionary that is used to store twin model inputs and parameters values
        to pass to the internal evaluation initialization method.
        """
        if not os.path.exists(json_filepath):
            msg = "Provided configuration filepath (for evaluation initialization) does not exist."
            msg += f"\nProvided filepath is: {json_filepath}"
            msg += "\nProvide an existing filepath to initialize the twin model evaluation."
            raise self._raise_error(msg)
        try:
            with open(json_filepath) as file:
                cfg = json.load(file)
                return cfg
        except Exception as e:
            msg = "Something went wrong while reading the configuration file."
            msg += f"n{str(e)}"
            self._raise_error(msg)

    def _update_inputs(self, inputs: dict):
        """Update input values with the given dictionary."""
        for name, value in inputs.items():
            if name in self._inputs:
                self._inputs[name] = value
                self._twin_runtime.twin_set_input_by_name(input_name=name, value=value)

    def _update_outputs(self):
        """Update output values with twin model results at the current evaluation time."""
        self._outputs = dict(zip(self._twin_runtime.twin_get_output_names(), self._twin_runtime.twin_get_outputs()))
        if self.tbrom_count > 0:
            for key, item in self._tbroms.items():
                if item._hasoutmcs:
                    self._update_tbrom_outmcs(item)

    def _update_parameters(self, parameters: dict):
        """Update parameter values with the given dictionary."""
        for name, value in parameters.items():
            if name in self._parameters:
                self._parameters[name] = value
                self._twin_runtime.twin_set_param_by_name(param_name=name, value=value)

    def _tbrom_resource_directory(self, rom_name: str):
        """
        Get the path of the resource directory associated with the given ROM name.
        """
        if self._twin_runtime is None:
            msg = self._error_msg_not_instantiated()
            self._raise_error(msg)

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        return self._twin_runtime.twin_get_rom_resource_directory(rom_name)

    def _warns_if_interpolation_is_forced(self, nb_points, tm_ncells, tm_nbpoints):
        msg = (
            f"[MeshProjection]Switching interpolate flag from False to True. Number of TBROM points = "
            f"{nb_points}, number of mesh cells = {tm_ncells}, number of mesh points = "
            f"{tm_nbpoints}."
        )
        self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_WARNING)

    def _warns_if_input_key_not_found(self, inputs: dict):
        if inputs is not None:
            for _input in inputs:
                if _input not in self.inputs:
                    if _input != "Time":
                        msg = f"Provided input ({_input}) has not been found in the model inputs."
                        self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_WARNING)

    def _warns_if_parameter_key_not_found(self, parameters: dict):
        if parameters is not None:
            for param in parameters:
                if param not in self.parameters:
                    msg = f"Provided parameter ({param}) has not been found in the model parameters."
                    self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_WARNING)

    def _field_input_port_name(self, field: str, mode_idx: int, rom_name: str = None):
        if self.tbrom_count > 1:
            return field + "_mode_" + str(mode_idx) + "_" + rom_name
        return field + "_mode_" + str(mode_idx)

    def _field_output_port_name(self, mode_idx: int, rom_name: str = None):
        if self.tbrom_count > 1:
            return "outField" + "_mode_" + str(mode_idx) + "_" + rom_name
        return "outField" + "_mode_" + str(mode_idx)

    def _snapshot_filename(self, rom_name: str, named_selection: str = None):
        if named_selection is not None:
            filename = self._tbroms[rom_name].field_output_name
            filename += "_"
            filename += named_selection
            filename += "_"
            filename += "{:.6f}".format(self.evaluation_time)
            filename += ".bin"
            return filename

        filename = self._tbroms[rom_name].field_output_name
        filename += "_"
        filename += "{:.6f}".format(self.evaluation_time)
        filename += ".bin"
        return filename

    def _points_filename(self, rom_name: str, named_selection: str = None):
        if named_selection is not None:
            filename = self._tbroms[rom_name].field_output_name
            filename += "_"
            filename += named_selection
            filename += "_points.bin"
            return filename

        filename = self._tbroms[rom_name].field_output_name
        filename += "_points.bin"
        return filename

    def _tbrom_init(self, tbrom: TbRom):
        """
        Initialize the tbrom attributes and connect with the Twin inputs/outputs.
        """
        if tbrom.field_input_count > 0:
            infmcs = dict()
            hasinfmcs = dict()
            for field in tbrom.field_input_names:
                inmcs = dict()
                for i in range(0, len(tbrom._infbasis[field])):
                    input_port_name = self._field_input_port_name(field, i, tbrom.name)
                    for key, item in self.inputs.items():
                        if input_port_name in key:
                            inmcs.update({key: item})
                            break
                            # e.g. inputfield_mode_1 has been found but we don't want to pick inputfield_mode_10 yet
                if len(inmcs) == len(tbrom._infbasis[field]):
                    infmcs.update({field: inmcs})
                    hasinfmcs.update({field: True})
                else:
                    hasinfmcs.update({field: False})
            tbrom._infmcs = infmcs
            tbrom._hasinfmcs = hasinfmcs

        outmcs = dict()
        for i in range(1, tbrom.nb_modes + 1):
            output_port_name = self._field_output_port_name(i, tbrom.name)
            for key, item in self.outputs.items():
                if output_port_name in key:
                    outmcs.update({key: item})
                    break
                    # e.g. outField_mode_1 has been found but we don't want to pick outField_mode_10 yet
        if len(outmcs) == tbrom.nb_modes:
            tbrom._outmcs = outmcs
            tbrom._hasoutmcs = True

        tbrom._outputfilespath = os.path.join(self.tbrom_directory_path, tbrom.name)

    def _update_tbrom_outmcs(self, tbrom: TbRom):
        """
        Update tbrom attributes based on Twin's current outputs states
        """
        for key, item in tbrom._outmcs.items():
            tbrom._outmcs[key] = self.outputs[key]
        tbrom._update_output_field()

    def _update_field_inputs(self, field_inputs: dict):
        for tbrom_name, field_inputs in field_inputs.items():
            if self._check_tbrom_input_field_dict_is_valid(tbrom_name, field_inputs):
                tbrom = self._tbroms[tbrom_name]
                for name, snp in field_inputs.items():
                    self._update_field_input(tbrom, name, snp)

    def _update_field_input(
        self,
        tbrom: TbRom,
        field_input_name: str,
        snapshot: Union[str, Path, np.ndarray],
        update_twin_runtime: bool = True,
    ):
        """
        Update Twin's current inputs states based on tbrom attributes
        """
        tbrom._reduce_field_input(field_input_name, snapshot)
        infmcs = {}
        for mc_name, mc_value in tbrom._infmcs[field_input_name].items():
            self.inputs[mc_name] = mc_value
            infmcs[mc_name] = mc_value
            if update_twin_runtime:
                self._twin_runtime.twin_set_input_by_name(input_name=mc_name, value=self.inputs[mc_name])
        return infmcs

    @property
    def evaluation_is_initialized(self):
        """Indicator for if the evaluation has been initialized."""
        if self._twin_runtime is None:
            self._raise_error("Twin model has not been successfully instantiated.")
        return self._twin_runtime._is_model_initialized

    @property
    def evaluation_time(self):
        """
        Floating point number that is the current twin model evaluation time in seconds.
        """
        return self._evaluation_time

    @property
    def inputs(self):
        """
        Dictionary with input values at the current evaluation time.
        """
        return self._inputs

    @property
    def initialization_time(self):
        """
        Floating point number that is the time at which the twin model has been initialized.
        The value is given in seconds since the epoch."""
        return self._initialization_time

    @property
    def instantiation_time(self):
        """
        Floating point number that is the time at which the twin model has been instantiated.
        The value is given in seconds since the epoch."""
        return self._instantiation_time

    @property
    def outputs(self):
        """
        Dictionary with output values at the current evaluation time.
        """
        return self._outputs

    @property
    def parameters(self):
        """
        Dictionary with parameter values at the current evaluation time.
        """
        return self._parameters

    @property
    def model_filepath(self):
        """
        Filepath for the twin model that has been verified and loaded.
        If the filepath is not valid, ``None`` is returned.
        """
        return self._model_filepath

    @property
    def tbrom_info(self):
        """
        Dictionary with TBROM model names included in the twin model and their corresponding 3D visualization
        capabilities. Such capabilities include snapshots and optionally generated images.
        If a twin model has not been initialized, or if there is no TBROM in the twin model, ``None`` is
        returned.
        """
        return self._tbrom_info

    @property
    def tbrom_names(self):
        """
        List of available TBROM names. If there is no TBROM in the twin model, an empty list is returned.

        It is the name of TBROM component that was used in the subsheet to compile the TWIN file in Twin Builder.
        """
        if self._tbrom_info is not None:
            return list(self._tbrom_info)
        return []

    @property
    def tbrom_directory_path(self):
        """
        TBROM directory path. This is the directory where temporary TBROM files are stored.

        This folder is created at TBROM initialization by the Twin Runtime SDK.
        It contains one folder per TBROM with folder name equals to TBROM name.
        """
        return os.path.join(self.model_dir, self.TBROM_FOLDER_NAME)

    @property
    def tbrom_count(self):
        """
        Return number of TBROM contained in the Twin. If a twin model has not been initialized, or if there is no TBROM
        in the twin model, it returns 0.
        """
        return len(self.tbrom_names)

    def initialize_evaluation(
        self, parameters: dict = None, inputs: dict = None, field_inputs: dict = None, json_config_filepath: str = None
    ):
        """
        Initialize evaluation of a twin model.

        A twin model can be initialized with either a dictionary of parameters values and/or input (start) values
        or a JSON configuration file. For more information, see the examples.

        Using a JSON configuration file overrides using a dictionary of parameter values and/or input (start) values.

        If no inputs are given in the arguments or in the configuration file, calling this method
        resets inputs to their default values. The behavior is the same for parameters.

        Default values are kept for parameters and inputs that are not found in the provided dictionaries
        or configuration file. For example, the start value of the twin model is kept.

        After this method is called and the initialization time is updated, the evaluation time is reset to zero.

        This method must be called:

        - Before evaluating the twin model.
        - If you want to update parameters values between multiple twin evaluations. In this case,
          the twin model is reset.

        .. note::
            if field inputs are supplied for a TBROM, they will override any input mode coefficient inputs for
            that ROM that are included in ``inputs``.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary of parameter values ({"name": value}) to use for the next evaluation.
        inputs : dict, optional
            Dictionary of input values ({"name": value}) to use for twin model initialization.
        field_inputs : dict, optional
            Dictionary of input fields snapshots ({"tbromname": {"inputfieldname": snapshotpath}}) to use for twin model
            initialization.
        json_config_filepath : str, optional
            Filepath to a JSON configuration file to use to initialize the evaluation.

        Examples
        --------
        >>> import json
        >>> from pytwin import TwinModel
        >>>
        >>> # Example 1 - Using a config file and scalar inputs
        >>> config = {"version": "0.1.0", "model": {"inputs": {"input-name-1": 1., "input-name-2": 2.}, \
        >>> "parameters": {"param-name-1": 1.,"param-name-2": 2.}}}
        >>> with open('path_to_your_config.json', 'w') as f:
        >>>     f.write(json.dumps(config))
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation(json_config_filepath='path_to_your_config.json')
        >>> outputs = twin_model.outputs
        >>>
        >>> # Example 2 - Using a dictionary and field inputs from disk
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation()
        >>> romname = twin_model.tbrom_names[0]
        >>> fieldname = twin_model.get_field_input_names(romname)[0]
        >>> twin_model.initialize_evaluation(field_inputs={romname: {fieldname:'path_to_the_snapshot.bin'}})
        >>> results = {'Time': twin_model.evaluation_time, 'Outputs': twin_model.outputs}
        >>>
        >>> # Example 3 - Using a dictionary and field inputs from memory
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation()
        >>> romname = twin_model.tbrom_names[0]
        >>> fieldname = twin_model.get_field_input_names(romname)[0]
        >>> snapshot = np.array([3.14, 2.71, 9.81, 6.02])
        >>> twin_model.initialize_evaluation(field_inputs={romname: {fieldname:snapshot}})
        >>> results = {'Time': twin_model.evaluation_time, 'Outputs': twin_model.outputs}
        """
        self._log_key = "InitializeEvaluation"

        if json_config_filepath is None:
            self._log_key += "WithDictionary"
            self._initialize_evaluation(parameters=parameters, inputs=inputs, field_inputs=field_inputs)
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
            self._initialize_evaluation(parameters=_parameters, inputs=_inputs, field_inputs=field_inputs)

    def evaluate_step_by_step(self, step_size: float, inputs: dict = None, field_inputs: dict = None):
        """
        Evaluate the twin model at time instant `t` plus a step size given inputs at time instant `t`.

        Twin model evaluation must have been initialized before calling this evaluation method.
        For more information, see the :func:`pytwin.TwinModel.initialize_evaluation` method.

        .. note::
            if field_inputs are supplied for a TBROM, they will override any input mode coefficient inputs for
            that ROM that are included in ``inputs``.

        Parameters
        ----------
        step_size : float
            Step size in seconds to reach the next time step. The value must be positive.
        inputs : dict (optional)
            Dictionary of scalar input values ({"name": value}) at time instant `t`. An input is not updated if
            the associated key is not found in the twin model's ``input_names`` property. If values for
            inputs are not provided in the dictionary, their current values are kept.
        field_inputs : dict (optional)
            Dictionary of input fields snapshots ({"tbromname": {"inputfieldname": snapshot}}) to use for twin model
            evaluation. ``snapshot`` may be a Numpy array, or string for path of snapshot file.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>>
        >>> # Example 1 - Evaluate step by step with scalar inputs and scalar outputs
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation()
        >>> twin_model.evaluate_step_by_step(step_size=0.1, inputs={'input1': 1., 'input2': 2.})
        >>> scalar_results = {'Time': twin_model.evaluation_time, 'Outputs': twin_model.outputs}
        >>>
        >>> # Example 2 - Evaluate step by step with field input from disk and field output to disk
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation()
        >>> romname = twin_model.tbrom_names[0]
        >>> fieldname = twin_model.get_field_input_names(romname)[0]
        >>> twin_model.evaluate_step_by_step(step_size=0.1, inputs={'input1': 1., 'input2': 2.},
        >>>                                  field_inputs={romname: {fieldname:'path_to_the_snapshot.bin'}})
        >>> field_output = twin_model.generate_snapshot(rom_name=romname, on_disk=True)
        >>> field_results = {'Time': twin_model.evaluation_time, 'Field': field_output}
        >>>
        >>> # Example 3 - Evaluate step by step with field input from memory and field output in memory
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> twin_model.initialize_evaluation()
        >>> romname = twin_model.tbrom_names[0]
        >>> fieldname = twin_model.get_field_input_names(romname)[0]
        >>> snapshot = np.array([3.14, 2.71, 9.81, 6.02])
        >>> twin_model.evaluate_step_by_step(step_size=0.1, inputs={'input1': 1., 'input2': 2.},
        >>>                                  field_inputs={romname: {fieldname:snapshot}})
        >>> field_output = twin_model.generate_snapshot(rom_name=romname, on_disk=False)
        >>> field_results = {'Time': twin_model.evaluation_time, 'Field': field_output}
        """
        self._log_key = "EvaluateStepByStep"

        if self._twin_runtime is None:
            msg = self._error_msg_not_instantiated()
            self._raise_error(msg)

        if not self.evaluation_is_initialized:
            msg = self._error_msg_for_not_initialized()
            self._raise_error(msg)

        if step_size <= 0.0:
            msg = f"Step size must be greater than zero. The value provided was {step_size}.)"
            self._raise_error(msg)

        self._warns_if_input_key_not_found(inputs)
        if inputs is not None:
            self._update_inputs(inputs)

        if field_inputs is not None:
            self._update_field_inputs(field_inputs)

        try:
            self._twin_runtime.twin_simulate(self._evaluation_time + step_size)
            self._evaluation_time += step_size
            self._update_outputs()
        except Exception as e:
            msg = f"Something went wrong during evaluation at time step {self._evaluation_time}:"
            msg += f"\n{str(e)}"
            msg += f"Reinitialize the model evaluation and restart the evaluation."
            msg += f"\nFor more information, see the model log file: {self.model_log}."
            self._raise_error(msg)

    def evaluate_batch(self, inputs_df: pd.DataFrame, field_inputs: dict = None):
        """
        Evaluate the twin model with historical input values given in a data frame.

        .. note::
            if field_inputs are supplied for a TBROM, they will override any input mode coefficient inputs for
            that ROM that are included in ``inputs_df``.

        Parameters
        ----------
        inputs_df: pandas.DataFrame
            Historical input values stored in a Pandas dataframe. It must have a 'Time' column and all history
            for the twin model inputs that you want to simulate. The dataframe must have one input per column,
            starting at time instant `t=0.(s)`. If a twin model input is not found in a dataframe column,
            this input is kept constant to its initialization value. The column header must match with a
            twin model input name.
        field_inputs : dict (optional)
            Dictionary of snapshot file paths or snapshot Numpy arrays that must be used as field input at all
            time instants given by the 'inputs_df' argument. One file path or array must be given per time
            instant, for a field input of a TBROM included in the twin model, using following dictionary format:
            {"tbrom_name": {"field_input_name": [snapshot_t0, snapshot_t1, ... ]}}

        Returns
        -------
        output_df: pandas.DataFrame
            Twin output values associated with the input values stored in the Pandas dataframe.

        Raises
        ------
        TwinModelError:
            If the :func:`pytwin.TwinModel.initialize_evaluation` method has not been called before.
            If there is no 'Time' column in the input values stored in the Pandas dataframe.
            If there is no time instant `t=0.s` in the input values stored in the Pandas dataframe.
            If the list of snapshots given as field inputs has not one snapshot per time instant.
            If the snapshots given as field inputs are not Numpy arrays or paths to snapshot files.
            If the field inputs dictionary has bad TBROM or field input names.

        Examples
        --------
        >>> import pandas as pd
        >>> from pytwin import TwinModel
        >>>
        >>> # Example 1 - Batch evaluation with scalar inputs and scalar outputs
        >>> twin_model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> inputs_df = pd.DataFrame({'Time': [0., 1., 2.], 'input1': [1., 2., 3.], 'input2': [1., 2., 3.]})
        >>> twin_model.initialize_evaluation(inputs={'input1': 1., 'input2': 1.})
        >>> scalar_outputs_df = twin_model.evaluate_batch(inputs_df=inputs_df)
        >>>
        >>> # Example 2 - Batch evaluation with field inputs from disk and field output
        >>> model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> romname = model.tbrom_names[0]
        >>> fieldname = twin_model.get_field_input_names(romname)[0]
        >>> snapshot_filepath_t0 = 'path_to_snapshot_t0.twin'
        >>> twin_model.initialize_evaluation(field_inputs={romname: {fieldname: snapshot_filepath_t0})
        >>> inputs_df = pd.DataFrame({'Time': [0., 1., 2.]})
        >>> snapshot_filepaths = ['path_to_snapshot_t0.bin', 'path_to_snapshot_t1.bin', 'path_to_snapshot_t2.bin']
        >>> batch_results = twin_model.evaluate_batch(inputs_df=inputs_df,\
        field_inputs={romname: {fieldname: snapshot_filepaths})
        >>> output_snapshots = twin_model.generate_snapshot_batch(batch_results, romname)
        >>>
        >>> # Example 3 - Batch evaluation with field inputs from memory and field output
        >>> model = TwinModel(model_filepath='path_to_your_twin_model.twin')
        >>> romname = model.tbrom_names[0]
        >>> fieldname = twin_model.get_field_input_names(romname)[0]
        >>> snapshot_t0 = np.array([3.14, 2.71, 9.81, 6.02])
        >>> twin_model.initialize_evaluation(field_inputs={romname: {fieldname: snapshot_t0})
        >>> inputs_df = pd.DataFrame({'Time': [0., 1., 2.]})
        >>> snapshots = [snapshot_t0, snapshot_t1, snapshot_t2]
        >>> batch_results = twin_model.evaluate_batch(inputs_df=inputs_df,\
        field_inputs={romname: {fieldname: snapshots})
        >>> output_snapshots = twin_model.generate_snapshot_batch(batch_results, romname)
        """
        self._log_key = "EvaluateBatch"

        if self._twin_runtime is None:
            msg = self._error_msg_not_instantiated()
            self._raise_error(msg)

        if not self.evaluation_is_initialized:
            msg = self._error_msg_for_not_initialized()
            self._raise_error(msg)

        _inputs_df = inputs_df.copy()
        if "Time" not in _inputs_df:
            msg = self._error_msg_no_time_column_in_batch(_inputs_df.columns)
            self._raise_error(msg)

        t0 = _inputs_df["Time"][0]
        if not np.isclose(t0, 0.0, atol=np.spacing(0.0)):
            msg = self._error_msg_no_time_zero_in_batch(t0)
            self._raise_error(msg)

        if field_inputs is not None:
            t_count = _inputs_df.shape[0]
            for tbrom_name, field_inputs_dict in field_inputs.items():
                if self._check_tbrom_input_field_dict_is_valid(tbrom_name, field_inputs_dict, t_count):
                    for field_name, snapshots in field_inputs_dict.items():
                        for i, snapshot in enumerate(snapshots):
                            infmcs = self._update_field_input(
                                tbrom=self._tbroms[tbrom_name],
                                field_input_name=field_name,
                                snapshot=snapshot,
                            )
                            mc_idx = 0
                            for mc_name, mc_value in infmcs.items():
                                header_name = self._field_input_port_name(field_name, mc_idx, tbrom_name)
                                if i == 0:
                                    _inputs_df[header_name] = [0.0] * t_count
                                _inputs_df.at[i, header_name] = mc_value
                                mc_idx += 1

        try:
            # Ensure SDK conventions are fulfilled
            _inputs_df = self._create_dataframe_inputs(_inputs_df)
            _output_col_names = ["Time"] + list(self._outputs.keys())

            return self._twin_runtime.twin_simulate_batch_mode(
                input_df=_inputs_df, output_column_names=_output_col_names
            )
        except Exception as e:
            msg = f"Something went wrong during batch evaluation:"
            msg += f"\n{str(e)}"
            msg += f"\nReinitialize the model evaluation and restart the evaluation."
            msg += f"\nFor more information, see the model log file: {self.model_log}."
            self._raise_error(msg)

    def get_available_view_names(self, rom_name: str):
        """
        Get a list of view names for a ROM (reduced order model) in the twin model.

        Parameters
        ----------
        rom_name : str
            Name of the ROM in the twin model. To get a list of available ROMs,
            see the :attr:`pytwin.TwinModel.get_available_view_names` attribute.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object does not include any TBROMs.
            If the provided ROM name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.get_available_view_names(rom_name=model.tbrom_names[0])
        """
        self._log_key = "GetImageViewNames"

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        view_names = list(self._tbrom_info[rom_name][self.TBROM_VIEWS_KEY].values())

        if len(view_names) == 0:
            msg = f"No views are available for the given ROM name: {rom_name}."
            self._log_message(msg, level=PyTwinLogLevel.PYTWIN_LOG_WARNING)

        return view_names

    def get_image_filepath(self, rom_name: str, view_name: str, evaluation_time: float = 0.0):
        """
        Get the image file that was created by the given ROM at the given time instant. The image file shows the field
        results of the ROM in the given predefined view.

        Parameters
        ----------
        rom_name : str
            Name of the ROM. To get a list of available ROMs, see
            the :attr:`pytwin.TwinModel.tbrom_names` attribute.
        view_name : str
            View name associated with the rendering view in which ROM results are displayed. To get
            a list of available rendering view names for a given ROM, use the
            :func:`pytwin.TwinModel.get_available_view_names` method.
        evaluation_time: float, optional
            Evaluation time at which to get the image file. The default is ``0.0``. If no
            image file is available at the time specified, this method returns ``None``.
            Two evaluation times can be distinguished up to six digits after the comma.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object has not been initialized.
            If ``TwinModel`` object does not include any TBROMs.
            If the provided ROM name is not available.
            If the provided view name is not available.

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
            msg = "Twin model has not been initialized. "
            msg += "Initialize evaluation before calling this method."
            self._raise_error(msg)

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        view_names = self.get_available_view_names(rom_name)
        if view_name not in view_names:
            msg = self._error_msg_view_names(view_name, rom_name, view_names)
            self._raise_error(msg)

        filename = f"{view_name}_"
        filename += f"{format(evaluation_time, self.TBROM_FILENAME_TIME_FORMAT)}"
        filename += f"{self.TBROM_IMAGE_EXT}"
        filepath = os.path.join(self.tbrom_directory_path, rom_name, filename)

        if not os.path.exists(filepath):
            msg = self._error_msg_view_filepath_does_not_exist(rom_name, view_name, evaluation_time, filepath)
            self._log_message(msg, level=PyTwinLogLevel.PYTWIN_LOG_WARNING)

        return filepath

    def get_geometry_filepath(self, rom_name: str):
        """
        Get the geometry file associated with a ROM available in the twin model.

        The geometry file contains the coordinates of the points that are used to define the
        geometrical support of the ROM field output.

        Parameters
        ----------
        rom_name : str
            Name of the ROM. To get a list of available ROMs, see the
            :attr:`pytwin.TwinModel.tbrom_names` attribute.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` obbject does not include any TBROMs.
            If the given ROM name is not available.
            If the given geometry file cannot be found for the ROM.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> available_rom_names = model.tbrom_names
        >>> geometry_filepath = TwinModel.get_geometry_filepath(rom_name=available_rom_names[0])
        """
        self._log_key = "GetGeometryFilePath"

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        filepath = self._check_tbrom_points_file(rom_name)
        return filepath

    def get_rom_directory(self, rom_name):
        """
        Get the working directory path for a ROM in the twin model.

        Parameters
        ----------
        rom_name : str
            Name of the ROM. To get a list of available ROMs, see the
            :attr:`pytwin.TwinModel.tbrom_names` attribute.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object does not include any TBROMs.
            If the provided ROM name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.get_rom_directory(model.tbrom_names[0])
        """
        self._log_key = "GetRomDirectory"

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        return os.path.join(self.tbrom_directory_path, rom_name)

    def get_named_selections(self, rom_name):
        """
        Get the list of named selections associated to the TBROM named rom_name.

        Parameters
        ----------
        rom_name : str
            Name of the TBROM. To get a list of available TBROMs, see the
            :attr:`pytwin.TwinModel.tbrom_names` attribute.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object does not include any TBROMs.
            If the provided ROM name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.get_named_selections(model.tbrom_names[0])
        """
        self._log_key = "GetNamedSelections"

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        tbrom = self._tbroms[rom_name]

        return tbrom.named_selections

    def get_field_input_names(self, rom_name):
        """
        Get the list of input fields names associated to the TBROM named rom_name.

        Parameters
        ----------
        rom_name : str
            Name of the ROM. To get a list of available ROMs, see the
            :attr:`pytwin.TwinModel.tbrom_names` attribute.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object does not include any TBROMs.
            If the provided ROM name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.get_field_input_names(model.tbrom_names[0])
        """
        self._log_key = "GetFieldInputNames"

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        tbrom = self._tbroms[rom_name]

        return tbrom.field_input_names

    def get_field_output_name(self, rom_name):
        """
        Get the output field name associated to the TBROM named rom_name.

        Parameters
        ----------
        rom_name : str
            Name of the ROM. To get a list of available ROMs, see the
            :attr:`pytwin.TwinModel.tbrom_names` attribute.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object does not include any TBROMs.
            If the provided ROM name is not available.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> model.get_field_output_name(model.tbrom_names[0])
        """
        self._log_key = "GetFieldOutputName"

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        tbrom = self._tbroms[rom_name]

        return tbrom.field_output_name

    def get_snapshot_filepath(self, rom_name: str, evaluation_time: float = 0.0):
        """
        Get the snapshot file that was created by the given ROM at the given time instant.

        The snapshot file contains the field results of the ROM.

        Parameters
        ----------
        rom_name : str
            Name of the ROM. To get a list of available ROMs, see the
            :attr:`pytwin.TwinModel.tbrom_names` attribute.
        evaluation_time: float, optional
            Evaluation time at which to get the snapshot file. The default is ``0.00`. If no
            snapshot file is available at this evaluation time, the method returns ``None``.
            Two evaluation times can be distinguished up to six digits after the comma.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object has not been initialized.
            If ``TwinModel`` object does not include any TBROMs.
            If provided ROM name is not available.

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
            msg = "Twin model has not been initialized. "
            msg += "Initialize evaluation before calling the snapshot file getter."
            self._raise_error(msg)

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        filename = f"{self.TBROM_SNAPSHOT_FILE_PREFIX}"
        filename += f"{format(evaluation_time, self.TBROM_FILENAME_TIME_FORMAT)}"
        filename += f"{self.TBROM_SNAPSHOT_EXT}"
        filepath = os.path.join(self.tbrom_directory_path, rom_name, filename)

        if not os.path.exists(filepath):
            msg = self._error_msg_output_snapshot_filepath_does_not_exist(rom_name, evaluation_time, filepath)
            self._log_message(msg, level=PyTwinLogLevel.PYTWIN_LOG_WARNING)

        return filepath

    def load_state(self, model_id: str, evaluation_time: float, epsilon: float = 1e-8):
        """
        Load a state that has been saved by a twin model instantiated with the same TWIN file.

        .. note::
           Calling this method replaces evaluation initialization.

        Parameters
        ----------
        model_id : str
            ID of the model that saved the state.
        evaluation_time : float
            Evaluation time at which the state was saved.
        epsilon : float, optional
            Absolute period that is added before and after the evaluation time to account for
            round-off error while searching the saved state. The default value is ``1e-8``.
            The search is performed in the interval [t-epsilon, t+epsilon] with `t` being
            the evaluation time. The first saved state found in this interval is loaded.

        Raises
        ------
        TwinModelError:
            If no state has been saved by the model with the given model ID and same model name
            as the one calling this method.

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
            self._initialize_evaluation(parameters=ss.parameters, inputs=ss.inputs, runtime_init=False)
            self._twin_runtime.twin_load_state(ss_filepath)
            self._evaluation_time = ss.time

            BUG732106_WORKAROUND = True
            if BUG732106_WORKAROUND:
                # Rather we call a step-by-step evaluation with a small time step OR we use the registry outputs
                # self.evaluate_step_by_step(step_size=ss.time * 1e-12, inputs=ss.inputs)
                self._outputs = ss.outputs
            else:
                self._update_outputs()

        except Exception as e:
            msg = f"Something went wrong while loading the state:"
            msg += f"\n{str(e)}"
            self._raise_error(msg)

    def save_state(self):
        """
        Save the state of the twin model after its initialization and after step-by-step evaluation.

        This method should be used in conjunction with the :func:`pytwin.TwinModel.load_state` method.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> # Instantiate a twin model, initialize it, and evaluate it step by step until you want to save its state
        >>> model1 = TwinModel('model.twin')
        >>> model1.initialize_evaluation()
        >>> model1.evaluate_step_by_step(step_size=0.1)
        >>> model1.save_state()
        >>> # Instantiate a new twin model with the same TWIN file and load the saved state
        >>> model2 = TwinModel('model.twin')
        >>> model2.load_state(model_id=model1.id, evaluation_time=model1.evaluation_time)
        >>> model2.evaluate_step_by_step(step_size=0.1)
        """
        self._log_key = "SaveState"

        try:
            # Lazy init saved state registry for this twin model
            if self._ss_registry is None:
                self._ss_registry = SavedStateRegistry(model_id=self.id, model_name=self.name)

            # Store saved state metadata
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
            msg = f"Something went wrong while saving the state:"
            msg += f"\n{str(e)}."
            self._raise_error(msg)

    def generate_snapshot(self, rom_name: str, on_disk: bool = True, named_selection: str = None):
        """
        Generate a field snapshot based on current states of the Twin, either in memory or on disk, for the full field
        or a specific part. It returns the field data as an array if in memory, or the path of the snapshot written on
        disk.

        Parameters
        ----------
        rom_name : str
            Name of the TBROM considered to generate the snapshot.
        on_disk : bool
            Whether the snapshot file is saved on disk (True which is the default) or returned in memory (False).
        named_selection : str (optional)
            Named selection on which the snasphot has to be generated.

        Returns
        -------
        str | np.ndarray
            Path to snapshot written to disk if on_disk == True, else array of snapshot field data

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object has not been initialized.
            If rom_name is not included in the Twin's list of TBROM
            If TBROM hasn't its mode coefficients outputs connected to the twin's outputs.
            If name_selection is not included in the TBROM's list of Named Selections

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> # Instantiate a twin model, initialize it, and evaluate it step by step until you want to save its state
        >>> model1 = TwinModel('model.twin')
        >>> model1.initialize_evaluation()
        >>> romname = model1.tbrom_names[0]
        >>> nslist = model1.get_named_selections(romname)
        >>> fieldresults = model1.generate_snapshot(romname, False, nslist[0])
        """
        self._log_key = "GenerateSnapshot"

        if not self.evaluation_is_initialized:
            msg = self._error_msg_for_not_initialized()
            self._raise_error(msg)

        self._check_tbrom_snapshot_generation_args(rom_name, named_selection)

        try:
            output_filename = self._snapshot_filename(rom_name, named_selection)
            output_filepath = os.path.join(self._tbroms[rom_name]._outputfilespath, output_filename)
            return self._tbroms[rom_name]._generate_snapshot(on_disk, output_filepath, named_selection)

        except Exception as e:
            msg = f"Something went wrong while generating the snapshot:"
            msg += f"\n{str(e)}."
            self._raise_error(msg)

    def generate_snapshot_batch(
        self, batch_results: pd.DataFrame, rom_name: str, on_disk: bool = True, named_selection: str = None
    ):
        """
        Generate several field snapshots based on historical batch results of the Twin, either in memory or on disk, for
        the full field or a specific named selection. It returns a list of the field data as an array if in memory, or a
        list of the paths of the different snapshots written on disk.

        Parameters
        ----------
        batch_results : pandas.DataFrame
            Historical output values stored in a Pandas dataframe. It must have a 'Time' column and all the time
            instants for the twin model outputs that you want to post process, with one output per column.
        rom_name : str
            Name of the TBROM considered to generate the snapshot.
        on_disk : bool
            Whether the snapshot file is saved on disk (True which is the default) or returned in memory (False).
        named_selection : str (optional)
            Named selection on which the snasphot has to be generated.

        Returns
        -------
        list[str] | list[np.ndarray]
            List of paths to snapshots written to disk if on_disk == True, else list of arrays of snapshot field data

        Raises
        ------
        TwinModelError:
            If the :func:`pytwin.TwinModel.initialize_evaluation` method has not been called before.
            If rom_name is not included in the Twin's list of TBROM
            If TBROM hasn't its mode coefficients outputs connected to the twin's outputs.
            If name_selection is not included in the TBROM's list of Named Selections

        Examples
        --------
        >>> import pandas as pd
        >>> from pytwin import TwinModel
        >>> # Instantiate a twin model, initialize it, and evaluate it step by step until you want to save its state
        >>> model = TwinModel('model.twin')
        >>> inputs_df = pd.DataFrame({'Time': [0., 1., 2.], 'input1': [1., 2., 3.], 'input2': [1., 2., 3.]})
        >>> model.initialize_evaluation(inputs={'input1': 1., 'input2': 1.})
        >>> romname = model.tbrom_names[0]
        >>> nslist = model.get_named_selections(romname)
        >>> outputs_df = model.evaluate_batch(inputs_df=inputs_df)
        >>> fieldresults = model.generate_snapshot_batch(outputs_df, romname, nslist[0])
        """
        self._log_key = "GenerateSnapshotBatch"

        if not self.evaluation_is_initialized:
            msg = self._error_msg_for_not_initialized()
            self._raise_error(msg)

        try:
            columns = batch_results.columns[1::]
            outputsnapshots = []

            current_time = self._evaluation_time
            current_outputs = self._outputs

            for i, row in batch_results.iterrows():
                time = row["Time"]
                self._evaluation_time = time
                outputs = dict(zip(list(columns), row[list(columns)]))

                # Update output values with twin model results at the current evaluation time.
                self._outputs = outputs
                if self.tbrom_count > 0:
                    for key, tbrom in self._tbroms.items():
                        if tbrom._hasoutmcs:
                            self._update_tbrom_outmcs(tbrom)

                # Generate the snapshot at the current evaluation time.
                outputsnapshots.append(self.generate_snapshot(rom_name, on_disk, named_selection))

            self._evaluation_time = current_time
            self._outputs = current_outputs
            if self.tbrom_count > 0:
                for key, tbrom in self._tbroms.items():
                    if tbrom._hasoutmcs:
                        self._update_tbrom_outmcs(tbrom)

            return outputsnapshots

        except Exception as e:
            msg = f"Something went wrong while generating the snapshot:"
            msg += f"\n{str(e)}."
            self._raise_error(msg)

    def generate_points(self, rom_name: str, on_disk: bool = True, named_selection: str = None):
        """
        Generate a points file either in memory or on disk, for the full domain or a specific part. It returns the
        points data as an array if in memory, or the path of the points file written on disk.

        Parameters
        ----------
        rom_name: str
            Name of the TBROM considered to generate the point file.
        on_disk: bool
            Whether the point file is saved on disk (True) or returned in memory (False).
        named_selection: str (optional)
            Named selection on which the point file has to be generated.

        Returns
        -------
        str | np.ndarray
            Path to points file written to disk if on_disk == True, else array of points data

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object has not been initialized.
            If rom_name is not included in the Twin's list of TBROM
            If the TBROM does not have any point file available
            If name_selection is not included in the TBROM's list of Named Selections

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> # Instantiate a twin model, initialize it, and evaluate it step by step until you want to save its state
        >>> model1 = TwinModel('model.twin')
        >>> model1.initialize_evaluation()
        >>> romname = model1.tbrom_names[0]
        >>> nslist = model1.get_named_selections(romname)
        >>> points = model1.generate_points(romname, False, nslist[0])
        """
        self._log_key = "GeneratePoints"

        if not self.evaluation_is_initialized:
            msg = self._error_msg_for_not_initialized()
            self._raise_error(msg)

        try:
            if self._check_tbrom_points_generation_args(rom_name, named_selection):
                output_file = self._points_filename(rom_name, named_selection)
                output_file_path = os.path.join(self._tbroms[rom_name]._outputfilespath, output_file)
                return self._tbroms[rom_name]._generate_points(on_disk, output_file_path, named_selection)

        except Exception as e:
            msg = f"Something went wrong while generating the points file:"
            msg += f"\n{str(e)}."
            self._raise_error(msg)

    def project_tbrom_on_mesh(
        self,
        rom_name: str,
        target_mesh: "pv.DataSet",
        interpolate: bool,
        named_selection: str = None,
        nodal_values: bool = False,
        sharpness: float = 5.0,
        radius: float = 0.0001,
        strategy: str = "closest_point",
        null_value: float = 0.0,
        n_points: int = None,
        all_points: bool = False,
    ):
        """
        Project the field ROM data onto a targeted mesh, using the current states of the TwinModel. The returned PyVista
        DataSet object is automatically updated whenever the TwinModel is evaluated.

        Parameters
        ----------
        rom_name : str
            Name of the TBROM considered to project the results.
        target_mesh: pyvista.DataSet
            PyVista DataSet object of the targeted mesh.
        interpolate: bool
            Flag to indicate whether the point cloud data are interpolated (True) or not (False) on the targeted mesh.
            Interpolation is recommended when point cloud data and mesh data are not ordered in the same way, and when
            the target mesh is different from the one used to generate the ROM. Interpolation is automatically enforced
            if the target mesh size (i.e. number of cells/points) is different from the point cloud size.
        nodal_values: bool (optional)
            Control whether the interpolated results are returned as nodal values, or cell values (default)
        named_selection: str (optional)
            Named selection from the ROM (i.e. subset of points cloud) that will be projected on the targeted mesh. The
            default is ``None``, in which case the entire domain is considered.
        sharpness : float, default: 5.0
            Set the sharpness (i.e., falloff) of the Gaussian interpolation kernel. As the sharpness increases the
            effects of distant points are reduced.
        radius : float, default: 0.0001
            Specify the radius within which the interpolation basis points must lie.
        strategy : str, default: "closest_point"
            Specify a strategy to use when encountering a "null" point during the interpolation process. Null points
            occur when the local neighborhood (of nearby points to interpolate from) is empty. If the strategy is set to
            ``'mask_points'``, then only cells with some or all valid points (according to the ``all_points`` setting)
            are included in the returned PyVista DataSet. If the strategy is set to ``'null_value'``, then the output
            data value(s) are set to the ``null_value`` (specified in the output point data). Finally, the strategy
            ``'closest_point'`` is to simply use the closest point to perform the interpolation.
        null_value : float, default: 0.0
            Specify the null point value. When a null point is encountered then all components of field ROM data
            associated with that point are set to this value.
        n_points : int, optional
            If given, specifies the number of the closest points used to form the interpolation basis. This will
            invalidate the radius argument in favor of an N closest points approach. This typically has poorer results.
        all_points: bool, default: False
            When ``strategy='mask_points'``, when this value is ``True`` only cells where all points are valid are kept.
            When ``False`` cells are kept if any of their points are valid and invalid points are given the
            ``null_value``.

        Returns
        -------
        pyvista.DataSet
            PyVista DataSet object of the targeted mesh with projected field ROM data.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object has not been initialized.
            If rom_name is not included in the Twin's list of TBROM
            If TBROM hasn't its mode coefficients outputs connected to the twin's outputs
            If target_mesh is not a valid grid dataset
            If name_selection is not included in the TBROM's list of Named Selections
            If interpolate is True and no points file is available with the TBROM
            If strategy is ``'mask_points'`` and all points are removed.

        TwinModelWarning:
            If interpolate is False and the targeted mesh has a number of cells and points different from TBROM point
            cloud. In that case, interpolate is automatically switched to True.

        See Also
        --------
        pyvista.DataSetFilters.interpolate :
            Detailed description of ``sharpness``, ``radius``, ``strategy``, ``null_value`` and ``n_points`` parameters.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> import pyvista as pv
        >>> # Instantiate a twin model, initialize it, and evaluate it step by step until you want to save its state
        >>> model = TwinModel('model.twin')
        >>> romname = model.tbrom_names[0]
        >>> target_mesh = pv.read('mesh.vtk')
        >>> rom_results_on_mesh = model.project_tbrom_on_mesh(romname, target_mesh)
        >>> model.initialize_evaluation()
        """
        self._log_key = "MeshProjection"

        if not self.evaluation_is_initialized:
            msg = self._error_msg_for_not_initialized()
            self._raise_error(msg)

        try:
            if self._check_tbrom_mesh_projection_args(rom_name, target_mesh, named_selection):
                if named_selection is None:
                    nb_points = self._tbroms[rom_name].nb_points
                else:
                    nb_points = len(self._tbroms[rom_name]._named_selection_indexes(named_selection))
                if not interpolate and (target_mesh.n_cells != nb_points and target_mesh.n_points != nb_points):
                    self._warns_if_interpolation_is_forced(nb_points, target_mesh.n_cells, target_mesh.n_points)
                    interpolate_flag = True
                else:
                    interpolate_flag = interpolate
                if interpolate_flag:
                    self._check_tbrom_points_file(rom_name)
                self._tbroms[rom_name]._project_on_mesh(
                    target_mesh,
                    interpolate_flag,
                    named_selection,
                    nodal_values=nodal_values,
                    sharpness=sharpness,
                    radius=radius,
                    strategy=strategy,
                    null_value=null_value,
                    n_points=n_points,
                    all_points=all_points,
                )
                self._update_tbrom_outmcs(self._tbroms[rom_name])
                return self._tbroms[rom_name].field_on_mesh

        except Exception as e:
            msg = f"Something went wrong while projecting on target mesh:"
            msg += f"\n{str(e)}."
            self._raise_error(msg)

    def get_tbrom_output_field(self, rom_name: str):
        """
        Return the TBROM output field as a PyVista DataSet object, in the form of point cloud data. The resulting field
        is based on current states of the TwinModel and is automatically updated whenever the TwinModel is evaluated.

        Parameters
        ----------
        rom_name : str
            Name of the ROM. To get a list of available ROMs, see the
            :attr:`pytwin.TwinModel.tbrom_names` attribute.

        Returns
        -------
        pyvista.DataSet
            PyVista DataSet object of the TBROM output field in the form of point cloud data.

        Raises
        ------
        TwinModelError:
            If ``TwinModel`` object does not include any TBROMs.
            If the provided ROM name is not available.
            If the TBROM does not have any point file available.
            If TBROM hasn't its mode coefficients outputs connected to the twin's outputs.

        Examples
        --------
        >>> from pytwin import TwinModel
        >>> model = TwinModel(model_filepath='path_to_twin_model_with_TBROM_in_it.twin')
        >>> romname = model.tbrom_names[0]
        >>> rom_results = model.get_tbrom_output_field(romname)
        >>> model.initialize_evaluation()
        """
        self._log_key = "GetPointsData"

        if self.tbrom_info is None:
            msg = self._error_msg_no_tbrom()
            self._raise_error(msg)

        if rom_name not in self.tbrom_names:
            msg = self._error_msg_for_rom_name(rom_name)
            self._raise_error(msg)

        tbrom = self._tbroms[rom_name]

        if not tbrom.has_point_file:
            msg = self._check_tbrom_points_file(rom_name)
            self._raise_error(msg)

        if not tbrom._hasoutmcs:
            msg = self._error_msg_for_rom_output_connection(rom_name)
            raise self._raise_error(msg)

        return tbrom.field_on_points


class TwinModelError(Exception):
    def __str__(self):
        return f"[TwinModelError] {self.args[0]}"
