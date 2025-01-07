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
import uuid

import numpy as np
from pytwin import get_pytwin_logger
from pytwin.evaluate.model import Model


class SavedState:
    """
    Provides the metadata of the saved state of a twin model.
    """

    ID_KEY = "id"
    TIME_KEY = "time"
    INPUTS_KEY = "inputs"
    OUTPUTS_KEY = "outputs"
    PARAMETERS_KEY = "parameters"

    def __init__(self):
        self._id = f"{uuid.uuid4()}"[0:24].replace("-", "")
        self.time = None
        self.inputs = None
        self.outputs = None
        self.parameters = None

    def dump(self):
        var = dict()
        var[self.ID_KEY] = self._id
        var[self.TIME_KEY] = self.time
        var[self.INPUTS_KEY] = self.inputs
        var[self.OUTPUTS_KEY] = self.outputs
        var[self.PARAMETERS_KEY] = self.parameters
        return var

    def load(self, json_dict: dict):
        self._check_given_dict(json_dict)
        self._id = json_dict[self.ID_KEY]
        self.time = json_dict[self.TIME_KEY]
        self.inputs = json_dict[self.INPUTS_KEY]
        self.outputs = json_dict[self.OUTPUTS_KEY]
        self.parameters = json_dict[self.PARAMETERS_KEY]

    @staticmethod
    def _raise_error(msg):
        logger = get_pytwin_logger()
        logger.error(msg)
        raise SavedStateError(msg)

    def _check_given_dict(self, json_dict):
        requested_keys = [self.ID_KEY, self.TIME_KEY, self.INPUTS_KEY, self.OUTPUTS_KEY, self.PARAMETERS_KEY]
        for key in requested_keys:
            if key not in json_dict:
                msg = f"Metadata is corrupted. No '{key}' key was found."
                msg += f"\nGiven dictionary is: {json_dict}."
                self._raise_error(msg)


class SavedStateError(Exception):
    def __str__(self):
        return f"[SavedStateError] {self.args[0]}"


class SavedStateRegistry:
    """
    Manages a registry of saved states for the twin model.

    This class registers metadata associated with the saved state, persists
    the saved state, and provides methods for appending and extracting saved states.
    """

    SAVED_STATES_KEY = "saved_states"

    def __init__(self, model_id: str, model_name: str):
        self._model_id = None
        self._model_name = None
        self._saved_states = None

        self._check_model_dir_exists(model_id, model_name)
        self._model_id = model_id
        self._model_name = model_name

        # Backup folder creation
        if not os.path.exists(self.backup_folderpath):
            os.mkdir(self.backup_folderpath)

    @property
    def backup_folderpath(self):
        model = Model()
        model._id = self._model_id
        model._model_name = self._model_name
        return os.path.join(model.model_dir, "backup")

    @property
    def registry_filename(self):
        return "registry.json"

    @property
    def registry_filepath(self):
        return os.path.join(self.backup_folderpath, self.registry_filename)

    def append_saved_state(self, ss: SavedState):
        if self._saved_states is None:
            self._saved_states = []
        self._saved_states.append(ss)
        self._write_registry()

    def extract_saved_state(self, simulation_time: float, epsilon: float):
        self._read_registry()
        return self._search_saved_state(simulation_time, epsilon)

    def return_saved_state_filepath(self, ss: SavedState):
        return os.path.join(self.backup_folderpath, f"saved_state{ss._id}.bin")

    @staticmethod
    def _raise_error(msg):
        logger = get_pytwin_logger()
        logger.error(msg)
        raise SavedStateRegistryError(msg)

    def _check_model_dir_exists(self, model_id: str, model_name: str):
        model = Model()
        model._id = model_id
        model._model_name = model_name
        wd = model.model_dir
        if not os.path.exists(wd):
            msg = f"Model directory ({wd}) does not exist."
            msg += "\nUse an existing model ID or model name."
            msg += f" (Given model_id:{model_id}, model_name:{model_name})"
            msg += " to instantiate a SavedStateRegistry."
            self._raise_error(msg)

    def _check_given_dict(self, json_dict):
        requested_keys = [self.SAVED_STATES_KEY]
        for key in requested_keys:
            if key not in requested_keys:
                if key not in json_dict:
                    msg = f"Metadata is corrupted. No '{key}' key was found."
                    msg += f"\n{json_dict}"
                    self._raise_error(msg)

    def _dump(self):
        var = dict()
        var[self.SAVED_STATES_KEY] = []
        for ss in self._saved_states:
            var[self.SAVED_STATES_KEY].append(ss.dump())
        return var

    def _load(self, json_dict: dict):
        self._check_given_dict(json_dict)
        # Load saved states
        self._saved_states = []
        for ss_dict in json_dict[self.SAVED_STATES_KEY]:
            ss = SavedState()
            ss.load(ss_dict)
            self._saved_states.append(ss)

    def _read_registry(self):
        try:
            with open(self.registry_filepath, "r", encoding="utf-8") as fp:
                self._load(json_dict=json.load(fp))
        except Exception as e:
            msg = f"Something went wrong while reading the registry file {self.registry_filename}."
            msg += f"\n{str(e)}"
            self._raise_error(msg)

    def _search_saved_state(self, evaluation_time: float, epsilon: float):
        time_instants = np.array([ss.time for ss in self._saved_states])
        tl = evaluation_time - epsilon
        tr = evaluation_time + epsilon
        idx = np.where((time_instants > tl) & (time_instants < tr))

        if len(idx[0]) == 0:
            msg = f"No state at simulation time {evaluation_time} was found."
            self._raise_error(msg)

        if len(idx[0]) > 1:
            times = []
            for i in range(len(idx[0])):
                ss = self._saved_states[idx[0][i]]
                times.append(ss.time)
            msg = (
                f"[SavedStateRegistry]Multiple saved states were found. The first one is "
                f"\nused at simulation time {times[0]}."
            )
            logger = get_pytwin_logger()
            logger.warning(msg)

        idx = idx[0][0]

        return self._saved_states[idx]

    def _write_registry(self):
        try:
            # Save current registry to registry file
            with open(self.registry_filepath, "w", encoding="utf-8") as fp:
                json.dump(self._dump(), fp, indent=4)
        except Exception as e:
            msg = f"Something went wrong while writing the registry file {self.registry_filename}."
            msg += f"\n{str(e)}"
            self._raise_error(msg)


class SavedStateRegistryError(Exception):
    def __str__(self):
        return f"[SavedStateRegistryError] {self.args[0]}"
