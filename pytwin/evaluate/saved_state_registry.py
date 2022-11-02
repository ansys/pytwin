import json
import os
import uuid

import numpy as np

from pytwin import get_pytwin_logger
from pytwin.evaluate.model import Model


class SavedState:
    """
    MetaData of the twin model on user save state request.
    """
    ID_KEY = 'id'
    TIME_KEY = 'time'
    INPUTS_KEY = 'inputs'
    OUTPUTS_KEY = 'outputs'
    PARAMETERS_KEY = 'parameters'

    def __init__(self):
        self.id = f'{uuid.uuid4()}'[0:8]
        self.time = None
        self.inputs = None
        self.outputs = None
        self.parameters = None

    def dump(self):
        var = dict()
        var[self.ID_KEY] = self.id
        var[self.TIME_KEY] = self.time
        var[self.INPUTS_KEY] = self.inputs
        var[self.OUTPUTS_KEY] = self.outputs
        var[self.PARAMETERS_KEY] = self.parameters

    def load(self, json_dict: dict):
        self._check_given_dict(json_dict)
        self.id = json_dict[self.ID_KEY]
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
        requested_keys = [self.ID_KEY,
                          self.TIME_KEY,
                          self.INPUTS_KEY,
                          self.OUTPUTS_KEY,
                          self.PARAMETERS_KEY]
        for key in requested_keys:
            if key not in requested_keys:
                if key not in json_dict:
                    msg = f'Meta data is corrupted! No \'{key}\' key was found!'
                    msg += f'\n{json_dict}'
                    self._raise_error(msg)


class SavedStateError(Exception):
    def __str__(self):
        return f'[SavedStateError] {self.args[0]}'


class SavedStateRegistry:
    """
    This class manages a registry of twin model saved states. It registers meta-data associated to saved state, persists
    it and provide append and extract saved state methods.
    """
    SAVED_STATES_KEY = 'saved_states'

    def __init__(self, model_id: str, model_name: str):
        self._model_id = None
        self._model_name = None
        self._saved_states = None

        self._check_model_dir_exists(model_id, model_name)
        self._model_id = model_id
        self._model_name = model_name

    @property
    def backup_folderpath(self):
        model = Model()
        model._id = self._model_id
        model._name = self._model_name
        return os.path.join(model.model_dir, 'backup')

    @property
    def registry_filename(self):
        return 'registry.json'

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
        ss = self._search_saved_state(simulation_time, epsilon)
        return self.return_saved_state_filepath(ss)

    def return_saved_state_filepath(self, ss: SavedState):
        return os.path.join(self.backup_folderpath, f'saved_state{ss.id}.bin')

    @staticmethod
    def _raise_error(msg):
        logger = get_pytwin_logger()
        logger.error(msg)
        raise SavedStateRegistryError(msg)

    def _check_model_dir_exists(self, model_id: str, model_name: str):
        model = Model()
        model._id = model_id
        model._name = model_name
        wd = model.model_dir
        if not os.path.exists(wd):
            msg = f'Model directory ({wd}) does not exist!'
            msg += f'\nPlease use existing model id (given {model_id}) and model name (given {model_name})'
            msg += ' to instantiate a SavedStateRegistry!'
            self._raise_error(msg)

    def _check_given_dict(self, json_dict):
        requested_keys = [self.SAVED_STATES_KEY]
        for key in requested_keys:
            if key not in requested_keys:
                if key not in json_dict:
                    msg = f'Meta data is corrupted! No \'{key}\' key was found!'
                    msg += f'\n{json_dict}'
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
        self.saved_states = []
        for ss in json_dict[self.SAVED_STATES_KEY]:
            self.saved_states.append(SavedState().load(ss))

    def _read_registry(self):
        try:
            with open(self.registry_filename, 'r') as fp:
                self._load(json_dict=json.load(fp))
        except Exception as e:
            msg = f'Something went wrong while reading registry file {self.registry_filename}!'
            msg += f'\n{str(e)}'
            self._raise_error(msg)

    def _search_saved_state(self, simulation_time: float, epsilon: float):
        time_instants = np.array([ss.time for ss in self._saved_states])
        tl = simulation_time * (1 - epsilon)
        tr = simulation_time * (1 + epsilon)
        idx = np.where((time_instants > tl) & (time_instants < tr))

        if len(idx[0]) == 0:
            msg = f'No state at simulation time {simulation_time} was found!'
            self._raise_error(msg)

        if len(idx[0]) > 0:
            times = [ss.time for ss in self._saved_states[idx[0]]]
            msg = f'Multiple saved states was found! Using first one, at simulation time {times[0]}'
            logger = get_pytwin_logger()
            logger.warning(msg)

        idx = idx[0][0]

        return self._saved_states[idx]

    def _write_registry(self):
        try:
            # Lazy backup folder creation
            if not os.path.exists(self.backup_folderpath):
                os.mkdir(self.backup_folderpath)
            # Save current registry to registry file
            with open(self.registry_filename, 'w') as fp:
                json.dump(self._dump(), fp)
        except Exception as e:
            msg = f'Something went wrong while writing registry file {self.registry_filename}!'
            msg += f'\n{str(e)}'
            self._raise_error(msg)


class SavedStateRegistryError(Exception):
    def __str__(self):
        return f'[SavedStateRegistryError] {self.args[0]}'
