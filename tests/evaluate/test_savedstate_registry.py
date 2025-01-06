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

import os

from pytwin import get_pytwin_log_file
from pytwin.evaluate.model import Model
from pytwin.evaluate.saved_state_registry import (
    SavedState,
    SavedStateError,
    SavedStateRegistry,
    SavedStateRegistryError,
)

from tests.utilities import compare_dictionary

UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")
UNIT_TEST_MODEL_ID = "1234abcd"
UNIT_TEST_MODEL_NAME = "test_model"


def reinit_registry():
    import shutil

    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    test_model = Model()
    test_model._id = UNIT_TEST_MODEL_ID
    test_model._model_name = UNIT_TEST_MODEL_NAME
    os.mkdir(test_model.model_dir)
    return test_model


class TestSavedState:
    def test_dump(self):
        ref_dict = {
            SavedState.ID_KEY: "1234abcd",
            SavedState.TIME_KEY: 0.12345678,
            SavedState.INPUTS_KEY: {"input1": 1.0, "input2": 2.0},
            SavedState.OUTPUTS_KEY: {"output1": 11.0, "output2": 22.0},
            SavedState.PARAMETERS_KEY: {"param1": 0.1, "param2": 0.2},
        }

        ss = SavedState()

        ss._id = ref_dict[SavedState.ID_KEY]
        ss.time = ref_dict[SavedState.TIME_KEY]
        ss.inputs = ref_dict[SavedState.INPUTS_KEY]
        ss.outputs = ref_dict[SavedState.OUTPUTS_KEY]
        ss.parameters = ref_dict[SavedState.PARAMETERS_KEY]

        dumped_dict = ss.dump()

        assert compare_dictionary(ref_dict, dumped_dict)

    def test_unique_id(self):
        id_store = []
        ss_count = 10000
        # Run test
        for i in range(ss_count):
            ss = SavedState()
            id_store.append(ss._id)
            assert id_store.count(ss._id) == 1

    def test_load_dump_x2(self):
        ref_dict = {
            SavedState.ID_KEY: "1234abcd",
            SavedState.TIME_KEY: 0.12345678,
            SavedState.INPUTS_KEY: {"input1": 1.0, "input2": 2.0},
            SavedState.OUTPUTS_KEY: {"output1": 11.0, "output2": 22.0},
            SavedState.PARAMETERS_KEY: {"param1": 0.1, "param2": 0.2},
        }

        ss = SavedState()
        ss.load(ref_dict)
        dumped_dict = ss.dump()
        ss.load(dumped_dict)
        dumped_dict2 = ss.dump()

        assert compare_dictionary(dumped_dict, dumped_dict2)

    def test_raise_error(self):
        wrong_dictionaries = [
            {},
            {SavedState.ID_KEY: "1234abcd"},
            {SavedState.ID_KEY: "1234abcd", SavedState.TIME_KEY: 0.12345678},
            {
                SavedState.ID_KEY: "1234abcd",
                SavedState.TIME_KEY: 0.12345678,
                SavedState.INPUTS_KEY: {"input1": 1.0, "input2": 2.0},
            },
            {
                SavedState.ID_KEY: "1234abcd",
                SavedState.TIME_KEY: 0.12345678,
                SavedState.INPUTS_KEY: {"input1": 1.0, "input2": 2.0},
                SavedState.OUTPUTS_KEY: {"output1": 11.0, "output2": 22.0},
            },
        ]
        for wrong_dict in wrong_dictionaries:
            try:
                ss = SavedState()
                ss.load(wrong_dict)
            except SavedStateError as e:
                assert "Metadata is corrupted." in str(e)


class TestSavedStateRegistry:
    def test_append_and_extract_saved_state(self):
        # Initialize unit test
        test_model = reinit_registry()
        ss1_dict = {
            SavedState.ID_KEY: "1234abcd",
            SavedState.TIME_KEY: 0.12345678,
            SavedState.INPUTS_KEY: {"input1": 1.0, "input2": 2.0},
            SavedState.OUTPUTS_KEY: {"output1": 11.0, "output2": 22.0},
            SavedState.PARAMETERS_KEY: {"param1": 0.1, "param2": 0.2},
        }
        ss1 = SavedState()
        ss1.load(ss1_dict)
        ss2_dict = {
            SavedState.ID_KEY: "1234abcdBIS",
            SavedState.TIME_KEY: 1.12345678,
            SavedState.INPUTS_KEY: {"input1": 11.0, "input2": 22.0},
            SavedState.OUTPUTS_KEY: {"output1": 111.0, "output2": 222.0},
            SavedState.PARAMETERS_KEY: {"param1": 1.1, "param2": 2.2},
        }
        ss2 = SavedState()
        ss2.load(ss2_dict)

        # Test appended SavedState are written in registry file
        ssr = SavedStateRegistry(model_id=test_model.id, model_name=test_model.name)
        ssr.append_saved_state(ss1)
        ssr.append_saved_state(ss2)
        with open(ssr.registry_filepath, "r") as ssr_fp:
            ssr_str = ssr_fp.readlines()
        assert ss1_dict[SavedState.ID_KEY] in "".join(ssr_str)
        assert ss2_dict[SavedState.ID_KEY] in "".join(ssr_str)

        # Test extracted SavedState are consistent with appended one
        ssr = SavedStateRegistry(model_id=test_model.id, model_name=test_model.name)
        extracted_ss1 = ssr.extract_saved_state(simulation_time=0.12345678, epsilon=1e-8)
        extracted_ss2 = ssr.extract_saved_state(simulation_time=1.12345678, epsilon=1e-8)
        assert compare_dictionary(extracted_ss1.dump(), ss1.dump())
        assert compare_dictionary(extracted_ss2.dump(), ss2.dump())

        # Test SavedState extraction with large epsilon
        ssr = SavedStateRegistry(model_id=test_model.id, model_name=test_model.name)
        extracted_ss1 = ssr.extract_saved_state(simulation_time=0.12345678, epsilon=2.0)
        assert compare_dictionary(extracted_ss1.dump(), ss1.dump())
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as fp:
            log_lines = fp.readlines()
        assert "Multiple saved states were found. The first one is " in "".join(log_lines)

    def test_raise_error(self):
        # Raise error if model dir does not exist
        try:
            SavedStateRegistry(model_id="unknown", model_name="unknown")
        except SavedStateRegistryError as e:
            assert "Use an existing model ID or model name." in str(e)
