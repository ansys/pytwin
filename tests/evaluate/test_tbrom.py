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
import sys

import numpy as np
import pandas as pd
from pytwin import (
    TwinModel,
    TwinModelError,
    download_file,
    read_binary,
    snapshot_to_array,
    write_binary,
)
from pytwin.evaluate import tbrom
from pytwin.settings import get_pytwin_log_file
import pyvista as pv


def reinit_settings():
    import shutil

    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
DYNAROM_HX_23R1 = os.path.join(os.path.dirname(__file__), "data", "HX_scalarDRB_23R1_other.twin")
RC_HEAT_CIRCUIT_23R1 = os.path.join(os.path.dirname(__file__), "data", "RC_heat_circuit_23R1.twin")

MESH_FILE = os.path.join(os.path.dirname(__file__), "data", "mesh.vtk")

UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")

"""
TEST_TB_ROM1
Twin with no TBROM -> nbTBROM = 0
"""
TEST_TB_ROM1 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_1.twin")

"""
TEST_TB_ROM2
Twin with 1 TBROM and 2 input fields but no input field connected, no output field connected
-> nbTBROM = 1, NbInputField = 2, hasInputField = (False, False), hasOutputField = False
"""
TEST_TB_ROM2 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_2.twin")

"""
TEST_TB_ROM3
Twin with 1 TBROM and 2 input fields both connected, 1 output field connected
-> nbTBROM = 1, NbInputField = 2, hasInputField = (True, True), hasOutputField = True
"""
TEST_TB_ROM3 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_3.twin")

"""
TEST_TB_ROM4
Twin with 1 TBROM and 2 input fields, 1st partially connected, second fully connected, 1 output field connected
-> nbTBROM = 1, NbInputField = 2, hasInputField = (True, False), hasOutputField = True
"""
TEST_TB_ROM4 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_4.twin")

"""
TEST_TB_ROM5
Twin with 1 TBROM and 1 input fields connected with error, 1 output field connected
-> nbTBROM = 1, NbInputField = 1, hasInputField = (False), hasOutputField = True
"""
TEST_TB_ROM5 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_5.twin")

"""
TEST_TB_ROM6
Twin with 2 TBROM, 1st has no connection, second has 2 input field connected, 1 output field connected,
-> nbTBROM = 2, NbInputField = (1, 2), hasInputField = False and (True, True), hasOutputField = False and True
"""
TEST_TB_ROM6 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_6.twin")

"""
TEST_TB_ROM7
Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected,
second has 2 input field connected with 1st field with errors, 1 output field connected
-> nbTBROM = 2, NbInputField = (1,2), hasInputField = True and (True, False), hasOutputField = True and True
"""
TEST_TB_ROM7 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_7.twin")

"""
TEST_TB_ROM8
Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected,
second has 2 input field connected, 1 output field connected
-> nbTBROM = 2, NbInputField = (1,2), hasInputField = (True, True) and True, hasOutputField = True and True
"""
TEST_TB_ROM8 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_8.twin")

"""
TEST_TB_ROM9
Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected with error,
second has 2 input field connected with second field with errors, 1 output field connected
-> nbTBROM = 2, NbInputField = (1,2), hasInputField = True and (True, False), hasOutputField = False and True
"""
TEST_TB_ROM9 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_9.twin")

"""
TEST_TB_ROM10
Twin with 1 TBROM but with 3D disabled at export time -> points file not available
"""
TEST_TB_ROM10 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_10.twin")

"""
TEST_TB_ROM11
Twin with 1 TBROM without named selection in settings
-> nbTBROM = 1, NbInputField = 0, hasInputField = False, hasOutputField = True
"""
TEST_TB_ROM11 = os.path.join(os.path.dirname(__file__), "data", "twin_no_ns.twin")

"""
TEST_TB_ROM12
Twin with 1 TBROM with two named selections in settings
-> nbTBROM = 1, NbInputField = 1, hasInputField = True, hasOutputField = True,
   named_selections = ['Group_1', 'Group_2']
"""
TEST_TB_ROM12 = os.path.join(os.path.dirname(__file__), "data", "ThermalTBROM_FieldInput_23R1.twin")

INPUT_SNAPSHOT = os.path.join(os.path.dirname(__file__), "data", "input_snapshot.bin")
INPUT_SNAPSHOT_WRONG = os.path.join(os.path.dirname(__file__), "data", "input_snapshot_wrong.bin")

"""
TEST_TB_ROM_TENSOR
Twin with 1 TBROM with tensor field
(https://github.com/ansys/pytwin/discussions/164)
"""
TEST_TB_ROM_TENSOR = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_stress_field.json")

"""
TEST_TB_ROM_NDOF
Twin with Dynamic ROM and NDOF not properly defined (bug 1168769 fixed in 2025R2)
"""
TEST_TB_ROM_NDOF = os.path.join(os.path.dirname(__file__), "data", "twin_ndof.twin")

"""
TEST_TB_ROM_CONSTRAINTS
Twin with 1 TBROM from SRB with constraints enabled (min/max = -0.00055/0.00044), deformation vector field
"""
TEST_TB_ROM_CONSTRAINTS = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_constraints.twin")

"""
TEST_TB_PFIELD_HISTORY
Twin with 1 TBROM of type parametric field history
"""
TEST_TB_PFIELD_HISTORY = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_pfieldhistory.twin")


def norm_vector_field(field: list):
    """Compute the norm of a vector field."""
    vec = field.reshape((-1, 3))
    return np.sqrt((vec * vec).sum(axis=1))

class TestTbRom:

    def test_instantiate_evaluation_tbrom1(self):
        """
        TEST_TB_ROM1
        Twin with no TBROM -> nbTBROM = 0
        """
        model_filepath = TEST_TB_ROM1
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 0

    def test_instantiate_evaluation_tbrom2(self):
        """
        TEST_TB_ROM2
        Twin with 1 TBROM and 2 input fields but no input field connected, no output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (False, False), hasOutputField = False
        """
        model_filepath = TEST_TB_ROM2
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 2
        assert tbrom1._hasoutmcs is False
        assert tbrom1._hasinfmcs["inputPressure"] is False
        assert tbrom1._hasinfmcs["inputTemperature"] is False

    def test_instantiate_evaluation_tbrom3(self):
        """
        TEST_TB_ROM3
        Twin with 1 TBROM and 2 input fields both connected, 1 output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, True), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 2
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputPressure"] is True
        assert tbrom1._hasinfmcs["inputTemperature"] is True

    def test_instantiate_evaluation_tbrom4(self):
        """
        TEST_TB_ROM4
        Twin with 1 TBROM and 2 input fields, 1st partially connected, second fully connected, 1 output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, False), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM4
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 2
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputPressure"] is True
        assert tbrom1._hasinfmcs["inputTemperature"] is False

    def test_instantiate_evaluation_tbrom5(self):
        """
        TEST_TB_ROM5
        Twin with 1 TBROM and 1 input fields connected with error, 1 output field connected
        -> nbTBROM = 1, NbInputField = 1, hasInputField = (False), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 1
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputTemperature"] is False

    def test_instantiate_evaluation_tbrom6(self):
        """
        TEST_TB_ROM6
        Twin with 2 TBROM, 1st has no connection, second has 2 input field connected, 1 output field connected,
        -> nbTBROM = 2, NbInputField = (1, 2), hasInputField = False and (True, True), hasOutputField = False and True
        """
        model_filepath = TEST_TB_ROM6
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 2
        tbrom1 = twinmodel._tbroms[twinmodel.tbrom_names[0]]  # tbrom with 1 field
        tbrom2 = twinmodel._tbroms[twinmodel.tbrom_names[1]]  # tbrom with 2 field
        assert tbrom1.field_input_count is 1
        assert tbrom2.field_input_count is 2
        assert tbrom1._hasoutmcs is False
        assert tbrom1._hasinfmcs["inputTemperature"] is False
        assert tbrom2._hasoutmcs is True
        assert tbrom2._hasinfmcs["inputPressure"] is True
        assert tbrom2._hasinfmcs["inputTemperature"] is True

    def test_instantiate_evaluation_tbrom7(self):
        """
        TEST_TB_ROM7
        Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected,
        second has 2 input field connected with 1st field with errors, 1 output field connected
        -> nbTBROM = 2, NbInputField = (1,2), hasInputField = True and (True, False), hasOutputField = True and True
        """
        model_filepath = TEST_TB_ROM7
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 2
        tbrom1 = twinmodel._tbroms[twinmodel.tbrom_names[0]]  # tbrom with 1 field
        tbrom2 = twinmodel._tbroms[twinmodel.tbrom_names[1]]  # tbrom with 2 fields
        assert tbrom1.field_input_count is 1
        assert tbrom2.field_input_count is 2
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputTemperature"] is True
        assert tbrom2._hasoutmcs is True
        assert tbrom2._hasinfmcs["inputPressure"] is True
        assert tbrom2._hasinfmcs["inputTemperature"] is False

    def test_instantiate_evaluation_tbrom8(self):
        """
        TEST_TB_ROM8
        Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected,
        second has 2 input field connected, 1 output field connected
        -> nbTBROM = 2, NbInputField = (1,2), hasInputField = (True, True) and True, hasOutputField = True and True
        """
        model_filepath = TEST_TB_ROM8
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 2
        tbrom1 = twinmodel._tbroms[twinmodel.tbrom_names[0]]  # tbrom with 1 field
        tbrom2 = twinmodel._tbroms[twinmodel.tbrom_names[1]]  # tbrom with 2 fields
        assert tbrom1.field_input_count is 1
        assert tbrom2.field_input_count is 2
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputTemperature"] is True
        assert tbrom2._hasoutmcs is True
        assert tbrom2._hasinfmcs["inputPressure"] is True  # pressure
        assert tbrom2._hasinfmcs["inputTemperature"] is True  # temperature

    def test_instantiate_evaluation_tbrom9(self):
        """
        TEST_TB_ROM9
        Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected with error,
        second has 2 input field connected with second field with errors, 1 output field connected
        -> nbTBROM = 2, NbInputField = (1,2), hasInputField = True and (True, False), hasOutputField = False and True
        """
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 2
        tbrom1 = twinmodel._tbroms[twinmodel.tbrom_names[0]]  # tbrom with 1 field
        tbrom2 = twinmodel._tbroms[twinmodel.tbrom_names[1]]  # tbrom with 2 fields
        assert tbrom1.field_input_count is 1
        assert tbrom2.field_input_count is 2
        assert tbrom1._hasoutmcs is False
        assert tbrom1._hasinfmcs["inputTemperature"] is True
        assert tbrom2._hasoutmcs is True
        assert tbrom2._hasinfmcs["inputPressure"] is True  # pressure
        assert tbrom2._hasinfmcs["inputTemperature"] is False  # temperature

    def test_instantiate_evaluation_tbrom11(self):
        """
        TEST_TB_ROM11
        Twin with 1 TBROM without named selection in settings
        -> nbTBROM = 1, NbInputField = 0, hasInputField = False, hasOutputField = True
        """
        model_filepath = TEST_TB_ROM11
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 1
        tbrom1 = twinmodel._tbroms[twinmodel.tbrom_names[0]]
        assert tbrom1.field_input_count is 0
        assert tbrom1._hasoutmcs is True
        assert twinmodel.get_named_selections(twinmodel.tbrom_names[0]) == []

    def test_instantiate_evaluation_tbrom12(self):
        """
        TEST_TB_ROM12
        Twin with 1 TBROM with two named selections in settings
        -> nbTBROM = 1, NbInputField = 1, hasInputField = True, hasOutputField = True,
        named_selections = ['Group_1', 'Group_2']
        """
        model_filepath = TEST_TB_ROM12
        twinmodel = TwinModel(model_filepath=model_filepath)
        assert twinmodel.tbrom_count is 1
        tbrom1 = twinmodel._tbroms[twinmodel.tbrom_names[0]]
        assert tbrom1.field_input_count is 1
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputTemperature"] is True
        assert twinmodel.get_named_selections(twinmodel.tbrom_names[0]) == ["Group_1", "Group_2"]

    def test_initialize_evaluation_with_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = twinmodel.tbrom_names[0]
        twinmodel.initialize_evaluation(field_inputs={romname: {"inputPressure": INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)

    def test_initialize_evaluation_with_numpy_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = twinmodel.tbrom_names[0]
        memory_snp = read_binary(INPUT_SNAPSHOT)
        twinmodel.initialize_evaluation(field_inputs={romname: {"inputPressure": memory_snp}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)

    def test_initialize_evaluation_with_input_field_exceptions(self):
        """
        Test with TEST_TB_ROM3
        Twin with 1 TBROM and 2 input fields both connected, 1 output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, True), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)

        # Raise an exception if provided rom name is not valid
        try:
            twinmodel.initialize_evaluation(field_inputs={"unknown_rom": {"unknown_infield": INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise an exception if provided field input name is not valid
        try:
            twinmodel.initialize_evaluation(
                field_inputs={twinmodel.tbrom_names[0]: {"unknown_infield": INPUT_SNAPSHOT}}
            )
        except TwinModelError as e:
            assert "[FieldName]" in str(e)

        # Raise an exception if provided snapshot path is None
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputPressure"
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: None}})
        except TwinModelError as e:
            assert "[InputSnapshotNone]" in str(e)

        # Raise en exception if provided snapshot is not string, Path or np.array
        memory_snp = read_binary(INPUT_SNAPSHOT)
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: memory_snp.tolist()}})
        except TwinModelError as e:
            assert "[InputSnapshotType]" in str(e)

        # Raise an exception if provided snapshot path does not exist
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: "unknown_snapshot_path"}})
            # exist
        except TwinModelError as e:
            assert "[InputSnapshotPath]" in str(e)

        # Raise en exception if provided snapshot is a np.array with wrong shape
        wrong_arr = np.zeros((memory_snp.shape[0], 3))
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: wrong_arr}})
        except TwinModelError as e:
            assert "[InputSnapshotShape]" in str(e)

        # Raise en exception if provided snapshot has the wrong size
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT_WRONG}})
        except TwinModelError as e:
            assert "[InputSnapshotSize]" in str(e)

        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: memory_snp[2:]}})
        except TwinModelError as e:
            assert "[InputSnapshotSize]" in str(e)

        """
        TEST_TB_ROM5
        Twin with 1 TBROM and 1 input fields connected with error, 1 output field connected
        -> nbTBROM = 1, NbInputField = 1, hasInputField = (False), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)

        # Raise an exception if field input is not connected.
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputTemperature"
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "[RomInputConnection]" in str(e)

    def test_evaluate_step_by_step_with_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputPressure"

        # Step t=0.0s
        twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

        # Step t=0.1s
        twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

        # Step t=0.2s
        twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

    def test_evaluate_step_by_step_with_numpy_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputPressure"
        memory_snp = read_binary(INPUT_SNAPSHOT)
        # Step t=0.0s
        twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: memory_snp}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

        # Step t=0.1s
        twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: memory_snp}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

        # Step t=0.2s
        twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: memory_snp}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

    def test_evaluate_step_by_step_with_input_field_exceptions(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()

        # Raise an exception if provided rom name is not valid
        romname = "unknown"
        fieldname = "unknown"
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise en exception if provided input field name is not valid
        romname = twinmodel.tbrom_names[0]
        fieldname = "unknown"
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "[FieldName]" in str(e)

        # Raise en exception if provided snapshot is None
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: None}})
        except TwinModelError as e:
            assert "[InputSnapshotNone]" in str(e)

        # Raise en exception if provided snapshot is not string, Path or np.array
        memory_snp = read_binary(INPUT_SNAPSHOT)
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: memory_snp.tolist()}})
        except TwinModelError as e:
            assert "[InputSnapshotType]" in str(e)

        # Raise en exception if provided snapshot is a np.array with wrong shape
        wrong_arr = np.zeros((memory_snp.shape[0], 3))
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: wrong_arr}})
        except TwinModelError as e:
            assert "[InputSnapshotShape]" in str(e)

        # Raise en exception if provided snapshot path does not exist
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: "unknown_path"}})
        except TwinModelError as e:
            assert "[InputSnapshotPath]" in str(e)

        # Raise an exception if provided snapshot has wrong size
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT_WRONG}})
        except TwinModelError as e:
            assert "[InputSnapshotSize]" in str(e)

        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: memory_snp}})
        except TwinModelError as e:
            assert "[InputSnapshotSize]" in str(e)

        # Raise an exception if provided field input is not connected
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputTemperature"
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "[RomInputConnection]" in str(e)

    def test_evaluate_batch_with_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputPressure"

        # Step t=0.0s
        twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

        batch_results = twinmodel.evaluate_batch(
            inputs_df=pd.DataFrame({"Time": [0.0, 0.1, 0.2]}),
            field_inputs={romname: {fieldname: [INPUT_SNAPSHOT, INPUT_SNAPSHOT, INPUT_SNAPSHOT]}},
        )

        assert np.isclose(batch_results["outField_mode_1"][0], -0.007815295084108557)
        assert np.isclose(batch_results["outField_mode_1"][1], -0.007815295084108557)
        assert np.isclose(batch_results["outField_mode_1"][2], -0.007815295084108557)

        assert np.isclose(batch_results["outField_mode_2"][0], -0.0019136501347937662)
        assert np.isclose(batch_results["outField_mode_2"][1], -0.0019136563369488168)
        assert np.isclose(batch_results["outField_mode_2"][2], -0.0019136563369488168)

        assert np.isclose(batch_results["outField_mode_3"][0], 0.0007345769427744131)
        assert np.isclose(batch_results["outField_mode_3"][1], 0.0007345833149719503)
        assert np.isclose(batch_results["outField_mode_3"][2], 0.0007345833149719503)

        assert np.isclose(batch_results["MaxDef"][0], 5.0352056308720146e-05)
        assert np.isclose(batch_results["MaxDef"][1], 5.035206128408094e-05)
        assert np.isclose(batch_results["MaxDef"][2], 5.035206128408094e-05)

    def test_evaluate_batch_with_numpy_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputPressure"
        memory_snp = read_binary(INPUT_SNAPSHOT)

        # Step t=0.0s
        twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: memory_snp}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

        batch_results = twinmodel.evaluate_batch(
            inputs_df=pd.DataFrame({"Time": [0.0, 0.1, 0.2]}),
            field_inputs={romname: {fieldname: [memory_snp, memory_snp, memory_snp]}},
        )

        assert np.isclose(batch_results["outField_mode_1"][0], -0.007815295084108557)
        assert np.isclose(batch_results["outField_mode_1"][1], -0.007815295084108557)
        assert np.isclose(batch_results["outField_mode_1"][2], -0.007815295084108557)

        assert np.isclose(batch_results["outField_mode_2"][0], -0.0019136501347937662)
        assert np.isclose(batch_results["outField_mode_2"][1], -0.0019136563369488168)
        assert np.isclose(batch_results["outField_mode_2"][2], -0.0019136563369488168)

        assert np.isclose(batch_results["outField_mode_3"][0], 0.0007345769427744131)
        assert np.isclose(batch_results["outField_mode_3"][1], 0.0007345833149719503)
        assert np.isclose(batch_results["outField_mode_3"][2], 0.0007345833149719503)

        assert np.isclose(batch_results["MaxDef"][0], 5.0352056308720146e-05)
        assert np.isclose(batch_results["MaxDef"][1], 5.035206128408094e-05)
        assert np.isclose(batch_results["MaxDef"][2], 5.035206128408094e-05)

    def test_evaluate_batch_with_input_field_exceptions(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()

        # Raise an exception if provided rom name is not valid
        romname = "unknown"
        fieldname = "unknown"
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}), field_inputs={romname: {fieldname: None}}
            )
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise an exception if provided input field name is not valid
        romname = twinmodel.tbrom_names[0]
        fieldname = "unknown"
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}), field_inputs={romname: {fieldname: []}}
            )
        except TwinModelError as e:
            assert "[FieldName]" in str(e)

        # Raise an exception if provided snapshot list is None
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}), field_inputs={romname: {fieldname: None}}
            )
        except TwinModelError as e:
            assert "[InputSnapshotListNone]" in str(e)

        # Raise an exception if provided snapshots are not string, Path or np.array
        memory_snp = read_binary(INPUT_SNAPSHOT)
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}),
                field_inputs={romname: {fieldname: [memory_snp.tolist(), memory_snp.tolist()]}},
            )
        except TwinModelError as e:
            assert "[InputSnapshotType]" in str(e)

        # Raise an exception if memory snapshot as list, same length at t_count
        short_snp = [1.0, 2]
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}),
                field_inputs={romname: {fieldname: [short_snp, short_snp]}},
            )
        except TwinModelError as e:
            assert "[InputSnapshotType]" in str(e)

        # Raise an exception if provided not as many snapshot paths as time instants
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}), field_inputs={romname: {fieldname: ["", "", ""]}}
            )
        except TwinModelError as e:
            assert "[InputSnapshotCount]" in str(e)

        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}), field_inputs={romname: {fieldname: [""]}}
            )
        except TwinModelError as e:
            assert "[InputSnapshotCount]" in str(e)

        # Raise an exception if provided snapshot path does not exist
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}),
                field_inputs={romname: {fieldname: ["unknown", "unknown"]}},
            )
        except TwinModelError as e:
            assert "[InputSnapshotPath]" in str(e)

        # Raise an exception if provided snapshot is a np.array with wrong shape
        wrong_arr = np.zeros((memory_snp.shape[0], 3))
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}),
                field_inputs={romname: {fieldname: [wrong_arr, wrong_arr]}},
            )
        except TwinModelError as e:
            assert "[InputSnapshotShape]" in str(e)

        # Raise an exception if provided snapshot has wrong size
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}),
                field_inputs={romname: {fieldname: [INPUT_SNAPSHOT_WRONG, INPUT_SNAPSHOT_WRONG]}},
            )
        except TwinModelError as e:
            assert "[InputSnapshotSize]" in str(e)

        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}),
                field_inputs={romname: {fieldname: [memory_snp[2:], memory_snp[2:]]}},
            )
        except TwinModelError as e:
            assert "[InputSnapshotSize]" in str(e)

        # Raise an exception if provided snapshots are not a list
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}), field_inputs={romname: {fieldname: INPUT_SNAPSHOT}}
            )
        except TwinModelError as e:
            assert "[InputSnapshotList]" in str(e)

        # Raise an exception if provided field input is not connected
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputTemperature"
        try:
            twinmodel.evaluate_batch(
                inputs_df=pd.DataFrame({"Time": [0.0, 1.0]}),
                field_inputs={romname: {fieldname: [INPUT_SNAPSHOT, INPUT_SNAPSHOT]}},
            )
        except TwinModelError as e:
            assert "[RomInputConnection]" in str(e)

    def test_generate_snapshot_with_tbrom_is_ok(self):
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[1]

        # Generate snapshot on disk
        snp_filepath = twinmodel.generate_snapshot(romname, True)
        snp_vec_on_disk = read_binary(snp_filepath)
        assert snp_vec_on_disk.shape[0] == 313266
        assert np.isclose(snp_vec_on_disk[0], 1.7188266861184398e-05)
        assert np.isclose(snp_vec_on_disk[-1], -1.3100502753567515e-05)

        # Generate snapshot in memory
        snp_vec_in_memory = twinmodel.generate_snapshot(romname, False)
        # snapshot in memory is ndarray with (number of points, field dimensionality)
        assert (
            snp_vec_in_memory.reshape(
                -1,
            ).shape[0]
            == snp_vec_on_disk.shape[0]
        )
        assert np.isclose(snp_vec_on_disk[0], snp_vec_in_memory[0, 0])
        assert np.isclose(snp_vec_on_disk[-1], snp_vec_in_memory[-1, -1])

        # Generate snapshot gives same results as twin_model probe
        max_snp = max(norm_vector_field(snp_vec_in_memory))
        assert np.isclose(max_snp, twinmodel.outputs["MaxDef"])

        # Generate snapshot on named selection
        # TODO LUCAS - Use another twin model with named selection smaller than whole model
        ns = twinmodel.get_named_selections(romname)
        snp_vec_ns = twinmodel.generate_snapshot(romname, False, named_selection=ns[0])
        assert (
            snp_vec_ns.reshape(
                -1,
            ).shape[0]
            == 313266
        )
        assert np.isclose(snp_vec_ns[0, 0], 1.7188266861184398e-05)
        assert np.isclose(snp_vec_ns[-1, -1], -1.3100502753567515e-05)

    def test_generate_snapshot_on_named_selection_with_tbrom_is_ok(self):
        model_filepath = TEST_TB_ROM12
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]

        # Generate snapshot on named selection
        ns = twinmodel.get_named_selections(romname)
        snp_vec_ns = twinmodel.generate_snapshot(romname, False, named_selection=ns[0])
        assert (
            snp_vec_ns.reshape(
                -1,
            ).shape[0]
            == 78594
        )
        if sys.platform != "linux":
            # TODO - Fix BUG881733
            assert np.isclose(snp_vec_ns[0, 0], 1.7188266859172047e-05)
            assert np.isclose(snp_vec_ns[-1, -1], -1.5316792773713332e-05)

    def test_generate_snapshot_with_tbrom_exceptions(self):
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = "unknown"

        # Raise an exception if twin model has not been initialized
        try:
            twinmodel.generate_snapshot(romname, False)
        except TwinModelError as e:
            assert "[Initialization]" in str(e)

        # Raise an exception if rom name is unknown
        twinmodel.initialize_evaluation()
        try:
            twinmodel.generate_snapshot(romname, False)
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise an exception if tbrom output is not connected
        romname = twinmodel.tbrom_names[1]
        try:
            twinmodel.generate_snapshot(romname, False)
        except TwinModelError as e:
            assert "[RomOutputConnection]" in str(e)

        # Raise en exception if named selection does not exist
        try:
            twinmodel.generate_snapshot(romname, False, "unknown")
        except TwinModelError as e:
            assert "[NamedSelection]" in str(e)

    def test_generate_snapshot_batch_with_tbrom_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputPressure"

        # Batch Evaluation
        twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        batch_results = twinmodel.evaluate_batch(
            inputs_df=pd.DataFrame({"Time": [0.0, 0.1, 0.2]}),
            field_inputs={romname: {fieldname: [INPUT_SNAPSHOT, INPUT_SNAPSHOT, INPUT_SNAPSHOT]}},
        )

        # Generate snapshot from batch results
        snapshot_paths = twinmodel.generate_snapshot_batch(batch_results, romname)
        assert len(snapshot_paths) == 3

        snp0 = read_binary(snapshot_paths[0])
        snp1 = read_binary(snapshot_paths[1])
        snp2 = read_binary(snapshot_paths[2])

        assert np.isclose(max(snp0), 4.4525419095601117e-05)
        assert np.isclose(max(snp1), 4.452541222688557e-05)
        assert np.isclose(max(snp2), 4.452541222688557e-05)

        max_snp0 = max(norm_vector_field(snp0))
        max_snp1 = max(norm_vector_field(snp1))
        max_snp2 = max(norm_vector_field(snp2))

        assert np.isclose(max_snp0, batch_results["MaxDef"][0])
        assert np.isclose(max_snp1, batch_results["MaxDef"][1])
        assert np.isclose(max_snp2, batch_results["MaxDef"][2])

    def test_generate_points_with_tbrom_is_ok(self):
        model_filepath = TEST_TB_ROM12
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]

        # Generate points on disk
        points_filepath = twinmodel.generate_points(romname, True)
        points_vec = read_binary(points_filepath)
        assert points_vec.shape[0] == 313266
        assert np.isclose(points_vec[0], 0.0)
        assert np.isclose(points_vec[-1], 38.919245779058635)

        # Generate points in memory
        points_vec2 = twinmodel.generate_points(romname, False)
        assert (
            points_vec.shape[0]
            == points_vec2.reshape(
                -1,
            ).shape[0]
        )
        assert np.isclose(points_vec[0], points_vec2[0, 0])
        assert np.isclose(points_vec[-1], points_vec2[-1, -1])

        # Generate points on named selection on disk
        ns = twinmodel.get_named_selections(romname)
        points_filepath_ns = twinmodel.generate_points(romname, True, named_selection=ns[0])
        points_vec_ns = read_binary(points_filepath_ns)
        assert points_vec_ns.shape[0] == 78594
        assert np.isclose(points_vec_ns[0], 0.0)
        assert np.isclose(points_vec_ns[-1], 68.18921187292435)

        # Generate points on named selection in memory
        points_vec_ns2 = twinmodel.generate_points(romname, False, named_selection=ns[0])
        assert (
            points_vec_ns.shape[0]
            == points_vec_ns2.reshape(
                -1,
            ).shape[0]
        )
        assert np.isclose(points_vec_ns[0], points_vec_ns2[0, 0])
        assert np.isclose(points_vec_ns[-1], points_vec_ns2[-1, -1])

    def test_generate_points_with_tbrom_exceptions(self):
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = "unknown"

        # Raise an exception if twin model not initialized
        try:
            twinmodel.generate_points(romname, False, "unknown")
        except TwinModelError as e:
            assert "[Initialization]" in str(e)

        twinmodel.initialize_evaluation()

        # Raise an exception if unknown rom name is given
        try:
            twinmodel.generate_points(romname, False, "unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise an exception if unknown named selection is given
        romname = twinmodel.tbrom_names[0]
        try:
            twinmodel.generate_points(romname, False, "unknown")
        except TwinModelError as e:
            assert "[NamedSelection]" in str(e)

        # Raise an exception if no point file is available
        model_filepath = TEST_TB_ROM10
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        nslist = twinmodel.get_named_selections(romname)
        try:
            twinmodel.generate_points(romname, False, nslist[0])
        except TwinModelError as e:
            assert "[GeometryFile]" in str(e)

    def test_tbrom_getters_that_do_not_need_initialization(self):
        reinit_settings()
        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files", force_download=True)
        twin = TwinModel(model_filepath=model_filepath)

        # Test rom name
        rom_name = twin.tbrom_names[0]
        assert rom_name == "ThermalROM23R1_1"

        # Test available view names
        view_names = twin.get_available_view_names(rom_name)
        assert view_names[0] == "View1"

        # Test geometry filepath
        points_filepath = twin.get_geometry_filepath(rom_name)
        assert "points.bin" in points_filepath

        # Test rom directory
        rom_dir = twin.get_rom_directory(rom_name)
        assert rom_name in rom_dir

        # Test sdk rom resources directory
        sdk_dir = twin._tbrom_resource_directory(rom_name)
        assert "resources" in sdk_dir

        # Test named selections
        ns = twin.get_named_selections(rom_name)
        assert ns[0] == "solid-part_2"

        # Test field input names
        names = twin.get_field_input_names(rom_name)
        assert names == []

    def test_tbrom_getters_exceptions_if_no_tbrom(self):
        # Raise an error if TWIN MODEL DOES NOT INCLUDE ANY TBROM
        reinit_settings()
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)

        # Test getters that do not need initialization
        try:
            twin._tbrom_resource_directory(rom_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        try:
            twin.get_geometry_filepath(rom_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        try:
            twin.get_available_view_names(rom_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        try:
            twin.get_rom_directory(rom_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        try:
            twin.get_named_selections(rom_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        try:
            twin.get_field_input_names(rom_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        # Test getters that need initialization
        twin.initialize_evaluation()
        try:
            twin.get_snapshot_filepath(rom_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        try:
            twin.get_image_filepath(rom_name="test", view_name="test")
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

    def test_tbrom_getters_exceptions_if_bad_rom_name(self):
        # Raise an error if getter is called with an unknown rom name
        reinit_settings()
        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twin = TwinModel(model_filepath=model_filepath)
        twin.initialize_evaluation()

        # Test getters that do not need initialization
        try:
            twin._tbrom_resource_directory(rom_name="unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        try:
            twin.get_geometry_filepath(rom_name="unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        try:
            twin.get_available_view_names(rom_name="unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        try:
            twin.get_rom_directory(rom_name="unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        try:
            twin.get_named_selections(rom_name="unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        try:
            twin.get_field_input_names(rom_name="unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Test getters that need initialization
        twin.initialize_evaluation()
        try:
            twin.get_snapshot_filepath(rom_name="unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        try:
            twin.get_image_filepath(rom_name="unknown", view_name="test")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

    def test_tbrom_getters_exceptions_other(self):
        reinit_settings()
        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twin = TwinModel(model_filepath=model_filepath)
        twin.initialize_evaluation()

        # Raise an error if IMAGE VIEW DOES NOT EXIST
        try:
            twin.get_image_filepath(rom_name=twin.tbrom_names[0], view_name="test")
        except TwinModelError as e:
            assert "[ViewName]" in str(e)

        # Raise an error if GEOMETRY POINT FILE HAS BEEN DELETED
        try:
            filepath = twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
            os.remove(filepath)
            twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
        except TwinModelError as e:
            assert "[GeometryFile]" in str(e)

    def test_tbrom_getters_warning(self):
        reinit_settings()
        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twin = TwinModel(model_filepath=model_filepath)
        twin.initialize_evaluation()

        # Raise a warning if SNAPSHOT FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_snapshot_filepath(rom_name=twin.tbrom_names[0], evaluation_time=1.234567)
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "[OutputSnapshotPath]" in "".join(log_str)

        # Raise a warning if IMAGE FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_image_filepath(
            rom_name=twin.tbrom_names[0],
            view_name=twin.get_available_view_names(twin.tbrom_names[0])[0],
            evaluation_time=1.234567,
        )
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "[ViewFilePath]" in "".join(log_str)

    def test_tbrom_projection_errors(self):
        reinit_settings()
        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twinmodel = TwinModel(model_filepath=model_filepath)
        mesh = pv.PolyData()
        romname = "unknown"

        # Raise an exception if twin model not initialized
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, False, "unknown")
        except TwinModelError as e:
            assert "[Initialization]" in str(e)

        twinmodel.initialize_evaluation()

        # Raise an exception if unknown rom name is given
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, False, "unknown")
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise an exception as the twin considered has not output MC connected
        romname = twinmodel.tbrom_names[0]
        nslist = twinmodel.get_named_selections(romname)
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, False, nslist[0])
        except TwinModelError as e:
            assert "[RomOutputConnection]" in str(e)

        # Raise an exception if mesh provided is not consistent
        model_filepath = download_file("ThermalTBROM_FieldInput_23R1.twin", "twin_files")
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, False, "unknown")
        except TwinModelError as e:
            assert "[PyVistaMesh]" in str(e)

        mesh = pv.read(MESH_FILE)

        # Raise an exception if unknown named selection is given
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, False, "unknown")
        except TwinModelError as e:
            assert "[NamedSelection]" in str(e)

        # Raise an exception if interpolate is True and no points file is available
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        filepath = twinmodel.get_geometry_filepath(rom_name=romname)
        os.remove(filepath)
        nslist = twinmodel.get_named_selections(romname)
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, True, nslist[0])
        except TwinModelError as e:
            assert "[GeometryFile]" in str(e)

        # Raise a warning if interpolation flag set to False and target mesh has not same size as point cloud
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        nslist = twinmodel.get_named_selections(romname)
        twinmodel.project_tbrom_on_mesh(romname, mesh, False, nslist[0])
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "Switching interpolate flag from False to True" in "".join(log_str)

        # Raise an exception if projection with masking removes all points
        model_filepath = download_file("ThermalTBROM_FieldInput_23R1.twin", "twin_files")
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        nslist = twinmodel.get_named_selections(romname)
        max_coord = twinmodel.generate_points(romname, on_disk=False).max()
        radius = max_coord / 10
        mesh = pv.Sphere(radius, (max_coord, max_coord, max_coord))
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, False, nslist[0], radius=radius, strategy="mask_points")
        except TwinModelError as e:
            assert "[TbRomInterpolation]" in str(e)

        # Raise an exception if any issue occurs during projection
        model_filepath = download_file("ThermalTBROM_FieldInput_23R1.twin", "twin_files")
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        nslist = twinmodel.get_named_selections(romname)
        try:
            twinmodel.project_tbrom_on_mesh(romname, mesh, False, nslist[0])
        except TwinModelError as e:
            assert "MeshProjection" in str(e)

    def test_tbrom_get_output_field_errors(self):
        reinit_settings()
        romname = "unknown"
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twinmodel = TwinModel(model_filepath=model_filepath)

        # Raise an exception if no tbrom available in the twin
        try:
            twinmodel.get_tbrom_output_field(romname)
        except TwinModelError as e:
            assert "[NoRom]" in str(e)

        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()

        # Raise an exception if unknown rom name is given
        try:
            twinmodel.get_tbrom_output_field(romname)
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise an exception if the twin considered has not output MC connected
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        try:
            twinmodel.get_tbrom_output_field(romname)
        except TwinModelError as e:
            assert "[RomOutputConnection]" in str(e)

        # Raise an exception if any issue occurs during the API execution
        model_filepath = download_file("ThermalTBROM_FieldInput_23R1.twin", "twin_files")
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        try:
            twinmodel.get_tbrom_output_field(romname)
        except TwinModelError as e:
            assert "GetPointsData" in str(e)

    def test_tbrom_new_instantiation_without_points(self):
        model_filepath = TEST_TB_ROM3
        try:
            twinmodel = TwinModel(model_filepath=model_filepath)  # instantiation should be fine without points
            romname = twinmodel.tbrom_names[0]
            twinmodel.get_tbrom_output_field(romname)  # retrieving the output field pyvista object should raise an
            # error since there is no point file
        except TwinModelError as e:
            assert "GeometryFile" in str(e)

    def test_read_write_api(self):
        scalar_field = np.array([1.0, 2.0, 3.0, 5.0])
        write_binary(os.path.join(os.path.dirname(__file__), "data", "snapshot_scalar.bin"), scalar_field)
        vector_field = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 3.0], [5.0, 3.0, 3.0], [5.0, 5.0, 6.0]])
        write_binary(os.path.join(os.path.dirname(__file__), "data", "snapshot_vector.bin"), vector_field)
        scalar_field_read = read_binary(os.path.join(os.path.dirname(__file__), "data", "snapshot_scalar.bin"))
        vector_field_read = read_binary(os.path.join(os.path.dirname(__file__), "data", "snapshot_vector.bin"))
        assert len(scalar_field_read) is 4
        assert len(vector_field_read) is 3 * 4

    def test_read_write_api_dtype(self):
        # Test for issue #321
        scalar_field = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.int64)
        write_binary(os.path.join(os.path.dirname(__file__), "data", "snapshot_scalar.bin"), scalar_field)
        scalar_field_read = read_binary(os.path.join(os.path.dirname(__file__), "data", "snapshot_scalar.bin"))
        assert np.all(np.equal(scalar_field, scalar_field_read))

    def test_snapshot_to_array_api(self):
        tensor_path = os.path.join(os.path.dirname(__file__), "data", "snapshot_tensor.bin")
        tensor_field = np.array(
            [
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
            ]
        )
        write_binary(tensor_path, tensor_field)
        geometry_path = os.path.join(os.path.dirname(__file__), "data", "geometry_vector.bin")
        geometry_field = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 3.0], [5.0, 3.0, 3.0], [5.0, 5.0, 6.0]])
        write_binary(geometry_path, geometry_field)
        vector_field_read = snapshot_to_array(tensor_path, geometry_path)
        assert vector_field_read.shape[0] == 4
        assert vector_field_read.shape[1] == 9

    def test_snapshot_to_array_api_mismatch(self):
        # Snapshot of length 24
        tensor_path = os.path.join(os.path.dirname(__file__), "data", "snapshot_tensor.bin")
        tensor_field = np.array(
            [
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
                [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
            ]
        )
        write_binary(tensor_path, tensor_field)

        # Snapshot of length 18 is not divisible by 4 points
        wrong_size_tensor = os.path.join(os.path.dirname(__file__), "data", "snapshot_wrong.bin")
        tensor_field = np.array(
            [[1.0, 2.0, 3.0, 5.0, 7.0, 11.0], [1.0, 2.0, 3.0, 5.0, 7.0, 11.0], [1.0, 2.0, 3.0, 5.0, 7.0, 11.0]]
        )
        write_binary(wrong_size_tensor, tensor_field)

        # Snapshot of length 12
        geometry_path = os.path.join(os.path.dirname(__file__), "data", "geometry_vector.bin")
        geometry_field = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 3.0], [5.0, 3.0, 3.0], [5.0, 5.0, 6.0]])
        write_binary(geometry_path, geometry_field)

        # Snapshot of length 8 is not divisible by 3
        wrong_geometry = os.path.join(os.path.dirname(__file__), "data", "geometry_wrong.bin")
        geometry_field = np.array([[1.0, 1.0], [1.0, 2.0], [5.0, 3.0], [5.0, 5.0]])
        write_binary(wrong_geometry, geometry_field)

        try:
            vector_field_read = snapshot_to_array(wrong_size_tensor, geometry_path)
        except ValueError as e:
            assert "Field snapshot length 18 must be divisible by the number of points 4." in str(e)
        try:
            vector_field_read = snapshot_to_array(tensor_field, wrong_geometry)
        except ValueError as e:
            assert "Geometry snapshot length must be divisible by 3." in str(e)

    def test_tbrom_tensor_field(self):
        model_filepath = TEST_TB_ROM_TENSOR
        [nsidslist, dimensionality, outputname, unit, timegrid] = tbrom._read_settings(
            model_filepath
        )  # instantiation should be fine without points
        assert int(dimensionality[0]) is 6

    def test_tbrom_fix_bug_1168769(self):
        model_filepath = TEST_TB_ROM_NDOF
        try:
            twinmodel = TwinModel(model_filepath=model_filepath)
        except TwinModelError as e:
            assert "cannot reshape array" not in str(e)

    def test_tbrom_srb_constraints(self):
        model_filepath = TEST_TB_ROM_CONSTRAINTS
        twinmodel = TwinModel(model_filepath=model_filepath)
