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

"""
TEST_TB_ROM_DROM
Twin with 1 dynamic field TBROM
"""
TEST_TB_ROM_DROM = os.path.join(os.path.dirname(__file__), "data", "twin_field_dyna_rom_cc.twin")


def norm_vector_field(field: list):
    """Compute the norm of a vector field."""
    vec = field.reshape((-1, 3))
    return np.sqrt((vec * vec).sum(axis=1))


class TestTbRom:

def test_tbrom_parametric_field_history(self):
        model_filepath = TEST_TB_ROM_CONSTRAINTS
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = twinmodel.tbrom_names[0]

        try:
            timegrid = twinmodel.get_tbrom_time_grid(romname)
        except TwinModelError as e:
            assert "not a parametric field history ROM" in str(e)

        model_filepath = TEST_TB_PFIELD_HISTORY
        twinmodel = TwinModel(model_filepath=model_filepath)
        romname = twinmodel.tbrom_names[0]

        timegrid = twinmodel.get_tbrom_time_grid(romname)

        assert len(timegrid) == 17

        assert twinmodel._tbroms[romname].isparamfieldhist == True

        twinmodel.initialize_evaluation()

        field_data = twinmodel.get_tbrom_output_field(romname)
        maxt0 = max(field_data[f"{twinmodel._tbroms[romname].field_output_name}-normed"])

        twinmodel.evaluate_step_by_step(100.0)
        field_data = twinmodel.get_tbrom_output_field(romname)
        maxt100 = max(field_data[f"{twinmodel._tbroms[romname].field_output_name}-normed"])

        twinmodel.evaluate_step_by_step(150.0)
        field_data = twinmodel.get_tbrom_output_field(romname)
        maxt250 = max(field_data[f"{twinmodel._tbroms[romname].field_output_name}-normed"])

        twinmodel.evaluate_step_by_step(100.0)
        field_data = twinmodel.get_tbrom_output_field(romname)
        maxt300 = max(field_data[f"{twinmodel._tbroms[romname].field_output_name}-normed"])

        log_file = get_pytwin_log_file()
        with open(log_file, "r") as f:
            lines = f.readlines()
        msg = "is larger than last time point"
        assert "".join(lines).count(msg) == 1

        # if sys.platform != "linux":
        assert np.isclose(maxt0, 0.8973744667566537)
        #    assert np.isclose(maxt100, 1.685669230751107)
        #    assert np.isclose(maxt250, 5.635884051349383)
        # assert np.isclose(maxt250, maxt300)

    # def test_tbrom_dynarom(self): #https://tfs.ansys.com:8443/tfs/ANSYS_Development/Portfolio/_workitems/edit/1362120
    #    model_filepath = TEST_TB_ROM_DROM
    #    twinmodel = TwinModel(model_filepath=model_filepath)
    #    romname = twinmodel.tbrom_names[0]
    #    assert twinmodel._tbroms[romname]._hasoutmcs is True
