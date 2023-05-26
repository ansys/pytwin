import os
import sys
import time

import pandas as pd
import pytest
from pytwin import TwinModel, TwinModelError, download_file
from pytwin.evaluate.tbrom import TbRom
from pytwin.settings import get_pytwin_log_file, get_pytwin_logger, get_pytwin_working_dir, modify_pytwin_working_dir

from tests.utilities import compare_dictionary

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
DYNAROM_HX_23R1 = os.path.join(os.path.dirname(__file__), "data", "HX_scalarDRB_23R1_other.twin")
RC_HEAT_CIRCUIT_23R1 = os.path.join(os.path.dirname(__file__), "data", "RC_heat_circuit_23R1.twin")

UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")

TEST_TB_ROM1 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_1.twin")
TEST_TB_ROM2 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_2.twin")
TEST_TB_ROM3 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_3.twin")
TEST_TB_ROM4 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_4.twin")
TEST_TB_ROM5 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_5.twin")
TEST_TB_ROM6 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_6.twin")
TEST_TB_ROM7 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_7.twin")
TEST_TB_ROM8 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_8.twin")
TEST_TB_ROM9 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_9.twin")
TEST_TB_ROM10 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_10.twin")
TEST_TB_ROM11 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_11.twin")
TEST_TB_ROM12 = os.path.join(os.path.dirname(__file__), "data", "twin_tbrom_12.twin")


def reinit_settings():
    import shutil

    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


class TestTbRom:
    def test_instantiation_initialization_Twin_with_TbRom(self):
        tbromname1field = "test23R1_1"
        tbromname2field = "test23R1_2f2"
        field1 = "inputPressure"
        field2 = "inputTemperature"
        model_filepath = TEST_TB_ROM1 # Twin with no TBROM -> nbTBROM = 0
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 0

        model_filepath = TEST_TB_ROM2 # Twin with 1 TBROM and 2 input fields but no input field connected, no output
        # field connected
        # -> nbTBROM = 1, NbInputField = 2, hasInputField = (False, False), hasOutputField = False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbrom[tbromname2field]
        assert tbrom1.NumberInputField is 2
        assert tbrom1.hasOutputModeCoefficients is False
        assert tbrom1.hasInputFieldsModeCoefficients(field1) is False
        assert tbrom1.hasInputFieldsModeCoefficients(field2) is False

        model_filepath = TEST_TB_ROM3 # Twin with 1 TBROM and 2 input fields both connected, 1 output
        # field connected
        # -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, True), hasOutputField = True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbrom[tbromname2field]
        assert tbrom1.NumberInputField is 2
        assert tbrom1.hasOutputModeCoefficients is True
        assert tbrom1.hasInputFieldsModeCoefficients(field1) is True
        assert tbrom1.hasInputFieldsModeCoefficients(field2) is True

        model_filepath = TEST_TB_ROM4 # Twin with 1 TBROM and 2 input fields, 1st partially connected, second
        # fully connected, 1 output field connected
        # -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, False), hasOutputField = True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbrom[tbromname2field]
        assert tbrom1.NumberInputField is 2
        assert tbrom1.hasOutputModeCoefficients is True
        assert tbrom1.hasInputFieldsModeCoefficients(field1) is True # pressure
        assert tbrom1.hasInputFieldsModeCoefficients(field2) is False # temperature

        model_filepath = TEST_TB_ROM5 # Twin with 1 TBROM and 1 input fields connected with error, 1 output
        # field connected
        # -> nbTBROM = 1, NbInputField = 1, hasInputField = (False), hasOutputField = True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbrom[tbromname1field]
        assert tbrom1.NumberInputField is 1
        field = tbrom1.NameInputFields[0]
        assert tbrom1.hasOutputModeCoefficients is True
        assert tbrom1.hasInputFieldsModeCoefficients(field) is False

        model_filepath = TEST_TB_ROM6 # Twin with 2 TBROM, 1st has 2 input field connected, 1 output field connected,
        # second has no connection
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, True) and False, hasOutputField = True and False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbrom[tbromname2field] # tbrom with 2 field
        tbrom2 = twinmodel._tbrom[tbromname1field] # tbrom with 1 field
        assert tbrom1.NumberInputField is 2
        assert tbrom2.NumberInputField is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasOutputModeCoefficients is True
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field1) is True # pressure
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field2) is True # temperature
        tb2field1 = field2
        assert tbrom2.hasOutputModeCoefficients is False
        assert tbrom2.hasInputFieldsModeCoefficients(tb2field1) is False

        model_filepath = TEST_TB_ROM7 # Twin with 2 TBROM, 1st has 2 input field connected with 1st field with errors,
        # 1 output field connected, second has 1 input field connected and 1 output field connected
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, False) and True, hasOutputField = True and True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbrom[tbromname2field] # tbrom with 2 field
        tbrom2 = twinmodel._tbrom[tbromname1field] # tbrom with 1 field
        assert tbrom1.NumberInputField is 2
        assert tbrom2.NumberInputField is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasOutputModeCoefficients is True
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field1) is True # pressure
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field2) is False # temperature
        tb2field1 = field2
        assert tbrom2.hasOutputModeCoefficients is True
        assert tbrom2.hasInputFieldsModeCoefficients(tb2field1) is True

        model_filepath = TEST_TB_ROM8 # Twin with 2 TBROM, 1st has 2 input field connected,
        # 1 output field connected, second has 1 input field connected and 1 output field connected
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, True) and True, hasOutputField = True and True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbrom[tbromname2field] # tbrom with 2 field
        tbrom2 = twinmodel._tbrom[tbromname1field] # tbrom with 1 field
        assert tbrom1.NumberInputField is 2
        assert tbrom2.NumberInputField is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasOutputModeCoefficients is True
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field1) is True # pressure
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field2) is True # temperature
        tb2field1 = field2
        assert tbrom2.hasOutputModeCoefficients is True
        assert tbrom2.hasInputFieldsModeCoefficients(tb2field1) is True

        model_filepath = TEST_TB_ROM9 # Twin with 2 TBROM, 1st has 2 input field connected with 1st field with errors,
        # 1 output field connected, second has 1 input field connected and 1 output field connected with error
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, False) and True, hasOutputField = True and False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbrom[tbromname2field] # tbrom with 2 field
        tbrom2 = twinmodel._tbrom[tbromname1field] # tbrom with 1 field
        assert tbrom1.NumberInputField is 2
        assert tbrom2.NumberInputField is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasOutputModeCoefficients is True
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field1) is True # pressure
        assert tbrom1.hasInputFieldsModeCoefficients(tb1field2) is False # temperature
        tb2field1 = field2
        assert tbrom2.hasOutputModeCoefficients is False
        assert tbrom2.hasInputFieldsModeCoefficients(tb2field1) is True
