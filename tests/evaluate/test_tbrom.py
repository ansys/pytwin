import os

from pytwin import TwinModel, TwinModelError

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

INPUT_SNAPSHOT = os.path.join(os.path.dirname(__file__), "data", "input_snapshot.bin")
INPUT_SNAPSHOT_WRONG = os.path.join(os.path.dirname(__file__), "data", "input_snapshot_wrong.bin")


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
        model_filepath = TEST_TB_ROM1  # Twin with no TBROM -> nbTBROM = 0
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 0

        model_filepath = TEST_TB_ROM2  # Twin with 1 TBROM and 2 input fields but no input field connected, no output
        # field connected
        # -> nbTBROM = 1, NbInputField = 2, hasInputField = (False, False), hasOutputField = False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbroms[tbromname2field]
        assert tbrom1.numberinputfields is 2
        assert tbrom1.hasoutmcs is False
        assert tbrom1.hasinfmcs(field1) is False
        assert tbrom1.hasinfmcs(field2) is False

        model_filepath = TEST_TB_ROM3  # Twin with 1 TBROM and 2 input fields both connected, 1 output
        # field connected
        # -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, True), hasOutputField = True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbroms[tbromname2field]
        assert tbrom1.numberinputfields is 2
        assert tbrom1.hasoutmcs is True
        assert tbrom1.hasinfmcs(field1) is True
        assert tbrom1.hasinfmcs(field2) is True

        model_filepath = TEST_TB_ROM4  # Twin with 1 TBROM and 2 input fields, 1st partially connected, second
        # fully connected, 1 output field connected
        # -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, False), hasOutputField = True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbroms[tbromname2field]
        assert tbrom1.numberinputfields is 2
        assert tbrom1.hasoutmcs is True
        assert tbrom1.hasinfmcs(field1) is True  # pressure
        assert tbrom1.hasinfmcs(field2) is False  # temperature

        model_filepath = TEST_TB_ROM5  # Twin with 1 TBROM and 1 input fields connected with error, 1 output
        # field connected
        # -> nbTBROM = 1, NbInputField = 1, hasInputField = (False), hasOutputField = True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 1
        tbrom1 = twinmodel._tbroms[tbromname1field]
        assert tbrom1.numberinputfields is 1
        field = tbrom1.nameinputfields[0]
        assert tbrom1.hasoutmcs is True
        assert tbrom1.hasinfmcs(field) is False

        model_filepath = TEST_TB_ROM6  # Twin with 2 TBROM, 1st has 2 input field connected, 1 output field connected,
        # second has no connection
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, True) and False, hasOutputField = True and False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbroms[tbromname2field]  # tbrom with 2 field
        tbrom2 = twinmodel._tbroms[tbromname1field]  # tbrom with 1 field
        assert tbrom1.numberinputfields is 2
        assert tbrom2.numberinputfields is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasoutmcs is True
        assert tbrom1.hasinfmcs(tb1field1) is True  # pressure
        assert tbrom1.hasinfmcs(tb1field2) is True  # temperature
        tb2field1 = field2
        assert tbrom2.hasoutmcs is False
        assert tbrom2.hasinfmcs(tb2field1) is False

        model_filepath = TEST_TB_ROM7  # Twin with 2 TBROM, 1st has 2 input field connected with 1st field with errors,
        # 1 output field connected, second has 1 input field connected and 1 output field connected
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, False) and True, hasOutputField = True and True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbroms[tbromname2field]  # tbrom with 2 field
        tbrom2 = twinmodel._tbroms[tbromname1field]  # tbrom with 1 field
        assert tbrom1.numberinputfields is 2
        assert tbrom2.numberinputfields is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasoutmcs is True
        assert tbrom1.hasinfmcs(tb1field1) is True  # pressure
        assert tbrom1.hasinfmcs(tb1field2) is False  # temperature
        tb2field1 = field2
        assert tbrom2.hasoutmcs is True
        assert tbrom2.hasinfmcs(tb2field1) is True

        model_filepath = TEST_TB_ROM8  # Twin with 2 TBROM, 1st has 2 input field connected,
        # 1 output field connected, second has 1 input field connected and 1 output field connected
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, True) and True, hasOutputField = True and True
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbroms[tbromname2field]  # tbrom with 2 field
        tbrom2 = twinmodel._tbroms[tbromname1field]  # tbrom with 1 field
        assert tbrom1.numberinputfields is 2
        assert tbrom2.numberinputfields is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasoutmcs is True
        assert tbrom1.hasinfmcs(tb1field1) is True  # pressure
        assert tbrom1.hasinfmcs(tb1field2) is True  # temperature
        tb2field1 = field2
        assert tbrom2.hasoutmcs is True
        assert tbrom2.hasinfmcs(tb2field1) is True

        model_filepath = TEST_TB_ROM9  # Twin with 2 TBROM, 1st has 2 input field connected with 1st field with errors,
        # 1 output field connected, second has 1 input field connected and 1 output field connected with error
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, False) and True, hasOutputField = True and False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.nb_tbrom is 2
        tbrom1 = twinmodel._tbroms[tbromname2field]  # tbrom with 2 field
        tbrom2 = twinmodel._tbroms[tbromname1field]  # tbrom with 1 field
        assert tbrom1.numberinputfields is 2
        assert tbrom2.numberinputfields is 1
        tb1field1 = field1
        tb1field2 = field2
        assert tbrom1.hasoutmcs is True
        assert tbrom1.hasinfmcs(tb1field1) is True  # pressure
        assert tbrom1.hasinfmcs(tb1field2) is False  # temperature
        tb2field1 = field2
        assert tbrom2.hasoutmcs is False
        assert tbrom2.hasinfmcs(tb2field1) is True

    def test_initialization_Twin_with_TbRom_inputField(self):
        model_filepath = TEST_TB_ROM3  # Twin with 1 TBROM and 2 input fields both connected, 1 output
        # field connected
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = "test"  # -> exception that rom name provided is not valid
        fieldname = "test"
        try:
            twinmodel.initialize_evaluation(inputfields={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "rom name provided" in str(e)

        romname = twinmodel.tbrom_names[0]
        fieldname = "test"  # -> exception that field name provided is not valid
        try:
            twinmodel.initialize_evaluation(inputfields={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "field name provided" in str(e)

        model_filepath = TEST_TB_ROM5  # Twin with 1 TBROM and 1 input fields connected with error, 1 output
        # field connected
        # -> exception that TBROM inputs are not properly connected to Twin inputs and therefore no field projection
        # can be performed
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_rom_inputfieldsnames(romname)[0]
        try:
            twinmodel.initialize_evaluation(inputfields={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "no common inputs" in str(e)

        model_filepath = TEST_TB_ROM3  # Twin with 1 TBROM and 2 input fields both connected, 1 output
        # field connected
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_rom_inputfieldsnames(romname)[0]
        try:
            twinmodel.initialize_evaluation(inputfields={romname: {fieldname: None}})  # -> not valid snapshot path
        except TwinModelError as e:
            assert "not a valid path" in str(e)

        try:
            twinmodel.initialize_evaluation(inputfields={romname: {fieldname: "test"}})  # -> snapshot path does not
            # exist
        except TwinModelError as e:
            assert "snapshot path does not exist" in str(e)

        try:
            twinmodel.initialize_evaluation(inputfields={romname: {fieldname: INPUT_SNAPSHOT_WRONG}})  # -> wrong
            # snapshot size
        except TwinModelError as e:
            assert "is not consistent with the input field size" in str(e)

        try:
            twinmodel.initialize_evaluation(
                inputfields={romname: {fieldname: INPUT_SNAPSHOT}}
            )  # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

    def test_step_by_step_Twin_with_TbRom_inputField(self):
        model_filepath = TEST_TB_ROM3  # Twin with 1 TBROM and 2 input fields both connected, 1 output
        # field connected
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = "test"  # -> exception that rom name provided is not valid
        fieldname = "test"
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, inputfields={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "rom name provided" in str(e)

        romname = twinmodel.tbrom_names[0]
        fieldname = "test"  # -> exception that field name provided is not valid
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, inputfields={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "field name provided" in str(e)

        model_filepath = TEST_TB_ROM5  # Twin with 1 TBROM and 1 input fields connected with error, 1 output
        # field connected
        # -> exception that TBROM inputs are not properly connected to Twin inputs and therefore no field projection
        # can be performed
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_rom_inputfieldsnames(romname)[0]
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, inputfields={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "no common inputs" in str(e)

        model_filepath = TEST_TB_ROM3  # Twin with 1 TBROM and 2 input fields both connected, 1 output
        # field connected
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_rom_inputfieldsnames(romname)[0]
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, inputfields={romname: {fieldname: None}})  # -> not valid
            # snapshot path
        except TwinModelError as e:
            assert "not a valid path" in str(e)

        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, inputfields={romname: {fieldname: "test"}})  # -> snapshot
            # path does not
            # exist
        except TwinModelError as e:
            assert "snapshot path does not exist" in str(e)

        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, inputfields={romname: {fieldname: INPUT_SNAPSHOT_WRONG}})
            # -> wrong snapshot size
        except TwinModelError as e:
            assert "is not consistent with the input field size" in str(e)

        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, inputfields={romname: {fieldname: INPUT_SNAPSHOT}})
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

    def test_snapshot_generation_Twin_with_TbRom_inputField(self):
        model_filepath = TEST_TB_ROM9  # Twin with 2 TBROM, 1st has 2 input field connected with 1st field with errors,
        # 1 output field connected, second has 1 input field connected and 1 output field connected with error
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, False) and True, hasOutputField = True and False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = "test"
        try:
            fieldresults = twinmodel.snapshot_generation(romname, False)
            # -> wrong rom name
        except TwinModelError as e:
            assert "rom name provided" in str(e)

        romname = twinmodel.tbrom_names[0]
        try:
            fieldresults = twinmodel.snapshot_generation(romname, False)
            # -> first tbrom has outputs connection error
        except TwinModelError as e:
            assert "no common outputs" in str(e)

        romname = twinmodel.tbrom_names[1]
        nslist = twinmodel.get_rom_nslist(romname)
        try:
            fieldresults = twinmodel.snapshot_generation(romname, False, nslist[0])
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.snapshot_generation(romname, False)
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.snapshot_generation(romname, True, nslist[0])
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.snapshot_generation(romname, True)
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.snapshot_generation(romname, False, "test")
            # -> "test" is not a valid named selection
        except TwinModelError as e:
            assert "provided named selection" in str(e)

    def test_points_generation_Twin_with_TbRom_inputField(self):
        model_filepath = TEST_TB_ROM9  # Twin with 2 TBROM, 1st has 2 input field connected with 1st field with errors,
        # 1 output field connected, second has 1 input field connected and 1 output field connected with error
        # -> nbTBROM = 2, NbInputField = (2,1), hasInputField = (True, False) and True, hasOutputField = True and False
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = "test"
        try:
            fieldresults = twinmodel.points_generation(romname, False)
            # -> wrong rom name
        except TwinModelError as e:
            assert "rom name provided" in str(e)

        romname = twinmodel.tbrom_names[0]
        try:
            fieldresults = twinmodel.points_generation(romname, False)
            # -> first tbrom has outputs connection error -> working as expected for points generation
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        romname = twinmodel.tbrom_names[1]
        nslist = twinmodel.get_rom_nslist(romname)
        try:
            fieldresults = twinmodel.points_generation(romname, False, nslist[0])
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.points_generation(romname, False)
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.points_generation(romname, True, nslist[0])
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.points_generation(romname, True)
            # -> working as expected
        except TwinModelError as e:
            print(str(e))
            assert "[TwinModelError]" not in str(e)

        try:
            fieldresults = twinmodel.points_generation(romname, False, "test")
            # -> "test" is not a valid named selection
        except TwinModelError as e:
            assert "provided named selection" in str(e)

    def test_points_generation_Twin_with_TbRom_noPointFile(self):
        model_filepath = TEST_TB_ROM10  # Twin with 1 TBROM but with 3D disabled at export time -> points file not
        # available
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        try:
            fieldresults = twinmodel.points_generation(romname, False)
            # -> point file available
        except TwinModelError as e:
            assert "not find the geometry file" in str(e)
