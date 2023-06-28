import os

import numpy as np
from pytwin import TwinModel, TwinModelError
from pytwin.evaluate.tbrom import TbRom

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
DYNAROM_HX_23R1 = os.path.join(os.path.dirname(__file__), "data", "HX_scalarDRB_23R1_other.twin")
RC_HEAT_CIRCUIT_23R1 = os.path.join(os.path.dirname(__file__), "data", "RC_heat_circuit_23R1.twin")

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

INPUT_SNAPSHOT = os.path.join(os.path.dirname(__file__), "data", "input_snapshot.bin")
INPUT_SNAPSHOT_WRONG = os.path.join(os.path.dirname(__file__), "data", "input_snapshot_wrong.bin")


class TestTbRom:
    def test_initialize_evaluation_tbrom1(self):
        """
        TEST_TB_ROM1
        Twin with no TBROM -> nbTBROM = 0
        """
        model_filepath = TEST_TB_ROM1
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.tbrom_count is 0

    def test_initialize_evaluation_tbrom2(self):
        """
        TEST_TB_ROM2
        Twin with 1 TBROM and 2 input fields but no input field connected, no output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (False, False), hasOutputField = False
        """
        model_filepath = TEST_TB_ROM2
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 2
        assert tbrom1._hasoutmcs is False
        assert tbrom1._hasinfmcs["inputPressure"] is False
        assert tbrom1._hasinfmcs["inputTemperature"] is False

    def test_initialize_evaluation_tbrom3(self):
        """
        TEST_TB_ROM3
        Twin with 1 TBROM and 2 input fields both connected, 1 output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, True), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 2
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputPressure"] is True
        assert tbrom1._hasinfmcs["inputTemperature"] is True

    def test_initialize_evaluation_tbrom4(self):
        """
        TEST_TB_ROM4
        Twin with 1 TBROM and 2 input fields, 1st partially connected, second fully connected, 1 output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, False), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM4
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 2
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputPressure"] is True
        assert tbrom1._hasinfmcs["inputTemperature"] is False

    def test_initialize_evaluation_tbrom5(self):
        """
        TEST_TB_ROM5
        Twin with 1 TBROM and 1 input fields connected with error, 1 output field connected
        -> nbTBROM = 1, NbInputField = 1, hasInputField = (False), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        assert twinmodel.tbrom_count is 1
        name = twinmodel.tbrom_names[0]
        tbrom1 = twinmodel._tbroms[name]
        assert tbrom1.field_input_count is 1
        assert tbrom1._hasoutmcs is True
        assert tbrom1._hasinfmcs["inputTemperature"] is False

    def test_initialize_evaluation_tbrom6(self):
        """
        TEST_TB_ROM6
        Twin with 2 TBROM, 1st has no connection, second has 2 input field connected, 1 output field connected,
        -> nbTBROM = 2, NbInputField = (1, 2), hasInputField = False and (True, True), hasOutputField = False and True
        """
        model_filepath = TEST_TB_ROM6
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
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

    def test_initialize_evaluation_tbrom7(self):
        """
        TEST_TB_ROM7
        Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected,
        second has 2 input field connected with 1st field with errors, 1 output field connected
        -> nbTBROM = 2, NbInputField = (1,2), hasInputField = True and (True, False), hasOutputField = True and True
        """
        model_filepath = TEST_TB_ROM7
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
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

    def test_initialize_evaluation_tbrom8(self):
        """
        TEST_TB_ROM8
        Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected,
        second has 2 input field connected, 1 output field connected
        -> nbTBROM = 2, NbInputField = (1,2), hasInputField = (True, True) and True, hasOutputField = True and True
        """
        model_filepath = TEST_TB_ROM8
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
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

    def test_initialize_evaluation_tbrom9(self):
        """
        TEST_TB_ROM9
        Twin with 2 TBROM, 1st has 1 input field connected and 1 output field connected with error,
        second has 2 input field connected with second field with errors, 1 output field connected
        -> nbTBROM = 2, NbInputField = (1,2), hasInputField = True and (True, False), hasOutputField = False and True
        """
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
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

    def test_initialize_evaluation_with_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.002555283807970279)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0013938020797122038)

    def test_initialize_evaluation_with_input_field_exceptions(self):
        """
        Test with TEST_TB_ROM3
        Twin with 1 TBROM and 2 input fields both connected, 1 output field connected
        -> nbTBROM = 1, NbInputField = 2, hasInputField = (True, True), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()

        # Raise an exception if provided rom name is not valid
        try:
            twinmodel.initialize_evaluation(field_inputs={"unknown_rom": {"unknown_infield": INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "rom name provided" in str(e)

        # Raise an exception if provided field input name is not valid
        try:
            twinmodel.initialize_evaluation(
                field_inputs={twinmodel.tbrom_names[0]: {"unknown_infield": INPUT_SNAPSHOT}}
            )
        except TwinModelError as e:
            assert "field name provided" in str(e)

        # Raise an exception if provided snapshot path is None
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: None}})
        except TwinModelError as e:
            assert "not a valid path" in str(e)

        # Raise an exception if provided snapshot path does not exist
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: "unknown_snapshot_path"}})
            # exist
        except TwinModelError as e:
            assert "snapshot path does not exist" in str(e)

        # Raise en exception if provided snapshot has the wrong size
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT_WRONG}})
        except TwinModelError as e:
            assert "is not consistent with the input field size" in str(e)

        """
        TEST_TB_ROM5
        Twin with 1 TBROM and 1 input fields connected with error, 1 output field connected
        -> nbTBROM = 1, NbInputField = 1, hasInputField = (False), hasOutputField = True
        """
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()

        # Raise an exception if field input is not connected.
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "no common inputs" in str(e)

    def test_evaluate_step_by_step_with_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 0.0)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], 0.0)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.002555283807970279)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0013938020797122038)

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
            assert "rom name provided" in str(e)

        # Raise en exception if provided input field name is not valid
        romname = twinmodel.tbrom_names[0]
        fieldname = "unknown"
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "field name provided" in str(e)

        # Raise en exception if provided snapshot path is None
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: None}})
        except TwinModelError as e:
            assert "not a valid path" in str(e)

        # Raise en exception if provided snapshot path does not exist
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: "unknown_path"}})
        except TwinModelError as e:
            assert "snapshot path does not exist" in str(e)

        # Raise an exception if provided snapshot has wrong size
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT_WRONG}})
        except TwinModelError as e:
            assert "is not consistent with the input field size" in str(e)

        # Raise an exception if provided field input is not connected
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "no common inputs" in str(e)

    def test_generate_snapshot_with_tbrom_is_ok(self):
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[1]

        # Generate snapshot on disk
        snp_filepath = twinmodel.generate_snapshot(romname, True)
        snp_vec_on_disk = TbRom._read_binary(snp_filepath)
        assert len(snp_vec_on_disk) == 313266
        assert np.isclose(snp_vec_on_disk[0], 1.7188266861184398e-05)
        assert np.isclose(snp_vec_on_disk[-1], -1.3100502753567515e-05)

        # Generate snapshot in memory
        snp_vec_in_memory = twinmodel.generate_snapshot(romname, False)
        assert len(snp_vec_in_memory) == len(snp_vec_on_disk)
        assert np.isclose(snp_vec_on_disk[0], snp_vec_in_memory[0])
        assert np.isclose(snp_vec_on_disk[-1], snp_vec_in_memory[-1])

        # Generate snapshot on named selection
        # TODO LUCAS - Use another twin model with named selection smaller than whole model
        ns = twinmodel.get_named_selections(romname)
        snp_vec_ns = twinmodel.generate_snapshot(romname, False, named_selection=ns[0])
        assert len(snp_vec_ns) == 313266
        assert np.isclose(snp_vec_ns[0], 1.7188266861184398e-05)
        assert np.isclose(snp_vec_ns[-1], -1.3100502753567515e-05)

    def test_generate_snapshot_with_tbrom_exceptions(self):
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()

        # Raise an exception if rom name is unknown
        romname = "unknown"
        try:
            twinmodel.generate_snapshot(romname, False)
        except TwinModelError as e:
            assert "rom name provided" in str(e)

        # Raise an exception if tbrom output is not connected
        romname = twinmodel.tbrom_names[1]
        try:
            twinmodel.generate_snapshot(romname, False)
        except TwinModelError as e:
            assert "no common outputs" in str(e)

        # Raise en exception if named selection does not exist
        try:
            twinmodel.generate_snapshot(romname, False, "unknown")
        except TwinModelError as e:
            assert "provided named selection" in str(e)

    def test_generate_points_with_tbrom_is_ok(self):
        # TODO LUCAS - Use another twin model with named selection smaller than whole model
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[1]
        nslist = twinmodel.get_named_selections(romname)

        # Generate points on disk
        points_filepath = twinmodel.generate_points(nslist[0], romname, True)
        points_vec = TbRom._read_binary(points_filepath)
        assert len(points_vec) == 313266
        assert np.isclose(points_vec[0], 0.0)
        assert np.isclose(points_vec[-1], 38.919245779058635)

        # Generate points in memory
        points_vec2 = twinmodel.generate_points(nslist[0], romname, False)
        assert len(points_vec) == len(points_vec2)
        assert np.isclose(points_vec[0], points_vec2[0])
        assert np.isclose(points_vec[-1], points_vec2[-1])

    def test_generate_points_with_tbrom_exceptions(self):
        model_filepath = TEST_TB_ROM9
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()

        # Raise an exception if unknown rom name is given
        romname = "unknown"
        try:
            twinmodel.generate_points("unknown", romname, False)
        except TwinModelError as e:
            assert "rom name provided" in str(e)

        # Raise an exception if unknown named selection is given
        romname = twinmodel.tbrom_names[0]
        try:
            twinmodel.generate_points("unknown", romname, False)
        except TwinModelError as e:
            assert "The provided named selection" in str(e)

        # Raise an exception if no point file is available
        model_filepath = TEST_TB_ROM10
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        nslist = twinmodel.get_named_selections(romname)
        try:
            twinmodel.generate_points(nslist[0], romname, False)
        except TwinModelError as e:
            assert "not find the geometry file" in str(e)
