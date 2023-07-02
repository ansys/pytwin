import os
import sys

import numpy as np
from pytwin import TwinModel, TwinModelError, download_file
from pytwin.evaluate.tbrom import TbRom
from pytwin.settings import get_pytwin_log_file


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

        # Raise an exception if provided snapshot path does not exist
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: "unknown_snapshot_path"}})
            # exist
        except TwinModelError as e:
            assert "[InputSnapshotPath]" in str(e)

        # Raise en exception if provided snapshot has the wrong size
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT_WRONG}})
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
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.initialize_evaluation(field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "[RomInputConnection]" in str(e)

    def test_evaluate_step_by_step_with_input_field_is_ok(self):
        model_filepath = TEST_TB_ROM3
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = "inputPressure"

        # First step
        twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        assert np.isclose(twinmodel.inputs["inputPressure_mode_0"], 18922.18290547577)
        assert np.isclose(twinmodel.inputs["inputPressure_mode_1"], -1303.3367783414574)
        assert np.isclose(twinmodel.outputs["outField_mode_1"], -0.007815295084108557)
        assert np.isclose(twinmodel.outputs["outField_mode_2"], -0.0019136501347937662)
        assert np.isclose(twinmodel.outputs["outField_mode_3"], 0.0007345769427744131)
        assert np.isclose(twinmodel.outputs["MaxDef"], 5.0352056308720146e-05)

        # Second step
        twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
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

        # Raise en exception if provided snapshot path is None
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: None}})
        except TwinModelError as e:
            assert "[InputSnapshotNone]" in str(e)

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

        # Raise an exception if provided field input is not connected
        model_filepath = TEST_TB_ROM5
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        fieldname = twinmodel.get_field_input_names(romname)[0]
        try:
            twinmodel.evaluate_step_by_step(step_size=0.1, field_inputs={romname: {fieldname: INPUT_SNAPSHOT}})
        except TwinModelError as e:
            assert "[RomInputConnection]" in str(e)

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
        romname = "unknown"

        # Raise an exception if twin model not initialized
        try:
            twinmodel.generate_points("unknown", romname, False)
        except TwinModelError as e:
            assert "[Initialization]" in str(e)

        twinmodel.initialize_evaluation()

        # Raise an exception if unknown rom name is given
        try:
            twinmodel.generate_points("unknown", romname, False)
        except TwinModelError as e:
            assert "[RomName]" in str(e)

        # Raise an exception if unknown named selection is given
        romname = twinmodel.tbrom_names[0]
        try:
            twinmodel.generate_points("unknown", romname, False)
        except TwinModelError as e:
            assert "[NamedSelection]" in str(e)

        # Raise an exception if no point file is available
        model_filepath = TEST_TB_ROM10
        twinmodel = TwinModel(model_filepath=model_filepath)
        twinmodel.initialize_evaluation()
        romname = twinmodel.tbrom_names[0]
        nslist = twinmodel.get_named_selections(romname)
        try:
            twinmodel.generate_points(nslist[0], romname, False)
        except TwinModelError as e:
            assert "[GeometryFile]" in str(e)

    def test_tbrom_getters_that_do_not_need_initialization(self):
        reinit_settings()
        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files")
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

    def test_tbrom_image_generation_at_initialization(self):
        reinit_settings()
        model_filepath = download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twin = TwinModel(model_filepath=model_filepath)
        twin.initialize_evaluation()

        # Verify IMAGE IS GENERATED AT INITIALIZATION
        if sys.platform != "linux":
            # TODO - Fix BUG755776
            fp = twin.get_image_filepath(
                rom_name=twin.tbrom_names[0],
                view_name=twin.get_available_view_names(twin.tbrom_names[0])[0],
                evaluation_time=0.0,
            )
            assert os.path.exists(fp)

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
