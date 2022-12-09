import os
import sys

import pytest

from pytwin import TwinModel, TwinModelError, examples
from pytwin.settings import get_pytwin_log_file, reinit_settings_for_unit_tests

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")

UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings():
    import shutil

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


class TestTwinModel:
    """
    def test_raised_errors_with_tbrom_resource_directory(self):
        wd = reinit_settings()
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Raise an error if TWIN MODEL IS NOT INITIALIZED
        with pytest.raises(TwinModelError) as e:
            twin._tbrom_resource_directory(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_geometry_filepath(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_snapshot_filepath(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_available_view_names(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name="test", view_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_rom_directory(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        twin.initialize_evaluation()
        # Raise an error if TWIN MODEL DOES NOT INCLUDE ANY TBROM
        with pytest.raises(TwinModelError) as e:
            twin._tbrom_resource_directory(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_geometry_filepath(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_snapshot_filepath(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_available_view_names(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name="test", view_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_rom_directory(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        model_filepath = examples.download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twin = TwinModel(model_filepath=model_filepath)
        twin.initialize_evaluation()
        # Raise an error if TWIN MODEL DOES NOT INCLUDE ANY TBROM NAMED 'test'
        with pytest.raises(TwinModelError) as e:
            twin._tbrom_resource_directory(rom_name="test")
        assert "Twin model does not include any TBROM named" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_geometry_filepath(rom_name="test")
        assert "Please call the geometry file getter with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_snapshot_filepath(rom_name="test")
        assert "Please call the snapshot file getter with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_available_view_names(rom_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name="test", view_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_rom_directory(rom_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        # Raise an error if IMAGE VIEW DOES NOT EXIST
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name=twin.tbrom_names[0], view_name="test")
        assert "Please call this method with a valid view name." in str(e)
        # Raise an error if GEOMETRY POINT FILE HAS BEEN DELETED
        with pytest.raises(TwinModelError) as e:
            filepath = twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
            os.remove(filepath)
            twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
        assert "Could not find the geometry file for given available rom_name" in str(e)
        # Raise a warning if SNAPSHOT FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_snapshot_filepath(rom_name=twin.tbrom_names[0], evaluation_time=1.234567)
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "Could not find the snapshot file for given available rom_name" in "".join(log_str)
        # Verify IMAGE IS GENERATED AT INITIALIZATION
        if sys.platform != "linux":
            # BUG751873
            fp = twin.get_image_filepath(
                rom_name=twin.tbrom_names[0],
                view_name=twin.get_available_view_names(twin.tbrom_names[0])[0],
                evaluation_time=0.0,
            )
            assert os.path.exists(fp)
        # Raise a warning if IMAGE FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_image_filepath(
            rom_name=twin.tbrom_names[0],
            view_name=twin.get_available_view_names(twin.tbrom_names[0])[0],
            evaluation_time=1.234567,
        )
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "Could not find the image file for given available rom_name" in "".join(log_str)
"""

    def test_raised_errors_with_tbrom_resource_directory(self):
        wd = reinit_settings()
        model_filepath = examples.download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twin = TwinModel(model_filepath=model_filepath)
        twin.initialize_evaluation()
        """
        # Raise an error if TWIN MODEL DOES NOT INCLUDE ANY TBROM NAMED 'test'
        with pytest.raises(TwinModelError) as e:
            twin._tbrom_resource_directory(rom_name="test")
        assert "Twin model does not include any TBROM named" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_geometry_filepath(rom_name="test")
        assert "Please call the geometry file getter with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_snapshot_filepath(rom_name="test")
        assert "Please call the snapshot file getter with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_available_view_names(rom_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name="test", view_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_rom_directory(rom_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        # Raise an error if IMAGE VIEW DOES NOT EXIST
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name=twin.tbrom_names[0], view_name="test")
        assert "Please call this method with a valid view name." in str(e)
        # Raise an error if GEOMETRY POINT FILE HAS BEEN DELETED
        with pytest.raises(TwinModelError) as e:
            filepath = twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
            os.remove(filepath)
            twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
        assert "Could not find the geometry file for given available rom_name" in str(e)
        # Raise a warning if SNAPSHOT FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_snapshot_filepath(rom_name=twin.tbrom_names[0], evaluation_time=1.234567)
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "Could not find the snapshot file for given available rom_name" in "".join(log_str)
        """
        # Raise a warning if IMAGE FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_image_filepath(
            rom_name=twin.tbrom_names[0],
            view_name=twin.get_available_view_names(twin.tbrom_names[0])[0],
            evaluation_time=1.234567,
        )

    def test_clean_unit_test(self):
        reinit_settings()
