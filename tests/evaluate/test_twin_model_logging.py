import os
import shutil

from pytwin import PYTWIN_LOGGING_OPT_NOLOGGING
from pytwin.settings import get_pytwin_log_file, get_pytwin_working_dir, modify_pytwin_logging

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings(create_new_temp_dir: bool = False):
    from pytwin.settings import reinit_settings_for_unit_tests

    session_id = reinit_settings_for_unit_tests(create_new_temp_dir)
    if os.path.exists(UNIT_TEST_WD):
        try:
            shutil.rmtree(UNIT_TEST_WD)
        except Exception as e:
            pass
    return UNIT_TEST_WD, session_id


class TestTwinModelLogging:
    def test_twin_model_no_logging(self):
        # Init unit test
        reinit_settings()
        # Twin Model does not log anything if PYTWIN_LOGGING_OPT_NOLOGGING
        modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_NOLOGGING)
        log_file = get_pytwin_log_file()
        assert log_file is None

    def test_clean_unit_test(self):
        reinit_settings()
        temp_wd = get_pytwin_working_dir()
        parent_dir = os.path.dirname(temp_wd)
        try:
            for dir_name in os.listdir(parent_dir):
                if dir_name not in temp_wd:
                    shutil.rmtree(os.path.join(parent_dir, dir_name))
        except Exception as e:
            pass
