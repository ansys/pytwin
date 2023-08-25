import os
import shutil

from pytwin import PYTWIN_LOGGING_OPT_NOLOGGING, TwinModel
from pytwin.settings import get_pytwin_log_file, get_pytwin_working_dir, modify_pytwin_logging

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings(create_new_temp_dir: bool = False):
    import shutil

    from pytwin.settings import reinit_settings_for_unit_tests

    session_id = reinit_settings_for_unit_tests(create_new_temp_dir)
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD, session_id


class TestTwinModelLogging:
    def test_twin_model_no_logging(self):
        from pytwin.settings import reinit_settings_session_id_for_unit_tests

        # Init unit test
        wd, session_id = reinit_settings(create_new_temp_dir=True)
        # Twin Model does not log anything if PYTWIN_LOGGING_OPT_NOLOGGING
        modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_NOLOGGING)
        log_file = get_pytwin_log_file()
        assert log_file is None
        twin = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        twin.initialize_evaluation()
        for i in range(100):
            new_inputs = {"Clutch1_in": 1.0 * i / 100, "Clutch2_in": 1.0 * i / 100}
            twin.evaluate_step_by_step(step_size=0.01, inputs=new_inputs)
        temp_dir = twin.model_temp
        assert os.path.exists(temp_dir)
        assert len(os.listdir(temp_dir)) == 0
        reinit_settings_session_id_for_unit_tests(session_id)

    def test_clean_unit_test(self):
        reinit_settings()
        temp_wd = get_pytwin_working_dir()
        try:
            shutil.rmtree(os.path.dirname(temp_wd))
        except Exception as e:
            pass
