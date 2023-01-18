import os

from pytwin import TwinModel, PYTWIN_LOGGING_OPT_NOLOGGING
from pytwin.settings import modify_pytwin_logging, get_pytwin_log_file

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings():
    import shutil

    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


class TestTwinModelLogging:
    def test_twin_model_no_logging(self):
        # Init unit test
        reinit_settings()
        # Twin Model does not log anything if PYTWIN_LOGGING_OPT_NOLOGGING
        modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_NOLOGGING)
        log_file = get_pytwin_log_file()
        assert log_file is None
        twin = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        twin.initialize_evaluation()
        for i in range(100):
            new_inputs = {"Clutch1_in": 1.0*i/100, "Clutch2_in": 1.0*i/100}
            twin.evaluate_step_by_step(step_size=0.01, inputs=new_inputs)
        temp_dir = twin.model_temp
        assert os.path.exists(temp_dir)
        assert len(os.listdir(temp_dir)) == 0
