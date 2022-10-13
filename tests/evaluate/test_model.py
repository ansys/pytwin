import os
from pytwin import PyTwinLogLevel
from pytwin import get_pytwin_log_file
from pytwin.evaluate.model import Model


UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), 'unit_test_wd')


def reinit_settings():
    from pytwin.settings import reinit_settings_for_unit_tests
    import shutil
    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


class TestModel:
    def test_each_model_has_a_unique_identifier(self):
        # Init test context
        reinit_settings()
        id_store = []
        model_count = 10000
        # Run test
        for i in range(model_count):
            model = Model()
            model._log_message('hello!', PyTwinLogLevel.PYTWIN_LOG_INFO)
            id_store.append(model._id)
            assert id_store.count(model._id) == 1
        with open(get_pytwin_log_file(), 'r') as f:
            assert len(f.readlines()) == model_count

    def test_multiple_model_log_in_same_logger(self):
        # Init test context
        reinit_settings()
        # Run test
        model1 = Model()
        model1._model_name = 'model1'
        model1._id = '1'
        model2 = Model()
        model2._model_name = 'model2'
        model2._id = '2'
        model1._log_message('Hello A from model 1!')
        model1._log_message('Hello B from model 1!')
        model2._log_message('Hello A from model 2!')
        model2._log_message('Hello B from model 2!')
        with open(get_pytwin_log_file(), 'r') as f:
            assert len(f.readlines()) == 4
