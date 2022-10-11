import os

from pytwin import set_pytwin_logging
from pytwin import PyTwinLogLevel
from pytwin.evaluate.model import Model


def reinit_logging(log_file):
    from pytwin.pytwin_logger import get_pytwin_logger
    from pytwin.settings import _reinit_settings
    _reinit_settings()
    logger = get_pytwin_logger()
    logger.handlers = []
    set_pytwin_logging(log_filepath=log_file)


class TestModel:
    def test_each_model_has_a_unique_identifier(self):
        # Init test context
        new_log_file = os.path.join(os.path.dirname(__file__), 'data', 'test_model_ids.log')
        if os.path.exists(new_log_file):
            os.remove(new_log_file)
        id_store = []
        model_count = 10000
        reinit_logging(new_log_file)
        # Run test
        for i in range(model_count):
            model = Model()
            model._log_message('hello!', PyTwinLogLevel.PYTWIN_LOG_ALL)
            id_store.append(model._id)
            assert id_store.count(model._id) == 1
        with open(new_log_file, 'r') as f:
            assert len(f.readlines()) == model_count

    def test_multiple_model_log_in_same_logger(self):
        # Init test context
        new_log_file = os.path.join(os.path.dirname(__file__), 'data', 'test_models_logging.log')
        if os.path.exists(new_log_file):
            os.remove(new_log_file)
        reinit_logging(new_log_file)
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
        with open(new_log_file, 'r') as f:
            assert len(f.readlines()) == 4
