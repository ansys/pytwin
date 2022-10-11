import os
import logging
import pytest
from pytwin import set_pytwin_logging
from pytwin import PYTWIN_LOG_WARNING
from pytwin import PYTWIN_NO_LOG
from pytwin import PyTwinLoggingError
from pytwin import get_pytwin_logger


def reinit_logging():
    from pytwin.settings import _reinit_settings
    _reinit_settings()
    logger = get_pytwin_logger()
    logger.handlers = []


class TestSetLoggingOption:
    def test_pytwin_logging_raises_error(self):
        # Reinit test context
        reinit_logging()
        # Raise error if level is unknown
        with pytest.raises(PyTwinLoggingError) as e:
            set_pytwin_logging(level='unknown')
        assert 'Provided log level is unknown' in str(e)
        # Raise error if mode is unknown
        with pytest.raises(PyTwinLoggingError) as e:
            set_pytwin_logging(mode='unknown', log_filepath='')
        assert 'Provided mode for FileHandler is unknown' in str(e)
        # Raise error if absolute path to log file does not exist
        with pytest.raises(PyTwinLoggingError) as e:
            wrong_path = os.path.join(os.path.dirname(__file__), 'unknown_folder', 'test.log')
            set_pytwin_logging(mode='a', log_filepath=wrong_path)
        assert 'Please provide a log filepath with existing absolute path.' in str(e)

    def test_pytwin_logging_option_filehandler(self):
        # Reinit test context
        reinit_logging()
        file_path = os.path.join(os.path.dirname(__file__), 'unit_test.log')
        if os.path.exists(file_path):
            os.remove(file_path)
        # Run test
        set_pytwin_logging(log_filepath=file_path, mode='a', level=PYTWIN_LOG_WARNING)
        logger = get_pytwin_logger()
        logger.warning('Hello from test_pytwin_logging_option_filehandler!')
        assert os.path.exists(file_path)

    def test_pytwin_logging_option_console(self):
        # Reinit test context
        reinit_logging()
        # Run test
        set_pytwin_logging()
        logger = get_pytwin_logger()
        logger.debug('Hello from test_pytwin_logging_option_console!')

    def test_pytwin_logging_option_no_log(self):
        # Reinit test context
        reinit_logging()
        # Run test
        set_pytwin_logging(level=PYTWIN_NO_LOG)
        logger = get_pytwin_logger()
        logger.debug('Hello from test_pytwin_logging_option_no_log!')
        assert len(logger.handlers) == 0

    def test_calling_multiple_time_twin_logging_has_no_effect(self):
        # Reinit test context
        reinit_logging()
        file_path = os.path.join(os.path.dirname(__file__), 'unit_test.log')
        if os.path.exists(file_path):
            os.remove(file_path)
        # Run test
        set_pytwin_logging(log_filepath=file_path, mode='a', level=PYTWIN_LOG_WARNING)
        set_pytwin_logging(level=PYTWIN_NO_LOG)
        logger = get_pytwin_logger()
        logger.warning('Hello from test_calling_multiple_time_twin_logging_has_no_effect!')
        assert os.path.exists(file_path)
