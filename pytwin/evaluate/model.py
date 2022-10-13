import uuid
import os
from pytwin import PyTwinLogLevel
from pytwin import get_pytwin_logger
from pytwin import pytwin_logging_is_enabled
from pytwin import get_pytwin_working_dir
from pytwin.settings import PYTWIN_SETTINGS


class Model:
    def __init__(self):
        self._id = f'{uuid.uuid4()}'[0:8]
        self._model_name = None
        self._log_key = None

    def _log_message(self, msg: str, level: PyTwinLogLevel = PyTwinLogLevel.PYTWIN_LOG_INFO):
        msg = f'[{self._model_name}.{self._id}][{self._log_key}] {msg}'
        logger = get_pytwin_logger()
        if pytwin_logging_is_enabled():
            if level == PyTwinLogLevel.PYTWIN_LOG_DEBUG:
                logger.debug(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_INFO:
                logger.info(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_WARNING:
                logger.warning(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_ERROR:
                logger.error(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_CRITICAL:
                logger.critical(msg)
                return

    def _raise_model_error(self, msg):
        pass

    def _raise_error(self, msg):
        self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_ERROR)
        self._raise_model_error(msg)

    @property
    def id(self):
        return self._id

    @property
    def model_dir(self):
        return os.path.join(get_pytwin_working_dir(), f'{self._model_name}.{self._id}')

    @property
    def model_temp(self):
        return os.path.join(get_pytwin_working_dir(), PYTWIN_SETTINGS.TEMP_WD_NAME)

    @property
    def model_log(self):
        return os.path.join(self.model_temp, f'{self._id}.log')

    @property
    def model_log_link(self):
        return os.path.join(self.model_dir, f'link.log')
