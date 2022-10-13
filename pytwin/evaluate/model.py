import uuid
from pytwin import PyTwinLogLevel
from pytwin import get_pytwin_logger
from pytwin import pytwin_logging_is_enabled


class Model:
    def __init__(self):
        self._id = f'{uuid.uuid4()}'[0:8]
        self._model_name = None

    def _log_message(self, msg: str, level: PyTwinLogLevel = PyTwinLogLevel.PYTWIN_LOG_INFO):
        msg = f'[{self._model_name}.{self._id}] {msg}'
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



