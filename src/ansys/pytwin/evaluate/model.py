import os
import uuid

from pytwin import PyTwinLogLevel, get_pytwin_logger, get_pytwin_working_dir, pytwin_logging_is_enabled
from pytwin.settings import PYTWIN_SETTINGS


class Model:
    """
    Model is the private base class to manage twin model evaluation. It handles model id, working directories paths,
    logging and error raising.

    A Base class can overload the raise_model_error(self, msg) method to provide its own exception.
    """

    def __init__(self):
        self._id = f"{uuid.uuid4()}"[0:24].replace("-", "")
        self._model_name = None
        self._log_key = None

    def _log_message(self, msg: str, level: PyTwinLogLevel = PyTwinLogLevel.PYTWIN_LOG_INFO):
        """
        Use this method in base class to log message at key steps in the code logic.
        """
        msg = f"[{self._model_name}.{self._id}][{self._log_key}] {msg}"
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
        """
        Over load this method in base class to provide your own exception class so that it can be caught explicitly.
        """
        raise ModelError(msg)

    def _raise_error(self, msg):
        """
        Use this method in base class to raise an error with meaningful message.
        """
        self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_ERROR)
        self._raise_model_error(msg)

    @property
    def id(self):
        """Model unique id"""
        return self._id

    @property
    def name(self):
        """Model name. Multiple models can share the same name."""
        return self._model_name

    @property
    def model_dir(self):
        """Model directory (within the global working directory)"""
        return os.path.join(get_pytwin_working_dir(), f"{self._model_name}.{self._id}")

    @property
    def model_temp(self):
        """Model temporary directory (within the global working directory). It is shared by all models."""
        return os.path.join(get_pytwin_working_dir(), PYTWIN_SETTINGS.TEMP_WD_NAME)

    @property
    def model_log(self):
        """Path to model log file that is used at TwinRuntime instantiation (because we don't know the model name before
        having instantiated it and this can't use the model_dir in the log file path that is given at instantiation)."""
        return os.path.join(self.model_temp, f"{self._id}.log")

    @property
    def model_log_link(self):
        """Path to symbolic link to the log file in the temporary folder and that is stored in the model_dir."""
        return os.path.join(self.model_dir, f"link.log")


class ModelError(Exception):
    def __str__(self):
        return f"[ModelError] {self.args[0]}"
