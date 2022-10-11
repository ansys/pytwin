import logging
import os
from enum import Enum
from pytwin.settings import _pytwin_logging_is_undefined
from pytwin.settings import _disable_pytwin_logging
from pytwin.settings import _set_pytwin_logging_option_to_file_handler
from pytwin.settings import _set_pytwin_logging_option_to_stream_handler
from pytwin.settings import _get_pytwin_logger_name


class PyTwinLogLevel(Enum):
    PYTWIN_LOG_ALL = 10  # DEBUG
    PYTWIN_LOG_WARNING = 30  # WARNING
    PYTWIN_LOG_ERROR = 40  # ERROR
    PYTWIN_LOG_FATAL = 50  # CRITICAL
    PYTWIN_NO_LOG = -1


def set_pytwin_logging(log_filepath: str = None,
                       level: PyTwinLogLevel = PyTwinLogLevel.PYTWIN_LOG_ALL,
                       mode: str = 'a'):
    """
    Set pytwin package logging options. You can: (1) redirect logging to a log file, (2) redirect logging to the console
    or (3) disable logging (see examples). Use level parameter to fine tune logging level. All pytwin objects (from the
    same python process) share the same logging options. Only the first call to this method has an effect.

    Parameters
    ----------
    log_filepath : str
        Absolute file path to redirect pytwin logging. It will be created by the pytwin logger if it does not exist.
        Same file is used for any pytwin instances. Keep it to None will redirect logging to console.
    level: PyTwinLogLevel
        Logging level you want to use for any pytwin instances. Enum values matches with standard logging module. use
        PyTwinLogLevel.PYTWIN_NO_LOG if you want no logging for pytwin objects.
    mode: str
        Mode in which the logging file will be opened. Allowed mode are: 'a'(default) or 'w'.

    Examples
    --------
    >>> # Append pytwin package logging to an existing application log file:
    >>> from pytwin import set_pytwin_logging
    >>> from pytwin import PYTWIN_LOG_WARNING
    >>> set_pytwin_logging(log_filepath='filepath_to_my_app.log', mode='a', level=PYTWIN_LOG_WARNING)

    >>> # Redirect pytwin package logging to the console:
    >>> from pytwin import set_pytwin_logging
    >>> set_pytwin_logging()

    >>> # Disable pytwin package logging:
    >>> from pytwin import set_pytwin_logging
    >>> from pytwin import PYTWIN_NO_LOG
    >>> set_pytwin_logging(level=PYTWIN_NO_LOG)
    """
    def _check_level_is_valid(_level: PyTwinLogLevel):
        if isinstance(_level, PyTwinLogLevel):
            if _level.name not in PyTwinLogLevel.__members__:
                msg = 'Error while setting pytwin logging options!'
                msg += f'\nProvided log level is unknown (provided: {_level})'
                msg += f'\nPlease choose among {PyTwinLogLevel.__members__} enum.'
                raise PyTwinLoggingError(msg)
        else:
            msg = 'Error while setting pytwin logging options!'
            msg += f'\nPlease use {PyTwinLogLevel} enum to set level argument value.'
            raise PyTwinLoggingError(msg)

    def _check_file_logging_mode_is_valid(_mode: str):
        allowed_modes = ['a', 'w']
        if _mode not in allowed_modes:
            msg = 'Error while setting pytwin logging options!'
            msg += f'\nProvided mode for FileHandler is unknown (provided: {_mode}).'
            msg += f'\nPlease choose among {allowed_modes}.'
            raise PyTwinLoggingError(msg)

    def _check_path_to_provided_log_file_exists(_filepath: str):
        abs_file_dir = os.path.split(_filepath)[0]
        file_name = os.path.split(_filepath)[-1]
        if not os.path.exists(abs_file_dir):
            msg = 'Error while setting pytwin logging options!'
            msg += f'\nFolder ({abs_file_dir}) of the log file ({file_name}) does not exists!'
            msg += f'\nPlease provide a log filepath within an existing folder.'
            raise PyTwinLoggingError(msg)

    if _pytwin_logging_is_undefined():
        _check_level_is_valid(level)

        if level == PyTwinLogLevel.PYTWIN_NO_LOG:
            _disable_pytwin_logging()
        else:
            if log_filepath is not None:
                # Case of Logging in a file
                _check_file_logging_mode_is_valid(mode)
                _check_path_to_provided_log_file_exists(log_filepath)
                log_handler = logging.FileHandler(filename=log_filepath, mode=mode)
                _set_pytwin_logging_option_to_file_handler(log_filepath, level)
            else:
                # Case of Logging in the console
                log_handler = logging.StreamHandler()
                _set_pytwin_logging_option_to_stream_handler(level)

            # This is the first call. We Create a pytwin logger in the logger hierarchy and set its level
            logger = logging.getLogger(_get_pytwin_logger_name())
            logger.setLevel(level.value)
            # Choose pytwin logging handler and set its level and format
            formatter = logging.Formatter(fmt='[%(asctime)s][pytwin] %(levelname)s: %(message)s',
                                          datefmt='%m/%d/%Y %I:%M:%S %p')
            log_handler.setLevel(level.value)
            log_handler.setFormatter(fmt=formatter)
            logger.addHandler(log_handler)


class PyTwinLoggingError(Exception):
    def __str__(self):
        return f'[pyTwin][LoggingError] {self.args[0]}'


def get_pytwin_logger():
    """
    Get pytwin logger
    """
    return logging.getLogger(_get_pytwin_logger_name())
