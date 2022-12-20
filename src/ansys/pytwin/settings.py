from enum import Enum
import logging
import os
import shutil
import tempfile


class PyTwinLogLevel(Enum):
    """
    Enum to choose logging level to be used by all pytwin objects.
    It follows standard python logging levels.
    """

    PYTWIN_LOG_DEBUG = logging.DEBUG
    PYTWIN_LOG_INFO = logging.INFO
    PYTWIN_LOG_WARNING = logging.WARNING
    PYTWIN_LOG_ERROR = logging.ERROR
    PYTWIN_LOG_CRITICAL = logging.CRITICAL


class PyTwinLogOption(Enum):
    """
    Enum to choose logging options to be used by all pytwin objects.

    PYTWIN_LOGGING_OPT_FILE:
        Redirect logging to the pytwin log file stored in the pytwin working directory.
    PYTWIN_LOGGING_OPT_CONSOLE:
        Redirect logging to console.
    PYTWIN_LOGGING_OPT_NOLOGGING:
        Disable pytwin logging.

    """

    PYTWIN_LOGGING_OPT_FILE = 0  # Redirect logging to pytwin log file
    PYTWIN_LOGGING_OPT_CONSOLE = 1  # Redirect logging to console
    PYTWIN_LOGGING_OPT_NOLOGGING = 2  # No logging


class PyTwinSettingsError(Exception):
    def __str__(self):
        return f"[pyTwin][SettingsError] {self.args[0]}"


def modify_pytwin_logging(
    new_option: PyTwinLogOption = PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE,
    new_level: PyTwinLogLevel = PyTwinLogLevel.PYTWIN_LOG_INFO,
):
    """
    Modify global pytwin logging. You can: (1) redirect logging to a log file, (2) redirect logging to the console
    or (3) disable logging (see examples). Use level parameter to fine tune logging level. All pytwin objects (from the
    same python process) share the same logging options.

    Parameters
    ----------
    new_option : PyTwinLogOption
        Option you want to use for pytwin logging.
    new_level: PyTwinLogLevel
        Level you want to use for pytwin logging.

    Raises
    ------
    PyTwinSettingsError
        If new_option is not a valid PyTwinLogOption attribute.
        if new_level is not a valid PyTwinLogLevel attribute.

    Examples
    --------
    >>> # Redirect logging to a file in the working directory
    >>> from pytwin import modify_pytwin_logging, get_pytwin_log_file
    >>> from pytwin import PYTWIN_LOGGING_OPT_FILE, PYTWIN_LOG_DEBUG
    >>> modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_FILE, new_level=PYTWIN_LOG_DEBUG)
    >>> print(get_pytwin_log_file())
    >>> # Redirect logging to the console
    >>> from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_CONSOLE
    >>> modify_pytwin_logging(PYTWIN_LOGGING_OPT_CONSOLE)
    >>> # Disable logging
    >>> from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_NOLOGGING
    >>> modify_pytwin_logging(PYTWIN_LOGGING_OPT_NOLOGGING)
    """

    def _check_log_level_is_valid(_level: PyTwinLogLevel):
        if isinstance(_level, PyTwinLogLevel):
            if _level.name not in PyTwinLogLevel.__members__:
                msg = "Error while setting pytwin logging!"
                msg += f"\nProvided log level is unknown (provided: {_level})"
                msg += f"\nPlease choose among {PyTwinLogLevel.__members__} enum."
                raise PyTwinSettingsError(msg)
        else:
            msg = "Error while setting pytwin logging options!"
            msg += f"\nPlease use {PyTwinLogLevel} enum to set level argument value."
            raise PyTwinSettingsError(msg)

    def _check_log_option_is_valid(_option: PyTwinLogOption):
        if isinstance(_option, PyTwinLogOption):
            if _option.name not in PyTwinLogOption.__members__:
                msg = "Error while setting pytwin logging!"
                msg += f"\nProvided log option is unknown (provided: {_option})"
                msg += f"\nPlease choose among {PyTwinLogOption.__members__} enum."
                raise PyTwinSettingsError(msg)
        else:
            msg = "Error while setting pytwin logging options!"
            msg += f"\nPlease use {PyTwinLogOption} enum to set option argument value."
            raise PyTwinSettingsError(msg)

    if new_option is not None:
        _check_log_option_is_valid(new_option)

    if new_level is not None:
        _check_log_level_is_valid(new_level)

    PYTWIN_SETTINGS.modify_logging(new_option=new_option, new_level=new_level)


def modify_pytwin_working_dir(new_path: str, erase: bool = True):
    """
    Modify global pytwin working directory.

    Parameters
    ----------
    new_path: str
        Absolute path to the working directory you want to use for pytwin package. It is created if it does not exist.
    erase: bool
        if True, erase non-empty existing working directory and create a new one. If False, use existing working
        directory as it is. Value has no effect if directory does not exist.

    Raises
    ------
    PyTwinSettingsError
        If provided path is None.
        If provided path does not exist AND some parent directories do not exist or last parent directory does not have
        writing permission.
        If erase is not a boolean.

    Examples
    --------
    >>> # Modify working directory
    >>> from pytwin import modify_pytwin_working_dir
    >>> modify_pytwin_working_dir('path_to_new_working_dir', erase=False)
    """

    def _check_wd_path_is_valid(_wd: str):
        if new_path is None:
            msg = "Error while setting pytwin working directory!"
            msg += "\nGiven path is None. Please give a valid path."
            raise PyTwinSettingsError(msg)
        parent_dir = os.path.split(_wd)[0]
        if not os.path.exists(_wd):
            # Check we can create the provided working dir if it does not exist.
            if not os.path.exists(parent_dir):
                msg = f"Error while setting pytwin working directory!"
                msg += f"\nSome parent directory ({parent_dir}) in the provided folder path ({_wd}) does not exists!"
                msg += f"\nPlease provide a folder path in which all parents exist."
                raise PyTwinSettingsError(msg)
            if not os.access(parent_dir, os.W_OK):
                msg = f"Error while setting pytwin working directory!"
                msg += f"\nParent directory ({parent_dir}) has not the writing permission."
                msg += f"\nPlease provide writing permission to '{parent_dir}'"
                raise PyTwinSettingsError(msg)

    def _check_wd_erase_is_valid(_erase: bool):
        if not isinstance(_erase, bool):
            msg = "Error while setting pytwin working directory!"
            msg += f"\n'erase' argument must be boolean (provided: {_erase})"
            raise PyTwinSettingsError(msg)

    _check_wd_path_is_valid(new_path)
    _check_wd_erase_is_valid(erase)
    PYTWIN_SETTINGS.modify_wd_dir(new_path=new_path, erase=erase)


def pytwin_logging_is_enabled():
    return PYTWIN_SETTINGS.LOGGING_OPTION != PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING


def get_pytwin_logger():
    """
    Get pytwin logger (if any).
    """
    return PYTWIN_SETTINGS.logger


def get_pytwin_log_file():
    """
    Get path to pytwin log file (if any).
    """
    return PYTWIN_SETTINGS.logfile


def get_pytwin_log_level():
    """
    Get path to pytwin log level.
    """
    return PYTWIN_SETTINGS.loglevel


def get_pytwin_working_dir():
    """
    Get path to pytwin working directory.
    """
    return PYTWIN_SETTINGS.working_dir


def reinit_settings_for_unit_tests():
    # Mutable attributes init
    _PyTwinSettings.LOGGING_OPTION = None
    _PyTwinSettings.LOGGING_LEVEL = None
    _PyTwinSettings.MULTI_PROCESS_IS_ENABLED = False
    _PyTwinSettings.WORKING_DIRECTORY_PATH = None
    logging.getLogger(_PyTwinSettings.LOGGER_NAME).handlers.clear()
    _PyTwinSettings().__init__()


class _PyTwinSettings(object):
    """
    This private class hosts pytwin package settings (that are mutable and immutable attributes) that are seen by all
    pytwin object instances. Helpers are provided to manipulate these attributes. Explicit modification of attributes is
    forbidden and may cause unexpected behavior.
    """

    # Below constants are mutable
    LOGGING_OPTION = None
    LOGGING_LEVEL = None
    MULTI_PROCESS_IS_ENABLED = False
    WORKING_DIRECTORY_PATH = None

    # Below constants are immutable
    LOGGER_NAME = "pytwin_logger"
    LOGGING_FILE_NAME = "pytwin.log"
    WORKING_DIRECTORY_NAME = "pytwin"
    TEMP_WD_NAME = ".temp"

    @property
    def logfile(self):
        if _PyTwinSettings.LOGGING_OPTION == PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING:
            return None
        if _PyTwinSettings.LOGGING_OPTION == PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE:
            return None
        if _PyTwinSettings.WORKING_DIRECTORY_PATH is None:
            msg = "Working directory has not been set!"
            raise PyTwinSettingsError(msg)
        return os.path.join(_PyTwinSettings.WORKING_DIRECTORY_PATH, _PyTwinSettings.LOGGING_FILE_NAME)

    @property
    def loglevel(self):
        return _PyTwinSettings.LOGGING_LEVEL

    @property
    def logger(self):
        if _PyTwinSettings.LOGGING_OPTION is None:
            msg = "Logging has not been set!"
            raise PyTwinSettingsError(msg)
        return logging.getLogger(_PyTwinSettings.LOGGER_NAME)

    @property
    def working_dir(self):
        if _PyTwinSettings.WORKING_DIRECTORY_PATH is None:
            msg = "Working directory has not been set!"
            raise PyTwinSettingsError(msg)
        return _PyTwinSettings.WORKING_DIRECTORY_PATH

    def __init__(self):
        self._initialize()

    @staticmethod
    def _add_default_file_handler_to_pytwin_logger(filepath: str, level: PyTwinLogLevel, mode: str = "w"):
        # Create logging handler
        formatter = logging.Formatter(
            fmt="[%(asctime)s][pytwin] %(levelname)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
        )
        log_handler = logging.FileHandler(filename=filepath, mode=mode)
        log_handler.setLevel(level.value)
        log_handler.setFormatter(fmt=formatter)
        # Add handler to pytwin logger
        logger = logging.getLogger(_PyTwinSettings.LOGGER_NAME)
        logger.setLevel(level.value)
        logger.addHandler(log_handler)

    @staticmethod
    def _add_default_stream_handler_to_pytwin_logger(level: PyTwinLogLevel):
        # Create logging handler
        formatter = logging.Formatter(
            fmt="[%(asctime)s][pytwin] %(levelname)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
        )
        log_handler = logging.StreamHandler()
        log_handler.setLevel(level.value)
        log_handler.setFormatter(fmt=formatter)
        # Add handler to pytwin logger
        logger = logging.getLogger(_PyTwinSettings.LOGGER_NAME)
        logger.setLevel(level.value)
        logger.addHandler(log_handler)

    @staticmethod
    def _initialize():
        pytwin_logger = logging.getLogger(_PyTwinSettings.LOGGER_NAME)
        pytwin_logger.handlers.clear()
        _PyTwinSettings._initialize_wd()
        _PyTwinSettings._initialize_logging()

    @staticmethod
    def _initialize_logging():
        """
        Default logging settings (log to file with info level)
        """
        # Set default logging settings
        _PyTwinSettings.LOGGING_OPTION = PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE
        _PyTwinSettings.LOGGING_LEVEL = PyTwinLogLevel.PYTWIN_LOG_INFO
        _PyTwinSettings._add_default_file_handler_to_pytwin_logger(
            filepath=os.path.join(_PyTwinSettings.WORKING_DIRECTORY_PATH, _PyTwinSettings.LOGGING_FILE_NAME),
            level=_PyTwinSettings.LOGGING_LEVEL,
        )

    @staticmethod
    def _initialize_wd():
        """
        Default working directory settings.
        """
        # Clean pytwin temporary directory, each time pytwin is imported.
        if _PyTwinSettings.MULTI_PROCESS_IS_ENABLED:
            pytwin_temp_dir = os.path.join(
                tempfile.gettempdir(), str(os.getpid()), _PyTwinSettings.WORKING_DIRECTORY_NAME
            )
        else:
            pytwin_temp_dir = os.path.join(tempfile.gettempdir(), _PyTwinSettings.WORKING_DIRECTORY_NAME)
        for i in range(5):
            # Loop to wait until logging file is freed
            try:
                if os.path.exists(pytwin_temp_dir):
                    shutil.rmtree(pytwin_temp_dir)
            except PermissionError as e:
                import time

                logging.warning(f"_PyTwinSettings failed to clear working dir (attempt #{i})! \n {str(e)}")
                time.sleep(1)

        os.mkdir(pytwin_temp_dir)
        _PyTwinSettings.WORKING_DIRECTORY_PATH = pytwin_temp_dir

    @staticmethod
    def _migration_due_to_new_wd(old_path: str, new_path: str):
        # Migrate file handler found in pytwin_logger
        pytwin_logger = logging.getLogger(_PyTwinSettings.LOGGER_NAME)

        has_file_handler = None
        for handler in pytwin_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                has_file_handler = True

        if has_file_handler:
            # Clear existing handlers, copy old log content into new one, add a new file handler to pytwin logger
            pytwin_logger.handlers.clear()
            old_logfile_path = os.path.join(old_path, _PyTwinSettings.LOGGING_FILE_NAME)
            new_logfile_path = os.path.join(new_path, _PyTwinSettings.LOGGING_FILE_NAME)
            shutil.copyfile(old_logfile_path, new_logfile_path)
            _PyTwinSettings.LOGGING_OPTION = PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE
            _PyTwinSettings._add_default_file_handler_to_pytwin_logger(
                filepath=new_logfile_path, level=_PyTwinSettings.LOGGING_LEVEL, mode="a"
            )
        # Migrate subfolder
        shutil.copytree(
            src=old_path,
            dst=new_path,
            ignore=shutil.ignore_patterns(f"{_PyTwinSettings.TEMP_WD_NAME}*"),
            dirs_exist_ok=True,
        )

    @staticmethod
    def modify_wd_dir(new_path: str, erase: bool):
        old_path = _PyTwinSettings.WORKING_DIRECTORY_PATH

        # Check new directory
        if os.path.exists(new_path):
            if len(os.listdir(new_path)) > 0:
                # New directory exists and it is not empty
                if erase:
                    shutil.rmtree(new_path)
                    os.mkdir(new_path)
        else:
            # New directory does not exist
            os.mkdir(new_path)

        _PyTwinSettings.WORKING_DIRECTORY_PATH = new_path
        if old_path is not None:
            _PyTwinSettings._migration_due_to_new_wd(old_path, new_path)

    @staticmethod
    def modify_logging(new_option: PyTwinLogOption, new_level: PyTwinLogLevel):

        pytwin_logger = logging.getLogger(_PyTwinSettings.LOGGER_NAME)

        # Modifications in case of new option
        if new_option is not None:
            if new_option != _PyTwinSettings.LOGGING_OPTION:
                # Update pytwin settings and clear existing handles
                _PyTwinSettings.LOGGING_OPTION = new_option
                pytwin_logger.handlers.clear()
                # Create new handles if needed
                if new_option == PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE:
                    _PyTwinSettings._add_default_file_handler_to_pytwin_logger(
                        filepath=os.path.join(
                            _PyTwinSettings.WORKING_DIRECTORY_PATH, _PyTwinSettings.LOGGING_FILE_NAME
                        ),
                        level=_PyTwinSettings.LOGGING_LEVEL,
                    )
                if new_option == PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE:
                    _PyTwinSettings._add_default_stream_handler_to_pytwin_logger(level=_PyTwinSettings.LOGGING_LEVEL)

        # Modifications in case of new level
        if new_level is not None:
            if new_level != _PyTwinSettings.LOGGING_LEVEL:
                pytwin_logger.setLevel(new_level.value)
                for handler in pytwin_logger.handlers:
                    handler.setLevel(new_level.value)


PYTWIN_SETTINGS = _PyTwinSettings()  # This instance is here to launch default settings initialization.
