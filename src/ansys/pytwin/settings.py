# Copyright (C) 2022 - 2025 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import atexit
from enum import Enum
import logging
import os
import shutil
import sys
import tempfile
import uuid


class PyTwinLogLevel(Enum):
    """
    Provides an enum for choosing the logging level for use by all PyTwin objects.
    PyTwin logging levels follow standard Python logging levels.

    PYTWIN_LOG_DEBUG:
        Provide detailed information that is typically of interest only when diagnosing problems.
    PYTWIN_LOG_INFO:
        Provide confirmation that things are working as expected.
    PYTWIN_LOG_WARNING:
        Provide an indication that something unexpected has happened or a problem might occur
        in the near future. For example, ``disk space low`` is a warning that is shown when
        the software is still working as expected but a problem might soon be encountered.
    PYTWIN_LOG_ERROR:
        Provide notice that due to a more serious problem, the software has not been able
        to perform some function.
    PYTWIN_LOG_CRITICAL:
       Provide notice of a serious error, indicating that the software may be unable to
       continue running.

    """

    PYTWIN_LOG_DEBUG = logging.DEBUG
    PYTWIN_LOG_INFO = logging.INFO
    PYTWIN_LOG_WARNING = logging.WARNING
    PYTWIN_LOG_ERROR = logging.ERROR
    PYTWIN_LOG_CRITICAL = logging.CRITICAL


class PyTwinLogOption(Enum):
    """
    Provides an enum for choosing the logging option for use by all PyTwin objects.

    PYTWIN_LOGGING_OPT_FILE:
        Redirect logging to the PyTwin log file stored in the PyTwin working directory.
    PYTWIN_LOGGING_OPT_CONSOLE:
        Redirect logging to the console.
    PYTWIN_LOGGING_OPT_NOLOGGING:
        Disable logging.

    """

    PYTWIN_LOGGING_OPT_FILE = 0  # Redirect logging to PyTwin log file
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
    Modify global PyTwin logging.

    You can choose to take these actions:

    - Redirect logging to a log file.
    - Redirect logging to the console.
    - Disable logging.

    All PyTwin objects from the same Python process share the same logging options.
    To fine tune the logging level, use the ``new_level`` parameter. For more
    information, see the examples.

    Parameters
    ----------
    new_option : PyTwinLogOption
        Option to use for PyTwin logging.
    new_level: PyTwinLogLevel
        Level to use for PyTwin logging.

    Raises
    ------
    PyTwinSettingsError
        If ``new_option`` is not a valid ``PyTwinLogOption`` attribute.
        If ``new_level`` is not a valid ``PyTwinLogLevel`` attribute.

    Examples
    --------
    >>> from pytwin import modify_pytwin_logging, get_pytwin_log_file
    >>> from pytwin import PYTWIN_LOGGING_OPT_FILE, PYTWIN_LOG_DEBUG
    >>> #
    >>> # Redirect logging to a file in the working directory and set logging level to DEBUG level
    >>> modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_FILE, new_level=PYTWIN_LOG_DEBUG)
    >>> print(get_pytwin_log_file())
    >>> #
    >>> # Redirect logging to the console
    >>> from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_CONSOLE
    >>> modify_pytwin_logging(PYTWIN_LOGGING_OPT_CONSOLE)
    >>> #
    >>> # Disable logging
    >>> from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_NOLOGGING
    >>> modify_pytwin_logging(PYTWIN_LOGGING_OPT_NOLOGGING)
    """

    def _check_log_level_is_valid(_level: PyTwinLogLevel):
        if isinstance(_level, PyTwinLogLevel):
            if _level.name not in PyTwinLogLevel.__members__:
                msg = "Error occurred while setting PyTwin logging level."
                msg += f"\nProvided log level is unknown (provided: {_level}.)"
                msg += f"\nChoose a value from the {PyTwinLogLevel.__members__} enum."
                raise PyTwinSettingsError(msg)
        else:
            msg = "Error occurred while setting PyTwin logging level."
            msg += f"\nUse the {PyTwinLogLevel} enum to set the log level."
            raise PyTwinSettingsError(msg)

    def _check_log_option_is_valid(_option: PyTwinLogOption):
        if isinstance(_option, PyTwinLogOption):
            if _option.name not in PyTwinLogOption.__members__:
                msg = "Error occurred while setting PyTwin logging option."
                msg += f"\nProvided logging option is unknown (provided: {_option}.)"
                msg += f"\nChoose a value from the {PyTwinLogOption.__members__} enum."
                raise PyTwinSettingsError(msg)
        else:
            msg = "Error occurred while setting PyTwin logging option."
            msg += f"\nUse the {PyTwinLogOption} enum to set the logging option."
            raise PyTwinSettingsError(msg)

    if new_option is not None:
        _check_log_option_is_valid(new_option)

    if new_level is not None:
        _check_log_level_is_valid(new_level)

    PYTWIN_SETTINGS.modify_logging(new_option=new_option, new_level=new_level)


def modify_pytwin_working_dir(new_path: str, erase: bool = True):
    """
    Modify the global PyTwin working directory.

    By default, a temporary directory is used by PyTwin as working directory. This temporary directory is automatically
    cleaned up at exit of the python process that imported pytwin. When this method is used, the new PyTwin working
    directory won't be deleted at python process exit. Note that this may lead to an overflow of the working directory.

    Parameters
    ----------
    new_path: str
        Absolute path to the working directory to use for PyTwin. The directory is created if it does not exist.
        This directory is kept alive at python process exit.
    erase: bool, optional
        Whether to erase a non-empty existing working directory. The default is ``True``,
        in which case the existing working directory is erased and a new one is created.
        If ``False``, the existing working directory is used as it is. This parameter has no
        effect if the directory does not exist.

    Raises
    ------
    PyTwinSettingsError
        If provided path is ``None``.

        If provided path does not exist and some parent directories do not exist or the last parent
        directory does not have write permission.

        If ``erase`` is not a Boolean value.

    Examples
    --------
    >>> # Modify working directory
    >>> from pytwin import modify_pytwin_working_dir
    >>> modify_pytwin_working_dir('path_to_new_working_dir', erase=False)
    """

    def _check_wd_path_is_valid(_wd: str):
        if new_path is None:
            msg = "Error occurred while setting the PyTwin working directory."
            msg += "\nGiven path is None. Provide a valid path."
            raise PyTwinSettingsError(msg)
        parent_dir = os.path.split(_wd)[0]
        if not os.path.exists(_wd):
            # Check if the provided working director can be created if it does not exist.
            if not os.path.exists(parent_dir):
                msg = f"Error occurred while setting the PyTwin working directory"
                msg += f"\nSome parent directory ({parent_dir}) in the provided path ({_wd}) does not exist."
                msg += f"\nProvide a folder path in which all parents exist."
                raise PyTwinSettingsError(msg)
            if not os.access(parent_dir, os.W_OK):
                msg = f"Error occurred while setting the PyTwin working directory."
                msg += f"\nParent directory ({parent_dir}) does not have write permission."
                msg += f"\nProvide write permission to '{parent_dir}'."
                raise PyTwinSettingsError(msg)

    def _check_wd_erase_is_valid(_erase: bool):
        if not isinstance(_erase, bool):
            msg = "Error occurred while setting the PyTwin working directory"
            msg += f"\n'erase' argument must be Boolean (provided: {_erase})."
            raise PyTwinSettingsError(msg)

    _check_wd_path_is_valid(new_path)
    _check_wd_erase_is_valid(erase)
    PYTWIN_SETTINGS.modify_wd_dir(new_path=new_path, erase=erase)


def pytwin_logging_is_enabled():
    return PYTWIN_SETTINGS.LOGGING_OPTION != PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING


def get_pytwin_logger():
    """
    Get the PyTwin logger (if any).
    """
    return PYTWIN_SETTINGS.logger


def get_pytwin_log_file():
    """
    Get the path to the PyTwin log file (if any).
    """
    return PYTWIN_SETTINGS.logfile


def get_pytwin_log_level():
    """
    Get PyTwin log level.
    """
    return PYTWIN_SETTINGS.loglevel


def get_pytwin_working_dir():
    """
    Get the path to the PyTwin working directory.
    """
    return PYTWIN_SETTINGS.working_dir


def reinit_settings_for_unit_tests():
    # Mutable attributes init
    _PyTwinSettings.LOGGING_OPTION = None
    _PyTwinSettings.LOGGING_LEVEL = None
    _PyTwinSettings.WORKING_DIRECTORY_PATH = None
    logging.getLogger(_PyTwinSettings.LOGGER_NAME).handlers.clear()
    PYTWIN_SETTINGS._initialize(keep_session_id=True)
    return PYTWIN_SETTINGS.SESSION_ID


class _PyTwinSettings(object):
    """
    This private class hosts PyTwin package settings (that are mutable and immutable attributes) that are seen by all
    PyTwin object instances. Helpers are provided to manipulate these attributes. Explicit modification of attributes is
    forbidden because they may cause unexpected behavior.
    """

    # Mutable constants
    LOGGING_OPTION = None
    LOGGING_LEVEL = None
    SESSION_ID = None
    WORKING_DIRECTORY_PATH = None
    TEMP_WORKING_DIRECTORY_PATH = None

    # Immutable constants
    LOGGER_NAME = "pytwin_logger"
    LOGGING_FILE_NAME = "pytwin.log"
    WORKING_DIRECTORY_NAME = "pytwin"
    TEMP_WD_NAME = ".temp"
    PYTWIN_START_MSG = "pytwin starts!"
    PYTWIN_END_MSG = "pytwin ends!"

    @property
    def logfile(self):
        if _PyTwinSettings.LOGGING_OPTION == PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING:
            return None
        if _PyTwinSettings.LOGGING_OPTION == PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE:
            return None
        if _PyTwinSettings.WORKING_DIRECTORY_PATH is None:
            msg = "Working directory has not been set."
            raise PyTwinSettingsError(msg)
        return os.path.join(_PyTwinSettings.WORKING_DIRECTORY_PATH, _PyTwinSettings.LOGGING_FILE_NAME)

    @property
    def loglevel(self):
        return _PyTwinSettings.LOGGING_LEVEL

    @property
    def logger(self):
        if _PyTwinSettings.LOGGING_OPTION is None:
            msg = "Logging option has not been set."
            raise PyTwinSettingsError(msg)
        return logging.getLogger(_PyTwinSettings.LOGGER_NAME)

    @property
    def working_dir(self):
        if _PyTwinSettings.WORKING_DIRECTORY_PATH is None:
            msg = "Working directory has not been set."
            raise PyTwinSettingsError(msg)
        return _PyTwinSettings.WORKING_DIRECTORY_PATH

    def __init__(self):
        self._initialize(keep_session_id=False)
        self.logger.info(_PyTwinSettings.PYTWIN_START_MSG)

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
    def _initialize(keep_session_id: bool):
        pytwin_logger = logging.getLogger(_PyTwinSettings.LOGGER_NAME)
        pytwin_logger.handlers.clear()

        if not keep_session_id:
            _PyTwinSettings.SESSION_ID = f"{uuid.uuid4()}"[0:24].replace("-", "")

        _PyTwinSettings._initialize_wd()
        _PyTwinSettings._initialize_logging()

    @staticmethod
    def _initialize_logging():
        """
        Provides default logging settings (log to file with INFO level).
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
        Provides default settings for the PyTwin working directory.
        """
        # Create a unique working directory for each python process that imports pytwin
        _PyTwinSettings.TEMP_WORKING_DIRECTORY_PATH = os.path.join(
            tempfile.gettempdir(), _PyTwinSettings.WORKING_DIRECTORY_NAME, _PyTwinSettings.SESSION_ID
        )
        os.makedirs(_PyTwinSettings.TEMP_WORKING_DIRECTORY_PATH, exist_ok=True)
        _PyTwinSettings.WORKING_DIRECTORY_PATH = _PyTwinSettings.TEMP_WORKING_DIRECTORY_PATH

    @staticmethod
    def _migration_due_to_new_wd(old_path: str, new_path: str):
        # Migrate file handler found in pytwin_logger
        pytwin_logger = logging.getLogger(_PyTwinSettings.LOGGER_NAME)

        has_file_handler = None
        for handler in pytwin_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                has_file_handler = True

        if has_file_handler:
            # Clear existing handlers, copy old log content into new one, add a new file handler to PyTwin logger
            pytwin_logger.handlers.clear()
            old_logfile_path = os.path.join(old_path, _PyTwinSettings.LOGGING_FILE_NAME)
            new_logfile_path = os.path.join(new_path, _PyTwinSettings.LOGGING_FILE_NAME)
            shutil.copyfile(old_logfile_path, new_logfile_path)
            _PyTwinSettings.LOGGING_OPTION = PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE
            _PyTwinSettings._add_default_file_handler_to_pytwin_logger(
                filepath=new_logfile_path, level=_PyTwinSettings.LOGGING_LEVEL, mode="a"
            )
        # Migrate subfolder
        if sys.version >= "3.8":
            shutil.copytree(
                src=old_path,
                dst=new_path,
                ignore=shutil.ignore_patterns(f"{_PyTwinSettings.TEMP_WD_NAME}*"),
                dirs_exist_ok=True,
            )
        else:
            """Copy a directory structure, overwriting existing files."""
            src = old_path
            dst = new_path
            for root, dirs, files in os.walk(src):
                if not os.path.isdir(root):
                    os.makedirs(root)
                for file in files:
                    rel_path = root.replace(src, "").lstrip(os.sep)
                    dest_path = os.path.join(dst, rel_path)
                    if _PyTwinSettings.TEMP_WD_NAME not in dest_path:
                        if not os.path.isdir(dest_path):
                            os.makedirs(dest_path)
                        shutil.copyfile(os.path.join(root, file), os.path.join(dest_path, file))

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
                # Update PyTwin settings and clear existing handles
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
                # fix ACE bug on logging level Jan 18th 2023 (test_modify_logging_level)
                _PyTwinSettings.LOGGING_LEVEL = new_level
                pytwin_logger.setLevel(new_level.value)
                for handler in pytwin_logger.handlers:
                    handler.setLevel(new_level.value)


PYTWIN_SETTINGS = _PyTwinSettings()  # This instance is here to launch default settings initialization.


@atexit.register
def cleanup_temp_pytwin_working_directory():
    pytwin_logger = PYTWIN_SETTINGS.logger
    pytwin_logger.info(PYTWIN_SETTINGS.PYTWIN_END_MSG)
    for handler in pytwin_logger.handlers:
        handler.close()
        pytwin_logger.removeHandler(handler)
    try:
        shutil.rmtree(PYTWIN_SETTINGS.TEMP_WORKING_DIRECTORY_PATH)
    except BaseException as e:
        msg = (
            "Something went wrong while trying to cleanup pytwin temporary directory! You might have to clean it up "
            "manually. "
        )
        msg += f" Error message:\n{str(e)}"
        print(msg)
