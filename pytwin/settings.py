import os
import shutil
import sys
import tempfile


class _PyTwinConstants(object):
    """
    (internal) This class hosts pytwin package constants that must be shared between all pytwin object instances.
    Explicit modifications of the constant values at runtime is forbidden and may cause unexpected behavior.
    """
    LOGGING = {'option': 'UNDEFINED'}
    LOGGER_NAME = 'pytwin_logger'
    WORKING_DIRECTORY = None


def _create_pytwin_temporary_dir():
    """
    (internal) Create pytwin temporary (delete existing one) and return path to it.
    """
    pytwin_temp_dir = os.path.join(tempfile.gettempdir(), 'pytwin')
    if os.path.exists(pytwin_temp_dir):
        shutil.rmtree(pytwin_temp_dir)
    os.mkdir(pytwin_temp_dir)
    return pytwin_temp_dir


def _disable_pytwin_logging():
    """
    (internal) Method used to disable pytwin logging
    """
    _PyTwinConstants.LOGGING = {'option': 'DISABLED'}


def _get_pytwin_logger_name():
    """
    (internal) Get pytwin logger name.
    """
    return _PyTwinConstants.LOGGER_NAME


def _pytwin_logging_is_undefined():
    """
    (internal) Method to know if the pytwin logging has already been setup by user or if it is still undefined.
    """
    return _PyTwinConstants.LOGGING['option'] == 'UNDEFINED'


def _reinit_settings():
    """
    (internal) This method is used by unitary tests only to reinitialize pytwin settings.
    """
    _PyTwinConstants.LOGGING = {'option': 'UNDEFINED'}
    _PyTwinConstants.LOGGER_NAME = 'pytwin_logger'
    _PyTwinConstants.WORKING_DIRECTORY = None


def _set_pytwin_logging_option_to_file_handler(log_filepath, level):
    """
    (internal) Set pytwin logging option to file handler and save log file path as well as logging level.
    """
    _PyTwinConstants.LOGGING['option'] = 'FILE_HANDLER'
    _PyTwinConstants.LOGGING['filepath'] = log_filepath
    _PyTwinConstants.LOGGING['level'] = level


def _set_pytwin_logging_option_to_stream_handler(level, stream=sys.stderr):
    """
    (internal) Set pytwin logging option to stream handler (sys.stderr is used by default) and save logging level.
    """
    _PyTwinConstants.LOGGING['option'] = 'STREAM_HANDLER'
    _PyTwinConstants.LOGGING['stream'] = stream
    _PyTwinConstants.LOGGING['level'] = level


def set_pytwin_working_dir(abs_path: str, erase: bool):
    """
    Set pytwin working directory. Only the first call to this method has an effect.

    Parameters
    ----------
    abs_path: str
        Absolute path to the working directory you want to use for pytwin package. It is created if it does not exist.
    erase: bool
        if True, erase non-empty existing working directory and create a new one. If False, use existing working
        directory as it is. Value has no effect if directory does not exist.

    Examples
    --------
    >>> from pytwin import set_pytwin_working_dir
    >>> set_pytwin_working_dir(abs_path='absolute_path_to_your_working_directory', erase=True)
    """
    if _PyTwinConstants.WORKING_DIRECTORY is None:
        if os.path.exists(abs_path):
            if len(os.listdir(abs_path)) == 0:
                # Directory exists and it is empty
                _PyTwinConstants.WORKING_DIRECTORY = abs_path
                return
            else:
                # Directory exists and it is not empty
                if erase:
                    shutil.rmtree(abs_path)
                    os.mkdir(abs_path)
                _PyTwinConstants.WORKING_DIRECTORY = abs_path
                return
        else:
            # Directory does not exist
            os.mkdir(abs_path)
            _PyTwinConstants.WORKING_DIRECTORY = abs_path
            return


def get_pytwin_logging_filepath():
    """
    Get pytwin logging file path if provided, return None if not.
    """
    if _PyTwinConstants.LOGGING['option'] == 'FILE_HANDLER':
        return _PyTwinConstants.LOGGING['filepath']
    return None


def get_pytwin_working_dir():
    """
    Get pytwin working directory if provided by user or lazily creates one in temp folder if not.

    Examples
    --------
    >>> from pytwin import get_pytwin_working_dir
    >>> wd = get_pytwin_working_dir()
    """
    if _PyTwinConstants.WORKING_DIRECTORY is None:
        # Lazily creates a temporary directory in temp folder (remove it if it exists)
        _PyTwinConstants.WORKING_DIRECTORY = _create_pytwin_temporary_dir()
    return _PyTwinConstants.WORKING_DIRECTORY


class PyTwinWorkingDirectoryError(Exception):
    def __str__(self):
        return f'[pyTwin][WorkingDirectoryError] {self.args[0]}'
