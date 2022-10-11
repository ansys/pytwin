import os
import shutil
import sys


class _PyTwinConstants(object):
    """
    (internal) This class host pytwin package constants that must be shared between all pytwin object instances.
    Explicit modifications of the constant values at runtime is forbidden and may cause unexpected behavior.
    """
    LOGGING = {'option': 'UNDEFINED'}
    LOGGER_NAME = 'pytwin_logger'
    WORKING_DIRECTORY = None


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


def set_pytwin_working_directory_path(working_directory_path: str, erase: bool = False):
    """
    Set pytwin working directory.

    Parameters
    ----------
    working_directory_path: str
        The empty working directory you want to use for pytwin package. The method creates it if it does not exist.
    erase: bool
        Erase non-empty existing working directory and create a new one (default to False).

    Raises
    ------
    PyTwinWorkingDirectoryError
        If erase is False and provided working directory exists and is not empty.
    """
    if os.path.exists(working_directory_path):
        if len(os.listdir(working_directory_path)) == 0:
            # Directory exists and it is empty
            _PyTwinConstants.WORKING_DIRECTORY = working_directory_path
            return
        else:
            # Directory exists and it is not empty
            if erase:
                # Remove directory and create new one
                shutil.rmtree(working_directory_path)
                os.mkdir(working_directory_path)
                _PyTwinConstants.WORKING_DIRECTORY = working_directory_path
                return
            else:
                # Raise exception
                msg = 'The directory provided to set pytwin working directory already exists and is not empty!'
                msg += f'\n({working_directory_path})'
                msg += '\nPlease remove existing directory, empty it or set \'erase\' argument to True.'
                raise PyTwinWorkingDirectoryError(msg)
    else:
        # Directory does not exist
        os.mkdir(working_directory_path)
        _PyTwinConstants.WORKING_DIRECTORY = working_directory_path
        return


def get_pytwin_logging_filepath():
    if _PyTwinConstants.LOGGING['option'] == 'FILE_HANDLER':
        return _PyTwinConstants.LOGGING['filepath']
    return None


def get_pytwin_working_directory_path():
    """
    Get pytwin working directory.
    """
    return _PyTwinConstants.WORKING_DIRECTORY


class PyTwinWorkingDirectoryError(Exception):
    def __str__(self):
        return f'[pyTwin][WorkingDirectoryError] {self.args[0]}'
