"""
pytwin.

library
"""
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version('pytwin')

from .pytwin_logger import PyTwinLogLevel
from .pytwin_logger import PyTwinLoggingError
from .pytwin_logger import set_pytwin_logging
from .pytwin_logger import get_pytwin_logger

PYTWIN_LOG_ALL = PyTwinLogLevel.PYTWIN_LOG_ALL
PYTWIN_LOG_WARNING = PyTwinLogLevel.PYTWIN_LOG_WARNING
PYTWIN_LOG_ERROR = PyTwinLogLevel.PYTWIN_LOG_ERROR
PYTWIN_LOG_FATAL = PyTwinLogLevel.PYTWIN_LOG_FATAL
PYTWIN_NO_LOG = PyTwinLogLevel.PYTWIN_NO_LOG
