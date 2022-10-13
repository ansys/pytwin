"""
pytwin.

library
"""
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version('pytwin')

"""
PUBLIC API TO PYTWIN SETTINGS 
"""
from .settings import get_pytwin_working_dir
from .settings import get_pytwin_log_file
from .settings import get_pytwin_logger
from .settings import modify_pytwin_logging
from .settings import modify_pytwin_working_dir
from .settings import PyTwinLogOption
from .settings import PyTwinLogLevel
from .settings import PyTwinSettingsError

PYTWIN_LOG_ALL = PyTwinLogLevel.PYTWIN_LOG_ALL
PYTWIN_LOG_WARNING = PyTwinLogLevel.PYTWIN_LOG_WARNING
PYTWIN_LOG_ERROR = PyTwinLogLevel.PYTWIN_LOG_ERROR
PYTWIN_LOG_FATAL = PyTwinLogLevel.PYTWIN_LOG_FATAL

PYTWIN_LOGGING_OPT_FILE = PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE
PYTWIN_LOGGING_OPT_CONSOLE = PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE
PYTWIN_LOGGING_OPT_NOLOGGING = PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING
