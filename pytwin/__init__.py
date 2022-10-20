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
from .settings import pytwin_logging_is_enabled

PYTWIN_LOG_DEBUG = PyTwinLogLevel.PYTWIN_LOG_DEBUG
PYTWIN_LOG_WARNING = PyTwinLogLevel.PYTWIN_LOG_WARNING
PYTWIN_LOG_ERROR = PyTwinLogLevel.PYTWIN_LOG_ERROR
PYTWIN_LOG_CRITICAL = PyTwinLogLevel.PYTWIN_LOG_CRITICAL

PYTWIN_LOGGING_OPT_FILE = PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE
PYTWIN_LOGGING_OPT_CONSOLE = PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE
PYTWIN_LOGGING_OPT_NOLOGGING = PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING

"""
PUBLIC API TO PYTWIN EVALUATE 
"""
from .evaluate import TwinModel
from .evaluate import TwinModelError

"""
PUBLIC API TO PYTWIN RUNTIME 
"""
from .twin_runtime import TwinRuntime
from .twin_runtime import TwinRuntimeError
