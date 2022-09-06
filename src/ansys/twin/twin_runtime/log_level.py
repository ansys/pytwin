from enum import Enum


class LogLevel(Enum):
    TWIN_LOG_ALL = 0
    TWIN_LOG_WARNING = 1
    TWIN_LOG_ERROR = 2
    TWIN_LOG_FATAL = 3
    TWIN_NO_LOG = 4
