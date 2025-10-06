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

"""
pytwin.

library
"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

# Checking if tqdm is installed.
# If it is, the default value for progress_bar is true.
try:
    from tqdm import tqdm  # noqa: F401

    _HAS_TQDM = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_TQDM = False


__version__ = importlib_metadata.version("pytwin")

"""
PUBLIC API TO PYTWIN SETTINGS
"""
from pytwin.settings import (
    PyTwinLogLevel,
    PyTwinLogOption,
    PyTwinSettingsError,
    get_pytwin_log_file,
    get_pytwin_log_level,
    get_pytwin_logger,
    get_pytwin_working_dir,
    modify_pytwin_logging,
    modify_pytwin_working_dir,
    pytwin_logging_is_enabled,
)

PYTWIN_LOG_DEBUG = PyTwinLogLevel.PYTWIN_LOG_DEBUG
PYTWIN_LOG_INFO = PyTwinLogLevel.PYTWIN_LOG_INFO
PYTWIN_LOG_WARNING = PyTwinLogLevel.PYTWIN_LOG_WARNING
PYTWIN_LOG_ERROR = PyTwinLogLevel.PYTWIN_LOG_ERROR
PYTWIN_LOG_CRITICAL = PyTwinLogLevel.PYTWIN_LOG_CRITICAL

PYTWIN_LOGGING_OPT_FILE = PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE
PYTWIN_LOGGING_OPT_CONSOLE = PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE
PYTWIN_LOGGING_OPT_NOLOGGING = PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING

"""
PUBLIC API TO PYTWIN EVALUATE
"""
from pytwin.evaluate.tbrom import (
    read_binary,
    read_snapshot_size,
    snapshot_to_array,
    write_binary,
)
from pytwin.evaluate.twin_model import TwinModel, TwinModelError

"""
PUBLIC API TO PYTWIN POSTPROCESSING
"""
from pytwin.postprocessing.postprocessing import (
    stress_strain_component,
)

"""
PUBLIC API TO PYTWIN RUNTIME
"""
from pytwin.twin_runtime.log_level import LogLevel
from pytwin.twin_runtime.twin_runtime_core import TwinRuntime, TwinRuntimeError

"""
PUBLIC API TO EXAMPLES
"""
from pytwin.examples.downloads import download_file, load_data
