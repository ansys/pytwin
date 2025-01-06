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

import os
import uuid

from pytwin import PyTwinLogLevel, get_pytwin_logger, get_pytwin_working_dir, pytwin_logging_is_enabled
from pytwin.settings import PYTWIN_SETTINGS


class Model:
    """
    Provides the private base class for managing twin model evaluation.

    This class handles the model ID, working directory paths, logging,
    and error raising.

    A child class can overload the ``raise_model_error(self, msg)`` method
    to provide its own exception.
    """

    def __init__(self):
        self._id = f"{uuid.uuid4()}"[0:24].replace("-", "")
        self._model_name = None
        self._log_key = None

    def _log_message(self, msg: str, level: PyTwinLogLevel = PyTwinLogLevel.PYTWIN_LOG_INFO):
        """
        Provides a base class method for logging messages at key steps in the code logic.
        """
        msg = f"[{self._model_name}.{self._id}][{self._log_key}] {msg}"
        logger = get_pytwin_logger()
        if pytwin_logging_is_enabled():
            if level == PyTwinLogLevel.PYTWIN_LOG_DEBUG:
                logger.debug(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_INFO:
                logger.info(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_WARNING:
                logger.warning(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_ERROR:
                logger.error(msg)
                return
            if level == PyTwinLogLevel.PYTWIN_LOG_CRITICAL:
                logger.critical(msg)
                return

    def _raise_model_error(self, msg):
        """
        Overload this method in the child class for supplying your own exception class to catch errors explicitly.
        """
        raise ModelError(msg)

    def _raise_error(self, msg):
        """
        Provides a base class method for raising an error with a meaningful message.
        """
        self._log_message(msg, PyTwinLogLevel.PYTWIN_LOG_ERROR)
        self._raise_model_error(msg)

    @property
    def id(self):
        """Model unique ID."""
        return self._id

    @property
    def name(self):
        """Model name. Multiple models can share the same name."""
        return self._model_name

    @property
    def model_dir(self):
        """Model directory (within the global working directory)."""
        return os.path.join(get_pytwin_working_dir(), f"{self._model_name}.{self._id}")

    @property
    def model_temp(self):
        """Model temporary directory (within the global working directory). This temporary directory
        is shared by all models."""
        return os.path.join(get_pytwin_working_dir(), PYTWIN_SETTINGS.TEMP_WD_NAME)

    @property
    def model_log(self):
        """Path to the model log file that is used at twin runtime instantiation."""
        return os.path.join(self.model_temp, f"{self._id}.log")

    @property
    def model_log_link(self):
        """Path to the symbolic link to the model log file."""
        return os.path.join(self.model_dir, f"link.log")


class ModelError(Exception):
    def __str__(self):
        return f"[ModelError] {self.args[0]}"
