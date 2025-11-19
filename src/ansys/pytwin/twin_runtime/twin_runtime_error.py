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

from enum import Enum


class TwinRuntimeError(Exception):
    def __init__(self, message, twin_runtime=None, twin_status=None):
        self.message = message
        self.twin_status = twin_status
        if twin_runtime is not None:
            self.dll_message = twin_runtime.twin_get_status_string()
            if twin_status is None:
                self.twin_status = twin_runtime._twin_status

    def add_message(self, new_message):
        self.message += "\n" + new_message


class PropertyStatusFlag(Enum):
    TWIN_VARPROP_OK = 0
    TWIN_VARPROP_NOTDEFINED = 1
    TWIN_VARPROP_NOTAPPLICABLE = 2
    TWIN_VARPROP_INVALIDVAR = 3
    TWIN_VARPROP_ERROR = 4


class PropertyNotDefinedError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status


class PropertyNotApplicableError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status


class PropertyInvalidError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status


class PropertyError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status
