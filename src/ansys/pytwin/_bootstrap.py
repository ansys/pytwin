# Copyright (C) 2022 - 2026 Synopsys, Inc. and ANSYS, Inc. All rights reserved.
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

"""Load native prerequisites before importing third-party packages."""

import ctypes
import os
import platform
import sys

_libstdcpp = None

if platform.system() == "Linux":
    _runtime_dir = getattr(sys, "_MEIPASS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_runtime"))
    _libstdcpp_path = os.path.join(str(_runtime_dir), "libstdc++.so.6")
    try:
        _libstdcpp = ctypes.CDLL(_libstdcpp_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise ImportError(
            f"Failed to preload PyTwin's bundled C++ runtime at {_libstdcpp_path}: {error}. "
            "Ensure the bundled library and its system dependencies are available. "
            "Import PyTwin before other native packages. If an incompatible libstdc++ is already loaded, "
            "restart Python with a compatible C++ runtime or preload the bundled library using LD_PRELOAD."
        ) from error
