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

import ctypes
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "ansys" / "pytwin"
BOOTSTRAP_PATH = PACKAGE_DIR / "_bootstrap.py"


def test_linux_preloads_bundled_runtime():
    with patch("platform.system", return_value="Linux"), patch("ctypes.CDLL") as load_library:
        namespace = runpy.run_path(str(BOOTSTRAP_PATH))

    load_library.assert_called_once_with(str(PACKAGE_DIR / "twin_runtime" / "libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)
    assert namespace["_libstdcpp"] is load_library.return_value


def test_frozen_application_runtime_path(tmp_path):
    with (
        patch("platform.system", return_value="Linux"),
        patch.object(sys, "_MEIPASS", str(tmp_path), create=True),
        patch("ctypes.CDLL") as load_library,
    ):
        runpy.run_path(str(BOOTSTRAP_PATH))

    load_library.assert_called_once_with(str(tmp_path / "libstdc++.so.6"), mode=ctypes.RTLD_GLOBAL)


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_other_platforms_do_not_preload(system):
    with patch("platform.system", return_value=system), patch("ctypes.CDLL") as load_library:
        namespace = runpy.run_path(str(BOOTSTRAP_PATH))

    load_library.assert_not_called()
    assert namespace["_libstdcpp"] is None


def test_preload_failure_has_actionable_error():
    error = OSError("missing dependency")
    with patch("platform.system", return_value="Linux"), patch("ctypes.CDLL", side_effect=error):
        with pytest.raises(ImportError, match="Failed to preload.*libstdc\\+\\+.*LD_PRELOAD") as exception:
            runpy.run_path(str(BOOTSTRAP_PATH))

    assert exception.value.__cause__ is error


@pytest.mark.parametrize(
    "statement",
    ["import pytwin", "from pytwin import TwinModel, TwinRuntime, read_binary", "import pytwin.evaluate.twin_model"],
)
def test_bootstrap_precedes_third_party_imports(statement):
    script = f"""
import builtins
import ctypes
import platform
import sys
from unittest.mock import patch

sys.path.insert(0, {str(PACKAGE_DIR.parent)!r})
original_import = builtins.__import__

class ThirdPartyImportReached(BaseException):
    pass

def check_import(name, *args, **kwargs):
    if name.split('.')[0] in ('tqdm', 'numpy', 'pandas'):
        load_library.assert_called_once_with(
            {str(PACKAGE_DIR / 'twin_runtime' / 'libstdc++.so.6')!r}, mode=ctypes.RTLD_GLOBAL
        )
        raise ThirdPartyImportReached()
    return original_import(name, *args, **kwargs)

with patch('platform.system', return_value='Linux'), patch('ctypes.CDLL') as load_library:
    with patch('builtins.__import__', side_effect=check_import):
        try:
            exec({statement!r})
        except ThirdPartyImportReached:
            pass
        else:
            raise AssertionError('No third-party import reached')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
