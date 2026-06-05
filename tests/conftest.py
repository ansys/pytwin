# Copyright (C) 2022 - 2026 ANSYS, Inc. and/or its affiliates.
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

# conftest.py
import os
import sys

import pytest


def pytest_collection_modifyitems(config, items):
    if sys.platform == "linux":
        for item in items:
            if "TestTbRom" in item.nodeid:
                item.add_marker(pytest.mark.forked)

# ---------------------------------------------------------------------------
# Coverage support for os.fork() + os._exit() used by pytest-forked on Linux
# ---------------------------------------------------------------------------
# pytest-forked forks a child with os.fork() and terminates it with os._exit(),
# which bypasses all Python cleanup (atexit, __del__, coverage finalisation).
# We wrap os.fork() so that the child:
#   1. Stops the inherited parent coverage instance (to avoid double-counting).
#   2. Starts a NEW coverage instance with a unique parallel data suffix.
#   3. Patches os._exit() to flush that data before the hard exit.
# ---------------------------------------------------------------------------
if sys.platform == "linux" and hasattr(os, "fork") and os.environ.get("COVERAGE_PROCESS_START"):

    import coverage as _coverage_module

    _real_fork = os.fork

    def _coverage_aware_fork():
        pid = _real_fork()

        if pid == 0:
            # ---- We are in the forked child process ----

            # 1. Stop and discard the inherited parent coverage instance so it
            #    does not interfere or write to the parent's data file.
            existing = _coverage_module.Coverage.current()
            if existing is not None:
                existing.stop()

            # 2. Start a brand-new coverage instance.
            #    data_suffix=True makes coverage.py append .HOSTNAME.PID.RANDOM
            #    to the filename, ensuring each child writes a unique file.
            _child_cov = _coverage_module.Coverage(
                config_file=os.environ["COVERAGE_PROCESS_START"],
                data_suffix=True,
            )
            _child_cov.start()

            # 3. Patch os._exit so coverage is saved before the hard kill.
            _real_exit = os._exit

            def _exit_with_coverage_save(code):
                _child_cov.stop()
                _child_cov.save()
                _real_exit(code)

            os._exit = _exit_with_coverage_save

        return pid

    os.fork = _coverage_aware_fork
