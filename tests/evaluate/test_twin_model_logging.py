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

import os
import shutil

from pytwin import PYTWIN_LOGGING_OPT_NOLOGGING
from pytwin.settings import get_pytwin_log_file, get_pytwin_working_dir, modify_pytwin_logging

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings():
    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        try:
            shutil.rmtree(UNIT_TEST_WD)
        except Exception as e:
            pass
    return UNIT_TEST_WD


class TestTwinModelLogging:
    def test_twin_model_no_logging(self):
        # Init unit test
        reinit_settings()
        # Twin Model does not log anything if PYTWIN_LOGGING_OPT_NOLOGGING
        modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_NOLOGGING)
        log_file = get_pytwin_log_file()
        assert log_file is None

    def test_clean_unit_test(self):
        reinit_settings()
        temp_wd = get_pytwin_working_dir()
        parent_dir = os.path.dirname(temp_wd)
        try:
            for dir_name in os.listdir(parent_dir):
                if dir_name not in temp_wd:
                    shutil.rmtree(os.path.join(parent_dir, dir_name))
        except Exception as e:
            pass
