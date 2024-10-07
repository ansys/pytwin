# Copyright (C) 2022 - 2024 ANSYS, Inc. and/or its affiliates.
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

from pytwin import PyTwinLogLevel, get_pytwin_log_file
from pytwin.evaluate.model import Model

UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings():
    import shutil

    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


class TestModel:
    def test_each_model_has_a_unique_identifier(self):
        # Init test context
        reinit_settings()
        id_store = []
        model_count = 10000
        # Run test
        for i in range(model_count):
            model = Model()
            model._log_message("hello!", PyTwinLogLevel.PYTWIN_LOG_INFO)
            id_store.append(model._id)
            assert id_store.count(model._id) == 1
        with open(get_pytwin_log_file(), "r") as f:
            assert len(f.readlines()) == model_count

    def test_multiple_model_log_in_same_logger(self):
        # Init test context
        reinit_settings()
        # Run test
        model1 = Model()
        model1._model_name = "model1"
        model1._id = "1"
        model2 = Model()
        model2._model_name = "model2"
        model2._id = "2"
        model1._log_message("Hello A from model 1!")
        model1._log_message("Hello B from model 1!")
        model2._log_message("Hello A from model 2!")
        model2._log_message("Hello B from model 2!")
        with open(get_pytwin_log_file(), "r") as f:
            assert len(f.readlines()) == 4
