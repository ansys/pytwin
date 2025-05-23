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
import shutil
import time
import tracemalloc

from pytwin import TwinModel
from pytwin.settings import get_pytwin_working_dir

TBROM_MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "ThermalTBROM_FieldInput_23R1.twin")
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


class TestTwinModelFinalize:
    def test_twin_model_finalizer_free_memory(self):
        # Init unit test
        reinit_settings()
        # TwinModel memory is freed at the end of a loop and its model directory is deleted
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()
        allocated_mem_size = ""
        model_dir = ""
        for i in range(4):
            twin_model = TwinModel(model_filepath=TBROM_MODEL_FILEPATH)
            model_dir_old = model_dir
            model_dir = twin_model.model_dir
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot, "lineno")
            allocated_mem_size_old = allocated_mem_size
            allocated_mem_size = f"{top_stats[0]}".split("size=")[1].split(",")[0].split("+")[1].split(" ")[0]
            time.sleep(0.25)
            if i > 1:
                # Current twin_model directory exists
                assert os.path.exists(model_dir)
                # Previous twin_model directory as been deleted
                assert not os.path.exists(model_dir_old)
                # Previous twin_model memory as been freed (allow for +/- 0.5% difference of memory
                assert (
                    1.005 * float(allocated_mem_size_old)
                    > float(allocated_mem_size)
                    > 0.995 * float(allocated_mem_size_old)
                )

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
