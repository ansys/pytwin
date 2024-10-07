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

from pytwin import TwinRuntime, TwinRuntimeError
import pytwin.examples.downloads as downloads
import pytwin.twin_runtime.twin_runtime_error as twin_runtime_error


class TestTwinRuntime:
    def test_evaluate_twin_status(self):
        model_fp = downloads.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        twin_runtime = TwinRuntime(model_fp, load_model=True)
        twin_runtime.twin_instantiate()
        # Test TwinRuntime warning
        TwinRuntime.evaluate_twin_status(1, twin_runtime, "unit_test_method")
        # Test TwinRuntime error
        try:
            TwinRuntime.evaluate_twin_status(2, twin_runtime, "unit_test_method")
        except TwinRuntimeError as e:
            assert "error" in str(e)
        # Test TwinRuntime fatal error
        try:
            TwinRuntime.evaluate_twin_status(3, twin_runtime, "unit_test_method")
        except TwinRuntimeError as e:
            assert "fatal error" in str(e)

    def test_evaluate_twin_prop_status(self):
        model_fp = downloads.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        twin_runtime = TwinRuntime(model_fp, load_model=True)
        twin_runtime.twin_instantiate()

        # Property error
        try:
            TwinRuntime.evaluate_twin_prop_status(4, twin_runtime, "unit_test_method", 0)
        except twin_runtime_error.PropertyError as e:
            assert "error" in str(e)

        # Property invalid error
        try:
            TwinRuntime.evaluate_twin_prop_status(3, twin_runtime, "unit_test_method", 0)
        except twin_runtime_error.PropertyInvalidError as e:
            assert "error" in str(e)

        # Property not applicable error
        try:
            TwinRuntime.evaluate_twin_prop_status(2, twin_runtime, "unit_test_method", 0)
        except twin_runtime_error.PropertyNotApplicableError as e:
            assert "error" in str(e)

        # Property not defined error
        try:
            TwinRuntime.evaluate_twin_prop_status(1, twin_runtime, "unit_test_method", 0)
        except twin_runtime_error.PropertyNotDefinedError as e:
            assert "error" in str(e)
