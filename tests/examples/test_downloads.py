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

import pytwin.examples.downloads as dld


class TestDownloads:
    def test_delete_downloads(self):
        assert dld.delete_downloads()

    def test_get_ext(self):
        assert dld.get_ext(__file__) == ".py"

    def test_decompress(self):
        dld.delete_downloads()
        unit_test_folder = os.path.join(dld.EXAMPLES_PATH, "unit_test_folder")
        if os.path.exists(unit_test_folder):
            shutil.rmtree(unit_test_folder)
        unit_test_zip_file = os.path.join(os.path.dirname(__file__), "data", "unit_test_folder.zip")
        dld._decompress(unit_test_zip_file)
        assert os.path.exists(unit_test_folder)
        assert len(os.listdir(unit_test_folder)) == 2

    def test_download_file(self):
        dld.delete_downloads()
        my_file_path = dld.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        assert os.path.exists(my_file_path)

    def test_cache_download_file(self):
        dld.delete_downloads()
        my_file_path = dld.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        first_modified_time = os.path.getmtime(my_file_path)
        my_file_path = dld.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=False)
        second_modified_time = os.path.getmtime(my_file_path)
        assert first_modified_time == second_modified_time

    def test_force_download_file(self):
        dld.delete_downloads()
        my_file_path = dld.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        first_modified_time = os.path.getmtime(my_file_path)
        my_file_path = dld.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        second_modified_time = os.path.getmtime(my_file_path)
        assert first_modified_time != second_modified_time

    def test_load_data(self):
        csv_input = dld.download_file("CoupledClutches_input.csv", "twin_input_files")
        data = dld.load_data(csv_input)
        assert not data.empty
