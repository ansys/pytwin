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

    def test_load_data(self):
        csv_input = dld.download_file("CoupledClutches_input.csv", "twin_input_files")
        data = dld.load_data(csv_input)
        assert not data.empty
