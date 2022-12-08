"""Functions to download sample datasets from the PyAnsys data repository.
Examples
--------
#>>> from pytwin import examples
#>>> filename = examples.download_file("CoupleClutches_22R2_other.twin", "twin_files")
#>>> filename
'/home/user/.local/share/TwinExamples/twin/CoupleClutches_22R2_other.twin'
"""
import ast
import csv
import os
import shutil
import tempfile
from typing import Optional
import urllib.request
import zipfile

import pandas as pd

temp_folder = tempfile.gettempdir()
# the REPO url needs to have "raw" and not "tree", otherwise xml file are downloaded instead of raw versions
EXAMPLES_REPO = "https://github.com/pyansys/example-data/raw/master/pytwin/"
EXAMPLES_PATH = os.path.join(temp_folder, "TwinExamples")


def get_ext(filename: str) -> str:
    """Extract the extension of a file."""
    ext = os.path.splitext(filename)[1].lower()
    return ext


def delete_downloads() -> bool:
    """Delete all downloaded examples to free space or update the files."""
    shutil.rmtree(EXAMPLES_PATH)
    os.makedirs(EXAMPLES_PATH)
    return True


def _decompress(filename: str) -> None:
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall(EXAMPLES_PATH)
    return zip_ref.close()


def _get_file_url(directory, filename=None):
    if not filename:
        return EXAMPLES_REPO + "/".join([directory])
    else:
        return EXAMPLES_REPO + "/".join([directory, filename])


def _retrieve_file(url, filename, directory, destination=None):
    """Download a file from an url"""
    # First check if file has already been downloaded
    if not destination:
        destination = EXAMPLES_PATH
    local_path = os.path.join(destination, directory, os.path.basename(filename))
    local_path_no_zip = local_path.replace(".zip", "")
    if os.path.isfile(local_path_no_zip) or os.path.isdir(local_path_no_zip):
        return local_path_no_zip

    urlretrieve = urllib.request.urlretrieve

    dirpath = os.path.dirname(local_path)
    if not os.path.isdir(destination):
        os.mkdir(destination)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    # Perform download
    _, resp = urlretrieve(url, local_path)
    return local_path


def _retrieve_folder(url, directory, destination=None):
    """Download a folder from an url"""
    # First check if folder exists
    if not destination:
        destination = EXAMPLES_PATH
    local_path = os.path.join(destination, directory)
    if os.path.isdir(local_path):
        return local_path

    with urllib.request.urlopen(url) as response:  # nosec
        data = response.read().decode("utf-8").split("\n")

    if not os.path.isdir(destination):
        os.mkdir(destination)
    if not os.path.isdir(local_path):
        os.makedirs(local_path)

    for line in data:
        if "js-navigation-open Link--primary" in line:
            filename = ast.literal_eval(line[line.find("title=") + len("title=") : line.rfind(" data-pjax")])
            _download_file(directory, filename, destination)
    return local_path


def _download_file(filename, directory, destination=None):
    if not filename:
        url = _get_file_url(directory)
        local_path = _retrieve_folder(url, directory, destination)
    else:
        url = _get_file_url(directory, filename)
        local_path = _retrieve_file(url, filename, directory, destination)

    return local_path


def download_file(
    file_name: str, directory: str, force_download: Optional[bool] = False, destination: Optional[str] = None
):
    """
    Download an example data file.
    Examples files are downloaded to a persistent cache to avoid downloading the same file twice.

    Parameters
    ----------
    file_name : str
        Path of the file in the examples folder.
    directory : str
        Subfolder storing the input file
    force_download : bool, optional
        Force to delete file and download file again. Default value is ``False``.
    destination : str, optional
        Path to download files to. The default is the user's temporary folder.

    Returns
    -------
    str
        Path to the folder containing all example data files.

    Examples
    --------
    Download an example result file and return the path of the file
    >>> from pytwin import examples
    >>> path = examples.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
    """
    if not destination:
        destination = EXAMPLES_PATH
    if force_download:
        local_path = os.path.join(destination, file_name)
        if os.path.exists(local_path):
            os.unlink(local_path)
    return _download_file(file_name, directory, destination)


def load_data(inputs: str):
    """Load a CSV input file into a Pandas Dataframe. Inputs is the path of the CSV file to be loaded,
    containing the Time column and all the Twin inputs data"""

    # Clean CSV headers if exported from Twin builder
    def clean_column_names(column_names):
        for name_index in range(len(column_names)):
            clean_header = column_names[name_index].replace('"', "").replace(" ", "").replace("]", "").replace("[", "")
            name_components = clean_header.split(".", 1)
            # The column name should match the last word after the "." in each column
            column_names[name_index] = name_components[-1]

        return column_names

    # #### Data loading (into Pandas DataFrame) and pre-processing ###### #
    # C engine can't read rows with quotes, reading just the first row
    input_header_df = pd.read_csv(inputs, header=None, nrows=1, sep=r",\s+", engine="python", quoting=csv.QUOTE_ALL)

    # Reading all values from the csv but skipping the first row
    inputs_df = pd.read_csv(inputs, header=None, skiprows=1)
    inputs_header_values = input_header_df.iloc[0][0].split(",")
    clean_column_names(inputs_header_values)
    inputs_df.columns = inputs_header_values

    return inputs_df
