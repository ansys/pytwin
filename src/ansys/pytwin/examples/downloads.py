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
EXAMPLES_REPO = "https://github.com/ansys/example-data/raw/master/pytwin/"
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
    """Download a file from a URL."""
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
    """Download a folder from a URL."""
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
    Download a file that is used for a PyTwin example.

    The files are downloaded from the PyTwin example files repository whose URL is given by the
    ``pytwin.examples.downloads.EXAMPLES_REPO`` constant. All example files are downloaded to a persistent cache to
    avoid downloading the same file twice.

    Parameters
    ----------
    file_name : str
        Name of the example file.
    directory : str
        Path to the directory in the example files repository where the example file is stored.
    force_download : bool, optional
        Whether to force deletion of an example file so that it can be downloaded again. The default is ``False``.
    destination : str, optional
        Path to download the example file to. The default is ``None``, in which case the example file is
        downloaded to the user's temporary folder.

    Returns
    -------
    str
        Path to the downloaded example file.

    Examples
    --------
    >>> from pytwin import download_file
    >>> path = download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
    """
    if not destination:
        destination = EXAMPLES_PATH
    if force_download:
        local_path = os.path.join(destination, directory, os.path.basename(file_name))
        if os.path.exists(local_path):
            os.unlink(local_path)
    return _download_file(file_name, directory, destination)


def load_data(inputs: str):
    """
    Load the input data from a CVS file into a Pandas dataframe.

    Parameters
    ----------
    inputs : str
        Path of the CSV file. This file must contain the ``Time`` column and all input data
        for the twin model.

    Returns
    -------
    inputs_df: pandas.DataFrame
        A Pandas dataframe storing time values and all corresponding input data.

    Examples
    --------
    >>> from pytwin import load_data, download_file
    >>> csv_input = download_file("CoupledClutches_input.csv", "twin_input_files")
    >>> twin_model_input_df = load_data(csv_input)
    """

    # Clean CSV headers if exported from Twin Builder
    def clean_column_names(column_names):
        for name_index in range(len(column_names)):
            clean_header = column_names[name_index].replace('"', "").replace(" ", "").replace("]", "").replace("[", "")
            name_components = clean_header.split(".", 1)
            # The column name should match the last word after the "." in each column
            column_names[name_index] = name_components[-1]

        return column_names

    # Read column header names
    input_header_df = pd.read_csv(inputs, header=None, nrows=1, sep=r",\s+", engine="python", quoting=csv.QUOTE_ALL)

    # Read data, clean header names, and assemble final dataframe
    inputs_df = pd.read_csv(inputs, header=None, skiprows=1)
    inputs_header_values = input_header_df.iloc[0][0].split(",")
    clean_column_names(inputs_header_values)
    inputs_df.columns = inputs_header_values

    return inputs_df
