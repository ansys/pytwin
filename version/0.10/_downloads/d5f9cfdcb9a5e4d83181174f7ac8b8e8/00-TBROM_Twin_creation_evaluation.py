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

""".. _ref_example_pyAEDT_Twin:

Static ROM creation, Twin generation and evaluation
---------------------------------------------------
This example shows how you can use PyAEDT and PyTwin together. PyAEDT is used on one side
to generate a static ROM based on training data, and then create and export a twin model.
PyTwin is used on the other side to load and evaluate the generated twin model.
Parametric field history results (i.e. transient results collected on a fixed time interval
for different parameters values) have been generated out of LS-DYNA simulations. Once the twin
is generated, it can be evaluated to make predictions of time series (e.g. transient displacement at
a given probe location)

.. note::
   This example requires an installation of Ansys Twin Builder 2025R1 or above.
"""

###############################################################################
# .. image:: /_static/TBROM_parametric_field_history.png
#   :width: 400pt
#   :align: center

# sphinx_gallery_thumbnail_path = '_static/TBROM_parametric_field_history.png'

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports, which include downloading and importing the
# input files.

import csv
import json
import os
import shutil
import zipfile

from ansys.aedt.core import TwinBuilder, generate_unique_project_name
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytwin import TwinModel, download_file, read_binary

training_data_zip = "training_data.zip"
download_folder = download_file(training_data_zip, "other_files", force_download=True)
training_data_folder = os.path.join(os.path.dirname(download_folder), "training_data")
if os.path.exists(training_data_folder) and os.path.isdir(training_data_folder):
    shutil.rmtree(training_data_folder)
# Unzip training data
with zipfile.ZipFile(download_folder) as zf:
    zf.extractall(training_data_folder)

###############################################################################
# Define auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define auxiliary functions for computing # the norm of the output field.


def norm_vector_history(field: np.ndarray, index: int):
    """Compute the norm of a vector field."""
    result = []
    for i in range(0, field.shape[0]):
        vec = field[i, :].reshape((-1, 3))
        result.append(np.sqrt((vec * vec).sum(axis=1))[index])
    return result


###############################################################################
# Project settings
# ~~~~~~~~~~~~~~~~
# Define Twin Builder project related settings and inputs files. The following code
# launches Twin Builder in graphical mode. You can change the Boolean parameter
# ``non_graphical`` to ``True`` to launch Twin Builder in non-graphical mode.
# You can also change the Boolean parameter ``new_thread`` to ``False`` to
# launch Twin Builder in an existing AEDT session if one is running.

source_build_conf_file = "SROMbuild.conf"
source_props_conf_file = "SROM_props.conf"  # Note : SROM_props.conf may need to be adapted if inputs names change!
desktop_version = "2025.1"
non_graphical = False
new_thread = True
confpath = os.path.join(training_data_folder, source_build_conf_file)
doefile = os.path.join(training_data_folder, "doe.csv")
settingsfile = os.path.join(training_data_folder, "settings.json")

###############################################################################
# ROM creation
# ~~~~~~~~~~~~
# Create a Twin Builder instance
tb = TwinBuilder(
    project=generate_unique_project_name(), version=desktop_version, non_graphical=non_graphical, new_desktop=new_thread
)

# Switch the current desktop configuration and the schematic environment to "Twin Builder" (Static ROM Builder is
# available with Twin Builder).
current_desktop_config = tb.odesktop.GetDesktopConfiguration()
current_schematic_environment = tb.odesktop.GetSchematicEnvironment()
tb.odesktop.SetDesktopConfiguration("Twin Builder")
tb.odesktop.SetSchematicEnvironment(1)

# Get the static ROM builder object
rom_manager = tb.odesign.GetROMManager()
static_rom_builder = rom_manager.GetStaticROMBuilder()

# Build the static ROM with specified configuration file
static_rom_builder.Build(confpath.replace("\\", "/"))

# Test if ROM was created successfully
static_rom_path = os.path.join(training_data_folder, "StaticRom.rom")
if os.path.exists(static_rom_path):
    tb.logger.info("Built intermediate rom file successfully at: %s", static_rom_path)
else:
    tb.logger.error("Intermediate rom file not found at: %s", static_rom_path)

# Create the ROM component definition in Twin Builder
rom_manager.CreateROMComponent(static_rom_path.replace("\\", "/"), "staticrom")

###############################################################################
# Twin composition and export
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grid distance for ease in calculations.
G = 0.00254

# Create the Twin Subsheet.
parentDesign = "ParentDesign"
subSheet = "SubSheet"
tb.create_subsheet(subSheet, parentDesign)
idSubSheet = len(tb.odesign.GetSubDesigns())
tb.set_active_design(parentDesign + "::SubSheet" + str(idSubSheet))

# Place the ROM component, parameterize and connect to port interfaces.
rom1 = tb.modeler.schematic.create_component("ROM1", "", "staticrom", [40 * G, 25 * G])
rom1.parameters["field_data_storage_period"] = "0"
rom1.parameters["store_snapshots"] = "1"
tb.modeler.schematic.add_pin_iports("ROM1", rom1.id)

tb.logger.info("Subsheet created, starting Twin compilation")

# twin compilation
twinname = "TwinModel"
tb.odesign.CompileAsTwin(twinname, ["1", "0.001", "1e-4", "1e-12"])

# twin export
twinFile = os.path.join(training_data_folder, twinname + ".twin")
tb.modeler.schematic.o_simmodel_manager.ExportTwinModel(twinname, twinFile, "twin", "other", "1", "1")

tb.logger.info("Twin compiled and exported. Closing Twin Builder.")

# Restore earlier desktop configuration and schematic environment.
tb.odesktop.SetDesktopConfiguration(current_desktop_config)
tb.odesktop.SetSchematicEnvironment(current_schematic_environment)

tb.release_desktop()

###############################################################################
# Twin evaluation
# ~~~~~~~~~~~~~~~
# Evaluate the exported Twin with PyTwin and post process the results

# Read column header names
input_header_df = pd.read_csv(doefile, header=None, nrows=1, sep=r";\s+", engine="python", quoting=csv.QUOTE_ALL)

# Read data, clean header names, and assemble final dataframe
inputs_df = pd.read_csv(doefile, header=None, skiprows=1, sep=";")
inputs_header_values = input_header_df.iloc[0][0].split(";")
inputs_df.columns = inputs_header_values
inputs = dict()
data_index = 0
for column in inputs_df.columns[1::]:
    inputs[column] = inputs_df[column][data_index]

# Read settings file
with open(settingsfile) as f:
    data = json.load(f)
timeGrid = data["timeSeries"]["timeStepsValues"]
romdim = int(data["dimensionality"][0])

# Load and evaluate the Twin
print("Loading model: {}".format(twinFile))
print("Twin inputs evaluated: {}".format(inputs))
twin_model = TwinModel(twinFile)
romname = twin_model.tbrom_names[0]
twin_model.initialize_evaluation(inputs=inputs)


###############################################################################
# Results post processing
# ~~~~~~~~~~~~~~~~~~~~~~~
# The time history predictions for a particular probe (point #23) are post processed and compared to reference results.

ref_snapshot = os.path.join(training_data_folder, "snapshots", inputs_df[inputs_df.columns[0]][data_index])
ref_data = read_binary(ref_snapshot).reshape(len(timeGrid), -1)
snapshot = twin_model.get_snapshot_filepath(romname)
snap_data = read_binary(snapshot).reshape(len(timeGrid), -1)
probe_id = 23
plt.plot(timeGrid, norm_vector_history(ref_data, probe_id), label="reference")
plt.plot(timeGrid, norm_vector_history(snap_data, probe_id), label="rom")
plt.legend(loc="lower left")
plt.title("Displacement vs Time for point {}".format(probe_id))
plt.show()

# Clean up the downloaded data
shutil.rmtree(training_data_folder)
