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

""".. _ref_example_TBROM_parametric_field_history:

3D parametric field history ROM example
---------------------------------------

This example shows how to use PyTwin to load and evaluate a twin model built upon a
parametric field history ROM. Such ROM, created with Static ROM Builder, has parameters
that can be changed from one evaluation to another, and will output field predictions over
a time grid. The example shows how to evaluate and post process the results at different
time points.
"""

###############################################################################
# .. image:: /_static/TBROM_pfieldhistory_t250.png
#   :width: 400pt
#   :align: center

# sphinx_gallery_thumbnail_path = '_static/TBROM_pfieldhistory_t250.png'

###############################################################################
# The example model is a simplified 3D mechanical frame tightened by two bolts.
# The bolts are represented as forces applied to the tips of the component.
# The magnitude of the deformation is dependent on the magnitude of both force
# parameters. Additionally, this component is made of a material that exhibits
# time-dependent behavior, allowing the structure to undergo a mechanical
# phenomenon known as creep. Essentially, the deformation of the structure
# changes over time, even under constant applied forces, which makes the problem
# suitable for parametric field history ROM.

###############################################################################
# .. note::
#   To be able to use the functionalities illustrated in this example, you
#   must have a twin with one or more TBROMs. The output mode coefficients for the TBROMs
#   must be enabled when exporting the TBROMs and connected to twin outputs following
#   these conventions:
#
#   - If there are multiple TBROMs in the twin, the format for the name of the twin
#     output must be ``outField_mode_{mode_index}_{tbrom_name}``.
#   - If there is a single TBROM in the twin, the format for the name of the twin
#     output must be ``outField_mode_{mode_index}``.
# .. image:: /_static/snapshot_generation.png
#   :width: 300pt
#   :align: center

###############################################################################
# .. note::
#   To be able to use the functionalities to visualize the results, you need to have a Twin with 1 or
#   more TBROM, for which its geometry is embedded when exporting the TBROMs to Twin Builder
# .. image:: /_static/point_generation.png
#   :width: 200pt
#   :align: center

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports, which include downloading and importing the input files.

import matplotlib.pyplot as plt
import numpy as np
from pytwin import TwinModel, download_file
import pyvista as pv

twin_file = download_file("TwinPFieldHistory_wGeo_25R2.twin", "twin_files", force_download=True)

###############################################################################
# Define ROM scalar inputs
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Define the ROM scalar inputs.

rom_inputs = {"testROM_25r2_1_force_1_Magnitude": 9500, "testROM_25r2_1_force_2_Magnitude": 16500}

###############################################################################
# Load the twin runtime and generate displacement results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the twin runtime, initialize the evaluation and display displacement results from the TBROM.

print("Loading model: {}".format(twin_file))
twin_model = TwinModel(twin_file)
twin_model.print_model_info()
romname = twin_model.tbrom_names[0]
twin_model.initialize_evaluation(parameters=rom_inputs)
field_data = twin_model.get_tbrom_output_field(romname)
plotter = pv.Plotter()
plotter.set_background("white")
plotter.add_axes()
plotter.add_mesh(field_data, scalar_bar_args={"color": "black"}, clim=[0.0, 6.0])
plotter.add_title("Time = {}".format(twin_model.evaluation_time))
plotter.show()
print(max(field_data.active_scalars))

###############################################################################
# .. image:: /_static/TBROM_pfieldhistory_t0.png
#   :width: 400pt
#   :align: center

###############################################################################
# Evaluate the twin at different time points (3D field visualization)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Because the twin is based on a parametric field history ROM, the entire field
# history has been computed during the Twin initialization. In order to visualize
# the results at different time points, the Twin can be advanced and simulated
# over time. A linear interpolation is performed in case the time selected
# is between two time steps of the field history.

# After 100 seconds
twin_model.evaluate_step_by_step(100.0)

field_data = twin_model.get_tbrom_output_field(romname)
plotter = pv.Plotter()
plotter.set_background("white")
plotter.add_axes()
plotter.add_mesh(field_data, scalar_bar_args={"color": "black"}, clim=[0.0, 6.0])
plotter.add_title("Time = {}".format(twin_model.evaluation_time))
plotter.show()
print(max(field_data.active_scalars))

###############################################################################
# .. image:: /_static/TBROM_pfieldhistory_t100.png
#   :width: 400pt
#   :align: center

# After 250 seconds
twin_model.evaluate_step_by_step(150.0)

field_data = twin_model.get_tbrom_output_field(romname)
plotter = pv.Plotter()
plotter.set_background("white")
plotter.add_axes()
plotter.add_mesh(field_data, scalar_bar_args={"color": "black"}, clim=[0.0, 6.0])
plotter.add_title("Time = {}".format(twin_model.evaluation_time))
plotter.show()
print(max(field_data.active_scalars))

###############################################################################
# .. image:: /_static/TBROM_pfieldhistory_t250.png
#   :width: 400pt
#   :align: center

###############################################################################
# Evaluate the twin at different time points (time series)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If we continue simulating the Twin over time, the field results won't change
# anymore since we have reached the end time of the field history. To evaluate the ROM
# again, it needs to be re-initialized, with the possibility to change input parameters values.
# In this section, we will see how to build and visualize a transient prediction
# at a given location.

timegrid = twin_model.get_tbrom_time_grid(romname)
print(timegrid)

fieldName = twin_model.get_field_output_name(romname) + "-normed"

point = np.array([0.0, 0.0, 0.0])
idx = field_data.find_closest_point(point)
outputValues = []

closest_pt = field_data.points[idx]
print(closest_pt)

twin_model.initialize_evaluation(parameters=rom_inputs)
outputValues.append(field_data.point_data[fieldName][idx])

for i in range(1, len(timegrid)):
    step = timegrid[i] - timegrid[i - 1]
    twin_model.evaluate_step_by_step(step)
    outputValues.append(field_data.point_data[fieldName][idx])

plt.plot(timegrid, outputValues)
plt.title("Displacement vs Time for point {}".format(np.round(closest_pt, 3)))
plt.show()
