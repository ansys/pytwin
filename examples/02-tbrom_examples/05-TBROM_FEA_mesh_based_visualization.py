""".. _ref_example_TBROM_FEA_mesh_projection:

3D field ROM example with FEA mesh based visualization
------------------------------------------------------

This example shows how PyTwin can be used to load and evaluate a twin model, and visualize
the ROM results on a FEA mesh.
Once the twin model is evaluated, PyDPF is used to read in the FEA mesh on which the ROM
results will be projected. PyVista is then used to perform the 3D visualization with different
options. For more information, see the
`PyDPF documentation <https://dpf.docs.pyansys.com/version/stable/getting_started/index.html>`_ and
`PyVista documentation <https://docs.pyvista.org/version/stable/index.html>`_.
"""

###############################################################################
# .. image:: /_static/TBROM_FEA_mesh_projection.png
#   :width: 400pt
#   :align: center

# sphinx_gallery_thumbnail_path = '_static/TBROM_FEA_mesh_projection'

###############################################################################
# .. note::
#   This example uses PyDPF which requires a DPF server installation (version 2024 R1-pre0 or higher)
#   as a first step. More information on how to download and install DPF server is available here :
#   https://dpf.docs.pyansys.com/version/stable/getting_started/index.html#install-dpf-server
#
#   To be able to use the functionalities to project an output field on a mesh, you
#   must have a twin with one or more TBROMs. The output mode coefficients for the TBROMs
#   must be enabled when exporting the TBROMs and connected to twin outputs with following
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
#   To be able to use the functionalities to project an output field on a mesh with interpolation,
#   you need to have a Twin with 1 or more TBROM, for which its geometry is embedded when
#   exporting the TBROMs to Twin Builder
# .. image:: /_static/point_generation.png
#   :width: 200pt
#   :align: center

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports, which include downloading and importing the input
# files.

import ansys.dpf.core as dpf
from pytwin import TwinModel, download_file
import pyvista as pv

twin_file = download_file("FEADeformationTBROM_23R2.twin", "twin_files", force_download=True)
fea_file = download_file("FEA_deformation.rst", "other_files", force_download=True)

###############################################################################
# Define ROM inputs
# ~~~~~~~~~~~~~~~~~
# Define the ROM inputs.
rom_inputs = {"Force_X_Component": 101.471}

###############################################################################
# Load the twin runtime and generate temperature results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the twin runtime, initialize and extract ROM related information.
print("Initializing the Twin")
twin_model = TwinModel(twin_file)
twin_model.initialize_evaluation(inputs=rom_inputs)
rom_name = twin_model.tbrom_names[0]
named_selections = twin_model.get_named_selections(rom_name)

###############################################################################
# Extract the FEA mesh information for projection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the Mechanical rst file through PyDPF and extract the mesh
print("Reading the FEA mesh")
ds = dpf.DataSources()
ds.set_result_file_path(fea_file)
streams = dpf.operators.metadata.streams_provider(data_sources=ds)

# extracting the grid associated to the fea model
whole_mesh = dpf.operators.mesh.mesh_provider(streams_container=streams).eval()
target_mesh = whole_mesh.grid

###############################################################################
# Project the TBROM field onto the targeted mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The projection is performed without interpolation (i.e. direct mapping of data)
print("Projecting the results on target mesh")
mesh_data = twin_model.project_tbrom_on_mesh(rom_name, target_mesh, False)

###############################################################################
# Post processing and field visualization using PyVista
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Creation of the scene and results to display
print("Post processing")
plotter = pv.Plotter()
plotter.set_background("white")
plotter.add_axes()
plotter.add_mesh(mesh_data, scalar_bar_args={"color": "black"})
# plotter.add_mesh(mesh_data, show_edges=True, scalar_bar_args={"color": "black"})
plotter.camera_position = [
    (-0.03962092584614037, 0.013987037327583286, 0.10356162483172728),
    (0.015819278345, 0.0001320060299999995, 0.005),
    (0.05026854357305644, 0.9925208192698814, -0.11124521042672807),
]
plotter.show()

###############################################################################
# .. image:: /_static/TBROM_FEA_mesh_projection.png
#   :width: 400pt
#   :align: center
