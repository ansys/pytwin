""".. _ref_post_processing_exhaust_manifold:

Postprocessing using PyVista and Matplotlib
-------------------------------------------
This example uses PyVista and Matplotlib to demonstrate PyFluent
postprocessing capabilities. The 3D model in this example
is an exhaust manifold that has high temperature flows passing
through it. The flow through the manifold is turbulent and
involves conjugate heat transfer.

"""
###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports and set the configuration.

import ansys.fluent.core as pyfluent


from ansys.fluent.visualization import set_config
from ansys.fluent.visualization.matplotlib import Plots
from ansys.fluent.visualization.pyvista import Graphics

from pytwin import TwinModel, examples

#set_config(blocking=True, set_view_on_display="isometric")

###############################################################################
# Download files and launch Fluent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Download the case and data files and launch Fluent as a service in solver
# mode with double precision and two processors. Read in the case and data
# files.

import_case = examples.download_file("T_Junction.cas.h5", "other_files")

solver_session = pyfluent.launch_fluent(
    precision="double", processor_count=2, start_transcript=False, mode="solver"
)

solver_session.tui.file.read_case(import_case)

###############################################################################
# Get graphics object
# ~~~~~~~~~~~~~~~~~~~
# Get the graphics object.

graphics = Graphics(session=solver_session)

###############################################################################
# Create graphics object for mesh display
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a graphics object for the mesh display.

mesh1 = graphics.Meshes["mesh-1"]

###############################################################################
# Show edges
# ~~~~~~~~~~
# Show edges on the mesh.

mesh1.show_edges = True

###############################################################################
# Get surfaces list
# ~~~~~~~~~~~~~~~~~
# Get the surfaccase list.

mesh1.surfaces_list = [
    "symmetry_solid",
    "convection",
]
mesh1.display("window-1")



solver_session.exit()
