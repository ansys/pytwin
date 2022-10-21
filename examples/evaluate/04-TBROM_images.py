""".. _ref_example_TBROM_images:

TBROM example for images generation
-----------------------------------

This example shows how PyTwin can be used to load and evaluate a Twin model in order to ROM visualization results in
the form of images with predefined views. The script takes user inputs to evaluate the ROM and will display the
corresponding images.
"""

# sphinx_gallery_thumbnail_path = '_static/TBROM_images_generation.png'

###############################################################################
# Import all necessary modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import matplotlib.pyplot as plt
import matplotlib.image as img

from pytwin import TwinModel
from pytwin import examples
from pytwin import get_pytwin_working_dir

twin_file = examples.download_file("ThermalTBROM_23R1_other.twin", "twin_files")


###############################################################################
# User inputs
# ~~~~~~~~~~~
# Defining user inputs

rom_inputs = {"main_inlet_temperature": 353.15, "side_inlet_temperature": 293.15}
rom_parameters = {"ThermalROM23R1_1_colorbar_min": 290, "ThermalROM23R1_1_colorbar_max": 360}

###############################################################################
# Loading the Twin Runtime and generate the temperature results from the TBROM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Loading model: {}'.format(twin_file))
twin_model = TwinModel(twin_file)

# TODO - following are SDK atomic calls, need to use TBROM class ultimately
twin_model._twin_runtime.twin_instantiate()

directory_path = os.path.join(get_pytwin_working_dir(), 'ROM_files')
visualization_info = twin_model._twin_runtime.twin_get_visualization_resources()
rom_name = ""
for model_name, data in visualization_info.items():
    twin_model._twin_runtime.twin_set_rom_image_directory(model_name, directory_path)
    rom_name = model_name

twin_model._initialize_evaluation(inputs=rom_inputs, parameters=rom_parameters)

###############################################################################
# Post-processing
# ~~~~~~~~~~~~~~~

image = img.imread(os.path.join(directory_path, rom_name, 'View1_0.000000.png'))
plt.imshow(image)
plt.show()

