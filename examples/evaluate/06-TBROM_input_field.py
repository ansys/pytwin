""".. _ref_example_TBROM_images:

3D field ROM example for input field snapshot projection as well as snapshot generation on demand
-------------------------------------------------------------------------------------------------

This example shows .....

.. note::
   To be able to use the input field snapshot projection capability, the ROM must be created with input field data,
   and the associated input mode coefficients must be exposed as Twin inputs
   To be able to use the output field snapshot generation on demand capability, the ROM must be exported with the option
   to expose mode coefficients as outputs, and these coefficients must be exposed as Twin outputs

"""

###############################################################################
# .. image:: /_static/TBROM_images_generation.png
#   :width: 400pt
#   :align: center

# sphinx_gallery_thumbnail_path = '_static/TBROM_images_generation.png'

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports, which include downloading and importing the input
# files.

import os
import struct
import numpy as np
from pytwin import TwinModel, download_file

twin_file = download_file("TwinModelInOutField_23R11.twin", "twin_files", force_download=False)

###############################################################################
# Define ROM inputs
# ~~~~~~~~~~~~~~~~~
# Define the ROM inputs.

rom_inputs = {"Pressure_Magnitude": 3000000.0}

###############################################################################
# Define auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~



def read_basis(fn):
    #print('Reading SVDBasis...')
    fr = open(fn,"rb")
    struct.unpack('cccccccccccccccc', fr.read(16))[0]
    nb_val = struct.unpack('Q', fr.read(8))[0]
    nb_modes = struct.unpack('Q', fr.read(8))[0]
    basis = []
    for i in range(nb_modes):
        vec = []
        for j in range(nb_val):
            vec.append (struct.unpack('d', fr.read(8))[0])
        basis.append(vec)
    fr.close()
    return basis

def read_binary(file):
    fr = open(file,"rb")
    nbdof = struct.unpack('Q', fr.read(8))[0]
    vec = []
    for i in range(nbdof):
        vec.append(struct.unpack('d', fr.read(8))[0])
    fr.close()
    return vec

def write_binary(fn, vec):
    fw = open(fn, "wb")
    fw.write(struct.pack("Q", len(vec)))
    for i in vec:
        fw.write(struct.pack("d", i))
    return True

# the following functions are prototyped (should be part of the TwinModel/TBROM class ultimately with associated attributes)

def snapshot_projection(twin_model, snapshot):
    modes_coef = []
    vec = read_binary(snapshot)
    vecnp = np.array(vec)
    basis = twin_model.inputSVD
    nb_modes = len(basis)
    for i in range(nb_modes):
        modenp = np.array(basis[i])
        coef = modenp.dot(vecnp)
        modes_coef.append(coef)
    twin_model.inputModeCoef = modes_coef # probably need to distinguish inputModeCoef from regular inputs

def snapshot_generation(twin_model, on_disk, output_file):
    basis = twin_model.outputSVD
    vec = np.zeros(len(basis[0]))
    nb_modes = len(basis)
    for i in range(nb_modes):
        modenp = np.array(basis[i])
        vec = vec + twin_model.outputModeCoef[i]*modenp # probably need to distinguish outputModeCoef from regular outputs
    if on_disk:
        write_binary(output_file, vec)
    else:
        return vec



###############################################################################
# Load the twin runtime and generate temperature results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the twin runtime and generate temperature results from the TBROM.

print("Loading model: {}".format(twin_file))
twin_model = TwinModel(twin_file)

twin_model.initialize_evaluation(inputs=rom_inputs) # twin_model needs to be initialized first before rom_name is available...

rom_name = twin_model.tbrom_names[0]

inputSVDpath = os.path.join(twin_model._tbrom_resource_directory(rom_name), "binaryInputField_inputTemperature", "basis.svd")
inputSVD = read_basis(inputSVDpath)

twin_model.inputSVD = inputSVD

outputSVDpath = os.path.join(twin_model._tbrom_resource_directory(rom_name), "binaryOutputField", "basis.svd")
outputSVD = read_basis(outputSVDpath)

twin_model.outputSVD = outputSVD

inputSnapshot = 'C:/Users/cpetre/TestTwin/inputTemperature/Snapshots/TEMP_6.bin'

snapshot_projection(twin_model, inputSnapshot) # project input field snapshot to get associated mode coefficients

for i in range(0, len(inputSVD)):
    input = {list(twin_model.inputs.keys())[i]:twin_model.inputModeCoef[i]}
    rom_inputs.update(input)
twin_model.initialize_evaluation(inputs=rom_inputs)
print(twin_model.inputs)

outputModeCoef = []
print(twin_model.outputs)
for key, item in twin_model.outputs.items():
    if "outField" in key:
        outputModeCoef.append(item)

print(outputModeCoef)
twin_model.outputModeCoef = outputModeCoef

snapshotfile = os.path.join(twin_model.tbrom_directory_path, rom_name, 'snapshot_test.bin')
snapshot_generation(twin_model, True, snapshotfile) # generation snapshot on the disk
res = snapshot_generation(twin_model, False, snapshotfile) # generation snasphot in memory
print(res)