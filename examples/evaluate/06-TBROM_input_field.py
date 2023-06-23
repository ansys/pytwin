""".. _ref_example_TBROM_images:

3D field ROM example for input field snapshot projection and snapshot generation on demand
------------------------------------------------------------------------------------------

This example shows how to use PyTwin to load and evaluate a twin model that has a
field ROM with inputs parameterized by both scalar and field data. The example shows
also how to evaluate the output field data in the form of snapshots.

.. note::
   To be able to use the functionalities to project an input field snapshot, you must have a
   twin with one or more TBROMs parameterized by input field data. Input mode coefficients
   for TBROMs are connected to the twin's inputs following these conventions:
   
   - If there are multiple TBROMs in the twin, the format for the name of the
      twin input is ``"input field name"_mode_"i"_"tbrom name"``.
   - If there is a single TBROM in the twin, the format for the name of the twin
      input is ``"input field name"_mode_"i"``.

   To be able to use the functionalities to generate an output field snapshot on demand, you
   must have a twin with one or more TBROMs. The output mode coefficients for the TBROMs
   must be enabled when exporting the TBROMs and connected to twin outputs with following
   these conventions:
   
   - If there are multiple TBROMs in the twin, the format for the name of the twin
     output is ``"outField_mode_"i"_"tbrom name"``.
   - If there is a single TBROM in the twin, the format for the name of the twin
     output is ``"outField_mode_"i"``.

   To be able to use the functionalities to generate points file on demand, you need to have a Twin with 1 or
   more TBROM, for which its geometry is embedded when exporting the ROM to Twin Builder
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
import math

import matplotlib.pyplot as plt
import pandas as pd
from pytwin import TwinModel, download_file

twin_file = download_file(
    "ThermalTBROM_FieldInput_23R1.twin", "twin_files", force_download=True
)  # , force_download=True)
inputfieldsnapshots = [
    download_file("TEMP_1.bin", "twin_input_files/inputFieldSnapshots", force_download=True),
    download_file("TEMP_2.bin", "twin_input_files/inputFieldSnapshots", force_download=True),
    download_file("TEMP_3.bin", "twin_input_files/inputFieldSnapshots", force_download=True),
]

###############################################################################
# Define ROM inputs
# ~~~~~~~~~~~~~~~~~
# Define the ROM inputs.

rom_inputs = [4000000, 5000000, 6000000]


###############################################################################
# Define auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define auxiliary functions for comparing and plotting the results from
# different input values evaluated on the twin model and for computing
# the norm of the output field.


def plot_result_comparison(results: pd.DataFrame):
    """Compare the results obtained from the different input values evaluated on
    the twin model. The results datasets are provided as Pandas dataframes. The
    function plots the results for a few variables of particular interest."""

    pd.set_option("display.precision", 12)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.expand_frame_repr", False)

    color = ["g"]
    # Output ordering: T_inner, T1_out, T_outer, T2_out, T3_out
    x_ind = 0
    y0_ind = 2
    y1_ind = 3
    y2_ind = 4

    # Plot simulation results (outputs versus input)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(18, 7))

    fig.subplots_adjust(hspace=0.5)
    fig.set_tight_layout({"pad": 0.0})

    axes0 = ax

    results.plot(x=x_ind, y=y0_ind, ax=axes0, ls="dashed", label="{}".format("Maximum Deformation (Twin output)"))
    results.plot(
        x=x_ind, y=y1_ind, ax=axes0, ls="-.", label="{}".format("Maximum Deformation (output field " "reconstruction)")
    )
    results.plot(
        x=x_ind,
        y=y2_ind,
        ax=axes0,
        ls="-.",
        label="{}".format("Maximum Deformation (output field reconstruction on Group_2)"),
    )

    axes0.set_title("T-junction deformation response")
    axes0.set_xlabel(results.columns[x_ind] + " [Pa]")
    axes0.set_ylabel("Deformation [m]")

    # Show plot
    plt.show()


def norm_vector_field(field: list):
    """Compute the norm of a vector field."""

    norm = []
    for i in range(0, int(len(field) / 3)):
        x = field[i * 3]
        y = field[i * 3 + 1]
        z = field[i * 3 + 2]
        norm.append(math.sqrt(x * x + y * y + z * z))
    return norm


###############################################################################
# Load the twin runtime and generate temperature results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the twin runtime and generate temperature results from the TBROM.

print("Loading model: {}".format(twin_file))
twin_model = TwinModel(twin_file)

###############################################################################
# Evaluate the twin with different input values and collect corresponding outputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Because the twin is based on a static model, two options can be considered:
#
# - Set the initial input value to evaluate and run the initialization function (current approach).
# - Create an input dataframe considering all input values to evaluate and run the batch function
#   to evaluate. In this case, to execute the transient simulation, a time dimension must be
#   arbitrarily defined.

results = []
input_name_all = list(twin_model.inputs.keys())
output_name_all = list(twin_model.outputs.keys())

# remove the TBROM related pins from the twin's list of inputs and outputs
input_name_without_mcs = []
for i in input_name_all:
    if "mode" not in i:
        input_name_without_mcs.append(i)
print(f"Twin physical inputs : {input_name_without_mcs}")
output_name_without_mcs = []
for i in output_name_all:
    if "mode" not in i:
        output_name_without_mcs.append(i)
print(f"Twin physical outputs : {output_name_without_mcs}")

# collect the TBROM and input field related information, the Twin must be initialized first
twin_model.initialize_evaluation()
print(f"TBROMs part of the Twin : {twin_model.tbrom_names}")
romname = twin_model.tbrom_names[0]
print(f"Input fields associated to the TBROM {romname} : {twin_model.get_rom_inputfieldsnames(romname)}")
fieldname = twin_model.get_rom_inputfieldsnames(romname)[0]
print(f"Named selections associated to the TBROM {romname} : {twin_model.get_rom_nslist(romname)}")
ns = twin_model.get_rom_nslist(romname)[1]

input_name = input_name_without_mcs[0]
for i in range(0, len(rom_inputs)):
    # Initialize twin with input values and collect output value
    dp = rom_inputs[i]
    dp_input = {input_name: dp}
    dp_field_input = {romname: {fieldname: inputfieldsnapshots[i]}}
    twin_model.initialize_evaluation(inputs=dp_input, inputfields=dp_field_input)
    outputs = [dp]
    for item in output_name_without_mcs:
        outputs.append(twin_model.outputs[item])
    outfield = twin_model.snapshot_generation(romname, False)  # generating the field output on the entire domain
    outfieldns = twin_model.snapshot_generation(romname, False, ns)  # generating the field output on "Group_2"
    twin_model.points_generation(romname, True, ns)  # generating the points file on "Group_2"
    twin_model.snapshot_generation(romname, True, ns)  # generating the field snapshot on "Group_2"
    outputs.append(max(norm_vector_field(outfield)))
    outputs.append(max(norm_vector_field(outfieldns)))
    results.append(outputs)
sim_results = pd.DataFrame(
    results, columns=[input_name] + output_name_without_mcs + ["MaxDefSnapshot", "MaxDefSnapshotNs"], dtype=float
)

###############################################################################
# Plot results
# ~~~~~~~~~~~~
# Plot the results and save the image on disk.

plot_result_comparison(sim_results)
