""".. _ref_example_heatExchangerRS:

Parametric twin evaluation of a response surface ROM
----------------------------------------------------
This example shows how you can use PyTwin to load and evaluate a twin model
and simulate multiple parametric variations. The model is based on a
response surface ROM created from a steady state thermal model of a heat
exchanger. The inputs are the minimum and maximum heat flows on the inner face.
The outputs are the inner temperature and the temperatures from three
temperature probes within the solid and outer temperature. The model is tested
against different input values to evaluate the corresponding temperature responses.
"""

###############################################################################
# .. image:: /_static/heatExchangerRS.png
#   :width: 400pt
#   :align: center

# sphinx_gallery_thumbnail_path = '_static/heatExchangerRS.png'

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports, which include downloading and importing the input files.

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from pytwin import TwinModel, download_file

twin_file = download_file("HeatExchangerRS_23R1_other.twin", "twin_files")

###############################################################################
# Define inputs and simulation settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the inputs and simulation settings.

heat_flow_min = 0.0
heat_flow_max = 50000.0
step = 50.0

###############################################################################
# Define auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define auxiliary functions for comparing and plotting the results from
# different input values evaluated on the twin model.


def plot_result_comparison(results: pd.DataFrame):
    """Compare the results obtained from the different input values evaluated on
    the twin model. The results datasets are provided as Pandas dataframes. The
    function plots the results for few variables of particular interest."""

    pd.set_option("display.precision", 12)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.expand_frame_repr", False)

    color = ["g"]
    # Output ordering: T_inner, T1_out, T_outer, T2_out, T3_out
    x_ind = 0
    y0_ind = 1
    y1_ind = 2
    y2_ind = 4
    y3_ind = 5
    y4_ind = 3

    # Plot simulation results (outputs versus input)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(18, 7))

    fig.subplots_adjust(hspace=0.5)
    fig.set_tight_layout({"pad": 0.0})

    axes0 = ax

    results.plot(x=x_ind, y=y0_ind, ax=axes0, ls="-.", label="{}".format("T inner"))
    results.plot(x=x_ind, y=y1_ind, ax=axes0, ls="-.", label="{}".format("T1"))
    results.plot(x=x_ind, y=y2_ind, ax=axes0, ls="-.", label="{}".format("T2"))
    results.plot(x=x_ind, y=y3_ind, ax=axes0, ls="-.", label="{}".format("T3"))
    results.plot(x=x_ind, y=y4_ind, ax=axes0, ls="-.", label="{}".format("T outer"))

    axes0.set_title("Heat Exchanger thermal response")
    axes0.set_xlabel(results.columns[x_ind] + " [W]")
    axes0.set_ylabel("Temperature [deg C]")

    # Show plot
    plt.show()


###############################################################################
# Load the twin runtime and instantiate it
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the twin runtime and instantiate it.

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
input_name = list(twin_model.inputs.keys())[0]
for dp in numpy.linspace(start=heat_flow_min, stop=heat_flow_max, num=int((heat_flow_max - heat_flow_min) / step + 1)):
    # Initialize twin with input values and collect output value

    dp_input = {input_name: dp}
    twin_model.initialize_evaluation(inputs=dp_input)
    outputs = [dp]
    for item in twin_model.outputs:
        outputs.append(twin_model.outputs[item])
    results.append(outputs)
    if dp % 1000 == 0.0:
        print("Simulating the model with input {}".format(dp))
sim_results = pd.DataFrame(results, columns=[input_name] + list(twin_model.outputs), dtype=float)


###############################################################################
# Plot results
# ~~~~~~~~~~~~
# Plotg the results and save the image on disk.

plot_result_comparison(sim_results)
