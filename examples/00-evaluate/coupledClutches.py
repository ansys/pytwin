"""pyTwin: Twin evaluation example --------------------------------------- This example shows how you can use pyTwin
to load and evaluate a Twin model. The model consists in a coupled clutches with 4 inputs (applied torque,
3 clutches opening) and 3 outputs (computed torque on each of the clutches) """

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports, which includes downloading and importing the input files
import csv
import platform

import matplotlib.pyplot as plt
import os
import pandas as pd
from src.ansys.twin.evaluate.evaluate import TwinModel
from src.ansys.twin import examples

twin_file = examples.download_file("CoupledClutches_23R1_other.twin", "twin_files")
csv_input = examples.download_file("CoupledClutches_input.csv", "twin_input_files")
twin_config = examples.download_file("CoupledClutches_config.json", "twin_input_files") # We will try to locate
# the model_setup, which provides default model start and parameter values override
cur_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


###############################################################################
# Auxiliary functions definition
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Definition of load_data function (loading Twin inputs data from CSV file) and plot_result_comparison for
# post processing the results

def load_data(inputs: str):
    """Load a CSV input file into a Pandas Dataframe. Inputs is the path of the CSV file to be loaded,
    containing the Time column and all the Twin inputs data"""

    # Clean CSV headers if exported from Twin builder
    def clean_column_names(column_names):
        for name_index in range(len(column_names)):
            clean_header = column_names[name_index].replace("\"", "").replace(" ", "").replace("]", "").replace("[", "")
            name_components = clean_header.split(".", 1)
            # The column name should match the last word after the "." in each column
            column_names[name_index] = name_components[-1]

        return column_names

    # #### Data loading (into Pandas DataFrame) and pre-processing ###### #
    # C engine can't read rows with quotes, reading just the first row
    input_header_df = pd.read_csv(inputs, header=None, nrows=1, sep=r',\s+',
                                  engine='python', quoting=csv.QUOTE_ALL)

    # Reading all values from the csv but skipping the first row
    inputs_df = pd.read_csv(inputs, header=None, skiprows=1)
    inputs_header_values = input_header_df.iloc[0][0].split(',')
    clean_column_names(inputs_header_values)
    inputs_df.columns = inputs_header_values

    return inputs_df


def plot_result_comparison(step_by_step_results: pd.DataFrame, batch_results: pd.DataFrame):
    """Compare the results obtained from 2 different simulations executed on the same TwinModel.
    The 2 results dataset are provided as Pandas Dataframe. The function will plot the different results for all the
    outputs and save the plot as a file "results.png" """
    pd.set_option('display.precision', 12)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.expand_frame_repr', False)

    # Plotting the runtime outputs
    columns = step_by_step_results.columns[1::]
    result_sets = 2  # Results from only step-by-step, batch_mode
    fig, ax = plt.subplots(ncols=result_sets, nrows=len(columns), figsize=(18, 7))
    if len(columns) == 1:
        single_column = True
    else:
        single_column = False

    fig.subplots_adjust(hspace=0.5)
    fig.set_tight_layout({"pad": .0})
    # x_data = runtime_result_df.iloc[:, 0]

    for ind, col_name in enumerate(columns):
        # Plot runtime results
        if single_column:
            axes0 = ax[0]
            axes1 = ax[1]

        else:
            axes0 = ax[ind, 0]
            axes1 = ax[ind, 1]

        step_by_step_results.plot(x=0, y=col_name, ax=axes0, ls=":", color='g',
                                  title='Twin Runtime - Step by Step')
        axes0.legend(loc=2)
        axes0.set_xlabel('Time [s]')

        # Plot Twin batch mode csv results
        batch_results.plot(x=0, y=col_name, ax=axes1, ls="-.", color='g',
                           title='Twin Runtime - Batch Mode')
        axes1.legend(loc=2)
        axes1.set_xlabel('Time [s]')

        if ind > 0:
            axes0.set_title('')
            axes1.set_title('')

    # Show plot
    plt.style.use('seaborn')
    plt.show()
    plt.savefig(os.path.join(cur_dir, 'results.png'))


###############################################################################
# Defining external files path
# ~~~~~~~~~~~~~~~~~~~
# Defining the runtime log path as well as loading the input data


runtime_log = os.path.join(cur_dir, 'model_{}.log'.format(platform.system()))
twin_model_input_df = load_data(csv_input)
data_dimensions = twin_model_input_df.shape
number_of_datapoints = data_dimensions[0] - 1

###############################################################################
# Loading the Twin Runtime and instantiating it
# ~~~~~~~~~~~~~~~~~~~
# Loading the Twin Runtime and instantiating it.


print('Loading model: {}'.format(twin_file))
twin_model = TwinModel(twin_file)

###############################################################################
# Setting up the initial settings of the Twin and initializing it
# ~~~~~~~~~~~~~~~~~~~
# Defining the initial inputs of the Twin, initializing it and collecting the initial outputs values


twin_model.initialize_evaluation(json_config_filepath=twin_config)
outputs = [twin_model.evaluation_time]
for item in twin_model.outputs:
    outputs.append(twin_model.outputs[item])

###############################################################################
# Step by step simulation mode
# ~~~~~~~~~~~~~~~~~~~
# Looping over all the input data, simulating the Twin one time step at a time and collecting corresponding outputs


sim_output_list_step = [outputs]
data_index = 0
while data_index < number_of_datapoints:
    # Gets the stop time of the current simulation step
    time_end = twin_model_input_df.iloc[data_index + 1][0]
    step = time_end - twin_model.evaluation_time
    inputs = dict()
    for column in twin_model_input_df.columns[1::]:
        inputs[column] = twin_model_input_df[column][data_index]
    twin_model.evaluate_step_by_step(step_size=step, inputs=inputs)
    outputs = [twin_model.evaluation_time]
    for item in twin_model.outputs:
        outputs.append(twin_model.outputs[item])
    sim_output_list_step.append(outputs)
    data_index += 1
results_step_pd = pd.DataFrame(sim_output_list_step, columns=['Time'] + list(twin_model.outputs),
                               dtype=float)

# ############################################################################## Batch simulation mode
# ~~~~~~~~~~~~~~~~~~~ Resetting/re-initializing the Twin and running it in batch mode (i.e. passing all the input
# data, simulating all the data points, and collecting all the outputs at once)


data_index = 0
inputs = dict()
for column in twin_model_input_df.columns[1::]:
    inputs[column] = twin_model_input_df[column][data_index]
twin_model.initialize_evaluation(inputs=inputs, json_config_filepath=twin_config)
outputs = [twin_model.evaluation_time]
for item in twin_model.outputs:
    outputs.append(twin_model.outputs[item])
results_batch_pd = twin_model.evaluate_batch(twin_model_input_df)

###############################################################################
# Post processing
# ~~~~~~~~~~~~~~~~~~~
# Plotting the different results and saving the image on disk

plot_result_comparison(results_step_pd, results_batch_pd)
