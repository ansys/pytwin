.. _ref_index_api_evaluate:

Evaluate
========

The :class:`TwinModel <ansys.pytwin.TwinModel` class implements a higher-level
abstraction to facilitate the manipulation and execution of a twin model. 

.. currentmodule:: pytwin

.. autosummary::
   :toctree: _autosummary

   TwinModel

Workflow example
----------------

This code shows how to set up and evaluate a twin model. 

.. code-block:: pycon

   >>> from pytwin import TwinModel, download_file, load_data

   # Download the input files
   >>> twin_file = download_file("CoupledClutches_23R1_other.twin", "twin_files")
   >>> csv_input = download_file("CoupledClutches_input.csv", "twin_input_files")
   >>> twin_config = download_file("CoupledClutches_config.json", "twin_input_files")

   # Load the CSV file containing the twin input data over time
   >>> twin_model_input_df = load_data(csv_input)
   # Load and instantiate the twin model
   >>> twin_model = TwinModel(twin_file)

   >>> inputs = dict()
   >>> for column in twin_model_input_df.columns[1::]:
   ...     inputs[column] = twin_model_input_df[column][0]
   ...

   # Initialize the twin model given initial input values and a configuration file for parameters values
   >>> twin_model.initialize_evaluation(inputs=inputs, json_config_filepath=twin_config)

   # Evaluate the twin model in batch mode and print the computed output values
   >>> results_batch_pd = twin_model.evaluate_batch(twin_model_input_df)
   >>> print(results_batch_pd)
          Time  Clutch1_torque  Clutch2_torque  Clutch3_torque
   0     0.000      -10.000000             0.0             0.0
   1     0.001       -9.999997             0.0             0.0
   2     0.002       -9.999999             0.0             0.0
   3     0.003       -9.999985             0.0             0.0
   4     0.004       -9.999956             0.0             0.0
   ...     ...             ...             ...             ...
   1496  1.496        0.000000             0.0             0.0
   1497  1.497        0.000000             0.0             0.0
   1498  1.498        0.000000             0.0             0.0
   1499  1.499        0.000000             0.0             0.0
   1500  1.500        0.000000             0.0             0.0

   [1501 rows x 4 columns]
