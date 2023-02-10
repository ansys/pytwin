.. _ref_index_api_sdk:

Twin runtimes
=============

The :class:`TwinRuntime <ansys.pytwin.TwinRuntime` class provides access to
twin runtime functionalities.

.. currentmodule:: pytwin

.. autosummary::
   :toctree: _autosummary

   TwinRuntime

Workflow example
----------------

This code shows how to load the ``TwinRuntime`` module and print twin information. 

.. code-block:: pycon

   >>> from pytwin import TwinRuntime, LogLevel
   >>> from pytwin import examples
   >>> twin_file = examples.download_file("CoupledClutches_23R1_other.twin", "twin_files")
   >>> twin_runtime = TwinRuntime(twin_model)  # Load the Runtime
   >>> twin_runtime.print_model_info(max_var_to_print=10)  # Print twin information
   ------------------------------------- Model Info -------------------------------------
   Twin Runtime Version: 2.4.0.0
   Model Name: CoupledClutchesTwin
   Number of outputs: 3
   Number of Inputs: 4
   Number of parameters: 3
   Default time end: 1.5
   Default step size: 0.001
   Default tolerance(Integration Accuracy): 0.0001

   Output names:
                Name  Type  ... Nominal              Description
   0  Clutch1_torque  Real  ...     1.0  TWIN_VARPROP_NOTDEFINED
   1  Clutch2_torque  Real  ...     1.0  TWIN_VARPROP_NOTDEFINED
   2  Clutch3_torque  Real  ...     1.0  TWIN_VARPROP_NOTDEFINED

   [3 rows x 9 columns]


   Input names:
            Name  Type  ... Nominal              Description
   0  Clutch1_in  Real  ...     1.0  TWIN_VARPROP_NOTDEFINED
   1  Clutch2_in  Real  ...     1.0  TWIN_VARPROP_NOTDEFINED
   2  Clutch3_in  Real  ...     1.0  TWIN_VARPROP_NOTDEFINED
   3   Torque_in  Real  ...     1.0  TWIN_VARPROP_NOTDEFINED

   [4 rows x 9 columns]


   Parameter names:
               Name  ...                                 Description
   0  solver.method  ...  Solver integration method (ADAMS=1, BDF=2)
   1  solver.abstol  ...                   Solver absolute tolerance
   2  solver.reltol  ...                   Solver relative tolerance

   [3 rows x 9 columns]
   >>> twin_runtime.twin_instantiate()  # Instantiate the Runtime
   >>> twin_runtime.twin_initialize()  # Initialize the twin simulation
   >>> print(twin_runtime.twin_get_outputs())  # Collect and print initial output values
   [0.0, 0.0, 0.0]
   >>> twin_runtime.twin_simulate(0.001)  # Simulate the twin until the end time
   >>> twin_runtime.twin_close()  # Close the Runtime
