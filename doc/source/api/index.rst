.. _ref_index_api:

=============
API reference
=============

This section describes the core Pythonic interfaces for Twin Runtimes.
Here, you can find the application programming interfaces for 
using the core Twin Runtime SDK functionalities, as well as an example
of higher level abstraction class to evaluate the Runtimes.

Twin Runtime SDK
----------------

Implementation of the :ref:`ref_index_api_sdk` class enabling access to the core
Twin Runtime SDK functionalities.

Evaluate
--------

:ref:`ref_index_api_evaluate` class is an example of higher level abstraction implementation
to facilitate the manipulation and execution of Twin Runtimes.

Global Settings
---------------

:ref:`ref_index_api_logging` describe the global settings (e.g. logging and working directory options) available from
PyTwin package and how to change them from their default values

Other Functions
---------------

Additional useful functions used for the examples implementation are described in :ref:`ref_index_api_example`.

.. currentmodule:: pytwin

.. autosummary::
   :toctree: _autosummary

.. toctree::
   :maxdepth: 4
   :hidden:

   sdk/index
   evaluate/index
   logging/index
   examples/index
