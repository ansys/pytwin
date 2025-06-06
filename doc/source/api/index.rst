.. _ref_index_api:

=============
API reference
=============

This section describes the core Pythonic interfaces for twin runtimes.
Here, you can find all APIs for consuming twin runtimes, from the lowest-level APIs
with the :class:`pytwin.TwinRuntime` class to the higher-level
APIs with the :class:`pytwin.TwinModel` class.

Twin runtimes
-------------

The :class:`TwinRuntime <pytwin.TwinRuntime>` class provides access to all
the twin runtime functionalities. It is the lowest-level API of the twin runtime SDK.
For a workflow example, see :ref:`ref_index_api_sdk`.

Evaluate
--------

The :class:`TwinModel <pytwin.TwinModel>` class implements a higher-level
abstraction to facilitate the manipulation and execution of a twin model. For more
information, see :ref:`ref_index_api_evaluate`.

Global settings
---------------

PyTwin provides global settings for configuring and changing both logging and directory options.
For more information, see :ref:`ref_index_api_logging`.

Post-processing
---------------

The post-processing module contains physics-specific functions for post-processing twin model results,
see :ref:`ref_index_postprocessing`.

Other functions
---------------

Some other PyTwin functions are available and used in examples. For more information,
see :ref:`ref_index_api_example`.

.. currentmodule:: pytwin

.. autosummary::
   :toctree: _autosummary

.. toctree::
   :maxdepth: 4
   :hidden:

   sdk/index
   evaluate/index
   logging/index
   postprocessing/index
   examples/index
