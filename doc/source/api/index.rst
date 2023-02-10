.. _ref_index_api:

=============
API reference
=============

This section describes the core Pythonic interfaces for twin runtimes.
Here, you can find the APIs for using twin runtime functionalities
the higher-level abstraction class for evaluating twin runtimes.

Twin runtimes
-------------

The :class:`TwinRuntime <ansys.pytwin.TwinRuntime` class provides access to
twin runtime functionalities. For a workflow example, see :ref:`ref_index_api_sdk`.

Evaluate
--------

The :class:`TwinModel <ansys.pytwin.TwinModel` class implements a higher-level
abstraction to facilitate the manipulation and execution of a twin model. For more
information, see :ref:`ref_index_api_sdk`.

Global settings
---------------

PyTwin provides global settings for configuring and changing both logging and directory options.
For more information, see :ref:`ref_index_api_logging`.

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
   examples/index
