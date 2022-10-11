PyTwin documentation |version|
================================

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   user_guide/index
   api/index
   examples/index
   contributing

Introduction
------------
Ansys Twin Builder and Twin Deployer are open solutions that allow engineers to create
simulation-based digital twins with Hybrid Analytics.

An analytics-driven, simulation-based digital twin is a connected, virtual replica of an
in-service physical asset, in the form of an integrated multidomain system simulation, that
mirrors the life and experience of the asset. Hybrid digital twins enable system design and
optimization and predictive maintenance, and they optimize industrial asset management.
By implementing Ansys Digital Twins, you can improve top-line revenue, manage bottom-line
costs and both gain and retain a competitive advantage.

.. image:: _static/ansys_dt.png
  :width: 800
  :alt: Ansys Digital Twins

What is PyTwin?
-----------------
PyTwin is part of the `PyAnsys <https://docs.pyansys.com>`_ ecosystem that
lets you use Ansys Digital Twins Runtimes within a Python environment of your choice in
conjunction with other PyAnsys libraries and external Python
libraries.

PyTwin implements different modules to facilitate the consumption of Twin Runtimes. It gives access to
the core Runtime SDK as well as a higher level abstraction class offering Python interfaces that can
be used in any client code for workflow development and Twin Runtimes integrations. You can use PyTwin
to programmatically interact with, and control the execution of Twin Runtimes.

Features
--------
Some of the many features in this primary PyTwin package, ``pytwin``,
allow you to:

- Use the Twin Runtimes Core SDK to build your own consumption workflows based on atomic API calls, see :ref:`ref_index_api_sdk`
- Get access and use the Twin Evaluate module, providing an higher level abstraction implmentation of Twin Runtimes consumption workflows for direct usage and integration, see :ref:`ref_index_api_evaluate`
- Get familiar with the PyTwin functionalities through multiple examples available, see :ref:`ref_example_gallery`

Documentation and issues
------------------------
In addition to installation and usage information, the PyFluent
documentation provides :ref:`ref_index_api`, :ref:`ref_example_gallery`,
and :ref:`ref_contributing` sections.

On the `PyTwin Issues <https://github.com/pyansys/pytwin/issues>`_ page, you can create
issues to submit questions, report bugs, and request new features. To reach
the PyAnsys support team, email `pyansys.support@ansys.com <pyansys.support@ansys.com>`_.

License
-------
PyTwin is licensed under the MIT license.

PyTwin makes no commercial claim over Ansys whatsoever. This library
extends the functionality of Ansys Twin Builder and Twin Deployer by adding
a Python interface
to the Digital Twins Runtimes SDK without changing the core behavior or license of the original
software's. The use of the interactive control of Ansys Digital Twins Runtimes through PyTwin
requires a legally licensed copy of Twin Deployer.

For more information, see the `Ansys Twin Builder and Twin Deployer <https://www.ansys.com/products/digital-twin/ansys-twin-builder>`_.
on the Ansys website.

Project index
-------------

* :ref:`genindex`
