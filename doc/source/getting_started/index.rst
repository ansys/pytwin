.. _getting_started:

===============
Getting started
===============
PyTwin provides Pythonic access to Ansys Digital Twins Runtimes.
These Runtimes are generated using `Ansys Twin Builder and Twin Deployer <https://www.ansys.com/products/digital-twin/ansys-twin-builder>`_.

To run PyTwin, you must have a license of Ansys Twin Deployer
installed locally. PyTwin supports Runtimes generated using 2023 R1 and later.



Install the package
-------------------
The ``pytwin`` package supports Python 3.7 through
Python 3.10 on Windows and Linux.

Install the latest release from `PyPi
<https://pypi.org/project/pytwin/>`_ with:

.. code::

   pip install pytwin

If you plan on doing local *development* of PyTwin with Git, install
the latest release with:

.. code::

   git clone https://github.com/pyansys/pytwin.git
   cd pytwin
   pip install pip -U
   pip install -e .
   python codegen/allapigen.py  # Generates the API files


Any changes that you make locally are reflected in your setup after you restart
the Python kernel.

Additional PyAnsys libraries
-----------------------------
You can also install and use these additional PyAnsys libraries:

- `PyAEDT <https://aedt.docs.pyansys.com//>`_, which provides
  access to Ansys Twin Builder for models creation and Digital Twins Runtimes generation.

