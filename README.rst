PyTwin
======
|pyansys| |python| |pypi| |codecov| |GH-CI| |MIT| |black| |pre-commit|

.. |pyansys| image:: https://img.shields.io/badge/Py-Ansys-ffc107.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAABDklEQVQ4jWNgoDfg5mD8vE7q/3bpVyskbW0sMRUwofHD7Dh5OBkZGBgW7/3W2tZpa2tLQEOyOzeEsfumlK2tbVpaGj4N6jIs1lpsDAwMJ278sveMY2BgCA0NFRISwqkhyQ1q/Nyd3zg4OBgYGNjZ2ePi4rB5loGBhZnhxTLJ/9ulv26Q4uVk1NXV/f///////69du4Zdg78lx//t0v+3S88rFISInD59GqIH2esIJ8G9O2/XVwhjzpw5EAam1xkkBJn/bJX+v1365hxxuCAfH9+3b9/+////48cPuNehNsS7cDEzMTAwMMzb+Q2u4dOnT2vWrMHu9ZtzxP9vl/69RVpCkBlZ3N7enoDXBwEAAA+YYitOilMVAAAAAElFTkSuQmCC
   :target: https://docs.pyansys.com/
   :alt: PyAnsys

.. |python| image:: https://img.shields.io/pypi/pyversions/pytwin?logo=pypi
   :target: https://pypi.org/project/pytwin/
   :alt: Python

.. |pypi| image:: https://img.shields.io/pypi/v/pytwin.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pytwin/
   :alt: PyPI

.. |codecov| image:: https://codecov.io/gh/ansys/pytwin/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/ansys/pytwin/
   :alt: Codecov

.. |GH-CI| image:: https://github.com/ansys/pytwin/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/ansys/pytwin/actions/workflows/ci.yml
   :alt: GH-CI

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black
   :alt: Black

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/ansys/pytwin/main.svg
   :target: https://results.pre-commit.ci/latest/github/ansys/pytwin/main
   :alt: pre-commit.ci status

Overview
--------
PyTwin is a Python package for consuming a digital twin model that has been exported from
Ansys Twin Builder as a TWIN file. Each TWIN file is a self-contained app (twin runtime)
that consists of two parts:

- The twin model, which is the simulation model created and compiled in Twin Builder
- The runtime SDK, which is the shared library for consuming the twin model within
  the deployment workflow.

Because PyTwin wraps a twin runtime in a Python interface, you do not need a Twin Builder
installation to deploy it.

For more information on digital twins and Ansys Twin Builder, see
`Digital Twins`_ and `Ansys Twin Builder`_ on the Ansys website.

Documentation
-------------
For comprehensive information on PyTwin, see the latest release `Documentation`_
and its sections:

* `Getting started`_
* `User guide`_
* `API reference`_
* `Examples`_
* `Contributing`_

Installation
------------
The ``pytwin`` package supports Python 3.9 through Python 3.12 on Windows and Linux.

Install the latest release from `PyPI <https://pypi.org/project/pytwin/>`_ with this
command:

.. code:: console

    pip install pytwin


If you plan on doing local *development* of PyTwin with Git, install
the latest release with this code:

.. code:: console

    git clone https://github.com/ansys/pytwin.git
    cd pytwin
    pip install pip -U
    pip install -e .


Dependencies
------------
The ``pytwin`` package requires access to an Ansys License Server
with the ``twin_builder_deployer`` feature available. For more information,
see `Getting started`_.


License and acknowledgments
---------------------------
PyTwin is licensed under the MIT license.


.. LINKS AND REFERENCES
.. _Digital Twins: https://www.ansys.com/products/digital-twin/
.. _Ansys Twin Builder: https://www.ansys.com/products/digital-twin/ansys-twin-builder
.. _Documentation: https://twin.docs.pyansys.com/
.. _Getting started: https://twin.docs.pyansys.com/version/stable/getting_started/index.html
.. _User guide: https://twin.docs.pyansys.com/version/stable/user_guide/index.html
.. _API reference: https://twin.docs.pyansys.com/version/stable/api/index.html
.. _Examples: https://twin.docs.pyansys.com/version/stable/examples/index.html
.. _Contributing: https://twin.docs.pyansys.com/version/stable/contributing.html
