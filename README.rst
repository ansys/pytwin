Pytwin
======
|pyansys| |python| |pypi| |codecov| |GH-CI| |MIT| |black|

.. |pyansys| image:: https://img.shields.io/badge/Py-Ansys-ffc107.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAABDklEQVQ4jWNgoDfg5mD8vE7q/3bpVyskbW0sMRUwofHD7Dh5OBkZGBgW7/3W2tZpa2tLQEOyOzeEsfumlK2tbVpaGj4N6jIs1lpsDAwMJ278sveMY2BgCA0NFRISwqkhyQ1q/Nyd3zg4OBgYGNjZ2ePi4rB5loGBhZnhxTLJ/9ulv26Q4uVk1NXV/f///////69du4Zdg78lx//t0v+3S88rFISInD59GqIH2esIJ8G9O2/XVwhjzpw5EAam1xkkBJn/bJX+v1365hxxuCAfH9+3b9/+////48cPuNehNsS7cDEzMTAwMMzb+Q2u4dOnT2vWrMHu9ZtzxP9vl/69RVpCkBlZ3N7enoDXBwEAAA+YYitOilMVAAAAAElFTkSuQmCC
   :target: https://docs.pyansys.com/
   :alt: PyAnsys

.. |python| image:: https://img.shields.io/badge/Python-%3E%3D3.9-blue
   :target: https://pypi.org/project/pytwin/
   :alt: Python

.. |pypi| image:: https://img.shields.io/pypi/v/pytwin-library.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pytwin/
   :alt: PyPI

.. |codecov| image:: https://codecov.io/gh/pyansys/pytwin/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/pyansys/pytwin/
   :alt: Codecov

.. |GH-CI| image:: https://github.com/pyansys/pytwin/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/pyansys/pytwin/actions/workflows/ci.yml
   :alt: GH-CI

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black
   :alt: Black

Overview
--------
PyTwin is a Python package that eases `Ansys Digital Twins`_ consumption workflows.


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
The ``pytwin`` package supports Python 3.8 through Python
3.10 on Windows and Linux.

Install the latest release from `PyPI
<https://pypi.org/project/pytwin/>`_ with:

.. code:: console

    pip install pytwin

If you plan on doing local *development* of PyTwin with Git, install
the latest release with:

.. code:: console

    git clone https://github.com/pyansys/pytwin.git
    cd pytwin
    pip install pip -U
    pip install -e .

Dependencies
------------
The ``pytwin`` package requires access to an Ansys License Server
with the ``twin_builder_deployer`` feature available (see the
`Getting started`_ section).


License and acknowledgments
---------------------------
PyTwin is licensed under the MIT license.

For more information on `Ansys Digital Twins`_, see the `Twin Builder`_
page on the Ansys website.

.. LINKS AND REFERENCES
.. _Ansys Digital Twins: https://www.ansys.com/products/digital-twin/
.. _Twin Builder: https://www.ansys.com/products/digital-twin/ansys-twin-builder
.. _Documentation: https://twin.docs.pyansys.com/
.. _Getting started: https://twin.docs.pyansys.com/release/0.1/getting_started/index.html
.. _User guide: https://twin.docs.pyansys.com/release/0.1/user_guide/index.html
.. _API reference: https://twin.docs.pyansys.com/release/0.1/api/index.html
.. _Examples: https://twin.docs.pyansys.com/release/0.1/examples/index.html
.. _Contributing: https://twin.docs.pyansys.com/release/0.1/contributing.html
