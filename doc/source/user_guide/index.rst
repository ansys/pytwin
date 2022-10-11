.. _ref_user_guide:

==========
User guide
==========
Anyone who wants to use PyTwin can import its Python modules and develop
Python code to integrate and deploy Ansys Digital Twins Runtimes, as explained in :ref:`ref_index_api` and demonstrated in :ref:`ref_example_gallery`

Global logging
--------------
You can control the global logging level at any time with:

.. code-block:: python

    >>> # Append pytwin package logging to an existing application log file:
    >>> from pytwin import set_pytwin_logging
    >>> from pytwin import PYTWIN_LOG_WARNING
    >>> set_pytwin_logging(log_filepath='filepath_to_my_app.log', mode='a', level=PYTWIN_LOG_WARNING)

    >>> # Redirect pytwin package logging to the console:
    >>> from pytwin import set_pytwin_logging
    >>> set_pytwin_logging()

    >>> # Disable pytwin package logging:
    >>> from pytwin import set_pytwin_logging
    >>> from pytwin import PYTWIN_NO_LOG
    >>> set_pytwin_logging(level=PYTWIN_NO_LOG)
