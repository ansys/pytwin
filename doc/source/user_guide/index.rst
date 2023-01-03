.. _ref_user_guide:

==========
User guide
==========
Anyone who wants to use PyTwin can import its Python modules and develop
Python code to integrate and deploy Ansys Digital Twins Runtimes, as explained in :ref:`ref_index_api` and demonstrated in :ref:`ref_example_gallery`

Global settings
---------------
By default, the logging is enabled with PyTwin at a level of ``INFO``, and simulation output files will be generated in the ``%temp%/pytwin`` folder.
You can change these global settings at anytime using the following functions:

.. code-block:: python

    # Modify working directory:
    from pytwin import modify_pytwin_working_dir

    modify_pytwin_working_dir("path_to_new_working_dir", erase=False)

    # Redirect logging to a file in the working directory:
    from pytwin import modify_pytwin_logging, get_pytwin_log_file
    from pytwin import PYTWIN_LOGGING_OPT_FILE, PYTWIN_LOG_DEBUG

    modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_FILE, new_level=PYTWIN_LOG_DEBUG)
    print(get_pytwin_log_file())

    # Redirect pytwin package logging to the console:
    from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_CONSOLE

    modify_pytwin_logging(PYTWIN_LOGGING_OPT_CONSOLE)

    # Disable pytwin package logging:
    from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_NOLOGGING

    modify_pytwin_logging(PYTWIN_LOGGING_OPT_NOLOGGING)

See :ref:`ref_index_api_logging` for more information on the APIs available