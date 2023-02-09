.. _ref_user_guide:

==========
User guide
==========

Anyone who wants to use PyTwin can import its Python modules and develop
Python code to integrate and deploy Ansys Digital Twin Runtimes, as
explained in :ref:`ref_index_api` and demonstrated in :ref:`ref_example_gallery`.

Global settings
---------------

By default, logging is enabled in PyTwin at a level of ``INFO``. Simulation output
files are generated in the ``%temp%/pytwin`` folder. You can change these global
settings at anytime using these functions:

.. code-block:: python

    # Modify working directory
    from pytwin import modify_pytwin_working_dir

    modify_pytwin_working_dir("path_to_new_working_dir", erase=False)

    # Redirect logging to a file in the working directory
    from pytwin import modify_pytwin_logging, get_pytwin_log_file
    from pytwin import PYTWIN_LOGGING_OPT_FILE, PYTWIN_LOG_DEBUG

    modify_pytwin_logging(new_option=PYTWIN_LOGGING_OPT_FILE, new_level=PYTWIN_LOG_DEBUG)
    print(get_pytwin_log_file())

    # Redirect PyTwin logging to the console
    from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_CONSOLE

    modify_pytwin_logging(PYTWIN_LOGGING_OPT_CONSOLE)

    # Disable PyTwin logging:
    from pytwin import modify_pytwin_logging, PYTWIN_LOGGING_OPT_NOLOGGING

    modify_pytwin_logging(PYTWIN_LOGGING_OPT_NOLOGGING)


For information on all APIs, see :ref:`ref_index_api_logging`.