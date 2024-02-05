import logging
import os
import shutil
import tempfile

from pytwin import (
    PyTwinLogLevel,
    PyTwinLogOption,
    PyTwinSettingsError,
    get_pytwin_log_file,
    get_pytwin_logger,
    get_pytwin_working_dir,
    modify_pytwin_logging,
    modify_pytwin_working_dir,
    pytwin_logging_is_enabled,
)

UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings():
    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


class TestDefaultSettings:
    #def test_default_setting(self):
    #    # Working directory is created in temp folder
    #    wd = get_pytwin_working_dir()
    #    assert tempfile.gettempdir() in wd
    #    assert os.path.exists(wd)
    #    # Logging is redirected to a file with INFO level
    #    log_file = get_pytwin_log_file()
    #    logger = get_pytwin_logger()
    #    logger.debug("Hello 10")
    #    logger.info("Hello 20")
    #    logger.warning("Hello 30")
    #    logger.error("Hello 40")
    #    logger.critical("Hello 50")
    #    with open(log_file, "r") as f:
    #        lines = f.readlines()
    #    assert "Hello 10" not in lines
    #    assert len(lines) == 5
    #    assert os.path.exists(log_file)
    #    assert len(logger.handlers) == 1
    #    assert pytwin_logging_is_enabled()
    #
    def test_modify_logging_raises_error(self):
        # Init unit test
        reinit_settings()
        assert pytwin_logging_is_enabled()

        # Raises error if new_option is not a valid PyTwinLogOption attribute.
        try:
            modify_pytwin_logging(new_option="unknown")
        except PyTwinSettingsError as e:
            assert "Error occurred while setting PyTwin logging option." in str(e)

        # Raises error if new_option is not a valid PyTwinLogLevel attribute.
        try:
            modify_pytwin_logging(new_level="unknown")
        except PyTwinSettingsError as e:
            assert "Error occurred while setting PyTwin logging level." in str(e)

    def test_modify_logging_no_logging(self):
        from pytwin import TwinModel
        from pytwin.twin_runtime.log_level import LogLevel

        # Init unit test
        reinit_settings()
        assert pytwin_logging_is_enabled()
        # Disable logging
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING)
        level = TwinModel._get_runtime_log_level()
        logger = get_pytwin_logger()
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 0
        assert log_file is None
        assert not pytwin_logging_is_enabled()
        assert level == LogLevel.TWIN_NO_LOG

    def test_modify_logging_console(self):
        # Init unit test
        reinit_settings()
        assert pytwin_logging_is_enabled()
        # Redirect logging to console
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE)
        logger = get_pytwin_logger()
        logger.debug("Hello 10")  # Must verify interactively that it is not printed to the console
        logger.info("Hello 20")
        logger.warning("Hello 30")
        logger.error("Hello 40")
        logger.critical("Hello 50")
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 1
        assert log_file is None
        assert pytwin_logging_is_enabled()

    def test_modify_logging_level(self):
        from pytwin import TwinModel
        from pytwin.settings import get_pytwin_log_level
        from pytwin.twin_runtime.log_level import LogLevel

        # Init unit test
        reinit_settings()
        # Modify logging level works
        modify_pytwin_logging(new_level=PyTwinLogLevel.PYTWIN_LOG_CRITICAL)
        level = get_pytwin_log_level()
        runtime_level = TwinModel._get_runtime_log_level()
        log_file = get_pytwin_log_file()
        logger = get_pytwin_logger()
        logger.debug("Hello 10")
        logger.info("Hello 20")
        logger.warning("Hello 30")
        logger.error("Hello 40")
        logger.critical("Hello 50")
        with open(log_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert level == PyTwinLogLevel.PYTWIN_LOG_CRITICAL
        assert runtime_level == LogLevel.TWIN_LOG_FATAL
        # Modify logging level can be done dynamically
        modify_pytwin_logging(new_level=PyTwinLogLevel.PYTWIN_LOG_DEBUG)
        level = get_pytwin_log_level()
        runtime_level = TwinModel._get_runtime_log_level()
        logger.debug("Hello 10")
        logger.info("Hello 20")
        logger.warning("Hello 30")
        logger.error("Hello 40")
        logger.critical("Hello 50")
        with open(log_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 5 + 1
        assert level == PyTwinLogLevel.PYTWIN_LOG_DEBUG
        assert runtime_level == LogLevel.TWIN_LOG_ALL

    def test_modify_logging_multiple_times(self):
        # Init unit test
        reinit_settings()
        # Start by disabling the logging
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING)
        logger = get_pytwin_logger()
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 0
        assert log_file is None
        # Then change to console logging
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE)
        logger = get_pytwin_logger()
        logger.info("Hello console 20")
        logger.warning("Hello console 30")
        logger.error("Hello console 40")
        logger.critical("Hello console 50")
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 1
        assert log_file is None
        # Then change to file logging
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE)
        logger = get_pytwin_logger()
        assert len(logger.handlers) == 1
        logger.info("Hello file 20")
        logger.warning("Hello file 30")
        logger.error("Hello file 40")
        logger.critical("Hello file 50")
        log_file = get_pytwin_log_file()
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 4
        assert "console" not in lines

    def test_modify_working_dir_raises_error(self):
        # Init unit test
        reinit_settings()

        # Raises error if None is provided as working dir
        try:
            modify_pytwin_working_dir(new_path=None)
        except PyTwinSettingsError as e:
            assert "Error occurred while setting the PyTwin working directory." in str(e)

        # Raises error if provided path does not exist and parent directory does not exists
        try:
            modify_pytwin_working_dir(new_path=os.path.join(os.path.dirname(__file__), "unknown_folder", "wd"))
        except PyTwinSettingsError as e:
            assert "Provide a folder path in which all parents exist." in str(e)

        # Raises error if erase argument is not boolean
        try:
            modify_pytwin_working_dir(new_path=UNIT_TEST_WD, erase="wrong_type")
        except PyTwinSettingsError as e:
            assert "'erase' argument must be Boolean" in str(e)

    def test_modify_working_dir_with_not_existing(self):
        # Init unit test
        reinit_settings()
        assert not os.path.exists(UNIT_TEST_WD)
        # Not existing dir is created
        modify_pytwin_working_dir(new_path=UNIT_TEST_WD)
        assert os.path.exists(UNIT_TEST_WD)
        assert UNIT_TEST_WD == get_pytwin_working_dir()

    #def test_modify_working_dir_with_existing_erase_false(self):
    #    # Init unit test
    #    reinit_settings()
    #    os.mkdir(UNIT_TEST_WD)
    #    with open(os.path.join(UNIT_TEST_WD, "test.txt"), "w") as f:
    #        pass
    #    assert len(os.listdir(UNIT_TEST_WD)) == 1
    #    assert "test.txt" in os.listdir(UNIT_TEST_WD)
    #    # Existing dir with erase = false
    #    modify_pytwin_working_dir(new_path=UNIT_TEST_WD, erase=False)
    #    temp = os.listdir(UNIT_TEST_WD)
    #    assert len(os.listdir(UNIT_TEST_WD)) == 2  # test.txt + log file
    #    assert "test.txt" in os.listdir(UNIT_TEST_WD)
    #    assert os.path.split(get_pytwin_log_file())[-1] in os.listdir(UNIT_TEST_WD)
    #
    #def test_modify_working_dir_with_existing_erase_true(self):
    #    # Init unit test
    #    reinit_settings()
    #    os.mkdir(UNIT_TEST_WD)
    #    with open(os.path.join(UNIT_TEST_WD, "test.txt"), "w") as f:
    #        pass
    #    assert len(os.listdir(UNIT_TEST_WD)) == 1
    #    assert "test.txt" in os.listdir(UNIT_TEST_WD)
    #    # Existing dir with erase = false
    #    modify_pytwin_working_dir(new_path=UNIT_TEST_WD, erase=True)
    #    temp = os.listdir(UNIT_TEST_WD)
    #    assert len(os.listdir(UNIT_TEST_WD)) == 1  # log file
    #    assert "test.txt" not in os.listdir(UNIT_TEST_WD)
    #    assert os.path.split(get_pytwin_log_file())[-1] in os.listdir(UNIT_TEST_WD)
    #
    def test_modify_working_dir_migration(self):
        # Init unit test
        reinit_settings()
        os.mkdir(UNIT_TEST_WD)
        logger = get_pytwin_logger()
        msg_temp = "Hello from temp dir!"
        logger.info(msg_temp)
        log_file_in_temp = get_pytwin_log_file()
        with open(log_file_in_temp, "r") as f:
            lines_temp = f.readlines()
        assert len(lines_temp) == 1
        assert msg_temp in lines_temp[0]
        # Verifies log file is well migrated to new path
        modify_pytwin_working_dir(new_path=UNIT_TEST_WD)
        log_file_new = get_pytwin_log_file()
        assert log_file_new != log_file_in_temp
        msg_new = "Hello from new dir!"
        logger.info(msg_new)
        with open(log_file_new, "r") as f:
            lines_new = f.readlines()
        assert len(lines_new) == 2
        assert msg_temp in lines_new[0]
        assert msg_new in lines_new[1]

    def test_modify_logging_after_working_dir(self):
        # Init unit test
        reinit_settings()
        modify_pytwin_working_dir(new_path=UNIT_TEST_WD)
        # Disabling the logging
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING)
        logger = get_pytwin_logger()
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 0
        assert log_file is None
        # Then change to console logging
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE)
        logger = get_pytwin_logger()
        logger.info("Hello console 20")
        logger.warning("Hello console 30")
        logger.error("Hello console 40")
        logger.critical("Hello console 50")
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 1
        assert log_file is None
        # Then change to file logging
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_FILE)
        logger = get_pytwin_logger()
        assert len(logger.handlers) == 1
        logger.info("Hello file 20")
        logger.warning("Hello file 30")
        logger.error("Hello file 40")
        logger.critical("Hello file 50")
        log_file = get_pytwin_log_file()
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 4
        assert "console" not in lines

    def test_modify_working_after_logging_no_logging(self):
        # Init unit test
        reinit_settings()
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_NOLOGGING)
        logger = get_pytwin_logger()
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 0
        assert log_file is None
        # Modify working directory does not modify logging
        modify_pytwin_working_dir(new_path=UNIT_TEST_WD)
        logger = get_pytwin_logger()
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 0
        assert log_file is None

    def test_modify_working_after_logging_console(self):
        # Init unit test
        reinit_settings()
        modify_pytwin_logging(new_option=PyTwinLogOption.PYTWIN_LOGGING_OPT_CONSOLE)
        logger = get_pytwin_logger()
        msg = "Hello from console 1"
        logger.info(msg)
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 1
        assert log_file is None
        # Modify working directory does not modify logging
        modify_pytwin_working_dir(new_path=UNIT_TEST_WD)
        logger = get_pytwin_logger()
        msg = "Hello from console 2"
        logger.info(msg)
        log_file = get_pytwin_log_file()
        assert len(logger.handlers) == 1
        assert log_file is None

    def test_multiprocess_execution_pytwin_cleanup_is_safe(self):
        import subprocess
        import sys

        # Init unit test
        reinit_settings()

        # Verify that each new python process delete its own temp working dir without deleting others
        current_wd_dir_count = len(os.listdir(os.path.dirname(get_pytwin_working_dir())))
        result = subprocess.run([sys.executable, "-c", "import pytwin"], capture_output=True)
        new_wd_dir_count = len(os.listdir(os.path.dirname(get_pytwin_working_dir())))

        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        assert new_wd_dir_count == current_wd_dir_count

    def test_multiprocess_execution_keep_new_directory(self):
        import subprocess
        import sys

        # Init unit test
        wd = reinit_settings()

        # Verify that each new python process delete its own temp working dir without deleting others
        current_wd_dir_count = len(os.listdir(os.path.dirname(get_pytwin_working_dir())))
        main_logger_handler_count = len(logging.getLogger().handlers)
        pytwin_logger_handler_count = len(get_pytwin_logger().handlers)
        code = "import pytwin\n"
        code += f'pytwin.modify_pytwin_working_dir(new_path=r"{wd}")'
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        new_wd_dir_count = len(os.listdir(os.path.dirname(get_pytwin_working_dir())))

        assert len(result.stdout) == 0
        assert len(result.stderr) == 0
        assert new_wd_dir_count == current_wd_dir_count
        assert os.path.exists(wd)
        assert main_logger_handler_count == len(logging.getLogger().handlers)
        assert pytwin_logger_handler_count == len(get_pytwin_logger().handlers)

    def test_clean_unit_test(self):
        reinit_settings()
        temp_wd = get_pytwin_working_dir()
        parent_dir = os.path.dirname(temp_wd)
        try:
            for dir_name in os.listdir(parent_dir):
                if dir_name not in temp_wd:
                    shutil.rmtree(os.path.join(parent_dir, dir_name))
        except Exception as e:
            pass
