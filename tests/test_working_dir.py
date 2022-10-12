import shutil
import os
from pytwin import set_pytwin_working_dir
from pytwin import get_pytwin_working_dir


def reinit_unit_test():
    from pytwin.settings import _reinit_settings
    _reinit_settings()
    wd = os.path.join(os.path.dirname(__file__), 'unit_test_wd')
    if os.path.exists(wd):
        shutil.rmtree(wd)
    return wd


class TestPyTwinWorkingDir:
    def test_set_wd_creates_dir_if_does_not_exist(self):
        # Init unit test
        wd = reinit_unit_test()
        assert not os.path.exists(wd)
        # Test working directory is created if it does not exist (whatever erase argument)
        set_pytwin_working_dir(wd, False)
        assert os.path.exists(wd)
        shutil.rmtree(wd)
        set_pytwin_working_dir(wd, True)

    def test_set_wd_erase_false(self):
        # Init unit test
        wd = reinit_unit_test()
        assert not os.path.exists(wd)
        # Test erase argument in case working directory exist
        os.mkdir(wd)
        temp_file = os.path.join(wd, 'temp.txt')
        with open(temp_file, 'w') as f:
            pass
        assert len(os.listdir(wd)) == 1
        # erase = False makes we keep the existing dir
        set_pytwin_working_dir(wd, False)
        assert get_pytwin_working_dir() == wd
        assert len(os.listdir(wd)) == 1

    def test_set_wd_erase_true(self):
        # Init unit test
        wd = reinit_unit_test()
        assert not os.path.exists(wd)
        # Test erase argument in case working directory exist
        os.mkdir(wd)
        temp_file = os.path.join(wd, 'temp.txt')
        with open(temp_file, 'w') as f:
            pass
        assert len(os.listdir(wd)) == 1
        # erase = True makes we clean the existing dir
        set_pytwin_working_dir(wd, True)
        assert get_pytwin_working_dir() == wd
        assert len(os.listdir(wd)) == 0

    def test_set_wd_has_an_effect_only_at_first_call(self):
        # Init unit test
        wd = reinit_unit_test()
        assert not os.path.exists(wd)
        # First call creates the working dir
        set_pytwin_working_dir(wd, False)
        assert os.path.exists(wd)
        # Second call does not create a new one
        wd2 = wd + '2'
        set_pytwin_working_dir(wd, False)
        assert not os.path.exists(wd2)

    def test_get_lazy_init_temporary_dir(self):
        # First call creates it if it does not exist
        wd = get_pytwin_working_dir()
        assert os.path.exists(wd)
        temp_file = os.path.join(wd, 'temp.txt')
        with open(temp_file, 'w') as f:
            pass
        assert len(os.listdir(wd)) == 1
        # Second call does nothing
        assert get_pytwin_working_dir() == wd
        assert os.path.exists(wd)
        assert len(os.listdir(wd)) == 1
        # New session will clean existing temporary dir
        reinit_unit_test()
        wd = get_pytwin_working_dir()
        assert os.path.exists(wd)
        assert len(os.listdir(wd)) == 0

    def test_clean_unit_test(self):
        reinit_unit_test()
