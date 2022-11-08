"""

@author: Grayson Wen <grayson.wen@ansys.com>
@date: 8/20/2020
"""
import os


def get_hook_dirs():
    """
    Tell PyInstaller where to find hooks provided by this distribution;
    this is referenced by the :ref:`hook registration <hook_registration>`.
    This function returns a list containing only the path to this
    directory, which is the location of these hooks.
    """
    return [os.path.dirname(__file__)]


def get_PyInstaller_tests():
    """
    Tell PyInstaller where to find tests of the hooks provided by this
    distribution; this is referenced by the :ref:`tests registration
    <tests_registration>`. This function returns a list containing only
    the path to this directory, which is the location of these tests.
    """
    return [os.path.dirname(__file__)]

