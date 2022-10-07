"""
pytwin.

library
"""
from pytwin._version import __version__  # noqa: F401

_VERSION_INFO = None
"""Global variable indicating the version of the PyFluent package - Empty by default"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

def version_info() -> str:
    """Method returning the version of PyFluent being used.

    Returns
    -------
    str
        The PyFluent version being used.

    Notes
    -------
    Only available in packaged versions. Otherwise it will return __version__.
    """
    return _VERSION_INFO if _VERSION_INFO is not None else __version__
