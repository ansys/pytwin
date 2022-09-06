"""
twin.

library
"""
# TODO : check with Maxime/PyAnsys how to properly use that code from the template
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version('ansys-twin')
