"""Module containing a set of decorators."""

ERROR_GRAPHICS_REQUIRED = (
    "Graphics are required for this method. Please install the ``graphics`` target "
    "to use this method. You can install it by running `pip install pytwin[graphics]`."
)
"""Message to display when graphics are required for a method."""

__GRAPHICS_AVAILABLE = None
"""Global variable to store the result of the graphics imports."""


def run_if_graphics_required():
    """Check if graphics are available."""
    global __GRAPHICS_AVAILABLE
    if __GRAPHICS_AVAILABLE is None:
        try:
            # Attempt to perform the imports
            import pyvista  # noqa: F401

            __GRAPHICS_AVAILABLE = True
        except (ModuleNotFoundError, ImportError):
            __GRAPHICS_AVAILABLE = False

    if __GRAPHICS_AVAILABLE is False:
        raise ImportError(ERROR_GRAPHICS_REQUIRED)


def needs_graphics(method):
    """Decorate a method as requiring graphics.

    Parameters
    ----------
    method : callable
        Method to decorate.

    Returns
    -------
    callable
        Decorated method.
    """
    def wrapper(*args, **kwargs):
        run_if_graphics_required()
        return method(*args, **kwargs)
    return wrapper
