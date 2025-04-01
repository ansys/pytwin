# Copyright (C) 2022 - 2025 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
