"""Load native prerequisites before importing third-party packages."""

import ctypes
import os
import platform
import sys

_libstdcpp = None

if platform.system() == "Linux":
    _runtime_dir = getattr(sys, "_MEIPASS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_runtime"))
    _libstdcpp_path = os.path.join(str(_runtime_dir), "libstdc++.so.6")
    try:
        _libstdcpp = ctypes.CDLL(_libstdcpp_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise ImportError(
            f"Failed to preload PyTwin's bundled C++ runtime at {_libstdcpp_path}: {error}. "
            "Ensure the bundled library and its system dependencies are available. "
            "Import PyTwin before other native packages. If an incompatible libstdc++ is already loaded, "
            "restart Python with a compatible C++ runtime or preload the bundled library using LD_PRELOAD."
        ) from error