#------------------------------------------------------------------------------
# (c) 2020-2024 ANSYS, Inc. All rights reserved.
#------------------------------------------------------------------------------
import sys
from pathlib import Path

__version__ = "1.15.7"
__anssversion__ = "2025.2"


PACKAGE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.resolve()))
