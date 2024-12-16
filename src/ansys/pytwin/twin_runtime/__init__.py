#------------------------------------------------------------------------------
# (c) 2020-2024 ANSYS, Inc. All rights reserved.
#------------------------------------------------------------------------------
import sys
from pathlib import Path

__version__ = "1.12.3"
__anssversion__ = "2025.1"


PACKAGE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.resolve()))
