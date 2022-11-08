"""

@author: Adriano Maron <adriano.maron@ansys.com>
@date: 01/06/2022
"""
import io
import os
import re
from glob import glob
from pathlib import Path
from typing import Tuple, List
import platform
from PyInstaller.utils.hooks import collect_submodules

CUR_DIR = Path(__file__).parent
MODULE_ROOT = (CUR_DIR / '..').resolve().absolute()
_WINDOWS = platform.system() == 'Windows'


def get_build(file=str(MODULE_ROOT / '__build__.py')):
    path = os.path.realpath(file)
    _ns = globals()  # {}
    with io.open(path, encoding="utf8") as f:
        s = f.read()
        # c = compile(f.read(), '<string>', 'exec')
        exec(s, globals(), _ns)
    return _ns


def glob_file(pattern):
    module_folder = str(MODULE_ROOT)
    li = glob(f'{module_folder}/{pattern}', recursive=True)

    # Drop dir paths
    li = filter(lambda _: Path(_).is_file(), li)
    li = list(li)

    # Get dir
    li_dir = map(lambda i: Path(i).parent, li)
    # Drop prepending module folder
    #   e.g. module_folder/path/file -> path/file
    #   not str.lstrip()
    li_dir = map(lambda i: str(i)[len(module_folder)+1:], li_dir)
    # Convert '' to '.'
    li_dir = map(lambda i: '.' if i == '' else i, li_dir)

    # li = map(lambda i: i.replace('\\', '/'), li)

    return list(zip(li, li_dir))


# Map the some prefix to ./ e.g. "client/" -> "./"
def _map_to_root(li: List[Tuple[str, str]], prefix):
    for i, (from_, to_) in enumerate(li):
        li[i] = (from_, re.sub(rf'^{prefix}[\\/]?', './', to_))
    return list(li)


__build__ = get_build()
platform_tag = 'windows-only' if _WINDOWS else 'linux-only'

# ========================================================================================
# ================ PyInstaller global variables (the purpose of the hook) ================
# ========================================================================================


# Example: datas = [ ('/usr/share/icons/education_*.png', 'icons') ]
datas = []
for _pattern in __build__['data_files'].collect(platform_tag):
    datas += glob_file(_pattern)

binaries = []
for _pattern in __build__['dll_files'].collect(platform_tag):
    datas += glob_file(_pattern)

for _pattern in __build__['binary_files'].collect(platform_tag):
    datas += glob_file(_pattern)
