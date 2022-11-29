"""

@author: Grayson Wen <grayson.wen@ansys.com>
@date: 9/17/2020
"""
from enum import Enum
from collections import defaultdict
from functools import reduce
from typing import Union, Optional, Dict, List, Set, Generic, TypeVar
import platform
from pathlib import Path

if platform.system() == 'Windows':
    os_version = 'win64'
else:
    os_version = 'linux64'

CUR = Path(__file__).parent.resolve()


class Tag(Enum):
    WindowsOnly = 'windows-only'
    LinuxOnly = 'linux-only'


# Type var for the path
T = TypeVar('T')


class PathCollection(Generic[T]):

    def __init__(self):
        super().__init__()
        self._collection_common = set()
        self._collections: Dict[Tag, Set[T]] = defaultdict(set)

    def add_batch(self, v_list: List[T], tag: Tag = None):
        for v in v_list:
            self.add(v, tag)

    def add(self, v: T, tag: Tag = None):
        if tag is None:
            self._collection_common.add(v)
            return

        tag = Tag(tag)
        self._collections[tag].add(v)

    def collect(self, tag: Union[Tag, str]) -> List[T]:
        """
        Return only the matching tag.

        e.g. collect('windows')
        """
        tag = Tag(tag)
        return list(self._collections[tag] | self._collection_common)

    def collect_all(self) -> List[T]:
        if not self._collections.values():
            return []

        return list(
            reduce(lambda a, b: a | b, self._collections.values())
            | self._collection_common
        )


data_files = PathCollection()
binary_files = PathCollection()
dll_files = PathCollection()

root_twin_runtime_sdk = os_version

# ------------------ data_files ------------------
data_files.add(f'*.xml')

# ------------------ binary_files ------------------
binary_files.add(f'licensingclient/winx64/ansyscl.exe', Tag.WindowsOnly)
binary_files.add(f'licensingclient/linx64/ansyscl', Tag.LinuxOnly)

# ------------------ dll_files ------------------
dll_files.add_batch([
        f'{root_twin_runtime_sdk}/*.dll',
        f'messages/*.dll'
    ], Tag.WindowsOnly)

dll_files.add_batch([
        f'{root_twin_runtime_sdk}/*.so',
        f'{root_twin_runtime_sdk}/*.so.*',
        f'messages/*.so',
        f'{root_twin_runtime_sdk}/lib/*.so',
        f'{root_twin_runtime_sdk}/lib/*.so.*'
    ], Tag.LinuxOnly)
