from __future__ import annotations
import sys
import pytest
from run_section21 import execute_author_source


def test_original_annotations_are_not_changed_by_launcher_future():
    execute_author_source(b"def f(x: int): pass\nassert f.__annotations__['x'] is int\n",
                          "original.py")


def test_module_annotations_resolve_like_a_normal_script():
    execute_author_source(b"""from __future__ import annotations
from dataclasses import dataclass
from typing import get_type_hints
class LocalType: pass
@dataclass
class Record:
    value: LocalType
assert get_type_hints(Record)['value'] is LocalType
assert __file__ == 'original.py'
""", "original.py")


def test_main_module_restored_after_exception():
    previous = sys.modules["__main__"]
    with pytest.raises(RuntimeError, match="author failure"):
        execute_author_source(b"raise RuntimeError('author failure')", "original.py")
    assert sys.modules["__main__"] is previous
