"""Mutation API tests."""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

import toml_edit


def _reparses(src: str) -> dict[str, Any]:
    """Sanity check that a rendered document is still valid TOML."""
    return tomllib.loads(src)


# ---------------------------------------------------------------------------
# Scalar set/get/del
# ---------------------------------------------------------------------------


def test_replace_scalar_preserves_surrounding_format() -> None:
    src = "# header comment\nname = 'old'  # inline\nport = 80\n"
    doc = toml_edit.parse(src)
    doc["name"] = "new"
    out = toml_edit.dumps(doc)
    assert out == '# header comment\nname = "new"  # inline\nport = 80\n'
    assert _reparses(out)["name"] == "new"


def test_add_top_level_key_appends() -> None:
    src = "name = 'foo'\n"
    doc = toml_edit.parse(src)
    doc["count"] = 3
    out = toml_edit.dumps(doc)
    assert out == "name = 'foo'\ncount = 3\n"


def test_add_top_level_key_when_only_section_exists() -> None:
    src = "[srv]\nport = 8080\n"
    doc = toml_edit.parse(src)
    doc["name"] = "demo"
    out = toml_edit.dumps(doc)
    # Pre-header section is created at index 0; a blank line separates
    # the new top-level key from the following ``[srv]`` header.
    assert out == 'name = "demo"\n\n[srv]\nport = 8080\n'
    assert _reparses(out) == {"name": "demo", "srv": {"port": 8080}}


def test_add_key_inside_existing_section() -> None:
    src = "[srv]\nport = 80\n"
    doc = toml_edit.parse(src)
    srv = doc["srv"]
    assert isinstance(srv, toml_edit.Table)
    srv["host"] = "127.0.0.1"
    out = toml_edit.dumps(doc)
    assert out == '[srv]\nport = 80\nhost = "127.0.0.1"\n'
    assert _reparses(out) == {"srv": {"port": 80, "host": "127.0.0.1"}}


def test_delete_scalar_removes_line_with_leading_trivia() -> None:
    src = "a = 1\n# this comment belongs to b\nb = 2\nc = 3\n"
    doc = toml_edit.parse(src)
    del doc["b"]
    out = toml_edit.dumps(doc)
    assert out == "a = 1\nc = 3\n"


def test_delete_missing_key_raises_keyerror() -> None:
    doc = toml_edit.parse("a = 1\n")
    with pytest.raises(KeyError):
        del doc["missing"]


def test_set_overwrites_dotted_prefix() -> None:
    src = "[a]\nb.c = 1\n"
    doc = toml_edit.parse(src)
    a = doc["a"]
    assert isinstance(a, toml_edit.Table)
    a["b"] = 2
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"a": {"b": 2}}


def test_set_overwrites_implicit_child_table() -> None:
    src = "[a.b]\nx = 1\n"
    doc = toml_edit.parse(src)
    doc["a"] = 5
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"a": 5}


def test_quoted_key_when_bare_invalid() -> None:
    doc = toml_edit.parse("")
    doc["weird key.com"] = 1
    out = toml_edit.dumps(doc)
    assert '"weird key.com"' in out
    assert _reparses(out) == {"weird key.com": 1}


# ---------------------------------------------------------------------------
# Inline table mutation
# ---------------------------------------------------------------------------


def test_inline_table_replace() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = toml_edit.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, toml_edit.Table)
    obj["a"] = 99
    out = toml_edit.dumps(doc)
    assert out == "obj = { a = 99, b = 2 }\n"


def test_inline_table_append() -> None:
    src = "obj = { a = 1 }\n"
    doc = toml_edit.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, toml_edit.Table)
    obj["b"] = 2
    out = toml_edit.dumps(doc)
    assert "a = 1" in out
    assert "b = 2" in out
    assert _reparses(out) == {"obj": {"a": 1, "b": 2}}


def test_inline_table_delete_last_clears_trailing_comma() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = toml_edit.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, toml_edit.Table)
    del obj["b"]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1}}


# ---------------------------------------------------------------------------
# Array mutation
# ---------------------------------------------------------------------------


def test_array_append() -> None:
    src = "xs = [1, 2, 3]\n"
    doc = toml_edit.parse(src)
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs.append(4)
    assert list(xs) == [1, 2, 3, 4]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_pop() -> None:
    doc = toml_edit.parse("xs = [10, 20, 30]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    v = xs.pop()
    assert v == 30
    assert list(xs) == [10, 20]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [10, 20]}


def test_array_setitem_int() -> None:
    doc = toml_edit.parse("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs[1] = 22
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 3]}


def test_array_setitem_slice() -> None:
    doc = toml_edit.parse("xs = [1, 2, 3, 4]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs[1:3] = [22, 33, 44]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 33, 44, 4]}


def test_array_delitem_slice() -> None:
    doc = toml_edit.parse("xs = [1, 2, 3, 4]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    del xs[1:3]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 4]}


def test_array_clear_and_append() -> None:
    doc = toml_edit.parse("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs.clear()
    xs.append("hi")
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": ["hi"]}


def test_array_extend_iadd() -> None:
    doc = toml_edit.parse("xs = []\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs.extend([1, 2])
    xs += [3, 4]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_sort_reverse() -> None:
    doc = toml_edit.parse("xs = [3, 1, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs.sort()
    assert list(xs) == [1, 2, 3]
    xs.reverse()
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [3, 2, 1]}


def test_array_imul() -> None:
    doc = toml_edit.parse("xs = [1, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs *= 3
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 1, 2, 1, 2]}


def test_array_remove() -> None:
    doc = toml_edit.parse("xs = [1, 2, 3, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs.remove(2)
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 3, 2]}


def test_array_insert() -> None:
    doc = toml_edit.parse("xs = [1, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, toml_edit.Array)
    xs.insert(1, 2)
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3]}


# Every Array/AoT mutator must be wired through the CST so the
# rendered output stays in sync with in-memory mutations.
@pytest.mark.parametrize(
    "name",
    [
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "sort",
        "reverse",
        "__setitem__",
        "__delitem__",
        "__iadd__",
        "__imul__",
    ],
)
def test_every_array_mutator_is_overridden(name: str) -> None:
    array_method = getattr(toml_edit.Array, name, None)
    list_method = getattr(list, name, None)
    assert array_method is not None
    assert list_method is not None
    assert array_method is not list_method, (
        f"Array.{name} must be overridden so mutation routes through CST"
    )


# ---------------------------------------------------------------------------
# Container assignment / deep clone
# ---------------------------------------------------------------------------


def test_assigning_array_deep_clones() -> None:
    src = "src = [1, 2, 3]\n"
    doc = toml_edit.parse(src)
    src_arr = doc["src"]
    assert isinstance(src_arr, toml_edit.Array)
    doc["dst"] = src_arr
    dst = doc["dst"]
    assert isinstance(dst, toml_edit.Array)
    dst.append(99)
    assert list(src_arr) == [1, 2, 3]
    assert list(dst) == [1, 2, 3, 99]
    out = toml_edit.dumps(doc)
    parsed = _reparses(out)
    assert parsed == {"src": [1, 2, 3], "dst": [1, 2, 3, 99]}


def test_assigning_dict_creates_inline_table() -> None:
    doc = toml_edit.parse("")
    doc["obj"] = {"a": 1, "b": "two"}
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1, "b": "two"}}


def test_assigning_list_creates_inline_array() -> None:
    doc = toml_edit.parse("")
    doc["nums"] = [1, 2, 3]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"nums": [1, 2, 3]}


def test_replace_scalar_with_array() -> None:
    doc = toml_edit.parse("x = 1\n")
    doc["x"] = [True, False]
    out = toml_edit.dumps(doc)
    assert _reparses(out) == {"x": [True, False]}
