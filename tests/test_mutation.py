"""Mutation API tests."""

from __future__ import annotations

import tomllib

import pytest

import tomle


def _reparses(src: str) -> dict[str, object]:
    """Sanity check that a rendered document is still valid TOML."""
    return tomllib.loads(src)


# ---------------------------------------------------------------------------
# Scalar set/get/del
# ---------------------------------------------------------------------------


def test_replace_scalar_preserves_surrounding_format() -> None:
    src = "# header comment\nname = 'old'  # inline\nport = 80\n"
    doc = tomle.parse(src)
    doc["name"] = "new"
    out = tomle.dumps(doc)
    assert out == '# header comment\nname = "new"  # inline\nport = 80\n'
    assert _reparses(out)["name"] == "new"


def test_add_top_level_key_appends() -> None:
    src = "name = 'foo'\n"
    doc = tomle.parse(src)
    doc["count"] = 3
    out = tomle.dumps(doc)
    assert out == "name = 'foo'\ncount = 3\n"


def test_add_top_level_key_when_only_section_exists() -> None:
    src = "[srv]\nport = 8080\n"
    doc = tomle.parse(src)
    doc["name"] = "demo"
    out = tomle.dumps(doc)
    # Pre-header section is created at index 0
    assert out == 'name = "demo"\n[srv]\nport = 8080\n'
    assert _reparses(out) == {"name": "demo", "srv": {"port": 8080}}


def test_add_key_inside_existing_section() -> None:
    src = "[srv]\nport = 80\n"
    doc = tomle.parse(src)
    srv = doc["srv"]
    assert isinstance(srv, tomle.Table)
    srv["host"] = "127.0.0.1"
    out = tomle.dumps(doc)
    assert out == '[srv]\nport = 80\nhost = "127.0.0.1"\n'
    assert _reparses(out) == {"srv": {"port": 80, "host": "127.0.0.1"}}


def test_delete_scalar_removes_line_with_leading_trivia() -> None:
    src = "a = 1\n# this comment belongs to b\nb = 2\nc = 3\n"
    doc = tomle.parse(src)
    del doc["b"]
    out = tomle.dumps(doc)
    assert out == "a = 1\nc = 3\n"


def test_delete_missing_key_raises_keyerror() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(KeyError):
        del doc["missing"]


def test_set_dotted_prefix_conflict_raises() -> None:
    src = "[a]\nb.c = 1\n"
    doc = tomle.parse(src)
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    with pytest.raises(tomle.TOMLEditError):
        a["b"] = 2


def test_set_existing_child_table_conflict_raises() -> None:
    src = "[a.b]\nx = 1\n"
    doc = tomle.parse(src)
    with pytest.raises(tomle.TOMLEditError):
        doc["a"] = 5  # 'a' is implicit parent of [a.b]


def test_quoted_key_when_bare_invalid() -> None:
    doc = tomle.parse("")
    doc["weird key.com"] = 1
    out = tomle.dumps(doc)
    assert '"weird key.com"' in out
    assert _reparses(out) == {"weird key.com": 1}


# ---------------------------------------------------------------------------
# Inline table mutation
# ---------------------------------------------------------------------------


def test_inline_table_replace() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = tomle.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, tomle.Table)
    obj["a"] = 99
    out = tomle.dumps(doc)
    assert out == "obj = { a = 99, b = 2 }\n"


def test_inline_table_append() -> None:
    src = "obj = { a = 1 }\n"
    doc = tomle.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, tomle.Table)
    obj["b"] = 2
    out = tomle.dumps(doc)
    assert "a = 1" in out
    assert "b = 2" in out
    assert _reparses(out) == {"obj": {"a": 1, "b": 2}}


def test_inline_table_delete_last_clears_trailing_comma() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = tomle.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, tomle.Table)
    del obj["b"]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1}}


# ---------------------------------------------------------------------------
# Array mutation
# ---------------------------------------------------------------------------


def test_array_append() -> None:
    src = "xs = [1, 2, 3]\n"
    doc = tomle.parse(src)
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs.append(4)
    assert list(xs) == [1, 2, 3, 4]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_pop() -> None:
    doc = tomle.parse("xs = [10, 20, 30]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    v = xs.pop()
    assert v == 30
    assert list(xs) == [10, 20]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [10, 20]}


def test_array_setitem_int() -> None:
    doc = tomle.parse("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs[1] = 22
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 3]}


def test_array_setitem_slice() -> None:
    doc = tomle.parse("xs = [1, 2, 3, 4]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs[1:3] = [22, 33, 44]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 33, 44, 4]}


def test_array_delitem_slice() -> None:
    doc = tomle.parse("xs = [1, 2, 3, 4]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    del xs[1:3]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [1, 4]}


def test_array_clear_and_append() -> None:
    doc = tomle.parse("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs.clear()
    xs.append("hi")
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": ["hi"]}


def test_array_extend_iadd() -> None:
    doc = tomle.parse("xs = []\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs.extend([1, 2])
    xs += [3, 4]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_sort_reverse() -> None:
    doc = tomle.parse("xs = [3, 1, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs.sort()
    assert list(xs) == [1, 2, 3]
    xs.reverse()
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [3, 2, 1]}


def test_array_imul() -> None:
    doc = tomle.parse("xs = [1, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs *= 3
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 1, 2, 1, 2]}


def test_array_remove() -> None:
    doc = tomle.parse("xs = [1, 2, 3, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs.remove(2)
    out = tomle.dumps(doc)
    assert _reparses(out) == {"xs": [1, 3, 2]}


def test_array_insert() -> None:
    doc = tomle.parse("xs = [1, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomle.Array)
    xs.insert(1, 2)
    out = tomle.dumps(doc)
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
    array_method = getattr(tomle.Array, name, None)
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
    doc = tomle.parse(src)
    src_arr = doc["src"]
    assert isinstance(src_arr, tomle.Array)
    doc["dst"] = src_arr
    dst = doc["dst"]
    assert isinstance(dst, tomle.Array)
    dst.append(99)
    assert list(src_arr) == [1, 2, 3]
    assert list(dst) == [1, 2, 3, 99]
    out = tomle.dumps(doc)
    parsed = _reparses(out)
    assert parsed == {"src": [1, 2, 3], "dst": [1, 2, 3, 99]}


def test_assigning_dict_creates_inline_table() -> None:
    doc = tomle.parse("")
    doc["obj"] = {"a": 1, "b": "two"}
    out = tomle.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1, "b": "two"}}


def test_assigning_list_creates_inline_array() -> None:
    doc = tomle.parse("")
    doc["nums"] = [1, 2, 3]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"nums": [1, 2, 3]}


def test_replace_scalar_with_array() -> None:
    doc = tomle.parse("x = 1\n")
    doc["x"] = [True, False]
    out = tomle.dumps(doc)
    assert _reparses(out) == {"x": [True, False]}
