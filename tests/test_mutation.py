"""Mutation API tests."""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

import tomlrt


def _reparses(src: str) -> dict[str, Any]:
    """Sanity check that a rendered document is still valid TOML."""
    return tomllib.loads(src)


# ---------------------------------------------------------------------------
# Scalar set/get/del
# ---------------------------------------------------------------------------


def test_replace_scalar_preserves_surrounding_format() -> None:
    src = "# header comment\nname = 'old'  # inline\nport = 80\n"
    doc = tomlrt.parse(src)
    doc["name"] = "new"
    out = tomlrt.dumps(doc)
    assert out == '# header comment\nname = "new"  # inline\nport = 80\n'
    assert _reparses(out)["name"] == "new"


def test_add_top_level_key_appends() -> None:
    src = "name = 'foo'\n"
    doc = tomlrt.parse(src)
    doc["count"] = 3
    out = tomlrt.dumps(doc)
    assert out == "name = 'foo'\ncount = 3\n"


def test_add_top_level_key_when_only_section_exists() -> None:
    src = "[srv]\nport = 8080\n"
    doc = tomlrt.parse(src)
    doc["name"] = "demo"
    out = tomlrt.dumps(doc)
    # Pre-header section is created at index 0; a blank line separates
    # the new top-level key from the following ``[srv]`` header.
    assert out == 'name = "demo"\n\n[srv]\nport = 8080\n'
    assert _reparses(out) == {"name": "demo", "srv": {"port": 8080}}


def test_add_key_inside_existing_section() -> None:
    src = "[srv]\nport = 80\n"
    doc = tomlrt.parse(src)
    srv = doc["srv"]
    assert isinstance(srv, tomlrt.Table)
    srv["host"] = "127.0.0.1"
    out = tomlrt.dumps(doc)
    assert out == '[srv]\nport = 80\nhost = "127.0.0.1"\n'
    assert _reparses(out) == {"srv": {"port": 80, "host": "127.0.0.1"}}


def test_delete_scalar_removes_line_with_leading_trivia() -> None:
    src = "a = 1\n# this comment belongs to b\nb = 2\nc = 3\n"
    doc = tomlrt.parse(src)
    del doc["b"]
    out = tomlrt.dumps(doc)
    assert out == "a = 1\nc = 3\n"


def test_delete_missing_key_raises_keyerror() -> None:
    doc = tomlrt.parse("a = 1\n")
    with pytest.raises(KeyError):
        del doc["missing"]


def test_set_overwrites_dotted_prefix() -> None:
    src = "[a]\nb.c = 1\n"
    doc = tomlrt.parse(src)
    a = doc["a"]
    assert isinstance(a, tomlrt.Table)
    a["b"] = 2
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": {"b": 2}}


def test_set_overwrites_implicit_child_table() -> None:
    src = "[a.b]\nx = 1\n"
    doc = tomlrt.parse(src)
    doc["a"] = 5
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"a": 5}


def test_quoted_key_when_bare_invalid() -> None:
    doc = tomlrt.parse("")
    doc["weird key.com"] = 1
    out = tomlrt.dumps(doc)
    assert '"weird key.com"' in out
    assert _reparses(out) == {"weird key.com": 1}


# ---------------------------------------------------------------------------
# Inline table mutation
# ---------------------------------------------------------------------------


def test_inline_table_replace() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = tomlrt.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, tomlrt.Table)
    obj["a"] = 99
    out = tomlrt.dumps(doc)
    assert out == "obj = { a = 99, b = 2 }\n"


def test_inline_table_append() -> None:
    src = "obj = { a = 1 }\n"
    doc = tomlrt.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, tomlrt.Table)
    obj["b"] = 2
    out = tomlrt.dumps(doc)
    assert "a = 1" in out
    assert "b = 2" in out
    assert _reparses(out) == {"obj": {"a": 1, "b": 2}}


def test_inline_table_delete_last_clears_trailing_comma() -> None:
    src = "obj = { a = 1, b = 2 }\n"
    doc = tomlrt.parse(src)
    obj = doc["obj"]
    assert isinstance(obj, tomlrt.Table)
    del obj["b"]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1}}


# ---------------------------------------------------------------------------
# Array mutation
# ---------------------------------------------------------------------------


def test_array_append() -> None:
    src = "xs = [1, 2, 3]\n"
    doc = tomlrt.parse(src)
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs.append(4)
    assert list(xs) == [1, 2, 3, 4]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_pop() -> None:
    doc = tomlrt.parse("xs = [10, 20, 30]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    v = xs.pop()
    assert v == 30
    assert list(xs) == [10, 20]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [10, 20]}


def test_array_setitem_int() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs[1] = 22
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 3]}


def test_array_setitem_slice() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3, 4]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs[1:3] = [22, 33, 44]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 22, 33, 44, 4]}


def test_array_delitem_slice() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3, 4]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    del xs[1:3]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 4]}


def test_array_clear_and_append() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs.clear()
    xs.append("hi")
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": ["hi"]}


def test_array_extend_iadd() -> None:
    doc = tomlrt.parse("xs = []\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs.extend([1, 2])
    xs += [3, 4]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 3, 4]}


def test_array_sort_reverse() -> None:
    doc = tomlrt.parse("xs = [3, 1, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs.sort()
    assert list(xs) == [1, 2, 3]
    xs.reverse()
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [3, 2, 1]}


def test_array_imul() -> None:
    doc = tomlrt.parse("xs = [1, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs *= 3
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 2, 1, 2, 1, 2]}


def test_array_remove() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3, 2]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs.remove(2)
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"xs": [1, 3, 2]}


def test_array_insert() -> None:
    doc = tomlrt.parse("xs = [1, 3]\n")
    xs = doc["xs"]
    assert isinstance(xs, tomlrt.Array)
    xs.insert(1, 2)
    out = tomlrt.dumps(doc)
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
    array_method = getattr(tomlrt.Array, name, None)
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
    doc = tomlrt.parse(src)
    src_arr = doc["src"]
    assert isinstance(src_arr, tomlrt.Array)
    doc["dst"] = src_arr
    dst = doc["dst"]
    assert isinstance(dst, tomlrt.Array)
    dst.append(99)
    assert list(src_arr) == [1, 2, 3]
    assert list(dst) == [1, 2, 3, 99]
    out = tomlrt.dumps(doc)
    parsed = _reparses(out)
    assert parsed == {"src": [1, 2, 3], "dst": [1, 2, 3, 99]}


def test_assigning_dict_creates_inline_table() -> None:
    doc = tomlrt.parse("")
    doc["obj"] = {"a": 1, "b": "two"}
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"obj": {"a": 1, "b": "two"}}


def test_assigning_list_creates_inline_array() -> None:
    doc = tomlrt.parse("")
    doc["nums"] = [1, 2, 3]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"nums": [1, 2, 3]}


def test_replace_scalar_with_array() -> None:
    doc = tomlrt.parse("x = 1\n")
    doc["x"] = [True, False]
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {"x": [True, False]}


# ---------------------------------------------------------------------------
# AoT mutators (pop / clear / __delitem__) — list-style mutation surface
# ---------------------------------------------------------------------------


def _aot_doc() -> tomlrt.Document:
    return tomlrt.parse(
        '[[pkg]]\nname = "a"\n\n'
        "[pkg.dep]\nx = 1\n\n"
        '[[pkg]]\nname = "b"\n\n'
        '[[pkg]]\nname = "c"\n'
    )


def test_aot_pop_default_removes_last_entry_and_owned_subsections() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    popped = aot.pop()
    assert isinstance(popped, tomlrt.Table)
    assert popped["name"] == "c"
    assert len(aot) == 2
    out = tomlrt.dumps(doc)
    assert _reparses(out) == {
        "pkg": [
            {"name": "a", "dep": {"x": 1}},
            {"name": "b"},
        ],
    }


def test_aot_pop_first_entry_takes_owned_subsections_with_it() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    popped = aot.pop(0)
    assert popped["name"] == "a"
    out = tomlrt.dumps(doc)
    assert "[pkg.dep]" not in out  # the owned sub-section went with entry 0
    assert _reparses(out) == {"pkg": [{"name": "b"}, {"name": "c"}]}


def test_aot_pop_negative_index() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    popped = aot.pop(-2)
    assert popped["name"] == "b"
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [{"name": "a", "dep": {"x": 1}}, {"name": "c"}],
    }


def test_aot_pop_index_out_of_range_raises() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    with pytest.raises(IndexError, match="pop index out of range"):
        aot.pop(99)
    with pytest.raises(IndexError, match="pop index out of range"):
        aot.pop(-99)


def test_aot_clear_removes_all_entries_and_owned_subsections() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    aot.clear()
    assert len(aot) == 0
    out = tomlrt.dumps(doc)
    assert "[[pkg]]" not in out
    assert "[pkg.dep]" not in out
    assert _reparses(out) == {}


def test_aot_delitem_index_pops_one() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    del aot[1]
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [{"name": "a", "dep": {"x": 1}}, {"name": "c"}],
    }


def test_aot_delitem_slice_removes_range() -> None:
    doc = _aot_doc()
    aot = doc.aot("pkg")
    del aot[1:]
    assert _reparses(tomlrt.dumps(doc)) == {"pkg": [{"name": "a", "dep": {"x": 1}}]}


def test_aot_delitem_slice_with_step() -> None:
    doc = tomlrt.parse(
        "[[p]]\nn=1\n[[p]]\nn=2\n[[p]]\nn=3\n[[p]]\nn=4\n",
    )
    aot = doc.aot("p")
    del aot[::2]
    assert _reparses(tomlrt.dumps(doc)) == {"p": [{"n": 2}, {"n": 4}]}


# ---------------------------------------------------------------------------
# Array.sort(key=...), Array *= n, Array.table() type-error
# ---------------------------------------------------------------------------


def test_array_sort_with_key_callable() -> None:
    doc = tomlrt.parse('xs = ["bb", "a", "ccc"]\n')
    xs = doc.array("xs")
    xs.sort(key=lambda v: len(str(v)))
    assert _reparses(tomlrt.dumps(doc)) == {"xs": ["a", "bb", "ccc"]}


def test_array_imul_zero_clears() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3]\n")
    xs = doc.array("xs")
    xs *= 0
    assert list(xs) == []
    assert _reparses(tomlrt.dumps(doc)) == {"xs": []}


def test_array_imul_negative_clears() -> None:
    doc = tomlrt.parse("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs *= -3
    assert list(xs) == []


def test_array_imul_repeats_items() -> None:
    doc = tomlrt.parse("xs = [1, 2]\n")
    xs = doc.array("xs")
    xs *= 3
    assert _reparses(tomlrt.dumps(doc)) == {"xs": [1, 2, 1, 2, 1, 2]}


def test_array_table_typed_accessor_raises_on_non_table_item() -> None:
    doc = tomlrt.parse("xs = [1, 2]\n")
    xs = doc.array("xs")
    with pytest.raises(TypeError, match="not a Table"):
        xs.table(0)


def test_array_array_typed_accessor_raises_on_non_array_item() -> None:
    doc = tomlrt.parse("xs = [1, 2]\n")
    xs = doc.array("xs")
    with pytest.raises(TypeError, match="not an Array"):
        xs.array(0)


# ---------------------------------------------------------------------------
# AoT.add — append-and-return-handle convenience
# ---------------------------------------------------------------------------


def test_aot_add_returns_new_table_view() -> None:
    doc = tomlrt.parse("")
    aot = doc.set_aot("pkg")
    pkg = aot.add({"name": "foo"})
    assert isinstance(pkg, tomlrt.Table)
    assert pkg["name"] == "foo"
    assert _reparses(tomlrt.dumps(doc)) == {"pkg": [{"name": "foo"}]}


def test_aot_add_default_empty_returns_blank_entry_for_population() -> None:
    doc = tomlrt.parse("")
    aot = doc.set_aot("pkg")
    pkg = aot.add()
    assert dict(pkg) == {}
    pkg["name"] = "bar"
    pkg.set_table("dep", {"x": 1})
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [{"name": "bar", "dep": {"x": 1}}],
    }


def test_aot_add_returned_view_stays_live_across_subsequent_adds() -> None:
    doc = tomlrt.parse("")
    aot = doc.set_aot("pkg")
    first = aot.add({"name": "a"})
    aot.add({"name": "b"})
    aot.add({"name": "c"})
    # The handle returned earlier still refers to the right entry.
    first["version"] = "1.0"
    assert _reparses(tomlrt.dumps(doc)) == {
        "pkg": [
            {"name": "a", "version": "1.0"},
            {"name": "b"},
            {"name": "c"},
        ],
    }


def test_aot_add_blank_separates_consecutive_entries() -> None:
    doc = tomlrt.parse("")
    aot = doc.set_aot("pkg")
    aot.add({"name": "a"})
    aot.add({"name": "b"})
    out = tomlrt.dumps(doc)
    # Same blank-separation behaviour as append, since add wraps it.
    assert 'name = "a"\n\n[[pkg]]' in out


# ---------------------------------------------------------------------------
# Array.append/extend/insert/__setitem__ accept dict & list at type level
# ---------------------------------------------------------------------------


def test_array_append_dict_synthesises_inline_table() -> None:
    doc = tomlrt.parse("xs = []\n")
    arr = doc.array("xs")
    arr.append({"a": 1})
    out = tomlrt.dumps(doc)
    assert "{ a = 1 }" in out
    parsed = _reparses(out)
    assert parsed == {"xs": [{"a": 1}]}


def test_array_append_list_synthesises_inline_array() -> None:
    doc = tomlrt.parse("xs = []\n")
    arr = doc.array("xs")
    arr.append([1, 2, 3])
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [[1, 2, 3]]}


def test_array_extend_mixed_python_containers() -> None:
    doc = tomlrt.parse("xs = []\n")
    arr = doc.array("xs")
    arr.extend([{"a": 1}, [1, 2], "three"])
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [{"a": 1}, [1, 2], "three"]}


def test_array_insert_dict() -> None:
    doc = tomlrt.parse("xs = [1, 3]\n")
    arr = doc.array("xs")
    arr.insert(1, {"k": "v"})
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [1, {"k": "v"}, 3]}


def test_array_setitem_replaces_with_dict() -> None:
    doc = tomlrt.parse("xs = [1, 2, 3]\n")
    arr = doc.array("xs")
    arr[1] = {"k": "v"}
    parsed = _reparses(tomlrt.dumps(doc))
    assert parsed == {"xs": [1, {"k": "v"}, 3]}


# ---------------------------------------------------------------------------
# to_dict / to_list: deep snapshot helpers
# ---------------------------------------------------------------------------


def test_table_to_dict_returns_plain_dict() -> None:
    doc = tomlrt.parse(
        """
        title = "demo"
        [owner]
        name = "alice"
        """
    )
    snap = doc.to_dict()
    assert type(snap) is dict
    assert type(snap["owner"]) is dict
    assert snap == {"title": "demo", "owner": {"name": "alice"}}


def test_table_to_dict_recursively_flattens_aot_and_array() -> None:
    doc = tomlrt.parse(
        """
        xs = [1, [2, 3], { k = "v" }]

        [[pkg]]
        name = "a"
        deps = ["x", "y"]

        [[pkg]]
        name = "b"
        """
    )
    snap = doc.to_dict()
    assert snap == {
        "xs": [1, [2, 3], {"k": "v"}],
        "pkg": [
            {"name": "a", "deps": ["x", "y"]},
            {"name": "b"},
        ],
    }
    # Every container is a real dict/list, not a tomlrt view.
    assert type(snap["xs"]) is list
    assert type(snap["xs"][2]) is dict
    assert type(snap["pkg"]) is list
    assert type(snap["pkg"][0]) is dict
    assert type(snap["pkg"][0]["deps"]) is list


def test_table_to_dict_isinstance_dict() -> None:
    doc = tomlrt.parse("[tool]\nname = 'x'\n")
    snap = doc.to_dict()
    assert isinstance(snap, dict)
    assert isinstance(snap["tool"], dict)


def test_table_to_dict_independent_of_document_mutations() -> None:
    doc = tomlrt.parse("a = 1\n[t]\nb = 2\n")
    snap = doc.to_dict()
    doc["a"] = 99
    doc.table("t")["b"] = 99
    assert snap == {"a": 1, "t": {"b": 2}}


def test_array_to_list_returns_plain_list() -> None:
    doc = tomlrt.parse('xs = [1, "two", { k = "v" }]\n')
    snap = doc.array("xs").to_list()
    assert type(snap) is list
    assert type(snap[2]) is dict
    assert snap == [1, "two", {"k": "v"}]


def test_aot_to_list_returns_list_of_dicts() -> None:
    doc = tomlrt.parse(
        """
        [[pkg]]
        name = "a"

        [[pkg]]
        name = "b"
        nested = { x = 1 }
        """
    )
    snap = doc.aot("pkg").to_list()
    assert type(snap) is list
    assert all(type(t) is dict for t in snap)
    assert snap == [{"name": "a"}, {"name": "b", "nested": {"x": 1}}]


def test_to_dict_round_trip_is_data_equivalent_to_tomllib() -> None:
    src = """
    title = "demo"
    xs = [1, 2, 3]

    [owner]
    name = "alice"

    [[pkg]]
    name = "a"
    """
    assert tomlrt.parse(src).to_dict() == _reparses(src)
