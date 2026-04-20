"""Tests for the comments side-channel and inline-table promotion."""

from __future__ import annotations

import pytest

import toml_edit

# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def test_eol_comment_present() -> None:
    src = 'name = "ada"  # the lovelace\n'
    doc = toml_edit.parse(src)
    assert doc.comments["name"] == "the lovelace"
    assert "name" in doc.comments


def test_eol_comment_absent_means_key_not_in_mapping() -> None:
    src = 'name = "ada"\n'
    doc = toml_edit.parse(src)
    assert "name" not in doc.comments
    with pytest.raises(KeyError):
        _ = doc.comments["name"]


def test_eol_comment_unknown_key_raises_keyerror() -> None:
    doc = toml_edit.parse("a = 1\n")
    with pytest.raises(KeyError):
        _ = doc.comments["missing"]


def test_eol_comment_iter_yields_only_commented_keys() -> None:
    src = "a = 1  # one\nb = 2\nc = 3  # three\n"
    doc = toml_edit.parse(src)
    assert list(doc.comments) == ["a", "c"]
    assert dict(doc.comments) == {"a": "one", "c": "three"}
    assert len(doc.comments) == 2


def test_leading_comments_present() -> None:
    src = "# a section\n# of two lines\nname = 1\n"
    doc = toml_edit.parse(src)
    assert doc.leading_comments["name"] == ("a section", "of two lines")
    assert "name" in doc.leading_comments


def test_leading_comments_absent_raises_on_get() -> None:
    doc = toml_edit.parse("name = 1\n")
    assert "name" not in doc.leading_comments
    with pytest.raises(KeyError):
        _ = doc.leading_comments["name"]


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_set_eol_comment_on_uncommented_key() -> None:
    doc = toml_edit.parse('name = "ada"\n')
    doc.comments["name"] = "the lovelace"
    assert toml_edit.dumps(doc) == 'name = "ada" # the lovelace\n'
    assert doc.comments["name"] == "the lovelace"


def test_set_eol_comment_replaces_existing() -> None:
    src = 'name = "ada"  # old\n'
    doc = toml_edit.parse(src)
    doc.comments["name"] = "new"
    assert toml_edit.dumps(doc) == 'name = "ada"  # new\n'


def test_set_eol_comment_to_empty_string_removes() -> None:
    src = 'name = "ada"  # old\n'
    doc = toml_edit.parse(src)
    doc.comments["name"] = ""
    # Clearing also drops the inline whitespace that separated the
    # comment from the value, so we don't render `name = "ada"  \n`.
    assert toml_edit.dumps(doc) == 'name = "ada"\n'
    assert "name" not in doc.comments


def test_del_eol_comment_removes_it() -> None:
    src = 'name = "ada"  # old\n'
    doc = toml_edit.parse(src)
    del doc.comments["name"]
    assert toml_edit.dumps(doc) == 'name = "ada"\n'
    assert "name" not in doc.comments
    with pytest.raises(KeyError):
        del doc.comments["name"]


def test_set_eol_comment_accepts_text_with_hash_prefix() -> None:
    doc = toml_edit.parse("a = 1\n")
    doc.comments["a"] = "## emphasised"
    assert toml_edit.dumps(doc) == "a = 1 ## emphasised\n"


def test_set_eol_comment_rejects_newline() -> None:
    doc = toml_edit.parse("a = 1\n")
    with pytest.raises(toml_edit.TOMLEditError):
        doc.comments["a"] = "no\nway"


def test_set_leading_comments_replaces_block() -> None:
    src = "# old comment\nname = 1\n"
    doc = toml_edit.parse(src)
    doc.leading_comments["name"] = ("fresh", "block")
    assert toml_edit.dumps(doc) == "# fresh\n# block\nname = 1\n"


def test_set_leading_comments_to_empty_clears_block() -> None:
    src = "# noisy\n# preamble\nname = 1\n"
    doc = toml_edit.parse(src)
    doc.leading_comments["name"] = ()
    assert toml_edit.dumps(doc) == "name = 1\n"
    assert "name" not in doc.leading_comments


def test_del_leading_comments_clears_block() -> None:
    src = "# above\nname = 1\n"
    doc = toml_edit.parse(src)
    del doc.leading_comments["name"]
    assert toml_edit.dumps(doc) == "name = 1\n"


def test_set_leading_comments_preserves_indent_in_subtable() -> None:
    src = "[tbl]\n    # explanation\n    x = 1\n"
    doc = toml_edit.parse(src)
    tbl = doc["tbl"]
    assert isinstance(tbl, toml_edit.Table)
    assert tbl.leading_comments["x"] == ("explanation",)
    tbl.leading_comments["x"] = ("replaced",)
    assert toml_edit.dumps(doc) == "[tbl]\n    # replaced\n    x = 1\n"


# ---------------------------------------------------------------------------
# Bulk-shaped operations (the reason side-channels exist)
# ---------------------------------------------------------------------------


def test_dict_of_comments_round_trips_via_update() -> None:
    src = "a = 1\nb = 2\nc = 3\n"
    doc = toml_edit.parse(src)
    doc.comments.update({"a": "first", "c": "third"})
    assert dict(doc.comments) == {"a": "first", "c": "third"}
    assert toml_edit.dumps(doc) == "a = 1 # first\nb = 2\nc = 3 # third\n"


def test_comments_view_is_live_not_snapshot() -> None:
    doc = toml_edit.parse("a = 1  # original\n")
    view = doc.comments
    doc.comments["a"] = "updated"
    assert view["a"] == "updated"


def test_comments_view_repr_shows_pairs() -> None:
    doc = toml_edit.parse("a = 1  # one\n")
    r = repr(doc.comments)
    assert "'a': 'one'" in r


# ---------------------------------------------------------------------------
# Inline tables and promotion
# ---------------------------------------------------------------------------


def test_comments_on_inline_table_raises_with_helpful_message() -> None:
    src = 'pkg = { name = "toml_edit", version = "0.1" }\n'
    doc = toml_edit.parse(src)
    pkg = doc["pkg"]
    assert isinstance(pkg, toml_edit.Table)
    with pytest.raises(toml_edit.TOMLEditError, match="comment API"):
        pkg.comments["name"] = "x"
    with pytest.raises(toml_edit.TOMLEditError, match="comment API"):
        _ = pkg.leading_comments


def test_inline_table_promotion_basic() -> None:
    src = 'pkg = { name = "toml_edit", version = "0.1" }\n'
    doc = toml_edit.parse(src)
    promoted = doc.promote_inline("pkg")
    assert isinstance(promoted, toml_edit.Table)
    assert promoted["name"] == "toml_edit"
    assert promoted["version"] == "0.1"
    assert toml_edit.dumps(doc) == '[pkg]\nname = "toml_edit"\nversion = "0.1"\n'


def test_inline_table_promotion_preserves_leading_comments() -> None:
    src = '# the package\npkg = { name = "toml_edit" }\n'
    doc = toml_edit.parse(src)
    doc.promote_inline("pkg")
    assert toml_edit.dumps(doc) == '# the package\n[pkg]\nname = "toml_edit"\n'


def test_inline_table_promotion_preserves_eol_comment_on_header() -> None:
    src = 'pkg = { name = "toml_edit" }  # describes pkg\n'
    doc = toml_edit.parse(src)
    doc.promote_inline("pkg")
    assert toml_edit.dumps(doc) == '[pkg]  # describes pkg\nname = "toml_edit"\n'


def test_inline_promotion_then_set_comment_on_member() -> None:
    src = 'pkg = { name = "toml_edit", version = "0.1" }\n'
    doc = toml_edit.parse(src)
    promoted = doc.promote_inline("pkg")
    promoted.comments["version"] = "calver soon"
    assert toml_edit.dumps(doc) == (
        '[pkg]\nname = "toml_edit"\nversion = "0.1" # calver soon\n'
    )


def test_inline_promotion_inserts_after_parent_block() -> None:
    src = "[parent]\na = 1\npkg = { x = 10 }\n[other]\nb = 2\n"
    doc = toml_edit.parse(src)
    parent = doc["parent"]
    assert isinstance(parent, toml_edit.Table)
    parent.promote_inline("pkg")
    assert toml_edit.dumps(doc) == (
        "[parent]\na = 1\n[parent.pkg]\nx = 10\n[other]\nb = 2\n"
    )


def test_promote_non_inline_raises() -> None:
    doc = toml_edit.parse("a = 1\n")
    with pytest.raises(toml_edit.TOMLEditError, match="not an inline table"):
        doc.promote_inline("a")


def test_promote_unknown_key_raises_keyerror() -> None:
    doc = toml_edit.parse("a = 1\n")
    with pytest.raises(KeyError):
        doc.promote_inline("missing")


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


def test_round_trip_after_set_and_clear() -> None:
    doc = toml_edit.parse("x = 1\n")
    doc.comments["x"] = "trailing"
    doc.leading_comments["x"] = ("above",)
    out = toml_edit.dumps(doc)
    assert out == "# above\nx = 1 # trailing\n"
    again = toml_edit.parse(out)
    assert again.comments["x"] == "trailing"
    assert again.leading_comments["x"] == ("above",)


# ---------------------------------------------------------------------------
# Header comment API
# ---------------------------------------------------------------------------


def test_header_comment_present() -> None:
    src = "[server] # in DC1\nhost = 'a'\n"
    doc = toml_edit.parse(src)
    assert doc.table("server").header_comment == "in DC1"


def test_header_comment_absent() -> None:
    src = "[server]\nhost = 'a'\n"
    doc = toml_edit.parse(src)
    assert doc.table("server").header_comment is None


def test_header_comment_set_round_trips() -> None:
    doc = toml_edit.parse("[server]\nhost = 'a'\n")
    doc.table("server").header_comment = "in DC1"
    assert toml_edit.dumps(doc) == "[server] # in DC1\nhost = 'a'\n"


def test_header_comment_replace_existing() -> None:
    doc = toml_edit.parse("[server] # old\nhost = 'a'\n")
    doc.table("server").header_comment = "new"
    assert toml_edit.dumps(doc) == "[server] # new\nhost = 'a'\n"


def test_header_comment_clear_with_empty_string() -> None:
    doc = toml_edit.parse("[server] # old\nhost = 'a'\n")
    doc.table("server").header_comment = ""
    assert toml_edit.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_clear_with_none() -> None:
    doc = toml_edit.parse("[server] # old\nhost = 'a'\n")
    doc.table("server").header_comment = None
    assert toml_edit.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_del() -> None:
    doc = toml_edit.parse("[server] # old\nhost = 'a'\n")
    del doc.table("server").header_comment
    assert doc.table("server").header_comment is None


def test_header_leading_comments_extract_block_only() -> None:
    src = "# old archived note\n\n# active 1\n# active 2\n[server]\nhost = 'a'\n"
    doc = toml_edit.parse(src)
    # Only the *contiguous* block above the header counts.
    assert doc.table("server").header_leading_comments == ("active 1", "active 2")


def test_header_leading_comments_round_trip() -> None:
    src = "# above\n[server]\nhost = 'a'\n"
    doc = toml_edit.parse(src)
    assert toml_edit.dumps(doc) == src


def test_header_leading_comments_set_preserves_older_block() -> None:
    src = "# old archived note\n\n# active\n[server]\nhost = 'a'\n"
    doc = toml_edit.parse(src)
    doc.table("server").header_leading_comments = ("brand new",)
    out = toml_edit.dumps(doc)
    # Older blank-separated comment must remain untouched.
    assert out == ("# old archived note\n\n# brand new\n[server]\nhost = 'a'\n")


def test_header_leading_comments_set_on_empty() -> None:
    doc = toml_edit.parse("[server]\nhost = 'a'\n")
    doc.table("server").header_leading_comments = ("hello", "world")
    assert toml_edit.dumps(doc) == "# hello\n# world\n[server]\nhost = 'a'\n"


def test_header_leading_comments_clear_with_empty_tuple() -> None:
    doc = toml_edit.parse("# above\n[server]\nhost = 'a'\n")
    doc.table("server").header_leading_comments = ()
    assert toml_edit.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_leading_comments_del() -> None:
    doc = toml_edit.parse("# above\n[server]\nhost = 'a'\n")
    del doc.table("server").header_leading_comments
    assert toml_edit.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_on_aot_entry() -> None:
    src = "[[items]]\nname = 'a'\n\n[[items]]\nname = 'b'\n"
    doc = toml_edit.parse(src)
    items = doc["items"]
    assert isinstance(items, toml_edit.AoT)
    items[0].header_comment = "first"
    items[1].header_leading_comments = ("about the second",)
    out = toml_edit.dumps(doc)
    assert out == (
        "[[items]] # first\nname = 'a'\n\n# about the second\n[[items]]\nname = 'b'\n"
    )


def test_header_comment_on_document_raises() -> None:
    doc = toml_edit.parse("a = 1\n")
    with pytest.raises(toml_edit.TOMLEditError):
        _ = doc.header_comment
    with pytest.raises(toml_edit.TOMLEditError):
        doc.header_comment = "x"
    with pytest.raises(toml_edit.TOMLEditError):
        _ = doc.header_leading_comments


def test_header_comment_on_inline_table_raises() -> None:
    doc = toml_edit.parse("a = { x = 1, y = 2 }\n")
    a = doc["a"]
    assert isinstance(a, toml_edit.Table)
    with pytest.raises(toml_edit.TOMLEditError):
        _ = a.header_comment
    with pytest.raises(toml_edit.TOMLEditError):
        _ = a.header_leading_comments


def test_header_comment_on_implicit_parent_raises() -> None:
    # `parent` exists logically but has no `[parent]` section in source.
    doc = toml_edit.parse("[parent.child]\nx = 1\n")
    parent = doc["parent"]
    assert isinstance(parent, toml_edit.Table)
    with pytest.raises(toml_edit.TOMLEditError):
        _ = parent.header_comment
    with pytest.raises(toml_edit.TOMLEditError):
        _ = parent.header_leading_comments


# ---------------------------------------------------------------------------
# Pre-existing leading_comments bug fix: only the trailing block counts
# ---------------------------------------------------------------------------


def test_leading_comments_extract_block_only() -> None:
    src = "# old archived note\n\n# active 1\n# active 2\nname = 'x'\n"
    doc = toml_edit.parse(src)
    assert doc.leading_comments["name"] == ("active 1", "active 2")


def test_leading_comments_set_preserves_older_block() -> None:
    src = "# old archived note\n\n# active\nname = 'x'\n"
    doc = toml_edit.parse(src)
    doc.leading_comments["name"] = ("brand new",)
    assert toml_edit.dumps(doc) == ("# old archived note\n\n# brand new\nname = 'x'\n")


# ---------------------------------------------------------------------------
# Array element comments (Phase B)
# ---------------------------------------------------------------------------


def test_array_eol_comments_read_multiline() -> None:
    src = "arr = [\n  1, # one\n  2, # two\n  3, # three\n]\n"
    doc = toml_edit.parse(src)
    arr = doc.array("arr")
    assert dict(arr.comments) == {0: "one", 1: "two", 2: "three"}


def test_array_eol_comment_read_last_no_trailing_comma() -> None:
    src = "arr = [\n  1,\n  2 # last\n]\n"
    doc = toml_edit.parse(src)
    arr = doc.array("arr")
    assert dict(arr.comments) == {1: "last"}


def test_array_leading_comments_read() -> None:
    src = "arr = [\n  # before 0\n  1,\n  # before 1a\n  # before 1b\n  2,\n]\n"
    doc = toml_edit.parse(src)
    arr = doc.array("arr")
    assert dict(arr.leading_comments) == {
        0: ("before 0",),
        1: ("before 1a", "before 1b"),
    }


def test_array_round_trip_with_comments() -> None:
    src = "arr = [\n  1, # one\n  2, # two\n]\n"
    doc = toml_edit.parse(src)
    assert toml_edit.dumps(doc) == src


def test_array_set_eol_on_single_line_promotes_to_multiline() -> None:
    doc = toml_edit.parse("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.comments[1] = "two"
    out = toml_edit.dumps(doc)
    re = toml_edit.parse(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.comments) == {1: "two"}


def test_array_set_eol_on_last_item_no_comma_breaks_before_close() -> None:
    doc = toml_edit.parse("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.comments[2] = "last"
    out = toml_edit.dumps(doc)
    # `]` must not be on the same line as the comment.
    assert "# last\n" in out
    assert out.rstrip().endswith("]")
    re = toml_edit.parse(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.comments) == {2: "last"}


def test_array_set_eol_on_last_item_with_trailing_comma() -> None:
    doc = toml_edit.parse("arr = [\n  1,\n  2,\n]\n")
    arr = doc.array("arr")
    arr.comments[1] = "second"
    out = toml_edit.dumps(doc)
    re = toml_edit.parse(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2]
    assert dict(re_arr.comments) == {1: "second"}


def test_array_replace_existing_eol_comment() -> None:
    doc = toml_edit.parse("arr = [\n  1, # old\n  2,\n]\n")
    arr = doc.array("arr")
    arr.comments[0] = "new"
    out = toml_edit.dumps(doc)
    assert "# new" in out
    assert "# old" not in out


def test_array_delete_eol_comment() -> None:
    doc = toml_edit.parse("arr = [\n  1, # one\n  2,\n]\n")
    arr = doc.array("arr")
    del arr.comments[0]
    out = toml_edit.dumps(doc)
    assert "# one" not in out
    re = toml_edit.parse(out)
    assert list(re.array("arr")) == [1, 2]


def test_array_set_leading_on_single_line_promotes_to_multiline() -> None:
    doc = toml_edit.parse("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.leading_comments[1] = ("before two",)
    out = toml_edit.dumps(doc)
    re = toml_edit.parse(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.leading_comments) == {1: ("before two",)}


def test_array_set_leading_on_first_item() -> None:
    doc = toml_edit.parse("arr = [1, 2, 3]\n")
    arr = doc.array("arr")
    arr.leading_comments[0] = ("first",)
    out = toml_edit.dumps(doc)
    re = toml_edit.parse(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.leading_comments) == {0: ("first",)}


def test_array_set_multiple_leading_lines() -> None:
    doc = toml_edit.parse("arr = [\n  1,\n  2,\n]\n")
    arr = doc.array("arr")
    arr.leading_comments[1] = ("line one", "line two")
    out = toml_edit.dumps(doc)
    re = toml_edit.parse(out)
    re_arr = re.array("arr")
    assert dict(re_arr.leading_comments) == {1: ("line one", "line two")}


def test_array_delete_leading_comments() -> None:
    src = "arr = [\n  # before\n  1,\n  2,\n]\n"
    doc = toml_edit.parse(src)
    arr = doc.array("arr")
    del arr.leading_comments[0]
    out = toml_edit.dumps(doc)
    assert "# before" not in out
    re = toml_edit.parse(out)
    assert list(re.array("arr")) == [1, 2]


def test_array_append_migrates_last_eol_comment() -> None:
    doc = toml_edit.parse("arr = [\n  1,\n  2 # last\n]\n")
    arr = doc.array("arr")
    assert dict(arr.comments) == {1: "last"}
    arr.append(3)
    out = toml_edit.dumps(doc)
    re = toml_edit.parse(out)
    re_arr = re.array("arr")
    assert list(re_arr) == [1, 2, 3]
    # The EOL must still belong to item 1, not item 2.
    assert dict(re_arr.comments) == {1: "last"}


def test_array_comments_view_contains_iter_len() -> None:
    src = "arr = [\n  1, # one\n  2,\n  3, # three\n]\n"
    doc = toml_edit.parse(src)
    arr = doc.array("arr")
    assert 0 in arr.comments
    assert 1 not in arr.comments
    assert 2 in arr.comments
    assert 99 not in arr.comments
    assert sorted(arr.comments) == [0, 2]
    assert len(arr.comments) == 2


def test_array_comments_view_empty_array() -> None:
    doc = toml_edit.parse("arr = []\n")
    arr = doc.array("arr")
    assert len(arr.comments) == 0
    assert list(arr.comments) == []
    assert len(arr.leading_comments) == 0
    with pytest.raises(KeyError):
        _ = arr.comments[0]
    with pytest.raises(KeyError):
        _ = arr.leading_comments[0]


def test_array_comments_non_int_key_raises() -> None:
    doc = toml_edit.parse("arr = [1, 2]\n")
    arr = doc.array("arr")
    with pytest.raises(TypeError):
        _ = arr.comments["x"]  # type: ignore[index]
    with pytest.raises(TypeError):
        arr.comments["x"] = "v"  # type: ignore[index]


def test_array_comments_out_of_range_raises() -> None:
    doc = toml_edit.parse("arr = [1, 2]\n")
    arr = doc.array("arr")
    with pytest.raises(KeyError):
        arr.comments[5] = "nope"
    with pytest.raises(KeyError):
        arr.leading_comments[5] = ("nope",)
    with pytest.raises(KeyError):
        del arr.comments[5]


def test_array_comment_with_hash_prefix_normalised() -> None:
    doc = toml_edit.parse("arr = [1, 2]\n")
    arr = doc.array("arr")
    arr.comments[0] = "# already-prefixed"
    re = toml_edit.parse(toml_edit.dumps(doc))
    # We don't double-up the `#`.
    assert dict(re.array("arr").comments) == {0: "already-prefixed"}


def test_array_set_value_via_indexing_preserves_eol_comment() -> None:
    src = "arr = [\n  1, # one\n  2, # two\n]\n"
    doc = toml_edit.parse(src)
    arr = doc.array("arr")
    arr[0] = 99
    re = toml_edit.parse(toml_edit.dumps(doc))
    re_arr = re.array("arr")
    assert list(re_arr) == [99, 2]
    # Comment ownership shouldn't change.
    assert dict(re_arr.comments) == {0: "one", 1: "two"}


# ---------------------------------------------------------------------------
# Typed accessors: Table.array / .table / .aot, Array.array / .table
# ---------------------------------------------------------------------------


def test_table_array_returns_array() -> None:
    doc = toml_edit.parse("xs = [1, 2]\n")
    arr = doc.array("xs")
    assert isinstance(arr, toml_edit.Array)
    arr.comments[0] = "first"
    assert "# first" in toml_edit.dumps(doc)


def test_table_table_returns_table() -> None:
    doc = toml_edit.parse("[server]\nport = 80\n")
    tbl = doc.table("server")
    assert isinstance(tbl, toml_edit.Table)
    tbl.header_comment = "production"
    assert "# production" in toml_edit.dumps(doc)


def test_table_aot_returns_aot() -> None:
    doc = toml_edit.parse("[[products]]\nname = 'a'\n[[products]]\nname = 'b'\n")
    aot = doc.aot("products")
    assert isinstance(aot, toml_edit.AoT)
    assert len(aot) == 2


def test_table_array_wrong_kind_raises_typeerror() -> None:
    doc = toml_edit.parse("x = 1\n")
    with pytest.raises(TypeError, match="not an Array"):
        doc.array("x")


def test_table_table_wrong_kind_raises_typeerror() -> None:
    doc = toml_edit.parse("x = 1\n")
    with pytest.raises(TypeError, match="not a Table"):
        doc.table("x")


def test_table_aot_wrong_kind_raises_typeerror() -> None:
    doc = toml_edit.parse("x = 1\n")
    with pytest.raises(TypeError, match="not an AoT"):
        doc.aot("x")


def test_table_typed_accessors_propagate_keyerror() -> None:
    doc = toml_edit.parse("x = 1\n")
    with pytest.raises(KeyError):
        doc.array("missing")
    with pytest.raises(KeyError):
        doc.table("missing")
    with pytest.raises(KeyError):
        doc.aot("missing")


def test_array_array_returns_nested_array() -> None:
    doc = toml_edit.parse("xs = [[1, 2], [3, 4]]\n")
    inner = doc.array("xs").array(0)
    assert isinstance(inner, toml_edit.Array)
    assert list(inner) == [1, 2]


def test_array_table_returns_nested_inline_table() -> None:
    doc = toml_edit.parse("xs = [{a = 1}, {a = 2}]\n")
    tbl = doc.array("xs").table(0)
    assert isinstance(tbl, toml_edit.Table)
    assert tbl["a"] == 1


def test_array_array_wrong_kind_raises_typeerror() -> None:
    doc = toml_edit.parse("xs = [1, 2]\n")
    with pytest.raises(TypeError, match="not an Array"):
        doc.array("xs").array(0)
