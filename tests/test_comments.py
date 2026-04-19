"""Tests for the comments side-channel and inline-table promotion."""

from __future__ import annotations

from typing import cast

import pytest

import tomle

# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def test_eol_comment_present() -> None:
    src = 'name = "ada"  # the lovelace\n'
    doc = tomle.parse(src)
    assert doc.comments["name"] == "the lovelace"
    assert "name" in doc.comments


def test_eol_comment_absent_means_key_not_in_mapping() -> None:
    src = 'name = "ada"\n'
    doc = tomle.parse(src)
    assert "name" not in doc.comments
    with pytest.raises(KeyError):
        _ = doc.comments["name"]


def test_eol_comment_unknown_key_raises_keyerror() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(KeyError):
        _ = doc.comments["missing"]


def test_eol_comment_iter_yields_only_commented_keys() -> None:
    src = "a = 1  # one\nb = 2\nc = 3  # three\n"
    doc = tomle.parse(src)
    assert list(doc.comments) == ["a", "c"]
    assert dict(doc.comments) == {"a": "one", "c": "three"}
    assert len(doc.comments) == 2


def test_leading_comments_present() -> None:
    src = "# a section\n# of two lines\nname = 1\n"
    doc = tomle.parse(src)
    assert doc.leading_comments["name"] == ("a section", "of two lines")
    assert "name" in doc.leading_comments


def test_leading_comments_absent_raises_on_get() -> None:
    doc = tomle.parse("name = 1\n")
    assert "name" not in doc.leading_comments
    with pytest.raises(KeyError):
        _ = doc.leading_comments["name"]


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_set_eol_comment_on_uncommented_key() -> None:
    doc = tomle.parse('name = "ada"\n')
    doc.comments["name"] = "the lovelace"
    assert tomle.dumps(doc) == 'name = "ada" # the lovelace\n'
    assert doc.comments["name"] == "the lovelace"


def test_set_eol_comment_replaces_existing() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomle.parse(src)
    doc.comments["name"] = "new"
    assert tomle.dumps(doc) == 'name = "ada"  # new\n'


def test_set_eol_comment_to_empty_string_removes() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomle.parse(src)
    doc.comments["name"] = ""
    # Clearing also drops the inline whitespace that separated the
    # comment from the value, so we don't render `name = "ada"  \n`.
    assert tomle.dumps(doc) == 'name = "ada"\n'
    assert "name" not in doc.comments


def test_del_eol_comment_removes_it() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomle.parse(src)
    del doc.comments["name"]
    assert tomle.dumps(doc) == 'name = "ada"\n'
    assert "name" not in doc.comments
    with pytest.raises(KeyError):
        del doc.comments["name"]


def test_set_eol_comment_accepts_text_with_hash_prefix() -> None:
    doc = tomle.parse("a = 1\n")
    doc.comments["a"] = "## emphasised"
    assert tomle.dumps(doc) == "a = 1 ## emphasised\n"


def test_set_eol_comment_rejects_newline() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(tomle.TOMLEditError):
        doc.comments["a"] = "no\nway"


def test_set_leading_comments_replaces_block() -> None:
    src = "# old comment\nname = 1\n"
    doc = tomle.parse(src)
    doc.leading_comments["name"] = ("fresh", "block")
    assert tomle.dumps(doc) == "# fresh\n# block\nname = 1\n"


def test_set_leading_comments_to_empty_clears_block() -> None:
    src = "# noisy\n# preamble\nname = 1\n"
    doc = tomle.parse(src)
    doc.leading_comments["name"] = ()
    assert tomle.dumps(doc) == "name = 1\n"
    assert "name" not in doc.leading_comments


def test_del_leading_comments_clears_block() -> None:
    src = "# above\nname = 1\n"
    doc = tomle.parse(src)
    del doc.leading_comments["name"]
    assert tomle.dumps(doc) == "name = 1\n"


def test_set_leading_comments_preserves_indent_in_subtable() -> None:
    src = "[tbl]\n    # explanation\n    x = 1\n"
    doc = tomle.parse(src)
    tbl = doc["tbl"]
    assert isinstance(tbl, tomle.Table)
    assert tbl.leading_comments["x"] == ("explanation",)
    tbl.leading_comments["x"] = ("replaced",)
    assert tomle.dumps(doc) == "[tbl]\n    # replaced\n    x = 1\n"


# ---------------------------------------------------------------------------
# Bulk-shaped operations (the reason side-channels exist)
# ---------------------------------------------------------------------------


def test_dict_of_comments_round_trips_via_update() -> None:
    src = "a = 1\nb = 2\nc = 3\n"
    doc = tomle.parse(src)
    doc.comments.update({"a": "first", "c": "third"})
    assert dict(doc.comments) == {"a": "first", "c": "third"}
    assert tomle.dumps(doc) == "a = 1 # first\nb = 2\nc = 3 # third\n"


def test_comments_view_is_live_not_snapshot() -> None:
    doc = tomle.parse("a = 1  # original\n")
    view = doc.comments
    doc.comments["a"] = "updated"
    assert view["a"] == "updated"


def test_comments_view_repr_shows_pairs() -> None:
    doc = tomle.parse("a = 1  # one\n")
    assert repr(doc.comments) == "_TableCommentsView({'a': 'one'})"


# ---------------------------------------------------------------------------
# Inline tables and promotion
# ---------------------------------------------------------------------------


def test_comments_on_inline_table_raises_with_helpful_message() -> None:
    src = 'pkg = { name = "tomle", version = "0.1" }\n'
    doc = tomle.parse(src)
    pkg = doc["pkg"]
    assert isinstance(pkg, tomle.Table)
    with pytest.raises(tomle.TOMLEditError, match="comment API"):
        pkg.comments["name"] = "x"
    with pytest.raises(tomle.TOMLEditError, match="comment API"):
        _ = pkg.leading_comments


def test_inline_table_promotion_basic() -> None:
    src = 'pkg = { name = "tomle", version = "0.1" }\n'
    doc = tomle.parse(src)
    promoted = doc.promote_inline("pkg")
    assert isinstance(promoted, tomle.Table)
    assert promoted["name"] == "tomle"
    assert promoted["version"] == "0.1"
    assert tomle.dumps(doc) == '[pkg]\nname = "tomle"\nversion = "0.1"\n'


def test_inline_table_promotion_preserves_leading_comments() -> None:
    src = '# the package\npkg = { name = "tomle" }\n'
    doc = tomle.parse(src)
    doc.promote_inline("pkg")
    assert tomle.dumps(doc) == '# the package\n[pkg]\nname = "tomle"\n'


def test_inline_table_promotion_preserves_eol_comment_on_header() -> None:
    src = 'pkg = { name = "tomle" }  # describes pkg\n'
    doc = tomle.parse(src)
    doc.promote_inline("pkg")
    assert tomle.dumps(doc) == '[pkg]  # describes pkg\nname = "tomle"\n'


def test_inline_promotion_then_set_comment_on_member() -> None:
    src = 'pkg = { name = "tomle", version = "0.1" }\n'
    doc = tomle.parse(src)
    promoted = doc.promote_inline("pkg")
    promoted.comments["version"] = "calver soon"
    assert tomle.dumps(doc) == (
        '[pkg]\nname = "tomle"\nversion = "0.1" # calver soon\n'
    )


def test_inline_promotion_inserts_after_parent_block() -> None:
    src = (
        "[parent]\n"
        "a = 1\n"
        "pkg = { x = 10 }\n"
        "[other]\n"
        "b = 2\n"
    )
    doc = tomle.parse(src)
    parent = doc["parent"]
    assert isinstance(parent, tomle.Table)
    parent.promote_inline("pkg")
    assert tomle.dumps(doc) == (
        "[parent]\n"
        "a = 1\n"
        "[parent.pkg]\n"
        "x = 10\n"
        "[other]\n"
        "b = 2\n"
    )


def test_promote_non_inline_raises() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(tomle.TOMLEditError, match="not an inline table"):
        doc.promote_inline("a")


def test_promote_unknown_key_raises_keyerror() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(KeyError):
        doc.promote_inline("missing")


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


def test_round_trip_after_set_and_clear() -> None:
    doc = tomle.parse("x = 1\n")
    doc.comments["x"] = "trailing"
    doc.leading_comments["x"] = ("above",)
    out = tomle.dumps(doc)
    assert out == "# above\nx = 1 # trailing\n"
    again = tomle.parse(out)
    assert again.comments["x"] == "trailing"
    assert again.leading_comments["x"] == ("above",)


# ---------------------------------------------------------------------------
# Header comment API
# ---------------------------------------------------------------------------


def test_header_comment_present() -> None:
    src = "[server] # in DC1\nhost = 'a'\n"
    doc = tomle.parse(src)
    assert cast("tomle.Table", doc["server"]).header_comment == "in DC1"


def test_header_comment_absent() -> None:
    src = "[server]\nhost = 'a'\n"
    doc = tomle.parse(src)
    assert cast("tomle.Table", doc["server"]).header_comment is None


def test_header_comment_set_round_trips() -> None:
    doc = tomle.parse("[server]\nhost = 'a'\n")
    cast("tomle.Table", doc["server"]).header_comment = "in DC1"
    assert tomle.dumps(doc) == "[server] # in DC1\nhost = 'a'\n"


def test_header_comment_replace_existing() -> None:
    doc = tomle.parse("[server] # old\nhost = 'a'\n")
    cast("tomle.Table", doc["server"]).header_comment = "new"
    assert tomle.dumps(doc) == "[server] # new\nhost = 'a'\n"


def test_header_comment_clear_with_empty_string() -> None:
    doc = tomle.parse("[server] # old\nhost = 'a'\n")
    cast("tomle.Table", doc["server"]).header_comment = ""
    assert tomle.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_clear_with_none() -> None:
    doc = tomle.parse("[server] # old\nhost = 'a'\n")
    cast("tomle.Table", doc["server"]).header_comment = None
    assert tomle.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_del() -> None:
    doc = tomle.parse("[server] # old\nhost = 'a'\n")
    del cast("tomle.Table", doc["server"]).header_comment
    assert cast("tomle.Table", doc["server"]).header_comment is None


def test_header_leading_comments_extract_block_only() -> None:
    src = (
        "# old archived note\n"
        "\n"
        "# active 1\n"
        "# active 2\n"
        "[server]\nhost = 'a'\n"
    )
    doc = tomle.parse(src)
    # Only the *contiguous* block above the header counts.
    assert cast("tomle.Table", doc["server"]).header_leading_comments == ("active 1", "active 2")


def test_header_leading_comments_round_trip() -> None:
    src = "# above\n[server]\nhost = 'a'\n"
    doc = tomle.parse(src)
    assert tomle.dumps(doc) == src


def test_header_leading_comments_set_preserves_older_block() -> None:
    src = (
        "# old archived note\n"
        "\n"
        "# active\n"
        "[server]\nhost = 'a'\n"
    )
    doc = tomle.parse(src)
    cast("tomle.Table", doc["server"]).header_leading_comments = ("brand new",)
    out = tomle.dumps(doc)
    # Older blank-separated comment must remain untouched.
    assert out == (
        "# old archived note\n"
        "\n"
        "# brand new\n"
        "[server]\nhost = 'a'\n"
    )


def test_header_leading_comments_set_on_empty() -> None:
    doc = tomle.parse("[server]\nhost = 'a'\n")
    cast("tomle.Table", doc["server"]).header_leading_comments = ("hello", "world")
    assert tomle.dumps(doc) == "# hello\n# world\n[server]\nhost = 'a'\n"


def test_header_leading_comments_clear_with_empty_tuple() -> None:
    doc = tomle.parse("# above\n[server]\nhost = 'a'\n")
    cast("tomle.Table", doc["server"]).header_leading_comments = ()
    assert tomle.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_leading_comments_del() -> None:
    doc = tomle.parse("# above\n[server]\nhost = 'a'\n")
    del cast("tomle.Table", doc["server"]).header_leading_comments
    assert tomle.dumps(doc) == "[server]\nhost = 'a'\n"


def test_header_comment_on_aot_entry() -> None:
    src = "[[items]]\nname = 'a'\n\n[[items]]\nname = 'b'\n"
    doc = tomle.parse(src)
    items = doc["items"]
    assert isinstance(items, tomle.AoT)
    items[0].header_comment = "first"
    items[1].header_leading_comments = ("about the second",)
    out = tomle.dumps(doc)
    assert out == (
        "[[items]] # first\nname = 'a'\n\n"
        "# about the second\n[[items]]\nname = 'b'\n"
    )


def test_header_comment_on_document_raises() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(tomle.TOMLEditError):
        _ = doc.header_comment
    with pytest.raises(tomle.TOMLEditError):
        doc.header_comment = "x"
    with pytest.raises(tomle.TOMLEditError):
        _ = doc.header_leading_comments


def test_header_comment_on_inline_table_raises() -> None:
    doc = tomle.parse("a = { x = 1, y = 2 }\n")
    a = doc["a"]
    assert isinstance(a, tomle.Table)
    with pytest.raises(tomle.TOMLEditError):
        _ = a.header_comment
    with pytest.raises(tomle.TOMLEditError):
        _ = a.header_leading_comments


def test_header_comment_on_implicit_parent_raises() -> None:
    # `parent` exists logically but has no `[parent]` section in source.
    doc = tomle.parse("[parent.child]\nx = 1\n")
    parent = doc["parent"]
    assert isinstance(parent, tomle.Table)
    with pytest.raises(tomle.TOMLEditError):
        _ = parent.header_comment
    with pytest.raises(tomle.TOMLEditError):
        _ = parent.header_leading_comments


# ---------------------------------------------------------------------------
# Pre-existing leading_comments bug fix: only the trailing block counts
# ---------------------------------------------------------------------------


def test_leading_comments_extract_block_only() -> None:
    src = (
        "# old archived note\n"
        "\n"
        "# active 1\n"
        "# active 2\n"
        "name = 'x'\n"
    )
    doc = tomle.parse(src)
    assert doc.leading_comments["name"] == ("active 1", "active 2")


def test_leading_comments_set_preserves_older_block() -> None:
    src = (
        "# old archived note\n"
        "\n"
        "# active\n"
        "name = 'x'\n"
    )
    doc = tomle.parse(src)
    doc.leading_comments["name"] = ("brand new",)
    assert tomle.dumps(doc) == (
        "# old archived note\n"
        "\n"
        "# brand new\n"
        "name = 'x'\n"
    )


# ---------------------------------------------------------------------------
# Array element comments (Phase B)
# ---------------------------------------------------------------------------


def test_array_eol_comments_read_multiline() -> None:
    src = "arr = [\n  1, # one\n  2, # two\n  3, # three\n]\n"
    doc = tomle.parse(src)
    arr = cast("tomle.Array", doc["arr"])
    assert dict(arr.comments) == {0: "one", 1: "two", 2: "three"}


def test_array_eol_comment_read_last_no_trailing_comma() -> None:
    src = "arr = [\n  1,\n  2 # last\n]\n"
    doc = tomle.parse(src)
    arr = cast("tomle.Array", doc["arr"])
    assert dict(arr.comments) == {1: "last"}


def test_array_leading_comments_read() -> None:
    src = "arr = [\n  # before 0\n  1,\n  # before 1a\n  # before 1b\n  2,\n]\n"
    doc = tomle.parse(src)
    arr = cast("tomle.Array", doc["arr"])
    assert dict(arr.leading_comments) == {0: ("before 0",), 1: ("before 1a", "before 1b")}


def test_array_round_trip_with_comments() -> None:
    src = "arr = [\n  1, # one\n  2, # two\n]\n"
    doc = tomle.parse(src)
    assert tomle.dumps(doc) == src


def test_array_set_eol_on_single_line_promotes_to_multiline() -> None:
    doc = tomle.parse("arr = [1, 2, 3]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.comments[1] = "two"
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    re_arr = cast("tomle.Array", re["arr"])
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.comments) == {1: "two"}


def test_array_set_eol_on_last_item_no_comma_breaks_before_close() -> None:
    doc = tomle.parse("arr = [1, 2, 3]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.comments[2] = "last"
    out = tomle.dumps(doc)
    # `]` must not be on the same line as the comment.
    assert "# last\n" in out
    assert out.rstrip().endswith("]")
    re = tomle.parse(out)
    re_arr = cast("tomle.Array", re["arr"])
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.comments) == {2: "last"}


def test_array_set_eol_on_last_item_with_trailing_comma() -> None:
    doc = tomle.parse("arr = [\n  1,\n  2,\n]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.comments[1] = "second"
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    re_arr = cast("tomle.Array", re["arr"])
    assert list(re_arr) == [1, 2]
    assert dict(re_arr.comments) == {1: "second"}


def test_array_replace_existing_eol_comment() -> None:
    doc = tomle.parse("arr = [\n  1, # old\n  2,\n]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.comments[0] = "new"
    out = tomle.dumps(doc)
    assert "# new" in out
    assert "# old" not in out


def test_array_delete_eol_comment() -> None:
    doc = tomle.parse("arr = [\n  1, # one\n  2,\n]\n")
    arr = cast("tomle.Array", doc["arr"])
    del arr.comments[0]
    out = tomle.dumps(doc)
    assert "# one" not in out
    re = tomle.parse(out)
    assert list(cast("tomle.Array", re["arr"])) == [1, 2]


def test_array_set_leading_on_single_line_promotes_to_multiline() -> None:
    doc = tomle.parse("arr = [1, 2, 3]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.leading_comments[1] = ("before two",)
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    re_arr = cast("tomle.Array", re["arr"])
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.leading_comments) == {1: ("before two",)}


def test_array_set_leading_on_first_item() -> None:
    doc = tomle.parse("arr = [1, 2, 3]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.leading_comments[0] = ("first",)
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    re_arr = cast("tomle.Array", re["arr"])
    assert list(re_arr) == [1, 2, 3]
    assert dict(re_arr.leading_comments) == {0: ("first",)}


def test_array_set_multiple_leading_lines() -> None:
    doc = tomle.parse("arr = [\n  1,\n  2,\n]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.leading_comments[1] = ("line one", "line two")
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    re_arr = cast("tomle.Array", re["arr"])
    assert dict(re_arr.leading_comments) == {1: ("line one", "line two")}


def test_array_delete_leading_comments() -> None:
    src = "arr = [\n  # before\n  1,\n  2,\n]\n"
    doc = tomle.parse(src)
    arr = cast("tomle.Array", doc["arr"])
    del arr.leading_comments[0]
    out = tomle.dumps(doc)
    assert "# before" not in out
    re = tomle.parse(out)
    assert list(cast("tomle.Array", re["arr"])) == [1, 2]


def test_array_append_migrates_last_eol_comment() -> None:
    doc = tomle.parse("arr = [\n  1,\n  2 # last\n]\n")
    arr = cast("tomle.Array", doc["arr"])
    assert dict(arr.comments) == {1: "last"}
    arr.append(3)
    out = tomle.dumps(doc)
    re = tomle.parse(out)
    re_arr = cast("tomle.Array", re["arr"])
    assert list(re_arr) == [1, 2, 3]
    # The EOL must still belong to item 1, not item 2.
    assert dict(re_arr.comments) == {1: "last"}


def test_array_comments_view_contains_iter_len() -> None:
    src = "arr = [\n  1, # one\n  2,\n  3, # three\n]\n"
    doc = tomle.parse(src)
    arr = cast("tomle.Array", doc["arr"])
    assert 0 in arr.comments
    assert 1 not in arr.comments
    assert 2 in arr.comments
    assert 99 not in arr.comments
    assert sorted(arr.comments) == [0, 2]
    assert len(arr.comments) == 2


def test_array_comments_view_empty_array() -> None:
    doc = tomle.parse("arr = []\n")
    arr = cast("tomle.Array", doc["arr"])
    assert len(arr.comments) == 0
    assert list(arr.comments) == []
    assert len(arr.leading_comments) == 0
    with pytest.raises(KeyError):
        _ = arr.comments[0]
    with pytest.raises(KeyError):
        _ = arr.leading_comments[0]


def test_array_comments_non_int_key_raises() -> None:
    doc = tomle.parse("arr = [1, 2]\n")
    arr = cast("tomle.Array", doc["arr"])
    with pytest.raises(TypeError):
        _ = arr.comments["x"]  # type: ignore[index]
    with pytest.raises(TypeError):
        arr.comments["x"] = "v"  # type: ignore[index]


def test_array_comments_out_of_range_raises() -> None:
    doc = tomle.parse("arr = [1, 2]\n")
    arr = cast("tomle.Array", doc["arr"])
    with pytest.raises(KeyError):
        arr.comments[5] = "nope"
    with pytest.raises(KeyError):
        arr.leading_comments[5] = ("nope",)
    with pytest.raises(KeyError):
        del arr.comments[5]


def test_array_comment_with_hash_prefix_normalised() -> None:
    doc = tomle.parse("arr = [1, 2]\n")
    arr = cast("tomle.Array", doc["arr"])
    arr.comments[0] = "# already-prefixed"
    re = tomle.parse(tomle.dumps(doc))
    # We don't double-up the `#`.
    assert dict(cast("tomle.Array", re["arr"]).comments) == {0: "already-prefixed"}


def test_array_set_value_via_indexing_preserves_eol_comment() -> None:
    src = "arr = [\n  1, # one\n  2, # two\n]\n"
    doc = tomle.parse(src)
    arr = cast("tomle.Array", doc["arr"])
    arr[0] = 99
    re = tomle.parse(tomle.dumps(doc))
    re_arr = cast("tomle.Array", re["arr"])
    assert list(re_arr) == [99, 2]
    # Comment ownership shouldn't change.
    assert dict(re_arr.comments) == {0: "one", 1: "two"}
