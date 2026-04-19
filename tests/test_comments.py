"""Tests for the comments side-channel and inline-table promotion."""

from __future__ import annotations

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
    assert tomle.dumps(doc) == 'name = "ada"  \n'
    assert "name" not in doc.comments


def test_del_eol_comment_removes_it() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomle.parse(src)
    del doc.comments["name"]
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
