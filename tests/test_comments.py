"""Tests for the comment API and inline-table promotion."""

from __future__ import annotations

import pytest

import tomle


def test_get_eol_comment_returns_text_without_marker() -> None:
    src = 'name = "ada"  # the lovelace\n'
    doc = tomle.parse(src)
    assert doc.comment("name") == "the lovelace"


def test_get_eol_comment_returns_none_when_absent() -> None:
    src = 'name = "ada"\n'
    doc = tomle.parse(src)
    assert doc.comment("name") is None


def test_get_eol_comment_unknown_key_raises_keyerror() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(KeyError):
        doc.comment("missing")


def test_set_eol_comment_on_uncommented_key() -> None:
    src = 'name = "ada"\n'
    doc = tomle.parse(src)
    doc.set_comment("name", "the lovelace")
    assert tomle.dumps(doc) == 'name = "ada" # the lovelace\n'
    assert doc.comment("name") == "the lovelace"


def test_set_eol_comment_replaces_existing() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomle.parse(src)
    doc.set_comment("name", "new")
    out = tomle.dumps(doc)
    # Original double-space trailing run is preserved; only the
    # comment payload changes.
    assert out == 'name = "ada"  # new\n'


def test_set_eol_comment_none_removes_existing() -> None:
    src = 'name = "ada"  # old\n'
    doc = tomle.parse(src)
    doc.set_comment("name", None)
    assert tomle.dumps(doc) == 'name = "ada"  \n'
    assert doc.comment("name") is None


def test_set_eol_comment_accepts_text_with_hash_prefix() -> None:
    doc = tomle.parse("a = 1\n")
    doc.set_comment("a", "## emphasised")
    assert tomle.dumps(doc) == "a = 1 ## emphasised\n"


def test_set_eol_comment_rejects_newline() -> None:
    doc = tomle.parse("a = 1\n")
    with pytest.raises(tomle.TOMLEditError):
        doc.set_comment("a", "no\nway")


def test_get_leading_comments_returns_block_above_key() -> None:
    src = "# a section\n# of two lines\nname = 1\n"
    doc = tomle.parse(src)
    assert doc.leading_comments("name") == ["a section", "of two lines"]


def test_set_leading_comments_replaces_existing_block() -> None:
    src = "# old comment\nname = 1\n"
    doc = tomle.parse(src)
    doc.set_leading_comments("name", ["fresh", "block"])
    assert tomle.dumps(doc) == "# fresh\n# block\nname = 1\n"


def test_set_leading_comments_to_empty_clears_block() -> None:
    src = "# noisy\n# preamble\nname = 1\n"
    doc = tomle.parse(src)
    doc.set_leading_comments("name", [])
    assert tomle.dumps(doc) == "name = 1\n"


def test_leading_comments_preserve_indent_in_subtable() -> None:
    src = "[tbl]\n    # explanation\n    x = 1\n"
    doc = tomle.parse(src)
    tbl = doc["tbl"]
    assert isinstance(tbl, tomle.Table)
    assert tbl.leading_comments("x") == ["explanation"]
    tbl.set_leading_comments("x", ["replaced"])
    assert tomle.dumps(doc) == "[tbl]\n    # replaced\n    x = 1\n"


def test_comment_api_on_inline_table_raises_with_helpful_message() -> None:
    src = 'pkg = { name = "tomle", version = "0.1" }\n'
    doc = tomle.parse(src)
    pkg = doc["pkg"]
    assert isinstance(pkg, tomle.Table)
    with pytest.raises(tomle.TOMLEditError, match="comment API"):
        pkg.set_comment("name", "anything")


def test_inline_table_promotion_basic() -> None:
    src = 'pkg = { name = "tomle", version = "0.1" }\n'
    doc = tomle.parse(src)
    promoted = doc.promote_inline("pkg")
    assert isinstance(promoted, tomle.Table)
    assert promoted["name"] == "tomle"
    assert promoted["version"] == "0.1"
    out = tomle.dumps(doc)
    assert out == '[pkg]\nname = "tomle"\nversion = "0.1"\n'


def test_inline_table_promotion_preserves_leading_comments() -> None:
    src = '# the package\npkg = { name = "tomle" }\n'
    doc = tomle.parse(src)
    doc.promote_inline("pkg")
    out = tomle.dumps(doc)
    assert out == '# the package\n[pkg]\nname = "tomle"\n'


def test_inline_table_promotion_preserves_eol_comment_on_header() -> None:
    src = 'pkg = { name = "tomle" }  # describes pkg\n'
    doc = tomle.parse(src)
    doc.promote_inline("pkg")
    out = tomle.dumps(doc)
    assert out == '[pkg]  # describes pkg\nname = "tomle"\n'


def test_inline_promotion_then_set_comment_on_member() -> None:
    src = 'pkg = { name = "tomle", version = "0.1" }\n'
    doc = tomle.parse(src)
    promoted = doc.promote_inline("pkg")
    promoted.set_comment("version", "calver soon")
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
    out = tomle.dumps(doc)
    assert out == (
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


# Note: the "target [path] section already exists" branch in
# promote_inline is defensive — the parser refuses any source that
# would let an inline-table value coexist with a same-path standard
# table, and the mutation API blocks creating one after the fact.
# The check stays in place to guard against future internal
# manipulation that bypasses these gates.


def test_round_trip_after_set_comment_and_set_leading() -> None:
    src = "x = 1\n"
    doc = tomle.parse(src)
    doc.set_comment("x", "trailing")
    doc.set_leading_comments("x", ["above"])
    out = tomle.dumps(doc)
    assert out == "# above\nx = 1 # trailing\n"
    # Re-parse should preserve everything verbatim.
    again = tomle.parse(out)
    assert again.comment("x") == "trailing"
    assert again.leading_comments("x") == ["above"]
