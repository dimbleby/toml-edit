"""Array views.

`Array(list)` backs an inline-array TOML value (`[1, 2, 3]`). `AoT`
(array-of-tables) is a `list[Table]` whose entries are individually
backed by `AoTEntry` records and whose elements are `Table` views.

Phase 2 surface: read-only access (length, indexing, iteration, value
decoding for plain inline-array elements). Mutation arrives in
Phase 3 (inline) and Phase 4 (AoT and multi-line array).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Self, SupportsIndex

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    from tomlrt._container import Container, Document, Table
    from tomlrt._values import ArrayValue


class Array(list[Any]):
    """An inline array (`[ ... ]`)."""

    __slots__ = ("_multiline", "_value")

    def __init__(self) -> None:
        super().__init__()
        self._value: ArrayValue | None = None
        self._multiline: bool = False

    def to_list(self) -> list[Any]:
        """Materialise a plain-Python ``list`` (recursive)."""
        from tomlrt._container import _to_python  # noqa: PLC0415

        return [_to_python(x) for x in self]

    def array(self, index: int) -> Array:
        """Return ``self[index]`` typed as a nested `Array`."""
        v = self[index]
        if not isinstance(v, Array):
            msg = f"item at {index} is {type(v).__name__}, not an Array"
            raise TypeError(msg)
        return v

    def table(self, index: int) -> Table:
        """Return ``self[index]`` typed as a `Table`."""
        from tomlrt._container import Table  # noqa: PLC0415

        v = self[index]
        if not isinstance(v, Table):
            msg = f"item at {index} is {type(v).__name__}, not a Table"
            raise TypeError(msg)
        return v

    def get_array(self, index: int, default: Any = None) -> Any:
        """Like `array(index)` but returns ``default`` for out-of-range."""
        if index < -len(self) or index >= len(self):
            return default
        return self.array(index)

    def get_table(self, index: int, default: Any = None) -> Any:
        """Like `table(index)` but returns ``default`` for out-of-range."""
        if index < -len(self) or index >= len(self):
            return default
        return self.table(index)


class AoT(list["Table"]):
    """A `[[name]]`-style array of tables.

    Carries minimal logical-attachment state (`_layout_root`, `_path`,
    `_parent`) that survives the empty-AoT case (where `self[-1]`
    cannot be consulted as a fallback parent reference).
    """

    __slots__ = ("_layout_root", "_parent", "_path")

    def __init__(self) -> None:
        super().__init__()
        self._layout_root: Document | None = None
        self._path: tuple[str, ...] = ()
        self._parent: Container | None = None

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise a list of plain-Python ``dict``s (recursive)."""
        return [t.to_dict() for t in self]

    def add(self, body: object | None = None) -> Table:
        """Append a fresh ``[[path]]`` entry and return its `Table` view.

        ``body`` may be a Mapping (initial body content) or ``None``
        (empty entry). The AoT must be attached to a document.
        """
        from tomlrt import _layout_ops  # noqa: PLC0415
        from tomlrt._container import Table  # noqa: PLC0415

        if self._layout_root is None:
            list.append(self, _make_unattached_entry(body))
            return self[-1]
        result = _layout_ops.add_aot_entry(self, body)
        assert isinstance(result, Table)
        return result

    # ------------------------------------------------------------------
    # Supported list-mutator surface (Phase 4-minimal).
    # Anything not implemented here is overridden below to fail closed
    # rather than corrupt the doc-stream via inherited `list` behaviour.
    # ------------------------------------------------------------------

    @override
    def pop(self, index: SupportsIndex = -1) -> Table:
        from tomlrt import _layout_ops  # noqa: PLC0415

        idx = int(index)
        if self._layout_root is None:
            return list.pop(self, idx)
        result = _layout_ops.remove_aot_entry(self, idx)
        from tomlrt._container import Table  # noqa: PLC0415

        assert isinstance(result, Table)
        return result

    @override
    def __delitem__(self, key: SupportsIndex | slice) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if isinstance(key, slice):
            msg = "AoT slice deletion is deferred to a later phase"
            raise NotImplementedError(msg)
        if self._layout_root is None:
            list.__delitem__(self, key)
            return
        _layout_ops.remove_aot_entry(self, int(key))

    @override
    def clear(self) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            list.clear(self)
            return
        while len(self) > 0:
            _layout_ops.remove_aot_entry(self, -1)

    @override
    def __setitem__(  # type: ignore[override]
        self, index: int | slice, value: object
    ) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if isinstance(index, slice):
            msg = "AoT slice assignment is deferred to a later phase"
            raise NotImplementedError(msg)
        if self._layout_root is None:
            assert isinstance(value, dict)
            list.__setitem__(self, index, _make_unattached_entry(value))
            return
        _layout_ops.replace_aot_entry(self, index, value)

    @override
    def append(self, value: Table | dict[str, Any]) -> None:
        # Same semantics as `add(body)` but with no return value (list API).
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            assert isinstance(value, dict)
            list.append(self, _make_unattached_entry(value))
            return
        _layout_ops.add_aot_entry(self, value)

    @override
    def extend(self, values: Any) -> None:
        for v in values:
            self.append(v)

    @override
    def insert(self, index: SupportsIndex, value: Any) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            assert isinstance(value, dict)
            list.insert(self, index, _make_unattached_entry(value))
            return
        # Materialise as add() then renormalise into position.
        new_entry = _layout_ops.add_aot_entry(self, value)
        from tomlrt._container import Table  # noqa: PLC0415

        assert isinstance(new_entry, Table)
        # `add` appended; move into position via renormalise.
        idx = int(index)
        n = len(self)
        if idx < 0:
            idx = max(0, n + idx)
        idx = min(idx, n - 1)
        new_order: list[Table] = list(self)
        new_order.pop()  # remove from tail
        new_order.insert(idx, new_entry)
        if new_order != list(self):
            _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def remove(self, value: Any) -> None:
        for i, t in enumerate(self):
            if t is value or t == value:
                del self[i]
                return
        msg = "AoT.remove(x): x not in AoT"
        raise ValueError(msg)

    @override
    def reverse(self) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        if self._layout_root is None:
            list.reverse(self)
            return
        new_order = list(reversed(self))
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def sort(self, *args: Any, **kwargs: Any) -> None:
        from tomlrt import _layout_ops  # noqa: PLC0415

        new_order = sorted(self, *args, **kwargs)
        if self._layout_root is None:
            list.clear(self)
            for t in new_order:
                list.append(self, t)
            return
        _layout_ops.renormalise_aot_order(self, new_order)

    @override
    def __iadd__(self, other: Any) -> Self:  # type: ignore[override]
        msg = "AoT.__iadd__ is deferred to a later phase"
        raise NotImplementedError(msg)

    @override
    def __imul__(self, n: SupportsIndex) -> Self:
        msg = "AoT.__imul__ is deferred to a later phase"
        raise NotImplementedError(msg)


def _make_unattached_entry(body: object | None) -> Table:
    """Build a fresh unattached `Table` view as an AoT-entry placeholder."""
    from tomlrt._container import Table  # noqa: PLC0415

    t = Table()
    if body is not None:
        assert isinstance(body, dict)
        for k, v in body.items():
            dict.__setitem__(t, k, v)
    return t


__all__ = ["AoT", "Array"]
