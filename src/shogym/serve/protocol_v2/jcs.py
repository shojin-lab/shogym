"""Canonical JSON: one byte string per value, so two parties hash the same thing.

Every identity in this protocol is a hash over serialized JSON, which only works if the
serialization is a function of the value and of nothing else. So keys are emitted in UTF-16
code unit order, no whitespace is inserted anywhere, strings use the shortest escape the
grammar allows, and numbers are integers only: this protocol never puts a float on the wire,
and refusing one here keeps the promise instead of inheriting a float printer's rounding.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from shogym.serve.protocol_v2.errors import WireFormatError

# The two-character escapes. Every other control character below 0x20 gets \u00xx, and every
# character at or above 0x20 other than a quote or a backslash is emitted as itself.
_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}
# Integers are serialized as doubles are, so an integer a double cannot hold exactly has no
# canonical form and is refused rather than printed at a length another writer would not pick.
_MAX_EXACT_INTEGER = 2**53 - 1
_MAX_DEPTH = 64


def encode(value: Any) -> bytes:
    """Return the canonical UTF-8 bytes of ``value``."""
    out: List[str] = []
    _write(value, out, 0)
    return "".join(out).encode("utf-8")


def _write(value: Any, out: List[str], depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise WireFormatError("a value nested deeper than this layer serializes")
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        _write_string(value, out)
    elif isinstance(value, float):
        raise WireFormatError("a number must be an integer, and this one is a float")
    elif isinstance(value, int):
        _write_integer(value, out)
    elif isinstance(value, Mapping):
        _write_object(value, out, depth)
    elif isinstance(value, (list, tuple)):
        _write_array(value, out, depth)
    else:
        raise WireFormatError(f"{type(value).__name__} has no canonical JSON encoding")


def _write_string(value: str, out: List[str]) -> None:
    out.append('"')
    for character in value:
        code = ord(character)
        escape = _ESCAPES.get(code)
        if escape is not None:
            out.append(escape)
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        elif 0xD800 <= code <= 0xDFFF:
            raise WireFormatError("a string must be Unicode scalar values, and this one is not")
        else:
            out.append(character)
    out.append('"')


def _write_integer(value: int, out: List[str]) -> None:
    if not -_MAX_EXACT_INTEGER <= value <= _MAX_EXACT_INTEGER:
        raise WireFormatError("an integer this large has no canonical JSON form")
    out.append(str(value))


def _write_object(value: Mapping[Any, Any], out: List[str], depth: int) -> None:
    keys = list(value)
    for key in keys:
        if not isinstance(key, str):
            raise WireFormatError("an object key must be a string")
    # UTF-16 code unit order, which is what the canonical form specifies and is not code point
    # order: a character outside the basic plane sorts by its leading surrogate, so it comes
    # before U+FB33 rather than after it. Encoding to UTF-16 big endian makes the byte
    # comparison the code unit comparison. Surrogates pass through here and are refused by
    # _write_string, which says why.
    keys.sort(key=lambda key: key.encode("utf-16-be", "surrogatepass"))
    out.append("{")
    for index, key in enumerate(keys):
        if index:
            out.append(",")
        _write_string(key, out)
        out.append(":")
        _write(value[key], out, depth + 1)
    out.append("}")


def _write_array(value: Sequence[Any], out: List[str], depth: int) -> None:
    out.append("[")
    for index, item in enumerate(value):
        if index:
            out.append(",")
        _write(item, out, depth + 1)
    out.append("]")
