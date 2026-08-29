"""The one error this layer raises."""

from __future__ import annotations


class WireFormatError(ValueError):
    """A value does not have the exact form the wire requires.

    Raised on the way in (a payload that is missing a field, carries an unknown one, or holds
    a wrong type, a wrong version, or an out-of-range value) and on the way out (a value with
    no canonical encoding). Nothing is repaired, defaulted, or coerced first, so the caller
    sees the original input, not a partly normalized copy of it.
    """
