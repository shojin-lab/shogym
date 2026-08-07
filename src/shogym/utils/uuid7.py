"""Minimal UUIDv7 (RFC 9562) generator.

The serve layer mints one per episode as the `_session_id` that keys
per-episode state on stateful MCP servers. Time-ordered, so session ids sort
by creation time. Local implementation because the stdlib gains `uuid.uuid7`
only in Python 3.14.
"""

import os
import time
import uuid


def uuid7() -> uuid.UUID:
    """Return a new UUIDv7: 48-bit unix-ms timestamp + 74 random bits."""
    unix_ts_ms = time.time_ns() // 1_000_000
    rand_a = int.from_bytes(os.urandom(2)) & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8)) & 0x3FFF_FFFF_FFFF_FFFF
    value = (unix_ts_ms & 0xFFFF_FFFF_FFFF) << 80
    value |= 0x7 << 76  # version 7
    value |= rand_a << 64
    value |= 0b10 << 62  # RFC 9562 variant
    value |= rand_b
    return uuid.UUID(int=value)
