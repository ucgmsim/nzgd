"""Per-trace bit-deterministic fingerprint for the dedup hash pass."""

import hashlib
import math
import struct
from typing import Iterable, Sequence

# Fixed 8-byte sentinel for NULL/NaN values. Distinct from any IEEE 754 double
# representation by construction (the leading 4 bytes are zero, the trailing
# 4 are ASCII 'NaN_').
_NULL_SENTINEL = b"\x00\x00\x00\x00NaN_"

# Type discriminator bytes — prefixed before each encoded value so that a
# string representation of a number never collides with the numeric encoding,
# and NULL/NaN never collides with either.
_TYPE_NULL = b"\x00"
_TYPE_NUMERIC = b"\x01"
_TYPE_STRING = b"\x02"


def _encode_value(v) -> bytes:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return _TYPE_NULL + _NULL_SENTINEL
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return _TYPE_NUMERIC + struct.pack("<d", float(v))
    encoded = str(v).encode("utf-8")
    return _TYPE_STRING + struct.pack("<I", len(encoded)) + encoded


def compute_trace_hash(rows: Iterable[Sequence[float | None]]) -> bytes:
    """Compute a 16-byte blake2b digest of a sorted measurement trace.

    `rows` must already be sorted by depth (ascending). Each row is a tuple of
    floats (or `None`/`NaN`) in a fixed column order; the caller chooses the
    order (e.g. depth, qc, fs, u2 for CPT). NaN and NULL are both mapped to a
    fixed 8-byte sentinel so they hash identically. Finite floats are packed
    as little-endian IEEE 754 doubles; no rounding is applied.

    Two traces producing the same digest are byte-identical after this
    normalisation, which is the strongest possible "same data" claim.
    """
    h = hashlib.blake2b(digest_size=16)
    for row in rows:
        for v in row:
            h.update(_encode_value(v))
    return h.digest()
