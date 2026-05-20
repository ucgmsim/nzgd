"""Per-trace bit-deterministic fingerprint for the dedup hash pass."""

import hashlib
import math
import struct
from typing import Iterable, Sequence

# Fixed 8-byte sentinel for NULL/NaN values. Distinct from any IEEE 754 double
# representation by construction (the leading 4 bytes are zero, the trailing
# 4 are ASCII 'NaN_').
_NULL_SENTINEL = b"\x00\x00\x00\x00NaN_"


def _encode_value(v: float | None) -> bytes:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return _NULL_SENTINEL
    return struct.pack("<d", float(v))


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
