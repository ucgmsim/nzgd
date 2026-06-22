"""Tests for atomic CSV-gz writing."""

import pandas as pd

from nzgd.metadata.io_utils import atomic_write_csv_gz


def test_atomic_write_csv_gz_roundtrip(tmp_path):
    """A written file reads back equal to the input frame."""
    df = pd.DataFrame({"nzgd_id": [1, 2], "region": ["Canterbury", "unclassified"]})
    out = tmp_path / "x.csv.gz"
    atomic_write_csv_gz(df, out)
    assert pd.read_csv(out).equals(df)


def test_atomic_write_csv_gz_overwrites(tmp_path):
    """Writing over an existing file replaces it, leaving no temp file behind."""
    out = tmp_path / "x.csv.gz"
    atomic_write_csv_gz(pd.DataFrame({"a": [1]}), out)
    atomic_write_csv_gz(pd.DataFrame({"a": [2]}), out)
    assert pd.read_csv(out)["a"].tolist() == [2]
    assert list(tmp_path.iterdir()) == [out]
