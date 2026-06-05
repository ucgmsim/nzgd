"""Tests for alternative LOCA_ID matching strategies in ags_miner.

These tests verify that the fallback matching strategies correctly handle
cases where simple substring matching fails.
"""

import re
from pathlib import Path

import pytest

from nzgd import constants
from nzgd.extract.bh.ags_miner import (
    _collect_loca_ids_and_table_counts,
    _find_matching_loca_from_investigation,
    _try_bh_t_pattern_match,
    _try_bracket_pattern_match,
    _try_context_based_match,
    _try_single_prefix_match,
)
from nzgd.extract.bh.ags_parser import load_ags_tables


class TestSimpleMatching:
    """Test simple substring matching (primary method)."""

    def test_simple_substring_match(self):
        """Test that simple substring matching works for standard cases."""
        loca_ids_with_counts = [("BH434", 2), ("BH430", 1)]
        investigation_id = "City Rail Link Stage 4 [BH434]"
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result == "BH434"


class TestSinglePrefixMatch:
    """Test single prefix matching strategy."""

    def test_single_prefix_match_basic(self):
        """Test single prefix match for case where prefix appears once."""
        loca_ids_with_counts = [("BH01", 1), ("HA1", 1), ("HA2", 1)]
        investigation_id = "91 River Road BH, HAs, DCPs"
        result = _try_single_prefix_match(loca_ids_with_counts, investigation_id)
        assert result == "BH01"

    def test_single_prefix_match_integration(self):
        """Test single prefix match through main function."""
        loca_ids_with_counts = [("BH01", 1), ("HA1", 1), ("HA2", 1)]
        investigation_id = "91 River Road BH, HAs, DCPs"
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result == "BH01"

    def test_single_prefix_match_multiple_prefixes(self):
        """Test that single prefix match matches when one prefix appears once."""
        loca_ids_with_counts = [("BH01", 1), ("BH02", 1), ("HA1", 1)]
        investigation_id = "91 River Road BH, HAs, DCPs"
        result = _try_single_prefix_match(loca_ids_with_counts, investigation_id)
        # "HA" appears once, so should match HA1 (BH appears twice, so not matched)
        assert result == "HA1"
    
    def test_single_prefix_match_no_single_prefix(self):
        """Test that single prefix match returns None when all prefixes appear multiple times."""
        loca_ids_with_counts = [("BH01", 1), ("BH02", 1), ("HA1", 1), ("HA2", 1)]
        investigation_id = "91 River Road BH, HAs, DCPs"
        result = _try_single_prefix_match(loca_ids_with_counts, investigation_id)
        # Both BH and HA appear multiple times, so should return None
        assert result is None


class TestBHTPatternMatch:
    """Test BH-t pattern matching strategy."""

    def test_bh_t_pattern_match_basic(self):
        """Test BH-t pattern matching extracts number correctly."""
        loca_ids_with_counts = [("BH2", 1), ("BH5", 1)]
        investigation_id = "BH-t2"
        result = _try_bh_t_pattern_match(loca_ids_with_counts, investigation_id)
        assert result == "BH2"

    def test_bh_t_pattern_match_integration(self):
        """Test BH-t pattern match through main function."""
        loca_ids_with_counts = [("BH2", 1), ("BH5", 1)]
        investigation_id = "BH-t2"
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result == "BH2"

    def test_bh_t_pattern_match_bh5(self):
        """Test BH-t pattern matching for BH-t5 case."""
        loca_ids_with_counts = [("BH2", 1), ("BH5", 1)]
        investigation_id = "BH-t5"
        result = _try_bh_t_pattern_match(loca_ids_with_counts, investigation_id)
        assert result == "BH5"

    def test_bh_t_pattern_match_no_match(self):
        """Test BH-t pattern returns None when no pattern found."""
        loca_ids_with_counts = [("BH2", 1), ("BH5", 1)]
        investigation_id = "No BH-t pattern here"
        result = _try_bh_t_pattern_match(loca_ids_with_counts, investigation_id)
        assert result is None


class TestBracketPatternMatch:
    """Test bracket pattern matching strategy."""

    def test_bracket_pattern_match_basic(self):
        """Test bracket pattern matching with prefix and number."""
        loca_ids_with_counts = [("MF_Boring3", 2), ("MF_Boring4", 2)]
        investigation_id = "Michael Fowler Centre [MF_BH3]"
        result = _try_bracket_pattern_match(loca_ids_with_counts, investigation_id)
        assert result == "MF_Boring3"

    def test_bracket_pattern_match_integration(self):
        """Test bracket pattern match through main function."""
        loca_ids_with_counts = [("MF_Boring3", 2), ("MF_Boring4", 2)]
        investigation_id = "Michael Fowler Centre [MF_BH3]"
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result == "MF_Boring3"

    def test_bracket_pattern_match_multiple_numbers(self):
        """Test bracket pattern matching with multiple LOCA_IDs having same number."""
        loca_ids_with_counts = [
            ("CSD_Bore3", 1),
            ("MF_Boring3", 2),
            ("TH_BH03", 1),
        ]
        investigation_id = "Michael Fowler Centre [MF_BH3]"
        result = _try_bracket_pattern_match(loca_ids_with_counts, investigation_id)
        # Should prefer MF prefix
        assert result == "MF_Boring3"

    def test_bracket_pattern_match_no_bracket(self):
        """Test bracket pattern returns None when no bracket found."""
        loca_ids_with_counts = [("MF_Boring3", 2)]
        investigation_id = "No bracket here"
        result = _try_bracket_pattern_match(loca_ids_with_counts, investigation_id)
        assert result is None


class TestContextBasedMatch:
    """Test context-based matching strategy."""

    def test_context_based_match_town_hall(self):
        """Test context-based matching for Town Hall case."""
        loca_ids_with_counts = [
            ("CSD_Bore2", 1),
            ("TH_BH01", 1),
            ("TH_BH02", 1),
            ("VS_BH2", 1),
        ]
        investigation_id = "Wellington Town Hall GENZWELL16138AA [Aurecon-BH02]"
        result = _try_context_based_match(loca_ids_with_counts, investigation_id)
        assert result == "TH_BH02"

    def test_context_based_match_integration(self):
        """Test context-based match through main function."""
        loca_ids_with_counts = [
            ("CSD_Bore2", 1),
            ("TH_BH01", 1),
            ("TH_BH02", 1),
            ("VS_BH2", 1),
        ]
        investigation_id = "Wellington Town Hall GENZWELL16138AA [Aurecon-BH02]"
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result == "TH_BH02"

    def test_context_based_match_no_context(self):
        """Test context-based match returns None when no context keyword found."""
        loca_ids_with_counts = [("TH_BH02", 1)]
        investigation_id = "No context keywords here [BH02]"
        result = _try_context_based_match(loca_ids_with_counts, investigation_id)
        assert result is None


class TestRealWorldCases:
    """Test with actual AGS files from problematic cases."""

    @pytest.mark.parametrize(
        "nzgd_id,investigation_id,expected_match",
        [
            (31205, "91 River Road BH, HAs, DCPs", "BH01"),
            (97660, "BH-t2", "BH2"),
            (99137, "BH-t5", "BH5"),
            (124644, "Michael Fowler Centre GENZWELL16138AA [MF_BH3]", "MF_Boring3"),
            (124645, "Michael Fowler Centre GENZWELL16138AA [MF_BH4]", "MF_Boring4"),
            (124646, "Michael Fowler Centre GENZWELL16138AA [MF_BH5]", "MF_Boring5"),
            (
                124647,
                "Wellington Town Hall GENZWELL16138AA [Aurecon-BH02]",
                "TH_BH02",
            ),
        ],
    )
    def test_real_world_cases(
        self, nzgd_id: int, investigation_id: str, expected_match: str
    ):
        """Test matching with actual AGS files from problematic cases."""
        nzgd_dir = constants.NZGD_SOURCE_DATA_DIR / str(nzgd_id)
        ags_files = list(nzgd_dir.glob("*.ags"))

        if not ags_files:
            pytest.skip(f"No AGS file found for NZGD ID {nzgd_id}")

        tables, _headings = load_ags_tables(ags_files[0])
        loca_ids_with_counts = _collect_loca_ids_and_table_counts(tables)

        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )

        assert result == expected_match, (
            f"NZGD {nzgd_id}: Expected {expected_match}, got {result}"
        )


class TestStrategyOrder:
    """Test that strategies are tried in the correct order."""

    def test_simple_match_takes_priority(self):
        """Test that simple substring match is tried first."""
        # Both simple match and single prefix match would work
        loca_ids_with_counts = [("BH434", 2), ("BH01", 1)]
        investigation_id = "City Rail Link Stage 4 [BH434] BH"
        # Simple match should win
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result == "BH434"

    def test_fallback_order(self):
        """Test that fallback strategies are tried in correct order."""
        # No simple match, but single prefix should work
        loca_ids_with_counts = [("BH01", 1), ("HA1", 1)]
        investigation_id = "91 River Road BH"
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result == "BH01"

    def test_no_match_returns_none(self):
        """Test that None is returned when no strategy matches."""
        loca_ids_with_counts = [("UNKNOWN1", 1), ("UNKNOWN2", 1)]
        investigation_id = "No matching pattern here"
        result = _find_matching_loca_from_investigation(
            loca_ids_with_counts, investigation_id
        )
        assert result is None

    def test_none_investigation_id(self):
        """Test that None investigation_id returns None."""
        loca_ids_with_counts = [("BH01", 1)]
        result = _find_matching_loca_from_investigation(loca_ids_with_counts, None)
        assert result is None

    def test_empty_investigation_id(self):
        """Test that empty investigation_id returns None."""
        loca_ids_with_counts = [("BH01", 1)]
        result = _find_matching_loca_from_investigation(loca_ids_with_counts, "")
        assert result is None

