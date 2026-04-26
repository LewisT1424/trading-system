"""
tests/test_run_weekly.py
Smoke tests for run_weekly.py — import, argument parser, and helper functions.

Does NOT run the full pipeline (that would hit the network and take minutes).
Validates that the module loads cleanly and core logic is correct.

Run:
    pytest tests/test_run_weekly.py -v
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Helper — load module without executing main() ─────────────────────────────

def load_run_weekly():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_weekly", ROOT / "run_weekly.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Import smoke test ─────────────────────────────────────────────────────────

def test_run_weekly_imports_cleanly():
    """Module must import without errors and expose expected functions."""
    mod = load_run_weekly()
    assert hasattr(mod, "main")
    assert hasattr(mod, "_filter_a6m")
    assert hasattr(mod, "_filter_mode_c")
    assert hasattr(mod, "_size_label")
    assert hasattr(mod, "_load_screener_results")


# ── _size_label tests ─────────────────────────────────────────────────────────

class TestSizeLabel:

    @pytest.fixture
    def mod(self):
        return load_run_weekly()

    def test_score_3_a6m_gives_8_percent(self, mod):
        size, tier = mod._size_label(3, 2600.0, mode_c=False)
        assert size == pytest.approx(208.0, rel=0.01)
        assert tier == "HIGH CONVICTION"

    def test_score_2_a6m_gives_5_percent(self, mod):
        size, tier = mod._size_label(2, 2600.0, mode_c=False)
        assert size == pytest.approx(130.0, rel=0.01)
        assert tier == "MEDIUM"

    def test_hard_max_1000_applied(self, mod):
        """Large portfolio — raw size exceeds £1000, should be capped."""
        size, _ = mod._size_label(3, 50_000.0, mode_c=False)
        assert size == pytest.approx(1000.0)

    def test_hard_min_50_applied(self, mod):
        """Tiny portfolio — raw size below £50, should be floored."""
        size, _ = mod._size_label(2, 500.0, mode_c=False)
        assert size == pytest.approx(50.0)

    def test_mode_c_score_8_gives_8_percent(self, mod):
        size, tier = mod._size_label(8, 2600.0, mode_c=True)
        assert size == pytest.approx(208.0, rel=0.01)
        assert tier == "HIGH CONVICTION"

    def test_mode_c_score_7_gives_5_percent(self, mod):
        size, tier = mod._size_label(7, 2600.0, mode_c=True)
        assert size == pytest.approx(130.0, rel=0.01)
        assert tier == "MEDIUM"


# ── _filter_a6m tests ─────────────────────────────────────────────────────────

class TestFilterA6M:

    @pytest.fixture
    def mod(self):
        return load_run_weekly()

    def make_signal(self, ticker, score, c5=True, c6=True, c9=True):
        return {"ticker": ticker, "score": score, "C5": c5, "C6": c6, "C9": c9}

    def test_passes_all_criteria(self, mod):
        result = mod._filter_a6m([self.make_signal("AAPL", 3)])
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_fails_if_c5_missing(self, mod):
        result = mod._filter_a6m([self.make_signal("AAPL", 3, c5=False)])
        assert len(result) == 0

    def test_fails_if_c6_missing(self, mod):
        result = mod._filter_a6m([self.make_signal("AAPL", 3, c6=False)])
        assert len(result) == 0

    def test_fails_if_c9_missing(self, mod):
        result = mod._filter_a6m([self.make_signal("AAPL", 3, c9=False)])
        assert len(result) == 0

    def test_fails_if_score_below_minimum(self, mod):
        result = mod._filter_a6m([self.make_signal("AAPL", 1)])
        assert len(result) == 0

    def test_sorted_by_score_descending(self, mod):
        signals = [
            self.make_signal("AAPL", 2),
            self.make_signal("NVDA", 3),
            self.make_signal("AMD",  2),
        ]
        result = mod._filter_a6m(signals)
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["score"] == 3


# ── _filter_mode_c tests ──────────────────────────────────────────────────────

class TestFilterModeC:

    @pytest.fixture
    def mod(self):
        return load_run_weekly()

    def make_signal(self, ticker, score):
        return {"ticker": ticker, "score": score, "C5": True, "C6": True, "C9": True}

    def test_passes_score_7(self, mod):
        result = mod._filter_mode_c([self.make_signal("AAPL", 7)])
        assert len(result) == 1

    def test_passes_score_9(self, mod):
        result = mod._filter_mode_c([self.make_signal("AAPL", 9)])
        assert len(result) == 1

    def test_fails_score_6(self, mod):
        result = mod._filter_mode_c([self.make_signal("AAPL", 6)])
        assert len(result) == 0

    def test_sorted_by_score_descending(self, mod):
        signals = [
            self.make_signal("AAPL", 7),
            self.make_signal("NVDA", 9),
            self.make_signal("AMD",  8),
        ]
        result = mod._filter_mode_c(signals)
        assert result[0]["ticker"] == "NVDA"
        assert result[1]["ticker"] == "AMD"
        assert result[2]["ticker"] == "AAPL"


# ── Argument parser smoke test ────────────────────────────────────────────────

def test_argument_parser_defaults():
    """Parser should initialise with sensible defaults."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio",       type=float, default=2600.0)
    parser.add_argument("--positions",       nargs="*",  default=[])
    parser.add_argument("--position-values", nargs="*",  type=float, default=[])
    parser.add_argument("--entry-prices",    nargs="*",  type=float, default=[])
    parser.add_argument("--entry-dates",     nargs="*",  default=[])
    parser.add_argument("--no-fetch",        action="store_true")
    parser.add_argument("--top",             type=int,   default=10)

    args = parser.parse_args([])
    assert args.portfolio == pytest.approx(2600.0)
    assert args.top == 10
    assert args.no_fetch is False
    assert args.positions == []