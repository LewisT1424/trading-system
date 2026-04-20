"""
tests/test_sizer.py
Production-grade tests for risk/sizer.py

Test categories:
    1.  Initialisation — correct cash and deployed calculations
    2.  Score allocation — correct £ amounts at known portfolio sizes
    3.  Hard max — £1,000 cap never exceeded
    4.  Hard min — £50 minimum enforced
    5.  Cash reserve — 10% always held back
    6.  No pyramiding — same ticker blocked
    7.  Max positions — 10 position hard limit
    8.  Cash constrained — reduced size when cash is tight
    9.  Insufficient cash — blocked when below minimum
    10. Invalid score — rejected cleanly
    11. SizeResult fields — correct values on every path
    12. Review portfolio — trim flags, review flags, clean state
    13. Constants — pinned so accidental changes are caught
    14. Edge cases — zero positions, full deployment, exact boundary values
    15. Known values — your actual portfolio (£2,600 ISA)

Run:
    pytest tests/test_sizer.py -v
    pytest tests/test_sizer.py::TestScoreAllocation -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "risk"))

from sizer import (
    Sizer,
    SizeResult,
    SCORE_ALLOCATION,
    MAX_POSITIONS,
    CASH_RESERVE_PCT,
    MAX_POSITION_PCT,
    HARD_MAX_GBP,
    HARD_MIN_GBP,
    STOP_LOSS_PCT,
)


# ── 1. Constants — pinned ─────────────────────────────────────────────────────

class TestConstants:
    """
    Pin every constant so accidental changes are caught immediately.
    These values are load-bearing — changing them changes real money decisions.
    """

    def test_score_3_allocation_is_8_pct(self):
        assert SCORE_ALLOCATION[3] == 0.08, (
            "Score 3 (high conviction) must be 8% of portfolio. "
            "Changing this changes every position size."
        )

    def test_score_2_allocation_is_5_pct(self):
        assert SCORE_ALLOCATION[2] == 0.05, (
            "Score 2 (medium conviction) must be 5% of portfolio."
        )

    def test_only_scores_2_and_3_defined(self):
        assert set(SCORE_ALLOCATION.keys()) == {2, 3}, (
            "A_6M only produces score 2 or 3. No other scores should be defined."
        )

    def test_max_positions_is_10(self):
        assert MAX_POSITIONS == 10

    def test_cash_reserve_is_10_pct(self):
        assert CASH_RESERVE_PCT == 0.10

    def test_max_position_pct_is_10_pct(self):
        assert MAX_POSITION_PCT == 0.10

    def test_hard_max_is_1000(self):
        assert HARD_MAX_GBP == 1_000

    def test_hard_min_is_50(self):
        assert HARD_MIN_GBP == 50

    def test_stop_loss_is_minus_30(self):
        assert STOP_LOSS_PCT == -30.0


# ── 2. Initialisation ─────────────────────────────────────────────────────────

class TestInitialisation:
    """Verify Sizer correctly computes cash, deployed, and free cash on init."""

    def test_no_positions_all_cash(self):
        s = Sizer(portfolio_value=3000)
        assert s.deployed == 0.0
        assert s.cash == 3000.0
        assert s.cash_reserve == 300.0
        assert s.cash_free == 2700.0
        assert s.n_open == 0

    def test_deployed_equals_sum_of_positions(self):
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 240, "NVDA": 150})
        assert s.deployed == 390.0

    def test_cash_equals_portfolio_minus_deployed(self):
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 240, "NVDA": 150})
        assert s.cash == 3000.0 - 390.0

    def test_cash_free_deducts_reserve(self):
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 240})
        # cash = 3000 - 240 = 2760
        # reserve = 3000 * 0.10 = 300
        # free = 2760 - 300 = 2460
        assert abs(s.cash_free - 2460.0) < 0.01

    def test_cash_free_never_negative(self):
        """Even if fully deployed, cash_free must be 0 not negative."""
        # Deploy 95% of portfolio — cash = 150, reserve = 300, free should be 0
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 2850})
        assert s.cash_free == 0.0

    def test_n_open_matches_position_count(self):
        s = Sizer(portfolio_value=3000, open_positions={"A": 100, "B": 100, "C": 100})
        assert s.n_open == 3

    def test_none_positions_treated_as_empty(self):
        s = Sizer(portfolio_value=3000, open_positions=None)
        assert s.n_open == 0
        assert s.deployed == 0.0

    def test_raises_on_zero_portfolio(self):
        with pytest.raises(ValueError):
            Sizer(portfolio_value=0)

    def test_raises_on_negative_portfolio(self):
        with pytest.raises(ValueError):
            Sizer(portfolio_value=-100)


# ── 3. Score allocation — known values ───────────────────────────────────────

class TestScoreAllocation:
    """
    Verify the exact £ amounts produced at known portfolio sizes.
    These are the numbers that determine real trade sizes.
    """

    def test_score_3_at_2600_portfolio(self):
        """Your actual portfolio: £2,600 * 8% = £208, rounded to £210."""
        s = Sizer(portfolio_value=2600)
        result = s.size("NVDA", score=3)
        assert result.can_trade is True
        # 2600 * 0.08 = 208 → rounded to nearest £10 = £210
        assert result.recommended_size == 210.0

    def test_score_2_at_2600_portfolio(self):
        """£2,600 * 5% = £130, rounded to £130."""
        s = Sizer(portfolio_value=2600)
        result = s.size("NVDA", score=2)
        assert result.can_trade is True
        assert result.recommended_size == 130.0

    def test_score_3_at_3000_portfolio(self):
        """£3,000 * 8% = £240."""
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=3)
        assert result.recommended_size == 240.0

    def test_score_2_at_3000_portfolio(self):
        """£3,000 * 5% = £150."""
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=2)
        assert result.recommended_size == 150.0

    def test_score_3_at_10000_portfolio(self):
        """£10,000 * 8% = £800."""
        s = Sizer(portfolio_value=10000)
        result = s.size("NVDA", score=3)
        assert result.recommended_size == 800.0

    def test_score_2_at_10000_portfolio(self):
        """£10,000 * 5% = £500."""
        s = Sizer(portfolio_value=10000)
        result = s.size("NVDA", score=2)
        assert result.recommended_size == 500.0

    def test_pct_of_portfolio_correct_for_score_3(self):
        """pct_of_portfolio should reflect the actual recommended size."""
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=3)
        expected_pct = result.recommended_size / 3000 * 100
        assert abs(result.pct_of_portfolio - expected_pct) < 0.01

    def test_score_3_larger_than_score_2(self):
        """Score 3 must always produce a larger position than score 2."""
        s = Sizer(portfolio_value=3000)
        r3 = s.size("NVDA", score=3)
        r2 = s.size("NVDA", score=2)
        assert r3.recommended_size > r2.recommended_size


# ── 4. Hard max cap ───────────────────────────────────────────────────────────

class TestHardMax:
    """£1,000 hard cap must never be exceeded regardless of portfolio size."""

    def test_score_3_capped_at_1000_on_large_portfolio(self):
        """£20,000 * 8% = £1,600 — must be capped at £1,000."""
        s = Sizer(portfolio_value=20000)
        result = s.size("NVDA", score=3)
        assert result.recommended_size <= HARD_MAX_GBP, (
            f"Hard max violated: £{result.recommended_size} > £{HARD_MAX_GBP}"
        )

    def test_score_2_capped_at_1000_on_large_portfolio(self):
        """£25,000 * 5% = £1,250 — must be capped at £1,000."""
        s = Sizer(portfolio_value=25000)
        result = s.size("NVDA", score=2)
        assert result.recommended_size <= HARD_MAX_GBP

    def test_hard_max_exact_boundary(self):
        """At exactly £12,500 portfolio, score 3 = £1,000 (8% = £1,000)."""
        s = Sizer(portfolio_value=12500)
        result = s.size("NVDA", score=3)
        assert result.recommended_size == 1000.0

    def test_hard_max_applies_at_any_scale(self):
        """Test multiple large portfolios — cap always enforced."""
        for pv in [15000, 50000, 100000, 500000]:
            s = Sizer(portfolio_value=pv)
            r = s.size("NVDA", score=3)
            assert r.recommended_size <= HARD_MAX_GBP, (
                f"Hard max violated at portfolio £{pv}: got £{r.recommended_size}"
            )


# ── 5. Hard min — below £50 blocked ──────────────────────────────────────────

class TestHardMin:
    """Positions below £50 must be blocked — not worth the friction."""

    def test_score_2_blocked_below_min_on_tiny_portfolio(self):
        """£800 * 5% = £40 — below £50 minimum, must block."""
        s = Sizer(portfolio_value=800)
        result = s.size("NVDA", score=2)
        assert result.can_trade is False, (
            "Position of £40 is below £50 minimum — should be blocked"
        )

    def test_score_3_blocked_below_min_on_tiny_portfolio(self):
        """£500 * 8% = £40 — below £50 minimum."""
        s = Sizer(portfolio_value=500)
        result = s.size("NVDA", score=3)
        assert result.can_trade is False

    def test_size_at_exact_minimum_boundary(self):
        """
        £625 * 8% = £50 exactly — at the minimum, should be tradeable.
        Rounded to nearest £10 = £50.
        """
        s = Sizer(portfolio_value=625)
        result = s.size("NVDA", score=3)
        # 625 * 0.08 = 50.0 → rounded to £50 → at minimum → can trade
        assert result.recommended_size >= HARD_MIN_GBP

    def test_reason_mentions_minimum_when_blocked(self):
        """Error message must explain why the trade is blocked."""
        s = Sizer(portfolio_value=500)
        result = s.size("NVDA", score=3)
        assert "minimum" in result.reason.lower() or "small" in result.reason.lower()


# ── 6. Cash reserve — 10% always held ────────────────────────────────────────

class TestCashReserve:
    """The 10% cash reserve must always be protected from deployment."""

    def test_reserve_not_available_for_new_positions(self):
        """
        With £3,000 portfolio and no positions:
        cash = £3,000
        reserve = £300
        free = £2,700
        A £2,800 position must be blocked — it would dip into the reserve.
        """
        s = Sizer(portfolio_value=3000)
        # Free cash is £2,700. A position of £2,800 exceeds free cash.
        # The sizer should reduce to £2,700 (cash constrained) not block entirely.
        result = s.size("NVDA", score=3)
        # Normal score 3 = £240, well within £2,700 free — should be tradeable
        assert result.can_trade is True
        assert result.recommended_size <= s.cash_free

    def test_free_cash_respects_reserve(self):
        """cash_free must never exceed (cash - reserve)."""
        for pv in [1000, 2600, 5000, 10000]:
            s = Sizer(portfolio_value=pv)
            expected_free = max(0, pv - pv * CASH_RESERVE_PCT)
            assert abs(s.cash_free - expected_free) < 0.01, (
                f"At £{pv} portfolio: expected free cash £{expected_free:.0f}, "
                f"got £{s.cash_free:.0f}"
            )

    def test_reserve_scales_with_portfolio(self):
        """Reserve must be 10% of current portfolio value."""
        for pv in [2600, 5000, 10000, 50000]:
            s = Sizer(portfolio_value=pv)
            assert abs(s.cash_reserve - pv * 0.10) < 0.01


# ── 7. No pyramiding ─────────────────────────────────────────────────────────

class TestNoPyramiding:
    """The same ticker cannot be opened twice — no pyramiding allowed."""

    def test_already_held_ticker_blocked(self):
        s = Sizer(portfolio_value=3000, open_positions={"NVDA": 240})
        result = s.size("NVDA", score=3)
        assert result.can_trade is False, (
            "NVDA already held — second entry must be blocked"
        )

    def test_already_held_ticker_blocked_regardless_of_score(self):
        s = Sizer(portfolio_value=3000, open_positions={"NVDA": 240})
        for score in [2, 3]:
            result = s.size("NVDA", score=score)
            assert result.can_trade is False

    def test_different_ticker_allowed_when_one_held(self):
        s = Sizer(portfolio_value=3000, open_positions={"NVDA": 240})
        result = s.size("AAPL", score=3)
        assert result.can_trade is True

    def test_pyramiding_reason_in_message(self):
        s = Sizer(portfolio_value=3000, open_positions={"NVDA": 240})
        result = s.size("NVDA", score=3)
        assert "pyramid" in result.reason.lower() or "already" in result.reason.lower()

    def test_size_is_zero_when_blocked_for_pyramiding(self):
        s = Sizer(portfolio_value=3000, open_positions={"NVDA": 240})
        result = s.size("NVDA", score=3)
        assert result.recommended_size == 0.0


# ── 8. Max positions hard limit ───────────────────────────────────────────────

class TestMaxPositions:
    """No new positions when 10 are already open."""

    def test_blocked_at_max_positions(self):
        positions = {f"TICK{i}": 200 for i in range(10)}
        s = Sizer(portfolio_value=5000, open_positions=positions)
        result = s.size("NEW", score=3)
        assert result.can_trade is False, (
            f"10 positions already open — 11th must be blocked"
        )

    def test_allowed_at_9_positions(self):
        positions = {f"TICK{i}": 200 for i in range(9)}
        s = Sizer(portfolio_value=10000, open_positions=positions)
        result = s.size("NEW", score=3)
        assert result.can_trade is True

    def test_max_positions_reason_in_message(self):
        positions = {f"TICK{i}": 200 for i in range(10)}
        s = Sizer(portfolio_value=5000, open_positions=positions)
        result = s.size("NEW", score=3)
        assert "max" in result.reason.lower() or "10" in result.reason

    def test_n_open_correct_at_max(self):
        positions = {f"TICK{i}": 200 for i in range(10)}
        s = Sizer(portfolio_value=5000, open_positions=positions)
        assert s.n_open == 10


# ── 9. Cash constrained — reduced size ───────────────────────────────────────

class TestCashConstrained:
    """
    When free cash is below the target size but above the minimum,
    the sizer should reduce the position to what's available — not block entirely.
    """

    def test_reduced_size_when_cash_constrained(self):
        """
        Portfolio: £3,000
        Deployed: £2,500 (positions near full)
        Cash: £500
        Reserve: £300
        Free: £200
        Score 3 target: £240 — exceeds £200 free
        Should return £200 (cash constrained), not block
        """
        s = Sizer(portfolio_value=3000, open_positions={"A": 2500})
        result = s.size("NVDA", score=3)
        assert result.can_trade is True, (
            "£200 free cash is above £50 minimum — should trade with reduced size"
        )
        assert result.recommended_size <= s.cash_free + 1  # allow £1 rounding

    def test_reduced_size_less_than_target(self):
        """Reduced size must be less than the full target size."""
        s = Sizer(portfolio_value=3000, open_positions={"A": 2500})
        full_target = 3000 * SCORE_ALLOCATION[3]
        result = s.size("NVDA", score=3)
        if result.can_trade:
            assert result.recommended_size <= full_target

    def test_reduced_reason_mentions_constraint(self):
        """Reason must explain the cash constraint."""
        s = Sizer(portfolio_value=3000, open_positions={"A": 2500})
        result = s.size("NVDA", score=3)
        if result.can_trade and result.recommended_size < 3000 * SCORE_ALLOCATION[3]:
            assert "constrained" in result.reason.lower() or "reduced" in result.reason.lower()

    def test_blocked_when_free_cash_below_minimum(self):
        """
        Free cash below £50 minimum — must block, not return tiny position.
        Portfolio: £3,000
        Deployed: £2,960 (nearly all deployed)
        Cash: £40
        Reserve: £300
        Free: max(0, 40 - 300) = 0
        """
        s = Sizer(portfolio_value=3000, open_positions={"A": 2960})
        result = s.size("NVDA", score=3)
        assert result.can_trade is False


# ── 10. Invalid score ─────────────────────────────────────────────────────────

class TestInvalidScore:
    """Scores other than 2 or 3 must be rejected cleanly."""

    def test_score_1_blocked(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=1)
        assert result.can_trade is False

    def test_score_4_blocked(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=4)
        assert result.can_trade is False

    def test_score_0_blocked(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=0)
        assert result.can_trade is False

    def test_invalid_score_reason_mentions_valid_scores(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=5)
        assert "2" in result.reason and "3" in result.reason

    def test_invalid_score_returns_zero_size(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=99)
        assert result.recommended_size == 0.0


# ── 11. SizeResult fields ─────────────────────────────────────────────────────

class TestSizeResultFields:
    """Every field of SizeResult must be correctly populated on all paths."""

    def test_ticker_preserved_in_result(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=3)
        assert result.ticker == "NVDA"

    def test_score_preserved_in_result(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=3)
        assert result.score == 3

    def test_portfolio_value_preserved(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=3)
        assert result.portfolio_value == 3000.0

    def test_cash_available_matches_sizer(self):
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 240})
        result = s.size("NVDA", score=3)
        assert result.cash_available == s.cash_free

    def test_n_open_positions_matches_sizer(self):
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 240, "AMD": 200})
        result = s.size("NVDA", score=3)
        assert result.n_open_positions == 2

    def test_can_trade_true_returns_positive_size(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=3)
        if result.can_trade:
            assert result.recommended_size > 0

    def test_can_trade_false_returns_zero_size_on_pyramiding(self):
        s = Sizer(portfolio_value=3000, open_positions={"NVDA": 240})
        result = s.size("NVDA", score=3)
        assert result.can_trade is False
        assert result.recommended_size == 0.0

    def test_str_method_does_not_crash(self):
        """__str__ must work on every result path without raising."""
        s = Sizer(portfolio_value=3000)
        for score in [2, 3]:
            result = s.size("NVDA", score=score)
            output = str(result)
            assert len(output) > 0

    def test_str_contains_ticker(self):
        s = Sizer(portfolio_value=3000)
        result = s.size("NVDA", score=3)
        assert "NVDA" in str(result)

    def test_str_contains_trade_or_skip(self):
        s = Sizer(portfolio_value=3000)
        output = str(s.size("NVDA", score=3))
        assert "TRADE" in output or "SKIP" in output


# ── 12. Review portfolio ──────────────────────────────────────────────────────

class TestReviewPortfolio:
    """Verify review_portfolio correctly identifies trim and review flags."""

    def test_position_above_10_pct_flagged_trim(self, capsys):
        """Position at 15% of portfolio must be flagged TRIM."""
        s = Sizer(portfolio_value=3000, open_positions={"INRG": 450})
        s.review_portfolio()
        output = capsys.readouterr().out
        assert "TRIM" in output, "15% position should be flagged TRIM"

    def test_position_below_minimum_flagged_review(self, capsys):
        """Position below £50 must be flagged REVIEW."""
        s = Sizer(portfolio_value=3000, open_positions={"MSTR": 37})
        s.review_portfolio()
        output = capsys.readouterr().out
        assert "REVIEW" in output, "£37 position should be flagged REVIEW"

    def test_all_positions_within_limits_shows_ok(self, capsys):
        """Clean portfolio must show all OK."""
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 240, "NVDA": 150})
        s.review_portfolio()
        output = capsys.readouterr().out
        assert "TRIM" not in output
        assert "All positions within limits" in output or "OK" in output

    def test_trim_recommendation_includes_target_amount(self, capsys):
        """Trim output must state what to trim to."""
        s = Sizer(portfolio_value=3000, open_positions={"INRG": 600})
        s.review_portfolio()
        output = capsys.readouterr().out
        assert "trim" in output.lower()

    def test_no_positions_prints_message(self, capsys):
        """Empty portfolio should print a message, not crash."""
        s = Sizer(portfolio_value=3000)
        s.review_portfolio()
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_stop_loss_reminder_always_printed(self, capsys):
        """Stop-loss reminder must always appear in review output."""
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 240})
        s.review_portfolio()
        output = capsys.readouterr().out
        assert "30%" in output or "stop" in output.lower()

    def test_exact_10_pct_position_is_ok(self, capsys):
        """Position at exactly 10% must be OK — not flagged TRIM."""
        s = Sizer(portfolio_value=3000, open_positions={"AAPL": 300})
        s.review_portfolio()
        output = capsys.readouterr().out
        assert "TRIM" not in output


# ── 13. Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary values and unusual inputs that must not crash or misbehave."""

    def test_single_position_exactly_at_hard_max(self):
        """£1,000 position on a large portfolio — exactly at hard max."""
        s = Sizer(portfolio_value=20000)
        result = s.size("NVDA", score=3)
        assert result.recommended_size == HARD_MAX_GBP

    def test_rounding_to_nearest_10(self):
        """All recommended sizes must be multiples of £10."""
        for pv in [1000, 2600, 3000, 5000, 7500, 10000]:
            s = Sizer(portfolio_value=pv)
            for score in [2, 3]:
                result = s.size("NVDA", score=score)
                if result.recommended_size > 0:
                    assert result.recommended_size % 10 == 0, (
                        f"At portfolio £{pv}, score {score}: "
                        f"size £{result.recommended_size} not a multiple of £10"
                    )

    def test_large_number_of_positions_counted_correctly(self):
        """9 positions — n_open must be exactly 9."""
        positions = {f"T{i}": 100 for i in range(9)}
        s = Sizer(portfolio_value=5000, open_positions=positions)
        assert s.n_open == 9

    def test_very_large_portfolio_still_caps_at_hard_max(self):
        s = Sizer(portfolio_value=1_000_000)
        result = s.size("NVDA", score=3)
        assert result.recommended_size == HARD_MAX_GBP

    def test_ticker_case_sensitivity(self):
        """NVDA and nvda are treated as different tickers."""
        s = Sizer(portfolio_value=3000, open_positions={"NVDA": 240})
        result_lower = s.size("nvda", score=3)
        # "nvda" is not "NVDA" — should not be blocked for pyramiding
        assert result_lower.can_trade is True


# ── 14. Your actual portfolio — known value tests ─────────────────────────────

class TestActualPortfolio:
    """
    Tests using your real post-rebalance portfolio (April 2026).
    These are integration-level checks that the sizer works for your
    specific situation, not just abstract inputs.
    """

    # Post-rebalance positions (approximate values)
    POSITIONS = {
        "PLTR":  466,
        "NFLX":  271,
        "ASML":  259,
        "AMD":   239,
        "CCJ":   141,
        "UPS":   140,
        "VST":   108,
        "MU":     94,
        "LMT":    89,
        "TCEHY":  75,
    }
    PORTFOLIO_VALUE = 2600

    def test_all_10_positions_held(self):
        s = Sizer(portfolio_value=self.PORTFOLIO_VALUE,
                  open_positions=self.POSITIONS)
        assert s.n_open == 10

    def test_11th_position_blocked_at_max(self):
        """With 10 positions open, any new signal must be blocked."""
        s = Sizer(portfolio_value=self.PORTFOLIO_VALUE,
                  open_positions=self.POSITIONS)
        result = s.size("NVDA", score=3)
        assert result.can_trade is False

    def test_pltr_blocked_for_pyramiding(self):
        """Already hold PLTR — must not be able to add."""
        s = Sizer(portfolio_value=self.PORTFOLIO_VALUE,
                  open_positions=self.POSITIONS)
        result = s.size("PLTR", score=3)
        assert result.can_trade is False

    def test_deployed_calculation_correct(self):
        """Total deployed must equal sum of all position values."""
        s = Sizer(portfolio_value=self.PORTFOLIO_VALUE,
                  open_positions=self.POSITIONS)
        expected_deployed = sum(self.POSITIONS.values())
        assert abs(s.deployed - expected_deployed) < 0.01

    def test_score_3_size_at_current_portfolio(self):
        """
        With fresh capital: £2,600 * 8% = £208 → rounded to £210.
        This is the exact size for a new score-3 signal.
        """
        s = Sizer(portfolio_value=self.PORTFOLIO_VALUE)
        result = s.size("NVDA", score=3)
        assert result.recommended_size == 210.0

    def test_score_2_size_at_current_portfolio(self):
        """£2,600 * 5% = £130."""
        s = Sizer(portfolio_value=self.PORTFOLIO_VALUE)
        result = s.size("NVDA", score=2)
        assert result.recommended_size == 130.0


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])