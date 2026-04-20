"""
risk/sizer.py
Position sizing for the A_6M systematic equity strategy.

Calibrated for:
    - Trading 212 ISA (zero commission)
    - Starting capital: £3,000
    - Monthly additions: £500
    - Strategy: A_6M — price signals only, 6M hold, weekly rebalances

Scoring scale (A_6M):
    Score 3 — all three price criteria pass (C5 + C6 + C9) — high conviction
    Score 2 — two price criteria pass — medium conviction

Sizing rules:
    Score 3:  8% of portfolio  (£240 at £3k, £800 at £10k)
    Score 2:  5% of portfolio  (£150 at £3k, £500 at £10k)
    Hard max: £1,000 per position — never exceed regardless of portfolio size
    Hard min: £50  per position — below this trading friction dominates
    Max open: 10 positions simultaneously
    Cash reserve: always keep 10% in cash — never fully deployed
    Stop-loss:    -30% triggers mandatory written thesis re-evaluation

Usage:
    # From Python
    from risk.sizer import Sizer
    sizer = Sizer(portfolio_value=3000, open_positions={"AAPL": 240})
    result = sizer.size("MSFT", score=3)
    print(result)

    # From terminal
    python risk/sizer.py --portfolio 3000 --score 3
    python risk/sizer.py --portfolio 3000 --score 2 --positions AAPL NVDA
    python risk/sizer.py --portfolio 3000 --review   # review all current positions
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_POSITIONS     = 10      # max open simultaneously
CASH_RESERVE_PCT  = 0.10    # always keep 10% cash uninvested
MAX_POSITION_PCT  = 0.10    # hard cap: 10% of portfolio per position
HARD_MAX_GBP      = 1_000   # hard cap in £ regardless of portfolio size
HARD_MIN_GBP      = 50      # minimum worth trading on Trading 212
STOP_LOSS_PCT     = -30.0   # % loss that triggers mandatory review

SCORE_ALLOCATION  = {
    3: 0.08,   # high conviction — 8% of portfolio
    2: 0.05,   # medium conviction — 5% of portfolio
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SizeResult:
    ticker:           str
    score:            int
    portfolio_value:  float
    recommended_size: float
    pct_of_portfolio: float
    n_open_positions: int
    cash_available:   float
    can_trade:        bool
    reason:           str

    def __str__(self) -> str:
        status = "✓ TRADE" if self.can_trade else "✗ SKIP"
        lines = [
            f"\n{'='*52}",
            f"  {self.ticker:6}  |  Score {self.score}  |  {status}",
            f"{'='*52}",
            f"  Portfolio value:    £{self.portfolio_value:>8,.0f}",
            f"  Recommended size:   £{self.recommended_size:>8,.0f}"
            f"  ({self.pct_of_portfolio:.1f}%)",
            f"  Free cash:          £{self.cash_available:>8,.0f}",
            f"  Open positions:     {self.n_open_positions} / {MAX_POSITIONS}",
            f"  Reason:             {self.reason}",
            f"{'='*52}\n",
        ]
        return "\n".join(lines)


# ── Sizer ─────────────────────────────────────────────────────────────────────

class Sizer:
    """
    Position sizer for A_6M strategy on Trading 212 ISA.

    Args:
        portfolio_value:   total current portfolio value in £
                           (cash + open positions marked to market)
        open_positions:    dict of {ticker: current_market_value}
                           for all currently open positions
        monthly_addition:  planned monthly contribution in £ (informational)
    """

    def __init__(
        self,
        portfolio_value:  float,
        open_positions:   dict[str, float] | None = None,
        monthly_addition: float = 500.0,
    ):
        if portfolio_value <= 0:
            raise ValueError(f"portfolio_value must be positive, got {portfolio_value}")

        self.portfolio_value  = portfolio_value
        self.open_positions   = open_positions or {}
        self.monthly_addition = monthly_addition

        self.n_open       = len(self.open_positions)
        self.deployed     = sum(self.open_positions.values())
        self.cash         = self.portfolio_value - self.deployed
        self.cash_reserve = self.portfolio_value * CASH_RESERVE_PCT
        self.cash_free    = max(0.0, self.cash - self.cash_reserve)

    def size(self, ticker: str, score: int) -> SizeResult:
        """
        Compute recommended position size for a new trade.

        Args:
            ticker: stock ticker symbol
            score:  signal score (2 or 3 for A_6M strategy)

        Returns:
            SizeResult with recommendation and full reasoning
        """
        # ── Validate score ────────────────────────────────────────────────────
        if score not in SCORE_ALLOCATION:
            return SizeResult(
                ticker=ticker, score=score,
                portfolio_value=self.portfolio_value,
                recommended_size=0, pct_of_portfolio=0,
                n_open_positions=self.n_open,
                cash_available=self.cash_free,
                can_trade=False,
                reason=f"Invalid score {score} — A_6M uses scores 2 or 3 only"
            )

        # ── Already holding ───────────────────────────────────────────────────
        if ticker in self.open_positions:
            return SizeResult(
                ticker=ticker, score=score,
                portfolio_value=self.portfolio_value,
                recommended_size=0, pct_of_portfolio=0,
                n_open_positions=self.n_open,
                cash_available=self.cash_free,
                can_trade=False,
                reason=f"Already holding {ticker} — no pyramiding allowed"
            )

        # ── Max positions reached ─────────────────────────────────────────────
        if self.n_open >= MAX_POSITIONS:
            return SizeResult(
                ticker=ticker, score=score,
                portfolio_value=self.portfolio_value,
                recommended_size=0, pct_of_portfolio=0,
                n_open_positions=self.n_open,
                cash_available=self.cash_free,
                can_trade=False,
                reason=f"Max positions reached ({self.n_open}/{MAX_POSITIONS})"
                       f" — wait for an exit before adding"
            )

        # ── Compute base size ─────────────────────────────────────────────────
        alloc_pct = SCORE_ALLOCATION[score]
        raw_size  = self.portfolio_value * alloc_pct

        # Apply caps
        size = min(raw_size, HARD_MAX_GBP)
        size = min(size, self.portfolio_value * MAX_POSITION_PCT)
        size = max(round(size / 10) * 10, 10)  # round to nearest £10, min £10
        pct  = size / self.portfolio_value * 100

        # ── Below minimum ─────────────────────────────────────────────────────
        if size < HARD_MIN_GBP:
            return SizeResult(
                ticker=ticker, score=score,
                portfolio_value=self.portfolio_value,
                recommended_size=size, pct_of_portfolio=pct,
                n_open_positions=self.n_open,
                cash_available=self.cash_free,
                can_trade=False,
                reason=f"Position too small (£{size:.0f} < £{HARD_MIN_GBP} minimum). "
                       f"Add more capital before trading this position."
            )

        # ── Sufficient free cash ──────────────────────────────────────────────
        if size <= self.cash_free:
            return SizeResult(
                ticker=ticker, score=score,
                portfolio_value=self.portfolio_value,
                recommended_size=size, pct_of_portfolio=pct,
                n_open_positions=self.n_open,
                cash_available=self.cash_free,
                can_trade=True,
                reason=f"Score {score} → {alloc_pct*100:.0f}% allocation. "
                       f"Full size available."
            )

        # ── Cash constrained — reduce size ────────────────────────────────────
        if self.cash_free >= HARD_MIN_GBP:
            reduced     = max(round(self.cash_free / 10) * 10, 10)
            reduced_pct = reduced / self.portfolio_value * 100
            return SizeResult(
                ticker=ticker, score=score,
                portfolio_value=self.portfolio_value,
                recommended_size=reduced, pct_of_portfolio=reduced_pct,
                n_open_positions=self.n_open,
                cash_available=self.cash_free,
                can_trade=True,
                reason=f"Cash constrained — reduced from £{size:.0f} to £{reduced:.0f}. "
                       f"Full size would require £{size:.0f} free cash."
            )

        # ── Insufficient cash ─────────────────────────────────────────────────
        return SizeResult(
            ticker=ticker, score=score,
            portfolio_value=self.portfolio_value,
            recommended_size=0, pct_of_portfolio=0,
            n_open_positions=self.n_open,
            cash_available=self.cash_free,
            can_trade=False,
            reason=f"Insufficient free cash (£{self.cash_free:.0f} available, "
                   f"£{HARD_MIN_GBP} minimum needed). "
                   f"Wait for a position to close or add capital."
        )

    def review_portfolio(self) -> None:
        """
        Review all open positions against sizing rules.
        Flags any position that has grown beyond its maximum allocation
        and outputs a trim recommendation.
        """
        if not self.open_positions:
            print("No open positions to review.")
            return

        print(f"\n{'='*52}")
        print(f"  PORTFOLIO REVIEW")
        print(f"  Portfolio value: £{self.portfolio_value:,.0f}")
        print(f"  Open positions:  {self.n_open}")
        print(f"{'='*52}")

        any_breach = False
        for ticker, value in self.open_positions.items():
            pct = value / self.portfolio_value * 100
            max_pct = MAX_POSITION_PCT * 100
            status = "OK"
            action = ""

            if pct > max_pct:
                status = "TRIM"
                trim_to = self.portfolio_value * MAX_POSITION_PCT
                trim_by = value - trim_to
                action  = f" → trim by £{trim_by:.0f} to £{trim_to:.0f}"
                any_breach = True
            elif value < HARD_MIN_GBP:
                status = "REVIEW"
                action  = f" → position below £{HARD_MIN_GBP} minimum"
                any_breach = True

            print(f"  {ticker:6}  £{value:>7,.0f}  ({pct:.1f}%)  {status}{action}")

        if not any_breach:
            print(f"\n  ✓ All positions within limits")

        # Stop-loss reminder
        print(f"\n  Stop-loss rule: any position down >30% requires written review")
        print(f"{'='*52}\n")

    def summary(self) -> None:
        """Print current portfolio state."""
        print(f"\n{'='*52}")
        print(f"  SIZER STATE")
        print(f"{'='*52}")
        print(f"  Portfolio value:  £{self.portfolio_value:>8,.0f}")
        print(f"  Deployed:         £{self.deployed:>8,.0f}  "
              f"({self.deployed/self.portfolio_value*100:.1f}%)")
        print(f"  Cash total:       £{self.cash:>8,.0f}  "
              f"({self.cash/self.portfolio_value*100:.1f}%)")
        print(f"  Cash reserve:     £{self.cash_reserve:>8,.0f}  "
              f"({CASH_RESERVE_PCT*100:.0f}% reserved)")
        print(f"  Free cash:        £{self.cash_free:>8,.0f}")
        print(f"  Open positions:   {self.n_open} / {MAX_POSITIONS}")
        print(f"\n  Score 3 target:   £{self.portfolio_value*SCORE_ALLOCATION[3]:>7,.0f}"
              f"  ({SCORE_ALLOCATION[3]*100:.0f}%)")
        print(f"  Score 2 target:   £{self.portfolio_value*SCORE_ALLOCATION[2]:>7,.0f}"
              f"  ({SCORE_ALLOCATION[2]*100:.0f}%)")
        print(f"{'='*52}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A_6M position sizer — Trading 212 ISA"
    )
    parser.add_argument("--portfolio", type=float, required=True,
                        help="Total portfolio value in £")
    parser.add_argument("--score", type=int, choices=[2, 3], default=None,
                        help="Signal score (2=medium, 3=high conviction)")
    parser.add_argument("--ticker", type=str, default="TICKER",
                        help="Stock ticker symbol")
    parser.add_argument("--positions", nargs="*", default=[],
                        help="Currently open tickers e.g. --positions AAPL NVDA MSFT")
    parser.add_argument("--position-values", nargs="*", type=float, default=[],
                        help="Current market values matching --positions")
    parser.add_argument("--review", action="store_true",
                        help="Review all open positions against sizing rules")
    parser.add_argument("--summary", action="store_true",
                        help="Print current portfolio state summary")
    args = parser.parse_args()

    # Build open positions dict
    open_positions = {}
    if args.positions:
        if args.position_values and len(args.position_values) == len(args.positions):
            open_positions = dict(zip(args.positions, args.position_values))
        else:
            # Assume POSITION_SIZE for each if values not provided
            from portfolio import POSITION_SIZE
            open_positions = {t: POSITION_SIZE for t in args.positions}

    sizer = Sizer(
        portfolio_value=args.portfolio,
        open_positions=open_positions,
    )

    if args.summary or (not args.score and not args.review):
        sizer.summary()

    if args.review:
        sizer.review_portfolio()

    if args.score:
        result = sizer.size(args.ticker, args.score)
        print(result)


if __name__ == "__main__":
    main()