"""tests/

Four test modules covering the four most critical correctness requirements:
  test_no_lookahead.py   — automated look-ahead bias detection
  test_universe_pit.py  — point-in-time universe integrity
  test_costs.py         — A-share cost model arithmetic
  test_metrics.py       — performance metric formulae

Run all tests:  pytest tests/ -v
"""

# ============================================================
# tests/test_no_lookahead.py
# ============================================================
# (Each module is a separate file in tests/; combined here for brevity.
#  Split into individual files when placing in the repo.)

import numpy as np
import pandas as pd
import pytest


# ── Look-ahead bias ──────────────────────────────────────────────────────────

class TestNoLookahead:
    """
    Verify that every predictive factor's value at date t has near-zero
    correlation with the same-day return at t.
    A non-trivial correlation would indicate look-ahead bias.
    """

    @pytest.fixture
    def mock_prices(self):
        """200 trading days, 30 stocks, random walk prices."""
        rng   = np.random.default_rng(0)
        dates = pd.bdate_range("2020-01-02", periods=200)
        codes = [f"S{i:03d}.SZ" for i in range(30)]
        log_rets = rng.normal(0.0005, 0.02, (200, 30))
        prices   = 100 * np.exp(np.cumsum(log_rets, axis=0))
        return pd.DataFrame(prices, index=dates, columns=codes)

    def _check_factor(self, factor_class, prices, **kwargs):
        from src.factors.base import BaseFactor
        f: BaseFactor = factor_class(**kwargs)
        factor_df = f.compute(prices)
        # Should not raise
        f.validate_no_lookahead(prices, factor_df, n_spot_checks=50)

    def test_mom5d_no_lookahead(self, mock_prices):
        from src.factors.momentum import Mom5d
        self._check_factor(Mom5d, mock_prices)

    def test_mom20d_no_lookahead(self, mock_prices):
        from src.factors.momentum import Mom20d
        self._check_factor(Mom20d, mock_prices)

    def test_zscore_no_lookahead(self, mock_prices):
        from src.factors.reversion import ZScore20d
        self._check_factor(ZScore20d, mock_prices)

    def test_rsi_no_lookahead(self, mock_prices):
        from src.factors.reversion import RSI14d
        self._check_factor(RSI14d, mock_prices)

    def test_shifted_factor_fails_correctly(self, mock_prices):
        """
        Sanity check: a factor that USES today's price SHOULD fail the test.
        """
        from src.factors.base import BaseFactor

        class LeakyFactor(BaseFactor):
            def compute(self, prices, **_):
                # Intentionally NO .shift(1) — this leaks today's price
                return prices.pct_change(5)

        leaky = LeakyFactor()
        factor_df = leaky.compute(mock_prices)
        with pytest.raises(AssertionError, match="Look-ahead"):
            leaky.validate_no_lookahead(mock_prices, factor_df, n_spot_checks=50)


# ── Point-in-Time Universe ────────────────────────────────────────────────────

class TestUniversePIT:
    """
    Spot-check that get_universe() returns the historically correct
    constituent list, not today's list.

    These tests use the ACTUAL Tushare data once fetched.
    They are skipped automatically if the data files don't exist yet.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        from src.utils import RAW
        path = RAW / "index_weight" / "000300.SH.parquet"
        if not path.exists():
            pytest.skip("Tushare data not yet fetched — run main.py --fetch first.")

    def test_universe_is_nonempty(self):
        from src.data.universe import get_universe
        u = get_universe("20200103", "000300.SH")
        assert len(u) > 100, "CSI 300 universe should have > 100 eligible stocks."

    def test_universe_size_reasonable(self):
        """Universe must be a subset of the 300-stock index (with filters, expect 180–300)."""
        from src.data.universe import get_universe
        u = get_universe("20190104", "000300.SH")
        assert 100 <= len(u) <= 300, f"Unexpected universe size: {len(u)}"

    def test_universe_changes_over_time(self):
        """Universe at two different dates must differ (index rebalances happen)."""
        from src.data.universe import get_universe
        u_2016 = set(get_universe("20160104", "000300.SH"))
        u_2023 = set(get_universe("20230103", "000300.SH"))
        overlap = u_2016 & u_2023
        # Some turnover is expected; if 100% overlap, PIT is probably broken.
        assert len(overlap) < len(u_2016), "Universe never changed — PIT may be broken."

    def test_no_future_stocks_in_historical_universe(self):
        """
        Any stock that first appears in the index AFTER a given date must NOT
        be in the universe on that date.
        """
        from src.data.universe import get_universe, _load_index_weight
        CHECK_DATE = pd.Timestamp("20180103")
        universe = set(get_universe(CHECK_DATE, "000300.SH"))

        weights = _load_index_weight("000300.SH")
        # Stocks whose first appearance in index is after CHECK_DATE
        first_appearances = weights.groupby("con_code")["trade_date"].min()
        future_stocks = first_appearances[first_appearances > CHECK_DATE].index

        leaked = universe & set(future_stocks)
        assert len(leaked) == 0, (
            f"Survivorship bias: {len(leaked)} future stocks found in {CHECK_DATE.date()} "
            f"universe: {list(leaked)[:5]}"
        )


# ── Transaction costs ─────────────────────────────────────────────────────────

class TestCosts:

    def test_buy_cost_greater_than_notional(self):
        from src.backtest.costs import buy_cost
        price, shares = 10.0, 100.0
        cost = buy_cost(price, shares, is_shanghai=True, run_id="csi300")
        assert cost > price * shares, "Buy cost must exceed notional."

    def test_sell_proceeds_less_than_notional(self):
        from src.backtest.costs import sell_proceeds
        price, shares = 10.0, 100.0
        proceeds = sell_proceeds(price, shares, is_shanghai=True, run_id="csi300")
        assert proceeds < price * shares, "Sell proceeds must be less than notional."

    def test_round_trip_csi300_reasonable(self):
        from src.backtest.costs import round_trip_cost_rate
        rt = round_trip_cost_rate("csi300")
        # Should be approximately 0.002–0.004 (0.20%–0.40%)
        assert 0.001 < rt < 0.006, f"Unexpected round-trip cost: {rt:.4%}"

    def test_csi500_more_expensive_than_csi300(self):
        from src.backtest.costs import round_trip_cost_rate
        assert round_trip_cost_rate("csi500") > round_trip_cost_rate("csi300")

    def test_stamp_duty_sell_only(self):
        """
        Buying and immediately selling should incur stamp duty once (on sell).
        A buy followed by a sell of the same notional should cost:
          buy_cost + sell_notional - sell_proceeds = total_cost
        Stamp duty (0.05%) should appear only once.
        """
        from src.backtest.costs import buy_cost, sell_proceeds
        from src.utils import load_config
        cfg = load_config()["backtest"]
        price, shares = 100.0, 1.0

        buy  = buy_cost(price, shares, is_shanghai=True, run_id="csi300")
        sell = sell_proceeds(price, shares, is_shanghai=True, run_id="csi300")

        total_cost = buy - price * shares + price * shares - sell
        stamp_duty_once = price * shares * cfg["stamp_duty_rate"]

        # Total cost should contain stamp duty approximately once
        assert abs(total_cost - (
            price * shares * (
                2 * cfg["commission_rate"]
                + cfg["stamp_duty_rate"]
                + 2 * cfg["exchange_fee_rate"]
                + 2 * cfg["slippage_csi300"]
            )
        )) < 1e-8


# ── Performance metrics ────────────────────────────────────────────────────────

class TestMetrics:

    @pytest.fixture
    def flat_equity(self):
        idx = pd.bdate_range("2020-01-02", periods=252)
        return pd.Series(np.ones(252) * 1_000_000, index=idx)

    @pytest.fixture
    def growing_equity(self):
        idx = pd.bdate_range("2020-01-02", periods=252)
        # 10% annual return compounded daily
        daily_r = (1.10) ** (1 / 252) - 1
        vals = 1_000_000 * (1 + daily_r) ** np.arange(252)
        return pd.Series(vals, index=idx)

    def test_sharpe_flat_is_nan_or_zero(self, flat_equity):
        from src.backtest.metrics import sharpe_ratio
        ret = np.log(flat_equity / flat_equity.shift(1)).dropna()
        sr = sharpe_ratio(ret)
        assert np.isnan(sr) or sr == 0.0

    def test_annual_return_10pct(self, growing_equity):
        from src.backtest.metrics import annualised_return
        ar = annualised_return(growing_equity)
        assert abs(ar - 0.10) < 0.005, f"Expected ~10% annual return, got {ar:.2%}"

    def test_max_drawdown_positive(self, growing_equity):
        from src.backtest.metrics import max_drawdown
        # Introduce a dip
        eq = growing_equity.copy()
        eq.iloc[100:120] *= 0.85
        mdd = max_drawdown(eq)
        assert mdd > 0, "Max drawdown must be positive when there is a dip."
        assert mdd < 1, "Max drawdown must be < 1 (100%)."

    def test_max_drawdown_monotone_increasing(self, growing_equity):
        from src.backtest.metrics import max_drawdown
        mdd = max_drawdown(growing_equity)
        assert mdd < 0.005, "Monotonically growing equity should have near-zero drawdown."

    def test_calmar_positive_for_positive_return(self, growing_equity):
        from src.backtest.metrics import calmar_ratio
        # Add a small dip so MDD > 0
        eq = growing_equity.copy()
        eq.iloc[50:60] *= 0.97
        calmar = calmar_ratio(eq)
        assert calmar > 0, "Calmar ratio must be positive for positive return with drawdown."
