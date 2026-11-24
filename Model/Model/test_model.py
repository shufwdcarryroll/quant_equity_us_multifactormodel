import numpy as np
import pandas as pd
import pytest

from model import (
    DATA_DIR,
    MULTI_FACTOR,
    SECTORS,
    TOP_N,
    XLC_PARTIAL_MONTH,
    build_factor_scores,
    build_target_weights,
    classify_regimes,
    load_inputs,
    max_drawdown,
    run_backtest,
    trend_gap,
)

CUTOFF = pd.Timestamp("2023-06-30")


@pytest.fixture(scope="module")
def inputs():
    return load_inputs()


def reverse_rows(table, rows):
    changed = table.copy()
    changed.loc[rows] = changed.loc[rows].to_numpy()[:, ::-1]
    return changed


def blank_frame():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame(0.0, index=dates, columns=SECTORS)


def test_returns_include_dividends():
    long = pd.read_csv(DATA_DIR / "raw" / "crsp_m_etfs_long.csv")
    assert (long["ret"] - long["retx"]).abs().gt(1e-9).sum() > 300


def test_monthly_returns_match_daily(inputs):
    returns, daily, _, _ = inputs
    compounded = (1 + daily).groupby(daily.index.to_period("M")).prod(min_count=1) - 1
    compounded.index = compounded.index.to_timestamp("M").normalize()
    assert (returns - compounded.reindex(returns.index)).abs().max().max() < 1e-5


def test_xlc_launch_month_removed(inputs):
    assert np.isnan(inputs[0].loc[XLC_PARTIAL_MONTH, "XLC"])


def test_future_data_does_not_change_scores(inputs):
    returns, daily, ratios, _ = inputs
    base = build_factor_scores(returns, daily, ratios)
    later_ratios = {
        name: reverse_rows(table, table.index > CUTOFF)
        for name, table in ratios.items()
    }
    changed = build_factor_scores(
        reverse_rows(returns, returns.index > CUTOFF),
        reverse_rows(daily, daily.index > CUTOFF),
        later_ratios,
    )
    for name, table in base.items():
        pd.testing.assert_frame_equal(table.loc[:CUTOFF], changed[name].loc[:CUTOFF])


def test_ratios_use_prior_month(inputs):
    returns, daily, ratios, _ = inputs
    base = build_factor_scores(returns, daily, ratios)
    cutoff_ratios = {
        name: reverse_rows(table, table.index == CUTOFF)
        for name, table in ratios.items()
    }
    changed = build_factor_scores(returns, daily, cutoff_ratios)
    next_month = CUTOFF + pd.offsets.MonthEnd(1)
    for factor in ("value", "quality"):
        pd.testing.assert_frame_equal(
            base[factor].loc[[CUTOFF]],
            changed[factor].loc[[CUTOFF]],
        )
        assert not base[factor].loc[next_month].equals(changed[factor].loc[next_month])


def test_portfolios_hold_top_three(inputs):
    returns, daily, ratios, _ = inputs
    targets = build_target_weights(build_factor_scores(returns, daily, ratios))
    weights = targets[MULTI_FACTOR]
    assert (weights.gt(0).sum(axis=1) == TOP_N).all()
    assert np.isclose(weights.sum(axis=1), 1).all()
    assert all(table.index.equals(weights.index) for table in targets.values())


def test_backtest_uses_next_month_return():
    weights = blank_frame()
    weights["XLK"] = 1.0
    returns = blank_frame()
    returns["XLK"] = [0.50, 0.10, -0.20]
    run = run_backtest(weights, returns, cost_bps=0)
    assert run["gross"].tolist() == pytest.approx([0.10, -0.20])


def test_cost_charged_on_traded_weight():
    weights = blank_frame()
    weights.loc[weights.index[0], "XLK"] = 1.0
    weights.loc[weights.index[1], "XLE"] = 1.0
    run = run_backtest(weights, blank_frame(), cost_bps=10)
    assert run["net"].tolist() == pytest.approx([-0.001, -0.002])
    assert run["one_way_turnover"].tolist() == pytest.approx([0.5, 1.0])


def test_trend_ignores_price_splits(inputs):
    assert trend_gap(inputs[0][SECTORS]).loc["2025-12-31"].abs().max() < 0.25


def test_risk_free_scale(inputs):
    risk_free = inputs[3]
    assert risk_free.loc["2024-01-31"] == pytest.approx(0.0045, abs=0.0005)
    assert risk_free.loc["2021-06-30"] < 0.0005


def test_regimes_use_prior_information(inputs):
    benchmark = inputs[0]["SPY"]
    changed = benchmark.copy()
    changed.loc[CUTOFF] = -0.5
    base = classify_regimes(benchmark)
    assert base.loc[CUTOFF] == classify_regimes(changed).loc[CUTOFF]


def test_max_drawdown_counts_first_month_loss():
    assert max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)
