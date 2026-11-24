from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR.parent / "data"
RESULTS_DIR = MODEL_DIR / "results"
PLOTS_DIR = MODEL_DIR / "plots"

BENCHMARK = "SPY"
SECTORS = "XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY".split()
FACTORS = ["momentum", "low_volatility", "trend", "value", "quality"]
MULTI_FACTOR = "multi_factor"
EQUAL_WEIGHT = "equal_weight"
PORTFOLIOS = [*FACTORS, MULTI_FACTOR, EQUAL_WEIGHT]
STRATEGIES = [*PORTFOLIOS, BENCHMARK]
COMPARISONS = [
    *[("factor_vs_equal_weight", name, EQUAL_WEIGHT) for name in FACTORS],
    *[("multi_vs_single", MULTI_FACTOR, name) for name in FACTORS],
    ("multi_vs_equal", MULTI_FACTOR, EQUAL_WEIGHT),
]
REGIMES = ["Recovery", "Mid-Cycle", "Late-Cycle", "Downturn"]
RATIO_COLUMNS = [
    "pe",
    "book_to_market",
    "dividend_yield",
    "roe",
    "roa",
    "gross_profitability",
    "debt_to_assets",
    "value_score",
    "quality_score",
]

MOMENTUM_WINDOWS = (3, 6, 12)
TREND_WINDOW = 10
VOLATILITY_DAYS = 252
FUNDAMENTAL_LAG = 1
TOP_N = 3
COST_BPS = 10.0
CVAR_LEVEL = 0.05
RANDOM_SEED = 2026
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_BLOCK = 6
XLC_PARTIAL_MONTH = pd.Timestamp("2018-06-30")
HOLDOUT_START = pd.Timestamp("2022-01-01")
ONE_MONTH_BILL_ID = 2000061


def read_returns(file_name):
    path = DATA_DIR / "raw" / file_name
    returns = pd.read_csv(path, index_col="date", parse_dates=True)
    return returns[[*SECTORS, BENCHMARK]].sort_index()


def load_risk_free(month_ends):
    path = DATA_DIR / "raw" / "treasury_monthly_raw_review.csv"
    treasury = pd.read_csv(path, parse_dates=["mcaldt"])
    bill = treasury[treasury["kytreasnox"] == ONE_MONTH_BILL_ID]
    bill = bill.set_index("mcaldt").sort_index()
    days_held = bill.index.to_series().diff().shift(-1).dt.days
    # compound daily yield
    next_return = (1 + bill["tmyld"]) ** days_held - 1
    next_return.index = next_return.index + pd.offsets.MonthEnd(0)
    return next_return.shift(1).reindex(month_ends).rename("risk_free")


def load_inputs():
    monthly = read_returns("etf_monthly_returns.csv")
    # drop partial launch month
    monthly.loc[XLC_PARTIAL_MONTH, "XLC"] = np.nan
    daily = read_returns("crsp_d_returns.csv")

    path = DATA_DIR / "processed" / "sector_factor_panel.csv"
    panel = pd.read_csv(path, parse_dates=["date"])
    panel = panel[panel["ticker"].isin(SECTORS)]
    ratios = {
        column: panel.pivot(index="date", columns="ticker", values=column)[SECTORS]
        for column in RATIO_COLUMNS
    }
    return monthly, daily, ratios, load_risk_free(monthly.index)


def zscore(frame):
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)


def average_zscores(frames):
    return zscore(sum(zscore(frame) for frame in frames) / len(frames))


def trend_gap(returns):
    index = (1 + returns).cumprod()
    return index / index.rolling(TREND_WINDOW).mean() - 1


def build_factor_scores(returns, daily, ratios, lag=FUNDAMENTAL_LAG):
    monthly = returns[SECTORS]
    momentum = [
        (1 + monthly).rolling(window).apply(np.prod, raw=True) - 1
        for window in MOMENTUM_WINDOWS
    ]
    volatility = daily[SECTORS].rolling(VOLATILITY_DAYS).std() * np.sqrt(252)
    volatility = volatility.resample("ME").last().reindex(returns.index)
    # prepared percentile scores
    prepared = {
        name: ratios[f"{name}_score"].reindex(returns.index).shift(lag)
        for name in ("value", "quality")
    }
    scores = {
        "momentum": average_zscores(momentum),
        "low_volatility": zscore(-volatility),
        "trend": zscore(trend_gap(monthly)),
        "value": zscore(prepared["value"]),
        "quality": zscore(prepared["quality"]),
    }

    # keep complete sector months
    missing = sum(table.isna().sum(axis=1) for table in scores.values())
    scores = {name: table.loc[missing == 0] for name, table in scores.items()}
    scores[MULTI_FACTOR] = average_zscores([scores[name] for name in FACTORS])
    return scores


def raw_ratio_scores(scores, ratios):
    index = scores[MULTI_FACTOR].index
    lagged = {
        name: table.shift(FUNDAMENTAL_LAG).reindex(index)
        for name, table in ratios.items()
    }
    value = [1 / lagged["pe"], lagged["book_to_market"], lagged["dividend_yield"]]
    quality = [
        lagged["roe"],
        lagged["roa"],
        lagged["gross_profitability"],
        -lagged["debt_to_assets"],
    ]
    variant = {
        **scores,
        "value": average_zscores(value),
        "quality": average_zscores(quality),
    }
    variant[MULTI_FACTOR] = average_zscores([variant[name] for name in FACTORS])
    return variant


def factor_correlations(scores):
    panel = pd.concat({name: scores[name].stack() for name in FACTORS}, axis=1)
    monthly = panel.groupby(level=0).corr(method="spearman")
    return monthly.groupby(level=1).mean().loc[FACTORS, FACTORS]


def top_n_weights(scores, top_n=TOP_N):
    ranks = scores[SECTORS].rank(axis=1, ascending=False, method="first")
    return ranks.le(top_n).astype(float) / top_n


def build_target_weights(scores, top_n=TOP_N):
    targets = {
        name: top_n_weights(scores[name], top_n)
        for name in [*FACTORS, MULTI_FACTOR]
    }
    index = scores[MULTI_FACTOR].index
    targets[EQUAL_WEIGHT] = pd.DataFrame(1 / len(SECTORS), index=index, columns=SECTORS)
    return targets


def run_backtest(weights, period_returns, cost_bps=COST_BPS):
    dates = weights.index[1:]
    held = weights[SECTORS].iloc[:-1].set_axis(dates)
    asset_returns = period_returns.loc[dates, SECTORS]
    gross = (held * asset_returns).sum(axis=1)
    # drift weights to month end
    drifted = (held * (1 + asset_returns)).div(1 + gross, axis=0).shift(1).fillna(0)
    traded = (held - drifted).abs().sum(axis=1)
    # charge cost on traded weight
    net = (1 - cost_bps / 10_000 * traded) * (1 + gross) - 1
    return pd.DataFrame({"gross": gross, "net": net, "one_way_turnover": traded / 2})


def run_strategies(targets, period_returns, benchmark, cost_bps=COST_BPS):
    runs = pd.concat(
        {
            name: run_backtest(weights, period_returns, cost_bps)
            for name, weights in targets.items()
        },
        axis=1,
    )
    net = runs.xs("net", axis=1, level=1)
    gross = runs.xs("gross", axis=1, level=1)
    net[BENCHMARK] = gross[BENCHMARK] = benchmark.reindex(runs.index)
    return net, gross, runs.xs("one_way_turnover", axis=1, level=1)


def next_day_returns(daily, month_ends):
    wealth = (1 + daily[SECTORS].fillna(0)).cumprod()
    # enter at next close
    positions = wealth.index.searchsorted(month_ends, side="right")
    entry = wealth.iloc[np.minimum(positions, len(wealth) - 1)].set_axis(month_ends)
    return (entry / entry.shift(1) - 1).dropna(how="all")


def annual_return(returns):
    return (1 + returns).prod() ** (12 / len(returns)) - 1


def sharpe_ratio(returns, risk_free):
    excess = returns.sub(risk_free.reindex(returns.index), axis=0)
    return excess.mean() / excess.std() * np.sqrt(12)


def max_drawdown(returns):
    wealth = np.concatenate([[1.0], (1 + returns).cumprod().to_numpy()])
    return (wealth / np.maximum.accumulate(wealth) - 1).min()


def study_periods(index):
    return {
        "full": index,
        "development": index[index < HOLDOUT_START],
        "holdout": index[index >= HOLDOUT_START],
    }


def held_weights(weights, dates):
    return weights.shift(1).reindex(dates)[SECTORS]


def performance_table(net, turnover, targets, risk_free):
    # exclude initial purchase
    rebalances = turnover.iloc[1:]
    rows = []
    for period, dates in study_periods(net.index).items():
        tail_months = int(np.ceil(len(dates) * CVAR_LEVEL))
        for strategy in STRATEGIES:
            returns = net.loc[dates, strategy]
            row = {
                "period": period,
                "strategy": strategy,
                "months": len(returns),
                "annual_return": annual_return(returns),
                "volatility": returns.std() * np.sqrt(12),
                "sharpe": sharpe_ratio(returns, risk_free),
                "max_drawdown": max_drawdown(returns),
                "cvar_95": returns.nsmallest(tail_months).mean(),
                "hit_rate_vs_equal_weight": (
                    returns > net.loc[dates, EQUAL_WEIGHT]
                ).mean(),
            }
            if strategy in targets:
                average_weight = held_weights(targets[strategy], dates).mean()
                row["avg_one_way_turnover"] = rebalances[strategy].reindex(dates).mean()
                row["effective_sectors"] = 1 / (average_weight**2).sum()
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_indices(length, rng):
    # random block restarts
    restart = rng.random((BOOTSTRAP_DRAWS, length)) < 1 / BOOTSTRAP_BLOCK
    starts = rng.integers(0, length, size=(BOOTSTRAP_DRAWS, length))
    indices = np.empty((BOOTSTRAP_DRAWS, length), dtype=int)
    indices[:, 0] = starts[:, 0]
    for month in range(1, length):
        continued = (indices[:, month - 1] + 1) % length
        indices[:, month] = np.where(restart[:, month], starts[:, month], continued)
    return indices


def bootstrap_metrics(returns, risk_free, indices):
    sample = returns[indices]
    excess = sample - risk_free[indices]
    annual = np.prod(1 + sample, axis=1) ** (12 / sample.shape[1]) - 1
    sharpe = excess.mean(axis=1) / excess.std(axis=1, ddof=1) * np.sqrt(12)
    return annual, sharpe


def summarize_draws(prefix, draws):
    tail = min((draws <= 0).mean(), (draws >= 0).mean())
    return {
        f"{prefix}_ci_low": np.quantile(draws, 0.025),
        f"{prefix}_ci_high": np.quantile(draws, 0.975),
        f"{prefix}_p": min(2 * tail, 1.0),
    }


def holm_adjust(p_values):
    return multipletests(p_values, method="holm")[1]


def pairwise_tests(net, risk_free):
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for period, dates in study_periods(net.index).items():
        sample = net.loc[dates]
        annual = annual_return(sample)
        sharpe = sharpe_ratio(sample, risk_free)
        indices = bootstrap_indices(len(dates), rng)
        rf = risk_free.reindex(dates).to_numpy()
        draws = {
            name: bootstrap_metrics(sample[name].to_numpy(), rf, indices)
            for name in PORTFOLIOS
        }
        for family, left, right in COMPARISONS:
            rows.append(
                {
                    "period": period,
                    "family": family,
                    "strategy": left,
                    "comparator": right,
                    "annual_return_diff": annual[left] - annual[right],
                    **summarize_draws("annual_diff", draws[left][0] - draws[right][0]),
                    "sharpe_diff": sharpe[left] - sharpe[right],
                    **summarize_draws("sharpe_diff", draws[left][1] - draws[right][1]),
                }
            )
    tests = pd.DataFrame(rows)
    groups = tests.groupby(["period", "family"])
    for column in ("annual_diff_p", "sharpe_diff_p"):
        tests[f"{column}_holm"] = groups[column].transform(holm_adjust)
    return tests


def sensitivity_table(returns, daily, ratios, scores, risk_free):
    price_signal = average_zscores([scores["momentum"], scores["trend"]])
    merged = {
        **scores,
        MULTI_FACTOR: average_zscores(
            [price_signal, scores["low_volatility"], scores["value"], scores["quality"]]
        ),
    }
    raw_ratios = raw_ratio_scores(scores, ratios)
    delayed = next_day_returns(daily, scores[MULTI_FACTOR].index)

    scenarios = [
        *[(f"cost_{c}bps", scores, returns, c, TOP_N) for c in (0, 5, 10, 25, 50)],
        *[(f"top_{n}", scores, returns, COST_BPS, n) for n in (2, 4)],
        *[
            (
                f"fundamental_lag_{lag}m",
                build_factor_scores(returns, daily, ratios, lag),
                returns,
                COST_BPS,
                TOP_N,
            )
            for lag in (0, 2)
        ],
        ("momentum_trend_merged", merged, returns, COST_BPS, TOP_N),
        ("raw_ratio_zscores", raw_ratios, returns, COST_BPS, TOP_N),
        ("next_day_execution", scores, delayed, COST_BPS, TOP_N),
    ]

    rows = []
    for name, variant, period_returns, cost, top_n in scenarios:
        targets = build_target_weights(variant, top_n)
        net, _, _ = run_strategies(targets, period_returns, returns[BENCHMARK], cost)
        annual = annual_return(net)
        sharpe = sharpe_ratio(net, risk_free)
        rows += [
            {
                "scenario": name,
                "strategy": strategy,
                "annual_return": annual[strategy],
                "sharpe": sharpe[strategy],
                "excess_vs_equal_weight": annual[strategy] - annual[EQUAL_WEIGHT],
            }
            for strategy in PORTFOLIOS
        ]
    return pd.DataFrame(rows)


def sector_holdings(targets, dates):
    held = {
        name: held_weights(weights, dates).mean()
        for name, weights in targets.items()
    }
    return pd.DataFrame(held).T


def classify_regimes(benchmark):
    wealth = (1 + benchmark).cumprod()
    # use prior month data
    long_return = wealth.pct_change(12, fill_method=None).shift(1)
    short_return = wealth.pct_change(3, fill_method=None).shift(1)
    conditions = [
        (long_return <= 0) & (short_return > 0),
        (long_return > 0) & (short_return > 0),
        (long_return > 0) & (short_return <= 0),
        (long_return <= 0) & (short_return <= 0),
    ]
    labels = np.select(conditions, REGIMES, default="")
    regimes = pd.Series(labels, index=benchmark.index, name="regime")
    return regimes.where(regimes != "")


def regime_tables(net, targets, regimes):
    regimes = regimes.reindex(net.index)
    excess = net[[*FACTORS, MULTI_FACTOR]].sub(net[EQUAL_WEIGHT], axis=0)
    performance = excess.groupby(regimes).mean().reindex(REGIMES).T
    performance.loc["months"] = regimes.value_counts().reindex(REGIMES)
    performance["hit_rate"] = (excess > 0).mean()
    held = held_weights(targets[MULTI_FACTOR], net.index)
    return performance, held.groupby(regimes).mean().reindex(REGIMES)
