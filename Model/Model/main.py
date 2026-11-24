import pandas as pd

from figures import create_figures
from model import (
    BENCHMARK,
    PLOTS_DIR,
    RESULTS_DIR,
    STRATEGIES,
    build_factor_scores,
    build_target_weights,
    classify_regimes,
    factor_correlations,
    load_inputs,
    pairwise_tests,
    performance_table,
    regime_tables,
    run_strategies,
    sector_holdings,
    sensitivity_table,
)


def long_table(tables, value_name):
    frames = [
        table.stack()
        .rename(value_name)
        .rename_axis(["date", "ticker"])
        .reset_index()
        .assign(strategy=name)
        for name, table in tables.items()
    ]
    return pd.concat(frames, ignore_index=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    returns, daily, ratios, risk_free = load_inputs()
    scores = build_factor_scores(returns, daily, ratios)
    targets = build_target_weights(scores)
    net, gross, turnover = run_strategies(targets, returns, returns[BENCHMARK])
    performance = performance_table(net, turnover, targets, risk_free)
    regimes = classify_regimes(returns[BENCHMARK])
    regime_performance, regime_weights = regime_tables(net, targets, regimes)

    indexed_tables = {
        "strategy_net_returns_monthly": net[STRATEGIES],
        "strategy_gross_returns_monthly": gross[STRATEGIES],
        "strategy_turnover_monthly": turnover,
        "factor_correlations": factor_correlations(scores),
        "regime_excess_returns": regime_performance,
        "regime_sector_weights": regime_weights,
        "sector_holdings": sector_holdings(targets, net.index),
    }
    plain_tables = {
        "factor_scores_monthly": long_table(scores, "score"),
        "target_weights_monthly": long_table(targets, "target_weight"),
        "performance_by_period": performance,
        "pairwise_tests": pairwise_tests(net, risk_free),
        "sensitivity": sensitivity_table(returns, daily, ratios, scores, risk_free),
    }
    for name, table in indexed_tables.items():
        table.to_csv(RESULTS_DIR / f"{name}.csv")
    for name, table in plain_tables.items():
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)

    create_figures(
        scores, net, performance, regimes, regime_performance, regime_weights
    )

    summary = performance.query("period == 'full'").set_index("strategy")
    columns = ["annual_return", "volatility", "sharpe", "max_drawdown"]
    print(f"Net performance {net.index[0]:%Y-%m} to {net.index[-1]:%Y-%m}")
    print(summary[columns].round(3).to_string())


if __name__ == "__main__":
    main()
