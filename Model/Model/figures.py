import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from model import (
    BENCHMARK,
    CVAR_LEVEL,
    EQUAL_WEIGHT,
    FACTORS,
    HOLDOUT_START,
    MULTI_FACTOR,
    PLOTS_DIR,
    REGIMES,
    STRATEGIES,
)

STRATEGY_STYLES = {
    "momentum": ("Momentum", "#0072B2"),
    "low_volatility": ("Low Volatility", "#009E73"),
    "trend": ("Trend", "#56B4E9"),
    "value": ("Value", "#E69F00"),
    "quality": ("Quality", "#CC79A7"),
    MULTI_FACTOR: ("Multi-Factor", "#0B2545"),
    EQUAL_WEIGHT: ("Equal Weight", "#D55E00"),
    BENCHMARK: ("SPY", "#666666"),
}
SECTOR_STYLES = {
    "XLE": ("Energy", "#1B3A5C"),
    "XLK": ("Info. Tech", "#2F6DA8"),
    "XLV": ("Health Care", "#6FA8DC"),
    "XLF": ("Financials", "#A9CCE8"),
    "XLI": ("Industrials", "#BFE3D0"),
    "XLC": ("Comm. Services", "#6DBE8B"),
    "XLB": ("Materials", "#D7E8A0"),
    "XLP": ("Cons. Staples", "#FBE7A1"),
    "XLU": ("Utilities", "#F9C784"),
    "XLY": ("Cons. Disc.", "#EE9A5A"),
    "XLRE": ("Real Estate", "#B9504B"),
}
LABELS = {name: style[0] for name, style in STRATEGY_STYLES.items()}
COLORS = {name: style[1] for name, style in STRATEGY_STYLES.items()}
SECTOR_NAMES = {name: style[0] for name, style in SECTOR_STYLES.items()}
REGIME_COLORS = {
    "Recovery": "#DCEFE3",
    "Mid-Cycle": "#DCE9F7",
    "Late-Cycle": "#FBF0CF",
    "Downturn": "#F8DEDC",
}
FACTOR_HEADERS = ["Mom.", "Low Vol", "Trend", "Value", "Quality", "Multi"]
LEGEND_FRAME = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
GRID_STYLE = {"color": "#E6E6E6", "linewidth": 0.5}
GUIDE_STYLE = {"color": "#333333", "linewidth": 0.7, "linestyle": ":"}
DOLLAR_FORMAT = FuncFormatter(lambda value, _: f"${value:.0f}")
PERCENT_FORMAT = PercentFormatter(1.0, decimals=0)
PLOT_STYLE = {
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
}


def new_figure(size, widths, heights, wspace, hspace=None):
    fig = plt.figure(figsize=size)
    grid = GridSpec(
        len(heights),
        2,
        width_ratios=widths,
        height_ratios=heights,
        wspace=wspace,
        hspace=hspace,
        figure=fig,
    )
    return fig, grid


def save_figure(fig, name):
    fig.savefig(PLOTS_DIR / f"{name}.pdf")
    plt.close(fig)


def value_of_100(returns):
    growth = 100 * (1 + returns).cumprod()
    growth.loc[returns.index[0] - pd.offsets.MonthEnd(1)] = 100.0
    return growth.sort_index()


def spread_labels(values, gap):
    order = np.argsort(values)
    placed = values[order].copy()
    for _ in range(200):
        for upper in range(1, len(placed)):
            shortfall = gap - (placed[upper] - placed[upper - 1])
            if shortfall > 1e-9:
                placed[upper - 1] -= shortfall / 2
                placed[upper] += shortfall / 2
    positions = np.empty(len(values))
    positions[order] = placed
    return positions


def leader_line(color, shrink):
    return dict(arrowstyle="-", linewidth=0.35, color=color, shrinkA=0, shrinkB=shrink)


def draw_table(ax, headers, rows, widths):
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=widths,
        cellLoc="right",
        bbox=[0, 0, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    highlight = STRATEGIES.index(MULTI_FACTOR) + 1
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if column == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "B"
            cell.set_linewidth(0.7)
        elif row == highlight:
            cell.set_facecolor("#DCE9F7")
            cell.set_text_props(fontweight="bold")


def draw_heatmap(ax, values, x_labels, y_labels, number_format, floor, bold_last=False):
    limit = max(floor, np.ceil(values.abs().to_numpy().max() * 2) / 2)
    image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    last_column = values.shape[1] - 1
    for (row, column), value in np.ndenumerate(values.round(2).to_numpy() + 0.0):
        ax.text(
            column,
            row,
            number_format.format(value),
            ha="center",
            va="center",
            color="white" if abs(value) > 0.62 * limit else "black",
            fontsize=8,
            fontweight="bold" if bold_last and column == last_column else "normal",
        )
    return image


def draw_score_quadrants(ax, level, change):
    x_max = np.ceil(level.abs().max() * 4 + 0.5) / 4
    y_max = np.ceil(change.abs().max() * 4 + 0.5) / 4
    quadrants = [
        (0, 0, "Recovery", "High and improving", "right", "top"),
        (-x_max, 0, "Mid-Cycle", "Low but improving", "left", "top"),
        (-x_max, -y_max, "Downturn", "Low and weakening", "left", "bottom"),
        (0, -y_max, "Late-Cycle", "High but weakening", "right", "bottom"),
    ]
    for left, bottom, color_key, label, h_align, v_align in quadrants:
        color = REGIME_COLORS[color_key]
        ax.add_patch(
            plt.Rectangle((left, bottom), x_max, y_max, color=color, lw=0, zorder=0)
        )
        text_x = left + (0.96 if h_align == "right" else 0.04) * x_max
        text_y = bottom + (0.95 if v_align == "top" else 0.05) * y_max
        ax.text(
            text_x, text_y, label, ha=h_align, va=v_align, fontsize=8, color="#333333"
        )

    ax.axhline(0, color="#333333", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="#333333", linewidth=0.6, linestyle="--")
    ax.scatter(level, change, color=COLORS[MULTI_FACTOR], s=20, zorder=3)

    label_y = spread_labels(change.to_numpy(), 0.085 * y_max)
    for ticker, x_value, y_value, text_y in zip(level.index, level, change, label_y):
        offset = 0.05 * x_max if x_value < 0.55 * x_max else -0.05 * x_max
        ax.annotate(
            SECTOR_NAMES[ticker],
            (x_value, y_value),
            xytext=(x_value + offset, text_y),
            ha="left" if offset > 0 else "right",
            va="center",
            fontsize=7.5,
            arrowprops=leader_line("#777777", 2),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.6),
            zorder=4,
        )
    ax.set(xlim=(-x_max, x_max), ylim=(-y_max, y_max))
    ax.set_xlabel("Multi-factor score")
    ax.set_ylabel("One-month change in score")
    ax.set_title("B. Score level and one-month change")


def plot_sector_scores(scores):
    composite = scores[MULTI_FACTOR]
    latest = pd.DataFrame(
        {name: scores[name].iloc[-1] for name in [*FACTORS, MULTI_FACTOR]}
    ).sort_values(MULTI_FACTOR, ascending=False)
    change = (composite.iloc[-1] - composite.iloc[-2]).reindex(latest.index)

    fig, grid = new_figure((10.5, 4.8), [1.2, 1], [1], 0.28)
    heat_ax = fig.add_subplot(grid[0])
    y_labels = [
        f"{rank}  {SECTOR_NAMES[ticker]} ({ticker})"
        for rank, ticker in enumerate(latest.index, 1)
    ]
    image = draw_heatmap(heat_ax, latest, FACTOR_HEADERS, y_labels, "{:.2f}", 2.0, True)
    heat_ax.set_title(f"A. Factor z-scores, {composite.index[-1]:%b %Y}", pad=24)
    colorbar = fig.colorbar(
        image, ax=heat_ax, orientation="horizontal", fraction=0.045, pad=0.04
    )
    colorbar.set_label("Cross-sectional z-score")
    colorbar.outline.set_linewidth(0.5)

    draw_score_quadrants(fig.add_subplot(grid[1]), latest[MULTI_FACTOR], change)
    save_figure(fig, "sector_factor_scores")


def draw_growth(ax, net):
    growth = value_of_100(net[STRATEGIES])
    for strategy in STRATEGIES:
        is_main = strategy == MULTI_FACTOR
        ax.plot(
            growth.index,
            growth[strategy],
            color=COLORS[strategy],
            linewidth=2.0 if is_main else 1.1,
            linestyle="--" if strategy == BENCHMARK else "-",
            label=LABELS[strategy],
            zorder=4 if is_main else 2,
        )
    ax.axvline(HOLDOUT_START, **GUIDE_STYLE)
    ax.text(
        HOLDOUT_START,
        0.02,
        f" From {HOLDOUT_START:%b %Y}",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="bottom",
    )

    ends = growth.iloc[-1]
    gap = (ends.max() - ends.min() + 60) * 0.052
    for strategy, position in zip(ends.index, spread_labels(ends.to_numpy(), gap)):
        ax.text(
            growth.index[-1] + pd.DateOffset(months=2),
            position,
            f"${ends[strategy]:.0f}",
            color="white",
            fontsize=7.5,
            fontweight="bold",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc=COLORS[strategy], ec="none"),
        )
    ax.set_xlim(growth.index[0], growth.index[-1] + pd.DateOffset(months=11))
    ax.set_ylabel("Value of $100, net of costs")
    ax.yaxis.set_major_formatter(DOLLAR_FORMAT)
    ax.grid(axis="y", **GRID_STYLE)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title(f"A. Growth of $100, {net.index[0]:%b %Y} to {net.index[-1]:%b %Y}")


def draw_relative_return(ax, net):
    wealth = (1 + net).cumprod()
    versus_equal = wealth[MULTI_FACTOR] / wealth[EQUAL_WEIGHT] - 1
    versus_spy = wealth[MULTI_FACTOR] / wealth[BENCHMARK] - 1
    equal_color = COLORS[EQUAL_WEIGHT]

    ax.fill_between(
        versus_equal.index, 0, versus_equal, color=equal_color, alpha=0.14, linewidth=0
    )
    ax.plot(
        versus_equal,
        color=equal_color,
        linewidth=1.3,
        label="Multi-Factor vs Equal Weight",
    )
    ax.plot(
        versus_spy,
        color=COLORS[BENCHMARK],
        linewidth=1.2,
        linestyle="--",
        label="Multi-Factor vs SPY",
    )
    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.axvline(HOLDOUT_START, **GUIDE_STYLE)
    ax.yaxis.set_major_formatter(PERCENT_FORMAT)
    ax.set_ylabel("Cumulative relative return")
    ax.grid(axis="y", **GRID_STYLE)
    ax.legend(loc="lower left", **LEGEND_FRAME)
    ax.set_title("C. Cumulative return relative to baselines")


def plot_strategy_performance(net, full):
    fig, grid = new_figure((11, 6), [1.25, 1], [1.08, 1], 0.24, 0.38)
    draw_growth(fig.add_subplot(grid[:, 0]), net)

    table_ax = fig.add_subplot(grid[0, 1])
    rows = [
        [
            LABELS[name],
            f"{row.annual_return:.1%}",
            f"{row.volatility:.1%}",
            f"{row.sharpe:.2f}",
            f"{row.max_drawdown:.1%}",
        ]
        for name, row in full.iterrows()
    ]
    headers = ["Strategy", "Return", "Vol.", "Sharpe", "Max DD"]
    draw_table(table_ax, headers, rows, [0.34, 0.165, 0.165, 0.165, 0.165])
    table_ax.set_title("B. Net performance, annualized")

    draw_relative_return(fig.add_subplot(grid[1, 1]), net)
    save_figure(fig, "strategy_performance")


def draw_return_histogram(ax, net):
    shown = [BENCHMARK, EQUAL_WEIGHT, MULTI_FACTOR]
    low = np.floor(net[shown].min().min() * 50) / 50
    high = np.ceil(net[shown].max().max() * 50) / 50
    bins = np.arange(low, high + 0.02, 0.02)
    for strategy in shown:
        is_main = strategy == MULTI_FACTOR
        ax.hist(
            net[strategy],
            bins=bins,
            histtype="stepfilled" if is_main else "step",
            facecolor=COLORS[strategy] if is_main else "none",
            edgecolor=COLORS[strategy],
            alpha=0.32 if is_main else 1,
            linewidth=1.2,
            label=LABELS[strategy],
        )

    cutoff = net[MULTI_FACTOR].quantile(CVAR_LEVEL)
    ax.axvspan(low, cutoff, color="#F8DEDC", zorder=0)
    ax.axvline(cutoff, color="#B2182B", linewidth=0.9, linestyle="--")
    ax.text(
        cutoff - 0.003,
        ax.get_ylim()[1] * 0.97,
        f"Worst {CVAR_LEVEL:.0%}\nMulti-Factor",
        color="#B2182B",
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.xaxis.set_major_formatter(PERCENT_FORMAT)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(low, high)
    ax.set_xlabel("Monthly net return")
    ax.set_ylabel("Number of months")
    ax.legend(loc="upper right")
    ax.set_title("A. Distribution of monthly returns")


def draw_risk_return(ax, full):
    label_y = spread_labels(full["annual_return"].to_numpy(), 0.0072)
    label_x = full["volatility"].max() + 0.004
    for (name, row), text_y in zip(full.iterrows(), label_y):
        point = (row.volatility, row.annual_return)
        size = 46 if name == MULTI_FACTOR else 26
        ax.scatter(*point, color=COLORS[name], s=size, zorder=3)
        ax.annotate(
            f"{LABELS[name]} ({row.sharpe:.2f})",
            point,
            xytext=(label_x, text_y),
            ha="left",
            va="center",
            fontsize=7.2,
            arrowprops=leader_line("#999999", 3),
        )
    ax.set_xlim(full["volatility"].min() - 0.006, full["volatility"].max() + 0.043)
    ax.set_ylim(label_y.min() - 0.006, label_y.max() + 0.006)
    ax.xaxis.set_major_formatter(PERCENT_FORMAT)
    ax.yaxis.set_major_formatter(PERCENT_FORMAT)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized net return")
    ax.grid(**GRID_STYLE)
    ax.set_title("B. Risk and return, Sharpe in brackets")


def plot_downside_risk(net, full):
    fig, grid = new_figure((11, 5.6), [1.2, 1], [1.05, 1], 0.24, 0.42)
    draw_return_histogram(fig.add_subplot(grid[:, 0]), net)
    draw_risk_return(fig.add_subplot(grid[0, 1]), full)

    table_ax = fig.add_subplot(grid[1, 1])
    rows = [
        [
            LABELS[name],
            f"{row.max_drawdown:.1%}",
            f"{row.cvar_95:.1%}",
            "n/a" if name == EQUAL_WEIGHT else f"{row.hit_rate_vs_equal_weight:.0%}",
        ]
        for name, row in full.iterrows()
    ]
    headers = ["Strategy", "Max DD", f"CVaR {1 - CVAR_LEVEL:.0%}", "Months > EW"]
    draw_table(table_ax, headers, rows, [0.36, 0.2, 0.22, 0.22])
    table_ax.set_title("C. Downside risk")
    save_figure(fig, "downside_risk")


def draw_regime_timeline(ax, net, regimes):
    labels = regimes.reindex(net.index).dropna()
    for _, block in labels.groupby((labels != labels.shift()).cumsum()):
        start = block.index[0] - pd.offsets.MonthEnd(1)
        color = REGIME_COLORS[block.iloc[0]]
        ax.axvspan(start, block.index[-1], color=color, linewidth=0)

    growth = value_of_100(net[BENCHMARK])
    ax.plot(growth, color=COLORS[MULTI_FACTOR], linewidth=1.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=REGIME_COLORS[label], ec="#777777", lw=0.5)
        for label in REGIMES
    ]
    ax.legend(
        handles,
        REGIMES,
        ncol=4,
        loc="upper left",
        handlelength=1,
        columnspacing=0.9,
        **LEGEND_FRAME,
    )
    ax.set_xlim(growth.index[0], growth.index[-1])
    ax.set_ylim(growth.min() * 0.94, growth.max() * 1.16)
    ax.yaxis.set_major_formatter(DOLLAR_FORMAT)
    ax.set_ylabel("SPY, value of $100")
    ax.set_title("A. Price-based market regimes")


def draw_regime_heatmap(ax, regime_performance):
    rows = [*FACTORS, MULTI_FACTOR]
    values = regime_performance.loc[rows, REGIMES].astype(float) * 100
    months = regime_performance.loc["months", REGIMES].astype(int)
    x_labels = [f"{label}\nn={months[label]}" for label in REGIMES]
    x_labels.append("Hit rate\nall months")
    y_labels = [LABELS[name] for name in rows]
    draw_heatmap(ax, values, x_labels, y_labels, "{:+.2f}", 0.5)
    ax.set_xlim(-0.5, len(REGIMES) + 0.5)
    for row, name in enumerate(rows):
        hit_rate = regime_performance.loc[name, "hit_rate"]
        ax.text(
            len(REGIMES), row, f"{hit_rate:.0%}", ha="center", va="center", fontsize=8
        )
    ax.set_title("B. Mean monthly return minus Equal Weight, %", pad=30)


def draw_regime_weights(ax, regime_weights):
    bottom = np.zeros(len(REGIMES))
    for ticker, (name, color) in SECTOR_STYLES.items():
        share = regime_weights[ticker].to_numpy()
        ax.bar(
            REGIMES,
            share,
            bottom=bottom,
            color=color,
            width=0.72,
            label=name,
            edgecolor="white",
            linewidth=0.4,
        )
        text_color = "white" if ticker in {"XLE", "XLK", "XLRE"} else "black"
        for position, (height, base) in enumerate(zip(share, bottom)):
            if height >= 0.045:
                y_value = base + height / 2
                ax.text(
                    position,
                    y_value,
                    f"{height:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                )
        bottom += share

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PERCENT_FORMAT)
    ax.set_ylabel("Average Multi-Factor weight")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        handlelength=1,
    )
    ax.set_title("C. Sector allocation by regime")


def plot_market_regimes(net, regimes, regime_performance, regime_weights):
    fig, grid = new_figure((11, 6.4), [1.2, 1], [1, 1.05], 0.22, 0.58)
    draw_regime_timeline(fig.add_subplot(grid[0, 0]), net, regimes)
    draw_regime_heatmap(fig.add_subplot(grid[1, 0]), regime_performance)
    draw_regime_weights(fig.add_subplot(grid[:, 1]), regime_weights)
    save_figure(fig, "market_regimes")


def create_figures(
    scores, net, performance, regimes, regime_performance, regime_weights
):
    plt.rcParams.update(PLOT_STYLE)
    full = performance.query("period == 'full'").set_index("strategy").loc[STRATEGIES]
    plot_sector_scores(scores)
    plot_strategy_performance(net, full)
    plot_downside_risk(net, full)
    plot_market_regimes(net, regimes, regime_performance, regime_weights)
