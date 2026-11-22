# US Equity Multi-Factor Model — Data & Modeling Guide

**University of Michigan | SIADS 699 Capstone**

This repository contains data extraction scripts, CRSP ETF datasets, sector fundamentals, factor datasets, and initial portfolio outputs for our US sector ETF multi-factor project. **For modeling, start with `data/processed/` rather than downloading the raw data again.**

## 1. Data available

The files below are included in the initial Git commit. Descriptions indicate their intended role; please inspect column names, coverage, and missing values before use.

| File | What to use it for |
| --- | --- |
| `data/raw/crsp_d_etfs_long.csv` | Daily CRSP ETF observations |
| `data/raw/crsp_m_etfs_long.csv` | Monthly CRSP ETF observations |
| `data/raw/crsp_d_returns.csv`, `crsp_m_returns.csv` | Daily/monthly return inputs |
| `data/raw/crsp_d_price.csv`, `crsp_m_price.csv` | Daily/monthly prices |
| `data/raw/crsp_d_volume.csv`, `crsp_m_volume.csv` | Trading volume |
| `data/raw/crsp_d_cap.csv`, `crsp_m_cap.csv` | Capitalization-related fields |
| `data/raw/crsp_d_retx.csv`, `crsp_m_retx.csv` | Return-excluding-distributions fields |
| `data/raw/etf_monthly_returns.csv` | Monthly ETF returns for backtests |
| `data/raw/etf_factor_signals.csv` | ETF-level factor signal inputs |
| `data/raw/security_identifier_review.csv` | ETF identifier checks |
| `data/raw/treasury_monthly_raw_review.csv` | Treasury data review |

## 2. Prepared datasets — start here

| File | Intended modeling use |
| --- | --- |
| `data/processed/sector_factor_panel.csv` | Main sector-by-period factor feature panel |
| `data/processed/factor_panel.csv` | Factor-level prepared panel |
| `data/processed/monthly_sector_factor_scores.csv` | Monthly sector rankings/scores |
| `data/processed/sector_fundamentals_monthly.csv` | Monthly sector fundamentals |
| `data/processed/sector_earnings_revisions.csv` | Earnings revisions signals |
| `data/processed/sector_returns_fundamentals.csv` | Combined return and fundamental inputs |

**Suggested starting pair:** `sector_factor_panel.csv` for features and `etf_monthly_returns.csv` for return targets/backtesting, after checking their dates, ETF identifiers, and alignment. Do not assume that their join keys or target variables are identical.

## 3. Modeling handoff

1. **Inspect the data:** Confirm ETF/sector identifiers, observation dates, available features, missing values, and sample coverage. Check `data/manifest.json` and `data/diagnostics/` for context.
2. **Define the target:** For return prediction, align today's factors with **subsequent** monthly returns. Never train on contemporaneous or future information.
3. **Build baselines:** Equal-weight sector ETFs, individual-factor rankings, and a combined multi-factor ranking portfolio. Compare with SPY where benchmark returns are available.
4. **Try additional models:** Ridge/regularized regression, Random Forest, or XGBoost, subject to sample size and time. Use chronological train/validation/test splits, not random splits.
5. **Evaluate:** Annualized return, volatility, Sharpe ratio, maximum drawdown, turnover, concentration, and transaction-cost-adjusted returns. Assess performance out of sample.

**Important checks:** Avoid look-ahead bias in financial statement publication dates, avoid data leakage when scaling/training, and verify whether CRSP return fields include distributions before selecting the backtest return series.

## 4. Existing results

| File | Contents suggested by filename |
| --- | --- |
| `data/results/target_weights_monthly.csv` | Portfolio target weights |
| `data/results/portfolio_returns_monthly.csv` | Monthly strategy returns |
| `data/results/portfolio_growth_1dollar.csv` | Portfolio growth series |
| `data/results/turnover_monthly_proxy.csv` | Turnover proxy |
| `data/results/performance_summary.csv` | Strategy performance summary |

These are existing outputs to review and validate, **not finalized or independently verified model results**.

## 5. Scripts and quick inspection

- `discover_wrds_data.py` — WRDS data discovery
- `download_fundamentals.py` — fundamental data extraction
- `download_earnings_revisions.py` — revisions data extraction
- `inscpect_fundaments.py` — fundamentals inspection (filename as committed)
- `main.py` — project entry script; review before running

Run from the repository root (with Python and pandas installed):

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

for name in [
    'data/processed/sector_factor_panel.csv',
    'data/processed/monthly_sector_factor_scores.csv',
    'data/processed/sector_fundamentals_monthly.csv',
    'data/processed/sector_earnings_revisions.csv',
    'data/raw/etf_monthly_returns.csv',
]:
    p = Path(name)
    if not p.exists():
        print(f'{name}: not found')
        continue
    df = pd.read_csv(p)
    print(f'\n{name} | rows={len(df)} | columns={len(df.columns)}')
    print('Fields:', list(df.columns))
    print('Missing values (top 5):', df.isna().sum().sort_values(ascending=False).head().to_dict())
PY
```

## 6. Team notes

The intention is to let the modeling team work directly from the prepared data, while we jointly agree on feature definitions, time alignment, models, and evaluation. Please flag missing fields or any required changes in our group chat before rebuilding the extraction pipeline.

**Data access:** WRDS/CRSP-derived datasets may be subject to redistribution restrictions. Confirm authorization before sharing any licensed raw or derived data in a public repository. Never commit WRDS credentials or GitHub tokens.
