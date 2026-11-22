"""Build compact monthly sector value/quality signals from WRDS Financial Ratios.

Run from the Capstone project root:
    python download_fundamentals.py

The WRDS server aggregates company-level records; no raw company data is saved.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import wrds

START_DATE = '2018-01-01'
END_DATE = '2026-11-21'   # exclusive
MIN_COMPANIES = 5
OUT = Path('data/processed')
OUT.mkdir(parents=True, exist_ok=True)

# Current GICS sector scheme. Historical classifications in the WRDS table
# are used as supplied; this is NOT an ETF-constituent valuation series.
SECTORS = {
    '10': ('XLE', 'Energy'),
    '15': ('XLB', 'Materials'),
    '20': ('XLI', 'Industrials'),
    '25': ('XLY', 'Consumer Discretionary'),
    '30': ('XLP', 'Consumer Staples'),
    '35': ('XLV', 'Health Care'),
    '40': ('XLF', 'Financials'),
    '45': ('XLK', 'Information Technology'),
    '50': ('XLC', 'Communication Services'),
    '55': ('XLU', 'Utilities'),
    '60': ('XLRE', 'Real Estate'),
}

# Exclude implausible extreme or meaningless values from each metric's
# median. Filters are research choices, not definitions from WRDS.
METRICS = {
    'book_to_market': ('bm', 0.02, 10.0),
    'pe': ('pe_exi', 0.0, 150.0),
    'dividend_yield': ('divyield', 0.0, 0.30),
    'roe': ('roe', -2.0, 2.0),
    'roa': ('roa', -1.0, 1.0),
    'gross_profitability': ('gprof', -1.0, 3.0),
    'debt_to_assets': ('debt_at', 0.0, 3.0),
}

# Aggregate on WRDS: just 11 x ~100 months, rather than millions of rows.
selects = [
    "date_trunc('month', public_date)::date AS month_start",
    'gsector',
    'COUNT(DISTINCT gvkey) AS companies',
]
for alias, (field, low, high) in METRICS.items():
    # PostgreSQL percentile_cont ignores NULLs.
    selects.append(
        f"percentile_cont(0.5) WITHIN GROUP "
        f"(ORDER BY CASE WHEN {field} >= {low} "
        f"AND {field} <= {high} THEN {field} END) AS {alias}"
    )

sql = f"""
SELECT {', '.join(selects)}
FROM wrdsapps_finratio.firm_ratio
WHERE public_date >= %(start)s
  AND public_date < %(end)s
  AND gsector IN ({', '.join(repr(k) for k in SECTORS)})
GROUP BY 1, 2
ORDER BY 1, 2
"""

print('Connecting to WRDS...')
db = wrds.Connection()
try:
    df = db.raw_sql(sql, params={'start': START_DATE, 'end': END_DATE},
                    date_cols=['month_start'])
finally:
    db.close()

if df.empty:
    raise RuntimeError('WRDS returned no sector financial ratios. Check subscription and dates.')

df['gsector'] = df['gsector'].astype(str).str.zfill(2)
df = df[df['gsector'].isin(SECTORS)].copy()
df['ticker'] = df['gsector'].map(lambda x: SECTORS[x][0])
df['sector'] = df['gsector'].map(lambda x: SECTORS[x][1])
df['date'] = df['month_start'].dt.to_period('M').dt.to_timestamp('M')
df.drop(columns='month_start', inplace=True)

for col in METRICS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['companies'] = pd.to_numeric(df['companies'], errors='coerce')
metric_cols = list(METRICS)
df.loc[df['companies'] < MIN_COMPANIES, metric_cols] = np.nan

# Sector cross-sectional ranks at each month's close. High = best.
# The composite is deliberately simple and transparent for a student project.
def rank_by_date(frame, column, higher_is_better=True):
    return frame.groupby('date')[column].rank(
        pct=True, ascending=higher_is_better
    )

# rank(ascending=True) assigns 1.0 to largest value.
df['value_bm_score'] = rank_by_date(df, 'book_to_market', True)
df['value_pe_score'] = rank_by_date(df, 'pe', False)
df['value_yield_score'] = rank_by_date(df, 'dividend_yield', True)
df['value_score'] = df[
    ['value_bm_score', 'value_pe_score', 'value_yield_score']
].mean(axis=1, skipna=True)

df['quality_roe_score'] = rank_by_date(df, 'roe', True)
df['quality_roa_score'] = rank_by_date(df, 'roa', True)
df['quality_gprof_score'] = rank_by_date(df, 'gross_profitability', True)
df['quality_leverage_score'] = rank_by_date(df, 'debt_to_assets', False)
df['quality_score'] = df[
    ['quality_roe_score', 'quality_roa_score',
     'quality_gprof_score', 'quality_leverage_score']
].mean(axis=1, skipna=True)

outfile = OUT / 'sector_fundamentals_monthly.csv'
df.sort_values(['date', 'ticker']).to_csv(outfile, index=False)
print(f'\nSaved {outfile} ({len(df):,} rows)')
print('Coverage:')
print(df.groupby('ticker').agg(
    months=('date','nunique'),
    first=('date','min'),
    last=('date','max'),
    value_nonnull=('value_score','count'),
    quality_nonnull=('quality_score','count')
).to_string())

# Optionally prepare a tidy file aligned with ETF return months.
returns_file = Path('data/raw/etf_monthly_returns.csv')
if not returns_file.exists():
    returns_file = Path('data/raw/crsp_m_returns.csv')

if returns_file.exists():
    ret = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    ret.index = pd.DatetimeIndex(ret.index).to_period('M').to_timestamp('M')
    ret.index.name = 'date'
    ret_long = ret.stack(dropna=True).rename('etf_return').reset_index()
    ret_long.columns = ['date','ticker','etf_return']
    combined = ret_long.merge(
        df.drop(columns=['sector']), on=['date','ticker'], how='left'
    )
    out2 = OUT / 'sector_returns_fundamentals.csv'
    combined.sort_values(['date','ticker']).to_csv(out2, index=False)
    print(f'Saved {out2} ({len(combined):,} rows)')
    print('NOTE: Signals formed at month-end t must trade in month t+1, not month t.')
else:
    print('ETF returns file not found; fundamental signals saved independently.')

print('NOTE: These are broad GICS sector medians, NOT precise ETF holdings-level fundamentals.')
print('NOTE: A score being present is not proof of point-in-time coverage or an investable backtest.')
