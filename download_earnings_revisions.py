"""Download sector-level analyst EPS-revision breadth from WRDS.

Inputs: WRDS ibes.statsum_epsus and wrdsapps_finratio.firm_ratio.
Outputs: data/processed/sector_earnings_revisions.csv and, when available,
         data/processed/sector_factor_panel.csv.

Only aggregated sector observations are downloaded; no raw analyst or company
records are saved. This is a broad GICS sector proxy, not ETF holdings data.
"""
from pathlib import Path
import pandas as pd
import wrds

START = "2018-01-01"
END_EXCLUSIVE = "2026-01-01"  # Extend only after checking available ETF data.
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = {
    "10": "XLE", "15": "XLB", "20": "XLI", "25": "XLY",
    "30": "XLP", "35": "XLV", "40": "XLF", "45": "XLK",
    "50": "XLC", "55": "XLU", "60": "XLRE",
}

# IBES statsum_epsus is a monthly snapshot of analyst consensus. FY1 ('1')
# means next annual EPS; numup and numdown count upward/downward revisions.
# For mapping only, use WRDS financial-ratio GICS assignment from PRIOR month:
# that prevents a later-in-month company label from leaking into a signal.
SQL = """
WITH fin_map AS (
    SELECT DISTINCT ON (
        f.cusip, date_trunc('month', f.public_date)
    )
        f.cusip,
        date_trunc('month', f.public_date)::date AS map_month,
        f.gsector
    FROM wrdsapps_finratio.firm_ratio f
    WHERE f.public_date >= (%(start)s::date - INTERVAL '1 month')
      AND f.public_date < %(end)s::date
      AND f.cusip IS NOT NULL
      AND f.gsector IN ('10','15','20','25','30','35','40','45','50','55','60')
    ORDER BY
        f.cusip, date_trunc('month', f.public_date),
        f.public_date DESC
),
ibes_monthly AS (
    SELECT DISTINCT ON (
        i.cusip, date_trunc('month', i.statpers)
    )
        i.cusip,
        i.statpers,
        date_trunc('month', i.statpers)::date AS ibes_month,
        i.numup,
        i.numdown,
        i.numest
    FROM ibes.statsum_epsus i
    WHERE i.statpers >= %(start)s::date
      AND i.statpers < %(end)s::date
      AND i.cusip IS NOT NULL
      AND i.measure = 'EPS'
      AND i.fiscalp = 'ANN'
      AND i.fpi = '1'
      AND i.estflag = 'P'
      AND i.curcode = 'USD'
      AND i.usfirm = 1
      AND i.numest >= 1
    ORDER BY
        i.cusip, date_trunc('month', i.statpers),
        i.statpers DESC, i.numest DESC
),
joined AS (
    SELECT
        (i.ibes_month + INTERVAL '1 month - 1 day')::date AS date,
        f.gsector,
        i.cusip,
        CASE WHEN i.numup >= 0 THEN i.numup ELSE NULL END AS numup,
        CASE WHEN i.numdown >= 0 THEN i.numdown ELSE NULL END AS numdown
    FROM ibes_monthly i
    INNER JOIN fin_map f
      ON i.cusip = f.cusip
     AND f.map_month = (i.ibes_month - INTERVAL '1 month')::date
)
SELECT
    date,
    gsector,
    COUNT(DISTINCT cusip) AS firm_count,
    SUM(COALESCE(numup, 0)) AS revisions_up,
    SUM(COALESCE(numdown, 0)) AS revisions_down,
    COUNT(*) FILTER (WHERE numup IS NOT NULL OR numdown IS NOT NULL)
        AS firms_with_revision_fields
FROM joined
GROUP BY date, gsector
ORDER BY date, gsector
"""


def main():
    print("Connecting to WRDS...")
    db = wrds.Connection()
    try:
        required = {
            "ibes": "statsum_epsus",
            "wrdsapps_finratio": "firm_ratio",
        }
        for lib, table in required.items():
            if table not in db.list_tables(library=lib):
                raise RuntimeError(f"Missing WRDS table {lib}.{table}")

        print("Aggregating monthly EPS revisions on WRDS; this may take a little time...")
        result = db.raw_sql(
            SQL,
            params={"start": START, "end": END_EXCLUSIVE},
            date_cols=["date"],
        )
    finally:
        db.close()

    if result.empty:
        raise RuntimeError(
            "No joined records. Check IBES filters, prior-month sector mapping, "
            "and available dates before changing the methodology."
        )

    result["gsector"] = result["gsector"].astype(str).str.zfill(2)
    result["ticker"] = result["gsector"].map(SECTOR_ETFS)
    for c in ["revisions_up", "revisions_down", "firm_count", "firms_with_revision_fields"]:
        result[c] = pd.to_numeric(result[c], errors="coerce")

    total = result["revisions_up"] + result["revisions_down"]
    result["eps_revision_breadth"] = (
        (result["revisions_up"] - result["revisions_down"])
        .div(total.where(total > 0))
    )
    # The ratio is undefined when no upward or downward revisions were counted.
    # Never replace this with zero: no activity is different from balanced activity.
    result = result[
        ["date", "ticker", "gsector", "firm_count", "firms_with_revision_fields",
         "revisions_up", "revisions_down", "eps_revision_breadth"]
    ].sort_values(["date", "ticker"])

    output = OUT / "sector_earnings_revisions.csv"
    result.to_csv(output, index=False)
    print(f"\nSaved {output} ({len(result):,} sector-month rows)")

    coverage = result.groupby("ticker").agg(
        months=("date", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
        months_with_signal=("eps_revision_breadth", "count"),
        median_firms=("firm_count", "median"),
    ).sort_index()
    print("\nCoverage (inspect for missing sectors or sparse history):")
    print(coverage.to_string())
    print("\nMonth range:", result["date"].min(), "to", result["date"].max())

    existing = OUT / "sector_returns_fundamentals.csv"
    if existing.exists():
        main_panel = pd.read_csv(existing)
        if {"date", "ticker"}.issubset(main_panel.columns):
            main_panel["date"] = pd.to_datetime(main_panel["date"])
            new = result.drop(columns=["gsector"]).copy()
            merged = main_panel.merge(
                new, on=["date", "ticker"], how="left", validate="many_to_one"
            )
            merged_path = OUT / "sector_factor_panel.csv"
            merged.to_csv(merged_path, index=False)
            print(f"\nSaved {merged_path} ({len(merged):,} rows)")
            print("EPS breadth non-null on merged panel:", merged["eps_revision_breadth"].notna().sum())
        else:
            print("\nSkipped merge: existing file needs date and ticker columns.")
    else:
        print("\nNo existing fundamentals panel found; sector revisions saved independently.")

    print("\nNOTE: This is a broad-sector proxy linked by 8-character CUSIP, not exact ETF holdings.")
    print("NOTE: Month t signals are for trades in month t+1, not same-month returns.")
    print("NOTE: Validate coverage and potential security identifier changes before final research use.")


if __name__ == "__main__":
    main()
