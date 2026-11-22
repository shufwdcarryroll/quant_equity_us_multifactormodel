
from pathlib import Path
import pandas as pd
import wrds

OUT = Path("data/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

TABLES = {
    "Financial ratios": ("wrdsapps_finratio", "firm_ratio"),
    "Earnings estimates": ("ibes", "statsum_epsus"),
    "S&P 500 constituents": ("comp", "spidx_cst"),
}

db = wrds.Connection()

try:
    with open(OUT / "fundamental_tables.txt", "w") as f:
        for label, (library, table) in TABLES.items():
            print(f"\n{'=' * 60}\n{label}", file=f)
            print(f"TABLE: {library}.{table}", file=f)

            try:
                schema = db.describe_table(
                    library=library, table=table
                )
                print("\nCOLUMNS:", file=f)
                print(schema.to_string(index=False), file=f)

                sample = db.raw_sql(
                    f"SELECT * FROM {library}.{table} LIMIT 3"
                )
                print("\nSAMPLE:", file=f)
                print(sample.to_string(index=False), file=f)

            except Exception as exc:
                print(f"ERROR: {exc}", file=f)

    print("Saved: data/diagnostics/fundamental_tables.txt")

finally:
    db.close()
