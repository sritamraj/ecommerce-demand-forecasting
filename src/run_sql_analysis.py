"""
Loads project data into a local SQLite database and runs the
business-question queries from:

    sql/sales_analysis.sql       -> Q1-Q8
    sql/advanced_sql.sql         -> Q9-Q11

SQLite tables:

    sales              -> data/sales_clean.csv
    daily_product_panel -> data/model_features.csv

Results are printed to the console and saved to:
    sql/query_results.txt

SQLite is used purely as a lightweight engine to execute and validate
standard SQL queries.
"""

from pathlib import Path
import re
import sqlite3

import pandas as pd


# Resolve paths relative to the repository root so the script is portable.
ROOT = Path(__file__).resolve().parents[1]

SALES_DATA_PATH = ROOT / "data" / "sales_clean.csv"
PANEL_DATA_PATH = ROOT / "data" / "model_features.csv"

SQL_PATHS = [
    ROOT / "sql" / "sales_analysis.sql",
    ROOT / "sql" / "advanced_sql.sql",
]

OUTPUT_PATH = ROOT / "sql" / "query_results.txt"


def run_sql_file(sql_path, conn, output_lines):
    """Execute all numbered SQL question blocks from one SQL file."""

    sql_text = sql_path.read_text(encoding="utf-8")

    # Split into named blocks on "-- Q<number>." markers.
    blocks = re.split(r"(?=-- Q\d+\.)", sql_text)

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        header = block.splitlines()[0]

        # Remove SQL comment lines before execution.
        stmt_lines = [
            line
            for line in block.splitlines()
            if not line.strip().startswith("--")
        ]

        stmt = "\n".join(stmt_lines).strip().rstrip(";")

        if not stmt:
            continue

        separator = "=" * 100

        output_lines.append(separator)
        output_lines.append(header)
        output_lines.append(separator)

        try:
            result = pd.read_sql_query(stmt, conn)

            output_lines.append(
                result.head(15).to_string(index=False)
            )

            if len(result) > 15:
                output_lines.append(
                    f"... ({len(result)} rows total)"
                )

        except Exception as exc:
            output_lines.append(f"ERROR: {exc}")

        output_lines.append("")


def main():
    # -------------------------------------------------------------
    # 1. Load source datasets
    # -------------------------------------------------------------

    sales_df = pd.read_csv(
        SALES_DATA_PATH,
        parse_dates=["date"],
    )

    panel_df = pd.read_csv(
        PANEL_DATA_PATH,
        parse_dates=["date"],
    )

    # -------------------------------------------------------------
    # 2. Create in-memory SQLite database
    # -------------------------------------------------------------

    conn = sqlite3.connect(":memory:")

    try:
        # Raw/cleaned transaction-level sales table.
        sales_df.to_sql(
            "sales",
            conn,
            index=False,
            if_exists="replace",
        )

        # Complete daily product-level panel used by forecasting.
        panel_df.to_sql(
            "daily_product_panel",
            conn,
            index=False,
            if_exists="replace",
        )

        output_lines = []

        # ---------------------------------------------------------
        # 3. Validate required SQL files
        # ---------------------------------------------------------

        for sql_path in SQL_PATHS:
            if not sql_path.exists():
                raise FileNotFoundError(
                    f"SQL file not found: {sql_path}"
                )

            output_lines.append("=" * 100)
            output_lines.append(
                f"SQL FILE: {sql_path.relative_to(ROOT)}"
            )
            output_lines.append("=" * 100)
            output_lines.append("")

            run_sql_file(
                sql_path,
                conn,
                output_lines,
            )

        # ---------------------------------------------------------
        # 4. Save combined SQL results
        # ---------------------------------------------------------

        output_text = "\n".join(output_lines)

        print(output_text)

        OUTPUT_PATH.write_text(
            output_text,
            encoding="utf-8",
        )

        print(f"Saved: {OUTPUT_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()