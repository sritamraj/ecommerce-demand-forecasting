"""
Loads data/sales_clean.csv into a local SQLite database and runs the
business-question queries from sql/sales_analysis.sql.

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

DATA_PATH = ROOT / "data" / "sales_clean.csv"
SQL_PATH = ROOT / "sql" / "sales_analysis.sql"
OUTPUT_PATH = ROOT / "sql" / "query_results.txt"


def main():
    df = pd.read_csv(DATA_PATH)

    conn = sqlite3.connect(":memory:")

    try:
        df.to_sql(
            "sales",
            conn,
            index=False,
            if_exists="replace",
        )

        sql_text = SQL_PATH.read_text(encoding="utf-8")

        # Split into named blocks on "-- Q<number>." markers.
        blocks = re.split(r"(?=-- Q\d+\.)", sql_text)

        output_lines = []

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