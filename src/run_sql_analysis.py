"""
Loads data/sales_clean.csv into a local SQLite database and runs the
business-question queries from sql/sales_analysis.sql, printing real
results (used to sanity-check the .sql file and to populate the README).
SQLite is used purely as a lightweight engine to execute and validate
standard SQL; the queries are portable to Postgres/MySQL/Redshift with
trivial syntax changes (documented in sql/sales_analysis.sql).
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect(":memory:")
df = pd.read_csv("/home/claude/ecommerce-demand-forecasting/data/sales_clean.csv")
df.to_sql("sales", conn, index=False, if_exists="replace")

with open("/home/claude/ecommerce-demand-forecasting/sql/sales_analysis.sql") as f:
    sql_text = f.read()

# split into named blocks on "-- Q<number>." markers
import re
blocks = re.split(r"(?=-- Q\d+\.)", sql_text)
for block in blocks:
    block = block.strip()
    if not block:
        continue
    header = block.splitlines()[0]
    # extract the actual statement (drop comment lines) for execution
    stmt_lines = [l for l in block.splitlines() if not l.strip().startswith("--")]
    stmt = "\n".join(stmt_lines).strip().rstrip(";")
    if not stmt:
        continue
    print("=" * 100)
    print(header)
    print("=" * 100)
    try:
        result = pd.read_sql_query(stmt, conn)
        print(result.head(15).to_string(index=False))
        if len(result) > 15:
            print(f"... ({len(result)} rows total)")
    except Exception as e:
        print("ERROR:", e)
    print()
