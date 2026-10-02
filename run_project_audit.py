"""
One-command reproducibility and quality audit for the project.

Run from the repository root:

    python run_project_audit.py

The audit stops immediately if any required stage fails.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


STAGES = [
    (
        "SOURCE COMPILATION",
        [sys.executable, "-m", "compileall", "-q", "src"],
    ),
    (
        "PRODUCTION MODEL TRAINING",
        [sys.executable, "src/production_model.py"],
    ),
    (
        "BATCH INFERENCE",
        [sys.executable, "src/run_batch_inference.py"],
    ),
    (
        "SQL ANALYSIS",
        [sys.executable, "src/run_sql_analysis.py"],
    ),
    (
        "TIME-SERIES CROSS-VALIDATION",
        [sys.executable, "src/run_time_series_cv.py"],
    ),
    (
        "FINAL HOLDOUT TEST",
        [sys.executable, "src/run_final_test.py"],
    ),
    (
        "INVENTORY SIMULATION",
        [sys.executable, "src/run_inventory_simulation.py"],
    ),
    (
        "INVENTORY SCENARIOS",
        [sys.executable, "src/run_inventory_scenarios.py"],
    ),
    (
        "DEMAND DRIVER AND ERROR ANALYSIS",
        [sys.executable, "src/demand_driver_analysis.py"],
    ),
    (
        "INVENTORY COST ANALYSIS",
        [sys.executable, "src/inventory_cost_analysis.py"],
    ),
    (
        "AUTOMATED TESTS",
        [sys.executable, "-m", "pytest", "-q"],
    ),
]


REQUIRED_ARTIFACTS = [
    # SQL
    "sql/query_results.txt",
    "sql/advanced_sql.sql",

    # Forecasting / validation
    "reports/cv_results_by_fold.csv",
    "reports/cv_results_summary.csv",
    "reports/gradient_boosting_oof_predictions.csv",
    "reports/final_test_predictions.csv",
    "reports/final_test_metrics.csv",

    # Inventory simulation
    "reports/inventory_normal_by_product.csv",
    "reports/inventory_normal_summary.csv",
    "reports/inventory_scenario_by_product.csv",
    "reports/inventory_scenario_summary.csv",

    # Safety-stock sensitivity
    "reports/inventory_z_sensitivity_summary.csv",

    # Inventory cost analysis
    "reports/inventory_cost_sensitivity.csv",

    # Demand-driver / root-cause-style analysis
    "reports/demand_driver_associations.csv",
    "reports/demand_driver_segments.csv",

    # Forecast error analysis
    "reports/forecast_error_by_product.csv",
    "reports/forecast_error_by_category.csv",
    "reports/forecast_error_by_condition.csv",
    "reports/final_test_predictions_with_error_analysis.csv",

    # Production inference
    "models/gradient_boosting_demand_model.joblib",
    "models/feature_columns.txt",
    "reports/batch_predictions.csv",

    # Deployment
    "src/api.py",
    "Dockerfile",
    "requirements-docker.txt",
    ".dockerignore",
]


def run_stage(name, command):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Command:", " ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=ROOT,
    )

    if result.returncode != 0:
        print()
        print("=" * 80)
        print(f"AUDIT FAILED: {name}")
        print("=" * 80)
        print(f"Exit code: {result.returncode}")
        sys.exit(result.returncode)


def check_sql_queries():
    """Verify that all expected SQL questions executed successfully."""

    print()
    print("=" * 80)
    print("SQL QUERY INTEGRITY")
    print("=" * 80)

    results_path = ROOT / "sql/query_results.txt"

    if not results_path.exists():
        print("sql/query_results.txt                         FAIL")
        sys.exit(1)

    text = results_path.read_text(
        encoding="utf-8"
    )

    expected_queries = [
        "-- Q1.",
        "-- Q2.",
        "-- Q3.",
        "-- Q4.",
        "-- Q5.",
        "-- Q6.",
        "-- Q7.",
        "-- Q8.",
        "-- Q9.",
        "-- Q10.",
        "-- Q11.",
    ]

    failed = False

    for query_marker in expected_queries:
        if query_marker not in text:
            print(
                f"{query_marker:<10} FAIL - missing from query results"
            )
            failed = True
        else:
            print(
                f"{query_marker:<10} PASS"
            )

    if "ERROR:" in text:
        print("SQL execution errors                  FAIL")
        failed = True
    else:
        print("SQL execution errors                  PASS")

    if failed:
        print()
        print("SQL QUERY INTEGRITY AUDIT: FAIL")
        sys.exit(1)

    print()
    print("SQL QUERY INTEGRITY AUDIT: PASS")


def check_artifacts():
    print()
    print("=" * 80)
    print("ARTIFACT INTEGRITY")
    print("=" * 80)

    failed = False

    for relative_path in REQUIRED_ARTIFACTS:
        path = ROOT / relative_path

        if (
            path.exists()
            and path.is_file()
            and path.stat().st_size > 0
        ):
            print(
                f"{relative_path:<65} PASS"
            )
        else:
            print(
                f"{relative_path:<65} FAIL"
            )
            failed = True

    if failed:
        print()
        print("ARTIFACT INTEGRITY AUDIT: FAIL")
        sys.exit(1)

    print()
    print("ARTIFACT INTEGRITY AUDIT: PASS")


def main():
    print("=" * 80)
    print("PROJECT REPRODUCIBILITY AND QUALITY AUDIT")
    print("=" * 80)
    print(f"Repository: {ROOT}")

    for name, command in STAGES:
        run_stage(name, command)

    check_sql_queries()

    check_artifacts()

    print()
    print("=" * 80)
    print("PROJECT AUDIT COMPLETE")
    print("=" * 80)

    print("All required stages completed successfully.")
    print()

    print("SOURCE COMPILATION:             PASS")
    print("PRODUCTION MODEL TRAINING:      PASS")
    print("BATCH INFERENCE:                PASS")
    print("SQL ANALYSIS:                   PASS")
    print("SQL QUERY INTEGRITY:            PASS")
    print("TIME-SERIES CV:                 PASS")
    print("FINAL HOLDOUT TEST:             PASS")
    print("INVENTORY SIMULATION:           PASS")
    print("INVENTORY SCENARIOS:            PASS")
    print("DEMAND DRIVER ANALYSIS:         PASS")
    print("INVENTORY COST ANALYSIS:        PASS")
    print("AUTOMATED TESTS:                PASS")
    print("ARTIFACT INTEGRITY:             PASS")
    print()

    print("REPRODUCIBILITY AUDIT:          PASS")


if __name__ == "__main__":
    main()