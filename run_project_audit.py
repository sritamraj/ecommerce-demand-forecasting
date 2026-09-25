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
        "AUTOMATED TESTS",
        [sys.executable, "-m", "pytest", "-q"],
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
]


REQUIRED_ARTIFACTS = [
    "sql/query_results.txt",
    "reports/cv_results_by_fold.csv",
    "reports/cv_results_summary.csv",
    "reports/gradient_boosting_oof_predictions.csv",
    "reports/final_test_predictions.csv",
    "reports/final_test_metrics.csv",
    "reports/inventory_normal_by_product.csv",
    "reports/inventory_normal_summary.csv",
    "reports/inventory_scenario_by_product.csv",
    "reports/inventory_scenario_summary.csv",
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


def check_artifacts():
    print()
    print("=" * 80)
    print("ARTIFACT INTEGRITY")
    print("=" * 80)

    failed = False

    for relative_path in REQUIRED_ARTIFACTS:
        path = ROOT / relative_path

        if path.exists() and path.is_file() and path.stat().st_size > 0:
            print(f"{relative_path:<55} PASS")
        else:
            print(f"{relative_path:<55} FAIL")
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

    check_artifacts()

    print()
    print("=" * 80)
    print("PROJECT AUDIT COMPLETE")
    print("=" * 80)
    print("All required stages completed successfully.")
    print()
    print("SOURCE COMPILATION:       PASS")
    print("AUTOMATED TESTS:          PASS")
    print("SQL ANALYSIS:             PASS")
    print("TIME-SERIES CV:           PASS")
    print("FINAL HOLDOUT TEST:       PASS")
    print("INVENTORY SIMULATION:     PASS")
    print("INVENTORY SCENARIOS:      PASS")
    print("ARTIFACT INTEGRITY:       PASS")
    print()
    print("REPRODUCIBILITY AUDIT:    PASS")


if __name__ == "__main__":
    main()