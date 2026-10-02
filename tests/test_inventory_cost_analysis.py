from pathlib import Path

import pandas as pd

from src.inventory_cost_analysis import (
    HOLDING_COST_PER_UNIT_DAY,
    ORDERING_COST_PER_UNIT,
    STOCKOUT_COST_PER_LOST_UNIT,
    calculate_costs,
    validate_input,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def test_cost_constants():
    assert HOLDING_COST_PER_UNIT_DAY == 1.00
    assert STOCKOUT_COST_PER_LOST_UNIT == 20.00
    assert ORDERING_COST_PER_UNIT == 0.10


def test_validate_input():
    path = (
        REPORTS
        / "inventory_z_sensitivity_summary.csv"
    )

    df = pd.read_csv(path)

    validate_input(df)


def test_cost_calculation():
    path = (
        REPORTS
        / "inventory_z_sensitivity_summary.csv"
    )

    df = pd.read_csv(path)

    result = calculate_costs(df)

    assert len(result) == 3

    assert set(
        round(float(x), 2)
        for x in result["z_score"]
    ) == {1.28, 1.65, 2.05}

    for column in [
        "holding_cost",
        "stockout_cost",
        "ordering_cost",
        "total_cost",
    ]:
        assert result[column].notna().all()
        assert (result[column] >= 0).all()


def test_total_cost_formula():
    path = (
        REPORTS
        / "inventory_z_sensitivity_summary.csv"
    )

    df = pd.read_csv(path)

    result = calculate_costs(df)

    expected = (
        result["holding_cost"]
        + result["stockout_cost"]
        + result["ordering_cost"]
    )

    assert (
        abs(
            result["total_cost"]
            - expected
        )
        < 0.01
    ).all()


def test_cost_report_exists():
    path = (
        REPORTS
        / "inventory_cost_sensitivity.csv"
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_expected_cost_values():
    path = (
        REPORTS
        / "inventory_cost_sensitivity.csv"
    )

    df = pd.read_csv(path)

    expected = {
        1.28: 999710.97,
        1.65: 1043220.74,
        2.05: 1094636.25,
    }

    for z, expected_cost in expected.items():

        row = df[
            df["z_score"].round(2) == z
        ]

        assert len(row) == 1

        actual_cost = float(
            row["total_cost"].iloc[0]
        )

        assert abs(
            actual_cost - expected_cost
        ) < 0.01