import pandas as pd
import pytest

from src.inventory_simulation import simulate_inventory


def make_test_product():
    dates = pd.date_range("2025-05-02", periods=20, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "actual_demand": [100.0] * 20,
            "forecast": [100.0] * 20,
        }
    )


def test_inventory_simulation_returns_expected_core_metrics():
    df = make_test_product()

    result = simulate_inventory(
        df_product=df,
        error_std=10.0,
    )

    required_keys = {
        "avg_inventory",
        "stockout_days",
        "cycle_service_level_%",
        "lost_units",
        "total_units_ordered",
        "avg_daily_forecast",
        "protection_period_days",
        "reorder_point",
        "target_level",
        "safety_stock",
        "inventory_capacity",
    }

    assert required_keys.issubset(result.keys())
    assert result["avg_daily_forecast"] == 100.0
    assert result["protection_period_days"] == 10
    assert result["safety_stock"] > 0
    assert result["reorder_point"] > 0


def test_protection_period_equals_lead_time_plus_review_period():
    df = make_test_product()

    result = simulate_inventory(
        df_product=df,
        error_std=10.0,
        lead_time=4,
        review_period=7,
    )

    assert result["protection_period_days"] == 11


def test_scenario_multiplier_changes_actual_demand_only():
    df = make_test_product()

    normal = simulate_inventory(
        df_product=df,
        error_std=10.0,
        demand_multiplier=1.0,
    )

    stressed = simulate_inventory(
        df_product=df,
        error_std=10.0,
        demand_multiplier=1.2,
    )

    assert normal["avg_daily_forecast"] == stressed["avg_daily_forecast"]
    assert normal["reorder_point"] == stressed["reorder_point"]
    assert stressed["lost_units"] > normal["lost_units"]


def test_capacity_constraint_is_applied():
    df = make_test_product()

    result = simulate_inventory(
        df_product=df,
        error_std=10.0,
        capacity_fraction=0.70,
    )

    assert result["inventory_capacity"] is not None
    assert result["inventory_capacity"] < result["target_level"]
    assert result["avg_inventory"] <= result["inventory_capacity"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"demand_multiplier": 0},
        {"demand_multiplier": -1},
        {"lead_time": -1},
        {"review_period": 0},
        {"review_period": -1},
        {"capacity_fraction": 0},
        {"capacity_fraction": -0.5},
        {"capacity_fraction": 1.1},
    ],
)
def test_invalid_inventory_parameters_raise_value_error(kwargs):
    df = make_test_product()

    with pytest.raises(ValueError):
        simulate_inventory(
            df_product=df,
            error_std=10.0,
            **kwargs,
        )