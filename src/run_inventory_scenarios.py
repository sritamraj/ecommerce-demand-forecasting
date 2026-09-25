import sys

sys.path.insert(0, ".")

import pandas as pd

from src.inventory_simulation import (
    LEAD_TIME_DAYS,
    REVIEW_PERIOD_DAYS,
    SERVICE_LEVEL_Z,
    simulate_inventory,
)


OOF_PATH = "reports/gradient_boosting_oof_predictions.csv"
TEST_PATH = "reports/final_test_predictions.csv"

PRODUCT_OUTPUT = "reports/inventory_scenario_by_product.csv"
SUMMARY_OUTPUT = "reports/inventory_scenario_summary.csv"


def load_inputs():
    oof = pd.read_csv(
        OOF_PATH,
        parse_dates=["date"],
    )

    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["date"],
    )

    test = test.rename(
        columns={
            "quantity": "actual_demand",
            "prediction": "forecast",
        }
    )

    return oof, test


def build_error_stats(oof):
    stats = (
        oof.groupby("product_id")["error"]
        .agg(
            error_std="std",
            oof_rows="size",
        )
        .reset_index()
    )

    if stats["error_std"].isna().any():
        raise ValueError(
            "Missing OOF error standard deviation."
        )

    return stats


def run_scenario(
    test,
    error_stats,
    demand_multiplier=1.0,
    spike=False,
    capacity_fraction=None,
):
    rows = []

    for product_id, group in test.groupby("product_id"):

        stats = error_stats[
            error_stats["product_id"] == product_id
        ]

        if len(stats) != 1:
            raise ValueError(
                f"Expected one uncertainty estimate "
                f"for {product_id}."
            )

        error_std = float(
            stats["error_std"].iloc[0]
        )

        d = (
            group
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        if spike:
            midpoint = len(d) // 2

            spike_end = min(
                midpoint + 5,
                len(d),
            )

            d.loc[
                midpoint:spike_end - 1,
                "actual_demand",
            ] *= 3.0

        result = simulate_inventory(
            d,
            error_std=error_std,
            demand_multiplier=demand_multiplier,
            capacity_fraction=capacity_fraction,
        )

        result["product_id"] = product_id
        result["oof_error_std"] = error_std
        result["capacity_fraction"] = capacity_fraction

        rows.append(result)

    return (
        pd.DataFrame(rows)
        .sort_values("product_id")
        .reset_index(drop=True)
    )


def summarize(
    product_results,
    scenario_name,
):
    capacity_fraction = product_results[
        "capacity_fraction"
    ].iloc[0]

    return {
        "scenario": scenario_name,
        "products": int(
            product_results["product_id"].nunique()
        ),
        "test_days": 60,
        "lead_time_days": LEAD_TIME_DAYS,
        "review_period_days": REVIEW_PERIOD_DAYS,
        "protection_period_days": (
            LEAD_TIME_DAYS
            + REVIEW_PERIOD_DAYS
        ),
        "service_level_z": SERVICE_LEVEL_Z,
        "capacity_fraction": (
            round(float(capacity_fraction), 2)
            if pd.notna(capacity_fraction)
            else None
        ),
        "mean_product_cycle_service_level_%": round(
            product_results[
                "cycle_service_level_%"
            ].mean(),
            2,
        ),
        "products_below_90pct_service": int(
            (
                product_results[
                    "cycle_service_level_%"
                ] < 90
            ).sum()
        ),
        "products_below_95pct_service": int(
            (
                product_results[
                    "cycle_service_level_%"
                ] < 95
            ).sum()
        ),
        "total_lost_units": round(
            product_results[
                "lost_units"
            ].sum(),
            1,
        ),
        "total_units_ordered": round(
            product_results[
                "total_units_ordered"
            ].sum(),
            1,
        ),
        "total_average_inventory": round(
            product_results[
                "avg_inventory"
            ].sum(),
            1,
        ),
    }


def main():

    print("=" * 70)
    print("INVENTORY STRESS TESTS")
    print("=" * 70)

    oof, test = load_inputs()

    error_stats = build_error_stats(oof)

    scenarios = [
        (
            "A - Normal demand",
            1.00,
            False,
            None,
        ),
        (
            "B - Demand +10%",
            1.10,
            False,
            None,
        ),
        (
            "C - Demand +20%",
            1.20,
            False,
            None,
        ),
        (
            "D - Unexpected 3x five-day spike",
            1.00,
            True,
            None,
        ),
        (
            "E - Inventory capacity at 70% of target",
            1.00,
            False,
            0.70,
        ),
    ]

    all_product_results = []
    summary_rows = []

    for (
        name,
        multiplier,
        spike,
        capacity_fraction,
    ) in scenarios:

        print()
        print(f"Running: {name}")

        product_results = run_scenario(
            test,
            error_stats,
            demand_multiplier=multiplier,
            spike=spike,
            capacity_fraction=capacity_fraction,
        )

        product_results["scenario"] = name

        all_product_results.append(
            product_results
        )

        summary_rows.append(
            summarize(
                product_results,
                name,
            )
        )

    product_results_all = pd.concat(
        all_product_results,
        ignore_index=True,
    )

    scenario_summary = pd.DataFrame(
        summary_rows
    )

    product_results_all.to_csv(
        PRODUCT_OUTPUT,
        index=False,
    )

    scenario_summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("SCENARIO SUMMARY")
    print("-" * 70)

    print(
        scenario_summary[
            [
                "scenario",
                "capacity_fraction",
                "mean_product_cycle_service_level_%",
                "products_below_90pct_service",
                "products_below_95pct_service",
                "total_lost_units",
                "total_units_ordered",
                "total_average_inventory",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {PRODUCT_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()