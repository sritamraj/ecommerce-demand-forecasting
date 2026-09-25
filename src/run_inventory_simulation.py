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

SUMMARY_PATH = "reports/inventory_normal_summary.csv"
PRODUCT_PATH = "reports/inventory_normal_by_product.csv"


def load_inputs():
    oof = pd.read_csv(
        OOF_PATH,
        parse_dates=["date"],
    )

    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["date"],
    )

    required_oof = {
        "product_id",
        "date",
        "error",
    }

    required_test = {
        "product_id",
        "date",
        "quantity",
        "prediction",
    }

    if not required_oof.issubset(oof.columns):
        raise ValueError(
            f"OOF file missing columns: "
            f"{sorted(required_oof - set(oof.columns))}"
        )

    if not required_test.issubset(test.columns):
        raise ValueError(
            f"Final-test file missing columns: "
            f"{sorted(required_test - set(test.columns))}"
        )

    return oof, test


def build_error_statistics(oof):
    error_stats = (
        oof.groupby("product_id")["error"]
        .agg(
            error_std="std",
            oof_rows="size",
        )
        .reset_index()
    )

    if error_stats["error_std"].isna().any():
        raise ValueError(
            "At least one product has missing OOF error_std."
        )

    return error_stats


def run_normal_scenario(oof, test):
    error_stats = build_error_statistics(oof)

    test = test.rename(
        columns={
            "quantity": "actual_demand",
            "prediction": "forecast",
        }
    )

    results = []

    for product_id, group in test.groupby("product_id"):

        stats = error_stats[
            error_stats["product_id"] == product_id
        ]

        if len(stats) != 1:
            raise ValueError(
                f"Expected exactly one error estimate "
                f"for {product_id}."
            )

        error_std = float(
            stats["error_std"].iloc[0]
        )

        result = simulate_inventory(
            group,
            error_std=error_std,
            demand_multiplier=1.0,
        )

        result["product_id"] = product_id
        result["oof_error_std"] = error_std
        result["oof_rows"] = int(
            stats["oof_rows"].iloc[0]
        )

        results.append(result)

    product_results = (
        pd.DataFrame(results)
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    return product_results


def build_portfolio_summary(product_results):
    summary = {
        "scenario": "A - Normal demand",
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
            product_results["lost_units"].sum(),
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
        "mean_safety_stock": round(
            product_results[
                "safety_stock"
            ].mean(),
            1,
        ),
        "mean_oof_error_std": round(
            product_results[
                "oof_error_std"
            ].mean(),
            2,
        ),
    }

    return pd.DataFrame([summary])


def main():
    print("=" * 70)
    print("INVENTORY SIMULATION — NORMAL DEMAND")
    print("=" * 70)

    oof, test = load_inputs()

    print(f"OOF rows:        {len(oof):,}")
    print(f"Final-test rows: {len(test):,}")

    print(
        f"Final-test date range: "
        f"{test['date'].min().date()} "
        f"to "
        f"{test['date'].max().date()}"
    )

    print(
        f"Products: {test['product_id'].nunique()}"
    )

    product_results = run_normal_scenario(
        oof,
        test,
    )

    summary = build_portfolio_summary(
        product_results
    )

    product_results.to_csv(
        PRODUCT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("NORMAL-DEMAND INVENTORY RESULTS")
    print("-" * 70)

    row = summary.iloc[0]

    print(
        f"Mean product cycle service level : "
        f"{row['mean_product_cycle_service_level_%']:.2f}%"
    )

    print(
        f"Products below 90% service       : "
        f"{int(row['products_below_90pct_service'])}"
    )

    print(
        f"Products below 95% service       : "
        f"{int(row['products_below_95pct_service'])}"
    )

    print(
        f"Total lost units                  : "
        f"{row['total_lost_units']:,.1f}"
    )

    print(
        f"Total units ordered               : "
        f"{row['total_units_ordered']:,.1f}"
    )

    print(
        f"Total average inventory           : "
        f"{row['total_average_inventory']:,.1f}"
    )

    print(
        f"Mean safety stock                 : "
        f"{row['mean_safety_stock']:,.1f}"
    )

    print()
    print(f"Saved: {PRODUCT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()