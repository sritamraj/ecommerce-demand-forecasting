import pandas as pd


INPUT_PATH = (
    "reports/inventory_z_sensitivity_summary.csv"
)

OUTPUT_PATH = (
    "reports/inventory_cost_sensitivity.csv"
)


# ------------------------------------------------------------------
# Illustrative business-cost assumptions.
# These are project assumptions, NOT Amazon's actual costs.
# ------------------------------------------------------------------

HOLDING_COST_PER_UNIT_DAY = 1.00
STOCKOUT_COST_PER_LOST_UNIT = 20.00
ORDERING_COST_PER_UNIT = 0.10


def calculate_costs(df):
    result = df.copy()

    result["holding_cost"] = (
        result["total_average_inventory"]
        * result["test_days"]
        * HOLDING_COST_PER_UNIT_DAY
    )

    result["stockout_cost"] = (
        result["total_lost_units"]
        * STOCKOUT_COST_PER_LOST_UNIT
    )

    result["ordering_cost"] = (
        result["total_units_ordered"]
        * ORDERING_COST_PER_UNIT
    )

    result["total_cost"] = (
        result["holding_cost"]
        + result["stockout_cost"]
        + result["ordering_cost"]
    )

    result["holding_cost"] = result[
        "holding_cost"
    ].round(2)

    result["stockout_cost"] = result[
        "stockout_cost"
    ].round(2)

    result["ordering_cost"] = result[
        "ordering_cost"
    ].round(2)

    result["total_cost"] = result[
        "total_cost"
    ].round(2)

    return result


def validate_input(df):
    required_columns = {
        "z_score",
        "test_days",
        "total_lost_units",
        "total_units_ordered",
        "total_average_inventory",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    expected_z = {1.28, 1.65, 2.05}

    actual_z = set(
        round(float(value), 2)
        for value in df["z_score"]
    )

    if actual_z != expected_z:
        raise ValueError(
            f"Unexpected z-score values: {sorted(actual_z)}"
        )


def main():

    print("=" * 70)
    print("INVENTORY COST SENSITIVITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    validate_input(df)

    result = calculate_costs(df)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("COST ASSUMPTIONS")
    print("-" * 70)
    print(
        f"Holding cost/unit/day: "
        f"₹{HOLDING_COST_PER_UNIT_DAY:.2f}"
    )
    print(
        f"Stockout cost/lost unit: "
        f"₹{STOCKOUT_COST_PER_LOST_UNIT:.2f}"
    )
    print(
        f"Ordering cost/unit: "
        f"₹{ORDERING_COST_PER_UNIT:.2f}"
    )

    print()
    print("COST SENSITIVITY")
    print("-" * 70)

    print(
        result[
            [
                "z_score",
                "mean_product_cycle_service_level_%",
                "total_lost_units",
                "total_average_inventory",
                "holding_cost",
                "stockout_cost",
                "ordering_cost",
                "total_cost",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()