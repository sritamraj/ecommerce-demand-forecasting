from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "benchmark" / "raw"
SERIES_FILE = ROOT / "data" / "benchmark" / "benchmark_series.csv"
OUTPUT_DIR = ROOT / "data" / "benchmark"
OUTPUT_FILE = OUTPUT_DIR / "public_benchmark_daily.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading selected benchmark series...")
    selected = pd.read_csv(SERIES_FILE)

    required_selection_cols = {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    }

    missing = required_selection_cols - set(selected.columns)

    if missing:
        raise ValueError(
            f"Missing selection columns: {sorted(missing)}"
        )

    print(f"Selected series: {len(selected)}")

    if selected["id"].nunique() != len(selected):
        raise ValueError("Duplicate benchmark series IDs detected.")

    print("Loading sales history...")

    day_columns = [
        f"d_{i}"
        for i in range(1, 1914)
    ]

    sales = pd.read_csv(
        RAW_DIR / "sales_train_validation.csv",
        usecols=["id"] + day_columns,
    )

    print("Loading calendar...")

    calendar = pd.read_csv(
        RAW_DIR / "calendar.csv",
        usecols=[
            "date",
            "d",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ],
    )

    print("Filtering selected series...")

    sales = sales[
        sales["id"].isin(selected["id"])
    ].copy()

    if sales["id"].nunique() != len(selected):
        raise ValueError(
            "Selected series mismatch: "
            "some benchmark series were not found."
        )

    print("Converting sales from wide to long format...")

    daily = sales.melt(
        id_vars=["id"],
        value_vars=day_columns,
        var_name="d",
        value_name="demand",
    )

    daily["d"] = daily["d"].astype(str)

    print("Joining series metadata...")

    metadata = selected[
        [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ]
    ].copy()

    daily = daily.merge(
        metadata,
        on="id",
        how="left",
        validate="many_to_one",
    )

    print("Joining calendar features...")

    calendar["d"] = calendar["d"].astype(str)

    # sales_train_validation.csv contains d_1 ... d_1913.
    # Restrict the calendar to exactly the same observation window.
    calendar = calendar[
        calendar["d"].isin(day_columns)
    ].copy()

    if len(calendar) != len(day_columns):
        raise ValueError(
            f"Unexpected calendar length: {len(calendar)}; "
            f"expected {len(day_columns)}"
        )

    daily = daily.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one",
    )

    print("Creating benchmark features...")

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily["demand"] = pd.to_numeric(
        daily["demand"],
        errors="coerce",
    )

    daily["is_weekend"] = (
        daily["wday"].isin([1, 7])
    ).astype(int)

    daily["has_event"] = (
        daily["event_name_1"].notna()
        | daily["event_name_2"].notna()
    ).astype(int)

    daily["event_type_present"] = (
        daily["event_type_1"].notna()
        | daily["event_type_2"].notna()
    ).astype(int)

    daily["snap"] = 0

    daily.loc[
        daily["state_id"] == "CA",
        "snap",
    ] = daily.loc[
        daily["state_id"] == "CA",
        "snap_CA",
    ]

    daily.loc[
        daily["state_id"] == "TX",
        "snap",
    ] = daily.loc[
        daily["state_id"] == "TX",
        "snap_TX",
    ]

    daily.loc[
        daily["state_id"] == "WI",
        "snap",
    ] = daily.loc[
        daily["state_id"] == "WI",
        "snap_WI",
    ]

    daily = daily.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    print("Validating daily panel...")

    expected_rows = (
        len(selected) * len(day_columns)
    )

    if len(daily) != expected_rows:
        raise ValueError(
            f"Unexpected row count: {len(daily)}; "
            f"expected {expected_rows}"
        )

    if daily["id"].nunique() != len(selected):
        raise ValueError(
            "Unexpected number of series."
        )

    if daily["date"].isna().any():
        raise ValueError(
            "Missing dates detected."
        )

    if daily["demand"].isna().any():
        raise ValueError(
            "Missing demand values detected."
        )

    if (daily["demand"] < 0).any():
        raise ValueError(
            "Negative demand detected."
        )

    duplicate_count = daily.duplicated(
        ["id", "date"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            "Duplicate id/date observations detected: "
            f"{duplicate_count}"
        )

    print("Adding leakage-safe lag features...")

    grouped = daily.groupby(
        "id",
        group_keys=False,
    )

    daily["lag_1"] = grouped["demand"].shift(1)

    daily["lag_7"] = grouped["demand"].shift(7)

    daily["lag_14"] = grouped["demand"].shift(14)

    daily["lag_28"] = grouped["demand"].shift(28)

    daily["rolling_mean_7"] = grouped[
        "demand"
    ].transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=7,
        )
        .mean()
    )

    daily["rolling_mean_14"] = grouped[
        "demand"
    ].transform(
        lambda x:
        x.shift(1)
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    daily["rolling_mean_28"] = grouped[
        "demand"
    ].transform(
        lambda x:
        x.shift(1)
        .rolling(
            28,
            min_periods=28,
        )
        .mean()
    )

    daily["rolling_std_7"] = grouped[
        "demand"
    ].transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=7,
        )
        .std()
    )

    daily["rolling_std_28"] = grouped[
        "demand"
    ].transform(
        lambda x:
        x.shift(1)
        .rolling(
            28,
            min_periods=28,
        )
        .std()
    )

    print("Creating final-holdout indicator...")

    max_date = daily["date"].max()

    final_test_start = (
        max_date
        - pd.Timedelta(days=27)
    )

    daily["is_final_test"] = (
        daily["date"] >= final_test_start
    ).astype(int)

    print("Final benchmark period:")

    print(
        f"  Maximum date: {max_date.date()}"
    )

    print(
        "  Final test start: "
        f"{final_test_start.date()}"
    )

    print("  Final test days: 28")

    output_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "date",
        "d",
        "demand",
        "weekday",
        "wday",
        "month",
        "year",
        "wm_yr_wk",
        "is_weekend",
        "has_event",
        "event_type_present",
        "snap",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "is_final_test",
    ]

    daily[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nBENCHMARK PREPARATION COMPLETE"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(daily):,}"
    )

    print(
        f"Series: {daily['id'].nunique()}"
    )

    print(
        "Date range: "
        f"{daily['date'].min().date()} "
        f"to "
        f"{daily['date'].max().date()}"
    )

    print(
        "Final-test rows: "
        f"{int(daily['is_final_test'].sum())}"
    )


if __name__ == "__main__":
    main()