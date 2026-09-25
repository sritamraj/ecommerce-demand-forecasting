import numpy as np
import pandas as pd


LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL_Z = 1.65


def simulate_inventory(
    df_product,
    error_std,
    demand_multiplier=1.0,
    lead_time=LEAD_TIME_DAYS,
    review_period=REVIEW_PERIOD_DAYS,
    z=SERVICE_LEVEL_Z,
    starting_inventory=None,
    capacity_fraction=None,
):
    """
    Simulate a periodic-review inventory policy.

    Forecasts are treated as the frozen demand plan.
    Scenario multipliers affect actual demand only.

    Safety stock is based on forecast-error variability estimated
    from development-period out-of-fold predictions.

    The protection period is lead time + review period because
    orders are placed only at periodic review points.

    If capacity_fraction is provided, maximum inventory position
    is constrained to that fraction of the unconstrained target level.
    """

    d = (
        df_product
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(d) == 0:
        raise ValueError("Product simulation received no rows.")

    required_columns = {
        "date",
        "actual_demand",
        "forecast",
    }

    missing = required_columns - set(d.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if demand_multiplier <= 0:
        raise ValueError(
            "demand_multiplier must be greater than 0."
        )

    if lead_time < 0:
        raise ValueError(
            "lead_time must be non-negative."
        )

    if review_period <= 0:
        raise ValueError(
            "review_period must be greater than 0."
        )

    # ---------------------------------------------------------------
    # Scenario demand
    # ---------------------------------------------------------------

    d["actual_scenario"] = (
        d["actual_demand"] * demand_multiplier
    )

    # ---------------------------------------------------------------
    # Forecast-based policy parameters
    # ---------------------------------------------------------------

    avg_daily_forecast = d["forecast"].mean()

    protection_period = (
        lead_time + review_period
    )

    safety_stock = (
        z
        * error_std
        * np.sqrt(protection_period)
    )

    reorder_point = (
        avg_daily_forecast * protection_period
        + safety_stock
    )

    target_level = reorder_point

    # ---------------------------------------------------------------
    # Capacity constraint
    # ---------------------------------------------------------------

    if capacity_fraction is not None:
        if not 0 < capacity_fraction <= 1:
            raise ValueError(
                "capacity_fraction must be between 0 and 1."
            )

        inventory_capacity = (
            target_level * capacity_fraction
        )
    else:
        inventory_capacity = None

    # ---------------------------------------------------------------
    # Initial inventory
    # ---------------------------------------------------------------

    if starting_inventory is None:
        if inventory_capacity is None:
            inventory = target_level
        else:
            inventory = min(
                target_level,
                inventory_capacity,
            )
    else:
        inventory = float(starting_inventory)

        if inventory_capacity is not None:
            inventory = min(
                inventory,
                inventory_capacity,
            )

    pipeline = []

    stockout_days = 0
    lost_units = 0.0
    inventory_levels = []
    total_units_ordered = 0.0

    # ---------------------------------------------------------------
    # Day-by-day simulation
    # ---------------------------------------------------------------

    for i in range(len(d)):

        # Receive orders arriving today.
        arrivals = [
            quantity
            for arrival_day, quantity in pipeline
            if arrival_day == i
        ]

        inventory += sum(arrivals)

        pipeline = [
            (arrival_day, quantity)
            for arrival_day, quantity in pipeline
            if arrival_day != i
        ]

        # -----------------------------------------------------------
        # Demand fulfillment
        # -----------------------------------------------------------

        demand_today = float(
            d.loc[i, "actual_scenario"]
        )

        if demand_today > inventory:

            stockout_days += 1

            lost_units += (
                demand_today - inventory
            )

            inventory = 0.0

        else:

            inventory -= demand_today

        inventory_levels.append(inventory)

        # -----------------------------------------------------------
        # Periodic review
        # -----------------------------------------------------------

        if i % review_period == 0:

            pipeline_units = sum(
                quantity
                for _, quantity in pipeline
            )

            inventory_position = (
                inventory
                + pipeline_units
            )

            if inventory_position < reorder_point:

                order_quantity = max(
                    target_level - inventory_position,
                    0.0,
                )

                # Enforce maximum inventory-position capacity.
                if inventory_capacity is not None:

                    available_capacity = max(
                        inventory_capacity - inventory_position,
                        0.0,
                    )

                    order_quantity = min(
                        order_quantity,
                        available_capacity,
                    )

                if order_quantity > 0:

                    pipeline.append(
                        (
                            i + lead_time,
                            order_quantity,
                        )
                    )

                    total_units_ordered += (
                        order_quantity
                    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    n_days = len(d)

    cycle_service_level = (
        100
        * (1 - stockout_days / n_days)
    )

    return {
        "avg_inventory": float(
            np.mean(inventory_levels)
        ),
        "stockout_days": int(
            stockout_days
        ),
        "cycle_service_level_%": round(
            cycle_service_level,
            2,
        ),
        "lost_units": round(
            lost_units,
            1,
        ),
        "total_units_ordered": round(
            total_units_ordered,
            1,
        ),
        "avg_daily_forecast": round(
            avg_daily_forecast,
            2,
        ),
        "protection_period_days": int(
            protection_period
        ),
        "reorder_point": round(
            reorder_point,
            1,
        ),
        "target_level": round(
            target_level,
            1,
        ),
        "safety_stock": round(
            safety_stock,
            1,
        ),
        "inventory_capacity": (
            round(inventory_capacity, 1)
            if inventory_capacity is not None
            else None
        ),
    }


print("Inventory simulation engine loaded.")