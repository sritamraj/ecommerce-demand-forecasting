import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from features import FEATURE_COLUMNS, TARGET_COLUMN
from forecasting import time_series_folds
from evaluation import evaluate


TEST_DAYS = 60
N_FOLDS = 4
TEST_SIZE_DAYS = 60


def build_models():
    return {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


print("=" * 70)
print("TIME-SERIES CROSS-VALIDATION")
print("=" * 70)


# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------

df = pd.read_csv(
    "data/model_features.csv",
    parse_dates=["date"],
)


# ---------------------------------------------------------------------
# 2. Freeze final test period
# ---------------------------------------------------------------------

max_date = df["date"].max()

test_start = (
    max_date
    - pd.Timedelta(days=TEST_DAYS - 1)
)

development = df[
    df["date"] < test_start
].copy()

final_test = df[
    df["date"] >= test_start
].copy()


assert development["date"].max() < final_test["date"].min()
assert final_test["date"].nunique() == TEST_DAYS


print(f"Full data:       {df.shape}")
print(f"Development:     {development.shape}")
print(f"Final test:      {final_test.shape}")
print(f"Development end: {development['date'].max().date()}")
print(
    f"Final test:      "
    f"{final_test['date'].min().date()} "
    f"to "
    f"{final_test['date'].max().date()}"
)


# ---------------------------------------------------------------------
# 3. Cross-validation
# ---------------------------------------------------------------------

fold_results = []

# Store only out-of-fold Gradient Boosting predictions.
# These are development-period predictions and are NOT final-test data.
oof_predictions = []


for fold_i, (train_mask, test_mask, bounds) in enumerate(
    time_series_folds(
        development["date"],
        n_folds=N_FOLDS,
        test_size_days=TEST_SIZE_DAYS,
    ),
    start=1,
):

    train = development.loc[train_mask].copy()
    validation = development.loc[test_mask].copy()

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    predictions = {
        "Seasonal Naive": validation["lag_7"].values
    }

    models = build_models()

    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        predictions[model_name] = np.clip(
            model.predict(X_validation),
            0,
            None,
        )

    # -------------------------------------------------------------
    # Save Gradient Boosting OOF predictions for inventory
    # calibration.
    # -------------------------------------------------------------

    gb_predictions = predictions["Gradient Boosting"]

    oof_fold = validation[
        [
            "product_id",
            "category",
            "date",
            TARGET_COLUMN,
        ]
    ].copy()

    oof_fold["prediction"] = gb_predictions

    oof_fold["error"] = (
        oof_fold[TARGET_COLUMN]
        - oof_fold["prediction"]
    )

    oof_fold["abs_error"] = (
        oof_fold["error"]
        .abs()
    )

    oof_fold["fold"] = fold_i

    oof_predictions.append(oof_fold)

    # -------------------------------------------------------------
    # Evaluate all models.
    # -------------------------------------------------------------

    for model_name, prediction in predictions.items():

        metrics = evaluate(
            y_validation.values,
            prediction,
        )

        metrics["fold"] = fold_i
        metrics["model"] = model_name

        fold_results.append(metrics)

    print(
        f"Fold {fold_i}: "
        f"{bounds['test_start'].astype('datetime64[D]')} -> "
        f"{bounds['test_end'].astype('datetime64[D]')} "
        f"completed"
    )


# ---------------------------------------------------------------------
# 4. Save CV metrics
# ---------------------------------------------------------------------

results = pd.DataFrame(
    fold_results
)

results.to_csv(
    "reports/cv_results_by_fold.csv",
    index=False,
)


# Aggregate fold metrics.
#
# The groupby/agg operation creates MultiIndex columns.
# Flatten them before writing to CSV so the report is clean and
# directly usable by downstream analysis.
summary = (
    results
    .groupby("model")[["MAE", "RMSE", "sMAPE_%"]]
    .agg(["mean", "std"])
    .sort_values(("MAE", "mean"))
)

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

summary.to_csv(
    "reports/cv_results_summary.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 5. Save development out-of-fold predictions
# ---------------------------------------------------------------------

oof_results = pd.concat(
    oof_predictions,
    ignore_index=True,
)

oof_results = oof_results.sort_values(
    ["product_id", "date"]
).reset_index(drop=True)


oof_results.to_csv(
    "reports/gradient_boosting_oof_predictions.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print()
print("=" * 70)
print("OOF INVENTORY CALIBRATION DATA")
print("=" * 70)

print(
    f"OOF rows:       {len(oof_results):,}"
)

print(
    f"OOF products:   {oof_results['product_id'].nunique():,}"
)

print(
    f"OOF date range: "
    f"{oof_results['date'].min().date()} "
    f"to "
    f"{oof_results['date'].max().date()}"
)

print(
    f"Mean OOF abs error: "
    f"{oof_results['abs_error'].mean():.4f}"
)

print()
print(
    "Saved: "
    "reports/gradient_boosting_oof_predictions.csv"
)

print()
print("Final 60-day holdout was NOT used for model selection.")
print("Final 60-day holdout was NOT used for inventory calibration.")
print("CV complete.")