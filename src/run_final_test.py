import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURE_COLUMNS, TARGET_COLUMN
from evaluation import evaluate


TEST_DAYS = 60


print("=" * 70)
print("FINAL FROZEN-MODEL TEST")
print("=" * 70)


# ---------------------------------------------------------------------
# 1. Load modeling data
# ---------------------------------------------------------------------

df = pd.read_csv(
    "data/model_features.csv",
    parse_dates=["date"],
)


# ---------------------------------------------------------------------
# 2. Freeze final holdout
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


print(f"Development rows: {len(development):,}")
print(f"Final test rows:  {len(final_test):,}")
print(f"Development end:  {development['date'].max().date()}")
print(
    f"Final test:       "
    f"{final_test['date'].min().date()} "
    f"to "
    f"{final_test['date'].max().date()}"
)


# ---------------------------------------------------------------------
# 3. Build the frozen model
# ---------------------------------------------------------------------

model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.05,
    max_iter=400,
    l2_regularization=1.0,
    random_state=42,
)


# ---------------------------------------------------------------------
# 4. Fit ONLY on development data
# ---------------------------------------------------------------------

X_development = development[FEATURE_COLUMNS]
y_development = development[TARGET_COLUMN]

X_final_test = final_test[FEATURE_COLUMNS]
y_final_test = final_test[TARGET_COLUMN]


print()
print("Fitting Gradient Boosting on development data...")

model.fit(
    X_development,
    y_development,
)


# ---------------------------------------------------------------------
# 5. Predict untouched final test
# ---------------------------------------------------------------------

final_predictions = np.clip(
    model.predict(X_final_test),
    0,
    None,
)


# ---------------------------------------------------------------------
# 6. Evaluate exactly once
# ---------------------------------------------------------------------

metrics = evaluate(
    y_final_test.values,
    final_predictions,
)


print()
print("=" * 70)
print("FINAL UNTOUCHED TEST RESULTS")
print("=" * 70)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


# ---------------------------------------------------------------------
# 7. Save predictions
# ---------------------------------------------------------------------

final_results = final_test[
    [
        "product_id",
        "category",
        "date",
        TARGET_COLUMN,
    ]
].copy()

final_results["prediction"] = final_predictions

final_results["error"] = (
    final_results["prediction"]
    - final_results[TARGET_COLUMN]
)

final_results["abs_error"] = (
    final_results["error"]
    .abs()
)

final_results.to_csv(
    "reports/final_test_predictions.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 8. Save metrics
# ---------------------------------------------------------------------

metrics_output = pd.DataFrame(
    [
        {
            "model": "Gradient Boosting",
            "test_start": final_test["date"].min().date(),
            "test_end": final_test["date"].max().date(),
            "test_rows": len(final_test),
            **metrics,
        }
    ]
)

metrics_output.to_csv(
    "reports/final_test_metrics.csv",
    index=False,
)


print()
print("Saved:")
print("  reports/final_test_predictions.csv")
print("  reports/final_test_metrics.csv")

print()
print("Final test evaluation complete.")