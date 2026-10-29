
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. File paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "Bhote_koshi.csv"
OUTPUT_DIR = BASE_DIR / "ml" / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(CSV_PATH)

print("Dataset loaded:", df.shape)

# 3. Prepare date and sort chronologically
df["date"] = pd.to_datetime(
    df["date"], dayfirst=True, errors="coerce"
)

df["river_discharge_m3s"] = pd.to_numeric(
    df["river_discharge_m3s"], errors="coerce"
)

df = df.dropna(subset=["date", "river_discharge_m3s"])
df = df.sort_values("date").drop_duplicates(
    subset=["date"], keep="last"
).reset_index(drop=True)

# 4. Define the target: discharge on the next calendar day
df["target_date"] = df["date"].shift(-1)
df["next_day_discharge"] = df["river_discharge_m3s"].shift(-1)

one_day_gap = (
    df["target_date"] - df["date"]
).dt.days == 1

df = df.loc[one_day_gap].copy()

# 5. Features available from the current day
weather_features = [
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "rain_mm",
    "temperature_mean_c",
    "dew_point_mean_c",
    "precipitation_hours",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh",
    "wind_gusts_max_kmh",
    "wind_direction_dominant_deg",
]

features = ["river_discharge_m3s"] + [
    col for col in weather_features if col in df.columns
]

# Add calendar information known in advance
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
features += ["month", "day_of_year"]

# Convert model features to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features + ["next_day_discharge"])

if len(df) < 30:
    raise ValueError(
        f"Only {len(df)} usable rows. Check dates and feature values."
    )

X = df[features]
y = df["next_day_discharge"]

# 6. Chronological 80/20 train-test split
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", features)

# 7. Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 8. Evaluate on unseen, later dates
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== RANDOM FOREST RESULTS =====")
print(f"MAE:  {mae:.4f} m3/s")
print(f"RMSE: {rmse:.4f} m3/s")
print(f"R2:   {r2:.4f}")

# 9. Save model and metadata
joblib.dump(
    {"model": model, "features": features},
    OUTPUT_DIR / "random_forest_model.joblib"
)

metrics = {
    "model": "RandomForestRegressor",
    "target": "next_day_discharge",
    "features": features,
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 10. Save actual vs predicted test values
results = pd.DataFrame({
    "date": df["target_date"].iloc[split:].values,
    "actual_discharge_m3s": y_test.values,
    "predicted_discharge_m3s": predictions,
})

results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("\nModel saved to:", OUTPUT_DIR / "random_forest_model.joblib")
print("Metrics saved to:", OUTPUT_DIR / "metrics.json")
print("Test predictions saved to:", OUTPUT_DIR / "test_predictions.csv")
