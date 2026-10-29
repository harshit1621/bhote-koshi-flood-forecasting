
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# 1. FIND AND LOAD DATASET
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

# Expected location: BhoteKoshiFlood/data/Bhote_koshi.csv
DATA_PATH = BASE_DIR / "data" / "Bhote_koshi.csv"

# Search project subfolders if not found at the expected path
if not DATA_PATH.exists():
    matches = list(BASE_DIR.rglob("Bhote_koshi.csv"))

    if matches:
        DATA_PATH = matches[0]
    else:
        raise FileNotFoundError(
            f"Cannot find Bhote_koshi.csv inside {BASE_DIR}.\n"
            "Place it in the data folder or check the filename."
        )

print("=" * 55)
print("BHOTE KOSHI DISCHARGE PREDICTION")
print("=" * 55)
print("Dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)

# ==================================================
# 2. CHECK REQUIRED COLUMNS
# ==================================================

features = [
    "river_discharge_m3s",
    "rain_mm",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh"
]

required_columns = ["date"] + features

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# ==================================================
# 3. CLEAN AND SORT DATA
# ==================================================

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date"])
df = df.sort_values("date")

# Keep one record per date
df = df.drop_duplicates(subset=["date"], keep="last")
df = df.reset_index(drop=True)

print("First date:", df["date"].min().date())
print("Last date: ", df["date"].max().date())

# ==================================================
# 4. CREATE NEXT-DAY TARGET
# ==================================================

# Shift discharge by one row to create tomorrow's target.
df["next_date"] = df["date"].shift(-1)
df["next_day_discharge"] = (
    df["river_discharge_m3s"].shift(-1)
)

# Retain only rows where the next record is exactly one day later
day_gap = df["next_date"] - df["date"]

df = df.loc[
    day_gap == pd.Timedelta(days=1)
].copy()

# Remove records with missing model inputs or target
df = df.dropna(
    subset=features + ["next_day_discharge"]
).reset_index(drop=True)

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} usable records remain. "
        "Check the dates and missing values in your CSV."
    )

X = df[features]
y = df["next_day_discharge"]

print("\nUsable records:", len(df))

# ==================================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==================================================

# First 80%: training data
# Last 20%: testing data
# Do not shuffle time-series observations.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records: ", len(X_test))

# ==================================================
# 6. TRAIN RANDOM FOREST REGRESSOR
# ==================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed.")

# ==================================================
# 7. PREDICT HISTORICAL TEST DATA
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# 8. CALCULATE EVALUATION METRICS
# ==================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE:  {mae:.4f} m3/s")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.4f} m3/s")
print(f"R2:   {r2:.4f}")

# ==================================================
# 9. SAVE TEST PREDICTIONS
# ==================================================

results = pd.DataFrame({
    # Date associated with the target discharge
    "Forecast_Date": (
        df["date"].iloc[split_index:].to_numpy()
        + np.timedelta64(1, "D")
    ),
    "Actual_Discharge_m3s": y_test.to_numpy(),
    "Predicted_Discharge_m3s": y_pred
})

results["Absolute_Error_m3s"] = (
    results["Actual_Discharge_m3s"]
    - results["Predicted_Discharge_m3s"]
).abs()

results.to_csv(
    BASE_DIR / "test_predictions.csv",
    index=False
)

print("\nTest results saved to:")
print(BASE_DIR / "test_predictions.csv")

print("\nFirst 10 test predictions:")
print(results.head(10).to_string(index=False))

# ==================================================
# 10. SAVE TRAINED MODEL
# ==================================================

joblib.dump(
    {
        "model": model,
        "features": features
    },
    BASE_DIR / "bhote_koshi_rf_model.pkl"
)

print("\nModel saved to:")
print(BASE_DIR / "bhote_koshi_rf_model.pkl")

# ==================================================
# 11. PREDICT THE NEXT DAY AFTER THE LATEST DATA
# ==================================================

# Use the latest row from the original dataset.
# This forecast is separate from the historical test evaluation.

latest_row = (
    pd.read_csv(DATA_PATH)
    .assign(
        date=lambda data: pd.to_datetime(
            data["date"],
            dayfirst=True,
            errors="coerce"
        )
    )
    .sort_values("date")
    .iloc[-1]
)

# Ensure the feature values are numeric
latest_values = pd.to_numeric(
    latest_row[features],
    errors="coerce"
)

if latest_values.isna().any():
    print(
        "\nCannot generate next-day forecast: "
        "latest observation contains missing feature values."
    )
else:
    latest_input = pd.DataFrame(
        [latest_values.to_dict()],
        columns=features
    )

    next_day_prediction = model.predict(latest_input)[0]

    latest_date = latest_row["date"]
    forecast_date = latest_date + pd.Timedelta(days=1)

    print("\n========== NEXT-DAY FORECAST ==========")
    print("Latest observation:", latest_date.strftime("%Y-%m-%d"))
    print("Forecast date:     ", forecast_date.strftime("%Y-%m-%d"))
    print(
        f"Predicted discharge: "
        f"{next_day_prediction:.4f} m3/s"
    )

    # Save the future forecast separately
    forecast_result = pd.DataFrame({
        "Last_Observed_Date": [latest_date],
        "Forecast_Date": [forecast_date],
        "Predicted_Discharge_m3s": [next_day_prediction]
    })

    forecast_result.to_csv(
        BASE_DIR / "next_day_forecast.csv",
        index=False
    )

    print("Forecast saved to next_day_forecast.csv")

    # Experimental threshold; not verified as an official threshold
    experimental_threshold = 4.08

    print(
        f"Experimental threshold: "
        f"{experimental_threshold:.2f} m3/s"
    )

    if next_day_prediction >= experimental_threshold:
        print("Prediction is at or above the experimental threshold.")
    else:
        print("Prediction is below the experimental threshold.")

print("\n========== PROCESS COMPLETED ==========")
