
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Bhote_koshi.csv"

THRESHOLD = 4.08  # Experimental threshold, NOT verified as official
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Features available in the dataset.
# Do not include warning_threshold or flood_warning because
# they may leak the target warning decision into the model.
FEATURES = [
    "river_discharge_m3s",
    "rain_mm",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh"
]

TARGET = "next_day_discharge"


# ============================================================
# 2. HELPER FUNCTION TO SAVE FILES SAFELY
# ============================================================

def safe_save_csv(dataframe, filename):
    """
    Save a CSV. If the target file is locked or inaccessible,
    try a timestamped filename instead.
    """
    output_path = BASE_DIR / filename

    try:
        dataframe.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        return output_path

    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alternate_path = BASE_DIR / (
            f"{Path(filename).stem}_{timestamp}.csv"
        )

        try:
            dataframe.to_csv(alternate_path, index=False)
            print("Could not write to the original file.")
            print(f"Saved using an alternative filename: {alternate_path}")
            return alternate_path

        except PermissionError as error:
            print(f"Could not save {filename}: {error}")
            print("Close Excel or other applications using this file.")
            return None


# ============================================================
# 3. LOAD DATASET
# ============================================================

if not DATA_PATH.exists():
    # Search within the project directory if the expected path
    # is incorrect.
    matches = list(BASE_DIR.rglob("Bhote_koshi.csv"))

    if matches:
        DATA_PATH = matches[0]
    else:
        raise FileNotFoundError(
            f"Dataset not found.\nExpected location: {DATA_PATH}\n"
            "Place Bhote_koshi.csv inside the data folder."
        )

print(f"\nDataset path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = ["date"] + FEATURES

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in CSV: {missing_columns}"
    )

# Parse dates and numerical features.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for column in FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows without valid dates and sort chronologically.
df = df.dropna(subset=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Keep one record per date.
# This assumes the CSV contains one daily observation per date.
df = df.drop_duplicates(subset=["date"], keep="last")
df = df.sort_values("date").reset_index(drop=True)

print(f"Dataset start: {df['date'].min().date()}")
print(f"Dataset end:   {df['date'].max().date()}")


# ============================================================
# 4. CREATE NEXT-DAY TARGET
# ============================================================

df["next_date"] = df["date"].shift(-1)
df[TARGET] = df["river_discharge_m3s"].shift(-1)

# Only retain rows where the next record is exactly one day later.
# This avoids accidentally treating a multi-day gap as next day.
df["date_difference"] = (
    df["next_date"] - df["date"]
).dt.days

df = df[df["date_difference"] == 1].copy()

# Remove rows missing required inputs or target.
df = df.dropna(subset=FEATURES + [TARGET])

df = df.sort_values("date").reset_index(drop=True)

if len(df) < 20:
    raise ValueError(
        "Not enough valid daily records to train and evaluate the model."
    )

print(f"Valid next-day records: {len(df)}")


# ============================================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

X = df[FEATURES]
y = df[TARGET]

split_index = int(len(df) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_input_dates = df["date"].iloc[split_index:]
test_forecast_dates = df["next_date"].iloc[split_index:]

print("\n========== DATA SPLIT ==========")
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ============================================================
# 6. TRAIN RANDOM FOREST REGRESSOR
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed.")


# ============================================================
# 7. EVALUATE RANDOM FOREST
# ============================================================

rf_predictions = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, rf_predictions)
mse = mean_squared_error(y_test, rf_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, rf_predictions)

print("\n========== RANDOM FOREST EVALUATION ==========")
print(f"MAE:  {mae:.4f} m3/s")
print(f"MSE:  {mse:.6f} (m3/s)^2")
print(f"RMSE: {rmse:.4f} m3/s")
print(f"R2:   {r2:.4f}")


# ============================================================
# 8. PERSISTENCE BASELINE
# Predict tomorrow's discharge as today's discharge.
# ============================================================

baseline_predictions = X_test["river_discharge_m3s"].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\n========== PERSISTENCE BASELINE ==========")
print(f"MAE:  {baseline_mae:.4f} m3/s")
print(f"MSE:  {baseline_mse:.6f} (m3/s)^2")
print(f"RMSE: {baseline_rmse:.4f} m3/s")
print(f"R2:   {baseline_r2:.4f}")


# ============================================================
# 9. SAVE HISTORICAL TEST PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Input_Date": test_input_dates.dt.strftime("%Y-%m-%d").to_numpy(),
    "Forecast_Date": test_forecast_dates.dt.strftime("%Y-%m-%d").to_numpy(),
    "Actual_Discharge_m3s": y_test.to_numpy(),
    "RF_Predicted_Discharge_m3s": rf_predictions,
    "Baseline_Predicted_Discharge_m3s": baseline_predictions,
    "Absolute_Error_m3s": np.abs(y_test.to_numpy() - rf_predictions)
})

# Apply the experimental threshold to predictions and actual values.
results["Actual_Above_Threshold"] = (
    results["Actual_Discharge_m3s"] >= THRESHOLD
)

results["RF_Predicted_Above_Threshold"] = (
    results["RF_Predicted_Discharge_m3s"] >= THRESHOLD
)

results["Threshold_Decision_Match"] = (
    results["Actual_Above_Threshold"]
    == results["RF_Predicted_Above_Threshold"]
)

safe_save_csv(results, "test_predictions.csv")


# ============================================================
# 10. THRESHOLD CLASSIFICATION CHECK
# ============================================================

actual_high = y_test.to_numpy() >= THRESHOLD
predicted_high = rf_predictions >= THRESHOLD

true_positive = np.sum(actual_high & predicted_high)
false_positive = np.sum(~actual_high & predicted_high)
false_negative = np.sum(actual_high & ~predicted_high)
true_negative = np.sum(~actual_high & ~predicted_high)

threshold_agreement = np.mean(actual_high == predicted_high)

print("\n========== EXPERIMENTAL THRESHOLD CHECK ==========")
print(f"Threshold: {THRESHOLD:.2f} m3/s")
print(f"True positives:  {true_positive}")
print(f"False positives: {false_positive}")
print(f"False negatives: {false_negative}")
print(f"True negatives:  {true_negative}")
print(f"Threshold agreement: {threshold_agreement:.2%}")

if np.sum(actual_high) > 0:
    miss_rate = false_negative / np.sum(actual_high)
    print(f"High-discharge miss rate: {miss_rate:.2%}")
else:
    print("High-discharge miss rate: undefined; no actual high days in test set.")

print(
    "\nWARNING: The 4.08 m3/s threshold is experimental. "
    "It is not confirmed as an official DHM warning threshold."
)


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

model_path = BASE_DIR / "bhote_koshi_rf_model.pkl"

try:
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved: {model_path}")
except PermissionError:
    print("\nCould not save the model. Close any application using it.")


# ============================================================
# 12. TRAIN FINAL MODEL ON ALL AVAILABLE HISTORICAL RECORDS
# ============================================================

# The test-set metrics above still come from the earlier
# chronological split. Retrain separately on all valid records
# for the next forecast.
final_model = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_model.fit(X, y)

final_model_path = BASE_DIR / "bhote_koshi_rf_model_final.pkl"

try:
    joblib.dump(final_model, final_model_path)
    print(f"Final model saved: {final_model_path}")
except PermissionError:
    print("Could not save the final model. Close any application using it.")


# ============================================================
# 13. FORECAST THE DAY AFTER THE LATEST OBSERVATION
# ============================================================

# Use the latest observed row in the original dataset.
# Do not use warning_threshold or flood_warning as model inputs.
latest_row = (
    pd.read_csv(DATA_PATH)
    .assign(date=lambda data: pd.to_datetime(data["date"], errors="coerce"))
    .dropna(subset=["date"])
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .iloc[-1]
)

latest_date = latest_row["date"]

latest_features = pd.DataFrame([{
    feature: pd.to_numeric(latest_row[feature], errors="coerce")
    for feature in FEATURES
}])

if latest_features.isna().any().any():
    print(
        "\nCould not produce the next-day forecast: "
        "the latest record has missing model features."
    )
else:
    latest_features = latest_features[FEATURES]

    forecast_discharge = float(
        final_model.predict(latest_features)[0]
    )

    forecast_date = latest_date + timedelta(days=1)

    if forecast_discharge >= THRESHOLD:
        threshold_result = "Above experimental threshold"
    else:
        threshold_result = "Below experimental threshold"

    forecast_output = pd.DataFrame([{
        "Last_Observed_Date": latest_date.strftime("%Y-%m-%d"),
        "Forecast_Date": forecast_date.strftime("%Y-%m-%d"),
        "Predicted_Discharge_m3s": forecast_discharge,
        "Experimental_Threshold_m3s": THRESHOLD,
        "Threshold_Result": threshold_result,
        "Validation_Status": "Not validated against actual future discharge"
    }])

    print("\n========== NEXT-DAY FORECAST ==========")
    print(f"Last observed date: {latest_date.date()}")
    print(f"Forecast date:      {forecast_date.date()}")
    print(f"Predicted discharge: {forecast_discharge:.4f} m3/s")
    print(f"Threshold result:   {threshold_result}")

    safe_save_csv(forecast_output, "next_day_forecast.csv")

print("\n========== PROCESS COMPLETED ==========")
