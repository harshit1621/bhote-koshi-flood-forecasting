
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
# 1. FIND AND LOAD THE DATASET
# ==================================================

# Folder containing train_model.py
BASE_DIR = Path(__file__).resolve().parent

# Expected CSV location:
# BhoteKoshiFlood/data/Bhote_koshi.csv
DATA_PATH = BASE_DIR / "data" / "Bhote_koshi.csv"

print("=" * 55)
print("BHOTE KOSHI NEXT-DAY DISCHARGE PREDICTION")
print("=" * 55)

print("Script folder:", BASE_DIR)
print("Dataset path:", DATA_PATH)

# If the expected path does not exist, search the project folder
if not DATA_PATH.exists():
    print("\nCSV not found in the expected data folder.")
    print("Searching inside the project directory...")

    matches = list(BASE_DIR.rglob("Bhote_koshi.csv"))

    if matches:
        DATA_PATH = matches[0]
        print("Dataset found at:", DATA_PATH)
    else:
        raise FileNotFoundError(
            "\nBhote_koshi.csv could not be found.\n"
            "Place the CSV inside a folder named 'data' "
            "next to train_model.py, or check its filename."
        )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)

# ==================================================
# 2. CHECK REQUIRED COLUMNS
# ==================================================

df.columns = df.columns.str.strip()

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
        f"\nMissing columns in CSV: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# ==================================================
# 3. CLEAN AND SORT DATA
# ==================================================

# Parse dates
df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

# Convert feature values to numeric
for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove invalid dates
df = df.dropna(subset=["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# Keep one record per date
df = df.drop_duplicates(
    subset=["date"],
    keep="last"
).reset_index(drop=True)

print("\nDate range:")
print("Start:", df["date"].min().date())
print("End:  ", df["date"].max().date())

# ==================================================
# 4. CREATE NEXT-DAY DISCHARGE TARGET
# ==================================================

# Target is the following calendar day's discharge.
# Do not accidentally use the next row when dates have gaps.

df["next_date"] = df["date"].shift(-1)

df["next_day_discharge"] = (
    df["river_discharge_m3s"].shift(-1)
)

day_gap = df["next_date"] - df["date"]

df = df.loc[
    day_gap == pd.Timedelta(days=1)
].copy()

print("\nRecords with a valid next-day target:", len(df))

# ==================================================
# 5. HANDLE MISSING VALUES
# ==================================================

target = "next_day_discharge"

df = df.dropna(
    subset=features + [target]
).reset_index(drop=True)

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} valid records remain. "
        "Check your dates and missing values."
    )

X = df[features]
y = df[target]

print("Final usable records:", len(df))

# ==================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==================================================

# First 80%: training
# Last 20%: testing
# Never shuffle time-series data.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records: ", len(X_test))

print(
    "Training target dates:",
    df["date"].iloc[:split_index].min().date(),
    "to",
    df["date"].iloc[:split_index].max().date()
)

print(
    "Testing target dates:",
    df["date"].iloc[split_index:].min().date(),
    "to",
    df["date"].iloc[split_index:].max().date()
)

# ==================================================
# 7. TRAIN RANDOM FOREST REGRESSOR
# ==================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# ==================================================
# 8. EVALUATE ON UNSEEN TEST DATA
# ==================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE:     {mae:.4f} m3/s")
print(f"RMSE:    {rmse:.4f} m3/s")
print(f"R2 score: {r2:.4f}")

# ==================================================
# 9. COMPARE WITH A SIMPLE BASELINE
# ==================================================

# Baseline: predict tomorrow's discharge as today's
baseline_pred = X_test["river_discharge_m3s"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)

print("\n========== BASELINE COMPARISON ==========")
print(f"Baseline MAE:  {baseline_mae:.4f} m3/s")
print(f"Baseline RMSE: {baseline_rmse:.4f} m3/s")

if mae < baseline_mae:
    print("Random Forest has lower MAE than the baseline.")
else:
    print("Baseline has equal or lower MAE than Random Forest.")

# ==================================================
# 10. SAVE TEST PREDICTIONS
# ==================================================

results = pd.DataFrame({
    "Date": df["date"].iloc[split_index:].to_numpy(),
    "Actual_Discharge": y_test.to_numpy(),
    "Predicted_Discharge": y_pred
})

results["Absolute_Error"] = (
    results["Actual_Discharge"]
    - results["Predicted_Discharge"]
).abs()

results.to_csv(
    BASE_DIR / "test_predictions.csv",
    index=False
)

print("\nTest predictions saved to:")
print(BASE_DIR / "test_predictions.csv")

# ==================================================
# 11. SAVE TRAINED MODEL
# ==================================================

joblib.dump(
    {
        "model": model,
        "features": features
    },
    BASE_DIR / "bhote_koshi_rf_model.pkl"
)

print("\nTrained model saved to:")
print(BASE_DIR / "bhote_koshi_rf_model.pkl")

# ==================================================
# 12. PREDICT THE NEXT CALENDAR DAY
# ==================================================

# Use the latest date in the original CSV, not an older
# row left behind after next-day target creation.
latest_row = (
    pd.read_csv(DATA_PATH)
    .assign(
        date=lambda x: pd.to_datetime(
            x["date"],
            dayfirst=True,
            errors="coerce"
        )
    )
    .sort_values("date")
    .iloc[-1]
)

# Check whether all latest input values are available
latest_features = pd.to_numeric(
    latest_row[features],
    errors="coerce"
)

if latest_features.isna().any():
    print(
        "\nCannot forecast the next day: "
        "the latest row has missing or invalid feature values."
    )
else:
    latest_input = pd.DataFrame(
        [latest_features.to_dict()],
        columns=features
    )

    next_day_prediction = model.predict(
        latest_input
    )[0]

    latest_date = latest_row["date"]
    forecast_date = latest_date + pd.Timedelta(days=1)

    print("\n========== NEXT-DAY FORECAST ==========")
    print("Latest observation:", latest_date.strftime("%Y-%m-%d"))
    print("Forecast date:     ", forecast_date.strftime("%Y-%m-%d"))

    print(
        f"Predicted discharge: "
        f"{next_day_prediction:.4f} m3/s"
    )

    # Experimental threshold only — not verified as official.
    EXPERIMENTAL_THRESHOLD = 4.08

    print(
        f"Experimental threshold: "
        f"{EXPERIMENTAL_THRESHOLD:.2f} m3/s"
    )

    if next_day_prediction >= EXPERIMENTAL_THRESHOLD:
        print("Result: PREDICTION IS AT OR ABOVE THE THRESHOLD")
    else:
        print("Result: PREDICTION IS BELOW THE THRESHOLD")

print("\n========== PROCESS COMPLETED ==========")
