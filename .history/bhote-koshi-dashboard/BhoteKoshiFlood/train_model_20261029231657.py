
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

# =====================================================
# 1. LOAD DATASET
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Bhote_koshi.csv"

# Search project folders if the expected path does not exist
if not DATA_PATH.exists():
    matches = list(BASE_DIR.rglob("Bhote_koshi.csv"))

    if not matches:
        raise FileNotFoundError(
            f"Cannot find Bhote_koshi.csv inside {BASE_DIR}.\n"
            "Check the CSV filename and folder."
        )

    DATA_PATH = matches[0]

print("Dataset path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# =====================================================
# 2. VALIDATE COLUMNS
# =====================================================

features = [
    "river_discharge_m3s",
    "rain_mm",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh"
]

required = ["date"] + features

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# =====================================================
# 3. CLEAN AND SORT DATA
# =====================================================

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date"])
df = df.sort_values("date")
df = df.drop_duplicates(subset=["date"], keep="last")
df = df.reset_index(drop=True)

print("Dataset start:", df["date"].min().date())
print("Dataset end:  ", df["date"].max().date())

# =====================================================
# 4. CREATE NEXT-CALENDAR-DAY TARGET
# =====================================================

df["next_date"] = df["date"].shift(-1)
df["next_day_discharge"] = (
    df["river_discharge_m3s"].shift(-1)
)

# Keep a row only if the next observation is exactly one
# calendar day later. This prevents gaps being treated as
# next-day predictions.
consecutive_days = (
    df["next_date"] - df["date"]
    == pd.Timedelta(days=1)
)

df = df.loc[consecutive_days].copy()

# Drop rows with missing features or target
df = df.dropna(
    subset=features + ["next_day_discharge"]
).reset_index(drop=True)

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} valid records remain. "
        "Check the dataset dates and missing values."
    )

X = df[features]
y = df["next_day_discharge"]

print("Valid next-day records:", len(df))

# =====================================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# =====================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records: ", len(X_test))

# =====================================================
# 6. TRAIN RANDOM FOREST
# =====================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed.")

# =====================================================
# 7. PREDICT HISTORICAL TEST DATA
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# 8. CALCULATE EVALUATION METRICS
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE:  {mae:.4f} m3/s")
print(f"MSE:  {mse:.6f} (m3/s)^2")
print(f"RMSE: {rmse:.4f} m3/s")
print(f"R2:   {r2:.4f}")

# =====================================================
# 9. SAVE HISTORICAL TEST PREDICTIONS
# =====================================================

# Each target belongs to the day after its input observation.
test_dates = (
    df["date"].iloc[split_index:].reset_index(drop=True)
    + pd.Timedelta(days=1)
)

results = pd.DataFrame({
    "Forecast_Date": test_dates,
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

print("\nSaved historical test predictions to:")
print(BASE_DIR / "test_predictions.csv")

# =====================================================
# 10. SAVE TRAINED MODEL
# =====================================================

joblib.dump(
    {"model": model, "features": features},
    BASE_DIR / "bhote_koshi_rf_model.pkl"
)

print("Saved model to:")
print(BASE_DIR / "bhote_koshi_rf_model.pkl")

# =====================================================
# 11. FORECAST THE DAY AFTER THE LATEST OBSERVATION
# =====================================================

# Read the latest row from the original CSV.
# Do not use the shortened next-day-target dataframe here.
original = pd.read_csv(DATA_PATH)
original.columns = original.columns.str.strip()

original["date"] = pd.to_datetime(
    original["date"],
    dayfirst=True,
    errors="coerce"
)

for col in features:
    original[col] = pd.to_numeric(
        original[col],
        errors="coerce"
    )

original = original.dropna(subset=["date"])
original = original.sort_values("date")

latest_row = original.iloc[-1]
latest_date = latest_row["date"]
forecast_date = latest_date + pd.Timedelta(days=1)

latest_features = pd.to_numeric(
    latest_row[features],
    errors="coerce"
)

if latest_features.isna().any():
    print(
        "\nCannot forecast: the latest row has missing "
        "or invalid feature values."
    )
else:
    latest_input = pd.DataFrame(
        [latest_features.to_dict()],
        columns=features
    )

    forecast = float(model.predict(latest_input)[0])

    forecast_df = pd.DataFrame({
        "Last_Observed_Date": [latest_date],
        "Forecast_Date": [forecast_date],
        "Predicted_Discharge_m3s": [forecast]
    })

    forecast_df.to_csv(
        BASE_DIR / "next_day_forecast.csv",
        index=False
    )

    print("\n========== NEXT-DAY FORECAST ==========")
    print("Last observed date:", latest_date.strftime("%Y-%m-%d"))
    print("Forecast date:     ", forecast_date.strftime("%Y-%m-%d"))
    print(f"Predicted discharge: {forecast:.4f} m3/s")

    print("\nSaved forecast to:")
    print(BASE_DIR / "next_day_forecast.csv")

print("\n========== COMPLETED ==========")
