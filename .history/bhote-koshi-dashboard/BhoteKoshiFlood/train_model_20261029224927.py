
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

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = Path("data/Bhote_koshi.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}\n"
        "Check your project folder and CSV path."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

# ==========================================
# 2. PREPROCESS DATA
# ==========================================

df.columns = df.columns.str.strip()

required_columns = [
    "date",
    "river_discharge_m3s",
    "rain_mm",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )

# Parse dates and sort chronologically
df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

# Convert model columns to numeric values
numeric_columns = [
    col for col in required_columns
    if col != "date"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Remove duplicate dates, if any
df = df.drop_duplicates(subset=["date"], keep="last")
df = df.sort_values("date").reset_index(drop=True)

# ==========================================
# 3. CREATE NEXT-DAY TARGET
# ==========================================

# Tomorrow's discharge is the value we want to predict
df["next_day_discharge"] = (
    df["river_discharge_m3s"].shift(-1)
)

# Verify that tomorrow is exactly one calendar day later
day_gap = df["date"].shift(-1) - df["date"]

df = df.loc[
    day_gap == pd.Timedelta(days=1)
].copy()

print("Records after date validation:", len(df))

# ==========================================
# 4. SELECT FEATURES
# ==========================================

features = [
    "river_discharge_m3s",
    "rain_mm",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh"
]

target = "next_day_discharge"

# Remove rows with missing values in selected columns
df = df.dropna(subset=features + [target])

if len(df) < 10:
    raise ValueError(
        "Too few valid records for a reliable train/test split."
    )

X = df[features]
y = df[target]

print("Valid records:", len(df))
print("Features:", features)

# ==========================================
# 5. SPLIT TRAINING AND TESTING DATA
# ==========================================

# First 80% for training, final 20% for testing
# Keep chronological order; do not shuffle.
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 6. TRAIN RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nRandom Forest model training completed!")

# ==========================================
# 7. TEST MODEL AND CHECK ACCURACY
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE:  {mae:.4f} m3/s")
print(f"RMSE: {rmse:.4f} m3/s")
print(f"R2 Score: {r2:.4f}")

# ==========================================
# 8. SAVE TEST PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Date": df["date"].iloc[split_index:].values,
    "Actual_Discharge": y_test.to_numpy(),
    "Predicted_Discharge": y_pred
})

results["Absolute_Error"] = (
    results["Actual_Discharge"]
    - results["Predicted_Discharge"]
).abs()

results.to_csv("test_predictions.csv", index=False)

print("\nTest predictions saved to test_predictions.csv")

# ==========================================
# 9. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    {"model": model, "features": features},
    "bhote_koshi_rf_model.pkl"
)

print("Trained model saved to bhote_koshi_rf_model.pkl")

# ==========================================
# 10. PREDICT NEXT-DAY DISCHARGE
# ==========================================

# Use the latest available day's measurements.
latest_row = df.iloc[-1]

latest_features = latest_row[features].to_frame().T

next_day_prediction = model.predict(latest_features)[0]

latest_date = latest_row["date"]
forecast_date = latest_date + pd.Timedelta(days=1)

print("\n========== NEXT-DAY FORECAST ==========")
print("Latest data date:", latest_date.strftime("%Y-%m-%d"))
print("Forecast date:", forecast_date.strftime("%Y-%m-%d"))
print(
    f"Predicted discharge: "
    f"{next_day_prediction:.4f} m3/s"
)

# Experimental threshold only; verify its source before operational use.
EXPERIMENTAL_THRESHOLD = 4.08

if next_day_prediction >= EXPERIMENTAL_THRESHOLD:
    print("Experimental threshold result: ABOVE THRESHOLD")
else:
    print("Experimental threshold result: BELOW THRESHOLD")

print("\nAll steps completed!")
