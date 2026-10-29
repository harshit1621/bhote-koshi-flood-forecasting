
from pathlib import Path
from datetime import timedelta

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Bhote Koshi Flood Prediction",
    page_icon="🌊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Bhote_koshi.csv"
MODEL_PATH = BASE_DIR / "bhote_koshi_rf_model_final.pkl"
PREDICTIONS_PATH = BASE_DIR / "test_predictions.csv"

# Experimental threshold only; not a verified official DHM threshold.
THRESHOLD = 4.08

FEATURES = [
    "river_discharge_m3s",
    "rain_mm",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh",
]

# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    required = ["date", "river_discharge_m3s"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["river_discharge_m3s"] = pd.to_numeric(
        df["river_discharge_m3s"], errors="coerce"
    )

    df = df.dropna(subset=["date", "river_discharge_m3s"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# --------------------------------------------------
# 3. DASHBOARD HEADER
# --------------------------------------------------
st.title("🌊 Bhote Koshi River Discharge Forecasting")
st.subheader("Next-Day Forecasting and Experimental Early Warning")

st.caption(
    "Research prototype using a Random Forest regression model. "
    "Predictions and the 4.08 m³/s threshold are experimental, "
    "not official flood warnings."
)

# --------------------------------------------------
# 4. CHECK FILES AND LOAD THEM
# --------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Place Bhote_koshi.csv inside the project's data folder."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info(
        "Run train_model.py first and make sure it creates "
        "bhote_koshi_rf_model_final.pkl."
    )
    st.stop()

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.stop()

if df.empty:
    st.error("The dataset contains no valid discharge records.")
    st.stop()

latest = df.iloc[-1]
latest_date = latest["date"]
forecast_date = latest_date + timedelta(days=1)
latest_discharge = float(latest["river_discharge_m3s"])

# --------------------------------------------------
# 5. PREDICT NEXT-DAY DISCHARGE
# --------------------------------------------------
missing_features = [column for column in FEATURES if column not in df.columns]

if missing_features:
    st.error(
        "The dataset is missing model input columns: "
        + ", ".join(missing_features)
    )
    st.info(
        "Check that FEATURES in app.py matches the feature columns "
        "used to train your saved Random Forest model."
    )
    st.stop()

input_data = latest[FEATURES].to_frame().T

for column in FEATURES:
    input_data[column] = pd.to_numeric(
        input_data[column], errors="coerce"
    )

if input_data.isna().any().any():
    st.error(
        "The latest record has missing or non-numeric model inputs. "
        "Check the latest row of your CSV before forecasting."
    )
    st.stop()

try:
    predicted_discharge = float(model.predict(input_data)[0])
except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------
st.sidebar.header("Project Information")
st.sidebar.write("**Study area:** Bhote Koshi, Rasuwagadhi")
st.sidebar.write("**Model:** Random Forest Regressor")
st.sidebar.write("**Forecast:** Next-day discharge")
st.sidebar.write("**Unit:** m³/s")
st.sidebar.write(f"**Experimental threshold:** {THRESHOLD:.2f} m³/s")

st.sidebar.warning(
    "This threshold has not been verified as an official DHM warning "
    "threshold. Do not use this dashboard as an operational warning system."
)

# --------------------------------------------------
# 7. KEY METRICS
# --------------------------------------------------
st.markdown("### Forecast Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted Discharge",
    f"{predicted_discharge:.3f} m³/s"
)

col2.metric(
    "Latest Observed Discharge",
    f"{latest_discharge:.3f} m³/s"
)

col3.metric(
    "Forecast Date",
    forecast_date.strftime("%d %b %Y")
)

col4.metric(
    "Experimental Threshold",
    f"{THRESHOLD:.2f} m³/s"
)

# --------------------------------------------------
# 8. EXPERIMENTAL WARNING
# --------------------------------------------------
st.markdown("### Experimental Threshold Check")

if predicted_discharge >= THRESHOLD:
    st.warning(
        f"Predicted discharge ({predicted_discharge:.3f} m³/s) is at "
        f"or above the experimental threshold ({THRESHOLD:.2f} m³/s). "
        "Further hydrological verification is required."
    )
else:
    st.info(
        f"Predicted discharge ({predicted_discharge:.3f} m³/s) is below "
        f"the experimental threshold ({THRESHOLD:.2f} m³/s). "
        "This does not establish that flooding is impossible."
    )

# --------------------------------------------------
# 9. HISTORICAL DISCHARGE CHART
# --------------------------------------------------
st.markdown("### Historical River Discharge")

chart_df = df[["date", "river_discharge_m3s"]].copy()

fig = px.line(
    chart_df,
    x="date",
    y="river_discharge_m3s",
    title="Observed Bhote Koshi River Discharge",
    labels={
        "date": "Date",
        "river_discharge_m3s": "Discharge (m³/s)",
    },
)

fig.add_hline(
    y=THRESHOLD,
    line_dash="dash",
    annotation_text="Experimental threshold",
    annotation_position="top left",
)

fig.add_trace(
    go.Scatter(
        x=[forecast_date],
        y=[predicted_discharge],
        mode="markers",
        name="Next-day prediction",
        marker=dict(size=12, symbol="diamond"),
    )
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------
st.markdown("### Random Forest Model Evaluation")

if PREDICTIONS_PATH.exists():
    try:
        predictions_df = pd.read_csv(PREDICTIONS_PATH)

        required_prediction_columns = [
            "Actual_Discharge_m3s",
            "RF_Predicted_Discharge_m3s",
        ]

        if all(
            column in predictions_df.columns
            for column in required_prediction_columns
        ):
            evaluation_df = predictions_df[
                required_prediction_columns
            ].apply(pd.to_numeric, errors="coerce").dropna()

            if len(evaluation_df) >= 2:
                actual = evaluation_df["Actual_Discharge_m3s"]
                predicted = evaluation_df["RF_Predicted_Discharge_m3s"]

                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, predicted)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{mae:.4f} m³/s")
                m2.metric("MSE", f"{mse:.5f}")
                m3.metric("RMSE", f"{rmse:.4f} m³/s")
                m4.metric("R² Score", f"{r2:.4f}")

                comparison_fig = go.Figure()

                comparison_fig.add_trace(
                    go.Scatter(
                        y=actual,
                        mode="lines",
                        name="Actual discharge",
                    )
                )

                comparison_fig.add_trace(
                    go.Scatter(
                        y=predicted,
                        mode="lines",
                        name="Predicted discharge",
                    )
                )

                comparison_fig.update_layout(
                    title="Actual vs Predicted Discharge on Test Data",
                    xaxis_title="Test observation",
                    yaxis_title="Discharge (m³/s)",
                )

                st.plotly_chart(
                    comparison_fig,
                    use_container_width=True
                )

                st.dataframe(
                    evaluation_df,
                    use_container_width=True
                )

                st.download_button(
                    label="Download Test Predictions CSV",
                    data=evaluation_df.to_csv(index=False).encode("utf-8"),
                    file_name="test_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "At least two valid test predictions are needed "
                    "to display the evaluation metrics."
                )
        else:
            st.warning(
                "The prediction CSV does not contain the expected columns: "
                "Actual_Discharge_m3s and RF_Predicted_Discharge_m3s."
            )

    except Exception as error:
        st.warning(f"Could not read test predictions: {error}")
else:
    st.info(
        "Model evaluation is unavailable because test_predictions.csv "
        "has not been generated yet. Run train_model.py first."
    )

# --------------------------------------------------
# 11. LATEST WEATHER CONDITIONS
# --------------------------------------------------
st.markdown("### Latest Recorded Weather and River Conditions")

weather_columns = [
    "rain_mm",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "temperature_mean_c",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh",
]

available_columns = [
    column for column in weather_columns if column in df.columns
]

if available_columns:
    st.dataframe(
        latest[available_columns].to_frame(
            name="Latest Recorded Value"
        ),
        use_container_width=True,
    )

# --------------------------------------------------
# 12. DATA TABLE
# --------------------------------------------------
with st.expander("View historical dataset"):
    st.dataframe(df, use_container_width=True)

st.caption(
    "Research use only. A discharge threshold alone cannot reliably "
    "determine flood risk. Official station thresholds, water level, "
    "rainfall forecasts, and hydrological validation are needed for "
    "a dependable early-warning system."
)
