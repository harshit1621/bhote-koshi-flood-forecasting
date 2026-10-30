# Bhote Koshi River Discharge Forecasting Using Random Forest

A machine learning-based project for next-day river discharge forecasting in the Bhote Koshi region of Nepal using Random Forest Regressor, with an interactive dashboard for forecast visualization and experimental threshold-based flood warnings.

## 1. Project Objective and Study Area

The objective of this project is to develop a machine learning model to forecast next-day river discharge in the Bhote Koshi region of Nepal.

### Key Objectives
- Predict next-day river discharge in cubic metres per second (m³/s).
- Train a Random Forest Regressor using historical hydrological data.
- Evaluate model performance using MAE, MSE, RMSE, and R².
- Visualize discharge forecasts through an interactive dashboard.
- Demonstrate an experimental threshold-based warning mechanism.

**Study Area:** Bhote Koshi region, Nepal.

## 2. Dataset Source and Observation Period

The project uses historical hydrological data for the Bhote Koshi region.

- **Dataset source:** [Insert actual Kaggle dataset or original data provider]
- **Study area:** Bhote Koshi, Nepal
- **Observation end date:** 31 August 2026
- **Prediction target:** Next-day river discharge (m³/s)
- **Input features:** Historical discharge and other available hydrological or meteorological variables.

The dataset is sorted chronologically, cleaned, and prepared for supervised learning. The target variable represents the following day's river discharge.

The exact observation station, start date, dataset source, units, and missing-value treatment should be documented for reproducibility.

## 3. Machine Learning Methodology

### Random Forest Regressor

Random Forest Regressor is an ensemble machine learning algorithm that combines predictions from multiple decision trees to estimate continuous values. It can capture nonlinear relationships between historical hydrological variables and river discharge.

### Workflow

1. Load and inspect the dataset.
2. Clean missing values and sort observations by date.
3. Create the next-day discharge prediction target.
4. Split the data chronologically into training and testing sets.
5. Train the Random Forest Regressor.
6. Generate predictions for the held-out test period.
7. Evaluate model performance using regression metrics.
8. Display forecasts and threshold comparisons on the dashboard.

To avoid data leakage, only information available at the forecast time should be used as input.

## 4. Model Evaluation

The Random Forest model is evaluated using four regression metrics.

| Metric | Description | Preferred Result |
|---|---|---|
| MAE | Mean Absolute Error | Lower |
| MSE | Mean Squared Error | Lower |
| RMSE | Root Mean Squared Error | Lower |
| R² | Coefficient of Determination | Generally closer to 1 |

### Random Forest Results

Replace the placeholders with actual values calculated on your held-out test set.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Random Forest Regressor | TBD | TBD | TBD | TBD |

These metrics measure how closely the predicted discharge matches the observed discharge over the evaluation period.

## 5. Dashboard Preview

The interactive dashboard displays:

- Predicted river discharge.
- Latest observed discharge.
- Forecast date.
- Experimental discharge threshold.
- Threshold-based classification.

## 7. Limitations of the Experimental Discharge Threshold

The dashboard currently uses an **experimental discharge threshold of 4.08 m³/s** to demonstrate threshold-based classification.

This threshold should not be considered an official flood-warning level unless it is verified against an authoritative source for the relevant hydrological station.

### Limitations

- Discharge alone may not represent all flood conditions.
- Rainfall, upstream inflows, and water-level changes can influence flood risk.
- Recursive forecasting may accumulate prediction errors over longer forecast horizons.
- Model performance depends on data quality, feature availability, and the representation of extreme events.
- A predicted discharge below the experimental threshold does not guarantee that the area is safe from flooding.

Future improvements may include additional rainfall and water-level data, official station-specific warning thresholds, and validation against documented flood events.

## Future Work

- Incorporate additional hydrological and meteorological variables.
- Evaluate multi-day forecasting performance.
- Investigate model performance during high-discharge events.
- Validate thresholds against official hydrological records.
- Improve dashboard visualizations and forecast uncertainty estimates.

## Disclaimer

This project is an academic forecasting experiment. Predictions and experimental thresholds are not substitutes for official hydrological forecasts, emergency warnings, or instructions from relevant authorities.

## Author

Harshit Pant 
Department of Computer Science and Engineering  
SRM Institute of Science and Technology

GitHub: [harshit1621](https://github.com/harshit1621)
