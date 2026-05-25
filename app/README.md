# NYC Pavement Truck Analysis App

An interactive Databricks + Streamlit project that analyzes how truck-route exposure relates to pavement condition in New York City and evaluates how well truck-related features predict pavement ratings.

## Live App

**Databricks App:**  
[NYC Pavement Truck Analysis App](https://pavement-truck-analysis-app-7474655415402641.aws.databricksapps.com)

## Overview

This project studies the relationship between truck-route exposure and pavement condition using NYC Open Data. It combines data cleaning, street-level feature engineering, exploratory analysis, predictive modeling, and an interactive Streamlit app deployed through Databricks Apps.

The project is designed to answer two related questions:

1. Do truck-route patterns appear to be associated with pavement condition?
2. Can truck-route and pavement-related features be used to predict pavement ratings with useful accuracy?

## Motivation

Road wear is influenced by many factors, including traffic load, road design, maintenance history, and environmental conditions. Because heavy vehicle traffic is often expected to contribute to pavement deterioration, this project explores whether truck-route exposure provides meaningful predictive signal for street-level pavement condition in NYC.

## Data Sources

This project uses two public NYC datasets.

| Dataset | Purpose | Example fields used |
|---------|---------|---------------------|
| NYC Pavement Condition data | Measures pavement quality and street attributes | `borough`, `onstreetna`, `systemrating`, `locationgeometry_stlength`, `ismultipass` |
| NYC Truck Route data | Measures truck-route exposure and route characteristics | `borough`, `street`, `streetcode`, `shape_leng`, `truckroute`, `routetype` |

## Methodology

The workflow followed these steps:

1. Load pavement-condition and truck-route datasets.
2. Standardize street and borough names for joining.
3. Create normalized street keys using `borough` and `street_norm`.
4. Aggregate truck-route data to the street level.
5. Engineer predictive features from truck-route exposure.
6. Join truck features back to pavement records using `borough` and `street_norm`.
7. Train regression models to predict pavement ratings.
8. Compare model performance and interpret the results visually.

## Engineered Features

The model uses the following features:

| Feature | Description |
|---------|-------------|
| `truck_segment_count` | Number of truck-data segments associated with a street |
| `truck_route_count` | Count of truck-route matches on the street |
| `local_routes` | Count of local truck routes |
| `through_routes` | Count of through truck routes |
| `log_truck_exposure` | Log-transformed truck segment count |
| `truck_intensity_ratio` | Truck-route count divided by truck segment count |
| `locationgeometry_stlength` | Pavement street-length field |
| `avg_truck_shape_leng` | Average truck-segment length |
| `ismultipass` | Pavement multipass indicator |

## Models Used

Two regression models were used:

- Random Forest Regressor
- Gradient Boosting Regressor

These models were trained on the engineered street-level feature set to estimate pavement `systemrating`.

## Key Findings

The analysis shows a real but weak predictive relationship between truck-route features and pavement ratings. Both Random Forest and Gradient Boosting produced \(R^2\) values of about 0.08, which means the current feature set explains only a small share of the variation in pavement condition.

The modeling workflow is not borough-only. Truck exposure was constructed by matching records on `borough` and `street_norm`, then joining those street-level features back to pavement records using the same keys.

Feature-importance results suggest that `locationgeometry_stlength` is the strongest predictor in the current setup, followed by truck-intensity variables and `ismultipass`. This suggests truck exposure contributes useful signal, but it is not the dominant driver of pavement rating.

At the borough level, Brooklyn has the highest average pavement rating at 7.28, while the Bronx has the lowest at 6.33. Manhattan shows the highest average matched truck-segment exposure at 0.43, while Queens is lower at 0.22.

Model error is still fairly large. An RMSE of about 3.12 on a 0–10 pavement scale suggests the predictions are not highly precise, and the prediction samples indicate the model tends to compress values toward the middle.

Overall, truck-route intensity appears to be associated with pavement condition, but it explains only a limited share of pavement-rating variation in the current street-matched model.

## Model Performance Summary

| Model | Approx. \(R^2\) | Approx. RMSE | Interpretation |
|------|------------------|--------------|----------------|
| Random Forest | 0.08 | 3.12 | Captures limited signal, but overall explanatory power is weak |
| Gradient Boosting | 0.08 | 3.12 | Similar performance, also weak predictive power |

## Borough-Level Takeaways

| Borough | Finding |
|---------|---------|
| Brooklyn | Highest average pavement rating: 7.28 |
| Bronx | Lowest average pavement rating: 6.33 |
| Manhattan | Highest average matched truck-segment exposure: 0.43 |
| Queens | Lower matched truck-segment exposure: 0.22 |

## Visualizations

The notebook and app include visualizations such as:

- Average pavement rating by borough
- Average truck-segment exposure by borough
- Truck exposure vs. pavement rating
- Pavement rating distribution
- Borough-level box plots
- Actual vs. predicted pavement ratings
- Random Forest feature importance

### Suggested README Figures

If you want to make this README stronger, add screenshots of these visuals:

1. Borough average pavement rating chart
2. Truck exposure vs. pavement rating scatter plot
3. Actual vs. predicted model chart
4. Feature-importance chart
5. One distribution chart or box plot

## App Features

The Streamlit app is organized into three main sections:

- **Borough Overview**: borough summaries and descriptive charts
- **Prediction Model**: model metrics, feature importance, and actual-vs-predicted analysis
- **Street Data**: joined street-level records and engineered features

## Repository Structure

```text
github_dbx_rep/
├── README.md
└── app/
    ├── app.py
    ├── app.yaml
    └── requirements.txt
```

## Deployment Steps

This project is deployed through Databricks Apps using GitHub.

1. Push the project files to GitHub.
2. Keep the Streamlit app files inside the `app/` folder.
3. In Databricks Apps, choose **Deploy**.
4. Select **From Git**.
5. Set:
   - **Git reference:** `main`
   - **Reference type:** `Branch`
   - **Source code path:** `app`
6. Deploy the app and open the generated app URL.

## Requirements

Main Python packages used in the app:

- `streamlit`
- `pandas`
- `plotly`
- `scikit-learn`
- `databricks-sdk`
- `databricks-sql-connector`

## Limitations

This model has low explanatory power, so it should not be interpreted as a strong predictive system. Pavement condition is likely influenced by additional variables not included in the current feature set, such as traffic volume beyond truck routes, maintenance history, construction quality, weather exposure, and roadway age.

The street-name matching approach is useful for exploratory analysis, but it may also introduce noise if street identifiers do not align perfectly across datasets.

## Future Improvements

Possible next steps include:

- Add more roadway and traffic features
- Use more precise spatial matching instead of street-name matching alone
- Incorporate maintenance or resurfacing history
- Test additional regression models
- Evaluate feature interactions and non-linear effects
- Add notebook visual outputs directly into the README

## Author

**Alia Kasem**  
All Foundation of Empirical Research Final Project 2026 
