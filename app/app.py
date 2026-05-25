import math
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="NYC Pavement + Truck Route Exposure",
    layout="wide"
)

st.title("NYC Pavement Condition + Truck Route Exposure")
st.write(
    "Street-level analysis of pavement ratings, truck exposure, and prediction models using NYC Open Data."
)

PAVEMENT_URL = "https://data.cityofnewyork.us/resource/6yyb-pb25.csv?$limit=498012"
TRUCK_URL = "https://data.cityofnewyork.us/resource/jjja-shxy.csv?$limit=50000"

FEATURE_COLS = [
    "truck_segment_count",
    "truck_route_count",
    "local_routes",
    "through_routes",
    "log_truck_exposure",
    "truck_intensity_ratio",
    "locationgeometry_stlength",
    "avg_truck_shape_leng",
    "ismultipass",
]


def clean_text(series, mode=None):
    s = series.fillna("").astype(str).str.strip()
    if mode == "upper":
        return s.str.upper()
    if mode == "title":
        return s.str.title()
    return s


@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    pavement = pd.read_csv(PAVEMENT_URL, low_memory=False)
    truck = pd.read_csv(TRUCK_URL, low_memory=False)

    if "boroughname" in pavement.columns:
        pavement = pavement.rename(columns={"boroughname": "borough"})
    if "boroname" in truck.columns:
        truck = truck.rename(columns={"boroname": "borough"})

    for col in ["borough", "onstreetna", "systemrating", "locationgeometry_stlength", "ismultipass"]:
        if col not in pavement.columns:
            pavement[col] = None

    for col in ["borough", "street", "streetcode", "shape_leng", "truckroute", "routetype"]:
        if col not in truck.columns:
            truck[col] = None

    pavement["borough"] = clean_text(pavement["borough"], mode="title")
    truck["borough"] = clean_text(truck["borough"], mode="title")

    pavement["street_norm"] = clean_text(pavement["onstreetna"], mode="upper")
    truck["street_norm"] = clean_text(truck["street"], mode="upper")

    pavement["systemrating"] = pd.to_numeric(pavement["systemrating"], errors="coerce")
    pavement["locationgeometry_stlength"] = pd.to_numeric(
        pavement["locationgeometry_stlength"], errors="coerce"
    )
    pavement["ismultipass"] = pd.to_numeric(pavement["ismultipass"], errors="coerce")

    truck["streetcode"] = pd.to_numeric(truck["streetcode"], errors="coerce")
    truck["shape_leng"] = pd.to_numeric(truck["shape_leng"], errors="coerce")
    truck["truckroute"] = clean_text(truck["truckroute"], mode="upper")
    truck["routetype"] = clean_text(truck["routetype"], mode="title")

    pavement = pavement.drop_duplicates()
    truck = truck.drop_duplicates()

    pavement = pavement[
        (pavement["borough"] != "") &
        (pavement["street_norm"] != "") &
        (pavement["systemrating"].notna())
    ].copy()

    truck = truck[
        (truck["borough"] != "") &
        (truck["street_norm"] != "")
    ].copy()

    pavement["rating_group"] = pavement["systemrating"].apply(
        lambda x: "Good" if x >= 8 else ("Fair" if x >= 4 else "Poor")
    )

    truck_street = (
        truck.groupby(["borough", "street_norm"], as_index=False)
        .agg(
            truck_segment_count=("streetcode", "nunique"),
            truck_route_count=("truckroute", lambda s: (s == "Y").sum()),
            local_routes=("routetype", lambda s: (s == "Local").sum()),
            through_routes=("routetype", lambda s: (s == "Through").sum()),
            avg_truck_shape_leng=("shape_leng", "mean"),
        )
    )

    model_df = pavement.merge(
        truck_street,
        on=["borough", "street_norm"],
        how="left"
    )

    model_df = model_df.fillna({
        "truck_segment_count": 0,
        "truck_route_count": 0,
        "local_routes": 0,
        "through_routes": 0,
        "avg_truck_shape_leng": 0,
    })

    model_df["log_truck_exposure"] = model_df["truck_segment_count"].apply(math.log1p)

    model_df["truck_intensity_ratio"] = model_df.apply(
        lambda row: (
            row["truck_route_count"] / row["truck_segment_count"]
            if row["truck_segment_count"] > 0 else 0
        ),
        axis=1
    )

    for col in FEATURE_COLS:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

    borough_summary = (
        model_df.groupby("borough", as_index=False)
        .agg(
            pavement_rows=("street_norm", "count"),
            avg_rating=("systemrating", "mean"),
            avg_truck_segments=("truck_segment_count", "mean"),
            avg_truck_routes=("truck_route_count", "mean"),
            avg_truck_intensity=("truck_intensity_ratio", "mean"),
        )
        .sort_values("borough")
    )

    round_cols = [
        "avg_rating",
        "avg_truck_segments",
        "avg_truck_routes",
        "avg_truck_intensity",
    ]
    borough_summary[round_cols] = borough_summary[round_cols].round(2)

    return model_df, borough_summary


@st.cache_data(show_spinner=True)
def train_models(model_df):
    model_input = model_df[
        ["borough", "street_norm", "systemrating"] + FEATURE_COLS
    ].dropna(subset=["systemrating"]).copy()

    sampled = False
    max_model_rows = 100000
    original_rows = len(model_input)

    if len(model_input) > max_model_rows:
        model_input = model_input.sample(max_model_rows, random_state=42).reset_index(drop=True)
        sampled = True
    else:
        model_input = model_input.reset_index(drop=True)

    X = model_input[FEATURE_COLS].copy()
    y = model_input["systemrating"].astype(float).copy()
    meta = model_input[["borough", "street_norm", "systemrating"]].copy()

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    gbt = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    rf.fit(X_train, y_train)
    gbt.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    gbt_pred = gbt.predict(X_test)

    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting"],
        "RMSE": [
            math.sqrt(mean_squared_error(y_test, rf_pred)),
            math.sqrt(mean_squared_error(y_test, gbt_pred)),
        ],
        "R2": [
            r2_score(y_test, rf_pred),
            r2_score(y_test, gbt_pred),
        ],
    }).round(4)

    feature_importance_df = (
        pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": rf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    predictions_df = meta_test.reset_index(drop=True).copy()
    predictions_df["rf_prediction"] = rf_pred
    predictions_df["gbt_prediction"] = gbt_pred
    predictions_df["rf_error"] = predictions_df["systemrating"] - predictions_df["rf_prediction"]
    predictions_df["gbt_error"] = predictions_df["systemrating"] - predictions_df["gbt_prediction"]

    return {
        "metrics_df": metrics_df,
        "feature_importance_df": feature_importance_df,
        "predictions_df": predictions_df,
        "sampled": sampled,
        "original_rows": original_rows,
        "model_rows": len(model_input),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


model_df, borough_summary = load_and_prepare_data()
model_results = train_models(model_df)

metrics_df = model_results["metrics_df"]
feature_importance_df = model_results["feature_importance_df"]
predictions_df = model_results["predictions_df"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Joined Rows", f"{len(model_df):,}")
col2.metric("Average Rating", f"{model_df['systemrating'].mean():.2f}")
col3.metric("Model Rows", f"{model_results['model_rows']:,}")
col4.metric("Test Rows", f"{model_results['test_rows']:,}")

if model_results["sampled"]:
    st.info(
        f"Model training was capped at {model_results['model_rows']:,} rows "
        f"from {model_results['original_rows']:,} total rows to keep the app responsive."
    )

tab1, tab2, tab3 = st.tabs(["Borough Overview", "Prediction Model", "Street Data"])

with tab1:
    st.subheader("Borough Summary")
    st.dataframe(borough_summary, use_container_width=True)

    fig1 = px.bar(
        borough_summary,
        x="borough",
        y="avg_rating",
        color="borough",
        title="Average Pavement Rating by Borough"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        borough_summary,
        x="borough",
        y="avg_truck_segments",
        color="borough",
        title="Average Street-Level Truck Segment Exposure by Borough"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        borough_summary,
        x="avg_truck_intensity",
        y="avg_rating",
        size="pavement_rows",
        text="borough",
        color="borough",
        title="Truck Exposure vs Pavement Rating by Borough"
    )
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, use_container_width=True)

    rating_pdf = model_df[["borough", "systemrating"]].dropna().copy()

    fig4 = px.histogram(
        rating_pdf,
        x="systemrating",
        nbins=30,
        color="borough",
        title="Distribution of Pavement Ratings"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.box(
        rating_pdf,
        x="borough",
        y="systemrating",
        color="borough",
        title="Pavement Rating Distribution by Borough"
    )
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
    st.subheader("Model Performance")
    st.dataframe(metrics_df, use_container_width=True)

    selected_model = st.radio(
        "Prediction view",
        ["Random Forest", "Gradient Boosting"],
        horizontal=True
    )

    pred_col = "rf_prediction" if selected_model == "Random Forest" else "gbt_prediction"
    err_col = "rf_error" if selected_model == "Random Forest" else "gbt_error"

    plot_df = predictions_df.copy()
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)

    min_axis = min(plot_df["systemrating"].min(), plot_df[pred_col].min())
    max_axis = max(plot_df["systemrating"].max(), plot_df[pred_col].max())

    fig6 = px.scatter(
        plot_df,
        x="systemrating",
        y=pred_col,
        color="borough",
        title=f"Actual vs Predicted Pavement Ratings — {selected_model}",
        labels={
            "systemrating": "Actual Rating",
            pred_col: "Predicted Rating",
        }
    )
    fig6.add_shape(
        type="line",
        x0=min_axis,
        y0=min_axis,
        x1=max_axis,
        y1=max_axis,
        line=dict(color="black", dash="dash")
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Random Forest Feature Importance")
    fig7 = px.bar(
        feature_importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )
    fig7.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Prediction Sample")
    preview_cols = ["borough", "street_norm", "systemrating", pred_col, err_col]
    st.dataframe(
        predictions_df[preview_cols].head(100),
        use_container_width=True
    )

with tab3:
    st.subheader("Street-Level Data Sample")
    sample_cols = [
        "borough",
        "street_norm",
        "systemrating",
        "rating_group",
        "truck_segment_count",
        "truck_route_count",
        "local_routes",
        "through_routes",
        "avg_truck_shape_leng",
        "truck_intensity_ratio",
    ]
    st.dataframe(model_df[sample_cols].head(200), use_container_width=True)
