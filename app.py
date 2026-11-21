"""
app.py: Delivery Time Estimation app (Streamlit)

How to run (in the project folder):
    streamlit run app.py

Layout:
    Sidebar        -> the order form (all inputs)
    Tab 1          -> Predict ETA: predicted time, likely range, distance, map
    Tab 2          -> What-if: how the ETA changes when one factor changes
    Tab 3          -> Model insights: scores, model comparison, charts

It needs two files made by notebook 3:
    models/model.pkl          the trained pipeline (preprocessing + XGBoost)
    models/model_report.json  scores, prediction range and input options

Safety net: if model.pkl cannot be loaded on a computer (for example a cloud
server with different library versions), the app trains the same pipeline again
from data/processed/features_data.csv. This takes about 10 seconds, once.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# "Agg" draws charts as images without opening windows (safe inside a web app)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Basic settings
# ---------------------------------------------------------------

# page title in the browser tab, and use the full screen width
st.set_page_config(page_title="Delivery Time Estimation", layout="wide")

# the folder this file is in, so the paths work from anywhere
APP_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = APP_FOLDER / "models" / "model.pkl"
REPORT_PATH = APP_FOLDER / "models" / "model_report.json"
FEATURES_PATH = APP_FOLDER / "data" / "processed" / "features_data.csv"

# one colour for normal bars and one for the bar that matches the current order
COLOR = "#3B6EA5"
LIGHT_COLOR = "#B4B2A9"

TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]


# ---------------------------------------------------------------
# Load the report (scores, prediction range, input options)
# ---------------------------------------------------------------

# @st.cache_data reads the file only once, not again on every click
@st.cache_data
def load_report():
    with open(REPORT_PATH) as file:
        return json.load(file)


# show a clear message instead of a crash if the report is missing
try:
    report = load_report()
except Exception as error:
    st.error("Could not load models/model_report.json. Run notebook 3 (03_Models_and_Evaluation.ipynb) first.")
    st.caption(f"Details: {error}")
    st.stop()

# the error range learned in notebook 3 (about -4.7 to +5.0 minutes)
RANGE_LOW = report["prediction_range"]["low"]
RANGE_HIGH = report["prediction_range"]["high"]
OPTIONS = report["category_options"]


# ---------------------------------------------------------------
# Feature functions (exactly the same as in notebooks 2 and 3)
# ---------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    # straight-line distance between two points on the Earth's surface, in km
    R = 6371
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def make_features(order):
    # turn the raw order details (a dictionary) into the 15 model features (a one-row table)
    row = {
        "distance_km": haversine_km(order["restaurant_lat"], order["restaurant_lon"],
                                    order["delivery_lat"], order["delivery_lon"]),
        "multiple_deliveries": order["multiple_deliveries"],
        "rider_rating": order["rider_rating"],
        "is_high_rated": 1.0 if order["rider_rating"] >= 4.5 else 0.0,
        "rider_age": order["rider_age"],
        "is_age_30_plus": 1.0 if order["rider_age"] >= 30 else 0.0,
        "vehicle_condition": order["vehicle_condition"],
        "order_hour": order["order_hour"],
        "is_peak_hour": 1.0 if 19 <= order["order_hour"] <= 21 else 0.0,
        "weather": order["weather"],
        "traffic": order["traffic"],
        "city_type": order["city_type"],
        "festival": order["festival"],
        "vehicle_type": order["vehicle_type"],
        "order_type": order["order_type"],
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------
# Load the model (with a safety net)
# ---------------------------------------------------------------

def train_backup_model():
    # Rebuild the same pipeline as notebook 3: same 80/20 split, same preprocessing,
    # same tuned XGBoost settings. Used only if model.pkl cannot be loaded here.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBRegressor

    numeric = report["numeric_features"]
    categorical = report["categorical_features"]

    data = pd.read_csv(FEATURES_PATH)
    X = data[numeric + categorical]
    y = data["delivery_time_min"]

    # the same training rows as notebook 3 (the test rows stay unseen)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("fill_missing", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("fill_missing", SimpleImputer(strategy="most_frequent")),
                          ("one_hot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    # the settings RandomizedSearchCV picked in notebook 3
    xgboost_model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=8, subsample=0.7,
                                 colsample_bytree=1.0, min_child_weight=1, random_state=42, n_jobs=-1)

    pipeline = Pipeline([("pre", preprocessor), ("model", xgboost_model)])
    pipeline.fit(X_train, y_train)
    return pipeline


# @st.cache_resource keeps the model in memory, so it loads (or trains) only once
@st.cache_resource
def load_model():
    # an example order, used to check that the loaded model really works on this computer
    test_order = {"restaurant_lat": 12.914264, "restaurant_lon": 77.6784,
                  "delivery_lat": 12.954264, "delivery_lon": 77.7184,
                  "multiple_deliveries": 1, "rider_rating": 4.6, "rider_age": 32,
                  "vehicle_condition": 1, "order_hour": 20, "weather": "Fog", "traffic": "Jam",
                  "city_type": "Metropolitan", "festival": "No", "vehicle_type": "motorcycle",
                  "order_type": "Meal"}
    try:
        saved_model = joblib.load(MODEL_PATH)
        saved_model.predict(make_features(test_order))
        return saved_model, False          # False = the saved model.pkl is used
    except Exception:
        return train_backup_model(), True  # True = the model was rebuilt on this computer


try:
    with st.spinner("Loading the model..."):
        model, model_was_rebuilt = load_model()
except Exception as error:
    st.error("Could not load or rebuild the model. Check that the models/ and data/processed/ folders exist.")
    st.caption(f"Details: {error}")
    st.stop()


def predict_eta(order):
    # predicted delivery time in minutes for one order
    return float(model.predict(make_features(order))[0])


def show_chart(fig):
    # draw the chart on the page, then close it so memory doesn't fill up
    st.pyplot(fig)
    plt.close(fig)


def what_if_chart(labels, values, current_label, title):
    # bar chart of the ETA for each option; the current choice gets the dark colour
    colors = [COLOR if label == current_label else LIGHT_COLOR for label in labels]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar([str(label) for label in labels], values, color=colors)
    # write the minutes on top of each bar
    for position, value in enumerate(values):
        ax.text(position, value + 0.5, f"{value:.0f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(values) + 6)
    ax.set_ylabel("ETA (min)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------
# Sidebar: the order form
# ---------------------------------------------------------------

st.sidebar.header("Order details")

st.sidebar.subheader("Locations")
# default example: a restaurant in Bangalore and a customer about 6 km away
restaurant_lat = st.sidebar.number_input("Restaurant latitude", value=12.914264, format="%.6f")
restaurant_lon = st.sidebar.number_input("Restaurant longitude", value=77.678400, format="%.6f")
delivery_lat = st.sidebar.number_input("Customer latitude", value=12.954264, format="%.6f")
delivery_lon = st.sidebar.number_input("Customer longitude", value=77.718400, format="%.6f")

st.sidebar.subheader("Conditions")
city_type = st.sidebar.selectbox("City type", OPTIONS["city_type"], index=OPTIONS["city_type"].index("Metropolitan"))
weather = st.sidebar.selectbox("Weather", OPTIONS["weather"], index=OPTIONS["weather"].index("Fog"))
traffic = st.sidebar.selectbox("Traffic", TRAFFIC_ORDER, index=3)
festival = st.sidebar.selectbox("Festival day?", OPTIONS["festival"], index=0)
order_hour = st.sidebar.slider("Order hour (24 h)", min_value=8, max_value=23, value=20,
                               help="The data has orders between 8:00 and 23:59.")

st.sidebar.subheader("Rider and order")
rider_age = st.sidebar.slider("Rider age", min_value=20, max_value=39, value=32)
rider_rating = st.sidebar.slider("Rider rating", min_value=2.5, max_value=5.0, value=4.6, step=0.1)
vehicle_type = st.sidebar.selectbox("Vehicle type", OPTIONS["vehicle_type"], index=OPTIONS["vehicle_type"].index("motorcycle"))
vehicle_condition = st.sidebar.selectbox("Vehicle condition (0 to 3)", [0, 1, 2, 3], index=1)
multiple_deliveries = st.sidebar.selectbox("Extra orders the rider is carrying", [0, 1, 2, 3], index=1)
order_type = st.sidebar.selectbox("Order type", OPTIONS["order_type"], index=OPTIONS["order_type"].index("Meal"))

# collect every input into one dictionary (same keys as make_features expects)
order = {
    "restaurant_lat": restaurant_lat, "restaurant_lon": restaurant_lon,
    "delivery_lat": delivery_lat, "delivery_lon": delivery_lon,
    "city_type": city_type, "weather": weather, "traffic": traffic, "festival": festival,
    "order_hour": order_hour, "rider_age": rider_age, "rider_rating": rider_rating,
    "vehicle_type": vehicle_type, "vehicle_condition": vehicle_condition,
    "multiple_deliveries": multiple_deliveries, "order_type": order_type,
}

# a small note if the safety net was used
if model_was_rebuilt:
    st.sidebar.caption("Note: the model was rebuilt on this server from the saved features, "
                       "because the saved model.pkl did not match this server's library versions.")

# predict once for the current order; every tab uses these values
eta = predict_eta(order)
distance = haversine_km(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)


# ---------------------------------------------------------------
# Page header and tabs
# ---------------------------------------------------------------

st.title("Delivery Time Estimation")
st.caption(f"Predicts food delivery time from order, rider, weather and traffic details. "
           f"Model: {report['best_model']}, average error {report['test_metrics']['MAE']:.1f} min on unseen orders.")

tab_predict, tab_what_if, tab_insights = st.tabs(["Predict ETA", "What-if", "Model insights"])


# ---------------------------------------------------------------
# Tab 1: Predict ETA
# ---------------------------------------------------------------

with tab_predict:
    # warn if the distance is outside what the model learned from (1.5 to 21 km)
    if distance > 25:
        st.warning(f"The distance is {distance:.1f} km. The model learned from trips of 1.5 to 21 km, "
                   "so this prediction is less reliable. Check the coordinates.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted delivery time", f"{eta:.0f} min")
    col2.metric("Likely range", f"{eta + RANGE_LOW:.0f} to {eta + RANGE_HIGH:.0f} min")
    col3.metric("Distance (straight line)", f"{distance:.1f} km")

    st.caption(f"The range holds for {report['prediction_range']['coverage_percent']:.0f}% of real deliveries "
               f"in our test set of unseen orders.")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Order summary")
        summary = pd.DataFrame({
            "Detail": ["City type", "Weather", "Traffic", "Festival day", "Order time",
                       "Rider age", "Rider rating", "Vehicle", "Vehicle condition",
                       "Extra orders carried", "Order type"],
            "Value": [city_type, weather, traffic, festival, f"{order_hour}:00",
                      str(rider_age), str(rider_rating), vehicle_type, str(vehicle_condition),
                      str(multiple_deliveries), order_type],
        })
        st.dataframe(summary, hide_index=True)

    with right:
        st.subheader("Restaurant and customer")
        # st.map needs a table with "lat" and "lon" columns
        points = pd.DataFrame({"lat": [restaurant_lat, delivery_lat], "lon": [restaurant_lon, delivery_lon]})
        # zoom=11 shows about a whole city; size = dot radius in metres (needs internet for the map tiles)
        st.map(points, zoom=11, size=250)


# ---------------------------------------------------------------
# Tab 2: What-if
# ---------------------------------------------------------------

with tab_what_if:
    st.write(f"Current order: **{eta:.0f} min**. Each chart changes one factor and keeps everything else "
             "the same. The dark bar is your current choice.")

    col1, col2 = st.columns(2)

    with col1:
        # ETA for every traffic level
        values = [predict_eta({**order, "traffic": level}) for level in TRAFFIC_ORDER]
        show_chart(what_if_chart(TRAFFIC_ORDER, values, traffic, "Traffic"))

        # ETA for 0 to 3 extra orders
        extra_options = [0, 1, 2, 3]
        values = [predict_eta({**order, "multiple_deliveries": n}) for n in extra_options]
        show_chart(what_if_chart(extra_options, values, multiple_deliveries, "Extra orders carried"))

    with col2:
        # ETA for every weather type
        weather_options = OPTIONS["weather"]
        values = [predict_eta({**order, "weather": w}) for w in weather_options]
        show_chart(what_if_chart(weather_options, values, weather, "Weather"))

        # ETA on a normal day vs a festival day
        festival_options = OPTIONS["festival"]
        values = [predict_eta({**order, "festival": f}) for f in festival_options]
        show_chart(what_if_chart(festival_options, values, festival, "Festival day"))


# ---------------------------------------------------------------
# Tab 3: Model insights
# ---------------------------------------------------------------

with tab_insights:
    metrics = report["test_metrics"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best model", report["best_model"])
    col2.metric("Average error (MAE)", f"{metrics['MAE']:.2f} min")
    col3.metric("R² score", f"{metrics['R2']:.2f}")
    col4.metric("Within 5 minutes", f"{metrics['within_5_min_percent']:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Model comparison (test set)")
        comparison = pd.DataFrame(report["model_comparison_test"]).round(2)
        st.dataframe(comparison, hide_index=True)

        # bar chart of test MAE for every model, best one in the dark colour
        fig, ax = plt.subplots(figsize=(6, 3.4))
        colors = [COLOR if name == report["best_model"] else LIGHT_COLOR for name in comparison["model"]]
        ax.barh(comparison["model"], comparison["MAE"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Average error (min, lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        show_chart(fig)

    with right:
        st.subheader("What drives delivery time")
        # how much the error grows (in minutes) when each feature is shuffled
        importance = pd.Series(report["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(importance.index, importance.values, color=COLOR)
        ax.set_xlabel("Extra error when the feature is shuffled (min)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        show_chart(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Predicted vs actual")
        sample = np.array(report["actual_vs_predicted_sample"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sample[:, 0], sample[:, 1], s=6, alpha=0.4, color=COLOR)
        ax.plot([10, 55], [10, 55], color="black", linestyle="--")
        ax.set_xlabel("Actual time (min)")
        ax.set_ylabel("Predicted time (min)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        show_chart(fig)

    with right:
        st.subheader("Error by traffic level")
        traffic_error = pd.Series(report["error_by_traffic"]).reindex(TRAFFIC_ORDER)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(traffic_error.index, traffic_error.values, color=COLOR)
        ax.set_ylabel("Average error (min)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        show_chart(fig)
