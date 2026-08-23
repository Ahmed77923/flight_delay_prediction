from pathlib import Path
import sys
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "flight_delay_model.joblib"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_feature import build_features


st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #000000; }
    .block-container { max-width: 1120px; padding-top: 2.5rem; }
    .hero { background: #123b3a; color: white; padding: 2.4rem 2.8rem;
            border-radius: 10px; margin-bottom: 1.5rem; }
    .hero h1 { color: #f8fbfa; font-size: 2.5rem; margin: 0; }
    .hero p { color: #c8dcda; font-size: 1.05rem; margin: .6rem 0 0; }
    .result { background: #e1f2ed; border-left: 6px solid #16836f;
              border-radius: 8px; padding: 1.4rem 1.6rem; }
    .result-label { color: #35615a; font-size: .9rem; text-transform: uppercase;
                    letter-spacing: .08em; }
    .result-value { color: #123b3a; font-size: 2.8rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> Any:
    return joblib.load(MODEL_PATH)


def get_encoder_categories(model: Any, column: str) -> List[str]:
    preprocessor = model.named_steps["preprocessor"]
    for name, transformer, columns in preprocessor.transformers:
        if name == "onehot":
            fitted_encoder = preprocessor.named_transformers_[name]
            category_columns = dict(
                zip(columns, fitted_encoder.categories_)
            )
            return [str(value) for value in category_columns[column]]
    raise KeyError("The saved model does not contain a one-hot transformer.")


def build_inference_frame(
    carrier: str,
    origin: str,
    destination: str,
    flight_date: Any,
    departure_hour: int,
    departure_minute: int,
    arrival_hour: int,
    arrival_minute: int,
    distance: float,
    elapsed_time: float,
) -> pd.DataFrame:
    departure_code = departure_hour * 100 + departure_minute
    arrival_code = arrival_hour * 100 + arrival_minute

    raw_input = pd.DataFrame(
        [
            {
                "ARR_DELAY": np.nan,
                "CANCELLED": 0,
                "DIVERTED": 0,
                "CRS_ELAPSED_TIME": elapsed_time,
                "FL_DATE": pd.Timestamp(flight_date),
                "CRS_DEP_TIME": departure_code,
                "CRS_ARR_TIME": arrival_code,
                "OP_UNIQUE_CARRIER": carrier,
                "ORIGIN": origin,
                "DEST": destination,
                "DISTANCE": distance,
                "ARR_TIME": arrival_code,
                "DEP_TIME": departure_code,
                "TAIL_NUM": "INFERENCE",
            }
        ]
    )

    features = build_features(raw_input)

    # These values require prior observed flights and are unavailable for one flight.
    historical_columns = [
        "carrier_historical_delay",
        "route_historical_delay",
        "origin_historical_delay",
        "origin_recent_delay_7",
        "origin_recent_delay_30",
        "carrier_recent_delay_7",
        "carrier_recent_delay_30",
        "route_recent_delay_7",
        "route_recent_delay_30",
        "carrier_origin_historical_delay",
        "aircraft_previous_delay",
    ]
    features[historical_columns] = np.nan

    return features.drop(columns=["ARR_DELAY", "FL_DATE"])


try:
    model = load_model()
    model_type = type(model.named_steps["model"]).__name__
except Exception:
    model = None
    model_type = "Unavailable"


with st.sidebar:
    st.subheader("Model")
    if model is not None:
        st.success("Model Loaded")
        st.write(f"**Model type:** {model_type}")
        st.write("**Status:** Ready for Prediction")
    else:
        st.error("Model unavailable")
        st.write(f"Expected file: `{MODEL_PATH}`")


st.markdown(
    """
    <div class="hero">
        <h1>Flight Delay Prediction</h1>
        <p>Predict flight delay using a trained machine learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.stop()

carrier_options = get_encoder_categories(model, "OP_UNIQUE_CARRIER")
origin_options = get_encoder_categories(model, "ORIGIN")
destination_options = get_encoder_categories(model, "DEST")

st.subheader("Flight Information")
with st.form("prediction_form"):
    carrier_col, origin_col, destination_col = st.columns(3)
    with carrier_col:
        carrier = st.selectbox("Carrier", carrier_options)
    with origin_col:
        origin = st.selectbox("Origin", origin_options)
    with destination_col:
        destination = st.selectbox("Destination", destination_options)

    st.subheader("Schedule")
    date_col, departure_col, arrival_col = st.columns(3)
    with date_col:
        flight_date = st.date_input("Flight date")
    with departure_col:
        departure_time = st.time_input("Departure", value=pd.Timestamp("08:00").time())
    with arrival_col:
        arrival_time = st.time_input("Arrival", value=pd.Timestamp("11:00").time())

    distance_col, duration_col, action_col = st.columns([1, 1, 1])
    with distance_col:
        distance = st.number_input("Distance (miles)", min_value=0.0, value=1200.0, step=1.0)
    with duration_col:
        elapsed_time = st.number_input("Scheduled duration (minutes)", min_value=0.0, value=150.0, step=1.0)
    with action_col:
        st.write("")
        submitted = st.form_submit_button("Predict Flight Delay", type="primary", use_container_width=True)

if submitted:
    departure_hour = departure_time.hour
    departure_minute = departure_time.minute
    arrival_hour = arrival_time.hour
    arrival_minute = arrival_time.minute

    if distance <= 0:
        st.error("Distance must be greater than zero.")
    elif elapsed_time <= 0:
        st.error("Scheduled duration must be greater than zero.")
    elif origin == destination:
        st.error("Origin and destination must be different airports.")
    else:
        try:
            input_frame = build_inference_frame(
                carrier,
                origin,
                destination,
                flight_date,
                departure_hour,
                departure_minute,
                arrival_hour,
                arrival_minute,
                distance,
                elapsed_time,
            )
            prediction = float(model.predict(input_frame)[0])
            st.markdown(
                f"""
                <div class="result">
                    <div class="result-label">Predicted Flight Delay</div>
                    <div class="result-value">{prediction:.1f} minutes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except (KeyError, ValueError, TypeError) as exc:
            st.error(f"Unable to make a prediction from these inputs: {exc}")

with st.expander("Model features"):
    preprocessor = model.named_steps["preprocessor"]
    numerical_features = next(
        columns
        for name, _, columns in preprocessor.transformers
        if name == "numerical"
    )
    categorical_features = next(
        columns
        for name, _, columns in preprocessor.transformers
        if name == "onehot"
    )

    feature_col, categorical_col = st.columns(2)
    with feature_col:
        st.markdown("**Numerical features**")
        st.code("\n".join(str(feature) for feature in numerical_features))
    with categorical_col:
        st.markdown("**Categorical features**")
        st.code("\n".join(str(feature) for feature in categorical_features))

with st.expander("About this prediction"):
    st.write(
        "The saved preprocessing and LightGBM model are used directly. "
        "Derived date, schedule, route, cyclical, distance, and peak features "
        "reuse the project's existing feature-engineering function. Historical "
        "delay features require prior observed flights, so they are left missing "
        "and handled by the saved numerical imputer."
    )
