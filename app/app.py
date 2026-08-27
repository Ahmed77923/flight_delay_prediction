from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config


# ============================================================
# API CONFIGURATION
#
# The Streamlit UI is a thin HTTP client. It never loads the MLflow
# model and never runs feature engineering or inference itself -
# FastAPI (src/api/main.py) remains the only component responsible
# for that. Configure the API's base URL via the API_URL environment
# variable (see config/config.py -> Config.API.BASE_URL); it defaults
# to http://127.0.0.1:8000.
# ============================================================

API_URL = Config.API.BASE_URL.rstrip("/")
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"
MODEL_INFO_ENDPOINT = f"{API_URL}/model"

REQUEST_TIMEOUT_SECONDS = 10
HEALTH_CHECK_TIMEOUT_SECONDS = 3
HEALTH_CACHE_TTL_SECONDS = 15

# Static option lists for the carrier / airport selectors. These used to
# be read from the fitted OneHotEncoder inside the locally loaded model;
# since the UI no longer loads the model, they are now a fixed reference
# list of common IATA codes. The API accepts any string for these fields
# and applies OneHotEncoder(handle_unknown="ignore") server-side, so an
# unlisted code is still handled gracefully rather than crashing.
CARRIER_OPTIONS = [
    "AA", "AS", "B6", "DL", "F9", "G4", "HA", "NK", "UA", "WN",
]

AIRPORT_OPTIONS = [
    "ATL", "AUS", "BNA", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW",
    "EWR", "FLL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO", "MDW",
    "MIA", "MSP", "ORD", "PHL", "PHX", "SAN", "SEA", "SFO", "SLC", "TPA",
]

# ============================================================
# API CLIENT
# ============================================================

class PredictionAPIError(Exception):
    """Raised when the FastAPI prediction service can't fulfil a request."""


def _extract_detail(response: requests.Response) -> str:
    """
    Turn FastAPI's error body into a short, readable string instead of
    dumping the raw JSON/traceback-shaped structure at the user.
    """

    try:
        body = response.json()
    except ValueError:
        return ""

    detail = body.get("detail") if isinstance(body, dict) else None

    if isinstance(detail, str):
        return detail

    if isinstance(detail, list):
        # FastAPI/pydantic validation error shape:
        # [{"loc": ["body", "ORIGIN"], "msg": "Field required", ...}, ...]
        messages = []
        for error in detail:
            if not isinstance(error, dict):
                continue
            location = error.get("loc", [])
            field = str(location[-1]) if location else "input"
            message = error.get("msg", "invalid value")
            messages.append(f"{field}: {message}")
        return "; ".join(messages)

    if isinstance(detail, dict):
        error_text = detail.get("error")
        if isinstance(error_text, str):
            return error_text

    return ""


def call_predict_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a flight request to FastAPI's POST /predict and return the
    parsed JSON response. Raises PredictionAPIError with a
    user-friendly message on any failure - never lets a raw exception
    or traceback reach the UI.
    """

    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise PredictionAPIError(
            f"Cannot connect to prediction API at {API_URL}. "
            "Make sure the FastAPI service is running."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise PredictionAPIError(
            "The prediction service took too long to respond. Please try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise PredictionAPIError(
            "Cannot reach the prediction API."
        ) from exc

    if response.status_code == 422:
        detail = _extract_detail(response)
        raise PredictionAPIError(
            "Invalid flight data." + (f" {detail}" if detail else "")
        )

    if 400 <= response.status_code < 500:
        detail = _extract_detail(response)
        raise PredictionAPIError(
            "The prediction API rejected the request."
            + (f" {detail}" if detail else "")
        )

    if response.status_code >= 500:
        raise PredictionAPIError(
            "Prediction service returned an error. Please try again later."
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise PredictionAPIError(
            "Received an invalid response from the prediction service."
        ) from exc

    if not isinstance(data, dict) or "prediction" not in data:
        raise PredictionAPIError(
            "Received an unexpected response from the prediction service."
        )

    return data


@st.cache_data(ttl=HEALTH_CACHE_TTL_SECONDS, show_spinner=False)
def check_api_health() -> Dict[str, Any]:
    """
    Lightweight, cached GET /health check used only to render the
    sidebar status - cached so it isn't re-fetched on every rerun.
    """

    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "unreachable", "model_loaded": False}


@st.cache_data(ttl=HEALTH_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_model_info() -> Optional[Dict[str, Any]]:
    """Cached GET /model, used only to populate the informational expander."""

    try:
        response = requests.get(
            MODEL_INFO_ENDPOINT,
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


# ============================================================
# REQUEST PAYLOAD
# ============================================================

def build_predict_payload(
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
) -> Dict[str, Any]:
    """
    Map the Streamlit form fields onto the exact FastAPI FlightRequest
    field names. No feature engineering happens here - that is entirely
    FastAPI's / build_features()'s responsibility.
    """

    return {
        "FL_DATE": datetime.combine(
            flight_date,
            datetime.min.time(),
        ).isoformat(),
        "CRS_DEP_TIME": departure_hour * 100 + departure_minute,
        "CRS_ARR_TIME": arrival_hour * 100 + arrival_minute,
        "CRS_ELAPSED_TIME": elapsed_time,
        "DISTANCE": distance,
        "OP_UNIQUE_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": destination,
    }


# ============================================================
# PAGE
# ============================================================

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


with st.sidebar:
    st.subheader("Prediction API")

    health = check_api_health()

    if health.get("status") == "healthy" and health.get("model_loaded"):
        st.success("API Connected")
        st.write(f"**Model type:** {health.get('model_type', 'unknown')}")
        st.write("**Status:** Ready for Prediction")
    else:
        st.error("API unavailable")
        st.write(f"Expected URL: `{API_URL}`")
        st.caption(
            "Start FastAPI with `uvicorn src.api.main:app` "
            "and reload this page."
        )

    st.caption(f"Endpoint: `{PREDICT_ENDPOINT}`")


st.markdown(
    """
    <div class="hero">
        <h1>Flight Delay Prediction</h1>
        <p>Predict flight delay using the trained model served by the FastAPI prediction API.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Flight Information")
with st.form("prediction_form"):
    carrier_col, origin_col, destination_col = st.columns(3)
    with carrier_col:
        carrier = st.selectbox("Carrier", CARRIER_OPTIONS)
    with origin_col:
        origin = st.selectbox("Origin", AIRPORT_OPTIONS)
    with destination_col:
        destination = st.selectbox("Destination", AIRPORT_OPTIONS, index=1)

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
        payload = build_predict_payload(
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

        try:
            with st.spinner("Contacting prediction API..."):
                result = call_predict_api(payload)

            prediction = float(result["prediction"])

            st.markdown(
                f"""
                <div class="result">
                    <div class="result-label">Predicted Arrival Delay</div>
                    <div class="result-value">{prediction:.2f} minutes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if prediction <= 0:
                st.caption("A negative value means the flight is predicted to arrive early.")

        except PredictionAPIError as exc:
            st.error(str(exc))

with st.expander("Model features"):
    model_info = fetch_model_info()

    if model_info is None:
        st.info("Model information is unavailable - the prediction API could not be reached.")
    else:
        expected_columns = model_info.get("expected_columns", [])
        categorical = [c for c in expected_columns if c in {
            "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "route",
            "departure_period", "carrier_origin",
        }]
        numerical = [c for c in expected_columns if c not in categorical]

        feature_col, categorical_col = st.columns(2)
        with feature_col:
            st.markdown("**Numerical features**")
            st.code("\n".join(str(f) for f in numerical) or "unavailable")
        with categorical_col:
            st.markdown("**Categorical features**")
            st.code("\n".join(str(f) for f in categorical) or "unavailable")

        st.caption(
            f"Model type: {model_info.get('model_type', 'unknown')} | "
            f"Historical features used: {model_info.get('historical_features_used', 'unknown')}"
        )

with st.expander("About this prediction"):
    st.write(
        "This page is a thin client: it sends your flight details as JSON to the "
        f"FastAPI prediction service (`POST {PREDICT_ENDPOINT}`), which builds the "
        "model's input features and runs the trained MLflow pipeline. No model is "
        "loaded and no prediction happens inside this Streamlit app. "
        "The model only uses information available at request time - it does not "
        "look up historical or prior-flight delay statistics."
    )
