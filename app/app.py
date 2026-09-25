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
from batch_page import render_batch_page


# ============================================================
# API CONFIGURATION
# ============================================================

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
    :root {
        --ink: #e8f1f2;
        --muted: #8ea7ad;
        --panel: #10222d;
        --panel-strong: #152d39;
        --line: #274552;
        --aqua: #66d6cf;
        --amber: #f3b65b;
        --danger: #ff8f85;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 0%, rgba(57, 113, 122, .22), transparent 28rem),
            linear-gradient(135deg, #071219 0%, #0b1a23 48%, #08141c 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1160px;
        padding: 3.5rem 2rem 4rem;
    }

    [data-testid="stSidebar"] {
        background: #091820;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.2rem; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--ink); }
    [data-testid="stSidebar"] .stCaption { color: var(--muted); }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: .45rem;
        margin: .55rem 0 2.2rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label {
        position: relative;
        min-height: 2.65rem;
        box-sizing: border-box;
        padding: .72rem .8rem;
        border: 1px solid transparent;
        border-radius: 3px;
        color: var(--muted);
        cursor: pointer;
        transition: background .2s ease, border-color .2s ease, color .2s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        border-color: var(--line);
        color: var(--ink);
        background: rgba(39, 69, 82, .34);
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        border-color: #438d8b;
        background: linear-gradient(90deg, #1d696d, #154e57);
        color: #f4fbfa;
        box-shadow: inset 3px 0 0 var(--amber);
    }

    [data-testid="stSidebar"] [role="radiogroup"] > label p {
        color: inherit;
        font-weight: 700;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #123442, #0e252f 62%, #132b32);
        border: 1px solid #2d5b64;
        border-radius: 4px;
        color: var(--ink);
        padding: 2.7rem 2.8rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 18px 42px rgba(0, 0, 0, .18);
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -4rem;
        top: -5rem;
        width: 18rem;
        height: 18rem;
        border: 1px solid rgba(102, 214, 207, .25);
        border-radius: 50%;
        box-shadow: 0 0 0 2.2rem rgba(102, 214, 207, .05),
                    0 0 0 4.5rem rgba(102, 214, 207, .035);
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        color: #f4fbfa;
        font-family: Georgia, serif;
        font-size: 3rem;
        font-weight: 400;
        letter-spacing: 0;
        line-height: 1.05;
        margin: 0;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 42rem;
        color: #b9d1d0;
        font-family: "Trebuchet MS", sans-serif;
        font-size: 1rem;
        line-height: 1.6;
        margin: .9rem 0 0;
    }

    h2, h3 { color: var(--ink); font-family: Georgia, serif; font-weight: 400; }
    label, [data-testid="stMarkdownContainer"] p { color: #c2d2d4; }

    [data-testid="stForm"],
    [data-testid="stExpander"],
    [data-testid="stFileUploader"] section {
        background: rgba(16, 34, 45, .76);
        border: 1px solid var(--line);
        border-radius: 4px;
    }

    [data-testid="stForm"] { padding: 1.3rem 1.4rem .5rem; }
    [data-testid="stExpander"] { margin-top: 1rem; }
    [data-testid="stExpander"] summary p { color: var(--ink); font-weight: 600; }

    [data-baseweb="select"] > div,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input,
    [data-testid="stNumberInput"] input {
        background: #0b1b24;
        border-color: var(--line);
        color: var(--ink);
    }

    [data-baseweb="select"] > div:focus-within,
    [data-testid="stDateInput"] input:focus,
    [data-testid="stTimeInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--aqua);
        box-shadow: 0 0 0 1px var(--aqua);
    }

    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button {
        min-height: 2.8rem;
        border: 1px solid #438d8b;
        border-radius: 3px;
        background: #1d696d;
        color: #f4fbfa;
        font-weight: 700;
        transition: background .2s ease, border-color .2s ease, transform .2s ease;
    }

    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--aqua);
        background: #278486;
        transform: translateY(-1px);
    }

    .result {
        background: linear-gradient(120deg, #122e35, #10252d);
        border: 1px solid #3d7778;
        border-left: 5px solid var(--amber);
        border-radius: 4px;
        padding: 1.5rem 1.7rem;
        margin: 1.5rem 0;
    }

    .result-label {
        color: var(--amber);
        font-family: "Trebuchet MS", sans-serif;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .result-value {
        color: #f4fbfa;
        font-family: Georgia, serif;
        font-size: 3.2rem;
        line-height: 1.1;
        margin-top: .35rem;
    }

    [data-testid="stMetric"] {
        background: rgba(16, 34, 45, .76);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--aqua); }
    [data-testid="stProgressBar"] > div > div { background: var(--aqua); }
    [data-testid="stAlert"] { border-radius: 4px; }

    @media (max-width: 700px) {
        .block-container { padding: 2rem 1rem 3rem; }
        .hero { padding: 2rem 1.4rem; }
        [data-testid="stForm"] { padding: 1rem .8rem .3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    page = st.radio(
        "Navigation",
        ["Single Prediction", "Batch Prediction"],
    )

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


if page == "Batch Prediction":
    render_batch_page()
    st.stop()


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
