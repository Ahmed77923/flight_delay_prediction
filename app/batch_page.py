"""
Batch Prediction page of the Streamlit dashboard.

Like the rest of the UI this is a thin HTTP client: the CSV is uploaded to
FastAPI (POST /batch/predict), progress is polled from
GET /batch/status/{id} and the result is downloaded through
GET /batch/results/{id}. Nothing is predicted inside Streamlit, and no
filesystem is shared with the API container.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from config.config import Config
from src.batch.schemas import REQUIRED_COLUMNS


API_URL = Config.API.BASE_URL.rstrip("/")
BATCH_SUBMIT_ENDPOINT = f"{API_URL}/batch/predict"
BATCH_STATUS_ENDPOINT = f"{API_URL}/batch/status"
BATCH_RESULTS_ENDPOINT = f"{API_URL}/batch/results"

UPLOAD_TIMEOUT_SECONDS = 120
STATUS_TIMEOUT_SECONDS = 5
DOWNLOAD_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 2

TERMINAL_STATUSES = {"completed", "failed"}

STATE_BATCH_ID = "batch_id"
STATE_STATUS = "batch_status"
STATE_RESULTS = "batch_results"


# ============================================================
# API CLIENT
# ============================================================

class BatchAPIError(Exception):
    """A batch API call failed; the message is safe to show to the user."""


def _api_error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""

    if isinstance(body, dict):
        for key in ("error", "detail"):
            if isinstance(body.get(key), str):
                return body[key]

    return ""


def _request(method: str, url: str, timeout: int, **kwargs) -> requests.Response:
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise BatchAPIError(
            f"Cannot connect to the prediction API at {API_URL}. "
            "Make sure the FastAPI service is running."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BatchAPIError(
            "The prediction service took too long to respond."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise BatchAPIError("Cannot reach the prediction API.") from exc

    if response.status_code >= 500:
        raise BatchAPIError(
            _api_error_message(response)
            or "The prediction service returned an error. Please try again later."
        )

    if response.status_code >= 400:
        raise BatchAPIError(
            _api_error_message(response)
            or "The prediction API rejected the request."
        )

    return response


def submit_batch(filename: str, content: bytes) -> Dict[str, Any]:
    response = _request(
        "POST",
        BATCH_SUBMIT_ENDPOINT,
        UPLOAD_TIMEOUT_SECONDS,
        files={"file": (filename, content, "text/csv")},
    )

    try:
        data = response.json()
    except ValueError as exc:
        raise BatchAPIError(
            "Received an invalid response from the prediction service."
        ) from exc

    if not isinstance(data, dict) or "batch_id" not in data:
        raise BatchAPIError(
            "Received an unexpected response from the prediction service."
        )

    return data


def fetch_batch_status(batch_id: str) -> Dict[str, Any]:
    response = _request(
        "GET",
        f"{BATCH_STATUS_ENDPOINT}/{batch_id}",
        STATUS_TIMEOUT_SECONDS,
    )

    try:
        return response.json()
    except ValueError as exc:
        raise BatchAPIError(
            "Received an invalid response from the prediction service."
        ) from exc


def fetch_batch_results(batch_id: str) -> bytes:
    return _request(
        "GET",
        f"{BATCH_RESULTS_ENDPOINT}/{batch_id}",
        DOWNLOAD_TIMEOUT_SECONDS,
    ).content


# ============================================================
# DATASET INSPECTION
# ============================================================

@st.cache_data(show_spinner=False)
def analyze_csv(content: bytes) -> Dict[str, Any]:
    """
    Client-side pre-check so obvious problems show up before uploading.
    The API repeats and extends this validation.
    """

    read_options = {
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8-sig",
    }

    try:
        # Header + a few rows for the preview, then a chunked row count:
        # the whole file is never held in memory (uploads can be ~2M rows).
        preview = pd.read_csv(io.BytesIO(content), nrows=5, **read_options)
        rows = sum(
            len(chunk)
            for chunk in pd.read_csv(
                io.BytesIO(content),
                usecols=[0],
                chunksize=200_000,
                **read_options,
            )
        )
    except pd.errors.EmptyDataError:
        return {"error": "The file is empty."}
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        return {"error": "The file is not a valid UTF-8 CSV."}

    missing: List[str] = [
        column for column in REQUIRED_COLUMNS if column not in preview.columns
    ]

    return {
        "error": None,
        "rows": rows,
        "columns": len(preview.columns),
        "missing": missing,
        "preview": preview,
    }


def format_duration(seconds: Optional[float]) -> str:
    return "-" if seconds is None else f"{seconds:,.1f} s"


# ============================================================
# STATUS PANEL
# ============================================================

def _reset_batch_state() -> None:
    for key in (STATE_BATCH_ID, STATE_STATUS, STATE_RESULTS):
        st.session_state.pop(key, None)


def _render_status(status: Dict[str, Any]) -> None:
    state = status.get("status", "unknown")
    rows = status.get("rows")
    processed = status.get("rows_processed", 0)

    st.markdown(f"**Batch ID:** `{status.get('batch_id', '-')}`")

    status_col, rows_col, time_col = st.columns(3)
    status_col.metric("Status", str(state).upper())
    # While running, the progress bar below shows "x of total".
    rows_text = f"{rows if state == 'completed' else processed:,}"

    rows_col.metric("Rows processed", rows_text)
    time_col.metric(
        "Execution time", format_duration(status.get("duration_seconds"))
    )

    if state in ("pending", "running") and rows:
        st.progress(
            min(processed / rows, 1.0),
            text=f"{processed:,} of {rows:,} rows",
        )
    elif state == "pending":
        st.info("Waiting for a worker to start this batch...")
    elif state == "running":
        st.info("Validating the file and loading the model...")


def _render_result(batch_id: str, status: Dict[str, Any]) -> None:
    state = status.get("status")

    if state == "failed":
        st.error(
            "Batch failed: "
            + (status.get("error") or "an unexpected error occurred.")
        )
        return

    if state != "completed":
        return

    st.success("✓ Batch completed")
    st.write(
        f"Rows processed: **{status.get('rows', 0):,}** · "
        f"Duration: **{format_duration(status.get('duration_seconds'))}**"
    )

    if STATE_RESULTS not in st.session_state:
        try:
            with st.spinner("Preparing download..."):
                st.session_state[STATE_RESULTS] = fetch_batch_results(batch_id)
        except BatchAPIError as exc:
            st.error(f"Could not retrieve the predictions. {exc}")
            return

    st.download_button(
        "Download Predictions",
        data=st.session_state[STATE_RESULTS],
        file_name=f"{batch_id}_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def _status_panel(batch_id: str) -> None:
    """
    Polls the API every few seconds until the batch reaches a final state.
    Only this fragment reruns while polling, so the upload widget and
    the rest of the page are left alone.
    """

    known = st.session_state.get(STATE_STATUS) or {}
    was_active = known.get("status") not in TERMINAL_STATUSES

    @st.fragment(run_every=POLL_INTERVAL_SECONDS if was_active else None)
    def panel() -> None:
        try:
            status = fetch_batch_status(batch_id)
            st.session_state[STATE_STATUS] = status
        except BatchAPIError as exc:
            st.warning(f"Could not refresh the batch status. {exc}")
            status = st.session_state.get(STATE_STATUS)

        if not status:
            return

        _render_status(status)
        _render_result(batch_id, status)

        if was_active and status.get("status") in TERMINAL_STATUSES:
            # Full rerun so the fragment is re-created without polling.
            st.rerun()

    panel()


# ============================================================
# PAGE
# ============================================================

def render_batch_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Batch Prediction</h1>
            <p>Upload a CSV of flights and download the predicted arrival delay for every row.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Upload flight data")
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Required columns: " + ", ".join(REQUIRED_COLUMNS),
    )

    dataset_valid = False
    content = b""

    if uploaded is not None:
        content = uploaded.getvalue()
        info = analyze_csv(content)

        st.markdown("**Dataset**")
        st.write(f"File: `{uploaded.name}`")

        if info["error"]:
            st.error(info["error"])
        else:
            st.write(f"Rows: **{info['rows']:,}**")

            if info["rows"] == 0:
                st.error("The file contains no data rows.")
            elif info["missing"]:
                st.error(
                    "Missing required columns: " + ", ".join(info["missing"])
                )
            else:
                dataset_valid = True
                st.success("Dataset looks valid.")

            with st.expander("Preview"):
                st.dataframe(info["preview"], width="stretch")
    else:
        st.caption(
            "Required columns: " + ", ".join(REQUIRED_COLUMNS) + ". "
            "All other columns are kept in the output."
        )

    known = st.session_state.get(STATE_STATUS) or {}
    running = (
        STATE_BATCH_ID in st.session_state
        and known.get("status") not in TERMINAL_STATUSES
    )

    if st.button(
        "Run Batch Prediction",
        type="primary",
        disabled=not dataset_valid or running,
    ):
        _reset_batch_state()

        try:
            with st.spinner("Uploading..."):
                result = submit_batch(uploaded.name, content)
            st.session_state[STATE_BATCH_ID] = result["batch_id"]
            st.session_state[STATE_STATUS] = {
                "batch_id": result["batch_id"],
                "status": result.get("status", "pending"),
                "rows": info["rows"],
            }
            # Re-render the whole page so the button is disabled while
            # the batch runs.
            st.rerun()
        except BatchAPIError as exc:
            st.error(str(exc))

    batch_id = st.session_state.get(STATE_BATCH_ID)

    if batch_id:
        st.divider()
        st.subheader("Batch Status")
        _status_panel(batch_id)

        # Analysis needs the finished predictions, which the status panel
        # has already fetched for the download button.
        results = st.session_state.get(STATE_RESULTS)
        finished = (st.session_state.get(STATE_STATUS) or {}).get("status")

        if finished == "completed" and results:
            # Imported here: batch_analysis is a sibling module that
            # Streamlit resolves from the script directory.
            from batch_analysis import render_batch_analysis

            render_batch_analysis(batch_id, results)
