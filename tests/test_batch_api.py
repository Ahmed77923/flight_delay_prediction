import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.config import Config
from src.api import main
from src.api.model_loader import get_state
from src.batch.schemas import BatchStatus


@pytest.fixture
def client(fake_model, monkeypatch):
    """API with the fake batch model already installed in the shared state."""

    def fake_load_pipeline():
        # The fake_model fixture already populated the shared state.
        assert get_state().model is fake_model

    monkeypatch.setattr(main, "load_pipeline", fake_load_pipeline)

    with TestClient(main.app) as api_client:
        yield api_client


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def upload(client, content: bytes, filename="flights.csv", content_type="text/csv"):
    return client.post(
        "/batch/predict",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


def wait_for_terminal(client, batch_id, timeout=15):
    deadline = time.time() + timeout

    while time.time() < deadline:
        body = client.get(f"/batch/status/{batch_id}").json()

        if body["status"] in ("completed", "failed"):
            return body

        time.sleep(0.05)

    pytest.fail(f"batch {batch_id} did not finish: {body}")


# ============================================================
# POST /batch/predict
# ============================================================

def test_submit_returns_batch_id_immediately(client, flights):
    response = upload(client, csv_bytes(flights(10)))

    assert response.status_code == 202

    body = response.json()

    assert body["batch_id"].startswith("batch-")
    assert body["status"] in ("pending", "running", "completed")


def test_full_flow_submit_status_results(client, flights, batch_dirs):
    source = flights(12)

    submitted = upload(client, csv_bytes(source)).json()
    batch_id = submitted["batch_id"]

    status = wait_for_terminal(client, batch_id)

    assert status["status"] == "completed"
    assert status["batch_id"] == batch_id
    assert status["rows"] == 12
    assert status["rows_processed"] == 12
    assert status["duration_seconds"] >= 0
    assert status["input_filename"] == "flights.csv"
    assert status["output_file"] == f"{batch_id}_predictions.csv"
    assert status["prediction_column"] == "predicted_arr_delay"
    assert status["model_version"] == "test-1"
    assert status["error"] is None

    results = client.get(f"/batch/results/{batch_id}")

    assert results.status_code == 200
    assert results.headers["content-type"].startswith("text/csv")
    assert f"{batch_id}_predictions.csv" in results.headers["content-disposition"]

    frame = pd.read_csv(io.BytesIO(results.content), dtype=str)

    assert len(frame) == 12
    assert list(frame.columns) == list(source.columns) + ["predicted_arr_delay"]
    assert frame["CRS_DEP_TIME"].iloc[0] == "0630"
    assert float(frame["predicted_arr_delay"].iloc[1]) == 6.0  # 600 / 100


def test_responses_never_expose_server_paths(client, flights, batch_dirs):
    batch_id = upload(client, csv_bytes(flights(5))).json()["batch_id"]
    wait_for_terminal(client, batch_id)

    text = client.get(f"/batch/status/{batch_id}").text

    assert str(batch_dirs) not in text
    assert "model_uri" not in text
    assert "fake://model" not in text
    assert "/" not in client.get(f"/batch/status/{batch_id}").json()["output_file"]


def test_uploaded_file_name_is_not_used_as_a_path(client, flights, batch_dirs):
    response = upload(
        client, csv_bytes(flights(3)), filename="../../etc/evil name.csv"
    )

    assert response.status_code == 202

    batch_id = response.json()["batch_id"]
    status = wait_for_terminal(client, batch_id)

    assert status["input_filename"] == "evil name.csv"
    assert (batch_dirs / "input" / f"{batch_id}.csv").is_file()
    assert not (batch_dirs.parent / "etc").exists()


def test_rejects_non_csv_filename(client, flights):
    response = upload(client, csv_bytes(flights(3)), filename="flights.txt")

    assert response.status_code == 400
    assert response.json() == {
        "status": "failed",
        "error": "Only .csv files are accepted.",
    }


def test_rejects_non_csv_content_type(client, flights):
    response = upload(
        client,
        csv_bytes(flights(3)),
        content_type="application/x-msdownload",
    )

    assert response.status_code == 400


def test_rejects_missing_file(client):
    assert client.post("/batch/predict").status_code == 422


def test_rejects_empty_file(client):
    response = upload(client, b"")

    assert response.status_code == 400
    assert response.json()["status"] == "failed"
    assert "empty" in response.json()["error"]


def test_rejects_missing_columns_with_useful_error(client, flights):
    df = flights(3).drop(columns=["ORIGIN", "DEST"])

    response = upload(client, csv_bytes(df))

    assert response.status_code == 400
    assert response.json() == {
        "status": "failed",
        "error": "Missing required columns: DEST, ORIGIN",
    }


def test_rejects_binary_garbage(client):
    response = upload(client, b"\x00\x01\x02\xff\xfe\x00garbage")

    assert response.status_code == 400
    assert response.json()["status"] == "failed"


def test_rejects_oversized_upload(client, flights, monkeypatch, batch_dirs):
    monkeypatch.setattr(Config.BATCH, "MAX_UPLOAD_MB", 0)

    response = upload(client, csv_bytes(flights(3)))

    assert response.status_code == 413
    assert response.json()["status"] == "failed"
    # Nothing is left behind by a rejected upload.
    assert list((batch_dirs / "input").iterdir()) == []


def test_rejected_upload_leaves_no_files(client, flights, batch_dirs):
    upload(client, csv_bytes(flights(3).drop(columns=["ORIGIN"])))

    assert list((batch_dirs / "input").iterdir()) == []
    assert not (batch_dirs / "metadata").exists() or not list(
        (batch_dirs / "metadata").iterdir()
    )


# ============================================================
# FAILURES DURING THE JOB
# ============================================================

def test_bad_values_fail_the_batch_with_a_clean_message(client, flights):
    df = flights(5)
    df.loc[3, "DISTANCE"] = "not-a-number"

    batch_id = upload(client, csv_bytes(df)).json()["batch_id"]
    status = wait_for_terminal(client, batch_id)

    assert status["status"] == "failed"
    assert "DISTANCE" in status["error"]
    assert "Traceback" not in status["error"]
    assert status["output_file"] is None


def test_results_unavailable_for_failed_batch(client, flights):
    df = flights(5)
    df.loc[0, "DISTANCE"] = ""

    batch_id = upload(client, csv_bytes(df)).json()["batch_id"]
    wait_for_terminal(client, batch_id)

    response = client.get(f"/batch/results/{batch_id}")

    assert response.status_code == 409
    assert response.json()["status"] == "failed"


# ============================================================
# GET /batch/status, /batch/results
# ============================================================

def test_status_unknown_batch_is_404(client):
    response = client.get("/batch/status/batch-19990101-001")

    assert response.status_code == 404
    assert response.json() == {"error": "Batch not found."}


@pytest.mark.parametrize(
    "bad_id",
    ["..", "%2e%2e%2fsecret", "..%2F..%2Fetc%2Fpasswd", "a%00b", "x" * 200],
)
def test_status_and_results_reject_traversal_ids(client, bad_id):
    assert client.get(f"/batch/status/{bad_id}").status_code == 404
    assert client.get(f"/batch/results/{bad_id}").status_code == 404


def test_results_unknown_batch_is_404(client):
    assert client.get("/batch/results/nope").status_code == 404


def test_results_not_ready_is_409(client, store):
    store.create(batch_id="pending-one", source="api")

    response = client.get("/batch/results/pending-one")

    assert response.status_code == 409
    assert response.json()["status"] == "pending"


def test_results_only_serve_files_inside_output_dir(client, store, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("top,secret\n1,2\n")

    store.create(batch_id="sneaky", output_file=secret)
    store.update("sneaky", status=BatchStatus.COMPLETED)

    response = client.get("/batch/results/sneaky")

    assert response.status_code == 404
    assert "secret" not in response.text


def test_results_missing_output_file_is_404(client, store):
    store.create(batch_id="gone", source="api")
    store.update("gone", status=BatchStatus.COMPLETED)

    assert client.get("/batch/results/gone").status_code == 404


# ============================================================
# STARTUP RECOVERY, METRICS, ROOT
# ============================================================

def test_startup_fails_batches_interrupted_by_restart(
    fake_model, store, monkeypatch
):
    store.create(batch_id="orphan", source="api")
    store.update("orphan", status=BatchStatus.RUNNING)

    monkeypatch.setattr(main, "load_pipeline", lambda: None)

    with TestClient(main.app) as api_client:
        body = api_client.get("/batch/status/orphan").json()

    assert body["status"] == "failed"
    assert "restarted" in body["error"]


def test_batch_metrics_are_exposed(client, flights):
    batch_id = upload(client, csv_bytes(flights(4))).json()["batch_id"]
    wait_for_terminal(client, batch_id)

    text = client.get("/metrics").text

    for name in (
        "batch_jobs_total",
        "batch_jobs_success_total",
        "batch_jobs_failed_total",
        "batch_rows_processed_total",
        "batch_duration_seconds_count",
    ):
        assert name in text

    # Existing API metrics are still there.
    assert "http_requests_total" in text


def test_root_lists_batch_endpoints(client):
    endpoints = client.get("/").json()["endpoints"]

    assert "/batch/predict" in endpoints
    assert "/predict" in endpoints
