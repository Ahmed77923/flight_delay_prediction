import time

import pytest
from prometheus_client import REGISTRY

from src.batch import runner
from src.batch.errors import BatchModelError
from src.batch.runner import create_batch, run_batch, submit_batch
from src.batch.schemas import BatchStatus
from src.batch.store import BatchStore, is_valid_batch_id


def metric(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


# ============================================================
# METADATA STORE
# ============================================================

def test_auto_ids_are_sequential_and_unique(store):
    first = store.create()
    second = store.create()

    assert first.batch_id != second.batch_id
    assert first.batch_id.startswith("batch-")
    assert first.batch_id.endswith("-001")
    assert second.batch_id.endswith("-002")


def test_new_batch_is_pending_with_default_paths(store):
    meta = store.create(batch_id="abc")

    assert meta.status == BatchStatus.PENDING
    assert meta.input_file.endswith("input/abc.csv")
    assert meta.output_file.endswith("output/abc_predictions.csv")
    assert store.metadata_path("abc").is_file()


def test_duplicate_batch_id_is_rejected(store):
    store.create(batch_id="september-2026")

    with pytest.raises(Exception, match="already exists"):
        store.create(batch_id="september-2026")


@pytest.mark.parametrize(
    "bad_id",
    ["../evil", "a/b", "a\\b", "", ".hidden", "-x", "x" * 65, "a b", "a\x00b"],
)
def test_invalid_batch_ids_are_rejected(store, bad_id):
    assert not is_valid_batch_id(bad_id)

    with pytest.raises(Exception, match="Invalid batch id"):
        store.create(batch_id=bad_id)

    assert store.get(bad_id) is None


def test_update_persists_and_get_returns_it(store):
    store.create(batch_id="b1")

    store.update("b1", status=BatchStatus.RUNNING, rows=12)

    meta = store.get("b1")
    assert meta.status == BatchStatus.RUNNING
    assert meta.rows == 12


def test_get_unknown_returns_none(store):
    assert store.get("missing") is None


def test_fail_interrupted_only_touches_api_jobs(store):
    store.create(batch_id="api-run", source="api")
    store.update("api-run", status=BatchStatus.RUNNING)
    store.create(batch_id="cli-run", source="cli")
    store.update("cli-run", status=BatchStatus.RUNNING)
    store.create(batch_id="api-done", source="api")
    store.update("api-done", status=BatchStatus.COMPLETED)

    assert store.fail_interrupted() == 1

    assert store.get("api-run").status == BatchStatus.FAILED
    assert "restarted" in store.get("api-run").error
    assert store.get("cli-run").status == BatchStatus.RUNNING
    assert store.get("api-done").status == BatchStatus.COMPLETED


# ============================================================
# RUN BATCH
# ============================================================

def test_run_batch_completes_and_records_metadata(
    store, flights, write_csv, fake_model, tmp_path
):
    meta = create_batch(
        store,
        input_path=write_csv(flights(10)),
        output_path=tmp_path / "result.csv",
        batch_id="run-ok",
    )
    jobs = metric("batch_jobs_total")
    ok = metric("batch_jobs_success_total")
    rows = metric("batch_rows_processed_total")

    done = run_batch(store, meta.batch_id)

    assert done.status == BatchStatus.COMPLETED
    assert done.rows == 10 and done.rows_processed == 10
    assert done.prediction_column == "predicted_arr_delay"
    assert done.model_uri == "fake://model"
    assert done.model_version == "test-1"
    assert done.started_at and done.completed_at
    assert done.duration_seconds is not None and done.duration_seconds >= 0
    assert done.error is None
    assert (tmp_path / "result.csv").is_file()

    assert store.get("run-ok") == done

    assert metric("batch_jobs_total") == jobs + 1
    assert metric("batch_jobs_success_total") == ok + 1
    assert metric("batch_rows_processed_total") == rows + 10


def test_run_batch_records_input_failure(
    store, flights, write_csv, fake_model, tmp_path
):
    df = flights(4).drop(columns=["ORIGIN", "DEST"])
    meta = create_batch(
        store,
        input_path=write_csv(df),
        output_path=tmp_path / "never.csv",
        batch_id="run-bad",
    )
    failed = metric("batch_jobs_failed_total")

    done = run_batch(store, meta.batch_id)

    assert done.status == BatchStatus.FAILED
    assert done.error == "Missing required columns: DEST, ORIGIN"
    assert not (tmp_path / "never.csv").exists()
    assert metric("batch_jobs_failed_total") == failed + 1


def test_run_batch_hides_internal_errors(
    store, flights, write_csv, fake_model, tmp_path, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("secret /internal/path detail")

    monkeypatch.setattr(runner, "predict_file", explode)

    meta = create_batch(
        store,
        input_path=write_csv(flights(3)),
        output_path=tmp_path / "o.csv",
        batch_id="run-boom",
    )

    done = run_batch(store, meta.batch_id)

    assert done.status == BatchStatus.FAILED
    assert "secret" not in done.error
    assert "path" not in done.error


def test_run_batch_reports_model_load_failure(
    store, flights, write_csv, tmp_path, monkeypatch
):
    def broken():
        raise BatchModelError("The prediction model could not be loaded.")

    monkeypatch.setattr(runner, "load_model", broken)

    meta = create_batch(
        store,
        input_path=write_csv(flights(3)),
        output_path=tmp_path / "o.csv",
        batch_id="run-nomodel",
    )

    done = run_batch(store, meta.batch_id)

    assert done.status == BatchStatus.FAILED
    assert done.error == "The prediction model could not be loaded."


def test_run_batch_reports_prediction_failure(
    store, flights, write_csv, fake_model, tmp_path, monkeypatch
):
    def broken(frame):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(fake_model, "predict", broken)

    meta = create_batch(
        store,
        input_path=write_csv(flights(3)),
        output_path=tmp_path / "o.csv",
        batch_id="run-predfail",
    )

    done = run_batch(store, meta.batch_id)

    assert done.status == BatchStatus.FAILED
    assert done.error == "Prediction failed."
    assert not (tmp_path / "o.csv").exists()


def test_run_batch_reports_unwritable_output(
    store, flights, write_csv, fake_model, tmp_path
):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")

    meta = create_batch(
        store,
        input_path=write_csv(flights(3)),
        output_path=blocker / "o.csv",
        batch_id="run-nowrite",
    )

    done = run_batch(store, meta.batch_id)

    assert done.status == BatchStatus.FAILED
    assert "could not be saved" in done.error or "not writable" in done.error


def test_create_batch_rejects_same_input_and_output(store, write_csv, flights):
    path = write_csv(flights(3))

    with pytest.raises(Exception, match="different from the input"):
        create_batch(store, input_path=path, output_path=path)


# ============================================================
# BACKGROUND EXECUTION
# ============================================================

def test_submit_batch_runs_in_background(
    store, flights, write_csv, fake_model, tmp_path
):
    meta = create_batch(
        store,
        input_path=write_csv(flights(10)),
        output_path=tmp_path / "bg.csv",
        batch_id="bg",
    )

    future = submit_batch(store, meta.batch_id)
    future.result(timeout=15)

    assert store.get("bg").status == BatchStatus.COMPLETED

    runner.shutdown_executor()
