from app.batch_page import analyze_csv, format_duration


def test_analyze_valid_csv(flights):
    info = analyze_csv(flights(4).to_csv(index=False).encode())

    assert info["error"] is None
    assert info["rows"] == 4
    assert info["missing"] == []
    assert len(info["preview"]) == 4


def test_analyze_reports_missing_columns(flights):
    content = flights(2).drop(columns=["ORIGIN", "DEST"]).to_csv(index=False)

    info = analyze_csv(content.encode())

    assert info["missing"] == ["DEST", "ORIGIN"]


def test_analyze_empty_and_invalid_files():
    assert analyze_csv(b"")["error"]
    assert analyze_csv(b"\xff\xfe\x00\x81")["error"]


def test_format_duration():
    assert format_duration(None) == "-"
    assert format_duration(18.44) == "18.4 s"
