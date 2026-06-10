from __future__ import annotations

from fastapi.testclient import TestClient

from aitrade.api import alpha as alpha_api
from aitrade.main import create_app


CSV_BYTES = b"""trade_date,symbol,open,high,low,close,volume
2024-01-02,000001,10,11,9,10.5,1000
2024-01-03,000001,10.5,11.2,10.2,10.9,1200
"""

TICK_CSV_BYTES = b"""datetime,vt_symbol,last_price,volume,turnover
2024-01-02 09:30:01,000001.SZSE,10.00,100,1000
2024-01-02 09:30:30,000001.SZSE,10.10,120,1212
2024-01-02 09:31:05,000001.SZSE,10.20,80,816
"""

MINUTE_BAR_CSV_BYTES = b"""datetime,symbol,open,high,low,close,volume
2024-01-02 09:30:00,000001,10,11,9,10.5,1000
2024-01-02 09:31:00,000001,10.5,11.2,10.2,10.9,1200
2024-01-02 09:32:00,000001,10.9,11.5,10.8,11.2,1100
2024-01-02 09:33:00,000001,11.2,11.6,11.0,11.4,900
"""

CSV_BATCH_A = b"""trade_date,vt_symbol,open,high,low,close,volume
2024-01-02,000001.SZSE,10,11,9,10.5,1000
2024-01-03,000001.SZSE,10.5,11.2,10.2,10.9,1200
2024-01-04,000001.SZSE,10.9,11.5,10.8,11.2,1100
"""

# 与 A 在重叠区（01-03 / 01-04）数据完全一致，向后扩展 01-05；用于验证一致重叠可合并。
CSV_BATCH_B = b"""trade_date,vt_symbol,open,high,low,close,volume
2024-01-03,000001.SZSE,10.5,11.2,10.2,10.9,1200
2024-01-04,000001.SZSE,10.9,11.5,10.8,11.2,1100
2024-01-05,000001.SZSE,11.3,11.8,11.1,11.5,900
"""

# 与 A 在重叠区（01-04）数据不一致，用于验证「重叠不一致拒绝合并」。
CSV_BATCH_B_CONFLICT = b"""trade_date,vt_symbol,open,high,low,close,volume
2024-01-03,000001.SZSE,10.5,11.2,10.2,10.9,1200
2024-01-04,000001.SZSE,11.0,11.6,10.9,11.3,1300
2024-01-05,000001.SZSE,11.3,11.8,11.1,11.5,900
"""

CSV_BATCH_NO_OVERLAP = b"""trade_date,vt_symbol,open,high,low,close,volume
2024-02-01,000001.SZSE,12,12.2,11.8,12.1,1000
2024-02-02,000001.SZSE,12.1,12.5,12.0,12.4,1200
"""

CSV_BATCH_OTHER_SYMBOL = b"""trade_date,vt_symbol,open,high,low,close,volume
2024-01-03,000002.SZSE,10.5,11.2,10.2,10.9,1200
2024-01-04,000002.SZSE,10.9,11.5,10.8,11.2,1100
"""

MINUTE_BATCH_GAP = b"""datetime,vt_symbol,open,high,low,close,volume
2024-01-02 09:30:00,000001.SZSE,10,11,9,10.5,1000
2024-01-02 09:32:00,000001.SZSE,10.5,11.2,10.2,10.9,1200
"""


def _upload_bar_batch(client: TestClient, csv_bytes: bytes, *, interval: str = "d", filename: str = "bars.csv") -> dict:
    response = client.post(
        "/api/alpha/bar-data/import",
        data={"interval": interval, "import_mode": "merge", "save_mode": "batch"},
        files={"file": (filename, csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["batches"]
    return payload["batches"][0]


def test_csv_preview_and_import_endpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        preview_response = client.post(
            "/api/alpha/bar-data/import/preview",
            files={"file": ("bars.csv", CSV_BYTES, "text/csv")},
        )
        assert preview_response.status_code == 200
        preview_payload = preview_response.json()

        assert preview_payload["total_rows"] == 2
        assert preview_payload["missing_required"] == []
        assert preview_payload["date_range"] == ["2024-01-02", "2024-01-03"]
        assert preview_payload["symbols"] == ["000001.SZSE"]

        import_response = client.post(
            "/api/alpha/bar-data/import",
            data={"interval": "d", "import_mode": "merge"},
            files={"file": ("bars.csv", CSV_BYTES, "text/csv")},
        )
        assert import_response.status_code == 200
        import_payload = import_response.json()

        assert import_payload["success"] is True
        assert import_payload["imported_count"] == 2
        assert import_payload["errors"] == []

        bar_list_response = client.get("/api/alpha/bar-data")
        assert bar_list_response.status_code == 200
        bar_list_payload = bar_list_response.json()

        assert bar_list_payload["daily"] == []

        resource_response = client.get("/api/alpha/data/resources")
        assert resource_response.status_code == 200
        resource_payload = resource_response.json()
        assert resource_payload["raw_bars"] == []
        assert len(resource_payload["raw_bar_batches"]) == 1
        assert resource_payload["raw_bar_batches"][0]["vt_symbol"] == "000001.SZSE"
        assert resource_payload["raw_bar_batches"][0]["interval"] == "d"


def test_csv_batch_upload_keeps_duplicate_contract_uploads_visible_without_official_write(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        first = _upload_bar_batch(client, CSV_BATCH_A, filename="first.csv")
        second = _upload_bar_batch(client, CSV_BATCH_B, filename="second.csv")

        assert first["key"] != second["key"]

        resource_response = client.get("/api/alpha/data/resources")
        assert resource_response.status_code == 200
        payload = resource_response.json()

        assert payload["raw_bars"] == []
        assert len(payload["raw_bar_batches"]) == 2
        assert {item["file_name"] for item in payload["raw_bar_batches"]} == {"first.csv", "second.csv"}
        assert {item["status"] for item in payload["raw_bar_batches"]} == {"pending"}


def test_csv_batch_merge_preview_and_merge_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        first = _upload_bar_batch(client, CSV_BATCH_A, filename="first.csv")
        second = _upload_bar_batch(client, CSV_BATCH_B, filename="second.csv")
        request = {"kind": "raw_bar", "keys": [first["key"], second["key"]]}

        preview_response = client.post("/api/alpha/data/resources/merge/preview", json=request)
        assert preview_response.status_code == 200
        preview = preview_response.json()
        assert preview["can_merge"] is True
        assert preview["intersection_start"].startswith("2024-01-03")
        assert preview["intersection_end"].startswith("2024-01-04")
        # 重叠区一致 -> 无冲突。
        assert preview["conflict_count"] == 0
        assert preview["estimated_rows"] == 4

        merge_response = client.post("/api/alpha/data/resources/merge", json=request)
        assert merge_response.status_code == 200
        merged = merge_response.json()
        assert merged["success"] is True
        assert merged["row_count"] == 4

        resource_payload = client.get("/api/alpha/data/resources").json()
        assert len(resource_payload["raw_bars"]) == 1
        assert resource_payload["raw_bars"][0]["row_count"] == 4
        assert {item["status"] for item in resource_payload["raw_bar_batches"]} == {"merged"}

        detail = client.get("/api/alpha/data/resources/raw_bar/d__000001.SZSE", params={"limit": 10}).json()
        row_20240104 = [row for row in detail["preview"] if row["datetime"].startswith("2024-01-04")][0]
        # 一致重叠：01-04 收盘取两批次一致值 11.2。
        assert row_20240104["close"] == 11.2


def test_csv_batch_merge_rejects_no_overlap_and_different_symbol(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        first = _upload_bar_batch(client, CSV_BATCH_A)
        no_overlap = _upload_bar_batch(client, CSV_BATCH_NO_OVERLAP, filename="later.csv")
        other_symbol = _upload_bar_batch(client, CSV_BATCH_OTHER_SYMBOL, filename="other.csv")

        no_overlap_response = client.post(
            "/api/alpha/data/resources/merge/preview",
            json={"kind": "raw_bar", "keys": [first["key"], no_overlap["key"]]},
        )
        assert no_overlap_response.status_code == 200
        assert no_overlap_response.json()["can_merge"] is False
        assert "数据无重叠" in no_overlap_response.json()["reason"]

        other_symbol_response = client.post(
            "/api/alpha/data/resources/merge/preview",
            json={"kind": "raw_bar", "keys": [first["key"], other_symbol["key"]]},
        )
        assert other_symbol_response.status_code == 200
        assert other_symbol_response.json()["can_merge"] is False
        assert "同一合约" in other_symbol_response.json()["reason"]


def test_csv_batch_merge_rejects_minute_gap(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        good = _upload_bar_batch(client, MINUTE_BAR_CSV_BYTES, interval="1m")
        gap = _upload_bar_batch(client, MINUTE_BATCH_GAP, interval="1m", filename="gap.csv")

        response = client.post(
            "/api/alpha/data/resources/merge/preview",
            json={"kind": "raw_bar", "keys": [good["key"], gap["key"]]},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["can_merge"] is False
        assert any("断档" in error for error in payload["errors"])


def test_relocate_raw_bar_interval(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        import_response = client.post(
            "/api/alpha/bar-data/import",
            data={"interval": "d", "import_mode": "merge", "save_mode": "official"},
            files={"file": ("minute_bars.csv", MINUTE_BAR_CSV_BYTES, "text/csv")},
        )
        assert import_response.status_code == 200

        detail_response = client.get("/api/alpha/data/resources/raw_bar/d__000001.SZSE")
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert detail_payload["interval"] == "d"

        relocate_response = client.patch(
            "/api/alpha/data/resources/raw_bar/d__000001.SZSE/interval",
            json={"interval": "1m"},
        )
        assert relocate_response.status_code == 200
        relocate_payload = relocate_response.json()
        assert relocate_payload["success"] is True
        assert relocate_payload["key"] == "1m__000001.SZSE"
        assert relocate_payload["interval"] == "1m"

        resource_response = client.get("/api/alpha/data/resources")
        intervals = {item["interval"] for item in resource_response.json()["raw_bars"]}
        assert intervals == {"1m"}


BAR_CSV_MISSING_CLOSE = b"""trade_date,symbol,open,high,low,volume
2024-01-02,000001,10,11,9,1000
"""


def test_bar_import_missing_required_field_returns_400(monkeypatch, tmp_path) -> None:
    """缺少必填字段（close）时服务端直接返回 400，不依赖前端拦截。"""
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/alpha/bar-data/import",
            data={"interval": "d", "import_mode": "merge", "save_mode": "official"},
            files={"file": ("bad.csv", BAR_CSV_MISSING_CLOSE, "text/csv")},
        )
        assert response.status_code == 400
        assert "close" in response.json()["detail"].lower()


def test_merge_succeeds_regardless_of_selection_order(monkeypatch, tmp_path) -> None:
    """一致重叠的批次可合并，且与 keys 传入顺序无关（内部按上传时间归并）。"""
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        first = _upload_bar_batch(client, CSV_BATCH_A, filename="first.csv")
        second = _upload_bar_batch(client, CSV_BATCH_B, filename="second.csv")

        request = {"kind": "raw_bar", "keys": [second["key"], first["key"]]}
        merge_response = client.post("/api/alpha/data/resources/merge", json=request)
        assert merge_response.status_code == 200
        assert merge_response.json()["success"] is True

        detail = client.get(
            "/api/alpha/data/resources/raw_bar/d__000001.SZSE", params={"limit": 10}
        ).json()
        row_20240104 = [r for r in detail["preview"] if r["datetime"].startswith("2024-01-04")][0]
        assert row_20240104["close"] == 11.2


def test_merge_rejects_inconsistent_overlap(monkeypatch, tmp_path) -> None:
    """重叠区数据不一致（同一时间点 OHLC 冲突）必须拒绝合并。"""
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        first = _upload_bar_batch(client, CSV_BATCH_A, filename="first.csv")
        conflict = _upload_bar_batch(client, CSV_BATCH_B_CONFLICT, filename="conflict.csv")

        request = {"kind": "raw_bar", "keys": [first["key"], conflict["key"]]}
        preview = client.post("/api/alpha/data/resources/merge/preview", json=request).json()
        assert preview["can_merge"] is False
        assert preview["conflict_count"] >= 1
        assert any("不一致" in e for e in preview["errors"])

        merge_response = client.post("/api/alpha/data/resources/merge", json=request)
        assert merge_response.status_code == 400


def test_single_batch_promotes_to_official(monkeypatch, tmp_path) -> None:
    """无正式资源时，单个批次可直接晋级为正式 K 线（无需重叠）。"""
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        batch = _upload_bar_batch(client, CSV_BATCH_A, filename="solo.csv")

        request = {"kind": "raw_bar", "keys": [batch["key"]]}
        preview = client.post("/api/alpha/data/resources/merge/preview", json=request).json()
        assert preview["can_merge"] is True
        assert preview["has_official"] is False

        merge_response = client.post("/api/alpha/data/resources/merge", json=request)
        assert merge_response.status_code == 200
        assert merge_response.json()["success"] is True

        resource_payload = client.get("/api/alpha/data/resources").json()
        assert len(resource_payload["raw_bars"]) == 1
        assert resource_payload["raw_bars"][0]["row_count"] == 3


def test_single_batch_merges_into_existing_official(monkeypatch, tmp_path) -> None:
    """已有正式 K 线时，单个重叠批次可并入现有正式资源并向后扩展。"""
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        # 先用 A 晋级出正式资源（01-02 ~ 01-04）。
        batch_a = _upload_bar_batch(client, CSV_BATCH_A, filename="a.csv")
        client.post("/api/alpha/data/resources/merge", json={"kind": "raw_bar", "keys": [batch_a["key"]]})

        # 再上传与正式资源重叠且一致、并扩展到 01-05 的 B。
        batch_b = _upload_bar_batch(client, CSV_BATCH_B, filename="b.csv")
        request = {"kind": "raw_bar", "keys": [batch_b["key"]]}
        preview = client.post("/api/alpha/data/resources/merge/preview", json=request).json()
        assert preview["can_merge"] is True
        assert preview["has_official"] is True

        merge_response = client.post("/api/alpha/data/resources/merge", json=request)
        assert merge_response.status_code == 200
        merged = merge_response.json()
        assert merged["success"] is True
        # 正式资源应扩展为 01-02 ~ 01-05 共 4 行。
        assert merged["row_count"] == 4

        resource_payload = client.get("/api/alpha/data/resources").json()
        assert resource_payload["raw_bars"][0]["row_count"] == 4


def test_tick_preview_and_import_endpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", tmp_path)

    app = create_app()
    with TestClient(app) as client:
        preview_response = client.post(
            "/api/alpha/ticks/import/preview",
            files={"file": ("ticks.csv", TICK_CSV_BYTES, "text/csv")},
        )
        assert preview_response.status_code == 200
        preview_payload = preview_response.json()

        assert preview_payload["data_kind"] == "tick"
        assert preview_payload["missing_required"] == []
        assert preview_payload["symbols"] == ["000001.SZSE"]

        import_response = client.post(
            "/api/alpha/ticks/import",
            data={"import_mode": "merge"},
            files={"file": ("ticks.csv", TICK_CSV_BYTES, "text/csv")},
        )
        assert import_response.status_code == 200
        import_payload = import_response.json()

        assert import_payload["success"] is True
        assert import_payload["imported_count"] == 3

        resource_response = client.get("/api/alpha/data/resources")
        assert resource_response.status_code == 200
        resource_payload = resource_response.json()
        assert resource_payload["raw_ticks"] == []
        assert len(resource_payload["raw_tick_batches"]) == 1
        assert resource_payload["raw_tick_batches"][0]["vt_symbol"] == "000001.SZSE"
