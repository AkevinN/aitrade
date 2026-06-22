"""选股 Tier-2 WF 报告读取端点测试。

覆盖 GET /api/cnn/screening/reports/{report_id}（cnn-screening-tier2-detail Task 1.2）：
- Property 1（隔离读取）：只读选股隔离 store；仅存在于生产 store 的 report_id 恒 404。
- Property 2（往返一致）：任意报告 dict 经 save_report 落盘后，端点读回 JSON 与原 dict 等价。
- Property 3（缺失即 404 且只读无副作用）：缺失 id → 404；调用端点不产生任何写入。

测试策略：
- monkeypatch ``aitrade.screening.store.build_screening_governance_store`` 指向 tmp 隔离 store，
  端点函数体内的 ``from ..screening.store import build_screening_governance_store`` 即解析到桩。
- 另建一个独立 tmp「生产替身」store，验证端点读不到只在它里面的报告（隔离）。
- Hypothesis 生成任意 JSON-native 报告 dict（``@settings(max_examples=100)``）。

Feature: cnn-screening-tier2-detail
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import aitrade.screening.store as screening_store
from aitrade.cnn.governance import CNNGovernanceStore
from aitrade.main import create_app

_BASE = "/api/cnn/screening/reports"

# ---------------------------------------------------------------------------
# Hypothesis 策略：JSON-native 报告 dict
# ---------------------------------------------------------------------------

_report_id_st = st.from_regex(r"[A-Za-z0-9_\-]{1,30}", fullmatch=True)

_json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-1_000_000_000, max_value=1_000_000_000)
    | st.floats(allow_nan=False, allow_infinity=False, width=64)
    | st.text(max_size=20)
)
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=4),
    max_leaves=15,
)
_report_body_st = st.dictionaries(st.text(min_size=1, max_size=8), _json_values, max_size=6)


# ---------------------------------------------------------------------------
# Fixture：tmp 隔离 store + 已打桩的 TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def screening_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """构造指向 tmp 的隔离 store，并把 build_screening_governance_store 打桩到它。

    Returns:
        (client, screening_store_instance, production_store_instance) 三元组。
        screening_store 是端点实际读取的隔离 store；production_store 是独立替身，
        用于验证端点读不到仅存在于它的报告。
    """
    screening = CNNGovernanceStore(root=tmp_path / "screening_gov")
    production = CNNGovernanceStore(root=tmp_path / "cnn_gov")
    monkeypatch.setattr(
        screening_store, "build_screening_governance_store", lambda: screening
    )
    app = create_app()
    # 刻意不用 `with TestClient(app)`：context-manager 形式会触发 FastAPI lifespan，
    # 启动 live 调度器（main.py:lifespan），它会向真实 .aitrade/live/scheduler_runs 落盘
    # （被 tests/conftest 的防泄漏兜底判失败）。本端点为纯读取、无需 lifespan，故直接构造。
    client = TestClient(app)
    yield client, screening, production


def _reports_file_count(store: CNNGovernanceStore) -> int:
    """隔离 store reports 目录下 .json 文件数（用于断言读取无副作用）。"""
    return len(list(store.reports_dir.glob("*.json")))


# ---------------------------------------------------------------------------
# Property 1 + Property 2：隔离命中 + 往返一致
# ---------------------------------------------------------------------------


class TestRoundTripAndIsolationHit:
    """写入隔离 store 的报告，端点 200 读回且与原 dict 等价。"""

    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(report_id=_report_id_st, body=_report_body_st)
    def test_saved_report_roundtrips_through_endpoint(
        self, screening_env, report_id: str, body: dict
    ) -> None:
        # Feature: cnn-screening-tier2-detail, Property 1 (hit), Property 2 (round-trip)
        client, screening, _ = screening_env
        report = {**body, "report_id": report_id}
        screening.save_report(report)

        resp = client.get(f"{_BASE}/{report_id}")

        assert resp.status_code == 200
        assert resp.json() == report

    def test_realistic_folds_report_roundtrips(self, screening_env) -> None:
        """具象 sanity：含 folds/summary 的真实形状报告往返不丢字段。"""
        client, screening, _ = screening_env
        report = {
            "report_id": "wf_20250622153012_a1b2c3",
            "type": "walk_forward",
            "name": "600000-tier2",
            "created_at": "2025-06-22T15:30:12",
            "request": {"name": "600000-tier2", "n_seeds": 3},
            "production_model": None,
            "folds": [
                {
                    "fold": 0,
                    "train": {"start": "2024-01-01", "end": "2024-06-01"},
                    "test": {"start": "2024-06-02", "end": "2024-09-01"},
                    "candidate_score": 0.61,
                    "cross_seed": {"mean": 0.61, "std": 0.04, "n": 3},
                    "candidate_statistics": {"total_return": 5.2, "sharpe_ratio": 1.2},
                    "production_model": None,
                    "production_score": None,
                    "score_delta": None,
                }
            ],
            "summary": {
                "fold_count": 1,
                "candidate_win_count": 0,
                "candidate_win_rate": 0.0,
                "n_seeds": 3,
                "avg_cross_seed_std": 0.04,
                "passed": False,
                "reasons": ["无生产模型"],
            },
        }
        screening.save_report(report)

        resp = client.get(f"{_BASE}/wf_20250622153012_a1b2c3")

        assert resp.status_code == 200
        assert resp.json() == report


# ---------------------------------------------------------------------------
# Property 1（miss）：隔离——读不到仅存在于生产 store 的报告
# ---------------------------------------------------------------------------


class TestIsolationMiss:
    """端点不串读生产 store。"""

    def test_report_only_in_production_store_is_404(self, screening_env) -> None:
        # Feature: cnn-screening-tier2-detail, Property 1 (isolation miss)
        client, _screening, production = screening_env
        rid = "wf_only_in_production"
        production.save_report({"report_id": rid, "marker": "production-only"})

        resp = client.get(f"{_BASE}/{rid}")

        assert resp.status_code == 404
        # 同一 id 在生产替身里确实存在，证明 404 来自隔离而非数据缺失
        assert production.get_report(rid) is not None


# ---------------------------------------------------------------------------
# Property 3：缺失即 404 + 读取无副作用
# ---------------------------------------------------------------------------


class TestMissingAndNoSideEffects:
    """缺失 id → 404；GET 不写任何文件。"""

    def test_missing_report_id_returns_404(self, screening_env) -> None:
        # Feature: cnn-screening-tier2-detail, Property 3 (missing -> 404)
        client, _screening, _ = screening_env
        resp = client.get(f"{_BASE}/does_not_exist")
        assert resp.status_code == 404
        assert "选股报告不存在" in resp.json()["detail"]

    def test_get_does_not_write_to_store(self, screening_env) -> None:
        # Feature: cnn-screening-tier2-detail, Property 3 (read-only, no side effect)
        client, screening, _ = screening_env
        screening.save_report({"report_id": "wf_seed", "k": 1})
        before = _reports_file_count(screening)

        client.get(f"{_BASE}/wf_seed")  # 命中
        client.get(f"{_BASE}/missing")  # 未命中

        assert _reports_file_count(screening) == before
