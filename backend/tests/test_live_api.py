"""
Decision_API 路由层单元/集成测试（交易操作台特性，任务 6.3）。

覆盖需求 1.1 / 1.2 / 1.5 / 1.6 / 1.7 / 4.1 / 4.2 / 4.3 / 5.3 / 6.7 / 7.3：

单元（同步端点 + 入口校验）：
- 1.5：模型不存在 → 404。
- 1.6：缺必填字段 → 400。
- 4.1：列出决策标识符集合。
- 4.2/4.3：单条决策详情；不存在 → 404。
- 7.3：接口文档标注无鉴权前置条件（route description）。

集成冒烟（异步任务 + 轮询）：
- 1.1/1.2：POST /api/live/decision 创建异步任务并返回 task_id。
- 1.4（关联）：轮询任务完成后校验 result.decision / result.risk_detail 结构。
- 5.3/6.7：决策日行情缺失 → 任务 FAILED 且 message 含「行情缺失」。

外部 I/O 全部桩化：`predict_cnn_signals` 与 `_load_close_price` 注入确定 signal/price；
`DecisionStore` 与模型库目录均用 `tmp_path` 隔离，不依赖真实 CNN/行情/网络。
"""

from __future__ import annotations

import time
from datetime import date, datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

from aitrade.api import live as live_api
from aitrade.live import orchestrator
from aitrade.live.decision import Decision, DecisionStore
from aitrade.live.decision_instant import make_signal_id
from aitrade.main import create_app


VT_SYMBOL = "000001.SZSE"
SCHEME = "eod_buy_v1"
MODEL = "测试"
TRADE_DATE = date(2026, 6, 9)
TRADE_DATE_STR = TRADE_DATE.isoformat()
# as_of 取决策日收盘后；Decision_Bar = 当日（与历史日频 signal_id 逐位一致）。
AS_OF = datetime(2026, 6, 9, 15, 5)
BAR_DT = datetime(2026, 6, 9, 15, 0)


def _decision(**over) -> Decision:
    base = dict(
        signal_id=make_signal_id(BAR_DT, "1d", SCHEME, "v3"),
        decision_bar_dt=BAR_DT.isoformat(),
        as_of=AS_OF.isoformat(),
        bar_freq="1d",
        scheme=SCHEME,
        action="buy",
        vt_symbol=VT_SYMBOL,
        volume=1000,
        price=10.0,
        signal=0.72,
        reason="概率达标且通过风控",
    )
    base.update(over)
    return Decision(**base)


def _signal_frame(
    signal: float, *, vt_symbol: str = VT_SYMBOL, trade_date: date = TRADE_DATE
) -> pl.DataFrame:
    """构造 predict_cnn_signals 同 schema 的桩输出：[datetime, vt_symbol, signal]。"""
    return pl.DataFrame(
        {
            "datetime": [datetime.combine(trade_date, datetime.min.time())],
            "vt_symbol": [vt_symbol],
            "signal": [float(signal)],
        }
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """构造隔离的 TestClient：决策存储用 tmp_path，模型库用 tmp_path 并预置一个模型文件。"""
    store = DecisionStore(tmp_path / "decisions")
    monkeypatch.setattr(live_api, "_store", store)

    model_dir = tmp_path / "cnn_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{MODEL}.pt").write_bytes(b"stub-model")
    monkeypatch.setattr(live_api, "CNN_MODEL_PATH", model_dir)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, store


def _request_body(**overrides) -> dict:
    body = {
        "model": MODEL,
        "vt_symbol": VT_SYMBOL,
        "scheme": SCHEME,
        "as_of": AS_OF.isoformat(),
        "bar_freq": "1d",
        "data_source": "upload",
        "portfolio": {"portfolio_value": 100000, "current_position": 0},
        "buy_threshold": 0.6,
        "model_version": "v3",
    }
    body.update(overrides)
    return body


def _stub_io(monkeypatch, *, signal: float, price: float) -> None:
    """桩化编排器外部 I/O：CNN 推理返回确定 signal、取价返回确定 price。"""
    monkeypatch.setattr(
        orchestrator, "predict_cnn_signals", lambda **kwargs: _signal_frame(signal)
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_close_price",
        lambda vt_symbol, instant: (float(price), "d"),
    )


def _poll_task(test_client: TestClient, task_id: str, timeout: float = 15.0) -> dict:
    """轮询任务直至 completed/failed，返回任务 dict。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = test_client.get(f"/api/alpha/tasks/{task_id}")
        assert resp.status_code == 200
        task = resp.json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


# ---------------------------------------------------------------------------
# 需求 1.5：模型不存在 → 404
# ---------------------------------------------------------------------------
def test_decision_model_not_found_returns_404(client) -> None:
    test_client, _ = client
    resp = test_client.post(
        "/api/live/decision", json=_request_body(model="不存在的模型")
    )
    assert resp.status_code == 404
    assert "模型不存在" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 需求 1.6：缺必填字段 → 400
# ---------------------------------------------------------------------------
def test_decision_missing_required_field_returns_400(client) -> None:
    test_client, _ = client
    # model 为空串：通过 Pydantic 类型校验，由路由入口业务兜底返回 400。
    resp = test_client.post("/api/live/decision", json=_request_body(model=""))
    assert resp.status_code == 400
    assert "缺少必填字段" in resp.json()["detail"]


def test_decision_missing_field_unprocessable_returns_422(client) -> None:
    """完全缺少必填字段（无 portfolio）由 Pydantic 校验拦截为 422。"""
    test_client, _ = client
    resp = test_client.post(
        "/api/live/decision",
        json={"model": MODEL, "vt_symbol": VT_SYMBOL, "scheme": SCHEME},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 需求 4.1：列出决策标识符集合
# ---------------------------------------------------------------------------
def test_list_decisions_empty_then_populated(client) -> None:
    test_client, store = client

    resp = test_client.get("/api/live/decisions")
    assert resp.status_code == 200
    assert resp.json() == {"signal_ids": []}

    # 预置一条决策后再次列出。
    decision = _decision()
    signal_id = decision.signal_id
    store.save(decision)
    resp2 = test_client.get("/api/live/decisions")
    assert resp2.status_code == 200
    assert resp2.json()["signal_ids"] == [signal_id.replace(":", "_")]


# ---------------------------------------------------------------------------
# 需求 4.2：单条决策详情
# ---------------------------------------------------------------------------
def test_get_decision_detail(client) -> None:
    test_client, store = client
    decision = _decision()
    signal_id = decision.signal_id
    store.save(decision)

    resp = test_client.get(f"/api/live/decisions/{signal_id}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["signal_id"] == signal_id
    assert payload["action"] == "buy"
    assert payload["vt_symbol"] == VT_SYMBOL
    assert payload["volume"] == 1000


# ---------------------------------------------------------------------------
# 需求 4.3：单条决策不存在 → 404
# ---------------------------------------------------------------------------
def test_get_decision_not_found_returns_404(client) -> None:
    test_client, _ = client
    resp = test_client.get("/api/live/decisions/不存在的signal_id")
    assert resp.status_code == 404
    assert "决策不存在" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 归档式删除：决策 + trace 整体移入 archive/，解除幂等占位
# ---------------------------------------------------------------------------
def test_delete_decision_archives_decision_and_trace(client, monkeypatch, tmp_path) -> None:
    from aitrade.live.decision_trace import DecisionTraceStore

    test_client, store = client
    trace_store = DecisionTraceStore(tmp_path / "decisions")
    monkeypatch.setattr(live_api, "_trace_store", trace_store)

    decision = _decision()
    store.save(decision)
    trace_store.save_if_absent(
        decision.signal_id,
        {
            "schema_version": 1,
            "run_id": "r1",
            "signal_id": decision.signal_id,
            "completed_sections": [],
            "sections": {},
        },
    )

    resp = test_client.delete(f"/api/live/decisions/{decision.signal_id}")
    assert resp.status_code == 200
    assert resp.json() == {
        "signal_id": decision.signal_id,
        "deleted": True,
        "trace_archived": True,
    }

    # 幂等占位解除：get/list/trace 均不再可见，同 signal_id 可重新产出决策。
    assert store.get(decision.signal_id) is None
    assert trace_store.get(decision.signal_id) is None
    assert test_client.get(f"/api/live/decisions/{decision.signal_id}").status_code == 404
    assert test_client.get("/api/live/decisions").json()["signal_ids"] == []

    # 审计痕迹保留：决策与 trace 各有一个带时间戳的归档文件。
    archived = sorted(p.name for p in (tmp_path / "decisions" / "archive").glob("*.json"))
    assert len(archived) == 2
    assert any(".trace." in name for name in archived)


def test_delete_decision_without_trace_still_archives_decision(client, monkeypatch, tmp_path) -> None:
    from aitrade.live.decision_trace import DecisionTraceStore

    test_client, store = client
    monkeypatch.setattr(live_api, "_trace_store", DecisionTraceStore(tmp_path / "decisions"))
    store.save(_decision())
    signal_id = _decision().signal_id

    resp = test_client.delete(f"/api/live/decisions/{signal_id}")
    assert resp.status_code == 200
    assert resp.json()["trace_archived"] is False
    assert store.get(signal_id) is None


def test_delete_decision_not_found_returns_404(client, monkeypatch, tmp_path) -> None:
    from aitrade.live.decision_trace import DecisionTraceStore

    test_client, _ = client
    monkeypatch.setattr(live_api, "_trace_store", DecisionTraceStore(tmp_path / "decisions"))
    resp = test_client.delete("/api/live/decisions/不存在的signal_id")
    assert resp.status_code == 404
    assert "决策不存在" in resp.json()["detail"]


def test_batch_delete_decisions_partial_success(client, monkeypatch, tmp_path) -> None:
    """批量删除部分成功：存在的归档（决策 + trace），缺失的归入 missing，不整体失败。"""
    from aitrade.live.decision_trace import DecisionTraceStore

    test_client, store = client
    trace_store = DecisionTraceStore(tmp_path / "decisions")
    monkeypatch.setattr(live_api, "_trace_store", trace_store)

    d1 = _decision()
    d2 = _decision(signal_id=make_signal_id(BAR_DT, "1d", "另一方案", "v3"), scheme="另一方案")
    store.save(d1)
    store.save(d2)
    trace_store.save_if_absent(
        d1.signal_id,
        {"schema_version": 1, "run_id": "r1", "signal_id": d1.signal_id,
         "completed_sections": [], "sections": {}},
    )

    resp = test_client.post(
        "/api/live/decisions/batch-delete",
        json={"signal_ids": [d1.signal_id, "不存在的id", d2.signal_id, d1.signal_id]},
    )
    assert resp.status_code == 200
    # 重复 id 去重、保持入参顺序。
    assert resp.json() == {
        "deleted": [d1.signal_id, d2.signal_id],
        "missing": ["不存在的id"],
    }

    # 两条决策与 d1 的 trace 均已归档不可见；归档目录留有 3 个文件。
    assert store.list_ids() == []
    assert trace_store.get(d1.signal_id) is None
    archived = list((tmp_path / "decisions" / "archive").glob("*.json"))
    assert len(archived) == 3


def test_batch_delete_decisions_empty_returns_400(client) -> None:
    test_client, _ = client
    resp = test_client.post("/api/live/decisions/batch-delete", json={"signal_ids": []})
    assert resp.status_code == 400
    assert "不能为空" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 需求 7.3：接口文档标注无鉴权前置条件
# ---------------------------------------------------------------------------
def test_openapi_marks_no_auth_precondition(client) -> None:
    test_client, _ = client
    schema = test_client.get("/openapi.json").json()
    description = schema["paths"]["/api/live/decision"]["post"]["description"]
    assert "无鉴权" in description


# ---------------------------------------------------------------------------
# 需求 1.1 / 1.2 / 1.4（集成冒烟）：POST → 轮询完成 → 校验 result 结构
# ---------------------------------------------------------------------------
def test_integration_decision_smoke(client, monkeypatch) -> None:
    test_client, store = client
    _stub_io(monkeypatch, signal=0.72, price=10.0)

    # 放宽风控限额，使达标信号产出 buy（默认单票上限 0.30 会拦截满仓买入）。
    body = _request_body(
        risk={"max_total_position_ratio": 0.95, "max_single_position_ratio": 0.95}
    )
    resp = test_client.post("/api/live/decision", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    assert task_id

    task = _poll_task(test_client, task_id)
    assert task["status"] == "completed", f"任务消息: {task.get('message')}"

    result = task["result"]
    decision = result["decision"]
    # action 合法且字段完整（需求 1.4）。
    assert decision["action"] in ("buy", "sell", "hold")
    for field_name in ("action", "volume", "price", "signal", "reason"):
        assert field_name in decision
    assert decision["action"] == "buy"
    assert decision["price"] == 10.0
    assert decision["signal"] == 0.72

    # risk_detail 为逐项明细列表，含 check/passed/detail。
    risk_detail = result["risk_detail"]
    assert isinstance(risk_detail, list) and len(risk_detail) == 5
    for item in risk_detail:
        assert {"check", "passed", "detail"} <= set(item.keys())

    # 决策已落盘，可经历史接口查询到。
    list_resp = test_client.get("/api/live/decisions")
    assert len(list_resp.json()["signal_ids"]) == 1


# ---------------------------------------------------------------------------
# 需求 5.3 / 6.7（集成冒烟）：行情缺失 → 任务 FAILED 且 message 含「行情缺失」
# ---------------------------------------------------------------------------
def test_integration_missing_quote_fails_task(client, monkeypatch) -> None:
    test_client, _ = client

    monkeypatch.setattr(
        orchestrator, "predict_cnn_signals", lambda **kwargs: _signal_frame(0.72)
    )

    def _raise(vt_symbol, instant):
        raise ValueError(f"决策时刻 {instant.as_of.isoformat()} 之前的 {vt_symbol} 行情缺失")

    monkeypatch.setattr(orchestrator, "_load_close_price", _raise)

    resp = test_client.post("/api/live/decision", json=_request_body())
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll_task(test_client, task_id)
    assert task["status"] == "failed"
    assert "行情缺失" in task["message"]


# ---------------------------------------------------------------------------
# 需求 8.4：一次运行后请求 /trace 返回六段完整 trace
# ---------------------------------------------------------------------------
def test_get_decision_trace_returns_six_sections(client, monkeypatch, tmp_path) -> None:
    test_client, store = client
    # 决策过程档案存储与 _store 同目录（tmp_path 为本测试函数级唯一目录，与 fixture 共享）。
    from aitrade.live.decision_trace import DecisionTraceStore

    trace_store = DecisionTraceStore(tmp_path / "decisions")
    monkeypatch.setattr(live_api, "_trace_store", trace_store)

    _stub_io(monkeypatch, signal=0.72, price=10.0)

    # 放宽风控限额，使达标信号产出 buy 并完整走完六段（非中止路径）。
    body = _request_body(
        risk={"max_total_position_ratio": 0.95, "max_single_position_ratio": 0.95}
    )
    resp = test_client.post("/api/live/decision", json=body)
    assert resp.status_code == 200
    task = _poll_task(test_client, resp.json()["task_id"])
    assert task["status"] == "completed", f"任务消息: {task.get('message')}"

    signal_id = task["result"]["decision"]["signal_id"]
    trace_resp = test_client.get(f"/api/live/decisions/{signal_id}/trace")
    assert trace_resp.status_code == 200
    trace = trace_resp.json()

    # 六段完整：completed_sections 与 sections 均覆盖六段。
    expected_sections = [
        "run_header", "inference", "pricing", "decision_logic", "risk", "result",
    ]
    assert trace["completed_sections"] == expected_sections
    assert set(trace["sections"].keys()) == set(expected_sections)
    assert trace["signal_id"] == signal_id


# ---------------------------------------------------------------------------
# 需求 8.5：未知 signal_id 返回 404 与说明消息
# ---------------------------------------------------------------------------
def test_get_decision_trace_not_found_returns_404(client, monkeypatch, tmp_path) -> None:
    test_client, _ = client
    from aitrade.live.decision_trace import DecisionTraceStore

    monkeypatch.setattr(
        live_api, "_trace_store", DecisionTraceStore(tmp_path / "decisions")
    )

    resp = test_client.get("/api/live/decisions/不存在的signal_id/trace")
    assert resp.status_code == 404
    assert "决策过程档案不存在" in resp.json()["detail"]
