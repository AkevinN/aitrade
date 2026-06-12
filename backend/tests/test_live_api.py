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
from aitrade.live.position_book import PortfolioState, PositionBook
from aitrade.live.rebalance_decision import RebalanceDecision, RebalanceItem, RebalanceStore
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


# =============================================================================
# Phase 3 M2：Rebalance 端点测试
# =============================================================================


def _make_rebalance_decision(
    signal_id: str = "rule_test_sig_001",
    portfolio_id: str = "p_test",
    items: list[RebalanceItem] | None = None,
    status: str = "proposed",
) -> RebalanceDecision:
    if items is None:
        items = [RebalanceItem(vt_symbol="000001.SZSE", action="buy", volume=1000, price=10.0)]
    return RebalanceDecision(
        signal_id=signal_id,
        decision_bar_dt="2026-06-01T15:00:00",
        as_of="2026-06-01T15:05:00",
        bar_freq="1d",
        scheme="rule:etf_momentum",
        portfolio_id=portfolio_id,
        items=items,
        target_portfolio={"000001.SZSE": 1000},
        status=status,
    )


@pytest.fixture
def rebalance_client(tmp_path, monkeypatch):
    """构造隔离的 TestClient，注入独立的 RebalanceStore 与 PositionBook。"""
    rb_store = RebalanceStore(tmp_path / "rebalances")
    pb = PositionBook(tmp_path / "portfolios")
    monkeypatch.setattr(live_api, "_rebalance_store", rb_store)
    monkeypatch.setattr(live_api, "_position_book", pb)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, rb_store, pb


# ---------------------------------------------------------------------------
# GET /api/live/rebalances
# ---------------------------------------------------------------------------


def test_list_rebalances_empty(rebalance_client) -> None:
    test_client, _, _ = rebalance_client
    resp = test_client.get("/api/live/rebalances")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


def test_list_rebalances_returns_summary(rebalance_client) -> None:
    test_client, rb_store, _ = rebalance_client
    d = _make_rebalance_decision()
    rb_store.save(d)

    resp = test_client.get("/api/live/rebalances")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["signal_id"] == d.signal_id
    assert items[0]["status"] == "proposed"
    assert items[0]["portfolio_id"] == "p_test"
    assert "created_at" in items[0]


# ---------------------------------------------------------------------------
# GET /api/live/rebalances/{signal_id}
# ---------------------------------------------------------------------------


def test_get_rebalance_detail(rebalance_client) -> None:
    test_client, rb_store, _ = rebalance_client
    d = _make_rebalance_decision()
    rb_store.save(d)

    resp = test_client.get(f"/api/live/rebalances/{d.signal_id}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["signal_id"] == d.signal_id
    assert payload["scheme"] == "rule:etf_momentum"
    assert isinstance(payload["items"], list)
    assert payload["items"][0]["vt_symbol"] == "000001.SZSE"


def test_get_rebalance_not_found_returns_404(rebalance_client) -> None:
    test_client, _, _ = rebalance_client
    resp = test_client.get("/api/live/rebalances/不存在的signal_id")
    assert resp.status_code == 404
    assert "调仓决策不存在" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/live/portfolios/{portfolio_id}
# ---------------------------------------------------------------------------


def test_get_portfolio_missing_returns_empty(rebalance_client) -> None:
    """账本文件缺失时返回空账本（positions={}），不 404。"""
    test_client, _, _ = rebalance_client
    resp = test_client.get("/api/live/portfolios/p_new")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["portfolio_id"] == "p_new"
    assert payload["positions"] == {}


def test_get_portfolio_with_positions(rebalance_client) -> None:
    test_client, _, pb = rebalance_client
    state = PortfolioState(portfolio_id="p1", positions={"000001.SZSE": 500})
    pb.save(state)

    resp = test_client.get("/api/live/portfolios/p1")
    assert resp.status_code == 200
    assert resp.json()["positions"]["000001.SZSE"] == 500


# ---------------------------------------------------------------------------
# POST /api/live/rebalances/{signal_id}/confirm — 全链路
# ---------------------------------------------------------------------------


def test_confirm_rebalance_full_flow(rebalance_client) -> None:
    """confirm 全链路：决策 proposed → confirmed，账本更新。"""
    test_client, rb_store, pb = rebalance_client
    d = _make_rebalance_decision(
        items=[RebalanceItem(vt_symbol="000001.SZSE", action="buy", volume=1000)]
    )
    rb_store.save(d)

    resp = test_client.post(f"/api/live/rebalances/{d.signal_id}/confirm")
    assert resp.status_code == 200
    payload = resp.json()
    # 决策状态更新
    assert payload["decision"]["status"] == "confirmed"
    assert payload["decision"]["confirmed_at"] != ""
    # 账本已更新
    assert payload["portfolio"]["positions"]["000001.SZSE"] == 1000
    assert payload["portfolio"]["last_signal_id"] == d.signal_id

    # 数据库层面也已更新
    updated_d = rb_store.get(d.signal_id)
    assert updated_d is not None
    assert updated_d.status == "confirmed"
    portfolio_state = pb.load("p_test")
    assert portfolio_state.positions["000001.SZSE"] == 1000


# ---------------------------------------------------------------------------
# POST /api/live/rebalances/{signal_id}/confirm — 404
# ---------------------------------------------------------------------------


def test_confirm_rebalance_not_found_returns_404(rebalance_client) -> None:
    test_client, _, _ = rebalance_client
    resp = test_client.post("/api/live/rebalances/不存在的id/confirm")
    assert resp.status_code == 404
    assert "调仓决策不存在" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/live/rebalances/{signal_id}/confirm — 409 已确认
# ---------------------------------------------------------------------------


def test_confirm_rebalance_already_confirmed_returns_409(rebalance_client) -> None:
    test_client, rb_store, _ = rebalance_client
    d = _make_rebalance_decision(status="confirmed")
    rb_store.save(d)

    resp = test_client.post(f"/api/live/rebalances/{d.signal_id}/confirm")
    assert resp.status_code == 409
    assert "已确认" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/live/rebalances/{signal_id}/confirm — 400 超卖
# ---------------------------------------------------------------------------


def test_confirm_rebalance_oversell_returns_400(rebalance_client) -> None:
    """卖出超过持仓 → 400。"""
    test_client, rb_store, pb = rebalance_client
    # 账本里只有 100 股
    pb.save(PortfolioState(portfolio_id="p_test", positions={"000001.SZSE": 100}))

    d = _make_rebalance_decision(
        items=[RebalanceItem(vt_symbol="000001.SZSE", action="sell", volume=200)]
    )
    rb_store.save(d)

    resp = test_client.post(f"/api/live/rebalances/{d.signal_id}/confirm")
    assert resp.status_code == 400
    assert "超过当前持仓" in resp.json()["detail"]

    # 决策状态未变
    d_after = rb_store.get(d.signal_id)
    assert d_after is not None
    assert d_after.status == "proposed"


# ---------------------------------------------------------------------------
# POST /api/live/rebalances/{signal_id}/confirm — 半完成态自愈
# ---------------------------------------------------------------------------


def test_confirm_rebalance_self_heal_half_done_state(rebalance_client) -> None:
    """半完成态自愈：账本已应用（last_signal_id 命中）但决策仍 proposed，
    模拟 apply_rebalance 成功而 update_status 在崩溃窗口未完成的场景。
    重试 confirm 应返回 200，决策转 confirmed，且账本持仓数不变。
    """
    test_client, rb_store, pb = rebalance_client

    d = _make_rebalance_decision(
        items=[RebalanceItem(vt_symbol="000001.SZSE", action="buy", volume=1000)]
    )
    rb_store.save(d)

    # 手动将账本置为"已应用"状态（模拟 apply_rebalance 成功但 update_status 未执行）
    from aitrade.live.position_book import PortfolioState
    half_done_book = PortfolioState(
        portfolio_id="p_test",
        positions={"000001.SZSE": 1000},
        last_signal_id=d.signal_id,
    )
    pb.save(half_done_book)

    # 此时决策仍为 proposed（update_status 崩溃窗口未完成）
    assert rb_store.get(d.signal_id).status == "proposed"

    # 重试 confirm → 应自愈返回 200
    resp = test_client.post(f"/api/live/rebalances/{d.signal_id}/confirm")
    assert resp.status_code == 200
    payload = resp.json()

    # 决策已补写为 confirmed
    assert payload["decision"]["status"] == "confirmed"
    assert payload["decision"]["confirmed_at"] != ""

    # 账本持仓数不变（未被二次应用）
    assert payload["portfolio"]["positions"]["000001.SZSE"] == 1000
    assert payload["portfolio"]["last_signal_id"] == d.signal_id

    # 存储层也已更新
    updated_d = rb_store.get(d.signal_id)
    assert updated_d is not None
    assert updated_d.status == "confirmed"

    # 账本仍是 1000 股（未变动）
    book_after = pb.load("p_test")
    assert book_after.positions.get("000001.SZSE") == 1000


# =============================================================================
# Task 3.7：POST /api/live/rebalance 全链路 + 调度分派用例
# =============================================================================


from aitrade.live.trading_plan import TradingPlan, TradingPlanStore  # noqa: E402


class _StubProvider:
    """可计数调用的确定性信号源桩（仿 test_rebalance_orchestrator.py）。"""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.call_count = 0

    def predict(self, start, end, on_progress=None) -> pl.DataFrame:
        self.call_count += 1
        if not self._rows:
            return pl.DataFrame(
                {"datetime": [], "vt_symbol": [], "signal": [], "close": []},
                schema={
                    "datetime": pl.Datetime,
                    "vt_symbol": pl.Utf8,
                    "signal": pl.Float64,
                    "close": pl.Float64,
                },
            )
        return pl.DataFrame(self._rows).with_columns(
            pl.col("datetime").cast(pl.Datetime)
        )


def _stub_rebalance_row(bar_dt, sym="000001.SZSE", signal=0.9, close=10.0) -> dict:
    return {"datetime": bar_dt, "vt_symbol": sym, "signal": signal, "close": close}


@pytest.fixture
def rebalance_api_client(tmp_path, monkeypatch):
    """构造隔离的 TestClient，注入独立的 RebalanceStore、PositionBook、TradingPlanStore。"""
    rb_store = RebalanceStore(tmp_path / "rebalances")
    pb = PositionBook(tmp_path / "portfolios")
    plan_store = TradingPlanStore(tmp_path / "plans")
    monkeypatch.setattr(live_api, "_rebalance_store", rb_store)
    monkeypatch.setattr(live_api, "_position_book", pb)
    monkeypatch.setattr(live_api, "_plan_store", plan_store)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, rb_store, pb, plan_store


# ---------------------------------------------------------------------------
# POST /api/live/rebalance — 内联模式全链路（completed + result 形态）
# ---------------------------------------------------------------------------


def test_rebalance_inline_mode_smoke(tmp_path, monkeypatch, rebalance_api_client) -> None:
    """内联模式：POST /api/live/rebalance 创建异步任务，轮询完成后验证 result 形态。"""
    test_client, rb_store, pb, _ = rebalance_api_client

    from datetime import date, datetime, time as dtime
    bar_dt = datetime.combine(date(2026, 6, 9), dtime(15, 0))

    provider = _StubProvider([_stub_rebalance_row(bar_dt)])
    from aitrade.backtest.registry import register_signal_source
    register_signal_source("stub_inline_src", lambda params: provider)

    body = {
        "plan_name": "inline_test",
        "signal_source": "stub_inline_src",
        "signal_params": {},
        "strategy_params": {"top_k": 1},
        "portfolio_id": "p_inline",
        "capital": 100_000.0,
        "as_of": datetime(2026, 6, 9, 15, 5).isoformat(),
        "bar_freq": "1d",
    }
    resp = test_client.post("/api/live/rebalance", json=body)
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    assert task_id

    task = _poll_task(test_client, task_id)
    assert task["status"] == "completed", f"任务消息: {task.get('message')}"

    result = task["result"]
    assert "decision" in result
    assert "idempotent_hit" in result
    assert result["idempotent_hit"] is False

    # 决策已落盘
    decision_dict = result["decision"]
    if decision_dict is not None:
        signal_id = decision_dict["signal_id"]
        assert rb_store.get(signal_id) is not None


def test_rebalance_inline_mode_missing_fields_returns_400(rebalance_api_client) -> None:
    """内联模式缺少必填字段 → 400。"""
    test_client, _, _, _ = rebalance_api_client
    resp = test_client.post("/api/live/rebalance", json={
        "plan_name": "",  # 内联模式但 plan_name 为空
        "signal_source": "",
    })
    assert resp.status_code == 400
    assert "必填" in resp.json()["detail"]


def test_rebalance_plan_id_not_found_returns_404(rebalance_api_client) -> None:
    """plan_id 不存在 → 404。"""
    test_client, _, _, _ = rebalance_api_client
    resp = test_client.post("/api/live/rebalance", json={"plan_id": "不存在的id"})
    assert resp.status_code == 404
    assert "交易计划不存在" in resp.json()["detail"]


def test_rebalance_plan_id_wrong_type_returns_400(rebalance_api_client) -> None:
    """plan_id 指向 cnn 计划（非 rule 类型）→ 400。"""
    test_client, _, _, plan_store = rebalance_api_client
    cnn_plan = TradingPlan(
        plan_id="cnn001",
        name="cnn计划",
        model="测试",
        vt_symbol="000001.SZSE",
        scheme="eod_buy",
        strategy_type="cnn",  # 非 rule
        portfolio={"portfolio_value": 100_000},
        trigger_times=["15:05"],
    )
    plan_store.save(cnn_plan)

    resp = test_client.post("/api/live/rebalance", json={"plan_id": "cnn001"})
    assert resp.status_code == 400
    assert "rule" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _trigger_plan 调度分派：rule 计划走新路径（仿既有 _trigger_plan 测试风格）
# ---------------------------------------------------------------------------


def test_trigger_plan_rule_dispatches_to_rebalance(tmp_path, monkeypatch) -> None:
    """rule 计划经 _trigger_plan 走 run_rebalance_decision 路径（而非 run_live_decision）。"""
    from datetime import date, datetime, time as dtime
    bar_dt = datetime.combine(date(2026, 6, 9), dtime(15, 0))

    provider = _StubProvider([_stub_rebalance_row(bar_dt)])
    from aitrade.backtest.registry import register_signal_source
    register_signal_source("stub_dispatch_src", lambda params: provider)

    rb_store = RebalanceStore(tmp_path / "rb")
    pb = PositionBook(tmp_path / "pb")
    monkeypatch.setattr(live_api, "_rebalance_store", rb_store)
    monkeypatch.setattr(live_api, "_position_book", pb)
    from aitrade.live.runtime_state import RuntimeStateStore
    from aitrade.live.portfolio_risk import PortfolioRiskConfig
    rs = RuntimeStateStore(tmp_path / "rs.json")
    monkeypatch.setattr(live_api, "_runtime_state", rs)
    monkeypatch.setattr(live_api, "_portfolio_risk_config", PortfolioRiskConfig())

    plan = TradingPlan(
        plan_id="rule001",
        name="rule_test",
        model="",
        vt_symbol="",
        scheme="rule:test",
        strategy_type="rule",
        signal_source="stub_dispatch_src",
        signal_params={},
        portfolio_id="p_dispatch",
        portfolio={"portfolio_value": 100_000},
        min_volume=100,
        trigger_times=["15:05"],
    )

    # 调用 _trigger_plan — 应不抛错，返回 dict 含 decision/idempotent_hit
    result = live_api._trigger_plan(plan)
    assert "decision" in result or "skipped_reason" in result
    assert "idempotent_hit" in result


def test_trigger_plan_cnn_still_uses_run_live_decision(tmp_path, monkeypatch) -> None:
    """cnn 计划经 _trigger_plan 仍走 run_live_decision（原有路径零改动）。

    用 monkeypatch sentinel 验证：run_live_decision 被调用，run_rebalance_decision 不被调用。
    """
    run_live_called: list[int] = [0]
    run_rebalance_called: list[int] = [0]

    def _sentinel_live(**kwargs) -> dict:
        run_live_called[0] += 1
        return {"decision": {}, "risk_detail": [], "idempotent_hit": False}

    def _sentinel_rebalance(**kwargs) -> dict:
        run_rebalance_called[0] += 1
        return {"decision": None, "idempotent_hit": False, "risk": [], "skipped_reason": "stub"}

    monkeypatch.setattr(live_api, "run_live_decision", _sentinel_live)
    monkeypatch.setattr(live_api, "run_rebalance_decision", _sentinel_rebalance)

    cnn_plan = TradingPlan(
        plan_id="cnn_p",
        name="cnn计划",
        model="测试",
        vt_symbol="000001.SZSE",
        scheme="eod",
        strategy_type="cnn",
        portfolio={"portfolio_value": 100_000},
        trigger_times=["15:05"],
    )

    live_api._trigger_plan(cnn_plan)

    assert run_live_called[0] == 1
    assert run_rebalance_called[0] == 0


# =============================================================================
# Lab 注入回归测试：_get_lab 是非空买入的必要条件
# =============================================================================

# 辅助常量与 helper —— 独立于上方 Decision 测试的常量
_LAB_SYM = "000001.SZSE"
_LAB_CLOSE = 12.34          # lab 中写入的 close 价格，断言时用
_LAB_BAR_DATE = date(2026, 6, 9)
_LAB_AS_OF = datetime(2026, 6, 9, 15, 5)  # 收盘后 5 min，bar 已可见


class _StubProvider3Col:
    """仅返回 [datetime, vt_symbol, signal] 三列的桩——与 etf_momentum 真实输出一致，无 close 列。

    买入价**无法**经 _get_price 路径 1（signal_df close 列）获取，
    只能走路径 2（AlphaLab）——这就是 lab 注入保护的核心路径。
    """

    def predict(self, start, end, on_progress=None) -> pl.DataFrame:
        bar_dt = datetime.combine(_LAB_BAR_DATE, datetime.min.time().replace(hour=15))
        return pl.DataFrame(
            {
                "datetime": [bar_dt],
                "vt_symbol": [_LAB_SYM],
                "signal": [0.9],
            }
        )


def _make_lab_bar_frame(close: float, bar_date: date) -> pl.DataFrame:
    """构造含单根日线的 DataFrame，写入 AlphaLab。"""
    dt = datetime(bar_date.year, bar_date.month, bar_date.day, 15, 0)
    return pl.DataFrame(
        {
            "datetime": [dt],
            "open": [close - 0.1],
            "high": [close + 0.5],
            "low": [close - 0.5],
            "close": [close],
            "volume": [1_000_000.0],
            "turnover": [close * 1_000_000.0],
            "open_interest": [0.0],
        }
    )


def _setup_lab_injection_client(tmp_path, monkeypatch):
    """公共夹具逻辑：注册桩信号源、创建隔离 TestClient 并返回 (test_client, rb_store)。"""
    from aitrade.backtest.registry import register_signal_source

    # 使用唯一信号源名，避免跨测试注册污染
    _SRC_NAME = f"stub_lab_test_{id(tmp_path)}"

    stub = _StubProvider3Col()
    register_signal_source(_SRC_NAME, lambda params: stub)

    rb_store = RebalanceStore(tmp_path / "rebalances")
    pb = PositionBook(tmp_path / "portfolios")
    plan_store = TradingPlanStore(tmp_path / "plans")
    monkeypatch.setattr(live_api, "_rebalance_store", rb_store)
    monkeypatch.setattr(live_api, "_position_book", pb)
    monkeypatch.setattr(live_api, "_plan_store", plan_store)

    app = create_app()
    test_client = TestClient(app)
    test_client.__enter__()

    return test_client, rb_store, _SRC_NAME


def _rebalance_body(signal_source: str, as_of: datetime = _LAB_AS_OF) -> dict:
    return {
        "plan_name": "lab_test",
        "signal_source": signal_source,
        "signal_params": {},
        "strategy_params": {"top_k": 1},
        "portfolio_id": "p_lab",
        "capital": 100_000.0,
        "as_of": as_of.isoformat(),
        "bar_freq": "1d",
    }


# ---------------------------------------------------------------------------
# 正例：lab 注入 → buy items 非空，price == lab close
# ---------------------------------------------------------------------------


def test_live_api_lab_injection_produces_buy_items(tmp_path, monkeypatch) -> None:
    """lab 注入正例：stub provider 三列（无 close），_get_lab 返回含行情的 AlphaLab，
    全链路 POST /rebalance 应 completed、buy items 非空、price 来自 lab close。
    """
    from aitrade.alpha.lab import AlphaLab

    # 构造含目标标的日线的 stub lab。
    lab = AlphaLab(tmp_path / "lab")
    lab.save_bar_frame(_LAB_SYM, "d", _make_lab_bar_frame(_LAB_CLOSE, _LAB_BAR_DATE))

    # monkeypatch _get_lab → stub lab（使 run_rebalance_decision 中 lab= 取到含行情的实例）。
    monkeypatch.setattr(live_api, "_get_lab", lambda: AlphaLab(tmp_path / "lab"))

    test_client, rb_store, src_name = _setup_lab_injection_client(tmp_path, monkeypatch)

    resp = test_client.post("/api/live/rebalance", json=_rebalance_body(src_name))
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll_task(test_client, task_id)
    assert task["status"] == "completed", f"任务消息: {task.get('message')}"

    result = task["result"]
    assert result["idempotent_hit"] is False

    decision_dict = result["decision"]
    assert decision_dict is not None, "决策不应为 None"

    buy_items = [it for it in decision_dict["items"] if it["action"] == "buy"]
    assert len(buy_items) > 0, (
        f"buy items 应非空（lab 行情已注入），实际 items={decision_dict['items']}, "
        f"risk_summary={decision_dict['risk_summary']}"
    )
    assert buy_items[0]["vt_symbol"] == _LAB_SYM
    assert buy_items[0]["price"] == pytest.approx(_LAB_CLOSE, rel=1e-6), (
        f"buy price 应来自 lab close={_LAB_CLOSE}，实际={buy_items[0]['price']}"
    )

    test_client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# 对照断言（方案 b）：_get_lab 返回 None → buy items 为空，risk_summary 含 pricing record
# ---------------------------------------------------------------------------


def test_live_api_lab_none_produces_empty_buy_items(tmp_path, monkeypatch) -> None:
    """lab 对照测试：_get_lab 返回 None（模拟生产端 lab= 接线被回退），
    stub provider 三列（无 close）→ buy items 为空，risk_summary 含 pricing record，
    证明 lab 是非空买入的必要条件。
    """
    # monkeypatch _get_lab → None（无法取价）。
    monkeypatch.setattr(live_api, "_get_lab", lambda: None)

    test_client, rb_store, src_name = _setup_lab_injection_client(tmp_path, monkeypatch)

    resp = test_client.post("/api/live/rebalance", json=_rebalance_body(src_name))
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll_task(test_client, task_id)
    assert task["status"] == "completed", f"任务消息: {task.get('message')}"

    result = task["result"]
    decision_dict = result["decision"]
    assert decision_dict is not None

    buy_items = [it for it in decision_dict["items"] if it["action"] == "buy"]
    assert len(buy_items) == 0, (
        f"lab=None 时 buy items 应为空（无法取价），实际 items={decision_dict['items']}"
    )

    # risk_summary 应含 pricing record（说明跳过原因已记录，用户可见）。
    pricing_records = [
        r for r in decision_dict["risk_summary"] if r.get("check") == "pricing"
    ]
    assert len(pricing_records) > 0, (
        f"lab=None 时 risk_summary 应含 pricing record，实际 risk_summary={decision_dict['risk_summary']}"
    )
    assert pricing_records[0]["passed"] is False

    test_client.__exit__(None, None, None)
