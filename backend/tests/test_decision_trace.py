"""
TraceBuilder 与 DecisionTraceStore 单元测试（交易操作台特性，需求 8.2 / 8.3 / 8.9 / 8.10）。

验证四件事：
1. TraceBuilder 逐段累积六段，`to_trace()["completed_sections"]` 与 SECTIONS 同序，
   且每段 payload 原样保留（需求 8.10 的过程档案结构基础）。
2. 里程碑日志为 INFO、明细日志为 DEBUG，且每行均带 `[run_id]` 前缀（过程日志可观测性）。
3. `save_if_absent` 幂等：首次写入返回 True 并落盘；二次返回 False 且绝不重写（需求 8.9）。
4. `get` 往返一致、未知 signal_id 返回 None；写入 .trace.json 绝不影响 sibling
   `{signal_id}.json`（需求 8.2 / 8.3）。

全部使用 tmp_path / caplog，不依赖外部网络。signal_id 含 ":" 以验证安全化规则。
"""

from __future__ import annotations

import json
import logging

from aitrade.live.decision import Decision, DecisionStore
from aitrade.live.decision_trace import DecisionTraceStore, TraceBuilder


# 含 ":" 的 signal_id，验证 replace(":", "_") 安全化
SIGNAL_ID = "2026-06-08:eod_buy_v1@v3"
RUN_ID = "run-20260608-0001"


def _build_full_trace(builder: TraceBuilder) -> None:
    """按 SECTIONS 顺序填满六段，每段携带可识别 payload。"""
    for name in TraceBuilder.SECTIONS:
        builder.set_section(name, {"section": name, "value": f"payload-{name}"})


# ---------------------------------------------------------------------------
# 1. 逐段累积：六段同序，payload 原样保留
# ---------------------------------------------------------------------------
def test_set_section_accumulates_six_sections_in_order() -> None:
    logger = logging.getLogger("aitrade.live.orchestrator")
    builder = TraceBuilder(RUN_ID, SIGNAL_ID, logger)

    _build_full_trace(builder)
    trace = builder.to_trace()

    assert trace["completed_sections"] == list(TraceBuilder.SECTIONS)
    assert trace["run_id"] == RUN_ID
    assert trace["signal_id"] == SIGNAL_ID
    assert trace["schema_version"] == 1
    # 每段 payload 原样保留
    assert set(trace["sections"].keys()) == set(TraceBuilder.SECTIONS)
    for name in TraceBuilder.SECTIONS:
        assert trace["sections"][name] == {"section": name, "value": f"payload-{name}"}


def test_to_trace_honors_overrides() -> None:
    logger = logging.getLogger("aitrade.live.orchestrator")
    builder = TraceBuilder(RUN_ID, SIGNAL_ID, logger)
    builder.set_section("run_header", {"k": "v"})

    trace = builder.to_trace(schema_version=2)

    assert trace["schema_version"] == 2
    assert trace["completed_sections"] == ["run_header"]
    assert trace["sections"]["run_header"] == {"k": "v"}


# ---------------------------------------------------------------------------
# 2. 里程碑 INFO / 明细 DEBUG，每行带 [run_id] 前缀
# ---------------------------------------------------------------------------
def test_milestone_is_info_and_detail_is_debug_with_run_id_prefix(caplog) -> None:
    logger = logging.getLogger("aitrade.live.orchestrator")
    builder = TraceBuilder(RUN_ID, SIGNAL_ID, logger)

    with caplog.at_level(logging.DEBUG, logger="aitrade.live.orchestrator"):
        builder.set_section(
            "inference",
            {"bars": 240},
            debug_detail={"signal_values": [0.1, 0.2, 0.3]},
        )

    records = [r for r in caplog.records if r.name == "aitrade.live.orchestrator"]
    info_records = [r for r in records if r.levelno == logging.INFO]
    debug_records = [r for r in records if r.levelno == logging.DEBUG]

    # 一条里程碑 INFO + 一条明细 DEBUG
    assert len(info_records) == 1
    assert len(debug_records) == 1
    assert "段完成" in info_records[0].getMessage()
    assert "明细" in debug_records[0].getMessage()
    # 每行均带 [run_id] 前缀
    prefix = f"[{RUN_ID}]"
    for rec in records:
        assert rec.getMessage().startswith(prefix)


def test_no_debug_record_when_detail_omitted(caplog) -> None:
    logger = logging.getLogger("aitrade.live.orchestrator")
    builder = TraceBuilder(RUN_ID, SIGNAL_ID, logger)

    with caplog.at_level(logging.DEBUG, logger="aitrade.live.orchestrator"):
        builder.set_section("pricing", {"close": 12.34})

    records = [r for r in caplog.records if r.name == "aitrade.live.orchestrator"]
    assert all(r.levelno == logging.INFO for r in records)
    assert len(records) == 1
    assert records[0].getMessage() == f"[{RUN_ID}] 段完成: pricing"


# ---------------------------------------------------------------------------
# 3. save_if_absent 幂等：首次 True 落盘，二次 False 不重写
# ---------------------------------------------------------------------------
def test_save_if_absent_is_idempotent_and_does_not_rewrite(tmp_path) -> None:
    store = DecisionTraceStore(tmp_path)
    first = {"schema_version": 1, "run_id": RUN_ID, "signal_id": SIGNAL_ID,
             "completed_sections": ["run_header"], "sections": {"run_header": {"k": "first"}}}
    second = {"schema_version": 1, "run_id": "run-other", "signal_id": SIGNAL_ID,
              "completed_sections": ["run_header"], "sections": {"run_header": {"k": "second"}}}

    # 首次写入
    assert store.save_if_absent(SIGNAL_ID, first) is True
    # 文件以安全化命名生成（":" → "_"）
    safe = SIGNAL_ID.replace("/", "_").replace(":", "_")
    trace_path = tmp_path / f"{safe}.trace.json"
    assert trace_path.exists()

    # 二次写入不同内容 → 返回 False 且磁盘内容不变
    assert store.save_if_absent(SIGNAL_ID, second) is False
    on_disk = json.loads(trace_path.read_text(encoding="utf-8"))
    assert on_disk == first
    assert on_disk != second


# ---------------------------------------------------------------------------
# 4. get 往返一致 / 未知返回 None
# ---------------------------------------------------------------------------
def test_get_round_trips_saved_trace(tmp_path) -> None:
    store = DecisionTraceStore(tmp_path)
    logger = logging.getLogger("aitrade.live.orchestrator")
    builder = TraceBuilder(RUN_ID, SIGNAL_ID, logger)
    _build_full_trace(builder)
    trace = builder.to_trace()

    assert store.save_if_absent(SIGNAL_ID, trace) is True
    assert store.exists(SIGNAL_ID) is True
    assert store.get(SIGNAL_ID) == trace


def test_get_unknown_signal_id_returns_none(tmp_path) -> None:
    store = DecisionTraceStore(tmp_path)
    assert store.exists("does-not-exist:ever") is False
    assert store.get("does-not-exist:ever") is None


# ---------------------------------------------------------------------------
# 5. 写入 trace 不影响 sibling {signal_id}.json（需求 8.3）
# ---------------------------------------------------------------------------
def test_trace_write_does_not_affect_sibling_decision_json(tmp_path) -> None:
    # 同目录下先用 DecisionStore 落一条 Decision（{safe}.json）
    decision_store = DecisionStore(tmp_path)
    decision = Decision(
        signal_id=SIGNAL_ID,
        decision_bar_dt="2026-06-08T15:00:00",
        as_of="2026-06-08T15:00:00",
        bar_freq="1d",
        scheme="eod_buy_v1",
        action="buy",
        vt_symbol="000415.SZSE",
        volume=100,
        price=12.34,
        signal=0.87,
        reason="signal>=threshold",
    )
    decision_store.save(decision)
    safe = SIGNAL_ID.replace("/", "_").replace(":", "_")
    decision_path = tmp_path / f"{safe}.json"
    assert decision_path.exists()
    decision_before = decision_path.read_text(encoding="utf-8")

    # 在同目录写 trace（{safe}.trace.json）
    trace_store = DecisionTraceStore(tmp_path)
    logger = logging.getLogger("aitrade.live.orchestrator")
    builder = TraceBuilder(RUN_ID, SIGNAL_ID, logger)
    _build_full_trace(builder)
    assert trace_store.save_if_absent(SIGNAL_ID, builder.to_trace()) is True

    # Decision 文件内容与 get 结果均不受影响
    assert decision_path.read_text(encoding="utf-8") == decision_before
    assert decision_store.get(SIGNAL_ID) == decision

    # DecisionStore.list_ids 只识别决策文件 {signal_id}.json，并排除 sibling
    # 的 {signal_id}.trace.json（需求 8.3）。
    ids = decision_store.list_ids()
    assert safe in ids
    # .trace.json 的 stem 为 "{safe}.trace"，不得被纳入决策 id 列表
    assert f"{safe}.trace" not in ids

# ---------------------------------------------------------------------------
# 6. 回归：list_ids 排除 sibling .trace.json（需求 8.3，对应任务 15 修订点）
# ---------------------------------------------------------------------------
def test_list_ids_excludes_sibling_trace_json(tmp_path) -> None:
    """同目录同时存在 {signal_id}.json 与 {signal_id}.trace.json 时，
    DecisionStore.list_ids 仅返回 {signal_id}.json 对应的决策 id，
    绝不把 sibling 的 {signal_id}.trace.json 纳入列表（需求 8.3）。"""
    store = DecisionStore(tmp_path)
    decision = Decision(
        signal_id=SIGNAL_ID,
        decision_bar_dt="2026-06-08T15:00:00",
        as_of="2026-06-08T15:00:00",
        bar_freq="1d",
        scheme="eod_buy_v1",
        action="buy",
        vt_symbol="000415.SZSE",
        volume=100,
        price=12.34,
        signal=0.87,
        reason="signal>=threshold",
    )
    store.save(decision)

    safe = SIGNAL_ID.replace("/", "_").replace(":", "_")
    # 手工在同目录写一个 sibling trace 文件（模拟 DecisionTraceStore 的落盘产物）
    trace_path = tmp_path / f"{safe}.trace.json"
    trace_path.write_text(json.dumps({"run_id": RUN_ID}), encoding="utf-8")
    assert trace_path.exists()

    ids = store.list_ids()
    # 仅返回决策文件对应 id
    assert ids == [safe]
    # 显式断言：trace 派生的 id 不在列表中
    assert f"{safe}.trace" not in ids
