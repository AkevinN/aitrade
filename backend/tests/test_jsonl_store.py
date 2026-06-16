"""
JsonlDayStore 测试套件（task-scheduler-observability Wave 1，任务 1）

覆盖：
- 按日分文件
- append 自动补 ts
- dedup_key 去重（同键当日只写一条）
- 重建去重：写后新建实例再 append 同键仍去重
- 坏行容忍（read_day 跳过 + warning）
- 只读属性（Hypothesis，TSO-7）
- IO 故障桩（monkeypatch open 抛错 → append 返回 False 不抛）
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.live.jsonl_store import JsonlDayStore


# ---------------------------------------------------------------------------
# 辅助：固定时钟工厂
# ---------------------------------------------------------------------------

def fixed_now(dt: datetime):
    return lambda: dt


DAY_A = date(2026, 6, 11)
DAY_B = date(2026, 6, 12)

NOW_A = datetime(2026, 6, 11, 9, 30, 0, tzinfo=timezone.utc)
NOW_B = datetime(2026, 6, 12, 15, 5, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 基础：按日分文件
# ---------------------------------------------------------------------------

def test_per_day_file(tmp_path: Path) -> None:
    """两个不同日期的写入应落到不同文件。"""
    store_a = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store_a.append({"event": "alpha"})

    store_b = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_B))
    store_b.append({"event": "beta"})

    assert (tmp_path / "2026-06-11.jsonl").exists()
    assert (tmp_path / "2026-06-12.jsonl").exists()

    records_a = store_a.read_day(DAY_A)
    records_b = store_b.read_day(DAY_B)
    assert len(records_a) == 1
    assert records_a[0]["event"] == "alpha"
    assert len(records_b) == 1
    assert records_b[0]["event"] == "beta"


# ---------------------------------------------------------------------------
# 自动补 ts
# ---------------------------------------------------------------------------

def test_append_auto_ts(tmp_path: Path) -> None:
    """append 时如果 event 不含 ts，应自动补充 ISO 时间戳。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"msg": "hello"})
    records = store.read_day(DAY_A)
    assert len(records) == 1
    assert "ts" in records[0]
    # 时间戳应能解析
    datetime.fromisoformat(records[0]["ts"])


def test_append_preserves_existing_ts(tmp_path: Path) -> None:
    """如果 event 已含 ts 字段，不覆盖。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"ts": "2026-01-01T00:00:00", "msg": "custom"})
    records = store.read_day(DAY_A)
    assert records[0]["ts"] == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# dedup_key 去重
# ---------------------------------------------------------------------------

def test_dedup_same_key_skipped(tmp_path: Path) -> None:
    """同 dedup_key 当日只写一条，第二次返回 False。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    r1 = store.append({"ev": "first"}, dedup_key="k1")
    r2 = store.append({"ev": "second"}, dedup_key="k1")
    assert r1 is True
    assert r2 is False
    records = store.read_day(DAY_A)
    assert len(records) == 1
    assert records[0]["ev"] == "first"


def test_dedup_different_keys_both_written(tmp_path: Path) -> None:
    """不同 dedup_key 都应写入。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"ev": "x"}, dedup_key="k1")
    store.append({"ev": "y"}, dedup_key="k2")
    records = store.read_day(DAY_A)
    assert len(records) == 2


def test_dedup_different_days_not_shared(tmp_path: Path) -> None:
    """dedup_key 去重只在当日有效，跨日相同 key 可重复写入。"""
    store_a = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store_b = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_B))
    r1 = store_a.append({"ev": "day-a"}, dedup_key="k1")
    r2 = store_b.append({"ev": "day-b"}, dedup_key="k1")
    assert r1 is True
    assert r2 is True


def test_dedup_rebuild_after_restart(tmp_path: Path) -> None:
    """写后新建实例（模拟重启），再 append 同键仍去重。"""
    # 第一个实例写入
    store1 = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store1.append({"ev": "orig"}, dedup_key="k1")

    # 新建实例（无内存 set），应回放文件重建去重
    store2 = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    r = store2.append({"ev": "dup"}, dedup_key="k1")
    assert r is False

    records = store2.read_day(DAY_A)
    assert len(records) == 1
    assert records[0]["ev"] == "orig"


def test_dedup_key_stored_in_file(tmp_path: Path) -> None:
    """有 dedup_key 的行应写入 _dedup 字段，以便重建。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"ev": "x"}, dedup_key="my-key")
    raw = (tmp_path / "2026-06-11.jsonl").read_text()
    obj = json.loads(raw.strip())
    assert obj.get("_dedup") == "my-key"


def test_no_dedup_key_no_dedup_field(tmp_path: Path) -> None:
    """无 dedup_key 的行不写入 _dedup 字段。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"ev": "plain"})
    raw = (tmp_path / "2026-06-11.jsonl").read_text()
    obj = json.loads(raw.strip())
    assert "_dedup" not in obj


# ---------------------------------------------------------------------------
# M2：read_day / read_range 返回结果不含 _dedup 内部字段
# ---------------------------------------------------------------------------

def test_m2_read_day_strips_dedup_field(tmp_path: Path) -> None:
    """M2：read_day 返回的记录不含 _dedup 内部字段（但文件本身仍保留以支持去重重建）。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"ev": "x"}, dedup_key="my-key")

    # 文件层仍有 _dedup 字段（供 _ensure_dedup_loaded 回放）
    raw = (tmp_path / "2026-06-11.jsonl").read_text()
    assert "_dedup" in raw, "文件层应保留 _dedup 供去重重建"

    # 读路径应剥离 _dedup
    records = store.read_day(DAY_A)
    assert len(records) == 1
    assert "_dedup" not in records[0], "read_day 返回记录不应含 _dedup 字段"


def test_m2_read_range_strips_dedup_field(tmp_path: Path) -> None:
    """M2：read_range 返回的记录不含 _dedup 内部字段。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"ev": "y"}, dedup_key="rng-key")

    records = store.read_range(DAY_A, DAY_A)
    assert len(records) == 1
    assert "_dedup" not in records[0], "read_range 返回记录不应含 _dedup 字段"


def test_m2_dedup_rebuild_still_works_after_read_strips(tmp_path: Path) -> None:
    """M2：read 剥离 _dedup 不影响去重重建（新实例仍能正确去重）。"""
    store1 = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store1.append({"ev": "orig"}, dedup_key="k-rebuild")

    # read_day 剥离 _dedup（对外投影）
    records = store1.read_day(DAY_A)
    assert "_dedup" not in records[0]

    # 新实例重建去重后，同 key 仍去重
    store2 = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    r = store2.append({"ev": "dup"}, dedup_key="k-rebuild")
    assert r is False, "去重重建不应受 read 投影影响，新实例仍能去重"


# ---------------------------------------------------------------------------
# 坏行容忍
# ---------------------------------------------------------------------------

def test_bad_line_skipped_with_warning(tmp_path: Path, caplog) -> None:
    """read_day 碰到 JSON 坏行应跳过并 warning，返回其余可解析记录。"""
    path = tmp_path / "2026-06-11.jsonl"
    path.write_text(
        '{"ev": "good1"}\n'
        "NOT_JSON_AT_ALL\n"
        '{"ev": "good2"}\n',
        encoding="utf-8",
    )
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    import logging
    with caplog.at_level(logging.WARNING):
        records = store.read_day(DAY_A)
    assert len(records) == 2
    assert records[0]["ev"] == "good1"
    assert records[1]["ev"] == "good2"
    assert any("JSON" in r.message or "json" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# read_range
# ---------------------------------------------------------------------------

def test_read_range_spans_days(tmp_path: Path) -> None:
    """read_range 应合并多日数据。"""
    store_a = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store_b = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_B))
    store_a.append({"day": "a"})
    store_b.append({"day": "b"})

    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    records = store.read_range(DAY_A, DAY_B)
    assert len(records) == 2


def test_read_range_predicate(tmp_path: Path) -> None:
    """predicate 过滤正常工作。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store.append({"kind": "skip"})
    store.append({"kind": "trigger"})

    records = store.read_range(DAY_A, DAY_A, predicate=lambda r: r.get("kind") == "trigger")
    assert len(records) == 1
    assert records[0]["kind"] == "trigger"


def test_read_range_limit(tmp_path: Path) -> None:
    """limit 参数应截断结果。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    for i in range(5):
        store.append({"i": i})
    records = store.read_range(DAY_A, DAY_A, limit=3)
    assert len(records) == 3


def test_read_range_reverse(tmp_path: Path) -> None:
    """reverse=True 应返回倒序（按日期）结果。"""
    store_a = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    store_b = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_B))
    store_a.append({"day": "a"})
    store_b.append({"day": "b"})

    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    records = store.read_range(DAY_A, DAY_B, reverse=True)
    assert records[0]["day"] == "b"
    assert records[1]["day"] == "a"


# ---------------------------------------------------------------------------
# IO 故障桩
# ---------------------------------------------------------------------------

def test_io_error_returns_false_does_not_raise(tmp_path: Path) -> None:
    """monkeypatch open 只在写入阶段抛 OSError → append 返回 False，不抛异常。"""
    store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
    # 用 patch.object 只替换 Path.open，不影响 JsonlDayStore 内部的 dedup 回放逻辑
    with patch("pathlib.Path.open", side_effect=OSError("disk full")):
        result = store.append({"ev": "x"})
    assert result is False


# ---------------------------------------------------------------------------
# Property TSO-7：查询只读
# Feature: task-scheduler-observability, Property TSO-7: 对任意调度日志/任务历史
# 查询调用序列，存储目录的全部文件内容（字节级）不变。
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    events=st.lists(
        st.fixed_dictionaries({"ev": st.text(max_size=20)}),
        min_size=1,
        max_size=20,
    )
)
def test_tso7_read_is_readonly(events: list[dict]) -> None:
    """
    # Feature: task-scheduler-observability, Property TSO-7: read_day / read_range
    # 任意 append 序列后，多次 read_* 调用文件字节不变。
    """
    # 每次 Hypothesis 调用使用独立临时目录（不依赖 function-scoped fixture）
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        store = JsonlDayStore(tmp_path, now_fn=fixed_now(NOW_A))
        for ev in events:
            store.append(ev)

        # 快照写入后的文件内容
        def snapshot() -> dict[str, bytes]:
            return {
                str(p): p.read_bytes()
                for p in tmp_path.rglob("*.jsonl")
            }

        before = snapshot()

        # 多次查询
        store.read_day(DAY_A)
        store.read_range(DAY_A, DAY_B)
        store.read_day(DAY_A)

        after = snapshot()
        assert before == after, "查询操作不应修改文件内容"
