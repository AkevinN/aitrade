"""
CNN 选股 Tier-2 按标的进程并行（E）的编排与等价性测试。

覆盖 cnn-screening-tier2-speedup 特性：
- worker `_evaluate_tier2_symbol`：成功 → 派生 verdict；异常 → "Tier-2 失败: " 失败 verdict（Property 4）。
- `_run_tier2_tasks`：并行(mw>1) 与串行(mw=1) 产出逐字段一致、结果按 symbol 键装配
  与完成顺序无关（Property 3）；单只失败隔离（Property 4）；mw<=1 不创建执行器（Property 5）。

测试策略：`_make_executor` monkeypatch 成 ThreadPoolExecutor 在**进程内**跑，使 stub
生效（生产用 ProcessPoolExecutor，spawn 子进程内 monkeypatch 不生效，无法廉价桩化真实
训练——真实多进程提速属手动验证）。`run_walk_forward_evaluate`/`derive_edge`/
`build_screening_governance_store` 全部桩化，不触真实训练/磁盘。

Feature: cnn-screening-tier2-speedup
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.models.governance import CNNWalkForwardRequest
from aitrade.screening import runner as runner_mod
from aitrade.screening.rules import ScreeningRules
from aitrade.screening.types import Tier2Verdict

_RULES = ScreeningRules()


def _wf_req(sym: str) -> CNNWalkForwardRequest:
    """构造一个最小 WF 请求，target_symbol 用于桩化路由与断言。"""
    return CNNWalkForwardRequest(
        name=f"t_{sym}",
        target_symbol=sym,
        start=date(2024, 1, 1),
        end=date(2025, 1, 1),
        train_days=480,
        test_days=90,
        step_days=90,
        n_seeds=1,
    )


def _fake_report(wf_req: CNNWalkForwardRequest, on_progress: Any = None, store: Any = None) -> dict:
    """桩 run_walk_forward_evaluate：返回带 target_symbol 的合成报告（验证路由不串号）。"""
    return {
        "report_id": f"r_{wf_req.target_symbol}",
        "request": {"target_symbol": wf_req.target_symbol},
        "folds": [{"candidate_score": 1.0}],
        "summary": {},
    }


def _fake_derive(report: dict, rules: ScreeningRules) -> Tier2Verdict:
    """桩 derive_edge：把报告的 target_symbol 原样带进 verdict，便于断言路由正确。"""
    sym = report["request"]["target_symbol"]
    return Tier2Verdict(vt_symbol=sym, evaluable=True, edge_ok=True, avg_score=1.0)


def _patch_common(monkeypatch, wf_fn=_fake_report) -> None:
    """桩化 worker 依赖的三处：store 构造、WF 评估、edge 派生。"""
    monkeypatch.setattr(runner_mod, "build_screening_governance_store", lambda: object())
    monkeypatch.setattr(runner_mod, "run_walk_forward_evaluate", wf_fn)
    monkeypatch.setattr(runner_mod, "derive_edge", _fake_derive)


def _patch_thread_executor(monkeypatch) -> list[int]:
    """把 _make_executor 换成 ThreadPoolExecutor（进程内，stub 生效）；返回调用记录列表。"""
    calls: list[int] = []

    def _factory(mw: int):
        calls.append(mw)
        return ThreadPoolExecutor(max_workers=mw)

    monkeypatch.setattr(runner_mod, "_make_executor", _factory)
    return calls


# ---------------------------------------------------------------------------
# worker 单测（Property 4）
# ---------------------------------------------------------------------------


def test_worker_success(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 4（成功路径）
    """worker 成功时返回 derive_edge 派生的 verdict，且 symbol 路由正确。"""
    _patch_common(monkeypatch)
    v = runner_mod._evaluate_tier2_symbol("000001.SZSE", _wf_req("000001.SZSE"), _RULES)
    assert v.evaluable is True
    assert v.vt_symbol == "000001.SZSE"


def test_worker_exception_degrades(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 4（异常降级）
    """worker 内 WF 评估抛异常时返回 evaluable=False 且 note 以 'Tier-2 失败: ' 开头。"""

    def _boom(*a: Any, **k: Any) -> dict:
        raise RuntimeError("explode")

    _patch_common(monkeypatch, wf_fn=_boom)
    v = runner_mod._evaluate_tier2_symbol("000002.SZSE", _wf_req("000002.SZSE"), _RULES)
    assert v.evaluable is False
    assert v.edge_ok is False
    assert v.note is not None and v.note.startswith("Tier-2 失败: ")


# ---------------------------------------------------------------------------
# _run_tier2_tasks 编排（Property 3 / 4 / 5）
# ---------------------------------------------------------------------------


def test_parallel_equals_serial(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 3: 并行 = 串行
    """同一批任务，mw=4 并行与 mw=1 串行产出逐字段一致，且 verdict 按自身 symbol 装配。"""
    _patch_common(monkeypatch)
    _patch_thread_executor(monkeypatch)
    syms = [f"00000{i}.SZSE" for i in range(1, 6)]
    tasks = [(s, _wf_req(s)) for s in syms]

    serial = runner_mod._run_tier2_tasks(list(tasks), _RULES, max_workers=1)
    parallel = runner_mod._run_tier2_tasks(list(tasks), _RULES, max_workers=4)

    assert set(serial) == set(syms)
    assert set(parallel) == set(syms)
    # 路由正确：每个 key 的 verdict 属于该 key 自己（并行不串号）。
    for k, v in parallel.items():
        assert v.vt_symbol == k
    # 逐字段一致。
    assert {k: v.model_dump() for k, v in serial.items()} == {
        k: v.model_dump() for k, v in parallel.items()
    }


def test_one_failure_isolated(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 4: 单只失败隔离
    """某只标的 WF 抛异常时仅该只降级，其余标的正常评估。"""

    def _wf(wf_req: CNNWalkForwardRequest, on_progress: Any = None, store: Any = None) -> dict:
        if wf_req.target_symbol == "000003.SZSE":
            raise RuntimeError("boom")
        return _fake_report(wf_req)

    _patch_common(monkeypatch, wf_fn=_wf)
    _patch_thread_executor(monkeypatch)
    syms = [f"00000{i}.SZSE" for i in range(1, 6)]
    tasks = [(s, _wf_req(s)) for s in syms]

    res = runner_mod._run_tier2_tasks(tasks, _RULES, max_workers=4)

    assert res["000003.SZSE"].evaluable is False
    assert res["000003.SZSE"].note.startswith("Tier-2 失败: ")
    for s in syms:
        if s != "000003.SZSE":
            assert res[s].evaluable is True, f"{s} 不应受其他标的失败影响"


def test_serial_path_does_not_create_executor(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 5: 串行回退等价
    """mw<=1 时走 inline 串行，不创建任何执行器。"""
    _patch_common(monkeypatch)
    calls = _patch_thread_executor(monkeypatch)
    syms = [f"00000{i}.SZSE" for i in range(1, 4)]
    tasks = [(s, _wf_req(s)) for s in syms]

    res = runner_mod._run_tier2_tasks(tasks, _RULES, max_workers=1)

    assert set(res) == set(syms)
    assert calls == [], "串行路径不应调用 _make_executor"


def test_single_task_runs_serial(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 5: 单任务走串行
    """任务数=1 时即便 mw>1 也走串行（不创建执行器）。"""
    _patch_common(monkeypatch)
    calls = _patch_thread_executor(monkeypatch)
    res = runner_mod._run_tier2_tasks([("000001.SZSE", _wf_req("000001.SZSE"))], _RULES, max_workers=8)
    assert set(res) == {"000001.SZSE"}
    assert calls == []


# ---------------------------------------------------------------------------
# 属性测试：任意标的数 × 任意失败子集，并行 = 串行（Property 3 / 4）
# ---------------------------------------------------------------------------


def test_property_parallel_equals_serial_any_failures(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 3/4: 任意失败子集下并行=串行
    """随机标的数与失败子集下，mw=4 与 mw=1 产出逐字段一致（结果与顺序/失败无关）。"""
    state: dict[str, set[str]] = {"fail": set()}

    def _wf(wf_req: CNNWalkForwardRequest, on_progress: Any = None, store: Any = None) -> dict:
        if wf_req.target_symbol in state["fail"]:
            raise RuntimeError("boom")
        return _fake_report(wf_req)

    _patch_common(monkeypatch, wf_fn=_wf)
    _patch_thread_executor(monkeypatch)

    @settings(max_examples=100, deadline=None)
    @given(
        n=st.integers(min_value=1, max_value=8),
        fail_idx=st.sets(st.integers(min_value=0, max_value=7)),
    )
    def _check(n: int, fail_idx: set[int]) -> None:
        syms = [f"sym{i:02d}.SZSE" for i in range(n)]
        state["fail"] = {syms[i] for i in fail_idx if i < n}
        tasks = [(s, _wf_req(s)) for s in syms]

        serial = runner_mod._run_tier2_tasks(list(tasks), _RULES, max_workers=1)
        parallel = runner_mod._run_tier2_tasks(list(tasks), _RULES, max_workers=4)

        assert set(serial) == set(syms) == set(parallel)
        for k, v in parallel.items():
            assert v.vt_symbol == k  # 路由正确，不串号
        assert {k: v.model_dump() for k, v in serial.items()} == {
            k: v.model_dump() for k, v in parallel.items()
        }

    _check()
