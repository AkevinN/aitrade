"""run_walk_forward_evaluate 的 governance store 注入向后兼容测试。

# Feature: cnn-stock-screening, Property 9: governance store 注入向后兼容

覆盖 Task 2.1/2.2：给 ``run_walk_forward_evaluate`` 新增可选 ``store`` 参数后，
- 不传 ``store`` 时仍解析/落盘到模块级全局生产 store（指向 CNN_GOVERNANCE_PATH）；
- 注入 ``store`` 时读写全部路由到该隔离 store，绝不触及全局生产 store。

为避免真实多折训练（慢），统一 monkeypatch 重型内部 ``_train_governance_model`` /
``_backtest_model``，使函数体（窗口生成、门禁、报告装配、落盘）在合成桩下快速跑通；
全局生产 store 亦 monkeypatch 到 ``tmp_path``，避免污染真实生产治理目录。
"""

from __future__ import annotations

from datetime import date

import pytest

from aitrade.cnn import governance as gov
from aitrade.cnn.governance import CNNGovernanceStore, run_walk_forward_evaluate
from aitrade.models.governance import CNNWalkForwardRequest


def _make_request() -> CNNWalkForwardRequest:
    """构造一个能生成单个 walk-forward 窗口的最小评估请求。

    Returns:
        train_days/test_days 之和恰好覆盖 [start, end] 区间、可生成 ≥1 个窗口的请求。
    """
    return CNNWalkForwardRequest(
        name="bc_test",
        target_symbol="000001.SZSE",
        start=date(2023, 1, 1),
        end=date(2025, 6, 1),
        train_days=720,
        test_days=90,
        n_seeds=1,
    )


@pytest.fixture
def fast_internals(monkeypatch):
    """桩化重型训练/回测内部，使 WF 评估秒级跑通且不触盘训练产物。

    - ``_train_governance_model`` 返回带 name 的假结果（不训真模型）。
    - ``_backtest_model`` 返回带固定 statistics 的假回测（让 _core_score 可计算）。
    """

    def fake_train(req, *, model_name, start, end, seed_index=0, on_progress=None):
        return {"name": model_name}

    def fake_backtest(*, model_name, name, start, end, capital, params):
        # 提供 _core_score(classification) 需要的字段，使候选得分可计算且为正。
        return {"statistics": {"total_return": 5.0, "sharpe_ratio": 1.2, "win_rate": 0.6}}

    monkeypatch.setattr(gov, "_train_governance_model", fake_train)
    monkeypatch.setattr(gov, "_backtest_model", fake_backtest)


def test_store_resolution_helper_logic():
    """store 解析逻辑：未注入回落全局，注入则用注入对象（不跑评估，直测语义）。

    # Feature: cnn-stock-screening, Property 9: governance store 注入向后兼容
    """
    # 全局别名与模块级生产 store 必须是同一对象（避免形参遮蔽后引用错对象）。
    assert gov._GLOBAL_STORE is gov.store
    # 复刻函数顶部的解析表达式，验证其分支语义。
    injected = CNNGovernanceStore  # 占位仅作非 None 标记
    resolved_default = None if None is not None else gov._GLOBAL_STORE
    resolved_injected = injected if injected is not None else gov._GLOBAL_STORE
    assert resolved_default is gov._GLOBAL_STORE
    assert resolved_injected is injected


def test_default_store_writes_to_global_production_store(tmp_path, monkeypatch, fast_internals):
    """不传 store：报告/历史落到模块级全局生产 store（CNN_GOVERNANCE_PATH 语义）。

    # Feature: cnn-stock-screening, Property 9: governance store 注入向后兼容
    """
    # 把全局生产 store 重定向到 tmp_path，避免污染真实生产治理目录；
    # _GLOBAL_STORE 是函数内部默认回落引用，必须一并指向同一隔离 store。
    global_store = CNNGovernanceStore(tmp_path / "global_gov")
    monkeypatch.setattr(gov, "store", global_store)
    monkeypatch.setattr(gov, "_GLOBAL_STORE", global_store)

    report = run_walk_forward_evaluate(_make_request())  # 不传 store

    report_id = report["report_id"]
    # 报告与历史确实写进了全局生产 store。
    assert global_store.report_path(report_id).exists()
    assert global_store.get_report(report_id) is not None
    history = global_store.history()
    assert history[-1]["event_type"] == "wf_evaluate_completed"
    assert history[-1]["payload"]["report_id"] == report_id
    # 返回结构保持不变（向后兼容关键字段齐全）。
    assert report["type"] == "walk_forward"
    assert "summary" in report and "passed" in report["summary"]
    assert report["summary"]["fold_count"] >= 1


def test_injected_store_routes_reads_and_writes_to_it_not_global(tmp_path, monkeypatch, fast_internals):
    """注入 store：读写全部路由到注入 store，全局生产 store 零写入。

    # Feature: cnn-stock-screening, Property 9: governance store 注入向后兼容
    """
    # 全局生产 store（重定向到隔离 tmp 目录，纯粹用于断言"未被写入"）。
    global_store = CNNGovernanceStore(tmp_path / "global_gov")
    monkeypatch.setattr(gov, "store", global_store)
    monkeypatch.setattr(gov, "_GLOBAL_STORE", global_store)
    # 选股隔离 store（模拟 SCREENING_GOVERNANCE_PATH）。
    screening_store = CNNGovernanceStore(tmp_path / "screening_gov")

    # 用埋点验证 get_production 来自注入 store 而非全局。
    global_get_calls: list[int] = []
    screening_get_calls: list[int] = []
    orig_global_get = global_store.get_production
    orig_screening_get = screening_store.get_production
    monkeypatch.setattr(
        global_store, "get_production", lambda: (global_get_calls.append(1), orig_global_get())[1]
    )
    monkeypatch.setattr(
        screening_store, "get_production", lambda: (screening_get_calls.append(1), orig_screening_get())[1]
    )

    report = run_walk_forward_evaluate(_make_request(), store=screening_store)
    report_id = report["report_id"]

    # 写：报告/历史落在注入 store。
    assert screening_store.report_path(report_id).exists()
    assert screening_store.get_report(report_id) is not None
    assert screening_store.history()[-1]["event_type"] == "wf_evaluate_completed"
    # 读：get_production 命中注入 store，未命中全局。
    assert screening_get_calls == [1]
    assert global_get_calls == []
    # 全局生产 store 完全没有该报告，且历史为空。
    assert not global_store.report_path(report_id).exists()
    assert global_store.history() == []
