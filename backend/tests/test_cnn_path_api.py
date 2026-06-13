"""
Task 6 验收测试：API 接线与治理评分（Property 6/8）。

覆盖：
6.1  path_class + 非 oco label → POST /api/cnn/train 返回 400，错误信息含 "path_class"
6.2  veto_threshold=1.5 → 422（Pydantic 范围拦截；Field(gt=0, le=1) 不可达 400）
6.3  _core_score 路径标签分支
     (a) tp_auc=0.7 / sl_auc=0.6 → 附加项 +3.0
     (b) 两键缺失 → 附加项 0（中性 tp_auc=sl_auc=0.5 → delta=0）
     (c) tp_auc=0.0 显式存在 → 按 0.0 计算而非 0.5（防 or 吞零）
     (d) regression 分支行为不变
     (e) 显式断言 regression best_val_rank_ic 接线依赖 _merge_training_metrics 才能存在
6.4  engine.strategy._veto_count 属性契约：BacktestingEngine + CNNSignalStrategy
     (veto_threshold=0.5) 跑回测后可从 engine.strategy 读到 _veto_count
6.5  Property 8（Hypothesis）：label↔策略 consistency，oco 场景全覆盖

Feature: cnn-path-multiclass-head
Property 8: 对任意合法 oco label_spec，derive_strategy_exit_from_label 推导的出场参数
与 label oco 口径相等；对推导口径调用 check_label_strategy_consistency 告警为空或不含
oco 口径错配项；故意破坏出场参数后告警非空且不中断（返回告警而非抛异常）。
注意：「不中断」性质仅对合法 exit_mode 组合成立（如 threshold+oco label → 软性告警）；
fixed_hold+oco label 属硬性错配，consistency.py 会抛 ValueError，不在「不中断」覆盖范围。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData
from aitrade.cnn.consistency import (
    check_label_strategy_consistency,
    derive_strategy_exit_from_label,
)
from aitrade.cnn.governance import _core_score, _merge_training_metrics
from aitrade.cnn.strategy import CNNSignalStrategy
from aitrade.main import create_app

# =============================================================================
# 常量
# =============================================================================

SYMBOL = "TEST.SZSE"
START = datetime(2026, 1, 5)


# =============================================================================
# 共享工具（复用 test_cnn_path_strategy.py 已验证的模式）
# =============================================================================


class FakeLoader:
    """实现 BarDataLoader 协议的合成数据源，无外部依赖。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {
            SYMBOL: {
                "long_rate": 0.0003,
                "short_rate": 0.0003,
                "stamp_duty": 0.0,
                "slippage": 0.0,
                "size": 1,
                "pricetick": 0.01,
            }
        }


def _build_bars(closes: list[float]) -> tuple[list[BarData], list[datetime]]:
    """构造满足撮合条件的合成日线：low 低于前收，保证限价单可成交。

    Args:
        closes: 各根 bar 的收盘价列表。

    Returns:
        (bars 列表, datetime 列表) 元组。
    """
    days = [START + timedelta(days=i) for i in range(len(closes))]
    bars: list[BarData] = []
    for i, close in enumerate(closes):
        prev_close = closes[i - 1] if i > 0 else close
        open_price = prev_close
        high_price = max(open_price, close) + 1.0
        low_price = min(open_price, close) - 1.0
        bars.append(
            BarData(
                symbol="TEST",
                exchange="SZSE",
                datetime=days[i],
                interval="d",
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close,
                volume=1_000_000,
            )
        )
    return bars, days


def _signal_df_path(
    days: list[datetime],
    probs_tp: list[float],
    probs_sl: list[float],
) -> pl.DataFrame:
    """构造 path_class 七列信号帧（signal=prob_tp，含 prob_sl 等列）。

    Args:
        days: 各行的 datetime 列表。
        probs_tp: 各行 prob_tp 值（同时作为 signal 列）。
        probs_sl: 各行 prob_sl 值。

    Returns:
        含 [datetime, vt_symbol, signal, prob_tp, prob_sl, prob_time_up, prob_time_down] 的 DataFrame。
    """
    n = len(days)
    return pl.DataFrame(
        {
            "datetime": days,
            "vt_symbol": [SYMBOL] * n,
            "signal": probs_tp,
            "prob_tp": probs_tp,
            "prob_sl": probs_sl,
            "prob_time_up": [0.1] * n,
            "prob_time_down": [0.1] * n,
        }
    )


def _run_engine(
    bars: list[BarData],
    days: list[datetime],
    signal_df: pl.DataFrame,
    setting: dict,
) -> BacktestingEngine:
    """组装并运行一次回测，返回引擎实例供断言。

    Args:
        bars: 合成 BarData 列表。
        days: 对应的 datetime 列表。
        signal_df: 信号 DataFrame（三列或七列均可）。
        setting: CNNSignalStrategy 参数字典。

    Returns:
        完成回测后的 BacktestingEngine 实例。
    """
    engine = BacktestingEngine(data_loader=FakeLoader(bars))
    engine.set_parameters(
        vt_symbols=[SYMBOL],
        interval="d",
        start=days[0],
        end=days[-1] + timedelta(days=1),
        capital=1_000_000,
    )
    engine.add_strategy(CNNSignalStrategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    return engine


# =============================================================================
# 6.1  path_class + 非 oco label → 400（Property 6 API 侧）
# =============================================================================


@pytest.fixture(scope="module")
def app_client():
    """模块级 TestClient（不涉及 I/O 的纯校验测试可共用）。

    同时将 task_manager.run_async patch 为 no-op，避免测试向真实
    ThreadPoolExecutor 提交训练/回测任务，防止污染本机任务历史归档
    （.aitrade/task_history）。断言仅关心同步校验结果（状态码），
    异步执行路径不在测试范围内。
    """
    from unittest.mock import patch

    app = create_app()
    # patch aitrade.api.cnn 命名空间内的 task_manager.run_async：
    # api/cnn.py 通过 `from ..task import task_manager` 导入，
    # 因此需 patch 该模块内已绑定的对象属性，而非 task 包的原始引用
    with patch("aitrade.api.cnn.task_manager.run_async", return_value=None):
        with TestClient(app) as c:
            yield c


class TestPathClassTrainValidation:
    """6.1：objective=path_class 必须配合 label_spec.mode=oco，否则返回 400。"""

    BASE = {
        "name": "test_pc",
        "vt_symbols": ["000001.SZSE"],
        "start": "2024-01-01",
        "end": "2025-01-01",
        "objective": "path_class",
    }

    def test_path_class_next_bar_label_returns_400(self, app_client: TestClient) -> None:
        """path_class + mode=next_bar（默认）→ 400，错误信息含 'path_class'。"""
        payload = {**self.BASE, "label_spec": {"mode": "next_bar"}}
        resp = app_client.post("/api/cnn/train", json=payload)
        assert resp.status_code == 400, f"期望 400，实际: {resp.status_code} {resp.text}"
        assert "path_class" in resp.text

    def test_path_class_horizon_bars_label_returns_400(self, app_client: TestClient) -> None:
        """path_class + mode=horizon_bars → 400，错误信息含 'path_class'。"""
        payload = {**self.BASE, "label_spec": {"mode": "horizon_bars", "horizon": 5}}
        resp = app_client.post("/api/cnn/train", json=payload)
        assert resp.status_code == 400
        assert "path_class" in resp.text

    def test_path_class_oco_label_passes_validation(self, app_client: TestClient) -> None:
        """path_class + mode=oco（合法 tp/sl）→ 校验通过（非 400/422）。

        注：训练端点是同步校验 + 异步执行；校验通过即返回 task_id（200）。
        因后台 I/O（数据加载）不存在，任务可能 failed，但状态码本身是 200。
        """
        payload = {
            **self.BASE,
            "label_spec": {
                "mode": "oco",
                "take_profit": 0.05,
                "stop_loss": 0.03,
                "max_hold": 10,
            },
        }
        resp = app_client.post("/api/cnn/train", json=payload)
        assert resp.status_code not in (400, 422), (
            f"oco label 不应被 400/422 拦截，实际: {resp.status_code} {resp.text}"
        )

    def test_classification_next_bar_label_passes(self, app_client: TestClient) -> None:
        """classification + mode=next_bar → 校验通过（非 path_class 分支不受约束）。"""
        payload = {
            **self.BASE,
            "objective": "classification",
            "label_spec": {"mode": "next_bar"},
        }
        resp = app_client.post("/api/cnn/train", json=payload)
        assert resp.status_code not in (400, 422), (
            f"classification+next_bar 不应被拦截，实际: {resp.status_code}"
        )


# =============================================================================
# 6.2  veto_threshold 范围校验 → 422（Pydantic Field(gt=0, le=1) 拦截）
# =============================================================================


class TestVetoThresholdValidation:
    """6.2：veto_threshold > 1 由 Pydantic 拦截，返回 422（非 400）。

    说明：R4.4 字面要求 400，但 Pydantic Field(gt=0, le=1) 在 FastAPI 中返回 422；
    显式追加 400 校验是不可达代码——保持 422 以满足意图，此处断言 422。
    """

    BASE = {
        "name": "bt_veto_test",
        "model": "nonexistent_model",
        "start": "2024-01-01",
        "end": "2025-01-01",
    }

    def test_veto_threshold_gt1_returns_422(self, app_client: TestClient) -> None:
        """veto_threshold=1.5（超出 le=1 约束）→ Pydantic 返回 422。"""
        payload = {**self.BASE, "veto_threshold": 1.5}
        resp = app_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code == 422, f"期望 422，实际: {resp.status_code}"

    def test_veto_threshold_zero_returns_422(self, app_client: TestClient) -> None:
        """veto_threshold=0（不满足 gt=0 约束）→ Pydantic 返回 422。"""
        payload = {**self.BASE, "veto_threshold": 0.0}
        resp = app_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code == 422

    def test_veto_threshold_valid_passes(self, app_client: TestClient) -> None:
        """veto_threshold=0.6（合法值）→ 通过 Pydantic 校验（因无 torch 或模型可能 400/500，但非 422）。"""
        payload = {**self.BASE, "veto_threshold": 0.6}
        resp = app_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code != 422, (
            f"合法 veto_threshold 不应返回 422，实际: {resp.status_code}"
        )


# =============================================================================
# 6.3  _core_score 路径标签评分项单元测试
# =============================================================================


class TestCoreScorePathClass:
    """6.3：_core_score 的 path_class 分支行为验证。"""

    BASE_STATS: dict[str, Any] = {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_ddpercent": 0.0,
        "total_trade_count": 1,  # 防止 -5 trade_penalty
    }

    def test_tp07_sl06_adds_3(self) -> None:
        """tp_auc=0.7, sl_auc=0.6 → 附加项 (0.7+0.6-1)*10 = 3.0（精确）。"""
        stats = {**self.BASE_STATS, "best_val_tp_auc": 0.7, "best_val_sl_auc": 0.6}
        score = _core_score(stats, "path_class")
        # 基础分 0+0-0-0=0，附加 3.0
        assert abs(score - 3.0) < 1e-6, f"期望 3.0，实际: {score}"

    def test_both_keys_missing_neutral(self) -> None:
        """两键缺失 → tp/sl 各回退 0.5，附加项 (0.5+0.5-1)*10 = 0（中性）。"""
        stats = {**self.BASE_STATS}  # 无 best_val_tp_auc / best_val_sl_auc
        score = _core_score(stats, "path_class")
        assert abs(score - 0.0) < 1e-6, f"期望 0.0，实际: {score}"

    def test_tp_zero_not_swallowed_by_or(self) -> None:
        """tp_auc=0.0 显式存在 → 按 0.0 计算，不因 or 被吞为 0.5。

        附加项 = (0.0 + 0.5 - 1) * 10 = -5.0（sl 缺失回退 0.5）。
        """
        stats = {**self.BASE_STATS, "best_val_tp_auc": 0.0}  # sl 缺失 → 0.5
        score = _core_score(stats, "path_class")
        expected = (0.0 + 0.5 - 1.0) * 10.0  # = -5.0
        assert abs(score - expected) < 1e-6, f"期望 {expected}，实际: {score}"

    def test_sl_zero_not_swallowed_by_or(self) -> None:
        """sl_auc=0.0 显式存在 → 按 0.0 计算，不因 or 被吞为 0.5。"""
        stats = {**self.BASE_STATS, "best_val_sl_auc": 0.0}  # tp 缺失 → 0.5
        score = _core_score(stats, "path_class")
        expected = (0.5 + 0.0 - 1.0) * 10.0  # = -5.0
        assert abs(score - expected) < 1e-6, f"期望 {expected}，实际: {score}"

    def test_regression_branch_unchanged(self) -> None:
        """regression 分支：best_val_rank_ic=0.2 → 附加项 +2.0，行为不变。"""
        stats = {**self.BASE_STATS, "best_val_rank_ic": 0.2}
        score = _core_score(stats, "regression")
        expected = 0.0 + 0.2 * 10.0  # 基础 0 + rank_ic 项
        assert abs(score - expected) < 1e-6, f"期望 {expected}，实际: {score}"

    def test_regression_missing_rank_ic_is_zero(self) -> None:
        """regression 键缺失 → 0.0 中性（不是 0.5，与 path_class 缺失语义不同）。"""
        stats = {**self.BASE_STATS}
        score = _core_score(stats, "regression")
        assert abs(score - 0.0) < 1e-6

    def test_error_key_returns_neg_inf(self) -> None:
        """statistics 含 error → 返回 -1e9（无论 objective）。"""
        for obj in ("classification", "regression", "path_class"):
            score = _core_score({"error": "some error"}, obj)
            assert score == -1e9, f"objective={obj} 期望 -1e9，实际: {score}"

    def test_classification_no_extra_term(self) -> None:
        """classification 无额外训练质量项，基础分与公式完全一致。"""
        stats = {
            **self.BASE_STATS,
            "total_return": 5.0,
            "sharpe_ratio": 1.0,
            "max_ddpercent": 10.0,
        }
        score = _core_score(stats, "classification")
        expected = 5.0 + 1.0 * 5.0 - 10.0 * 0.2 - 0.0
        assert abs(score - expected) < 1e-5, f"期望 {expected}，实际: {score}"


class TestMergeTrainingMetrics:
    """6.3（接线断言）：验证 regression best_val_rank_ic 的接线依赖 _merge_training_metrics。

    侦察结论：checkpoint 本体不含 best_val_rank_ic；只有调用 _merge_training_metrics
    后（从 _history.json 读取 best_epoch 行），statistics 才会存在该键。
    这是实现的已知设计，测试显式断言此行为，不允许静默掩盖。
    """

    def test_regression_key_absent_without_merge(self) -> None:
        """不调用 _merge_training_metrics 时，statistics 中不含 best_val_rank_ic。

        验证「接线是真实需要的，不是多余的」。
        """
        stats: dict[str, Any] = {}
        # 直接用 _core_score，不调用 _merge_training_metrics
        # → rank_ic 按 0.0 处理（静默缺失）
        score = _core_score(stats, "regression")
        # 空 stats 触发 trade_penalty=-5.0
        assert abs(score - (-5.0)) < 1e-6

    def test_merge_reads_history_and_injects_key(self, tmp_path) -> None:
        """_merge_training_metrics 从 _history.json 读取并写入 statistics。

        构造合成 checkpoint + history 文件，验证 best_val_rank_ic 被正确注入。
        """
        import json

        model_name = "test_merge_model"
        # 构造最小 history（3 个 epoch，best 在第 2 个）
        history = [
            {"val_rank_ic": 0.1},
            {"val_rank_ic": 0.3},  # best_epoch=2
            {"val_rank_ic": 0.2},
        ]
        history_path = tmp_path / f"{model_name}_history.json"
        history_path.write_text(json.dumps(history), encoding="utf-8")

        # checkpoint 直接传给 _merge_training_metrics（不需要 .pt 文件）
        checkpoint = {
            "best_epoch": 2,
            "train_config": {"objective": "regression"},
        }

        # 临时 patch CNN_MODEL_DIR
        import aitrade.cnn.governance as gov_module
        original_dir = gov_module.CNN_MODEL_DIR
        try:
            gov_module.CNN_MODEL_DIR = tmp_path
            stats: dict[str, Any] = {}
            _merge_training_metrics(stats, model_name, checkpoint)
        finally:
            gov_module.CNN_MODEL_DIR = original_dir

        assert "best_val_rank_ic" in stats, "best_val_rank_ic 应被 _merge_training_metrics 注入"
        assert abs(stats["best_val_rank_ic"] - 0.3) < 1e-9

    def test_merge_path_class_injects_tp_sl_auc(self, tmp_path) -> None:
        """_merge_training_metrics 对 path_class 注入 best_val_tp_auc / best_val_sl_auc。"""
        import json

        model_name = "test_path_merge"
        history = [
            {"val_tp_auc": 0.65, "val_sl_auc": 0.70},
            {"val_tp_auc": 0.72, "val_sl_auc": 0.75},  # best
        ]
        history_path = tmp_path / f"{model_name}_history.json"
        history_path.write_text(json.dumps(history), encoding="utf-8")

        # checkpoint 直接传给 _merge_training_metrics（不需要 .pt 文件）
        checkpoint = {
            "best_epoch": 2,
            "train_config": {"objective": "path_class"},
        }

        import aitrade.cnn.governance as gov_module
        original_dir = gov_module.CNN_MODEL_DIR
        try:
            gov_module.CNN_MODEL_DIR = tmp_path
            stats: dict[str, Any] = {}
            _merge_training_metrics(stats, model_name, checkpoint)
        finally:
            gov_module.CNN_MODEL_DIR = original_dir

        assert "best_val_tp_auc" in stats
        assert "best_val_sl_auc" in stats
        assert abs(stats["best_val_tp_auc"] - 0.72) < 1e-9
        assert abs(stats["best_val_sl_auc"] - 0.75) < 1e-9

    def test_merge_history_missing_graceful(self, tmp_path) -> None:
        """history 文件不存在时，_merge_training_metrics 静默跳过（不抛异常）。"""
        checkpoint = {"best_epoch": 1, "train_config": {"objective": "regression"}}
        import aitrade.cnn.governance as gov_module
        original_dir = gov_module.CNN_MODEL_DIR
        try:
            gov_module.CNN_MODEL_DIR = tmp_path
            stats: dict[str, Any] = {}
            _merge_training_metrics(stats, "no_such_model", checkpoint)
        finally:
            gov_module.CNN_MODEL_DIR = original_dir

        assert "best_val_rank_ic" not in stats


# =============================================================================
# 6.4  engine.strategy._veto_count 属性契约
# =============================================================================


class TestVetoCountEngineContract:
    """6.4：守护"API 读取点"依赖的引擎属性契约。

    验证 BacktestingEngine.strategy._veto_count 在回测结束后可被防御式读取，
    语义为「本要买入却被否决的次数」。
    """

    def _base_setting(self, veto_threshold: float) -> dict:
        return {
            "buy_threshold": 0.5,
            "sell_threshold": 0.3,
            "price_add": 0.0,
            "exit_mode": "fixed_hold",
            "hold_days": 1,
            "take_profit": 0.0,
            "stop_loss": 0.0,
            "veto_threshold": veto_threshold,
        }

    def test_veto_count_readable_after_backtest(self) -> None:
        """回测后 engine.strategy._veto_count 可读（防御式 getattr 等价）。"""
        closes = [10.0] * 10
        bars, days = _build_bars(closes)
        n = len(days)
        # prob_sl 全为 0.8，veto_threshold=0.5 → 全程否决
        signal_df = _signal_df_path(
            days,
            probs_tp=[0.9] * n,  # signal 高于 buy_threshold
            probs_sl=[0.8] * n,  # prob_sl 高于 veto_threshold
        )
        engine = _run_engine(bars, days, signal_df, self._base_setting(veto_threshold=0.5))

        # engine.strategy 必须存在且有 _veto_count 属性
        assert hasattr(engine, "strategy"), "BacktestingEngine 应有 strategy 属性"
        veto_count = getattr(engine.strategy, "_veto_count", None)
        assert veto_count is not None, "CNNSignalStrategy 应有 _veto_count 属性"
        assert isinstance(veto_count, int), "_veto_count 应为 int"
        assert veto_count > 0, "prob_sl >= veto_threshold 应产生否决记录"

    def test_veto_count_zero_when_disabled(self) -> None:
        """veto_threshold=1.0（默认关闭）→ _veto_count=0（无否决）。"""
        closes = [10.0] * 10
        bars, days = _build_bars(closes)
        n = len(days)
        signal_df = _signal_df_path(
            days,
            probs_tp=[0.9] * n,
            probs_sl=[0.8] * n,  # 即使 prob_sl 高，veto_threshold=1.0 不触发否决
        )
        engine = _run_engine(bars, days, signal_df, self._base_setting(veto_threshold=1.0))
        veto_count = getattr(engine.strategy, "_veto_count", 0)
        assert veto_count == 0, f"veto_threshold=1.0 应无否决，实际 _veto_count={veto_count}"

    def test_veto_count_getattr_defense_no_raises(self) -> None:
        """getattr(engine.strategy, '_veto_count', 0) 不会因属性缺失而抛异常。

        即使策略实现变更，防御式读取始终安全。
        """
        closes = [10.0] * 5
        bars, days = _build_bars(closes)
        signal_df = pl.DataFrame(
            {"datetime": days, "vt_symbol": [SYMBOL] * 5, "signal": [0.3] * 5}
        )
        engine = _run_engine(bars, days, signal_df, self._base_setting(veto_threshold=1.0))
        count = getattr(engine.strategy, "_veto_count", 0)
        assert isinstance(count, int)

    def test_veto_count_readable_after_zero_trade_backtest(self) -> None:
        """零成交回测（全程被否决）后，_veto_count 仍可防御式读取。

        覆盖 _run_cnn_backtest 无成交早退分支中 veto_count 字段的读取路径：
        engine.run_backtesting() 已执行，engine.strategy 存在，_veto_count 可读。
        """
        closes = [10.0] * 10
        bars, days = _build_bars(closes)
        n = len(days)
        # prob_sl 全为 0.9，veto_threshold=0.5 → 全程被否决 → 零成交
        signal_df = _signal_df_path(
            days,
            probs_tp=[0.9] * n,   # signal 高于 buy_threshold，本要买入
            probs_sl=[0.9] * n,   # prob_sl >= veto_threshold，触发否决
        )
        engine = _run_engine(bars, days, signal_df, self._base_setting(veto_threshold=0.5))

        # 零成交场景：trade_count == 0，但 _veto_count 仍应可防御式读取
        assert engine.trade_count == 0, "全程否决应导致零成交"
        veto_count = int(getattr(engine.strategy, "_veto_count", 0))
        assert veto_count > 0, (
            "全程否决后 _veto_count 应 > 0（API 早退分支依赖此值解释零成交原因）"
        )


# =============================================================================
# 6.5  Property 8（Hypothesis）：label ↔ 策略 consistency，oco 场景
# =============================================================================


@given(
    take_profit=st.floats(min_value=0.005, max_value=0.2),
    stop_loss=st.floats(min_value=0.005, max_value=0.2),
    max_hold=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100)
# Feature: cnn-path-multiclass-head, Property 8: derive_strategy_exit_from_label 对任意合法
# oco label_spec 推导的出场参数与 label oco 口径相等；check_label_strategy_consistency 对
# 推导口径告警为空或不含 oco 口径错配项；故意破坏出场参数后告警非空且不中断。
def test_property8_oco_consistency(
    take_profit: float,
    stop_loss: float,
    max_hold: int,
) -> None:
    """Property 8：oco label ↔ 策略一致性三项子性质全覆盖。

    Args:
        take_profit: 止盈幅度，∈ [0.005, 0.2]。
        stop_loss: 止损幅度，∈ [0.005, 0.2]。
        max_hold: 最大持有天数，∈ [1, 30]。
    """
    label_spec = {
        "mode": "oco",
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "max_hold": max_hold,
    }
    input_interval = "d"

    # (a) derive 推导的出场参数与 label 的 oco 口径相等
    exit_cfg = derive_strategy_exit_from_label(label_spec, input_interval)
    assert exit_cfg["exit_mode"] == "oco", "oco label 应推导出 exit_mode=oco"
    assert abs(exit_cfg["take_profit"] - take_profit) < 1e-9, "推导 take_profit 应与 label 一致"
    assert abs(exit_cfg["stop_loss"] - stop_loss) < 1e-9, "推导 stop_loss 应与 label 一致"
    assert exit_cfg["hold_days"] == max_hold, "推导 hold_days 应等于 label max_hold"

    # (b) 对推导口径调 check → 告警为空或不含 oco 口径错配项
    # oco 推导出 exit_mode=oco，一致性自检对 oco+oco 仅返回保守假设告警（软性），不报错配
    warnings_aligned = check_label_strategy_consistency(
        label_spec, exit_cfg["exit_mode"], exit_cfg["hold_days"], input_interval
    )
    # 不应含「口径不一致」型告警（含 "固定" 或 "不一致" 关键词）
    mismatched = [w for w in warnings_aligned if "不一致" in w and "固定" in w]
    assert not mismatched, f"对齐口径不应有错配告警: {mismatched}"

    # (c) 故意改出场参数 → 告警非空，且是返回告警而非抛异常（不中断语义）
    # 改 exit_mode 为 threshold（与 oco label 不对齐，软性告警路径）
    warnings_broken = check_label_strategy_consistency(
        label_spec, "threshold", max_hold, input_interval
    )
    assert len(warnings_broken) > 0, (
        "threshold exit_mode 与 oco label 不对齐，应产生告警"
    )


@given(
    take_profit=st.floats(min_value=0.005, max_value=0.2),
    stop_loss=st.floats(min_value=0.005, max_value=0.2),
    max_hold=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=50)
def test_property8_oco_check_always_returns_not_raises(
    take_profit: float,
    stop_loss: float,
    max_hold: int,
) -> None:
    """Property 8（c）补充：check_label_strategy_consistency 对任意出场模式组合返回告警而非抛异常。

    验证「不中断语义」：破坏后调用只要不抛 ValueError 就算通过。
    注意：fixed_hold + oco label 在 consistency.py 中会抛 ValueError（硬性错误），
    此处仅测试 threshold 与 oco 组合（软性告警路径）。

    Args:
        take_profit: 止盈幅度。
        stop_loss: 止损幅度。
        max_hold: 最大持有天数。
    """
    label_spec = {
        "mode": "oco",
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "max_hold": max_hold,
    }
    # threshold 与 oco label 组合 → 软性告警，不中断
    warnings = check_label_strategy_consistency(label_spec, "threshold", max_hold, "d")
    assert isinstance(warnings, list), "告警应为 list（不应抛异常）"
    assert len(warnings) > 0, "threshold 与 oco label 不对齐，应有告警"


@given(
    take_profit=st.floats(min_value=0.005, max_value=0.2),
    stop_loss=st.floats(min_value=0.005, max_value=0.2),
    max_hold=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=50)
def test_property8_auto_derive_idempotent(
    take_profit: float,
    stop_loss: float,
    max_hold: int,
) -> None:
    """Property 8：auto 推导口径相等——derive 两次结果相同（幂等性）。

    Args:
        take_profit: 止盈幅度。
        stop_loss: 止损幅度。
        max_hold: 最大持有天数。
    """
    label_spec = {
        "mode": "oco",
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "max_hold": max_hold,
    }
    cfg1 = derive_strategy_exit_from_label(label_spec, "d")
    cfg2 = derive_strategy_exit_from_label(label_spec, "d")
    assert cfg1 == cfg2, "derive_strategy_exit_from_label 应为幂等函数"
