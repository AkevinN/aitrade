"""
任务4 验收测试：信号帧末列恒含 objective 列（cnn-eval-honesty-fixes）。

覆盖范围：
  4.2  示例测试：三种 objective 输出帧含 objective 列且全行同值 == checkpoint objective；
         signal/prob_* 列不变；objective 在末尾。
  4.3  Property 4（Hypothesis）：自描述 + legacy 帧无该列时消费方行为不变。

Feature: cnn-eval-honesty-fixes
Property 4: 对任意 objective 的 CNN 推理输出，信号帧含 objective 列、全行同值且等于
checkpoint objective；既有列（signal/prob_*）不变；不含该列的 legacy 帧经所有消费方
处理行为不变。
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.cnn import model as cnn_model
from aitrade.cnn import storage as cnn_storage

# ---------------------------------------------------------------------------
# 共享常量（与 test_cnn_path_predictor.py 保持一致）
# ---------------------------------------------------------------------------

TARGET = "000001.SZSE"
LOOKBACK = 10
FEATURE_CHANNELS = 6
MAX_GROUP_WIDTH = 1
GROUP_COUNT = 1
FRAME_ROWS = 60

GROUPS = [{"role": "target", "name": "目标", "symbols": [TARGET]}]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _make_trading_frame(rows: int = FRAME_ROWS, seed: int = 0) -> pl.DataFrame:
    """构造连续工作日合成日线行情帧。

    行情列与 features.py 特征计算所需的 OHLCV + turnover + open_interest 保持一致。
    datetime 为工作日序列（跳过周六日），便于 predictor 的区间过滤正常工作。

    Args:
        rows: 行数；默认 60（足够 lookback=10 + warm-up 裕量）。
        seed: numpy 随机种子。

    Returns:
        polars DataFrame，列：datetime, open, high, low, close, volume, turnover, open_interest。
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    dt = datetime(2024, 1, 2)
    px = 100.0
    for _ in range(rows):
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        px *= 1 + rng.normal(0, 0.01)
        records.append({
            "datetime": dt,
            "open": float(px * 0.998),
            "high": float(px * 1.015),
            "low": float(px * 0.985),
            "close": float(px),
            "volume": float(1000 + rng.integers(0, 100)),
            "turnover": float(px * (1000 + rng.integers(0, 100))),
            "open_interest": 0.0,
        })
        dt += timedelta(days=1)
    return pl.DataFrame(records)


def _make_checkpoint(
    tmp_path,
    objective: str = "classification",
) -> str:
    """直接构造最小 checkpoint 并落盘，返回模型名称。

    不走 train_cnn_model，速度快，供示例测试与属性测试使用。

    Args:
        tmp_path: pytest tmp_path fixture，用作 CNN_MODEL_DIR。
        objective: 训练目标，"classification" / "regression" / "path_class"。

    Returns:
        落盘的模型名（不含 .pt 后缀）。
    """
    import torch
    from aitrade.cnn.network import create_market_cnn

    model = create_market_cnn(
        FEATURE_CHANNELS, LOOKBACK, MAX_GROUP_WIDTH, GROUP_COUNT,
        dropout=0.0, objective=objective,
    )
    model_config: dict[str, Any] = {
        "in_channels": FEATURE_CHANNELS,
        "time_steps": LOOKBACK,
        "max_group_width": MAX_GROUP_WIDTH,
        "group_count": GROUP_COUNT,
        "dropout": 0.0,
    }
    if objective == "path_class":
        model_config["num_classes"] = 4

    save_data = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "train_config": {
            "target_symbol": TARGET,
            "lookback": LOOKBACK,
            "input_data_kind": "bar",
            "input_interval": "d",
            "objective": objective,
            "observation_groups": GROUPS,
        },
        "normalization": {
            "channel_mean": [0.0] * FEATURE_CHANNELS,
            "channel_std": [1.0] * FEATURE_CHANNELS,
        },
    }
    name = f"obj_{objective}"
    torch.save(save_data, str(tmp_path / f"{name}.pt"))
    return name


def _run_predict(monkeypatch, tmp_path, objective: str) -> pl.DataFrame:
    """构造 checkpoint + 合成行情，执行推理，返回信号帧。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest tmp_path fixture，用作 CNN_MODEL_DIR。
        objective: 训练目标字符串。

    Returns:
        predict_cnn_signals 返回的 polars DataFrame。
    """
    frame = _make_trading_frame()

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    name = _make_checkpoint(tmp_path, objective=objective)
    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()

    from aitrade.cnn.predictor import predict_cnn_signals
    return predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)


# ---------------------------------------------------------------------------
# 4.2  示例测试：三种 objective 各一个
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
class TestObjectiveColumnPresent:
    """4.2 示例测试：三种 objective 的信号帧末列恒含 objective，全行同值。"""

    def test_classification_has_objective_column(self, monkeypatch, tmp_path) -> None:
        """classification 推理输出应含 objective 列（末列），全行值 == 'classification'。

        列数应为 4（原 3 列 + objective）；signal 列仍为数值。
        """
        df = _run_predict(monkeypatch, tmp_path, "classification")

        assert "objective" in df.columns, "classification 帧缺少 objective 列"
        assert df.columns[-1] == "objective", "objective 应在末列"
        assert len(df.columns) == 4, f"classification 帧应有 4 列，实得 {len(df.columns)}"
        assert df.columns == ["datetime", "vt_symbol", "signal", "objective"]

        vals = df["objective"].to_list()
        assert all(v == "classification" for v in vals), (
            f"objective 列应全行 == 'classification'，实得 {set(vals)}"
        )
        assert df.height > 0

        # signal 列仍是数值
        assert df["signal"].dtype in (pl.Float32, pl.Float64), (
            f"signal 列类型异常: {df['signal'].dtype}"
        )

    def test_regression_has_objective_column(self, monkeypatch, tmp_path) -> None:
        """regression 推理输出应含 objective 列（末列），全行值 == 'regression'。

        列数应为 4（原 3 列 + objective）；signal 列仍为数值。
        """
        df = _run_predict(monkeypatch, tmp_path, "regression")

        assert "objective" in df.columns, "regression 帧缺少 objective 列"
        assert df.columns[-1] == "objective", "objective 应在末列"
        assert len(df.columns) == 4, f"regression 帧应有 4 列，实得 {len(df.columns)}"
        assert df.columns == ["datetime", "vt_symbol", "signal", "objective"]

        vals = df["objective"].to_list()
        assert all(v == "regression" for v in vals), (
            f"objective 列应全行 == 'regression'，实得 {set(vals)}"
        )
        assert df.height > 0

        assert df["signal"].dtype in (pl.Float32, pl.Float64)

    def test_path_class_has_objective_column(self, monkeypatch, tmp_path) -> None:
        """path_class 推理输出应含 objective 列（末列），全行值 == 'path_class'。

        列数应为 8（原 7 列 + objective）；signal/prob_* 列不变。
        """
        df = _run_predict(monkeypatch, tmp_path, "path_class")

        assert "objective" in df.columns, "path_class 帧缺少 objective 列"
        assert df.columns[-1] == "objective", "objective 应在末列"
        assert len(df.columns) == 8, f"path_class 帧应有 8 列，实得 {len(df.columns)}"
        assert df.columns == [
            "datetime", "vt_symbol", "signal",
            "prob_tp", "prob_sl", "prob_time_up", "prob_time_down",
            "objective",
        ]

        vals = df["objective"].to_list()
        assert all(v == "path_class" for v in vals), (
            f"objective 列应全行 == 'path_class'，实得 {set(vals)}"
        )
        assert df.height > 0

        # prob_* 列仍是数值
        for col in ("prob_tp", "prob_sl", "prob_time_up", "prob_time_down"):
            assert df[col].dtype in (pl.Float32, pl.Float64), (
                f"{col} 列类型异常: {df[col].dtype}"
            )

    def test_objective_column_does_not_affect_signal_values(
        self, monkeypatch, tmp_path
    ) -> None:
        """追加 objective 列不改变 signal 列的数值（对比不含 objective 的 path_class 子集）。

        通过两次相同种子推理做完全相同断言——确保无数值副作用。
        """
        import torch
        from aitrade.cnn.network import create_market_cnn
        from aitrade.cnn.predictor import predict_cnn_signals

        frame = _make_trading_frame(seed=42)

        def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
            return frame

        monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
        monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

        torch.manual_seed(0)
        model = create_market_cnn(
            FEATURE_CHANNELS, LOOKBACK, MAX_GROUP_WIDTH, GROUP_COUNT,
            dropout=0.0, objective="path_class",
        )
        save_data: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": FEATURE_CHANNELS,
                "time_steps": LOOKBACK,
                "max_group_width": MAX_GROUP_WIDTH,
                "group_count": GROUP_COUNT,
                "dropout": 0.0,
                "num_classes": 4,
            },
            "train_config": {
                "target_symbol": TARGET,
                "lookback": LOOKBACK,
                "input_data_kind": "bar",
                "input_interval": "d",
                "objective": "path_class",
                "observation_groups": GROUPS,
            },
            "normalization": {
                "channel_mean": [0.0] * FEATURE_CHANNELS,
                "channel_std": [1.0] * FEATURE_CHANNELS,
            },
        }
        import torch as _torch
        _torch.save(save_data, str(tmp_path / "obj_signal_check.pt"))

        start_dt = frame["datetime"][30].date()
        end_dt = frame["datetime"][-1].date()
        df = predict_cnn_signals(model_name="obj_signal_check", start=start_dt, end=end_dt)

        # signal 仍等于 prob_tp（path_class 的固有关系，加列不应打破）
        for i, row in enumerate(df.to_dicts()):
            assert row["signal"] == row["prob_tp"], (
                f"第 {i} 行: signal={row['signal']} ≠ prob_tp={row['prob_tp']}"
            )
        # 四概率行和 ≈ 1
        prob_sum = (
            df["prob_tp"] + df["prob_sl"] + df["prob_time_up"] + df["prob_time_down"]
        ).to_list()
        for i, s in enumerate(prob_sum):
            assert abs(s - 1.0) < 1e-5, f"第 {i} 行概率和={s}"


# ---------------------------------------------------------------------------
# 4.3  Property 4（Hypothesis）：自描述 + legacy 兼容
# ---------------------------------------------------------------------------

# Feature: cnn-eval-honesty-fixes, Property 4:
# 对任意 objective 的 CNN 推理输出，信号帧含 objective 列、全行同值且等于
# checkpoint objective；既有列（signal/prob_*）不变；不含该列的 legacy 帧经
# 所有消费方处理行为不变。


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@given(
    seed=st.integers(min_value=0, max_value=9999),
    objective=st.sampled_from(["classification", "regression", "path_class"]),
)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property4_objective_column_self_describes(
    seed: int,
    objective: str,
    monkeypatch,
    tmp_path,
) -> None:
    """Property 4（自描述子句）：任意 objective 的推理帧含 objective 列、全行同值等于 checkpoint objective。

    覆盖三种 objective × 随机权重种子，验证：
    - objective 列存在且在末列；
    - 全行值 == 对应 objective 字符串；
    - signal 列保持数值 dtype；
    - 列数符合预期（classification/regression=4，path_class=8）。

    Feature: cnn-eval-honesty-fixes, Property 4: 对任意 objective 的 CNN 推理输出，
    信号帧含 objective 列、全行同值且等于 checkpoint objective；既有列（signal/prob_*）不变。
    """
    import torch
    from aitrade.cnn.network import create_market_cnn
    from aitrade.cnn.predictor import predict_cnn_signals

    frame = _make_trading_frame(seed=seed % 100)

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    torch.manual_seed(seed)
    model = create_market_cnn(
        FEATURE_CHANNELS, LOOKBACK, MAX_GROUP_WIDTH, GROUP_COUNT,
        dropout=0.0, objective=objective,
    )
    model_config: dict[str, Any] = {
        "in_channels": FEATURE_CHANNELS,
        "time_steps": LOOKBACK,
        "max_group_width": MAX_GROUP_WIDTH,
        "group_count": GROUP_COUNT,
        "dropout": 0.0,
    }
    if objective == "path_class":
        model_config["num_classes"] = 4

    save_data = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "train_config": {
            "target_symbol": TARGET,
            "lookback": LOOKBACK,
            "input_data_kind": "bar",
            "input_interval": "d",
            "objective": objective,
            "observation_groups": GROUPS,
        },
        "normalization": {
            "channel_mean": [0.0] * FEATURE_CHANNELS,
            "channel_std": [1.0] * FEATURE_CHANNELS,
        },
    }
    model_name = f"prop4_{objective}_{seed}"
    torch.save(save_data, str(tmp_path / f"{model_name}.pt"))

    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()
    df = predict_cnn_signals(model_name=model_name, start=start_dt, end=end_dt)

    # objective 列存在且在末列
    assert "objective" in df.columns, f"objective={objective}: 帧缺少 objective 列"
    assert df.columns[-1] == "objective", (
        f"objective={objective}: objective 列应在末列，实得 {df.columns}"
    )

    # 全行同值 == checkpoint objective
    assert df.height > 0, f"objective={objective}: 推理结果为空"
    vals = df["objective"].to_list()
    assert all(v == objective for v in vals), (
        f"objective={objective}: 列值不全等于 '{objective}'，实得 {set(vals)}"
    )

    # signal 列保持数值 dtype
    assert df["signal"].dtype in (pl.Float32, pl.Float64), (
        f"objective={objective}: signal 列类型异常 {df['signal'].dtype}"
    )

    # 列数符合预期
    expected_ncols = 8 if objective == "path_class" else 4
    assert len(df.columns) == expected_ncols, (
        f"objective={objective}: 期望 {expected_ncols} 列，实得 {len(df.columns)}: {df.columns}"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
def test_property4_legacy_frame_no_objective_col_runs_without_error() -> None:
    """Property 4（legacy 兼容子句）：不含 objective 列的三列信号帧喂给 CNNSignalStrategy，行为与改造前完全一致。

    构造一个手工三列 DataFrame（模拟规则策略/历史信号，无 objective 列），
    喂给 CNNSignalStrategy 跑一小段回测，断言不报错且能正常买卖——守护 R3.3。

    Feature: cnn-eval-honesty-fixes, Property 4: 不含该列的 legacy 帧经所有消费方
    处理行为不变。
    """
    from datetime import timedelta

    from aitrade.backtest.engine import BacktestingEngine
    from aitrade.backtest.types import BarData, Direction
    from aitrade.cnn.strategy import CNNSignalStrategy

    SYMBOL = "TEST.SZSE"
    START = datetime(2026, 1, 5)
    closes = [100.0 + i for i in range(8)]
    days = [START + timedelta(days=i) for i in range(len(closes))]
    bars: list[BarData] = []
    for i, close in enumerate(closes):
        prev = closes[i - 1] if i > 0 else close
        bars.append(BarData(
            symbol="TEST",
            exchange="SZSE",
            datetime=days[i],
            interval="d",
            open_price=prev,
            high_price=max(prev, close) + 1.0,
            low_price=min(prev, close) - 1.0,
            close_price=close,
            volume=1_000_000,
        ))

    # 三列信号帧（无 objective 列）——模拟规则策略或改造前 CNN 信号
    signal_df = pl.DataFrame({
        "datetime": days,
        "vt_symbol": [SYMBOL] * len(days),
        "signal": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9],
    })

    class _FakeLoader:
        def load_bar_data(self, vt_symbol, interval, start, end):
            return list(bars)

        def load_contract_settings(self):
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

    engine = BacktestingEngine(data_loader=_FakeLoader())
    engine.set_parameters(
        vt_symbols=[SYMBOL],
        interval="d",
        start=days[0],
        end=days[-1] + timedelta(days=1),
        capital=1_000_000,
    )
    engine.add_strategy(
        CNNSignalStrategy,
        {"buy_threshold": 0.6, "exit_mode": "threshold"},
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()   # 不应抛出任何异常

    trades = sorted(engine.get_all_trades(), key=lambda t: int(t.tradeid))
    buy_trades = [t for t in trades if t.direction == Direction.LONG]
    # 前三行 signal=0.9 >= 0.6，应有买入成交（回测引擎至少触发一次）
    assert len(buy_trades) >= 1, "三列信号帧应能正常触发买入（向后兼容）"
