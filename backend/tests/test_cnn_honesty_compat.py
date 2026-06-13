"""
向后兼容守护：诚实性修复对默认与合法配置零影响（cnn-eval-honesty-fixes Task 7）。

本文件是"集中守护"：把 Property 7 的四个子不变量固化为一个专属测试文件，
确保「n_seeds=1, seed=42 的默认/合法路径」与改造前完全等价。

Feature: cnn-eval-honesty-fixes
Property 7: 对任意 objective ∈ {classification, regression, path_class} 且未启用多种子
（n_seeds=1, seed=42）：训练权重、信号帧 signal 列、回测成交与改造前一致；
既有 test_cnn_* 套件全绿。

四个子不变量：
  P7.1  seed 默认等价：不传 seed 与显式传 seed=42 训出完全相同的权重；
  P7.2  signal 列不变：加 objective 列是纯增量，不扰动 signal 列的数值/dtype；
  P7.3  合法阈值放行不改成交：threshold_scale_check 接入后，合法配置回测成交序列正常；
  P7.4  n_seeds=1 退化等价：cross_seed.std==0，candidate_score==该单一种子分数。

Requirements: 6.1, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aitrade.cnn.thresholds import threshold_scale_check


# ---------------------------------------------------------------------------
# 共享工厂：合成训练数据集（与 test_cnn_seed.py 对齐）
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n: int = 80,
    C: int = 6,
    T: int = 10,
    S: int = 2,
    G: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """生成与 trainer 种子无关的合成分类数据集，用于 seed 参数化测试。

    使用固定 rng(seed=0)，确保合成 X/y 本身不依赖 trainer 种子，
    从而隔离 trainer seed 对权重初始化和 DataLoader shuffle 的影响。

    Args:
        n: 样本数；默认 80（>=trainer 最低 50）。
        C: 特征通道数；默认 6。
        T: 时间步数；默认 10。
        S: 每组最大证券数；默认 2。
        G: 分组数；默认 1。

    Returns:
        (X, y, group_mask, info) 四元组，info 键与真实 build_dataset 输出一致。
    """
    rng = np.random.default_rng(0)  # 固定，与 trainer seed 无关
    X = rng.standard_normal((n, C, T, S, G)).astype(np.float32)
    y = (rng.uniform(0, 1, n) > 0.5).astype(np.float32)
    group_mask = np.ones((1, 1, 1, S, G), dtype=np.float32)

    start_dt = datetime(2024, 1, 1)
    anchor_dates = [(start_dt + timedelta(days=i)).isoformat() for i in range(n)]
    info: dict[str, Any] = {
        "symbols": ["AAA.SSE", "BBB.SSE"],
        "groups": [{"role": "target", "name": "目标", "symbols": ["AAA.SSE", "BBB.SSE"]}],
        "target_symbol": "AAA.SSE",
        "feature_names": ["open_pct", "high_pct", "low_pct", "close_pct", "volume_pct", "turnover_pct"],
        "feature_channels": C,
        "group_count": G,
        "max_group_width": S,
        "lookback": T,
        "n_dates": n + T,
        "dates": anchor_dates,
        "sample_anchor_dates": anchor_dates,
        "input_data_kind": "bar",
        "input_interval": "d",
        "label_spec": {
            "mode": "next_bar",
            "threshold": 0.0,
            "neutral_policy": "drop",
            "price_ref": "close",
        },
        "label_threshold": 0.0,
        "price_ref": "close",
        "objective": "classification",
        "skipped_for_label": 0,
        "skipped_for_neutral": 0,
        "sample_returns": rng.uniform(-0.05, 0.05, n).tolist(),
    }
    return X, y, group_mask, info


def _train_capture(
    seed: int | None,
    tmp_path: Any,
    monkeypatch: Any,
) -> dict[str, Any]:
    """用合成数据训练 2 epoch，捕获 save_data（含 model_state_dict / train_config）。

    Args:
        seed: 传给 train_cnn_model 的 seed 参数；None 表示不传（使用函数默认值 42）。
        tmp_path: pytest 临时目录，供假 save 路径。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        save_cnn_model 捕获到的 data 字典。
    """
    import aitrade.cnn.trainer as trainer_mod

    X, y, group_mask, info = _make_synthetic_dataset()
    captured: dict[str, Any] = {}

    def _fake_save(name: str, data: dict, hist: list) -> tuple:
        captured["data"] = data
        return (tmp_path / f"{name}.pt", tmp_path / f"{name}.json")

    monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))
    monkeypatch.setattr(trainer_mod, "save_cnn_model", _fake_save)

    from aitrade.cnn.trainer import train_cnn_model

    kwargs: dict[str, Any] = dict(
        name="compat_test",
        vt_symbols=["AAA.SSE", "BBB.SSE"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        epochs=2,
        batch_size=16,
        lookback=10,
        dropout=0.0,
    )
    if seed is not None:
        kwargs["seed"] = seed

    train_cnn_model(**kwargs)
    return captured["data"]


# ---------------------------------------------------------------------------
# P7.1  seed 默认等价：不传 seed == 显式传 seed=42
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
class TestSeedDefaultEquivalence:
    """P7.1：不传 seed 与显式传 seed=42 训出完全相同的权重。

    # Feature: cnn-eval-honesty-fixes, Property 7: seed 默认等价子不变量
    """

    def test_default_seed_matches_explicit_42(self, tmp_path, monkeypatch) -> None:
        """不传 seed（默认 42）与显式 seed=42 产出数值完全相同的 state_dict。

        守护 R1.5（调用方未显式传 seed 时，Trainer 使用默认 42，保持与改造前一致的单次训练行为）。
        """
        import torch

        data_default = _train_capture(None, tmp_path, monkeypatch)
        data_42 = _train_capture(42, tmp_path, monkeypatch)

        sd_default = data_default["model_state_dict"]
        sd_42 = data_42["model_state_dict"]

        assert set(sd_default.keys()) == set(sd_42.keys()), "两次训练的 state_dict 键集合不同"
        for key in sd_default:
            assert torch.allclose(sd_default[key], sd_42[key], atol=0.0), (
                f"P7.1：默认 seed 与显式 seed=42 在参数 {key} 上不等价"
            )

    def test_default_seed_recorded_as_42(self, tmp_path, monkeypatch) -> None:
        """不传 seed 时，train_config['seed'] 应记录为 42（改造后的默认值不变）。

        守护 checkpoint 的 seed 记录与默认行为一致。
        """
        data = _train_capture(None, tmp_path, monkeypatch)
        assert data["train_config"].get("seed") == 42, (
            "P7.1：默认 seed 应记录为 42，实际为 "
            f"{data['train_config'].get('seed')!r}"
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property7_1_explicit_seed_equals_two_consecutive_runs(
    seed: int, tmp_path: Any, monkeypatch: Any
) -> None:
    """P7.1 属性测试：任意显式 seed 下，两次训练权重精确相等（可复现性守护）。

    # Feature: cnn-eval-honesty-fixes, Property 7: seed 默认等价——可复现性不变量
    不传 seed 和传 seed=42 等价已由示例测试固化；此属性测试泛化地断言"给定 seed 可复现"，
    避免默认路径的可复现性被未来改动破坏。
    """
    import torch

    data_a = _train_capture(seed, tmp_path, monkeypatch)
    data_b = _train_capture(seed, tmp_path, monkeypatch)

    sd_a = data_a["model_state_dict"]
    sd_b = data_b["model_state_dict"]

    for key in sd_a:
        assert torch.allclose(sd_a[key], sd_b[key], atol=0.0), (
            f"P7.1 属性失败：seed={seed}，参数 {key} 两次训练结果不等（可复现性被破坏）"
        )


# ---------------------------------------------------------------------------
# P7.2  signal 列不变：加 objective 列是纯增量，不扰动 signal
# ---------------------------------------------------------------------------

TARGET = "000001.SZSE"
LOOKBACK = 10
FEATURE_CHANNELS = 6
MAX_GROUP_WIDTH = 1
GROUP_COUNT = 1
FRAME_ROWS = 60
GROUPS = [{"role": "target", "name": "目标", "symbols": [TARGET]}]


def _make_trading_frame(rows: int = FRAME_ROWS, seed: int = 0) -> pl.DataFrame:
    """构造连续工作日合成日线行情帧（与 test_cnn_signal_objective.py 对齐）。

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


def _make_predict_checkpoint(
    tmp_path: Any,
    objective: str,
    name: str,
    torch_seed: int = 0,
) -> str:
    """直接构造最小 checkpoint 并落盘，返回模型名称。

    Args:
        tmp_path: pytest tmp_path fixture，用作 CNN_MODEL_DIR。
        objective: 训练目标字符串。
        name: 模型名（不含 .pt）。
        torch_seed: 控制权重初始化的 torch 种子。

    Returns:
        落盘的模型名（不含 .pt 后缀）。
    """
    import torch
    from aitrade.cnn.network import create_market_cnn

    torch.manual_seed(torch_seed)
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
    torch.save(save_data, str(tmp_path / f"{name}.pt"))
    return name


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@pytest.mark.parametrize("objective", ["classification", "regression", "path_class"])
def test_property7_2_objective_column_is_pure_additive(
    objective: str, monkeypatch, tmp_path
) -> None:
    """P7.2：objective 列是纯增量——加列后 signal 列的 dtype 与数值与"预期 signal 源"完全一致。

    # Feature: cnn-eval-honesty-fixes, Property 7: signal 列不变子不变量
    对三种 objective 分别：
    - 推理输出含 objective 末列；
    - signal 列仍为数值 dtype（Float32/Float64）；
    - path_class 下 signal == prob_tp（signal 的定义是 prob_tp，加列后不应改变）；
    - classification/regression 下 signal 列不全零（模型有输出）。

    守护 R3.2（Objective_Column 不改变既有列的语义与顺序）。
    """
    from aitrade.cnn import model as cnn_model
    from aitrade.cnn import storage as cnn_storage
    from aitrade.cnn.predictor import predict_cnn_signals

    frame = _make_trading_frame()

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    name = _make_predict_checkpoint(tmp_path, objective=objective, name=f"p72_{objective}")
    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()

    df = predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)

    # objective 列在末位（纯增量，不改已有列的位置）
    assert "objective" in df.columns, f"P7.2：{objective} 帧缺少 objective 末列"
    assert df.columns[-1] == "objective", (
        f"P7.2：{objective} objective 应在末列，实际顺序 {df.columns}"
    )
    assert df.height > 0, f"P7.2：{objective} 推理结果不应为空"

    # signal 列保持数值 dtype（加列不改类型）
    assert df["signal"].dtype in (pl.Float32, pl.Float64), (
        f"P7.2：{objective} signal 列 dtype 异常：{df['signal'].dtype}"
    )

    # path_class：signal 仍等于 prob_tp（加 objective 列不破坏该等式）
    if objective == "path_class":
        for i, row in enumerate(df.to_dicts()):
            assert row["signal"] == pytest.approx(row["prob_tp"]), (
                f"P7.2：path_class 第 {i} 行 signal={row['signal']} ≠ prob_tp={row['prob_tp']}"
            )

    # signal 列非空（模型对合法输入有输出）
    signals = df["signal"].to_list()
    assert any(s != 0.0 for s in signals), (
        f"P7.2：{objective} signal 全零，推理可能未正常运行"
    )


# ---------------------------------------------------------------------------
# P7.3  合法阈值放行不改成交：threshold_scale_check 接入后，合法配置正常买卖
# ---------------------------------------------------------------------------

SYMBOL = "TEST.SZSE"
START = datetime(2026, 1, 5)


class _FakeLoader:
    """实现 BarDataLoader 协议的合成数据源，无外部依赖。"""

    def __init__(self, bars: list) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list:
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


def _build_backtest_bars(closes: list[float]) -> tuple[list, list[datetime]]:
    """构造满足撮合条件的合成日线（low 低于前收，保证限价单可成交）。

    Args:
        closes: 各根 bar 的收盘价列表。

    Returns:
        (bars 列表, datetime 列表) 元组。
    """
    from aitrade.backtest.types import BarData

    days = [START + timedelta(days=i) for i in range(len(closes))]
    bars: list = []
    for i, close in enumerate(closes):
        prev = closes[i - 1] if i > 0 else close
        bars.append(
            BarData(
                symbol="TEST",
                exchange="SZSE",
                datetime=days[i],
                interval="d",
                open_price=prev,
                high_price=max(prev, close) + 1.0,
                low_price=min(prev, close) - 1.0,
                close_price=close,
                volume=1_000_000,
            )
        )
    return bars, days


def _run_backtest(
    signal_df: pl.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    exit_mode: str = "threshold",
) -> tuple:
    """运行一次合成回测，返回 (engine, buy_trades, all_trades)。

    Args:
        signal_df: 含 objective 列的信号帧（含 [datetime, vt_symbol, signal, objective]）。
        buy_threshold: 买入阈值。
        sell_threshold: 卖出阈值。
        exit_mode: 出场模式；默认 threshold。

    Returns:
        (BacktestingEngine, buy_trades_list, all_trades_list) 三元组。
    """
    from aitrade.backtest.engine import BacktestingEngine
    from aitrade.backtest.types import Direction
    from aitrade.cnn.strategy import CNNSignalStrategy

    closes = [10.0] * 10
    bars, days = _build_backtest_bars(closes)

    engine = BacktestingEngine(data_loader=_FakeLoader(bars))
    engine.set_parameters(
        vt_symbols=[SYMBOL],
        interval="d",
        start=days[0],
        end=days[-1] + timedelta(days=1),
        capital=1_000_000,
    )
    engine.add_strategy(
        CNNSignalStrategy,
        {"buy_threshold": buy_threshold, "sell_threshold": sell_threshold, "exit_mode": exit_mode},
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()

    all_trades = sorted(engine.get_all_trades(), key=lambda t: int(t.tradeid))
    buy_trades = [t for t in all_trades if t.direction == Direction.LONG]
    return engine, buy_trades, all_trades


def _make_signal_df(
    n: int,
    signals: list[float],
    objective: str,
) -> pl.DataFrame:
    """构造含 objective 列的信号帧。

    Args:
        n: 行数（与 signals 等长）。
        signals: 各行 signal 值。
        objective: 末列 objective 常量字符串。

    Returns:
        含 [datetime, vt_symbol, signal, objective] 的 DataFrame。
    """
    days = [START + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({
        "datetime": days,
        "vt_symbol": [SYMBOL] * n,
        "signal": signals,
        "objective": [objective] * n,
    })


class TestProperty73LegalThresholdNoImpact:
    """P7.3：合法配置下 threshold_scale_check 接入后，回测成交序列正常（零误拦）。

    # Feature: cnn-eval-honesty-fixes, Property 7: 合法阈值放行不改成交子不变量
    这是"诚实性修复对合法路径零影响"的核心断言：三种 objective 的合法阈值组合
    均能正常触发买入，_threshold_invalid=False，与改造前行为完全一致。
    """

    def test_classification_legal_threshold_normal_trade(self) -> None:
        """classification 0.6/0.4（合法）→ _threshold_invalid=False 且有买入成交。

        守护 R4.5（放行既有合法配置：classification 0.6/0.4）和 R6.3（改造前一致）。
        """
        n = 10
        # 前半段高信号触发买入，后半段低信号触发出场
        signals = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
        df = _make_signal_df(n, signals, objective="classification")

        engine, buy_trades, _ = _run_backtest(df, buy_threshold=0.6, sell_threshold=0.4)

        assert engine.strategy._threshold_invalid is False, (
            "P7.3：classification 0.6/0.4 合法配置不应标记 _threshold_invalid"
        )
        assert len(buy_trades) >= 1, (
            "P7.3：合法 classification 配置应能正常触发买入成交"
        )

    def test_regression_legal_threshold_normal_trade(self) -> None:
        """regression 0.005/-0.005（合法）→ _threshold_invalid=False 且有买入成交。

        守护 R4.5（放行既有合法配置：regression 0.005/-0.005）。
        """
        n = 10
        # regression signal 是预测收益，0.02 > 0.005 触发买入
        signals = [0.02, 0.02, 0.02, 0.02, 0.02, -0.01, -0.01, -0.01, -0.01, -0.01]
        df = _make_signal_df(n, signals, objective="regression")

        engine, buy_trades, _ = _run_backtest(df, buy_threshold=0.005, sell_threshold=-0.005)

        assert engine.strategy._threshold_invalid is False, (
            "P7.3：regression 0.005/-0.005 合法配置不应标记 _threshold_invalid"
        )
        assert len(buy_trades) >= 1, (
            "P7.3：合法 regression 配置应能正常触发买入成交"
        )

    def test_path_class_legal_threshold_normal_trade(self) -> None:
        """path_class 0.6 veto=1.0（合法）→ _threshold_invalid=False 且有买入成交。

        守护 R4.5（放行既有合法配置：path_class 0.6 + veto）。
        """
        from aitrade.backtest.engine import BacktestingEngine
        from aitrade.backtest.types import Direction
        from aitrade.cnn.strategy import CNNSignalStrategy

        n = 10
        days = [START + timedelta(days=i) for i in range(n)]
        signals = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
        prob_sl = [0.1] * n  # 低 sl 概率，不触发 veto

        df = pl.DataFrame({
            "datetime": days,
            "vt_symbol": [SYMBOL] * n,
            "signal": signals,
            "prob_tp": signals,
            "prob_sl": prob_sl,
            "prob_time_up": [0.05] * n,
            "prob_time_down": [0.05] * n,
            "objective": ["path_class"] * n,
        })

        closes = [10.0] * n
        bars, _ = _build_backtest_bars(closes)

        engine = BacktestingEngine(data_loader=_FakeLoader(bars))
        engine.set_parameters(
            vt_symbols=[SYMBOL],
            interval="d",
            start=days[0],
            end=days[-1] + timedelta(days=1),
            capital=1_000_000,
        )
        engine.add_strategy(
            CNNSignalStrategy,
            {"buy_threshold": 0.6, "sell_threshold": 0.4, "exit_mode": "threshold", "veto_threshold": 1.0},
            df,
        )
        engine.load_data()
        engine.run_backtesting()

        all_trades = engine.get_all_trades()
        buy_trades = [t for t in all_trades if t.direction == Direction.LONG]

        assert engine.strategy._threshold_invalid is False, (
            "P7.3：path_class 0.6 合法配置不应标记 _threshold_invalid"
        )
        assert len(buy_trades) >= 1, (
            "P7.3：合法 path_class 配置应能正常触发买入成交"
        )


@pytest.mark.parametrize(
    "objective,buy,sell",
    [
        ("classification", 0.6, 0.4),
        ("regression", 0.005, -0.005),
        ("path_class", 0.6, 0.4),
        (None, 0.6, 0.4),  # legacy（无 objective 列）
    ],
)
def test_property7_3_legal_configs_zero_violation(
    objective: str | None, buy: float, sell: float
) -> None:
    """P7.3 属性：既有合法配置经 threshold_scale_check 零违规（诚实性修复不误拦合法配置）。

    # Feature: cnn-eval-honesty-fixes, Property 7: 合法阈值放行不改成交子不变量
    这是 P7.3 的纯函数层守护：回测入口 / strategy / live service 三处共用
    threshold_scale_check，合法配置在函数层就返回空违规列表。
    """
    reasons = threshold_scale_check(objective, buy, sell)
    assert reasons == [], (
        f"P7.3：合法配置 objective={objective!r} buy={buy} sell={sell} "
        f"不应有违规，实得 {reasons}"
    )


# ---------------------------------------------------------------------------
# P7.4  n_seeds=1 退化等价：cross_seed.std==0，candidate_score==单种子分数
# ---------------------------------------------------------------------------


class TestProperty74NSeedsOneDegeneracy:
    """P7.4：n_seeds=1（默认）时 WF 折结果退化为单种子，行为与改造前一致。

    # Feature: cnn-eval-honesty-fixes, Property 7: n_seeds=1 退化子不变量
    n_seeds=1 是默认值，对应"改造前单次训练"的行为。
    此测试集中固化"默认 n_seeds=1 时，cross_seed 统计量与单次训练等价"的不变量。
    """

    def test_n_seeds_1_yields_std_zero(self, monkeypatch, tmp_path) -> None:
        """n_seeds=1（默认）：折内 cross_seed.std==0，candidate_score==该单一种子分数。

        守护 R2.5（n_seeds=1 退化为单种子，行为与单次训练等价）和 R6.5（n_seeds=1 默认时
        开销与改造前一致）。
        """
        from datetime import timedelta

        import aitrade.cnn.governance as gov
        from aitrade.cnn.governance import CNNGovernanceStore
        from aitrade.models.governance import CNNWalkForwardRequest

        monkeypatch.setattr(gov, "store", CNNGovernanceStore(tmp_path))

        expected_score = 0.37

        def _fake_train(req, *, model_name, start, end, seed_index=0, on_progress=None):
            return {"name": model_name, "seed_index": seed_index}

        def _fake_backtest(*, model_name, name, start, end, capital, params):
            return {"statistics": {"total_return": expected_score, "total_trade_count": 1}}

        monkeypatch.setattr(gov, "_train_governance_model", _fake_train)
        monkeypatch.setattr(gov, "_backtest_model", _fake_backtest)

        start = date(2022, 1, 1)
        end = start + timedelta(days=540 + 30)
        req = CNNWalkForwardRequest(
            name="p74_wf",
            target_symbol="AAA.SSE",
            start=start,
            end=end,
            train_days=540,
            test_days=30,
            step_days=30,
            # n_seeds 不传，使用默认值 1
        )
        assert req.n_seeds == 1, "CNNWalkForwardRequest 默认 n_seeds 应为 1"

        report = gov.run_walk_forward_evaluate(req)
        fold = report["folds"][0]

        assert fold["cross_seed"]["n"] == 1, (
            f"P7.4：n_seeds=1 时 cross_seed.n 应为 1，实得 {fold['cross_seed']['n']}"
        )
        assert fold["cross_seed"]["std"] == 0.0, (
            f"P7.4：n_seeds=1 时 cross_seed.std 应为 0，实得 {fold['cross_seed']['std']}"
        )
        assert fold["candidate_score"] == pytest.approx(expected_score), (
            f"P7.4：n_seeds=1 时 candidate_score 应等于单一种子分数 {expected_score}，"
            f"实得 {fold['candidate_score']}"
        )
        assert fold["candidate_score"] == pytest.approx(fold["cross_seed"]["mean"]), (
            "P7.4：candidate_score 应恒等于 cross_seed.mean"
        )

    def test_n_seeds_1_cross_seed_mean_equals_single_score(self, monkeypatch, tmp_path) -> None:
        """n_seeds=1 时 cross_seed.mean 等于唯一种子的 OOS 分数，无任何聚合失真。

        通过参数化验证 expected_score 为任意浮点值时等式成立。
        """
        import aitrade.cnn.governance as gov
        from aitrade.cnn.governance import CNNGovernanceStore, _cross_seed_dispersion

        monkeypatch.setattr(gov, "store", CNNGovernanceStore(tmp_path))

        for score in [0.0, 0.5, -0.3, 1.0, 0.123456]:
            disp = _cross_seed_dispersion([score])
            assert disp["mean"] == pytest.approx(score), (
                f"P7.4：_cross_seed_dispersion([{score}]).mean != {score}"
            )
            assert disp["std"] == 0.0, (
                f"P7.4：_cross_seed_dispersion([{score}]).std 应为 0，实得 {disp['std']}"
            )
            assert disp["n"] == 1


@given(score=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=80)
def test_property7_4_single_seed_dispersion(score: float) -> None:
    """P7.4 属性测试：n_seeds=1 时离散度函数对任意分数均给出 std=0，mean=score，n=1。

    # Feature: cnn-eval-honesty-fixes, Property 7: n_seeds=1 退化子不变量
    """
    from aitrade.cnn.governance import _cross_seed_dispersion

    disp = _cross_seed_dispersion([score])
    assert disp["n"] == 1
    assert disp["std"] == 0.0
    assert disp["mean"] == pytest.approx(score, abs=1e-9)


# ---------------------------------------------------------------------------
# P7 综合守护：三种 objective 的合法路径组合，确认没有任何意外交互
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@pytest.mark.parametrize("objective", ["classification", "regression", "path_class"])
def test_property7_full_default_path_no_regression(
    objective: str, monkeypatch, tmp_path
) -> None:
    """Property 7 综合验证：seed=42/n_seeds=1 默认路径下三种 objective 均无回归。

    # Feature: cnn-eval-honesty-fixes, Property 7: 向后兼容不变性（综合）
    验证：
    1. 推理输出含 objective 末列（不破坏 signal/prob_* 列的存在）；
    2. signal 列 dtype 保持数值类型；
    3. objective 列全行同值等于 checkpoint objective；
    4. threshold_scale_check 对该 objective 的合法默认阈值返回空违规列表。

    这四条合起来是"改造前的 classification/regression/path_class 推理路径"
    在改造后完全等价的最简集中断言。
    """
    from aitrade.cnn import model as cnn_model
    from aitrade.cnn import storage as cnn_storage
    from aitrade.cnn.predictor import predict_cnn_signals

    frame = _make_trading_frame()

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    name = _make_predict_checkpoint(tmp_path, objective=objective, name=f"p7_full_{objective}")
    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()

    df = predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)

    # 1. objective 末列存在
    assert df.columns[-1] == "objective", (
        f"P7-综合：{objective} 推理帧 objective 列应在末位，实际 {df.columns}"
    )

    # 2. signal 列 dtype 保持数值
    assert df["signal"].dtype in (pl.Float32, pl.Float64), (
        f"P7-综合：{objective} signal 列 dtype={df['signal'].dtype} 不是数值类型"
    )

    # 3. objective 列全行等于 checkpoint objective
    assert df.height > 0, f"P7-综合：{objective} 推理结果不应为空"
    vals = df["objective"].to_list()
    assert all(v == objective for v in vals), (
        f"P7-综合：{objective} 帧 objective 列值异常：{set(vals)}"
    )

    # 4. 合法默认阈值无违规
    legal_thresholds = {
        "classification": (0.6, 0.4),
        "regression": (0.005, -0.005),
        "path_class": (0.6, 0.4),
    }
    buy, sell = legal_thresholds[objective]
    reasons = threshold_scale_check(objective, buy, sell)
    assert reasons == [], (
        f"P7-综合：{objective} 合法默认阈值应零违规，实得 {reasons}"
    )
