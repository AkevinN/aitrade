"""
Task 4 验收测试：predict_cnn_signals 的 path_class 分支推理四概率列。

覆盖范围：
  4.1  示例测试：七列齐全、列序正确；
         旧 objective（classification/regression）输出恰好三列
  4.2  Property 3：对任意 path_class 推理输出，逐行 signal == prob_tp 严格相等
  4.3  Property 5：对任意 objective ∈ {classification, regression} 的输入，
         推理输出列（恰好三列）与改造前完全一致
  4.4  Property 7：对任意训练完成的 path_class 模型，保存→加载→推理后：
         objective 恢复为 path_class、输出头 4 维、Signal_Frame 含全部七列
  4.5  冗余校验：num_classes 被手工篡改为非 4 时抛 ValueError
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

from aitrade.cnn import model as cnn_model
from aitrade.cnn import storage as cnn_storage

# ---------------------------------------------------------------------------
# 共享常量
# ---------------------------------------------------------------------------

TARGET = "000001.SZSE"
LOOKBACK = 10
FEATURE_CHANNELS = 6
MAX_GROUP_WIDTH = 1
GROUP_COUNT = 1
FRAME_ROWS = 60  # 行数足够：lookback=10，start 取第 30 行日期

# 观测组：单组单证券（最简结构）
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
    # 起点设在 2024-01-02（工作日）
    dt = datetime(2024, 1, 2)
    px = 100.0
    for _ in range(rows):
        # 跳过周末
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
    objective: str = "path_class",
    num_classes_override: int | None = None,
) -> str:
    """直接构造最小 checkpoint 并落盘，返回模型名称。

    不走 train_cnn_model，速度快，供 Property 3/5 与示例测试使用。

    Args:
        tmp_path: pytest tmp_path fixture，用作 CNN_MODEL_DIR。
        objective: 训练目标，"path_class" / "classification" / "regression"。
        num_classes_override: 若非 None，将覆盖 model_config["num_classes"]，
            用于测试冗余校验。

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
    if num_classes_override is not None:
        model_config["num_classes"] = num_classes_override

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
    name = f"test_{objective}"
    torch.save(save_data, str(tmp_path / f"{name}.pt"))
    return name


# ---------------------------------------------------------------------------
# 4.1  示例测试：七列 / 三列 / 列序
# ---------------------------------------------------------------------------


class TestExampleTests:
    """4.1 示例测试：输出列数与列序。"""

    def test_path_class_seven_columns(self, monkeypatch, tmp_path) -> None:
        """path_class 推理输出应含七列，列序与文档一致。"""
        pytest.importorskip("torch")
        frame = _make_trading_frame()

        def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
            return frame

        monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
        monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

        name = _make_checkpoint(tmp_path, objective="path_class")
        start_dt = frame["datetime"][30].date()
        end_dt = frame["datetime"][-1].date()

        from aitrade.cnn.predictor import predict_cnn_signals
        df = predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)

        assert isinstance(df, pl.DataFrame)
        # path_class 输出八列（原七列 + objective 末列；Task 4 cnn-eval-honesty-fixes）
        expected_cols = ["datetime", "vt_symbol", "signal", "prob_tp", "prob_sl",
                         "prob_time_up", "prob_time_down", "objective"]
        assert df.columns == expected_cols, f"列序错误: {df.columns}"
        assert df.height > 0

    def test_classification_three_columns(self, monkeypatch, tmp_path) -> None:
        """classification 推理输出应恰好三列 [datetime, vt_symbol, signal]。"""
        pytest.importorskip("torch")
        frame = _make_trading_frame()

        def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
            return frame

        monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
        monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

        name = _make_checkpoint(tmp_path, objective="classification")
        start_dt = frame["datetime"][30].date()
        end_dt = frame["datetime"][-1].date()

        from aitrade.cnn.predictor import predict_cnn_signals
        df = predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)

        # classification 输出四列（原三列 + objective 末列；Task 4 cnn-eval-honesty-fixes）
        assert df.columns == ["datetime", "vt_symbol", "signal", "objective"]

    def test_regression_three_columns(self, monkeypatch, tmp_path) -> None:
        """regression 推理输出应恰好三列 [datetime, vt_symbol, signal]（+objective 末列）。"""
        pytest.importorskip("torch")
        frame = _make_trading_frame()

        def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
            return frame

        monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
        monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

        name = _make_checkpoint(tmp_path, objective="regression")
        start_dt = frame["datetime"][30].date()
        end_dt = frame["datetime"][-1].date()

        from aitrade.cnn.predictor import predict_cnn_signals
        df = predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)

        # regression 输出四列（原三列 + objective 末列；Task 4 cnn-eval-honesty-fixes）
        assert df.columns == ["datetime", "vt_symbol", "signal", "objective"]

    def test_path_class_on_meta_contains_objective(self, monkeypatch, tmp_path) -> None:
        """path_class 推理的 on_meta 回调应包含 objective='path_class'。"""
        pytest.importorskip("torch")
        frame = _make_trading_frame()

        def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
            return frame

        monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
        monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

        name = _make_checkpoint(tmp_path, objective="path_class")
        start_dt = frame["datetime"][30].date()
        end_dt = frame["datetime"][-1].date()

        metas: list[dict] = []
        from aitrade.cnn.predictor import predict_cnn_signals
        predict_cnn_signals(model_name=name, start=start_dt, end=end_dt, on_meta=metas.append)

        assert len(metas) == 1
        assert metas[0]["objective"] == "path_class"


# ---------------------------------------------------------------------------
# 4.5  冗余校验：num_classes 篡改
# ---------------------------------------------------------------------------


class TestNumClassesValidation:
    """4.5 冗余校验：num_classes 不等于 4 时抛 ValueError。"""

    def test_tampered_num_classes_raises(self, monkeypatch, tmp_path) -> None:
        """num_classes=3（手工篡改）→ ValueError，拒绝推理。"""
        pytest.importorskip("torch")
        frame = _make_trading_frame()

        def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
            return frame

        monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
        monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

        # 手工构造 num_classes=3 的 path_class checkpoint
        name = _make_checkpoint(tmp_path, objective="path_class", num_classes_override=3)
        start_dt = frame["datetime"][30].date()
        end_dt = frame["datetime"][-1].date()

        from aitrade.cnn.predictor import predict_cnn_signals
        with pytest.raises(ValueError, match="num_classes"):
            predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)

    def test_absent_num_classes_no_error(self, monkeypatch, tmp_path) -> None:
        """path_class checkpoint 缺少 num_classes 键（旧格式）→ 不报错，向后兼容。"""
        pytest.importorskip("torch")
        import torch
        frame = _make_trading_frame()

        def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
            return frame

        monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
        monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

        # 构造不含 num_classes 的 path_class checkpoint
        from aitrade.cnn.network import create_market_cnn
        model = create_market_cnn(FEATURE_CHANNELS, LOOKBACK, MAX_GROUP_WIDTH, GROUP_COUNT,
                                   dropout=0.0, objective="path_class")
        save_data = {
            "model_state_dict": model.state_dict(),
            "model_config": {    # 故意不含 num_classes
                "in_channels": FEATURE_CHANNELS,
                "time_steps": LOOKBACK,
                "max_group_width": MAX_GROUP_WIDTH,
                "group_count": GROUP_COUNT,
                "dropout": 0.0,
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
        torch.save(save_data, str(tmp_path / "test_no_num_classes.pt"))

        start_dt = frame["datetime"][30].date()
        end_dt = frame["datetime"][-1].date()

        from aitrade.cnn.predictor import predict_cnn_signals
        df = predict_cnn_signals(model_name="test_no_num_classes", start=start_dt, end=end_dt)
        assert df.height > 0
        assert "prob_tp" in df.columns


# ---------------------------------------------------------------------------
# Property 3：逐行 signal == prob_tp 严格相等
# ---------------------------------------------------------------------------

# Feature: cnn-path-multiclass-head, Property 3:
# 对任意 path_class 推理输出的 Signal_Frame，逐行 signal == prob_tp 严格相等。

# 注：max_examples=25 而非 100——模型推理（含权重随机化+前向传播）耗时不稳定
# （torch 首例有预热开销），25 例覆盖足够的随机权重空间，整文件目标 <60s。
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@given(seed=st.integers(min_value=0, max_value=9999))
@settings(
    max_examples=25,
    # deadline=None：torch 首例预热导致耗时超出 Hypothesis 默认 200ms 限制，禁用 deadline。
    deadline=None,
    # monkeypatch/tmp_path 是函数级 fixture，patch 在测试函数层面只设置一次：
    # CNN_MODEL_DIR → tmp_path（每例写入不同的 .pt 文件名，不会冲突）；
    # _load_market_frame 每次返回同一合成帧（不依赖重置）。故安全抑制此检查。
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property3_signal_equals_prob_tp(seed: int, monkeypatch, tmp_path) -> None:
    """Property 3：对任意 path_class 推理输出，逐行 signal == prob_tp 严格相等。"""
    import torch
    from aitrade.cnn.network import create_market_cnn
    from aitrade.cnn.predictor import predict_cnn_signals

    frame = _make_trading_frame(seed=seed % 100)

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    # 每例用随机种子重新初始化权重——域由权重随机化保证
    torch.manual_seed(seed)
    model = create_market_cnn(FEATURE_CHANNELS, LOOKBACK, MAX_GROUP_WIDTH, GROUP_COUNT,
                               dropout=0.0, objective="path_class")
    save_data = {
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
    model_name = f"prop3_{seed}"
    torch.save(save_data, str(tmp_path / f"{model_name}.pt"))

    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()
    df = predict_cnn_signals(model_name=model_name, start=start_dt, end=end_dt)

    # 核心断言：signal 恒等 prob_tp（浮点完全相等，因为两者来自同一 float(p[0])）
    assert df.height > 0, "推理结果为空"
    signals = df["signal"].to_list()
    prob_tps = df["prob_tp"].to_list()
    for i, (s, p) in enumerate(zip(signals, prob_tps, strict=True)):
        assert s == p, f"第 {i} 行: signal={s} ≠ prob_tp={p}"

    # 附加：四概率行和 ≈ 1
    prob_sum = (
        df["prob_tp"] + df["prob_sl"] + df["prob_time_up"] + df["prob_time_down"]
    ).to_list()
    for i, s in enumerate(prob_sum):
        assert abs(s - 1.0) < 1e-5, f"第 {i} 行概率和={s}，偏差过大"


# ---------------------------------------------------------------------------
# Property 5：classification / regression 输出恰好三列，与改造前一致
# ---------------------------------------------------------------------------

# Feature: cnn-path-multiclass-head, Property 5:
# 对任意 objective ∈ {classification, regression} 的输入，
# 推理输出列（恰好三列）与改造前完全一致。


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@pytest.mark.parametrize("objective", ["classification", "regression"])
def test_property5_legacy_objectives_three_columns(
    objective: str, monkeypatch, tmp_path
) -> None:
    """Property 5：classification/regression 输出恰好三列，列序与改造前一致。"""
    from aitrade.cnn.predictor import predict_cnn_signals

    frame = _make_trading_frame()

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    name = _make_checkpoint(tmp_path, objective=objective)
    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()

    df = predict_cnn_signals(model_name=name, start=start_dt, end=end_dt)

    # 四列：原三列 + objective 末列（Task 4 cnn-eval-honesty-fixes）
    assert df.columns == ["datetime", "vt_symbol", "signal", "objective"], (
        f"objective={objective}: 列序错误 {df.columns}"
    )
    assert df.height > 0, f"objective={objective}: 推理结果为空"

    # 列类型检查：signal 为数值浮点
    assert df["signal"].dtype in (pl.Float32, pl.Float64), (
        f"signal 列类型异常: {df['signal'].dtype}"
    )

    # signal 值域检查：classification 应在 [0,1]
    if objective == "classification":
        sig_min = df["signal"].min()
        sig_max = df["signal"].max()
        assert sig_min >= 0.0, f"classification signal 出现负值: {sig_min}"
        assert sig_max <= 1.0 + 1e-6, f"classification signal 超过 1: {sig_max}"


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
def test_property5_path_class_vs_legacy_column_count(monkeypatch, tmp_path) -> None:
    """Property 5 对照：path_class 七列 vs classification 三列，形成明确对照。"""
    from aitrade.cnn.predictor import predict_cnn_signals

    frame = _make_trading_frame()

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)
    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()

    name_path = _make_checkpoint(tmp_path, objective="path_class")
    df_path = predict_cnn_signals(model_name=name_path, start=start_dt, end=end_dt)

    name_cls = _make_checkpoint(tmp_path, objective="classification")
    df_cls = predict_cnn_signals(model_name=name_cls, start=start_dt, end=end_dt)

    # path_class 八列（原七列 + objective）；classification 四列（原三列 + objective）
    # Task 4 cnn-eval-honesty-fixes：objective 末列恒追加
    assert len(df_path.columns) == 8
    assert len(df_cls.columns) == 4


# ---------------------------------------------------------------------------
# Property 7：checkpoint 往返（端到端训练→保存→加载→推理）
# ---------------------------------------------------------------------------

# Feature: cnn-path-multiclass-head, Property 7:
# 对任意训练完成的 path_class 模型，保存→加载→推理后：
# objective 恢复为 path_class、输出头 4 维、Signal_Frame 含全部七列。


def _make_synthetic_dataset_for_trainer(
    n: int = 80,
    C: int = FEATURE_CHANNELS,
    T: int = LOOKBACK,
    S: int = MAX_GROUP_WIDTH,
    G: int = GROUP_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """构造 path_class 训练的合成数据集。

    四类标签均有分布，info 键与真实 build_dataset 输出一致，
    供 train_cnn_model 正常工作。

    Args:
        n: 样本数；默认 80（满足 >=50 的 trainer 最低要求）。
        C: 特征通道数。
        T: 时间步数（lookback）。
        S: 每组最大证券数。
        G: 分组数。

    Returns:
        (X, y, group_mask, info) 四元组，与 build_dataset 签名一致。
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, C, T, S, G)).astype(np.float32)
    # 四类均匀分布（0/1/2/3）
    y = np.tile(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32), n // 4 + 1)[:n]
    group_mask = np.ones((1, 1, 1, S, G), dtype=np.float32)

    start_dt = datetime(2024, 1, 1)
    anchor_dates = [(start_dt + timedelta(days=i)).isoformat() for i in range(n)]
    info: dict[str, Any] = {
        "symbols": [TARGET],
        "groups": GROUPS,
        "target_symbol": TARGET,
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
            "mode": "oco",
            "take_profit": 0.05,
            "stop_loss": 0.05,
            "max_hold": 10,
            "stop_first": True,
            "threshold": 0.0,
            "neutral_policy": "drop",
            "price_ref": "close",
        },
        "label_threshold": 0.0,
        "price_ref": "close",
        "objective": "path_class",
        "skipped_for_label": 0,
        "skipped_for_neutral": 0,
        "sample_returns": rng.uniform(-0.05, 0.05, n).tolist(),
        "class_distribution": {
            "tp_first": int(np.sum(y == 0.0)),
            "sl_first": int(np.sum(y == 1.0)),
            "time_up": int(np.sum(y == 2.0)),
            "time_down": int(np.sum(y == 3.0)),
        },
    }
    return X, y, group_mask, info


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
def test_property7_train_save_load_predict_roundtrip(tmp_path, monkeypatch) -> None:
    """Property 7：path_class 模型 训练→保存→加载→推理 往返完整性。

    验证：
    - objective 从 checkpoint 正确恢复为 "path_class"；
    - on_meta 回传 objective=="path_class"；
    - Signal_Frame 含全部七列；
    - 逐行 signal==prob_tp；
    - 四概率行和 ≈ 1。
    """
    import aitrade.cnn.trainer as trainer_mod
    from aitrade.cnn.predictor import predict_cnn_signals

    # ---- 阶段 A：训练 ---- #
    X, y, group_mask, info = _make_synthetic_dataset_for_trainer()

    # patch build_dataset（不走真实行情 IO）
    monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))

    # patch CNN_MODEL_DIR → tmp_path；trainer 内 save_cnn_model 读 storage 模块全局
    # CNN_MODEL_DIR，因此仅替换此变量即可让真实 save_cnn_model 落盘到 tmp_path。
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)

    from aitrade.cnn.trainer import train_cnn_model

    result = train_cnn_model(
        name="prop7_roundtrip",
        vt_symbols=[TARGET],
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        epochs=2,
        batch_size=16,
        lookback=LOOKBACK,
        dropout=0.0,
        objective="path_class",
        label_spec={
            "mode": "oco",
            "take_profit": 0.05,
            "stop_loss": 0.05,
            "max_hold": 10,
        },
    )

    model_name: str = result["name"]
    assert (tmp_path / f"{model_name}.pt").exists(), "模型文件未落盘"

    # ---- 阶段 B：推理 ---- #
    frame = _make_trading_frame(rows=FRAME_ROWS)

    def fake_loader(vt_symbol, _s, _e, *, input_data_kind, input_interval):
        return frame

    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)

    start_dt = frame["datetime"][30].date()
    end_dt = frame["datetime"][-1].date()

    metas: list[dict] = []
    df = predict_cnn_signals(
        model_name=model_name,
        start=start_dt,
        end=end_dt,
        on_meta=metas.append,
    )

    # ---- 断言 ---- #
    # objective 从 on_meta 回传
    assert len(metas) == 1, "on_meta 未被调用"
    assert metas[0]["objective"] == "path_class", (
        f"objective 未恢复: {metas[0]['objective']}"
    )

    # 八列齐全（原七列 + objective 末列；Task 4 cnn-eval-honesty-fixes）
    expected_cols = ["datetime", "vt_symbol", "signal", "prob_tp", "prob_sl",
                     "prob_time_up", "prob_time_down", "objective"]
    assert df.columns == expected_cols, f"列序错误: {df.columns}"
    assert df.height > 0, "推理结果为空"

    # 逐行 signal == prob_tp
    for i, row in enumerate(df.to_dicts()):
        assert row["signal"] == row["prob_tp"], (
            f"第 {i} 行: signal={row['signal']} ≠ prob_tp={row['prob_tp']}"
        )

    # 四概率行和 ≈ 1
    prob_sum = (
        df["prob_tp"] + df["prob_sl"] + df["prob_time_up"] + df["prob_time_down"]
    ).to_list()
    for i, s in enumerate(prob_sum):
        assert abs(s - 1.0) < 1e-5, f"第 {i} 行概率和={s}，偏差 {abs(s-1.0)}"
