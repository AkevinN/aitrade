"""
Task 13.1 验收测试：`predict_cnn_signals` 的可选 `on_meta` 回调（推理可观测信息采集）。

覆盖：
1. 既有调用方**不传** `on_meta` 时行为不变 —— 返回的 `signal_df` 仍满足原契约
   （列 `[datetime, vt_symbol, signal]`，目标标的一致）。
2. 传入收集器时，一次性吐出结构化推理元信息：字段齐全、`per_symbol_bars`
   与各证券**原始 bar 数**一致、且**不含任何凭证**。

外部 I/O 桩化：`_load_market_frame` 返回合成行情，模型库目录用 `tmp_path`，
模型权重由 `create_market_cnn` 现场构建后落盘 —— 不依赖真实行情/已训练模型。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from aitrade.cnn import model as cnn_model
from aitrade.cnn import storage as cnn_storage
from aitrade.cnn.predictor import predict_cnn_signals

# --- 测试常量 -------------------------------------------------------------
TARGET = "AAA.SSE"
OBS_A = "BBB.SSE"
OBS_B = "CCC.SSE"
LOOKBACK = 10
FEATURE_CHANNELS = 6
GROUP_COUNT = 2
MAX_GROUP_WIDTH = 2
START = date(2024, 1, 1)
END = date(2024, 3, 1)

# 各证券原始 bar 数刻意取不同值，验证 per_symbol_bars 反映「原始帧」行数
BAR_COUNTS = {TARGET: 40, OBS_A: 42, OBS_B: 40}

GROUPS = [
    {"role": "target", "name": "目标证券", "symbols": [TARGET]},
    {"role": "custom", "name": "观测组", "symbols": [OBS_A, OBS_B]},
]

# 注入到模型库目录的哨兵：用于断言 meta 中绝不出现凭证
SENTINEL_TOKEN = "tushare_secret_token_should_never_leak"


def _make_frame(count: int, seed: int) -> pl.DataFrame:
    """构造一段连续日线合成行情（含计算 6 通道特征所需的 OHLCV）。"""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    px = 100.0
    for index in range(count):
        px *= 1 + rng.normal(0, 0.02)
        rows.append(
            {
                "datetime": datetime(2024, 1, 1) + timedelta(days=index),
                "open": px * 0.997,
                "high": px * 1.02,
                "low": px * 0.98,
                "close": px,
                "volume": 1_000.0 + index,
                "turnover": (1_000.0 + index) * px,
                "open_interest": float(index),
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def stub_model(monkeypatch, tmp_path):
    """落盘一个最小可用 CNN checkpoint，并桩化行情加载 + 模型库目录。"""
    torch = pytest.importorskip("torch")
    from aitrade.cnn.network import create_market_cnn

    frames = {sym: _make_frame(count, seed) for seed, (sym, count) in enumerate(BAR_COUNTS.items())}

    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frames[vt_symbol]

    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)

    model = create_market_cnn(
        FEATURE_CHANNELS, LOOKBACK, MAX_GROUP_WIDTH, GROUP_COUNT, 0.5, objective="classification"
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": FEATURE_CHANNELS,
            "time_steps": LOOKBACK,
            "max_group_width": MAX_GROUP_WIDTH,
            "group_count": GROUP_COUNT,
            "dropout": 0.5,
        },
        "train_config": {
            "target_symbol": TARGET,
            "lookback": LOOKBACK,
            "input_data_kind": "bar",
            "input_interval": "d",
            "objective": "classification",
            "observation_groups": GROUPS,
            # 模拟一个携带凭证的训练配置：on_meta 绝不能把它泄漏出去
            "secret_token": SENTINEL_TOKEN,
        },
        "normalization": {
            "channel_mean": [0.0] * FEATURE_CHANNELS,
            "channel_std": [1.0] * FEATURE_CHANNELS,
        },
    }
    torch.save(checkpoint, str(tmp_path / "testmodel.pt"))
    return "testmodel"


def test_without_on_meta_preserves_original_contract(stub_model) -> None:
    """不传 on_meta：返回 signal_df 与原契约一致（列与目标标的不变）。"""
    signal_df = predict_cnn_signals(model_name=stub_model, start=START, end=END)

    assert isinstance(signal_df, pl.DataFrame)
    # classification 输出四列（原三列 + objective 末列；Task 4 cnn-eval-honesty-fixes）
    assert signal_df.columns == ["datetime", "vt_symbol", "signal", "objective"]
    assert signal_df.height > 0
    assert signal_df["vt_symbol"].unique().to_list() == [TARGET]
    # 推理点数 = total_steps - lookback + 1（区间覆盖全部对齐时间步）
    common_steps = min(BAR_COUNTS.values())
    assert signal_df.height == common_steps - LOOKBACK + 1


def test_on_meta_emits_complete_inference_meta(stub_model) -> None:
    """传入收集器：meta 字段齐全、per_symbol_bars 与各证券 bar 数一致、无凭证。"""
    collected: list[dict] = []

    signal_df = predict_cnn_signals(
        model_name=stub_model, start=START, end=END, on_meta=collected.append
    )

    # on_meta 仅触发一次
    assert len(collected) == 1
    meta = collected[0]

    expected_keys = {
        "target_symbol",
        "lookback",
        "input_interval",
        "objective",
        "observation_symbols",
        "observation_group_count",
        "warmup_start",
        "total_steps",
        "valid_points",
        "per_symbol_bars",
        "alignment_drop_rate",  # Task 6：对齐丢弃率回传
    }
    assert set(meta.keys()) == expected_keys

    assert meta["target_symbol"] == TARGET
    assert meta["lookback"] == LOOKBACK
    assert meta["input_interval"] == "d"
    assert meta["objective"] == "classification"
    assert meta["observation_symbols"] == [TARGET, OBS_A, OBS_B]
    assert meta["observation_group_count"] == len(GROUPS)

    # warmup_start == extended_start.isoformat()，其中 extended_start = start - lookback*2.5 天
    expected_warmup = (START - timedelta(days=int(LOOKBACK * 2.5))).isoformat()
    assert meta["warmup_start"] == expected_warmup

    # total_steps = 对齐后的公共时间步；valid_points = 落在区间内的推理点数
    common_steps = min(BAR_COUNTS.values())
    assert meta["total_steps"] == common_steps
    assert meta["valid_points"] == signal_df.height == common_steps - LOOKBACK + 1

    # per_symbol_bars 反映各证券「原始帧」行数（区别于对齐后的公共步数）
    assert meta["per_symbol_bars"] == BAR_COUNTS

    # 脱敏红线：meta 序列化后绝不含任何凭证
    serialized = json.dumps(meta, ensure_ascii=False, default=str)
    assert SENTINEL_TOKEN not in serialized
    assert "token" not in serialized.lower()
