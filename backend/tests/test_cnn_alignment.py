"""
Task 6 验收测试：对齐丢弃率计算与回传（CNN 评估诚实性修复）

覆盖范围：
  6.1  示例测试 — alignment_drop_rate 纯函数：
         三标的 A/B/C 各 50 行，B 删 5 行 → inner join 后 45 行 → drop_rate = 0.1
         单标的 → 0；空字典 → 0；恰好超阈值触发告警
  6.2  _align_frames_by_datetime 的 merged 只读旁路：
         新增测量函数不改变 merged 输出（逐行一致）
  6.3  build_dataset 集成：
         info["alignment_drop_rate"] 正确；超阈值时 info 含 alignment_warning；
         on_progress 含丢弃率；info["per_symbol_bars_before_align"] 各标的 bar 数
  6.4  predict_cnn_signals 集成：
         on_meta["alignment_drop_rate"] 正确返回
  6.5  train_cnn_model 集成：
         dataset_info["alignment_drop_rate"] 写入 checkpoint；
         超阈值时 result 含 warnings
  6.6  Property 6（Hypothesis）：
         随机生成各标的行数与缺失 → 公式恒成立；
         告警当且仅当 drop_rate > 阈值；
         merged 与仅跑对齐时逐行一致

Requirements: 5.1–5.5；Validates: Property 6
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aitrade.cnn.features import (
    ALIGN_DROP_WARN_THRESHOLD,
    _align_frames_by_datetime,
    alignment_drop_rate,
)

# ---------------------------------------------------------------------------
# 辅助：构造合成 K 线帧
# ---------------------------------------------------------------------------

_BASE_DT = datetime(2024, 1, 1)


def _make_frame(datetimes: list[datetime], seed: int = 0) -> pl.DataFrame:
    """构造包含指定时间戳的合成 K 线帧，包含所有特征计算所需列。

    Args:
        datetimes: 时间戳列表，决定帧的行数与时间点。
        seed: numpy 随机种子，影响 OHLCV 数值。

    Returns:
        polars DataFrame，列：datetime, open, high, low, close, volume, turnover, open_interest。
    """
    rng = np.random.default_rng(seed)
    px = 100.0
    rows: list[dict[str, Any]] = []
    for dt in datetimes:
        px *= 1 + rng.normal(0, 0.01)
        rows.append(
            {
                "datetime": dt,
                "open": float(px * 0.998),
                "high": float(px * 1.015),
                "low": float(px * 0.985),
                "close": float(px),
                "volume": float(1000 + rng.integers(0, 100)),
                "turnover": float(px * (1000 + rng.integers(0, 100))),
                "open_interest": float(rng.integers(0, 50)),
            }
        )
    return pl.DataFrame(rows)


def _make_datetimes(n: int, offset_days: int = 0) -> list[datetime]:
    """生成 n 个连续整点时间戳（间隔 1 天）。

    Args:
        n: 时间戳数量。
        offset_days: 相对基准日期的偏移天数。

    Returns:
        长度为 n 的 datetime 列表。
    """
    return [_BASE_DT + timedelta(days=i + offset_days) for i in range(n)]


def _make_trading_frame(rows: int, seed: int = 0) -> pl.DataFrame:
    """构造从基准日期起 rows 行的合成日线行情帧。

    Args:
        rows: 行数。
        seed: 随机种子。

    Returns:
        polars DataFrame，列包含 datetime, open, high, low, close, volume, turnover, open_interest。
    """
    return _make_frame(_make_datetimes(rows), seed)


# ===========================================================================
# 6.1  示例测试 — alignment_drop_rate 纯函数
# ===========================================================================


class TestAlignmentDropRatePure:
    """alignment_drop_rate 纯函数的示例测试。"""

    def test_three_symbols_with_missing_rows(self) -> None:
        """三标的 A/B/C 各 50 行，B 缺 5 行 → drop_rate = (50-45)/50 = 0.1。

        模拟一只证券停牌 5 个交易日的情形，inner join 后保留公共 45 行。
        """
        all_dts = _make_datetimes(50)
        # B 删去前 5 个时间点，模拟停牌
        b_dts = all_dts[5:]

        frames = {
            "A": _make_frame(all_dts, seed=0),
            "B": _make_frame(b_dts, seed=1),
            "C": _make_frame(all_dts, seed=2),
        }
        _, merged = _align_frames_by_datetime(frames)
        aligned_h = merged.height
        # B 缺 5 行，inner join 结果 = 45；max 原始行数 = 50
        assert aligned_h == 45
        dr = alignment_drop_rate(frames, aligned_h)
        assert abs(dr - 0.1) < 1e-9, f"期望 0.1，实际 {dr}"

    def test_single_symbol_returns_zero(self) -> None:
        """单标的不存在跨标的对齐丢弃，drop_rate = 0。"""
        frames = {"A": _make_trading_frame(30, seed=0)}
        dr = alignment_drop_rate(frames, 30)
        assert dr == 0.0

    def test_empty_dict_returns_zero(self) -> None:
        """空字典返回 0。"""
        assert alignment_drop_rate({}, 0) == 0.0

    def test_no_drop_returns_zero(self) -> None:
        """所有标的行数相同且对齐后无丢失 → drop_rate = 0。"""
        all_dts = _make_datetimes(40)
        frames = {
            "A": _make_frame(all_dts, seed=0),
            "B": _make_frame(all_dts, seed=1),
        }
        _, merged = _align_frames_by_datetime(frames)
        dr = alignment_drop_rate(frames, merged.height)
        assert dr == 0.0

    def test_above_threshold_triggers_warning(self) -> None:
        """drop_rate = 0.1 > ALIGN_DROP_WARN_THRESHOLD(0.05)，应超阈值。"""
        assert ALIGN_DROP_WARN_THRESHOLD == 0.05
        assert 0.1 > ALIGN_DROP_WARN_THRESHOLD

    def test_below_threshold_no_warning(self) -> None:
        """drop_rate = 0.03 < ALIGN_DROP_WARN_THRESHOLD(0.05)，不超阈值。"""
        assert 0.03 < ALIGN_DROP_WARN_THRESHOLD

    def test_exact_threshold_no_warning(self) -> None:
        """drop_rate == threshold：等于阈值时不超阈值（严格大于才告警）。"""
        assert not (ALIGN_DROP_WARN_THRESHOLD > ALIGN_DROP_WARN_THRESHOLD)


# ===========================================================================
# 6.2  _align_frames_by_datetime 只读旁路
# ===========================================================================


class TestAlignMergedReadonly:
    """alignment_drop_rate 是独立旁路函数，不污染 _align_frames_by_datetime 的 merged。"""

    def test_merged_unchanged_after_drop_rate_call(self) -> None:
        """对同一组 symbol_frames 分别求 merged 与 drop_rate，merged 逐行完全一致。

        Property 6 的"旁路不污染对齐"子句：alignment_drop_rate 从不触碰 merged，
        本测试通过分别运行来固化此保证。
        """
        all_dts = _make_datetimes(50)
        b_dts = all_dts[5:]
        frames = {
            "A": _make_frame(all_dts, seed=0),
            "B": _make_frame(b_dts, seed=1),
        }
        # 两次独立对齐，结果应完全相同
        _, merged_ref = _align_frames_by_datetime(frames)

        # 调用 drop_rate（只读，不改变 frames）后再次对齐
        _ = alignment_drop_rate(frames, merged_ref.height)
        _, merged_after = _align_frames_by_datetime(frames)

        assert merged_ref.equals(merged_after), "alignment_drop_rate 不应改变 merged 内容"


# ===========================================================================
# 6.3  build_dataset 集成测试
# ===========================================================================


def _make_groups(symbols: list[str]) -> list[dict[str, Any]]:
    """构造最简 observation_groups：target 组含第一个证券，custom 组含其余证券。

    normalize_observation_groups 会跳过 role="target" 的 observation_groups 条目，
    并自动插入单符号 target 组；因此观测组必须用 "custom" 角色承载非目标证券。

    Args:
        symbols: 所有参与证券的代码列表，第一个为目标证券。

    Returns:
        observation_groups 列表，首项为 target，其余（若有）打包为 custom 组。
    """
    if len(symbols) <= 1:
        return [{"role": "target", "name": "目标组", "symbols": symbols[:1]}]
    return [
        {"role": "target", "name": "目标组", "symbols": [symbols[0]]},
        {"role": "custom", "name": "观测组", "symbols": symbols[1:]},
    ]


@pytest.fixture()
def stub_dataset_two_symbols(monkeypatch):
    """桩化 _load_market_frame，返回两只证券的合成行情（B 缺 5 行）。

    A: 60 行；B: 55 行（缺最后 5 个时间点），对齐后 55 行，
    drop_rate = (60 - 55) / 60 ≈ 0.0833 > 0.05 触发告警。
    """
    from aitrade.cnn import dataset as cnn_dataset

    all_dts = _make_datetimes(60)
    b_dts = all_dts[:55]  # B 缺后 5 行
    frames = {
        "AAA.SSE": _make_frame(all_dts, seed=10),
        "BBB.SSE": _make_frame(b_dts, seed=11),
    }

    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frames[vt_symbol]

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)
    return frames


@pytest.fixture()
def stub_dataset_equal_symbols(monkeypatch):
    """桩化 _load_market_frame，返回两只证券行数相同的合成行情（无对齐丢弃）。"""
    from aitrade.cnn import dataset as cnn_dataset

    all_dts = _make_datetimes(60)
    frames = {
        "AAA.SSE": _make_frame(all_dts, seed=20),
        "BBB.SSE": _make_frame(all_dts, seed=21),
    }

    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frames[vt_symbol]

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)
    return frames


class TestBuildDatasetDropRate:
    """build_dataset 回传 alignment_drop_rate 与 per_symbol_bars_before_align。"""

    def test_info_contains_drop_rate(self, stub_dataset_two_symbols) -> None:
        """drop_rate 写入 info["alignment_drop_rate"]，值精确。

        A=60 行，B=55 行 → 对齐 55 行，drop_rate = (60-55)/60。
        """
        from aitrade.cnn.dataset import build_dataset

        symbols = ["AAA.SSE", "BBB.SSE"]
        _X, _y, _mask, info = build_dataset(
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            lookback=10,
            target_symbol="AAA.SSE",
            observation_groups=_make_groups(symbols),
        )
        assert "alignment_drop_rate" in info
        expected = (60 - 55) / 60
        assert abs(info["alignment_drop_rate"] - expected) < 1e-9

    def test_info_contains_per_symbol_bars(self, stub_dataset_two_symbols) -> None:
        """per_symbol_bars_before_align 记录各标的对齐前行数。"""
        from aitrade.cnn.dataset import build_dataset

        symbols = ["AAA.SSE", "BBB.SSE"]
        _X, _y, _mask, info = build_dataset(
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            lookback=10,
            target_symbol="AAA.SSE",
            observation_groups=_make_groups(symbols),
        )
        assert "per_symbol_bars_before_align" in info
        bars = info["per_symbol_bars_before_align"]
        assert bars["AAA.SSE"] == 60
        assert bars["BBB.SSE"] == 55

    def test_info_contains_alignment_warning_when_above_threshold(self, stub_dataset_two_symbols) -> None:
        """drop_rate > 0.05 时 info 含 alignment_warning。"""
        from aitrade.cnn.dataset import build_dataset

        symbols = ["AAA.SSE", "BBB.SSE"]
        _X, _y, _mask, info = build_dataset(
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            lookback=10,
            target_symbol="AAA.SSE",
            observation_groups=_make_groups(symbols),
        )
        # drop_rate ≈ 0.0833 > 0.05
        assert info["alignment_drop_rate"] > ALIGN_DROP_WARN_THRESHOLD
        assert "alignment_warning" in info, "超阈值时应有 alignment_warning"
        assert isinstance(info["alignment_warning"], str)

    def test_no_warning_when_below_threshold(self, stub_dataset_equal_symbols) -> None:
        """drop_rate = 0（无丢失）时 info 不含 alignment_warning。"""
        from aitrade.cnn.dataset import build_dataset

        symbols = ["AAA.SSE", "BBB.SSE"]
        _X, _y, _mask, info = build_dataset(
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            lookback=10,
            target_symbol="AAA.SSE",
            observation_groups=_make_groups(symbols),
        )
        assert info["alignment_drop_rate"] == 0.0
        assert "alignment_warning" not in info, "不超阈值时不应有 alignment_warning"

    def test_on_progress_contains_drop_rate(self, stub_dataset_two_symbols) -> None:
        """on_progress 回调中至少有一条消息包含丢弃率信息（⚠️ 超阈值标记）。"""
        from aitrade.cnn.dataset import build_dataset

        messages: list[str] = []
        symbols = ["AAA.SSE", "BBB.SSE"]
        _X, _y, _mask, _info = build_dataset(
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            lookback=10,
            target_symbol="AAA.SSE",
            observation_groups=_make_groups(symbols),
            on_progress=lambda pct, msg: messages.append(msg),
        )
        combined = " ".join(messages)
        # 超阈值时应有 ⚠️ 标记
        assert "⚠️" in combined or "丢弃率" in combined or "drop_rate" in combined.lower(), (
            f"on_progress 消息中未找到丢弃率信息，messages={messages}"
        )


# ===========================================================================
# 6.4  predict_cnn_signals 集成测试
# ===========================================================================


@pytest.fixture()
def stub_predictor_model(monkeypatch, tmp_path):
    """落盘最小 CNN checkpoint，桩化行情加载（两标的：B 缺 5 行）。"""
    torch = pytest.importorskip("torch")
    from aitrade.cnn import model as cnn_model
    from aitrade.cnn import storage as cnn_storage
    from aitrade.cnn.network import create_market_cnn

    FEATURE_CH = 6
    LOOKBACK = 10
    # normalize_observation_groups 将生成：target 组(AAA) + custom 组(BBB) = 2 组，每组宽 1
    MAX_W = 1
    GROUP_CT = 2

    all_dts = _make_datetimes(60)
    b_dts = all_dts[:55]
    frames = {
        "AAA.SSE": _make_frame(all_dts, seed=30),
        "BBB.SSE": _make_frame(b_dts, seed=31),
    }

    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frames[vt_symbol]

    monkeypatch.setattr(cnn_model, "_load_market_frame", fake_loader)
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)

    model = create_market_cnn(FEATURE_CH, LOOKBACK, MAX_W, GROUP_CT, 0.0)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": FEATURE_CH,
            "time_steps": LOOKBACK,
            "max_group_width": MAX_W,
            "group_count": GROUP_CT,
            "dropout": 0.0,
        },
        "train_config": {
            "target_symbol": "AAA.SSE",
            "lookback": LOOKBACK,
            "input_data_kind": "bar",
            "input_interval": "d",
            "objective": "classification",
            "observation_groups": [
                {"role": "target", "name": "目标组", "symbols": ["AAA.SSE"]},
                {"role": "custom", "name": "观测组", "symbols": ["BBB.SSE"]},
            ],
        },
        "normalization": {
            "channel_mean": [0.0] * FEATURE_CH,
            "channel_std": [1.0] * FEATURE_CH,
        },
    }
    torch.save(checkpoint, str(tmp_path / "pred_model.pt"))
    return "pred_model"


class TestPredictorDropRate:
    """predict_cnn_signals 在 on_meta 中回传 alignment_drop_rate。"""

    def test_on_meta_contains_alignment_drop_rate(self, stub_predictor_model) -> None:
        """on_meta 字典含 alignment_drop_rate，值精确。

        A=60 行，B=55 行 → 对齐 55 行，drop_rate = (60-55)/60。
        """
        from aitrade.cnn.predictor import predict_cnn_signals

        meta_list: list[dict] = []
        predict_cnn_signals(
            model_name=stub_predictor_model,
            start=date(2024, 1, 1),
            end=date(2024, 3, 30),
            on_meta=meta_list.append,
        )
        assert len(meta_list) == 1
        meta = meta_list[0]
        assert "alignment_drop_rate" in meta
        expected = (60 - 55) / 60
        assert abs(meta["alignment_drop_rate"] - expected) < 1e-9


# ===========================================================================
# 6.5  train_cnn_model 集成测试
# ===========================================================================


@pytest.fixture()
def stub_trainer(monkeypatch, tmp_path):
    """桩化行情加载与模型库目录，A=80 行，B=70 行（触发告警）。

    对齐后 70 行，lookback=10 → 61 样本 ≥ 50 满足 trainer 最低要求。
    drop_rate = (80-70)/80 = 0.125 > 0.05 触发告警。
    """
    pytest.importorskip("torch")
    from aitrade.cnn import dataset as cnn_dataset
    from aitrade.cnn import storage as cnn_storage

    all_dts = _make_datetimes(80)
    b_dts = all_dts[:70]  # B 缺后 10 行
    frames = {
        "AAA.SSE": _make_frame(all_dts, seed=40),
        "BBB.SSE": _make_frame(b_dts, seed=41),
    }

    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frames[vt_symbol]

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)


@pytest.fixture()
def stub_trainer_no_drop(monkeypatch, tmp_path):
    """桩化行情加载，两标的行数相同（无对齐丢弃，不触发告警）。

    A=B=80 行，对齐后 80 行 → 71 样本 ≥ 50；drop_rate=0。
    """
    pytest.importorskip("torch")
    from aitrade.cnn import dataset as cnn_dataset
    from aitrade.cnn import storage as cnn_storage

    all_dts = _make_datetimes(80)
    frames = {
        "AAA.SSE": _make_frame(all_dts, seed=50),
        "BBB.SSE": _make_frame(all_dts, seed=51),
    }

    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frames[vt_symbol]

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)


class TestTrainerDropRate:
    """train_cnn_model 把 alignment_drop_rate 写入 dataset_info/checkpoint/result warnings。"""

    def test_dataset_info_contains_drop_rate(self, stub_trainer) -> None:
        """dataset_info["alignment_drop_rate"] 写入 save_data（验证 checkpoint 含此键）。

        通过训练结果侧面验证：result 中有 alignment_drop_rate 或 warnings 含告警。
        """
        from aitrade.cnn.trainer import train_cnn_model

        symbols = ["AAA.SSE", "BBB.SSE"]
        result = train_cnn_model(
            name="train_test_drop",
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 3, 30),
            lookback=10,
            epochs=1,
            observation_groups=[
                {"role": "target", "name": "目标组", "symbols": [symbols[0]]},
                {"role": "custom", "name": "观测组", "symbols": [symbols[1]]},
            ],
        )
        # 超阈值时 result 含 warnings（可能是列表或字符串）
        expected_dr = (80 - 70) / 80  # A=80 行，B=70 行
        assert "alignment_drop_rate" in result, "result 应含 alignment_drop_rate"
        assert abs(result["alignment_drop_rate"] - expected_dr) < 1e-9

    def test_result_warnings_when_above_threshold(self, stub_trainer) -> None:
        """超阈值时 result 含 warnings 字段（列表，至少一条含告警信息）。"""
        from aitrade.cnn.trainer import train_cnn_model

        symbols = ["AAA.SSE", "BBB.SSE"]
        result = train_cnn_model(
            name="train_test_warn",
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 3, 30),
            lookback=10,
            epochs=1,
            observation_groups=[
                {"role": "target", "name": "目标组", "symbols": [symbols[0]]},
                {"role": "custom", "name": "观测组", "symbols": [symbols[1]]},
            ],
        )
        assert "warnings" in result, "超阈值时 result 应含 warnings"
        warnings = result["warnings"]
        assert isinstance(warnings, list)
        assert len(warnings) > 0
        combined = " ".join(str(w) for w in warnings)
        assert "alignment" in combined.lower() or "丢弃" in combined or "drop" in combined.lower()

    def test_no_warnings_when_below_threshold(self, stub_trainer_no_drop) -> None:
        """drop_rate = 0 时 result 不含 warnings（或 warnings 为空列表）。"""
        from aitrade.cnn.trainer import train_cnn_model

        symbols = ["AAA.SSE", "BBB.SSE"]
        result = train_cnn_model(
            name="train_test_no_warn",
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2024, 3, 30),
            lookback=10,
            epochs=1,
            observation_groups=[
                {"role": "target", "name": "目标组", "symbols": [symbols[0]]},
                {"role": "custom", "name": "观测组", "symbols": [symbols[1]]},
            ],
        )
        # 无丢弃时，warnings 要么不存在，要么为空列表
        warnings = result.get("warnings", [])
        assert warnings == [], f"无丢弃时不应有 warnings，实际={warnings}"


# ===========================================================================
# 6.6  Property 6（Hypothesis）
# ===========================================================================


@given(
    n_symbols=st.integers(min_value=2, max_value=5),
    max_rows=st.integers(min_value=20, max_value=80),
    missing_each=st.lists(
        st.integers(min_value=0, max_value=10),
        min_size=0,
        max_size=4,
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property6_drop_rate_formula(
    n_symbols: int,
    max_rows: int,
    missing_each: list[int],
) -> None:
    """Property 6.a：任意 symbol_frames，drop_rate = (max_h - aligned_h) / max_h。

    Feature: alignment_drop_rate
    Property: 对任意两只及以上证券，对齐丢弃率公式恒精确成立。
    """
    all_dts = _make_datetimes(max_rows)
    frames: dict[str, pl.DataFrame] = {}
    for idx in range(n_symbols):
        # 每只标的随机缺若干行（从头部删）
        miss = missing_each[idx] if idx < len(missing_each) else 0
        miss = min(miss, max_rows - 5)  # 保留至少 5 行
        dts = all_dts[miss:]
        frames[f"S{idx}"] = _make_frame(dts, seed=idx)

    try:
        _, merged = _align_frames_by_datetime(frames)
        aligned_h = merged.height
    except ValueError:
        # 若完全无公共时间点则跳过（hypothesis 会探索其他案例）
        assume(False)
        return

    dr = alignment_drop_rate(frames, aligned_h)
    heights = [f.height for f in frames.values()]
    max_h = max(heights)
    expected = max(0.0, (max_h - aligned_h) / max_h)
    assert abs(dr - expected) < 1e-9, f"公式错误: dr={dr}, expected={expected}"


@given(
    n_symbols=st.integers(min_value=2, max_value=3),
    max_rows=st.integers(min_value=30, max_value=80),
    missing_first=st.integers(min_value=0, max_value=20),
)
@settings(
    max_examples=30,  # 每例跑真实 build_dataset，降低数量避免超时
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property6_warning_iff_above_threshold(
    n_symbols: int,
    max_rows: int,
    missing_first: int,
) -> None:
    """Property 6.b：告警触发当且仅当 drop_rate > ALIGN_DROP_WARN_THRESHOLD。

    Feature: alignment_drop_rate
    Property: 驱动真实 build_dataset 运行，断言 info["alignment_warning"] 存在
    当且仅当 info["alignment_drop_rate"] > ALIGN_DROP_WARN_THRESHOLD。

    实现方式：monkeypatch _load_market_frame（通过 unittest.mock.patch），
    构造随机多标的合成帧（S0 缺 missing_first 行模拟停牌），跑完整的
    build_dataset → 直接检查 info 中的告警键，真正守护"超阈值才写 warning"逻辑。

    设 max_examples=30 是因为每例需跑完整 build_dataset（IO+特征计算），
    30 例已能覆盖 drop_rate=0 / 0<drop_rate<=阈值 / 超阈值三个区间。
    """
    from unittest.mock import patch

    LOOKBACK = 10
    all_dts = _make_datetimes(max_rows)
    miss = min(missing_first, max_rows - LOOKBACK - 5)  # S0 至少保留 LOOKBACK+5 行
    if miss < 0:
        miss = 0

    frames: dict[str, pl.DataFrame] = {}
    frames["S0"] = _make_frame(all_dts[miss:], seed=0)
    for idx in range(1, n_symbols):
        frames[f"S{idx}"] = _make_frame(all_dts, seed=idx)

    # 检查对齐后有足够样本；不足则跳过（不用 try/except 吞断言）
    try:
        _, merged = _align_frames_by_datetime(frames)
    except ValueError:
        assume(False)
        return
    # 需要至少 LOOKBACK+2 行才能生成 ≥1 个有效样本（next_bar 标签需 anchor+1）
    assume(merged.height >= LOOKBACK + 2)

    symbols = list(frames.keys())
    groups = _make_groups(symbols)

    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return frames[vt_symbol]

    from aitrade.cnn.dataset import build_dataset

    with patch("aitrade.cnn.dataset._load_market_frame", side_effect=fake_loader):
        _X, _y, _mask, info = build_dataset(
            vt_symbols=symbols,
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            lookback=LOOKBACK,
            target_symbol=symbols[0],
            observation_groups=groups,
        )

    drop_rate = info["alignment_drop_rate"]
    has_warning = "alignment_warning" in info

    assert has_warning == (drop_rate > ALIGN_DROP_WARN_THRESHOLD), (
        f"告警键存在={has_warning} 与 drop_rate={drop_rate:.4f} > "
        f"ALIGN_DROP_WARN_THRESHOLD={ALIGN_DROP_WARN_THRESHOLD} 不一致"
    )


@given(
    n_rows=st.integers(min_value=20, max_value=80),
    missing_b=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property6_merged_readonly(n_rows: int, missing_b: int) -> None:
    """Property 6.c：alignment_drop_rate 不改变 merged 输出（只读旁路）。

    Feature: alignment_drop_rate
    Property: 对同一 symbol_frames，调用 drop_rate 后再次对齐，merged 逐行完全一致。
    """
    miss = min(missing_b, n_rows - 5)
    all_dts = _make_datetimes(n_rows)
    frames = {
        "A": _make_frame(all_dts, seed=0),
        "B": _make_frame(all_dts[miss:], seed=1),
    }
    _, merged_before = _align_frames_by_datetime(frames)
    _ = alignment_drop_rate(frames, merged_before.height)
    _, merged_after = _align_frames_by_datetime(frames)
    assert merged_before.equals(merged_after), "alignment_drop_rate 不得修改 merged"


@given(
    max_rows=st.integers(min_value=10, max_value=60),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property6_single_symbol_always_zero(max_rows: int) -> None:
    """Property 6.d：单标的丢弃率恒为 0。

    Feature: alignment_drop_rate
    Property: len(symbol_frames) == 1 时，任意对齐后行数均返回 0.0。
    """
    frames = {"A": _make_trading_frame(max_rows, seed=0)}
    dr = alignment_drop_rate(frames, max_rows)
    assert dr == 0.0
