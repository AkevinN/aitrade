"""CNN 回归模式（方案A）回归测试：标签、网络头、评估指标。

均为快速单元测试，不做实际训练。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from aitrade.cnn import dataset as cnn_dataset
from aitrade.cnn.network import create_market_cnn
from aitrade.cnn.trainer import _regression_metrics, _pearson, _rankdata


def _make_frame(start: datetime, count: int) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    px = 100.0
    for index in range(count):
        px *= 1 + rng.normal(0, 0.02)
        rows.append(
            {
                "datetime": start + timedelta(days=index),
                "open": px * 0.997,
                "high": px * 1.01,
                "low": px * 0.99,
                "close": px,
                "volume": 1_000 + index,
                "turnover": (1_000 + index) * px,
                "open_interest": float(index),
            }
        )
    return pl.DataFrame(rows)


def test_build_dataset_regression_labels_are_continuous(monkeypatch) -> None:
    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return _make_frame(datetime(2024, 1, 1), 80)

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)

    X, y, _mask, info = cnn_dataset.build_dataset(
        vt_symbols=["AAA.SSE"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 31),
        lookback=10,
        target_symbol="AAA.SSE",
        input_interval="d",
        label_spec={"mode": "next_bar", "price_ref": "next_open"},
        objective="regression",
    )

    assert info["objective"] == "regression"
    # 回归标签是连续收益，不应只取 {0,1}
    assert not set(np.unique(y).tolist()).issubset({0.0, 1.0})
    # 样本收益与标签同序、同长
    assert len(info["sample_returns"]) == len(y) == X.shape[0]
    assert np.allclose(np.asarray(info["sample_returns"], dtype=np.float32), y, atol=1e-5)


def test_regression_threshold_drops_small_moves(monkeypatch) -> None:
    def fake_loader(vt_symbol, start, end, *, input_data_kind, input_interval):
        return _make_frame(datetime(2024, 1, 1), 80)

    monkeypatch.setattr(cnn_dataset, "_load_market_frame", fake_loader)

    _, y, _mask, info = cnn_dataset.build_dataset(
        vt_symbols=["AAA.SSE"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 31),
        lookback=10,
        target_symbol="AAA.SSE",
        input_interval="d",
        label_spec={"mode": "next_bar", "threshold": 0.01},
        objective="regression",
    )
    # 阈值去噪后，保留样本的 |收益| 都应大于阈值
    assert np.all(np.abs(y) > 0.01 - 1e-9)
    assert info["skipped_for_neutral"] >= 0


def test_regression_network_head_has_no_sigmoid() -> None:
    torch = pytest.importorskip("torch")
    reg = create_market_cnn(6, 10, 2, 3, 0.5, objective="regression")
    cls = create_market_cnn(6, 10, 2, 3, 0.5, objective="classification")
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in reg.modules())
    assert any(isinstance(m, torch.nn.Sigmoid) for m in cls.modules())


def test_regression_metrics_perfect_and_reversed() -> None:
    y_true = np.array([-0.02, -0.01, 0.01, 0.03])
    up_ratio = float(np.mean(y_true > 0))

    perfect = _regression_metrics(y_true, y_true.copy(), up_ratio)
    assert perfect["ic"] == 1.0
    assert perfect["rank_ic"] == 1.0
    assert perfect["dir_acc"] == 1.0
    assert perfect["mae"] == 0.0

    reversed_pred = _regression_metrics(y_true, -y_true, up_ratio)
    assert reversed_pred["ic"] == -1.0


def test_pearson_and_rankdata_edge_cases() -> None:
    # 常数序列方差为 0 → 相关无定义返回 None
    assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    # 并列取平均秩
    ranks = _rankdata([10.0, 10.0, 20.0])
    assert ranks[0] == ranks[1] == 1.5
    assert ranks[2] == 3.0
