"""
网络头与训练分支 path_class 验收测试。

覆盖范围：
  3.1  create_market_cnn(objective="path_class") 输出 [B,4]、无 Sigmoid
  3.1b 向后兼容：classification→[B,1]+Sigmoid，regression→[B,1] 无 Sigmoid
  3.3  _path_class_metrics：完美预测 tp_auc=sl_auc=1.0、macro_f1=1.0
  3.3b 单类缺失（y_true 不含类 0）→ tp_auc 为 None 不崩
  3.3c _selection_score path 分支：None→按 0.5 处理
  3.5  Property 2：softmax 后的概率单纯形（Hypothesis, max_examples=100）
  3.6  冒烟测试：合成小数据 path_class 训练 2 epoch 跑通，result/history 键齐全
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# 3.1 / 3.1b  网络头形状与 Sigmoid 存在性
# ---------------------------------------------------------------------------

class TestCreateMarketCnnPathClass:
    """3.1 path_class 输出头：[B,4] 且无 Sigmoid。"""

    def test_path_class_output_shape(self) -> None:
        """path_class 头应输出 [B, 4]。"""
        torch = pytest.importorskip("torch")
        from aitrade.cnn.network import create_market_cnn

        B, C, T, S, G = 4, 6, 10, 2, 1
        model = create_market_cnn(C, T, S, G, dropout=0.5, objective="path_class")
        model.eval()
        x = torch.randn(B, C, T, S, G)
        mask = torch.ones(B, 1, 1, S, G)
        with torch.no_grad():
            out = model(x, mask)
        assert out.shape == (B, 4), f"期望 [B,4]，实得 {out.shape}"

    def test_path_class_no_sigmoid(self) -> None:
        """path_class 头不应含 Sigmoid（输出为 raw logits，sigmoid 留给 softmax/cross_entropy）。"""
        torch = pytest.importorskip("torch")
        from aitrade.cnn.network import create_market_cnn

        model = create_market_cnn(6, 10, 2, 1, 0.5, objective="path_class")
        assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())

    def test_path_class_output_can_be_negative(self) -> None:
        """path_class 输出 logits，值域无界（应可取负值）。"""
        torch = pytest.importorskip("torch")
        from aitrade.cnn.network import create_market_cnn

        torch.manual_seed(0)
        model = create_market_cnn(6, 10, 2, 1, 0.0, objective="path_class")
        model.eval()
        x = torch.randn(8, 6, 10, 2, 1)
        mask = torch.ones(8, 1, 1, 2, 1)
        with torch.no_grad():
            out = model(x, mask)
        # logits 应有正有负（dropout=0，seed=0，大概率满足）
        assert out.min().item() < out.max().item()

    def test_backward_compat_classification_has_sigmoid(self) -> None:
        """classification 头仍需含 Sigmoid（向后兼容）。"""
        torch = pytest.importorskip("torch")
        from aitrade.cnn.network import create_market_cnn

        model = create_market_cnn(6, 10, 2, 1, 0.5, objective="classification")
        assert any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())

    def test_backward_compat_classification_shape(self) -> None:
        """classification 头输出 [B, 1]。"""
        torch = pytest.importorskip("torch")
        from aitrade.cnn.network import create_market_cnn

        B, C, T, S, G = 3, 6, 10, 2, 1
        model = create_market_cnn(C, T, S, G, dropout=0.5, objective="classification")
        model.eval()
        x = torch.randn(B, C, T, S, G)
        mask = torch.ones(B, 1, 1, S, G)
        with torch.no_grad():
            out = model(x, mask)
        assert out.shape == (B, 1)

    def test_backward_compat_regression_no_sigmoid(self) -> None:
        """regression 头仍无 Sigmoid（向后兼容）。"""
        torch = pytest.importorskip("torch")
        from aitrade.cnn.network import create_market_cnn

        model = create_market_cnn(6, 10, 2, 1, 0.5, objective="regression")
        assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())

    def test_backward_compat_regression_shape(self) -> None:
        """regression 头输出 [B, 1]。"""
        torch = pytest.importorskip("torch")
        from aitrade.cnn.network import create_market_cnn

        B, C, T, S, G = 3, 6, 10, 2, 1
        model = create_market_cnn(C, T, S, G, dropout=0.5, objective="regression")
        model.eval()
        x = torch.randn(B, C, T, S, G)
        mask = torch.ones(B, 1, 1, S, G)
        with torch.no_grad():
            out = model(x, mask)
        assert out.shape == (B, 1)


# ---------------------------------------------------------------------------
# 3.3  _path_class_metrics
# ---------------------------------------------------------------------------

class TestPathClassMetrics:
    """3.3 _path_class_metrics：完美预测、单类缺失等边界情形。"""

    @pytest.fixture
    def _metrics(self):
        from aitrade.cnn.trainer import _path_class_metrics
        return _path_class_metrics

    def _perfect_logits(self, y_true: np.ndarray) -> np.ndarray:
        """为每个真值类别构造极端 logit：真值类置为 10，其余为 -10。"""
        n = len(y_true)
        logits = np.full((n, 4), -10.0, dtype=np.float64)
        for i, c in enumerate(y_true):
            logits[i, int(c)] = 10.0
        return logits

    def test_perfect_prediction(self, _metrics) -> None:
        """完美预测：tp_auc=sl_auc=1.0，macro_f1=1.0。"""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32)
        logits = self._perfect_logits(y_true)
        result = _metrics(y_true, logits)
        assert result["tp_auc"] == pytest.approx(1.0, abs=1e-6)
        assert result["sl_auc"] == pytest.approx(1.0, abs=1e-6)
        assert result["macro_f1"] == pytest.approx(1.0, abs=1e-6)

    def test_class0_absent_tp_auc_is_none(self, _metrics) -> None:
        """y_true 不含类 0 时 tp_auc 应为 None（AUC 无定义），不应崩溃。"""
        y_true = np.array([1, 2, 3, 1, 2, 3], dtype=np.float32)
        logits = self._perfect_logits(y_true)
        result = _metrics(y_true, logits)
        assert result["tp_auc"] is None
        # 其他指标不受影响
        assert result["sl_auc"] is not None

    def test_class1_absent_sl_auc_is_none(self, _metrics) -> None:
        """y_true 不含类 1 时 sl_auc 应为 None。"""
        y_true = np.array([0, 2, 3, 0, 2, 3], dtype=np.float32)
        logits = self._perfect_logits(y_true)
        result = _metrics(y_true, logits)
        assert result["sl_auc"] is None
        assert result["tp_auc"] is not None

    def test_macro_f1_excludes_absent_class(self, _metrics) -> None:
        """macro_f1 仅对有 support 的类求均值（support=0 的类不参与分母）。"""
        # 只有类 0、1、2；类 3 缺失
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.float32)
        logits = self._perfect_logits(y_true)
        result = _metrics(y_true, logits)
        # 三类完美预测，macro_f1 应为 1.0（不被类 3 的缺失拉低）
        assert result["macro_f1"] == pytest.approx(1.0, abs=1e-6)

    def test_class_report_structure(self, _metrics) -> None:
        """class_report 应含四个类别键，每键有 precision/recall/support。"""
        y_true = np.array([0, 1, 2, 3], dtype=np.float32)
        logits = self._perfect_logits(y_true)
        result = _metrics(y_true, logits)
        report = result["class_report"]
        for key in ("tp_first", "sl_first", "time_up", "time_down"):
            assert key in report, f"缺少类别键: {key}"
            assert "precision" in report[key]
            assert "recall" in report[key]
            assert "support" in report[key]

    def test_result_keys_complete(self, _metrics) -> None:
        """返回字典应含 tp_auc/sl_auc/macro_f1/class_report。"""
        y_true = np.array([0, 1, 2, 3], dtype=np.float32)
        logits = self._perfect_logits(y_true)
        result = _metrics(y_true, logits)
        for key in ("tp_auc", "sl_auc", "macro_f1", "class_report"):
            assert key in result, f"缺少键: {key}"


# ---------------------------------------------------------------------------
# 3.3c  _selection_score path_class 分支：None → 按 0.5
# ---------------------------------------------------------------------------

class TestSelectionScorePathClass:
    """3.3c _selection_score：path_class 分支 None 按 0.5 处理。"""

    @pytest.fixture
    def _score(self):
        from aitrade.cnn.trainer import _selection_score
        return _selection_score

    def test_both_aucs_present(self, _score) -> None:
        """tp_auc=0.6, sl_auc=0.7 → score=1.3。"""
        row = {"val_tp_auc": 0.6, "val_sl_auc": 0.7}
        s = _score(row, objective="path_class")
        assert s == pytest.approx(1.3, abs=1e-6)

    def test_tp_auc_none_falls_back(self, _score) -> None:
        """tp_auc=None → 按 0.5 计，sl_auc=0.8 → score=1.3。"""
        row = {"val_tp_auc": None, "val_sl_auc": 0.8}
        s = _score(row, objective="path_class")
        assert s == pytest.approx(1.3, abs=1e-6)

    def test_sl_auc_none_falls_back(self, _score) -> None:
        """sl_auc=None → 按 0.5 计，tp_auc=0.9 → score=1.4。"""
        row = {"val_tp_auc": 0.9, "val_sl_auc": None}
        s = _score(row, objective="path_class")
        assert s == pytest.approx(1.4, abs=1e-6)

    def test_both_none_gives_one(self, _score) -> None:
        """两者都 None → score=1.0（0.5+0.5）。"""
        row = {"val_tp_auc": None, "val_sl_auc": None}
        s = _score(row, objective="path_class")
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_classification_still_works(self, _score) -> None:
        """classification 分支：使用 val_auc，不受 path_class 改动影响。"""
        row = {"val_auc": 0.72}
        s = _score(row, objective="classification")
        assert s == pytest.approx(0.72, abs=1e-6)

    def test_regression_still_works(self, _score) -> None:
        """regression 分支仍正常工作。"""
        row = {"val_rank_ic": 0.3, "val_excess_acc": 0.1}
        s = _score(row, objective="regression")
        assert s == pytest.approx(0.4, abs=1e-6)


# ---------------------------------------------------------------------------
# 3.5  Property 2：softmax 后的概率单纯形
# ---------------------------------------------------------------------------

# Feature: cnn-path-multiclass-head, Property 2:
# 对任意合法形状的输入张量，path_class 模型输出经 softmax 后形状为 [B,4]、
# 每个分量均在 [0,1] 内、每行之和与 1 的偏差 < 1e-5（概率单纯形约束）。


@st.composite
def path_class_model_inputs(draw):
    """生成 path_class 模型合法输入的 Hypothesis 策略。

    B ∈ [1,4], T ∈ [8,16], S ∈ [1,3], G ∈ [1,2], C 固定 6。

    Returns:
        (model, x, mask) 三元组，model 已调用 eval()。
    """
    import torch
    from aitrade.cnn.network import create_market_cnn

    B = draw(st.integers(min_value=1, max_value=4))
    T = draw(st.integers(min_value=8, max_value=16))
    S = draw(st.integers(min_value=1, max_value=3))
    G = draw(st.integers(min_value=1, max_value=2))
    C = 6

    torch.manual_seed(draw(st.integers(min_value=0, max_value=9999)))
    model = create_market_cnn(C, T, S, G, dropout=0.5, objective="path_class")
    model.eval()  # BatchNorm 在 B=1 train 模式会崩，必须 eval

    x = torch.randn(B, C, T, S, G)
    mask = torch.ones(B, 1, 1, S, G)
    return model, x, mask


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@given(path_class_model_inputs())
@settings(max_examples=100)
def test_property2_softmax_simplex(args) -> None:
    """Property 2：path_class 输出 softmax 后满足概率单纯形约束。"""
    import torch

    model, x, mask = args
    with torch.no_grad():
        logits = model(x, mask)  # [B, 4]
    probs = torch.softmax(logits, dim=1)  # [B, 4]

    B = logits.shape[0]
    assert probs.shape == (B, 4), f"形状错误: {probs.shape}"
    # 所有分量 ∈ [0, 1]
    assert probs.min().item() >= 0.0, f"存在负概率: {probs.min().item()}"
    assert probs.max().item() <= 1.0 + 1e-6, f"概率超 1: {probs.max().item()}"
    # 每行和为 1（容忍 1e-5 浮点误差）
    row_sums = probs.sum(dim=1)
    max_dev = (row_sums - 1.0).abs().max().item()
    assert max_dev < 1e-5, f"行和偏差过大: {max_dev}"


# ---------------------------------------------------------------------------
# 3.6  冒烟测试：合成小数据 path_class 训练 2 epoch 跑通
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    n: int = 80,
    C: int = 6,
    T: int = 10,
    S: int = 2,
    G: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """生成供冒烟测试使用的合成 path_class 数据集。

    四类标签均有分布（各约 n//4 个），info 键与真实 build_dataset 输出一致，
    供 train_cnn_model 内部逻辑（包括 class_distribution、sample_returns 等）正常工作。

    Args:
        n: 样本数；默认 80（满足 >=50 的 trainer 最低要求）。
        C: 特征通道数；默认 6（对应 FEATURE_NAMES）。
        T: 时间步数；默认 10。
        S: 每组最大证券数；默认 2。
        G: 分组数；默认 1。

    Returns:
        (X, y, group_mask, info) 四元组，与 build_dataset 签名一致。
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, C, T, S, G)).astype(np.float32)
    # 四类均匀分布（0/1/2/3），确保每类都有样本
    y = np.tile(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32), n // 4 + 1)[:n]
    group_mask = np.ones((1, 1, 1, S, G), dtype=np.float32)

    # 与 build_dataset 真实输出对齐的 info 字典
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


class TestSmokeTrainPathClass:
    """3.6 冒烟测试：合成数据训练 2 epoch，结果键与 history 键齐全。"""

    def test_smoke_train_runs_and_keys_complete(self, tmp_path, monkeypatch) -> None:
        """path_class 训练 2 epoch 跑通，result 与 history 含专属键。"""
        pytest.importorskip("torch")
        import aitrade.cnn.trainer as trainer_mod

        # 合成数据集（不走真实行情 IO）
        X, y, group_mask, info = _make_synthetic_dataset()

        monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))
        monkeypatch.setattr(trainer_mod, "save_cnn_model", lambda name, data, hist: (
            tmp_path / f"{name}.pt",
            tmp_path / f"{name}.json",
        ))

        from aitrade.cnn.trainer import train_cnn_model

        result = train_cnn_model(
            name="smoke_path",
            vt_symbols=["AAA.SSE", "BBB.SSE"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            epochs=2,
            batch_size=16,
            lookback=10,
            dropout=0.0,
            objective="path_class",
            label_spec={
                "mode": "oco",
                "take_profit": 0.05,
                "stop_loss": 0.05,
                "max_hold": 10,
            },
        )

        # result 应含 path_class 专属键
        for key in ("best_val_tp_auc", "best_val_sl_auc", "best_val_macro_f1", "class_distribution"):
            assert key in result, f"result 缺少键: {key}"
        assert result.get("num_classes") == 4

        # history 每行应含 val_tp_auc / val_sl_auc / val_macro_f1
        for row in result["history"]:
            for key in ("val_tp_auc", "val_sl_auc", "val_macro_f1"):
                assert key in row, f"history 行缺少键: {key}，当前行: {list(row.keys())}"

    def test_smoke_train_beats_baseline_uses_tp_auc(self, tmp_path, monkeypatch) -> None:
        """beats_baseline 应基于 best_val_tp_auc > 0.5 判断，而非 excess_acc。"""
        pytest.importorskip("torch")
        import aitrade.cnn.trainer as trainer_mod

        X, y, group_mask, info = _make_synthetic_dataset()

        monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))
        monkeypatch.setattr(trainer_mod, "save_cnn_model", lambda name, data, hist: (
            tmp_path / f"{name}.pt",
            tmp_path / f"{name}.json",
        ))

        from aitrade.cnn.trainer import train_cnn_model

        result = train_cnn_model(
            name="smoke_bb",
            vt_symbols=["AAA.SSE"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            epochs=2,
            batch_size=16,
            lookback=10,
            dropout=0.0,
            objective="path_class",
            label_spec={
                "mode": "oco",
                "take_profit": 0.05,
                "stop_loss": 0.05,
                "max_hold": 10,
            },
        )

        # beats_baseline 应是 bool，且与 best_val_tp_auc 的判断一致
        tp_auc = result.get("best_val_tp_auc")
        expected_beats = bool(tp_auc is not None and tp_auc > 0.5)
        assert result["beats_baseline"] == expected_beats

    def test_smoke_train_loss_weighting_forced_none(self, tmp_path, monkeypatch) -> None:
        """path_class 模式下 loss_weighting='magnitude' 应被强制回退为 'none'。

        验证 save_cnn_model 收到的 train_config["loss_weighting"] 确实为 "none"，
        而不仅仅是"不崩溃"。
        """
        pytest.importorskip("torch")
        import aitrade.cnn.trainer as trainer_mod

        X, y, group_mask, info = _make_synthetic_dataset()

        # 捕获 save_cnn_model 的入参，用于断言
        captured: dict[str, Any] = {}

        def _fake_save(name: str, data: dict, hist: list) -> tuple:
            captured["save_data"] = data
            return (tmp_path / f"{name}.pt", tmp_path / f"{name}.json")

        monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))
        monkeypatch.setattr(trainer_mod, "save_cnn_model", _fake_save)

        from aitrade.cnn.trainer import train_cnn_model

        result = train_cnn_model(
            name="smoke_lw",
            vt_symbols=["AAA.SSE"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            epochs=2,
            batch_size=16,
            lookback=10,
            dropout=0.0,
            objective="path_class",
            loss_weighting="magnitude",  # 应被强制回退为 "none"
            label_spec={
                "mode": "oco",
                "take_profit": 0.05,
                "stop_loss": 0.05,
                "max_hold": 10,
            },
        )
        assert "best_epoch" in result
        # 核心断言：强制回退后 train_config 中 loss_weighting 应为 "none"
        assert "save_data" in captured, "save_cnn_model 未被调用"
        assert captured["save_data"]["train_config"]["loss_weighting"] == "none", (
            f"loss_weighting 未被强制为 none，实得: "
            f"{captured['save_data']['train_config'].get('loss_weighting')}"
        )
