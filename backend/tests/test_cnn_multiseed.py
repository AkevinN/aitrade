"""
CNN 治理多种子真实循环验收测试（cnn-eval-honesty-fixes 任务 2）。

覆盖范围：
  2.4 示例测试：n_seeds=3 的小 WF（monkeypatch 训练/回测合成）——
             折内训出的 3 个模型互不相同（异种子异权重）；
             门禁候选分数 == cross_seed.mean。
  2.5 Property 3：跨种子离散度真实——
             N>=2 且分数不全等 -> std>0；N==1 -> std==0；
             门禁消费的候选分数恒等于 cross_seed.mean。

设计说明：治理流程含真实训练 + 回测，开销极重，故除「异种子异权重」一条
走真实 train_cnn_model（合成 build_dataset）外，其余均 mock 治理依赖
（_train_governance_model / _backtest_model / store），把验证聚焦在
多种子循环、聚合与门禁消费的逻辑正确性上。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import aitrade.cnn.governance as gov
from aitrade.cnn.governance import _cross_seed_dispersion
from aitrade.models.governance import (
    CNNBacktestParams,
    CNNPromotionGate,
    CNNTrainingParams,
    CNNWalkForwardRequest,
)


# ---------------------------------------------------------------------------
# _cross_seed_dispersion 纯函数单测（mean/std/n 与边界）
# ---------------------------------------------------------------------------


class TestCrossSeedDispersion:
    """_cross_seed_dispersion 的均值/标准差/样本数与边界行为。"""

    def test_multi_seed_mean_and_std(self) -> None:
        """多种子：mean 为算术均值，std 为总体标准差（ddof=0），n=种子数。"""
        disp = _cross_seed_dispersion([0.2, 0.4, 0.6])
        assert disp["n"] == 3
        assert disp["mean"] == pytest.approx(0.4)
        assert disp["std"] == pytest.approx(float(np.std([0.2, 0.4, 0.6])))
        assert disp["std"] > 0.0

    def test_single_seed_std_zero(self) -> None:
        """单种子（n=1）：std 恒为 0，mean 等于该唯一得分。"""
        disp = _cross_seed_dispersion([0.5])
        assert disp == {"mean": 0.5, "std": 0.0, "n": 1}

    def test_empty_scores_neutral(self) -> None:
        """空列表边界：mean/std=0，n=0，不抛异常。"""
        assert _cross_seed_dispersion([]) == {"mean": 0.0, "std": 0.0, "n": 0}

    def test_identical_scores_std_zero(self) -> None:
        """多种子但分数全等：std=0（无离散度）。"""
        disp = _cross_seed_dispersion([0.3, 0.3, 0.3])
        assert disp["std"] == pytest.approx(0.0)
        assert disp["mean"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# WF 折内多种子循环：mock 训练 + 回测，验证聚合与门禁消费
# ---------------------------------------------------------------------------


def _patch_single_fold(monkeypatch: Any, tmp_path: Any) -> CNNWalkForwardRequest:
    """构造单折 WF 并 mock 训练/回测/store，返回对应请求。

    将日期范围与 train/test/step days 设为恰好生成 1 个窗口，便于专注单折断言。
    _train_governance_model 被替换为只回名字的轻量假实现，按 seed_index 编号；
    _backtest_model 由各测试自行 monkeypatch 以注入可控 statistics；
    store 替换为 tmp_path 上的真实 store，避免污染全局持久化。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录，供独立 store 落盘。

    Returns:
        生成 1 个窗口的 CNNWalkForwardRequest（n_seeds 由调用方覆盖）。
    """
    monkeypatch.setattr(gov, "store", gov.CNNGovernanceStore(tmp_path))

    def _fake_train(req: CNNWalkForwardRequest, *, model_name: str, start, end, seed_index: int = 0, on_progress=None):
        # 不真训，仅回传一个带 seed_index 的可识别模型名。
        return {"name": f"{model_name}", "seed_index": seed_index}

    monkeypatch.setattr(gov, "_train_governance_model", _fake_train)

    # train 540 天 + test 30 天，区间恰够 1 个窗口。
    start = date(2022, 1, 1)
    end = start + timedelta(days=540 + 30)
    return CNNWalkForwardRequest(
        name="ms_wf",
        target_symbol="AAA.SSE",
        start=start,
        end=end,
        train_days=540,
        test_days=30,
        step_days=30,
        promotion_gate=CNNPromotionGate(),
    )


class TestWalkForwardMultiSeedFold:
    """折内多种子循环：候选分数取均值、cross_seed 真实、命名带 _s{idx}。"""

    def test_candidate_score_equals_cross_seed_mean(self, monkeypatch, tmp_path) -> None:
        """n_seeds=3、三个种子得分 [0.2,0.4,0.6] -> candidate_score==0.4 且 std>0。"""
        req = _patch_single_fold(monkeypatch, tmp_path)
        req = req.model_copy(update={"n_seeds": 3})

        # 按种子模型名末尾的 _s{idx} 注入不同分数：_s0->0.2 _s1->0.4 _s2->0.6。
        score_by_seed = {0: 0.2, 1: 0.4, 2: 0.6}

        def _fake_backtest(*, model_name: str, name: str, start, end, capital, params):
            seed_index = int(model_name.split("_s")[1].split("_")[0])
            # 让 _core_score 直接拿到我们想要的核心得分：用 total_return 注入，
            # 并给 1 笔交易避免空仓罚分；其余项为 0。
            return {"statistics": {"total_return": score_by_seed[seed_index], "total_trade_count": 1}}

        monkeypatch.setattr(gov, "_backtest_model", _fake_backtest)

        report = gov.run_walk_forward_evaluate(req)
        folds = report["folds"]
        assert len(folds) == 1
        fold = folds[0]

        assert fold["cross_seed"]["n"] == 3
        assert fold["candidate_seed_scores"] == pytest.approx([0.2, 0.4, 0.6])
        assert fold["cross_seed"]["mean"] == pytest.approx(0.4)
        # Property 3：N>=2 且分数不全等 -> std>0。
        assert fold["cross_seed"]["std"] > 0.0
        # 门禁消费的候选分数 == 跨种子均值。
        assert fold["candidate_score"] == pytest.approx(0.4)
        assert fold["candidate_score"] == pytest.approx(fold["cross_seed"]["mean"])

    def test_seed_models_named_distinctly(self, monkeypatch, tmp_path) -> None:
        """n_seeds=3 折内 3 个候选模型命名带 _s0/_s1/_s2，互不覆盖。"""
        req = _patch_single_fold(monkeypatch, tmp_path)
        req = req.model_copy(update={"n_seeds": 3})

        monkeypatch.setattr(
            gov,
            "_backtest_model",
            lambda **kw: {"statistics": {"total_return": 0.1, "total_trade_count": 1}},
        )

        report = gov.run_walk_forward_evaluate(req)
        models = report["folds"][0]["candidate_models"]
        assert len(models) == 3
        assert len(set(models)) == 3, "三个种子模型名不应重复（应带 _s{idx} 区分）"
        for idx, model in enumerate(models):
            assert f"_s{idx}_" in model, f"第 {idx} 个种子模型名缺少 _s{idx} 标记: {model}"

    def test_summary_avg_cross_seed_std(self, monkeypatch, tmp_path) -> None:
        """summary 含 avg_cross_seed_std 与 n_seeds；单折时等于该折 std。"""
        req = _patch_single_fold(monkeypatch, tmp_path)
        req = req.model_copy(update={"n_seeds": 3})

        score_by_seed = {0: 0.0, 1: 0.3, 2: 0.6}
        monkeypatch.setattr(
            gov,
            "_backtest_model",
            lambda **kw: {
                "statistics": {
                    "total_return": score_by_seed[int(kw["model_name"].split("_s")[1].split("_")[0])],
                    "total_trade_count": 1,
                }
            },
        )

        report = gov.run_walk_forward_evaluate(req)
        summary = report["summary"]
        assert summary["n_seeds"] == 3
        assert summary["avg_cross_seed_std"] == pytest.approx(report["folds"][0]["cross_seed"]["std"])

    def test_single_seed_std_zero_in_fold(self, monkeypatch, tmp_path) -> None:
        """n_seeds=1（默认）：折内 cross_seed.std==0，candidate_score==该单一得分。"""
        req = _patch_single_fold(monkeypatch, tmp_path)  # 默认 n_seeds=1
        assert req.n_seeds == 1

        monkeypatch.setattr(
            gov,
            "_backtest_model",
            lambda **kw: {"statistics": {"total_return": 0.42, "total_trade_count": 1}},
        )

        report = gov.run_walk_forward_evaluate(req)
        fold = report["folds"][0]
        assert fold["cross_seed"]["n"] == 1
        assert fold["cross_seed"]["std"] == 0.0
        assert fold["candidate_score"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Property 3：异种子异权重（走真实 train_cnn_model + 合成数据）
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n: int = 80, C: int = 6, T: int = 10, S: int = 2, G: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """生成与 trainer 种子无关的合成分类数据集（复用任务 1 的构造约定）。

    使用固定 rng(seed=0)，确保 X/y 不依赖 trainer 种子，从而隔离出 seed
    对权重初始化与 DataLoader shuffle 的影响。

    Args:
        n: 样本数；默认 80（>=trainer 最低 50）。
        C/T/S/G: 通道/时间步/每组宽/分组数。

    Returns:
        (X, y, group_mask, info) 四元组，info 键与真实 build_dataset 对齐。
    """
    rng = np.random.default_rng(0)
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
        "label_spec": {"mode": "next_bar", "threshold": 0.0, "neutral_policy": "drop", "price_ref": "close"},
        "label_threshold": 0.0,
        "price_ref": "close",
        "objective": "classification",
        "skipped_for_label": 0,
        "skipped_for_neutral": 0,
        "sample_returns": rng.uniform(-0.05, 0.05, n).tolist(),
    }
    return X, y, group_mask, info


def _govern_train_capture(seed_index: int, tmp_path: Any, monkeypatch: Any) -> dict[str, Any]:
    """经 governance._train_governance_model 用合成数据训练，捕获落盘 state_dict。

    验证 seed_index 真正下传 train_cnn_model（seed=BASE_SEED+seed_index），
    复用任务 1 已证明的「异 seed 异权重」，把链路落到治理入口上。

    Args:
        seed_index: 治理层的重复试验序号（映射为 seed=BASE_SEED+seed_index）。
        tmp_path: pytest 临时目录，供假 save 路径。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        save_cnn_model 捕获到的 data 字典（含 model_state_dict / train_config）。
    """
    import aitrade.cnn.trainer as trainer_mod

    X, y, group_mask, info = _make_synthetic_dataset()
    captured: dict[str, Any] = {}

    def _fake_save(name: str, data: dict, hist: list) -> tuple:
        captured["data"] = data
        return (tmp_path / f"{name}.pt", tmp_path / f"{name}.json")

    monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))
    monkeypatch.setattr(trainer_mod, "save_cnn_model", _fake_save)

    req = CNNWalkForwardRequest(
        name="ms_seed",
        target_symbol="AAA.SSE",
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        training_params=CNNTrainingParams(epochs=2, batch_size=16, lookback=10, dropout=0.0),
        backtest_params=CNNBacktestParams(),
    )
    gov._train_governance_model(
        req, model_name="ms_seed_model", start=date(2024, 1, 1), end=date(2024, 6, 30), seed_index=seed_index
    )
    return captured["data"]


class TestMultiSeedDistinctModels:
    """Property 3 之「异种子异权重」：治理入口下传 seed_index 训出不同模型。"""

    def test_three_seed_indices_yield_distinct_weights(self, tmp_path, monkeypatch) -> None:
        # Feature: cnn-eval-honesty-fixes, Property 3: 训练出的 N 个模型互不相同（权重不全等）
        data0 = _govern_train_capture(0, tmp_path, monkeypatch)
        data1 = _govern_train_capture(1, tmp_path, monkeypatch)
        data2 = _govern_train_capture(2, tmp_path, monkeypatch)

        # seed_index 已下传：train_config.seed == BASE_SEED + seed_index。
        assert data0["train_config"]["seed"] == gov.BASE_SEED + 0
        assert data1["train_config"]["seed"] == gov.BASE_SEED + 1
        assert data2["train_config"]["seed"] == gov.BASE_SEED + 2

        sds = [data0["model_state_dict"], data1["model_state_dict"], data2["model_state_dict"]]

        def _any_param_differs(sd_a: dict, sd_b: dict) -> bool:
            for key in sd_a:
                if not np.allclose(sd_a[key].cpu().numpy(), sd_b[key].cpu().numpy(), atol=1e-6):
                    return True
            return False

        assert _any_param_differs(sds[0], sds[1]), "seed_index 0 与 1 权重全等，种子未生效"
        assert _any_param_differs(sds[0], sds[2]), "seed_index 0 与 2 权重全等，种子未生效"
        assert _any_param_differs(sds[1], sds[2]), "seed_index 1 与 2 权重全等，种子未生效"


# ---------------------------------------------------------------------------
# Property 3 属性测试：离散度的真实性（N>=2 不全等->std>0；N=1->std==0）
# ---------------------------------------------------------------------------


class TestProperty3Dispersion:
    """Property 3：跨种子离散度真实，门禁消费均值。"""

    @settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        scores=st.lists(
            # 限定到 6 位小数粒度：真实 _core_score 即 round(..., 6)，
            # 不会产出亚正规级差异（否则 std 在浮点下会下溢成 0）。
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False).map(
                lambda v: round(v, 6)
            ),
            min_size=2,
            max_size=10,
        )
    )
    def test_dispersion_positive_when_scores_differ(self, scores: list[float]) -> None:
        # Feature: cnn-eval-honesty-fixes, Property 3: N>=2 且各折分数不全相等时 cross_seed.std>0
        disp = _cross_seed_dispersion(scores)
        assert disp["n"] == len(scores)
        # mean 始终等于算术均值（门禁消费的就是它）。
        assert disp["mean"] == pytest.approx(float(np.mean(scores)), abs=1e-9)
        if len(set(scores)) > 1:
            assert disp["std"] > 0.0
        else:
            assert disp["std"] == pytest.approx(0.0)

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(score=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    def test_single_seed_always_zero_std(self, score: float) -> None:
        # Feature: cnn-eval-honesty-fixes, Property 3: N=1 时 std==0
        disp = _cross_seed_dispersion([score])
        assert disp["n"] == 1
        assert disp["std"] == 0.0
        assert disp["mean"] == pytest.approx(score, abs=1e-9)
