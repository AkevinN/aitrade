"""
trainer seed 参数化验收测试。

覆盖范围：
  1.2  示例测试：同 seed 两次训练 state_dict 数值等价；
             不同 seed 至少一层权重差异 > 容差
  1.3a Property 1: 可复现且可分辨——同 seed 权重完全相同，异 seed 可观测不同
  1.3b Property 2: seed checkpoint 往返——train_config["seed"] 等于训练所用 seed
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# 合成数据集工厂（与 test_cnn_path_trainer.py 中保持一致，避免 IO）
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n: int = 80,
    C: int = 6,
    T: int = 10,
    S: int = 2,
    G: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """生成供 seed 测试使用的合成分类数据集。

    使用固定 rng(seed=0)，确保合成 X/y 本身不依赖 trainer 种子，
    从而隔离 trainer seed 对模型权重初始化和 DataLoader shuffle 的影响。

    Args:
        n: 样本数；默认 80（满足 trainer >=50 最低要求）。
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


def _train_with_seed(seed: int, tmp_path: Any, monkeypatch: Any) -> dict[str, Any]:
    """用合成数据训练 2 epoch，返回 save_cnn_model 捕获到的 save_data。

    Args:
        seed: 传给 train_cnn_model 的 seed 参数。
        tmp_path: pytest 临时目录，供假 save 路径使用。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        save_cnn_model 收到的 data 字典（含 model_state_dict / train_config 等）。
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

    train_cnn_model(
        name="seed_test",
        vt_symbols=["AAA.SSE", "BBB.SSE"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        epochs=2,
        batch_size=16,
        lookback=10,
        dropout=0.0,
        seed=seed,
    )
    return captured["data"]


# ---------------------------------------------------------------------------
# 1.2  示例测试：同 seed / 不同 seed 的权重比较
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    """1.2 示例测试：seed 控制可复现性与可分辨性。"""

    def test_same_seed_produces_identical_state_dict(self, tmp_path, monkeypatch) -> None:
        """同 seed 两次训练得到数值完全等价的 state_dict。"""
        pytest.importorskip("torch")
        import torch

        data_a = _train_with_seed(7, tmp_path, monkeypatch)
        data_b = _train_with_seed(7, tmp_path, monkeypatch)

        sd_a = data_a["model_state_dict"]
        sd_b = data_b["model_state_dict"]

        assert set(sd_a.keys()) == set(sd_b.keys()), "两次训练的 state_dict 键不一致"
        for key in sd_a:
            assert torch.allclose(sd_a[key], sd_b[key], atol=0.0), (
                f"同 seed=7 两次训练，参数 {key} 不等价"
            )

    def test_different_seeds_produce_different_weights(self, tmp_path, monkeypatch) -> None:
        """不同 seed 至少一层参数差异 > 1e-6。"""
        pytest.importorskip("torch")

        data_7 = _train_with_seed(7, tmp_path, monkeypatch)
        data_9 = _train_with_seed(9, tmp_path, monkeypatch)

        sd_7 = data_7["model_state_dict"]
        sd_9 = data_9["model_state_dict"]

        found_diff = False
        for key in sd_7:
            if key in sd_9 and sd_7[key].dtype.is_floating_point:
                max_diff = (sd_7[key] - sd_9[key]).abs().max().item()
                if max_diff > 1e-6:
                    found_diff = True
                    break

        assert found_diff, "seed=7 与 seed=9 的所有参数差异均 <= 1e-6，种子似乎未生效"

    def test_seed_written_to_train_config(self, tmp_path, monkeypatch) -> None:
        """train_config['seed'] 应等于传入的 seed 值。"""
        pytest.importorskip("torch")

        data = _train_with_seed(42, tmp_path, monkeypatch)
        assert "seed" in data["train_config"], "train_config 缺少 'seed' 键"
        assert data["train_config"]["seed"] == 42

    def test_default_seed_is_42(self, tmp_path, monkeypatch) -> None:
        """不传 seed 时默认值为 42，且写入 train_config。"""
        pytest.importorskip("torch")
        import aitrade.cnn.trainer as trainer_mod

        X, y, group_mask, info = _make_synthetic_dataset()
        captured: dict[str, Any] = {}

        def _fake_save(name: str, data: dict, hist: list) -> tuple:
            captured["data"] = data
            return (tmp_path / f"{name}.pt", tmp_path / f"{name}.json")

        monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))
        monkeypatch.setattr(trainer_mod, "save_cnn_model", _fake_save)

        from aitrade.cnn.trainer import train_cnn_model

        train_cnn_model(
            name="default_seed",
            vt_symbols=["AAA.SSE", "BBB.SSE"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 30),
            epochs=2,
            batch_size=16,
            lookback=10,
            dropout=0.0,
            # 不传 seed，期望默认值 42
        )
        assert captured["data"]["train_config"].get("seed") == 42, (
            "默认 seed 应为 42，但 train_config['seed'] 值不符"
        )


# ---------------------------------------------------------------------------
# 1.3a  Property 1：可复现且可分辨（Hypothesis）
# Feature: cnn-eval-honesty-fixes, Property 1:
# 对任意合法 seed，同 seed 两次训练得到数值等价的权重；
# 不同 seed 得到可观测不同的权重（至少一层参数差异 > 容差）。
# ---------------------------------------------------------------------------

_TRAIN_SEEDS = st.integers(min_value=0, max_value=2**31 - 1)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@given(seed_a=_TRAIN_SEEDS, seed_b=_TRAIN_SEEDS.filter(lambda x: x != 0))
@settings(
    max_examples=10,  # 每例需训练 2-3 次，整体控制在 <60s
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property1_reproducible_and_distinguishable(
    seed_a: int, seed_b: int, tmp_path: Any, monkeypatch: Any
) -> None:
    """Property 1: 同 seed 两次权重等价；seed_a vs seed_a^seed_b 权重可分辨（至少一层差异 > 1e-6）。

    使用 seed_a 作为"固定种子"，seed_a XOR seed_b 作为"另一种子"（异或确保两者不等）。
    两次 seed_a 训练结果精确相同；再用 seed_a^seed_b 训练，至少一层浮点参数差异 > 1e-6。
    """
    import torch

    seed_other = seed_a ^ seed_b  # 保证 seed_other != seed_a（seed_b != 0）

    data_a1 = _train_with_seed(seed_a, tmp_path, monkeypatch)
    data_a2 = _train_with_seed(seed_a, tmp_path, monkeypatch)
    data_other = _train_with_seed(seed_other, tmp_path, monkeypatch)

    sd_a1 = data_a1["model_state_dict"]
    sd_a2 = data_a2["model_state_dict"]
    sd_other = data_other["model_state_dict"]

    # --- 可复现性：同 seed 精确相同 ---
    for key in sd_a1:
        assert torch.allclose(sd_a1[key], sd_a2[key], atol=0.0), (
            f"Property 1 失败（可复现性）：seed={seed_a}，参数 {key} 两次结果不等"
        )

    # --- 可分辨性：异 seed 至少一层浮点参数明显不同 ---
    found_diff = any(
        sd_a1[k].dtype.is_floating_point
        and (sd_a1[k] - sd_other[k]).abs().max().item() > 1e-6
        for k in sd_a1
        if k in sd_other
    )
    assert found_diff, (
        f"Property 1 失败（可分辨性）：seed={seed_a} vs seed={seed_other}，"
        "所有浮点参数差异均 <= 1e-6"
    )


# ---------------------------------------------------------------------------
# 1.3b  Property 2: seed checkpoint 往返（Hypothesis）
# Feature: cnn-eval-honesty-fixes, Property 2:
# 对任意训练完成的模型，train_config["seed"] 等于训练所用 seed，且重载后可读。
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="需要 torch",
)
@given(seed=_TRAIN_SEEDS)
@settings(
    max_examples=15,  # 每例仅训练一次，较轻
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property2_seed_checkpoint_roundtrip(
    seed: int, tmp_path: Any, monkeypatch: Any
) -> None:
    """Property 2: train_config["seed"] 在 torch.save→torch.load 磁盘往返后与原始值相等。

    验证流程：
    1. 将 CNN_MODEL_DIR 重定向到 tmp_path，让真实 save_cnn_model 落盘到隔离目录。
    2. 用合成数据调用 train_cnn_model，触发真实 torch.save。
    3. 从磁盘用 torch.load 重新加载 checkpoint（而非读同一内存对象）。
    4. 断言加载后的 train_config["seed"] == 原始 seed。

    Args:
        seed: Hypothesis 生成的任意合法 seed。
        tmp_path: pytest 临时目录，隔离落盘文件，测试结束后自动清理。
        monkeypatch: pytest monkeypatch fixture。
    """
    import torch

    import aitrade.cnn.storage as cnn_storage
    import aitrade.cnn.trainer as trainer_mod

    X, y, group_mask, info = _make_synthetic_dataset()

    # 用 seed 值构造唯一模型名，避免多 example 之间串扰
    model_name = f"seed_rt_{seed}"

    # patch build_dataset（不走真实行情 IO）
    monkeypatch.setattr(trainer_mod, "build_dataset", lambda **_kw: (X, y, group_mask, info))

    # patch CNN_MODEL_DIR → tmp_path；save_cnn_model 读 storage 模块全局变量，
    # 仅替换此变量即可让真实 torch.save 落盘到隔离的临时目录。
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)

    from aitrade.cnn.trainer import train_cnn_model

    result = train_cnn_model(
        name=model_name,
        vt_symbols=["AAA.SSE", "BBB.SSE"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        epochs=2,
        batch_size=16,
        lookback=10,
        dropout=0.0,
        seed=seed,
    )

    # train_cnn_model 会在 name 后附加日期后缀，从返回值取实际名称
    actual_name: str = result["name"]
    pt_path = tmp_path / f"{actual_name}.pt"
    assert pt_path.exists(), f"模型文件未落盘：{pt_path}"

    # 真实磁盘往返：通过 torch.load 重新反序列化（而非读同一内存字典）
    loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    assert "train_config" in loaded, "checkpoint 缺少 'train_config' 键"
    assert "seed" in loaded["train_config"], "train_config 缺少 'seed' 键"
    reloaded_seed = loaded["train_config"]["seed"]
    assert reloaded_seed == seed, (
        f"Property 2 失败（磁盘往返）：训练 seed={seed}，"
        f"torch.load 后 train_config['seed']={reloaded_seed}"
    )
