"""
端到端冒烟测试：path_class 全链路（dataset→训练→保存→推理→veto 回测→统计）。

这是 .kiro/specs/cnn-path-multiclass-head/design.md 测试策略中的"集成/冒烟 1 例"：
与各任务的单元测试不同，本文件走**真实 build_dataset**（真实特征计算与 OCO 标签生成），
仅在底层行情加载处（_load_market_frame）注入合成数据，确保以下链路全程无 mock 跳过：

    合成行情帧
         ↓  _load_market_frame（monkeypatched）
    build_dataset（真实标签生成 + 特征工程）
         ↓
    train_cnn_model（真实训练 2 epoch + 真实 save_cnn_model 落盘）
         ↓
    predict_cnn_signals（七列信号帧）
         ↓
    CNNSignalStrategy + BacktestingEngine（veto 回测）
         ↓
    engine 统计可计算 + _veto_count 存在 + class_distribution 四键齐全

覆盖属性（来自 design.md）：
  - Property 3（逐行 signal==prob_tp）
  - Property 4（veto_threshold 否决 + _veto_count 统计）
  - Property 5（全链路守护：旧 objective 三列无副作用）
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# 合成行情帧构造
# ---------------------------------------------------------------------------

_TARGET = "000001.SZSE"
_LOOKBACK = 10
_FRAME_ROWS = 120  # 约 120 根日线，给四类路径足够出现概率


def _make_synthetic_frame(rows: int = _FRAME_ROWS, seed: int = 42) -> pl.DataFrame:
    """构造连续日线合成行情帧，保证四类 OCO 路径标签均有机会出现。

    价格序列加入适度波动（sigma=0.025），使止盈/止损线（take_profit=stop_loss=0.03）
    在 max_hold=5 内均有机会被触及，从而令四类路径（tp_first/sl_first/time_up/time_down）
    都有样本。列名与 features.py 特征计算要求的 OHLCV 保持一致。

    Args:
        rows: 行数；默认 120（给 lookback=10 留足够裕量）。
        seed: numpy 随机种子，保证可复现。

    Returns:
        polars DataFrame，列：datetime, open, high, low, close, volume, turnover, open_interest。
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    dt = datetime(2023, 6, 1)
    px = 100.0
    for _ in range(rows):
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
        # sigma=0.025，偶发±3% 止盈止损触发；max_hold=5 后时间止损
        ret = rng.normal(0, 0.025)
        px = px * (1.0 + ret)
        px = max(px, 1.0)  # 防止价格退化为零或负
        records.append({
            "datetime": dt,
            "open": float(px * 0.998),
            "high": float(px * (1.0 + abs(rng.normal(0, 0.012)))),
            "low": float(px * (1.0 - abs(rng.normal(0, 0.012)))),
            "close": float(px),
            "volume": float(1_000_000 + rng.integers(0, 100_000)),
            "turnover": float(px * (1_000_000 + rng.integers(0, 100_000))),
            "open_interest": 0.0,
        })
        dt += timedelta(days=1)
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# 回测辅助（复用 test_cnn_path_strategy.py 的 FakeLoader/harness 模式）
# ---------------------------------------------------------------------------

from aitrade.backtest.engine import BacktestingEngine  # noqa: E402
from aitrade.backtest.types import BarData  # noqa: E402
from aitrade.cnn.strategy import CNNSignalStrategy  # noqa: E402


class _FakeLoader:
    """实现 BarDataLoader 协议的合成数据源，无外部依赖。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list[BarData]:
        """返回全量合成 bar 列表，忽略日期过滤（测试足够短）。"""
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        """返回最小化合约参数，手续费/滑点均为零以简化断言。"""
        return {
            _TARGET: {
                "long_rate": 0.0003,
                "short_rate": 0.0003,
                "stamp_duty": 0.0,
                "slippage": 0.0,
                "size": 1,
                "pricetick": 0.01,
            }
        }


def _signal_df_to_bars(signal_df: pl.DataFrame) -> list[BarData]:
    """将推理信号帧对应的日期转换为合成 BarData 列表，供回测引擎消费。

    使用信号帧中的 datetime 列，以固定价格（close=100）构造最简 BarData，
    保证限价单撮合时 low < close <= high 成立。

    Args:
        signal_df: predict_cnn_signals 返回的七列信号 DataFrame；datetime 列用于对齐日期。

    Returns:
        与 signal_df 行数相同的 BarData 列表，按 datetime 升序排列。
    """
    bars = []
    for row in signal_df.to_dicts():
        dt = row["datetime"]
        bars.append(
            BarData(
                symbol=_TARGET.split(".")[0],
                exchange=_TARGET.split(".")[1],
                datetime=dt,
                interval="d",
                open_price=98.0,
                high_price=105.0,
                low_price=95.0,
                close_price=100.0,
                volume=1_000_000,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# 主测试函数
# ---------------------------------------------------------------------------


def test_path_class_e2e_full_pipeline(tmp_path, monkeypatch) -> None:
    """path_class 全链路端到端冒烟测试：dataset→训练→推理→veto 回测→统计。

    本测试不 mock build_dataset，走真实特征计算与 OCO 路径标签生成；
    仅在行情加载层（_load_market_frame）注入合成数据帧以避免本地文件依赖。

    设计意图（来自 design.md 测试策略）：
    - 验证全链路无缺失接线（训练 result 键完整、信号帧七列、回测可运行）；
    - 覆盖 Property 3（signal==prob_tp）、Property 4（veto 否决计数）；
    - 守护 Task 1~7 各模块协同正确。

    Args:
        tmp_path: pytest 提供的临时目录（conftest 已全局隔离 AITRADE_HOME，
            但 CNN_MODEL_DIR 需额外 patch 到 tmp_path 确保真实落盘）。
        monkeypatch: pytest monkeypatch fixture。
    """
    pytest.importorskip("torch")

    import aitrade.cnn.dataset as dataset_mod
    from aitrade.cnn import model as cnn_model
    from aitrade.cnn import storage as cnn_storage

    # ------------------------------------------------------------------
    # 阶段 0：准备合成行情帧 + patch 行情加载
    # ------------------------------------------------------------------
    frame = _make_synthetic_frame()

    def _fake_loader(vt_symbol: str, _start, _end, *, input_data_kind: str, input_interval: str):
        """拦截行情加载，返回合成帧（忽略日期参数）。"""
        return frame

    # patch dataset 命名空间（build_dataset 内部调用路径）
    monkeypatch.setattr(dataset_mod, "_load_market_frame", _fake_loader)
    # patch model 命名空间（predictor 内部调用路径）
    monkeypatch.setattr(cnn_model, "_load_market_frame", _fake_loader)
    # 重定向模型落盘目录（conftest 隔离 AITRADE_HOME，但 CNN_MODEL_DIR
    # 在 storage 模块顶层已固化；此处再 patch 以确保真实 save_cnn_model 写到 tmp_path）
    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)

    # ------------------------------------------------------------------
    # 阶段 1：训练（真实 build_dataset + 真实 save_cnn_model 落盘）
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    from aitrade.cnn.trainer import train_cnn_model

    result = train_cnn_model(
        name="e2e_smoke",
        vt_symbols=[_TARGET],
        start=date(2023, 6, 1),
        end=date(2023, 12, 31),
        epochs=2,
        batch_size=16,
        lookback=_LOOKBACK,
        dropout=0.0,
        objective="path_class",
        label_spec={
            "mode": "oco",
            "take_profit": 0.03,
            "stop_loss": 0.03,
            "max_hold": 5,
        },
    )

    elapsed_train = time.perf_counter() - t0

    # --- 断言训练结果 ---
    model_name: str = result["name"]
    assert (tmp_path / f"{model_name}.pt").exists(), "模型文件应真实落盘"

    # path_class 专属键
    assert "class_distribution" in result, "result 缺少 class_distribution"
    cd = result["class_distribution"]
    assert set(cd.keys()) == {"tp_first", "sl_first", "time_up", "time_down"}, (
        f"class_distribution 键不完整: {cd.keys()}"
    )
    # 四类路径标签均须有样本（当前固定种子 seed=42 下实测分布：
    # {tp_first:46, sl_first:46, time_up:7, time_down:5}，四类全非零；
    # 若未来改种子，合成行情参数需保证四类路径均可覆盖）
    assert all(v > 0 for v in cd.values()), (
        f"class_distribution 存在零样本类，合成数据未能覆盖四类路径: {cd}"
    )

    # ------------------------------------------------------------------
    # 阶段 2：推理（predict_cnn_signals，七列信号帧）
    # ------------------------------------------------------------------
    start_dt = frame["datetime"][_LOOKBACK + 5].date()
    end_dt = frame["datetime"][-1].date()

    from aitrade.cnn.predictor import predict_cnn_signals

    signal_df = predict_cnn_signals(
        model_name=model_name,
        start=start_dt,
        end=end_dt,
    )

    # --- 断言信号帧 ---
    # path_class 输出八列（原七列 + objective 末列；Task 4 cnn-eval-honesty-fixes）
    expected_cols = [
        "datetime", "vt_symbol", "signal",
        "prob_tp", "prob_sl", "prob_time_up", "prob_time_down", "objective",
    ]
    assert signal_df.columns == expected_cols, f"信号帧列序错误: {signal_df.columns}"
    assert signal_df.height > 0, "信号帧为空，推理无输出"

    # Property 3：逐行 signal == prob_tp
    for i, row in enumerate(signal_df.to_dicts()):
        assert row["signal"] == row["prob_tp"], (
            f"第 {i} 行 signal={row['signal']} ≠ prob_tp={row['prob_tp']}"
        )

    # 四概率行和 ≈ 1
    prob_sum = (
        signal_df["prob_tp"]
        + signal_df["prob_sl"]
        + signal_df["prob_time_up"]
        + signal_df["prob_time_down"]
    ).to_list()
    for i, s in enumerate(prob_sum):
        assert abs(s - 1.0) < 1e-5, f"第 {i} 行概率和={s}，偏差过大"

    # ------------------------------------------------------------------
    # 阶段 3：veto 回测（CNNSignalStrategy + BacktestingEngine）
    # ------------------------------------------------------------------
    bars = _signal_df_to_bars(signal_df)

    engine = BacktestingEngine(data_loader=_FakeLoader(bars))
    engine.set_parameters(
        vt_symbols=[_TARGET],
        interval="d",
        start=signal_df["datetime"][0],
        end=signal_df["datetime"][-1] + timedelta(days=2),
        capital=1_000_000,
    )
    engine.add_strategy(
        CNNSignalStrategy,
        {
            "buy_threshold": 0.3,   # 低阈值确保有信号触发
            "exit_mode": "oco",
            "take_profit": 0.03,
            "stop_loss": 0.03,
            "hold_days": 5,
            "veto_threshold": 0.3,  # 低否决阈值，prob_sl>=0.3 时否决
        },
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()

    # --- 断言回测统计 ---
    # engine 统计可计算（先 calculate_result 再 calculate_statistics，不崩溃即可）
    try:
        engine.calculate_result()
        engine.calculate_statistics()
    except Exception as exc:
        pytest.fail(f"calculate_result/calculate_statistics 异常: {exc}")

    # _veto_count > 0：当前数据实测否决触发 2 次（veto_threshold=0.3，buy_threshold=0.3，seed=42）；
    # >=0 对任意整数恒真，无守护力；若 veto 机制失效（计数始终为 0）此处会暴露问题
    assert hasattr(engine.strategy, "_veto_count"), "CNNSignalStrategy 缺少 _veto_count 属性"
    assert engine.strategy._veto_count > 0, (
        f"_veto_count={engine.strategy._veto_count}，veto 否决机制应至少触发一次"
        "（当前 buy_threshold=0.3，veto_threshold=0.3，seed=42 下实测=2）"
    )

    # ------------------------------------------------------------------
    # 阶段 4：全链路耗时报告（供 CI 参考，不断言）
    # ------------------------------------------------------------------
    elapsed_total = time.perf_counter() - t0
    print(
        f"\n[e2e 耗时] 训练={elapsed_train:.1f}s  总计={elapsed_total:.1f}s"
        f"  信号行数={signal_df.height}"
        f"  class_distribution={cd}"
        f"  veto_count={engine.strategy._veto_count}"
    )
