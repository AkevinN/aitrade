"""
Task 5 验收测试：CNNSignalStrategy 入场否决（veto_threshold + _entry_vetoed）。

覆盖：
5.2 示例测试
  ① path 信号帧（含 prob_sl 高值）+ veto_threshold=0.5 → 全程无买入成交，否决计数 > 0
  ② 同一信号帧 + veto_threshold=1.0（默认）→ 有买入（等效关闭）
  ③ 三列信号帧（无 prob_sl）+ 任意 veto_threshold → 行为与 veto=1.0 完全一致
  ④ 否决不影响出场：建仓后 prob_sl 飙高，到期出场照常
5.3 属性测试 Property 4（Hypothesis）：
  (a) 全程 veto → 零买入成交（三种 exit_mode）
  (b) 无 prob_sl 列时目标序列与 veto=1.0 基准完全一致

Feature: cnn-path-multiclass-head
Property 4: 对任意含 prob_sl 列的信号帧与 veto_threshold ∈ (0,1]，当某时点 prob_sl >=
veto_threshold 时，CNNSignalStrategy 在该时点不产生任何买入目标（三种 exit_mode 下均
成立）；当信号帧不含 prob_sl 列时，策略的目标序列与改造前实现完全一致。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.backtest.engine import BacktestingEngine
from aitrade.backtest.types import BarData, Direction
from aitrade.cnn.strategy import CNNSignalStrategy

SYMBOL = "TEST.SZSE"
START = datetime(2026, 1, 5)


# ---------------------------------------------------------------------------
# 共享夹具工具（复用 iteration0/oco 测试中已验证的模式）
# ---------------------------------------------------------------------------

class FakeLoader:
    """实现 BarDataLoader 协议的合成数据源，无外部依赖。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list[BarData]:
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


def _build_bars(closes: list[float]) -> tuple[list[BarData], list[datetime]]:
    """构造满足撮合条件的合成日线：low 低于前收，保证限价单可成交。

    Args:
        closes: 各根 bar 的收盘价列表。

    Returns:
        (bars 列表, datetime 列表) 元组。
    """
    days = [START + timedelta(days=i) for i in range(len(closes))]
    bars: list[BarData] = []
    for i, close in enumerate(closes):
        prev_close = closes[i - 1] if i > 0 else close
        open_price = prev_close
        high_price = max(open_price, close) + 1.0
        low_price = min(open_price, close) - 1.0
        bars.append(
            BarData(
                symbol="TEST",
                exchange="SZSE",
                datetime=days[i],
                interval="d",
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close,
                volume=1_000_000,
            )
        )
    return bars, days


def _signal_df_path(
    days: list[datetime],
    probs_tp: list[float],
    probs_sl: list[float],
) -> pl.DataFrame:
    """构造 path_class 七列信号帧（signal=prob_tp，含 prob_sl 等列）。

    Args:
        days: 各行的 datetime 列表。
        probs_tp: 各行 prob_tp 值（同时作为 signal 列）。
        probs_sl: 各行 prob_sl 值。

    Returns:
        含 [datetime, vt_symbol, signal, prob_tp, prob_sl, prob_time_up, prob_time_down] 的 DataFrame。
    """
    n = len(days)
    return pl.DataFrame(
        {
            "datetime": days,
            "vt_symbol": [SYMBOL] * n,
            "signal": probs_tp,
            "prob_tp": probs_tp,
            "prob_sl": probs_sl,
            "prob_time_up": [0.1] * n,
            "prob_time_down": [0.1] * n,
        }
    )


def _signal_df_3col(days: list[datetime], probs: list[float]) -> pl.DataFrame:
    """构造 classification/regression 三列信号帧（无 prob_sl）。

    Args:
        days: 各行的 datetime 列表。
        probs: 各行 signal 值。

    Returns:
        含 [datetime, vt_symbol, signal] 的 DataFrame。
    """
    return pl.DataFrame(
        {
            "datetime": days,
            "vt_symbol": [SYMBOL] * len(days),
            "signal": probs,
        }
    )


def _run(
    bars: list[BarData],
    days: list[datetime],
    signal_df: pl.DataFrame,
    setting: dict,
) -> BacktestingEngine:
    """组装并运行一次回测，返回引擎实例供断言。

    Args:
        bars: 合成 BarData 列表。
        days: 对应的 datetime 列表。
        signal_df: 信号 DataFrame（三列或七列均可）。
        setting: CNNSignalStrategy 参数字典。

    Returns:
        完成回测后的 BacktestingEngine 实例。
    """
    engine = BacktestingEngine(data_loader=FakeLoader(bars))
    engine.set_parameters(
        vt_symbols=[SYMBOL],
        interval="d",
        start=days[0],
        end=days[-1] + timedelta(days=1),
        capital=1_000_000,
    )
    engine.add_strategy(CNNSignalStrategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    return engine


def _trades(engine: BacktestingEngine) -> list:
    """按 tradeid 排序返回所有成交记录。"""
    return sorted(engine.get_all_trades(), key=lambda t: int(t.tradeid))


def _buy_trades(engine: BacktestingEngine) -> list:
    """返回所有买入（Direction.LONG）成交记录。"""
    return [t for t in _trades(engine) if t.direction == Direction.LONG]


# ---------------------------------------------------------------------------
# 5.2 示例测试
# ---------------------------------------------------------------------------

def test_veto_blocks_all_entries_when_prob_sl_high() -> None:
    """① path 信号帧 prob_sl 全程高值 + veto_threshold=0.5 → 无买入成交，否决计数 > 0。"""
    closes = [100.0 + i for i in range(6)]
    bars, days = _build_bars(closes)
    # prob_tp 全程 > buy_threshold(0.6)，prob_sl 全程 0.8 >= veto 0.5
    signal_df = _signal_df_path(days, [0.9] * 6, [0.8] * 6)

    engine = _run(
        bars, days, signal_df,
        {"buy_threshold": 0.6, "exit_mode": "threshold", "veto_threshold": 0.5},
    )

    assert len(_buy_trades(engine)) == 0, "全程否决应无任何买入成交"
    assert engine.strategy._veto_count > 0, "否决计数应 > 0"


def test_veto_disabled_at_1_allows_entry() -> None:
    """② 同一信号帧 + veto_threshold=1.0（默认）→ 有买入（等效关闭）。"""
    closes = [100.0 + i for i in range(6)]
    bars, days = _build_bars(closes)
    # prob_sl=0.8 < 1.0（veto 阈值），不触发否决
    signal_df = _signal_df_path(days, [0.9] * 6, [0.8] * 6)

    engine = _run(
        bars, days, signal_df,
        {"buy_threshold": 0.6, "exit_mode": "threshold", "veto_threshold": 1.0},
    )

    assert len(_buy_trades(engine)) > 0, "veto=1.0 等效关闭，应有买入成交"
    assert engine.strategy._veto_count == 0, "veto=1.0 下否决计数应为 0"


def test_no_prob_sl_column_behavior_matches_veto_1() -> None:
    """③ 三列信号帧（无 prob_sl）+ 低 veto_threshold → 行为与 veto=1.0 完全一致。"""
    closes = [100.0 + i for i in range(6)]
    bars, days = _build_bars(closes)
    probs = [0.9, 0.9, 0.1, 0.1, 0.9, 0.5]
    signal_3col = _signal_df_3col(days, probs)

    # 3列信号 + 低 veto（理论上应触发否决但无 prob_sl 列）
    engine_3col = _run(
        bars, days, signal_3col,
        {"buy_threshold": 0.6, "exit_mode": "threshold", "veto_threshold": 0.3},
    )
    # 3列信号 + veto=1.0（基准）
    engine_baseline = _run(
        bars, days, signal_3col,
        {"buy_threshold": 0.6, "exit_mode": "threshold", "veto_threshold": 1.0},
    )

    trades_3col = _trades(engine_3col)
    trades_baseline = _trades(engine_baseline)
    assert len(trades_3col) == len(trades_baseline), (
        "无 prob_sl 列时，任意 veto 值行为应与 veto=1.0 一致（向后兼容）"
    )
    for t1, t2 in zip(trades_3col, trades_baseline, strict=True):
        assert t1.direction == t2.direction
        assert t1.datetime.date() == t2.datetime.date()


def test_veto_does_not_affect_exit_fixed_hold() -> None:
    """④ 否决不影响出场：建仓后 prob_sl 飙高，fixed_hold 到期出场照常。

    构造：D0 低 prob_sl 允许建仓；D1 开始 prob_sl 飙高；验证 D1 成交建仓，
    hold_days=1 后 D2 正常平仓。
    """
    closes = [100.0 + i for i in range(5)]
    bars, days = _build_bars(closes)
    # D0 prob_sl=0.1（允许入场），其余行 prob_sl=0.9（否决——但已持仓不影响）
    probs_tp = [0.9, 0.5, 0.5, 0.5, 0.5]
    probs_sl = [0.1, 0.9, 0.9, 0.9, 0.9]
    signal_df = _signal_df_path(days, probs_tp, probs_sl)

    engine = _run(
        bars, days, signal_df,
        {"buy_threshold": 0.6, "exit_mode": "fixed_hold", "hold_days": 1, "veto_threshold": 0.5},
    )

    trades = _trades(engine)
    buy_trades = [t for t in trades if t.direction == Direction.LONG]
    sell_trades = [t for t in trades if t.direction == Direction.SHORT]

    assert len(buy_trades) == 1, "D0 信号应触发一次买入"
    assert len(sell_trades) == 1, "固定持有到期应有一次平仓出场"
    # 平仓是 D2（D1 建仓，计数 1，D2 出场决策，D2 成交）
    assert sell_trades[0].datetime.date() == days[2].date()


def test_veto_log_written_on_trigger() -> None:
    """① 否决触发时，engine.logs 中应有包含'否决买入'的条目。"""
    closes = [100.0 + i for i in range(4)]
    bars, days = _build_bars(closes)
    signal_df = _signal_df_path(days, [0.9] * 4, [0.8] * 4)

    engine = _run(
        bars, days, signal_df,
        {"buy_threshold": 0.6, "exit_mode": "fixed_hold", "hold_days": 1, "veto_threshold": 0.5},
    )

    veto_logs = [log for log in engine.logs if "否决买入" in log]
    assert len(veto_logs) > 0, "否决触发时 engine.logs 应含'否决买入'日志"


# ---------------------------------------------------------------------------
# 5.3 属性测试 Property 4（Hypothesis）
# ---------------------------------------------------------------------------
# max_examples=25：每例需启动 BacktestingEngine 跑完整回测，25 例约 15-30s；
# 避免单文件超 90s 的同时保持足够的随机覆盖。
# ---------------------------------------------------------------------------

_EXIT_MODES = st.sampled_from(["threshold", "fixed_hold", "oco"])


@st.composite
def _path_signal_strategy(draw):
    """生成 (bars, days, path_signal_df, veto_threshold, exit_mode, setting) 元组。

    随机域：
    - N=5~12 根日线
    - prob_tp ∈ [0.7, 0.99]（全程超买入阈值，确保信号足够强，否决效果明显）
    - prob_sl ∈ [veto_threshold, 1.0)（确保全程触发否决）
    - veto_threshold ∈ (0, 1.0)
    - exit_mode 三种之一
    """
    n = draw(st.integers(min_value=5, max_value=12))
    veto = draw(st.floats(min_value=0.01, max_value=0.99))
    exit_mode = draw(_EXIT_MODES)

    # prob_tp 全程 > 0.6（buy_threshold），保证入场条件满足——只差 veto 拦截
    probs_tp = [draw(st.floats(min_value=0.7, max_value=0.99)) for _ in range(n)]
    # prob_sl 全程 >= veto，保证全程否决
    probs_sl = [draw(st.floats(min_value=veto, max_value=1.0)) for _ in range(n)]

    closes = [100.0 + i for i in range(n)]
    bars, days = _build_bars(closes)
    signal_df = _signal_df_path(days, probs_tp, probs_sl)

    setting: dict = {"buy_threshold": 0.6, "exit_mode": exit_mode, "veto_threshold": veto}
    if exit_mode in ("fixed_hold", "oco"):
        setting["hold_days"] = 2
    if exit_mode == "oco":
        setting["take_profit"] = 0.5   # 很大，不易触发
        setting["stop_loss"] = 0.5

    return bars, days, signal_df, veto, exit_mode, setting


@given(params=_path_signal_strategy())
@settings(max_examples=25, deadline=None)
def test_property4a_full_veto_yields_zero_buys(params) -> None:
    """Property 4(a): 全程 prob_sl >= veto_threshold → 三种 exit_mode 均无买入成交。

    Feature: cnn-path-multiclass-head, Property 4。
    """
    bars, days, signal_df, veto, exit_mode, setting = params
    engine = _run(bars, days, signal_df, setting)
    assert len(_buy_trades(engine)) == 0, (
        f"全程 veto（veto_threshold={veto}, exit_mode={exit_mode}）下不应有买入成交"
    )


@st.composite
def _no_prob_sl_strategy(draw):
    """生成不含 prob_sl 列的信号帧参数，用于验证向后兼容。"""
    n = draw(st.integers(min_value=5, max_value=12))
    exit_mode = draw(_EXIT_MODES)
    veto = draw(st.floats(min_value=0.01, max_value=0.99))

    # prob 随机，可能超也可能不超买入阈值
    probs = [draw(st.floats(min_value=0.0, max_value=1.0)) for _ in range(n)]
    closes = [100.0 + i for i in range(n)]
    bars, days = _build_bars(closes)
    signal_df = _signal_df_3col(days, probs)

    setting_low_veto = {"buy_threshold": 0.6, "exit_mode": exit_mode, "veto_threshold": veto}
    setting_veto1 = {"buy_threshold": 0.6, "exit_mode": exit_mode, "veto_threshold": 1.0}
    if exit_mode in ("fixed_hold", "oco"):
        setting_low_veto["hold_days"] = 2
        setting_veto1["hold_days"] = 2
    if exit_mode == "oco":
        for s in (setting_low_veto, setting_veto1):
            s["take_profit"] = 0.5
            s["stop_loss"] = 0.5

    return bars, days, signal_df, setting_low_veto, setting_veto1, exit_mode


@given(params=_no_prob_sl_strategy())
@settings(max_examples=25, deadline=None)
def test_property4b_no_prob_sl_matches_veto1_baseline(params) -> None:
    """Property 4(b): 无 prob_sl 列时，任意 veto 值成交序列与 veto=1.0 基准完全一致。

    Feature: cnn-path-multiclass-head, Property 4。
    守护向后兼容：classification/regression 模型不含 prob_sl，否决应恒为 False。
    """
    bars, days, signal_df, setting_low_veto, setting_veto1, exit_mode = params
    engine_test = _run(bars, days, signal_df, setting_low_veto)
    engine_base = _run(bars, days, signal_df, setting_veto1)

    trades_test = _trades(engine_test)
    trades_base = _trades(engine_base)

    assert len(trades_test) == len(trades_base), (
        f"无 prob_sl 列时（exit_mode={exit_mode}），成交笔数应与 veto=1.0 一致，"
        f"实际 {len(trades_test)} vs {len(trades_base)}"
    )
    for t1, t2 in zip(trades_test, trades_base, strict=True):
        assert t1.direction == t2.direction, "成交方向应一致"
        assert t1.datetime.date() == t2.datetime.date(), "成交日期应一致"
