"""
CNN 回测「信号 ↔ 行情」对齐自测脚本（排查 P0：信号与 bar 的 datetime 粒度不匹配）。

背景
----
`api/cnn.py` 的 CNN 回测把引擎 interval 硬编码为 "d"，但 `predict_cnn_signals`
用的是模型训练时的 `input_interval`。若模型是分钟级（1m / 10m ...），信号的 datetime
是分钟时间戳，而引擎按日线加载，二者 datetime 永不相等：
`BacktestingEngine.get_signal` 永远返回空 → 全程无成交（且只在日志 warn，不报错）。

本脚本对每个模型对比两种 interval 下的对齐情况：
  - 当前行为：引擎 interval="d"
  - 修复方向：引擎 interval = 模型 input_interval
并据此定位 P0、验证修复方向。

用法
----
    cd backend && .venv/bin/python scripts/cnn_backtest_alignment_check.py [模型名 ...]
    不传模型名 → 自动检测 cnn_models 目录下所有模型。
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

# 支持直接 `python scripts/xxx.py` 运行：把 backend 根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from aitrade.alpha import AlphaLab  # noqa: E402
from aitrade.backtest.engine import BacktestingEngine  # noqa: E402
from aitrade.cnn.predictor import predict_cnn_signals  # noqa: E402
from aitrade.cnn.storage import CNN_MODEL_DIR  # noqa: E402
from aitrade.cnn.strategy import CNNSignalStrategy  # noqa: E402
from aitrade.config import ALPHA_LAB_PATH  # noqa: E402

# 诊断窗口（自训练起始日起，足够产出若干信号即可；越短跑得越快）
WINDOW_DAYS = 12

lab = AlphaLab(ALPHA_LAB_PATH)


def _to_date(value) -> date:
    """把 checkpoint 里的 start/end 容错转成 date。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _bar_datetimes(vt_symbol: str, interval: str, start: datetime, end: datetime) -> set[datetime]:
    """引擎视角加载 bar 后的 datetime 集合（与 get_signal 内部口径一致：去 tz）。"""
    bars = lab.load_bar_data(vt_symbol, interval, start, end)
    return {b.datetime.replace(tzinfo=None) for b in bars}


def _run_engine(target, interval, signal_df, start_dt, end_dt, buy, sell, capital=1_000_000):
    """按指定 interval 真实跑一遍回测，返回 (成交笔数, 回放bar数)。"""
    engine = BacktestingEngine(data_loader=lab)
    engine.set_parameters([target], interval, start_dt, end_dt, capital=capital)
    # 缺合约配置时给默认值（与 api/cnn.py 的兜底保持一致）
    if target not in engine.sizes:
        engine.sizes[target] = 1
        engine.priceticks[target] = 0.01
        engine.long_rates[target] = 0.0003
        engine.short_rates[target] = 0.0003
    engine.add_strategy(
        CNNSignalStrategy,
        {"buy_threshold": buy, "sell_threshold": sell},
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()
    return engine.trade_count, len(engine.dts)


def check_model(name: str) -> str:
    """检测单个模型的对齐情况，返回结论标签。"""
    print("=" * 72)
    print(f"模型: {name}")

    ck = torch.load(str(CNN_MODEL_DIR / f"{name}.pt"), map_location="cpu", weights_only=False)
    tc = ck["train_config"]
    interval = str(tc.get("input_interval", "d"))
    target = str(tc["target_symbol"])
    train_start = _to_date(tc.get("start"))
    train_end = _to_date(tc.get("end"))
    print(f"  input_interval={interval}  target={target}  训练区间={train_start} ~ {train_end}")

    start = train_start
    end = min(train_end, start + timedelta(days=WINDOW_DAYS))
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())
    print(f"  诊断窗口: {start} ~ {end}（落在训练区间内，仅验证对齐与成交，不代表绩效）")

    # 1) 生成信号
    signal_df = predict_cnn_signals(name, start, end)
    if signal_df.is_empty():
        print("  [跳过] 该窗口未产出任何信号")
        return "数据不足"

    sig_dts = {
        (d.replace(tzinfo=None) if isinstance(d, datetime) else d)
        for d in signal_df["datetime"].to_list()
    }
    s = signal_df["signal"]
    sample = [str(x) for x in signal_df["datetime"].head(3).to_list()]
    print(f"  信号: {signal_df.height} 条  datetime样例={sample}")
    print(f"        概率 min/mean/max = {s.min():.3f} / {s.mean():.3f} / {s.max():.3f}")

    # 阈值按信号分布微调，确保「对齐时」确实能触发买卖（避免阈值问题误判为无成交）
    q55, q45 = float(s.quantile(0.55)), float(s.quantile(0.45))
    buy = min(max(q55, 0.50), 0.99)
    sell = max(min(q45, buy - 0.01), 0.01)
    print(f"  自适应阈值: buy>{buy:.3f}  sell<{sell:.3f}")

    # 2) 当前行为：引擎 interval="d"
    d_dts = _bar_datetimes(target, "d", start_dt, end_dt)
    match_d = len(sig_dts & d_dts)
    trades_d, ndts_d = _run_engine(target, "d", signal_df, start_dt, end_dt, buy, sell)
    print(f"  [当前 interval='d']      日线bar={len(d_dts):>4}  信号∩bar={match_d:>4}  回放bar={ndts_d:>4}  成交={trades_d}")

    # 3) 修复方向：引擎 interval = 模型 input_interval
    iv_dts = _bar_datetimes(target, interval, start_dt, end_dt)
    match_iv = len(sig_dts & iv_dts)
    trades_iv, ndts_iv = _run_engine(target, interval, signal_df, start_dt, end_dt, buy, sell)
    print(f"  [修复 interval='{interval}']  bar={len(iv_dts):>4}  信号∩bar={match_iv:>4}  回放bar={ndts_iv:>4}  成交={trades_iv}")

    # 结论
    if match_d == 0 and match_iv > 0:
        verdict = "P0确认"
        print(f"  >>> 结论: ❌ P0 确认 — 当前日线对齐命中=0、成交=0；改用 interval='{interval}' 后对齐命中={match_iv}、成交={trades_iv}")
    elif match_d > 0 and trades_d > 0:
        verdict = "对齐正常"
        print("  >>> 结论: ✅ 对齐正常 — 当前 interval='d' 即有成交（该模型为日线）")
    else:
        verdict = "异常"
        print("  >>> 结论: ⚠️ 异常 — 信号或行情数据不足，需人工排查")
    return verdict


def main() -> int:
    names = sys.argv[1:]
    if not names:
        names = [f.stem for f in sorted(CNN_MODEL_DIR.glob("*.pt"))]
    if not names:
        print("未找到任何 CNN 模型，请先训练后再运行本脚本。")
        return 1

    print(f"待检测模型: {names}")
    results: dict[str, str] = {}
    for n in names:
        try:
            results[n] = check_model(n)
        except Exception as exc:  # noqa: BLE001
            print(f"  [错误] {n}: {exc}")
            traceback.print_exc()
            results[n] = "错误"

    print("=" * 72)
    print("汇总:")
    for n, v in results.items():
        print(f"  {n}: {v}")
    # 只要有任一模型命中 P0，返回非零便于 CI/脚本判断
    return 0 if all(v == "对齐正常" for v in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
