"""
CNN 数据集构建 —— 标签生成与分组样本装配。

从 model.py 抽出。数据加载、对齐、特征通道与分组张量装配复用 features 模块，
保证训练与推理走同一套底层逻辑。
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from ..backtest.oco import simulate_oco_exit
from .features import (
    FEATURE_NAMES,
    _align_frames_by_datetime,
    _build_grouped_tensor,
    _compute_features,
    _extract_aligned_bars,
    _load_market_frame,
    normalize_observation_groups,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径多分类标签常量（path_class objective）
# ---------------------------------------------------------------------------
# 四类出场路径对应的浮点标签值（y 数组以 float32 存储，trainer 侧再转 long）：
#   0 = 止盈先触发（TP First）
#   1 = 止损先触发（SL First）
#   2 = 时间止损 + 方向向上（Time Exit, Up）
#   3 = 时间止损 + 方向向下（Time Exit, Down）
PATH_TP_FIRST: float = 0.0
PATH_SL_FIRST: float = 1.0
PATH_TIME_UP: float = 2.0
PATH_TIME_DOWN: float = 3.0

# 类别名称元组——顺序与上方 PATH_TP_FIRST..PATH_TIME_DOWN 的 0~3 编码一一对应。
# 供 trainer 侧构建指标字典时共享，消除硬编码。
PATH_CLASS_NAMES: tuple[str, ...] = ("tp_first", "sl_first", "time_up", "time_down")


def _normalize_label_spec(label_spec: dict[str, Any] | None) -> dict[str, Any]:
    """规整 label 配置字典，补全缺省值并统一 enum 为字符串值。

    负责将 API 传入的 Pydantic 枚举（如 LabelMode.OCO）转为底层字符串（"oco"），
    并为 horizon_bars/oco/threshold/neutral_policy/price_ref 等字段设置默认值。

    Args:
        label_spec: 原始 label 配置字典；None 等价于 mode=next_bar 的默认配置。

    Returns:
        规整后的配置字典，所有枚举字段均为字符串，缺省字段已补全。

    Raises:
        ValueError: oco 模式的 max_hold < 1，或 take_profit/stop_loss 非正时抛出。
    """
    spec = dict(label_spec or {})
    mode = spec.get("mode") or "next_bar"
    # mode 可能来自 API 的 pydantic 枚举（LabelMode）；统一规整为字符串值，
    # 否则 build_dataset 中 str(LabelMode.OCO) 会得到 "LabelMode.OCO" 而非 "oco"，
    # 导致 OCO 标签分支失效（点对点模式因 enum 的 == 比较侥幸可用）。
    mode = getattr(mode, "value", mode)
    spec["mode"] = mode
    if mode == "horizon_bars":
        spec["horizon"] = int(spec.get("horizon") or 1)

    # OCO（三重障碍）路径依赖标签：持有期内先触发止盈/止损则按触发价与时刻计收益，
    # 全程未触发则在第 max_hold+1 根开盘按时间止损平仓（与 backtest/oco.simulate_oco_exit 同口径）。
    if mode == "oco":
        # max_hold 复用 horizon 语义（最大持有 bar 数）；缺省 10。
        # 注意区分"未提供"与"显式 0"：显式 0 应判非法，故用 in 判断而非 or。
        if "max_hold" in spec and spec.get("max_hold") is not None:
            max_hold = int(spec["max_hold"])
        elif spec.get("horizon"):
            max_hold = int(spec["horizon"])
        else:
            max_hold = 10
        if max_hold < 1:
            raise ValueError("oco 标签模式的 max_hold 必须 >= 1")
        spec["max_hold"] = max_hold
        take_profit = float(spec.get("take_profit") or 0.0)
        stop_loss = float(spec.get("stop_loss") or 0.0)
        if take_profit <= 0.0 or stop_loss <= 0.0:
            raise ValueError(
                "oco 标签模式需要正的 take_profit 与 stop_loss（如 0.03 表示 3%）"
            )
        spec["take_profit"] = take_profit
        spec["stop_loss"] = stop_loss
        # 同一根 bar 止盈止损都触发、日内先后未知时是否保守假设止损先到（默认 True，与回测一致）
        spec["stop_first"] = bool(spec.get("stop_first", True))

    # 最小有效波动阈值（去噪 dead-zone），单位为收益率：0.005 = 0.5%。
    # 0 表示关闭去噪，保持旧的 future_return > 0 二分类行为（向后兼容）。
    threshold = float(spec.get("threshold") or 0.0)
    spec["threshold"] = threshold if threshold > 0.0 else 0.0

    # 中性样本（|未来收益| <= 阈值）处理策略：drop=丢弃，negative=并入下跌类
    neutral_policy = str(spec.get("neutral_policy") or "drop").lower()
    if neutral_policy not in ("drop", "negative"):
        neutral_policy = "drop"
    spec["neutral_policy"] = neutral_policy

    # 标签收益计价口径（与回测撮合成交价一一对应，见 backtest/engine.fill_price_mode）：
    #   close      = 收盘到收盘（旧/研究口径，实盘吃不到）
    #   next_open  = 次开盘到次开盘（对齐 T+1 开盘成交）
    #   next_close = 次收盘到次收盘（对齐 T+1 收盘价 MOC 成交）
    #   next_vwap  = 次日均价到次日均价（对齐 T+1 全天均价 VWAP 成交）
    price_ref = str(spec.get("price_ref") or "close").lower()
    if price_ref not in ("close", "next_open", "next_close", "next_vwap"):
        price_ref = "close"
    spec["price_ref"] = price_ref
    return spec


def _label_from_return(
    future_return: float,
    threshold: float,
    neutral_policy: str,
) -> float | None:
    """把未来收益映射为方向标签。

    - threshold <= 0：保持旧行为（收益 > 0 记为上涨，其余为下跌），不丢样本。
    - threshold > 0：引入去噪 dead-zone：
        收益 > +阈值 → 1.0（上涨）
        收益 < -阈值 → 0.0（下跌）
        |收益| <= 阈值 视为噪声，按 neutral_policy 处理
        （drop → 返回 None 由调用方丢弃；negative → 并入下跌类 0.0）
    """
    if threshold <= 0.0:
        return 1.0 if future_return > 0 else 0.0
    if future_return > threshold:
        return 1.0
    if future_return < -threshold:
        return 0.0
    if neutral_policy == "negative":
        return 0.0
    return None


def _compute_label_return(
    anchor: int,
    future_index: int,
    price_ref: str,
    open_series: np.ndarray,
    close_series: np.ndarray,
    total_steps: int,
    vwap_series: np.ndarray | None = None,
) -> float | None:
    """按计价口径计算锚点到未来时刻的收益率。

    口径说明：
    - close（旧/研究口径）：close[anchor] → close[future_index]。
    - next_open（可执行）：open[anchor+1] → open[future_index+1]，
      对应「T 收盘出信号、T+1 开盘建仓、目标周期后开盘平仓」，
      剔除 close[anchor]→open[anchor+1] 的隔夜跳空。
    - next_close（可执行）：close[anchor+1] → close[future_index+1]，
      对应「T+1 收盘价(MOC)成交」。
    - next_vwap（可执行）：vwap[anchor+1] → vwap[future_index+1]，
      对应「T+1 全天均价(VWAP)成交」；vwap_series 为 None 时回退到 close。

    next_* 口径需要 anchor+1/future_index+1 的价格；越界返回 None（无前视）。

    Args:
        anchor: 当前锚点在时间序列中的下标。
        future_index: 未来出场时刻的下标。
        price_ref: 计价口径，"close" | "next_open" | "next_close" | "next_vwap"。
        open_series: 目标证券开盘价序列，形状 [T]。
        close_series: 目标证券收盘价序列，形状 [T]。
        total_steps: 时间序列总长度（即 T）。
        vwap_series: 均价序列，形状 [T]；仅 next_vwap 口径使用，None 时退化为 close。

    Returns:
        收益率（浮点，如 0.023 表示 +2.3%）；越界时返回 None。
    """
    if price_ref in ("next_open", "next_close", "next_vwap"):
        entry_index = anchor + 1
        exit_index = future_index + 1
        if entry_index >= total_steps or exit_index >= total_steps:
            return None
        if price_ref == "next_open":
            entry_series = exit_series = open_series
        elif price_ref == "next_close":
            entry_series = exit_series = close_series
        else:  # next_vwap
            entry_series = exit_series = (
                vwap_series if vwap_series is not None else close_series
            )
        base = max(float(entry_series[entry_index]), 1e-8)
        future_price = float(exit_series[exit_index])
    else:
        base = max(float(close_series[anchor]), 1e-8)
        future_price = float(close_series[future_index])
    return (future_price - base) / base


def _oco_label_value(
    anchor: int,
    open_series: np.ndarray,
    high_series: np.ndarray,
    low_series: np.ndarray,
    label_spec: dict[str, Any],
    objective: str,
    threshold: float,
    neutral_policy: str,
) -> tuple[str, float | None, float | None]:
    """OCO（三重障碍）路径依赖标签。

    建仓对齐 A 股 T+1：在 anchor+1 根开盘建仓，复用 backtest.oco.simulate_oco_exit
    逐根扫描持有期内的止盈/止损触发，到期未触发按时间止损（次开盘）平仓。

    返回 (status, label_value, future_return)：
    - "skip"   ：越界 / 无前视无法形成完整样本，调用方计入 skipped 并丢弃；
    - "neutral"：落入去噪 dead-zone，调用方计入 skipped_neutral 并丢弃；
    - "ok"     ：有效样本。

    标签口径：
    - regression：label = 真实出场收益 ret（先触止盈/止损即按触发价计，天然路径依赖）。
    - classification：止盈触发→1，止损触发→0，时间止损→按 ret 符号并套用去噪阈值。
    - path_class：四分类路径标签（使用模块级常量 PATH_TP_FIRST / PATH_SL_FIRST /
      PATH_TIME_UP / PATH_TIME_DOWN，值为 0.0~3.0，trainer 侧转 long 后用 CrossEntropyLoss）：
        * reason=="tp"  → PATH_TP_FIRST（0.0）
        * reason=="sl"  → PATH_SL_FIRST（1.0）
        * reason=="time" 且 ret >  threshold → PATH_TIME_UP（2.0）
        * reason=="time" 且 ret < -threshold → PATH_TIME_DOWN（3.0）
        * dead-zone（time 且 |ret| <= threshold）：
            neutral_policy=="negative" → PATH_TIME_DOWN；否则 → "neutral"（丢弃）
      threshold=0 的边界：ret>0→TIME_UP，ret<0→TIME_DOWN，ret==0 落 dead-zone（按 neutral_policy）；
      这与 classification 的 threshold<=0 "ret>0→1 否则→0" 有意不同——ret==0 无方向信息。

    Args:
        anchor: 锚点下标；建仓时刻为 anchor+1。
        open_series: 目标证券开盘价序列，形状 [T]。
        high_series: 目标证券最高价序列，形状 [T]。
        low_series: 目标证券最低价序列，形状 [T]。
        label_spec: 规整后的 OCO label 配置字典（take_profit/stop_loss/max_hold/stop_first）。
        objective: "regression" | "classification" | "path_class"。
        threshold: 去噪 dead-zone 阈值（收益率，>=0）。
        neutral_policy: "drop"（dead-zone 样本丢弃）或 "negative"（并入下跌/TIME_DOWN）。

    Returns:
        (status, label_value, future_return) 三元组：
        - status: "ok" | "neutral" | "skip"。
        - label_value: 有效时为浮点标签（regression→收益率；classification→0/1；
          path_class→0/1/2/3）；无效时为 None。
        - future_return: 实际出场收益率；skip 时为 None。
    """
    entry_index = anchor + 1
    if entry_index >= len(open_series):
        return ("skip", None, None)
    entry_price = float(open_series[entry_index])
    if entry_price <= 0:
        return ("skip", None, None)

    result = simulate_oco_exit(
        entry_index,
        entry_price,
        open_series,
        high_series,
        low_series,
        float(label_spec["take_profit"]),
        float(label_spec["stop_loss"]),
        int(label_spec["max_hold"]),
        stop_first=bool(label_spec.get("stop_first", True)),
    )
    if result is None:
        return ("skip", None, None)

    ret = float(result["ret"])
    reason = str(result["reason"])

    if objective == "regression":
        # 阈值仅用于剔除过小噪声样本；标签即连续出场收益
        if threshold > 0.0 and abs(ret) <= threshold:
            return ("neutral", None, ret)
        return ("ok", ret, ret)

    if objective == "path_class":
        # 四分类路径标签：按 OCO 出场原因细分方向
        if reason == "tp":
            return ("ok", PATH_TP_FIRST, ret)
        if reason == "sl":
            return ("ok", PATH_SL_FIRST, ret)
        # reason == "time"：按出场收益方向分 TIME_UP / TIME_DOWN；dead-zone 按 neutral_policy
        if ret > threshold:
            return ("ok", PATH_TIME_UP, ret)
        if ret < -threshold:
            return ("ok", PATH_TIME_DOWN, ret)
        # dead-zone：|ret| <= threshold（含 threshold=0 时 ret==0 的情形）
        if neutral_policy == "negative":
            return ("ok", PATH_TIME_DOWN, ret)
        return ("neutral", None, ret)

    # 分类：优先按触发原因定方向（与实盘止盈止损对齐）
    if reason == "tp":
        return ("ok", 1.0, ret)
    if reason == "sl":
        return ("ok", 0.0, ret)
    # 时间止损：按收益符号 + 去噪阈值映射
    label = _label_from_return(ret, threshold, neutral_policy)
    if label is None:
        return ("neutral", None, ret)
    return ("ok", label, ret)


def _build_session_last_index(datetimes: list[datetime]) -> tuple[dict[Any, int], list[Any]]:
    """构建「交易日 → 该日最后一根 bar 下标」的映射表。

    遍历 datetimes，每个交易日的最后出现下标即为该日最后一根 bar 的位置，
    供 session_close/next_session_close 标签模式定位出场时刻。

    Args:
        datetimes: 对齐后的时间序列，每项为 datetime 对象；顺序与 K 线行序一致。

    Returns:
        (day_to_last_index, ordered_days)：
        - day_to_last_index: date → 该日最后一根 bar 在 datetimes 中的下标。
        - ordered_days: 按首次出现顺序排列的交易日列表（date 对象）。
    """
    day_to_last_index: dict[Any, int] = {}
    ordered_days: list[Any] = []
    for index, dt in enumerate(datetimes):
        day_key = dt.date()
        if day_key not in day_to_last_index:
            ordered_days.append(day_key)
        day_to_last_index[day_key] = index
    return day_to_last_index, ordered_days


def _label_future_index(
    anchor: int,
    datetimes: list[datetime],
    label_spec: dict[str, Any],
    *,
    input_interval: str,
) -> int | None:
    """计算锚点对应的 label 未来出场时刻的下标。

    按 label_spec["mode"] 分派：
    - next_bar           → anchor + 1。
    - horizon_bars       → anchor + spec["horizon"]。
    - session_close      → 当日最后一根 bar（日线下禁用）。
    - next_session_close → 次交易日最后一根 bar。

    Args:
        anchor: 当前锚点在时间序列中的下标。
        datetimes: 对齐后的完整时间序列（datetime 对象列表）。
        label_spec: 规整后的 label 配置字典（已经 _normalize_label_spec 处理）。
        input_interval: K 线周期，如 "d"、"30m"；影响 session_close 的合法性校验。

    Returns:
        未来出场时刻在 datetimes 中的下标；越界或找不到时返回 None。

    Raises:
        ValueError: session_close 用于日线周期，或 mode 不在支持列表内时抛出。
    """
    mode = label_spec["mode"]
    if mode == "next_bar":
        future = anchor + 1
        return future if future < len(datetimes) else None

    if mode == "horizon_bars":
        future = anchor + int(label_spec.get("horizon") or 1)
        return future if future < len(datetimes) else None

    day_to_last_index, ordered_days = _build_session_last_index(datetimes)
    current_day = datetimes[anchor].date()

    if mode == "session_close":
        if input_interval == "d":
            raise ValueError("session_close 仅适用于日内数据，请选择分钟周期输入")
        future = day_to_last_index[current_day]
        return future if future > anchor else None

    if mode == "next_session_close":
        if current_day not in ordered_days:
            return None
        day_index = ordered_days.index(current_day)
        next_index = day_index + 1
        if next_index >= len(ordered_days):
            return None
        return day_to_last_index[ordered_days[next_index]]

    raise ValueError(f"不支持的标签模式: {mode}")


def build_dataset(
    vt_symbols: list[str],
    start: date,
    end: date,
    lookback: int = 30,
    target_symbol: str | None = None,
    on_progress: Optional[Callable] = None,
    observation_groups: list[dict[str, Any]] | None = None,
    input_data_kind: str = "bar",
    input_interval: str = "d",
    label_spec: dict[str, Any] | None = None,
    objective: str = "classification",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """构建分组感知的 CNN 训练样本集。

    加载各证券本地 K 线 → 按公共时间轴对齐 → 计算技术特征 →
    填入 [C, T, S, G] 张量 → 为每个锚点生成标签，返回样本矩阵与元数据。
    训练与推理共用同一套底层逻辑（features 模块），避免两处实现漂移。

    Args:
        vt_symbols: 参与训练的全量证券代码列表；observation_groups 为空时用于构造默认分组。
        start: 数据起始日期（含）。
        end: 数据结束日期（含）。
        lookback: 每个样本的回看 bar 数（时间窗口长度 T）。
        target_symbol: 预测目标证券代码；None 时取 vt_symbols[0]。
        on_progress: 进度回调 ``(percent: float, message: str) -> None``，可为 None。
        observation_groups: 语义分组配置列表；None 时退化为旧版兼容逻辑。
        input_data_kind: 数据种类，"bar"（K 线）或 "tick"（Tick 聚合）。
        input_interval: K 线周期，"d" | "1m" | "5m" | "10m" | "15m" | "30m" | "60m"。
        label_spec: label 配置字典（mode/horizon/threshold/price_ref 等）；None → 默认 next_bar。
        objective: 训练目标，"classification"（方向二分类）、"regression"（收益回归）
            或 "path_class"（四分类路径标签，需配合 label_spec.mode="oco"）。

    Returns:
        四元组 ``(X, y, group_mask, info)``：
        - X: 形状 [N, C, T, S, G]，float32，归一化前的特征张量（归一化在 trainer 中进行）。
        - y: 形状 [N]，float32；分类为 {0.0, 1.0}，回归为连续收益率，
          path_class 为 {0.0, 1.0, 2.0, 3.0}（trainer 侧转 long 后用 CrossEntropyLoss）。
        - group_mask: 形状 [1, 1, 1, S, G]，float32；有效证券位置为 1.0，占位为 0.0。
        - info: 元数据字典，含 symbols/groups/feature_names/dates/sample_anchor_dates 等，
          供 trainer 写入 checkpoint 和前端展示；sample_returns 键含每样本带符号收益，
          仅用于幅度加权训练，不写入 checkpoint。
          path_class 时额外含 "class_distribution" 键（tp_first/sl_first/time_up/time_down 各类样本数）。

    Raises:
        ValueError: 证券本地数据缺失、公共时间步不足、无法生成任何有效样本等情况时抛出；
            objective="path_class" 且 label_spec.mode≠"oco" 时抛出（在数据加载前触发）。
    """
    if on_progress:
        on_progress(5, "解析观测分组...")

    target_symbol, groups = normalize_observation_groups(
        target_symbol=target_symbol,
        observation_groups=observation_groups,
        vt_symbols=vt_symbols,
    )
    label_spec = _normalize_label_spec(label_spec)

    # path_class 路径标签依赖三重障碍判定，必须与 oco 标签模式配合使用；
    # 此校验在数据加载之前触发，属性测试无需真实行情即可覆盖。
    if objective == "path_class" and label_spec.get("mode") != "oco":
        raise ValueError(
            "objective=path_class 需要 label_spec.mode=oco"
            "（路径标签依赖三重障碍判定）"
        )

    ordered_symbols: list[str] = []
    for group in groups:
        for symbol in group["symbols"]:
            if symbol not in ordered_symbols:
                ordered_symbols.append(symbol)

    if on_progress:
        on_progress(10, f"加载 {len(ordered_symbols)} 个观测证券的 {input_interval} 数据...")

    symbol_frames: dict[str, pl.DataFrame] = {}
    missing_symbols: list[str] = []
    for index, vt_symbol in enumerate(ordered_symbols):
        try:
            symbol_frames[vt_symbol] = _load_market_frame(
                vt_symbol,
                start,
                end,
                input_data_kind=input_data_kind,
                input_interval=input_interval,
            )
            if on_progress:
                on_progress(
                    10 + 20 * (index + 1) / len(ordered_symbols),
                    f"已加载 {vt_symbol} ({index + 1}/{len(ordered_symbols)})",
                )
        except Exception as exc:
            logger.warning(f"加载 {vt_symbol} 失败: {exc}")
            missing_symbols.append(f"{vt_symbol}: {exc}")

    if missing_symbols:
        raise ValueError(
            "以下证券缺少本地数据，CNN 训练不会使用 mock 数据："
            + "; ".join(missing_symbols)
        )

    if on_progress:
        on_progress(32, "按公共时间轴对齐观测组...")

    symbols, aligned_df = _align_frames_by_datetime(symbol_frames)
    if target_symbol not in symbols:
        raise ValueError(f"目标证券 {target_symbol} 不在观测证券列表中")

    all_features: dict[str, np.ndarray] = {}
    for vt_symbol in symbols:
        all_features[vt_symbol] = _compute_features(_extract_aligned_bars(aligned_df, vt_symbol))

    total_steps = aligned_df.height
    if total_steps <= lookback:
        raise ValueError(
            f"公共时间步不足，当前仅 {total_steps}，至少需要大于 lookback({lookback})"
        )

    feature_channels = len(FEATURE_NAMES)
    group_count = len(groups)
    max_group_width = max(len(group["symbols"]) for group in groups)

    aligned, group_mask = _build_grouped_tensor(
        groups,
        all_features,
        feature_channels,
        total_steps,
        max_group_width,
        group_count,
    )

    target_close_series = aligned_df.select(pl.col(f"{target_symbol}__close")).to_numpy().reshape(-1)
    target_open_series = aligned_df.select(pl.col(f"{target_symbol}__open")).to_numpy().reshape(-1)
    # 目标证券日内高低价序列（列由 features 对齐时保留），供 OCO 标签判定止盈/止损触碰。
    target_high_series = aligned_df.select(pl.col(f"{target_symbol}__high")).to_numpy().reshape(-1)
    target_low_series = aligned_df.select(pl.col(f"{target_symbol}__low")).to_numpy().reshape(-1)
    # 目标证券的均价(VWAP)序列：成交额/成交量，缺失(量为 0 或无 turnover 列)回退到收盘价，
    # 供 price_ref=next_vwap 计价口径使用（与回测 fill_price_mode=vwap 成交对齐）。
    if f"{target_symbol}__turnover" in aligned_df.columns:
        target_turnover_series = aligned_df.select(
            pl.col(f"{target_symbol}__turnover")
        ).to_numpy().reshape(-1)
        target_volume_series = aligned_df.select(
            pl.col(f"{target_symbol}__volume")
        ).to_numpy().reshape(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            target_vwap_series = np.where(
                target_volume_series > 0,
                target_turnover_series / np.maximum(target_volume_series, 1e-12),
                target_close_series,
            ).astype(np.float64)
        # turnover 全为 0（部分数据源未提供）时 vwap 退化为 0，统一回退到收盘价
        target_vwap_series = np.where(
            target_vwap_series > 0, target_vwap_series, target_close_series
        )
    else:
        target_vwap_series = target_close_series
    aligned_dates: list[datetime] = aligned_df["datetime"].to_list()

    if on_progress:
        on_progress(
            48,
            f"构建语义张量: 特征={feature_channels}, 时间步={total_steps}, 宽度={max_group_width}, 分组={group_count}",
        )

    threshold: float = float(label_spec.get("threshold") or 0.0)
    neutral_policy: str = str(label_spec.get("neutral_policy") or "drop")
    price_ref: str = str(label_spec.get("price_ref") or "close")
    label_mode: str = str(label_spec.get("mode") or "next_bar")

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    returns_list: list[float] = []
    anchor_dates: list[str] = []
    skipped = 0
    skipped_neutral = 0

    for anchor in range(lookback - 1, total_steps):
        if label_mode == "oco":
            status, oco_label, oco_ret = _oco_label_value(
                anchor,
                target_open_series,
                target_high_series,
                target_low_series,
                label_spec,
                objective,
                threshold,
                neutral_policy,
            )
            if status == "skip":
                skipped += 1
                continue
            if status == "neutral":
                skipped_neutral += 1
                continue
            label_value = float(oco_label)
            future_return = float(oco_ret)
            snapshot = aligned[:, anchor - lookback + 1: anchor + 1, :, :]
            X_list.append(snapshot)
            y_list.append(label_value)
            returns_list.append(future_return)
            anchor_dates.append(aligned_dates[anchor].isoformat())
            continue

        future_index = _label_future_index(
            anchor,
            aligned_dates,
            label_spec,
            input_interval=input_interval,
        )
        if future_index is None:
            skipped += 1
            continue

        future_return = _compute_label_return(
            anchor,
            future_index,
            price_ref,
            target_open_series,
            target_close_series,
            total_steps,
            target_vwap_series,
        )
        if future_return is None:
            skipped += 1
            continue

        # 先判定标签：落入去噪 dead-zone 的样本直接丢弃，避免与 X 错位
        if objective == "regression":
            # 回归：标签即连续未来收益；阈值仅用于剔除过小噪声样本
            if threshold > 0.0 and abs(future_return) <= threshold:
                skipped_neutral += 1
                continue
            label_value: float = future_return
        else:
            label = _label_from_return(future_return, threshold, neutral_policy)
            if label is None:
                skipped_neutral += 1
                continue
            label_value = label

        snapshot = aligned[:, anchor - lookback + 1: anchor + 1, :, :]
        X_list.append(snapshot)
        y_list.append(label_value)
        returns_list.append(future_return)
        anchor_dates.append(aligned_dates[anchor].isoformat())

    if not X_list:
        raise ValueError(
            "没有生成任何有效样本，请扩大时间范围、调小标签阈值(threshold)或调整标签定义"
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # path_class 统计四类样本数，写入 info 供监控与展示
    class_distribution: dict[str, int] | None = None
    if objective == "path_class":
        # PATH_CLASS_NAMES 顺序与 PATH_TP_FIRST..PATH_TIME_DOWN 的 0~3 编码一一对应
        _labels = (PATH_TP_FIRST, PATH_SL_FIRST, PATH_TIME_UP, PATH_TIME_DOWN)
        class_distribution = {
            name: int(np.sum(y == lbl))
            for name, lbl in zip(PATH_CLASS_NAMES, _labels, strict=True)
        }

    if on_progress:
        neutral_note = f", 去噪丢弃={skipped_neutral}" if skipped_neutral > 0 else ""
        if objective == "regression":
            label_stat = f"收益均值={y.mean():.3%}, 标准差={y.std():.3%}"
        elif objective == "path_class" and class_distribution is not None:
            dist = class_distribution
            label_stat = (
                f"tp={dist['tp_first']}, sl={dist['sl_first']}, "
                f"time_up={dist['time_up']}, time_down={dist['time_down']}"
            )
        else:
            label_stat = f"正样本比例={y.mean():.2%}"
        on_progress(55, f"样本构建完成: X={X.shape}, {label_stat}{neutral_note}")

    info: dict[str, Any] = {
        "symbols": symbols,
        "groups": groups,
        "target_symbol": target_symbol,
        "feature_names": FEATURE_NAMES,
        "feature_channels": feature_channels,
        "group_count": group_count,
        "max_group_width": max_group_width,
        "lookback": lookback,
        "n_dates": total_steps,
        "dates": [dt.isoformat() for dt in aligned_dates],
        "sample_anchor_dates": anchor_dates,
        "input_data_kind": input_data_kind,
        "input_interval": input_interval,
        "label_spec": label_spec,
        "label_threshold": threshold,
        "price_ref": price_ref,
        "objective": objective,
        "skipped_for_label": skipped,
        "skipped_for_neutral": skipped_neutral,
        # 每样本带符号未来收益（与 X/y 同序），供训练侧做幅度加权；不写入 checkpoint
        "sample_returns": returns_list,
    }
    if class_distribution is not None:
        info["class_distribution"] = class_distribution
    return X, y, group_mask, info
