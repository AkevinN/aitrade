"""
标的画像（Symbol Profiling）指标纯函数。

本文件集中实现画像所需的全部指标计算，严格遵循以下约束：

- **纯函数、无 I/O、无副作用**：每个函数仅对内存中的 polars.DataFrame 做只读统计，
  不触碰文件系统、不调用 AlphaLab、不修改入参（Requirement 9）。
- **只读统计，不回填 / 不重采样**：缺口、零成交等以统计形式如实报告，绝不填补或
  改写原始数据（Requirement 3.5）。
- **附带有效样本量**：每个指标返回统一的轻量结构 `MetricResult(value, effective_sample)`，
  其中 `effective_sample` 表示参与该指标计算的有效样本量，供上层 rules.py 据此判定
  置信度（Confidence_Level）与样本不足降级（Requirement 7.1 / 7.2）。

本文件已实现**数据质量（data_quality）** 指标：
`count_valid_bars`、`gap_ratio`、`zero_volume_ratio`、`alignment_coverage`；
以及**流动性（liquidity）** 指标：`avg_turnover`、`intraday_concentration`，
**波动性（volatility）** 指标：`realized_volatility`、`atr_ratio`、`amplitude_quantiles`；
以及**可预测性（predictability）** 指标：`return_autocorr`、`hurst_exponent`、
`variance_ratio`、`adf_pvalue`、`skewness`、`kurtosis`。

可预测性指标的输入约定与前三组不同：它们接受 `numpy.ndarray`（收益序列或价格序列）而非
polars frame，由上层 profiler 先从行情 frame 抽取收益 / 价格序列后传入，便于复用通用数值算法
并做基于属性的测试。这些指标同样附带 `effective_sample`，样本不足 / 退化输入时返回定义良好的
结果（多为 `NaN`），供 rules.py 据此降级（Requirement 6.1 / 7）。

行情 frame 列约定（与 AlphaLab / aitrade.alpha.lab 一致）：
`datetime, open, high, low, close, volume, turnover, open_interest`。
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

# 行情 frame 列名常量，与 AlphaLab（aitrade/alpha/lab.py）保持一致
_DATETIME_COLUMN = "datetime"
_OHLC_COLUMNS = ("open", "high", "low", "close")
_VOLUME_COLUMN = "volume"
# 成交额列（缺失时以 close * volume 作为等效成交额代理）
_TURNOVER_COLUMN = "turnover"

# gap_ratio 中区分"时段内缺口"与"时段边界"（隔夜/午休/周末等）的相对步长上限。
# 这是一个结构性启发参数而非业务阈值：相邻间隔超过中位步长的该倍数时，
# 视为交易时段切换（隔夜/午休/周末）而非时段内缺失 bar，避免这些**周期性**的
# 大间隔被误计为缺口、污染数据质量评估。取 2.0 表示仅把"约一个 bar 的缺失"
# （间隔约为中位步长 2 倍）计为时段内缺口，更大的周期性间隔归为时段边界。
_SESSION_GAP_FACTOR = 2.0


class MetricResult(NamedTuple):
    """统一的轻量指标返回结构。

    所有指标纯函数都返回该结构，使上层能以一致方式读取数值并据有效样本量判定置信度。

    字段：
    - value: 指标数值。多数为标量（float / int）；部分指标按设计返回结构化值：
      `amplitude_quantiles` 返回 dict（分位 -> 振幅值），`intraday_concentration`
      在非分钟级周期下返回 None（表示该指标不适用 / 不可用）。样本不足等情况下的
      "降级抑制"由上层 rules.py 依据 effective_sample 决定，本层只如实返回计算结果。
    - effective_sample: 参与该指标计算的有效样本量（如有效 bar 数 / 目标行数），
      供 confidence 判定使用（Requirement 7.1）。
    """

    value: float | int | dict | None
    effective_sample: int


def _is_minute_interval(interval: str) -> bool:
    """判断是否为分钟级周期（如 1m / 5m / 30m）。

    本地实现一个轻量判定，避免引入 alpha 包的导入链，保持本模块纯净无副作用。
    规范化为小写后，形如 "<数字>m" 即视为分钟级。
    """
    canonical = (interval or "").strip().lower()
    return canonical.endswith("m") and canonical[:-1].isdigit()


def count_valid_bars(df: pl.DataFrame) -> MetricResult:
    """统计窗口内有效 bar 数量（Requirement 3.1）。

    有效 bar 定义为 OHLC 四列均非空且非 NaN 的行（具备可用价格信息）。
    纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param df: 已裁剪到 as_of 的窗口行情 frame
    :return: MetricResult，value=有效 bar 数（int），effective_sample=窗口内总行数
    """
    total = df.height
    if total == 0:
        return MetricResult(value=0, effective_sample=0)

    # 仅保留实际存在的 OHLC 列做有效性判定，缺列时跳过该列约束（防御性处理）
    present_ohlc = [c for c in _OHLC_COLUMNS if c in df.columns]
    if not present_ohlc:
        # 无任何 OHLC 列，无法判定有效性，按 0 个有效 bar 处理
        return MetricResult(value=0, effective_sample=total)

    # 有效 = 每个存在的 OHLC 列都非空且非 NaN
    valid_mask = pl.lit(True)
    for col in present_ohlc:
        col_expr = pl.col(col)
        valid_mask = valid_mask & col_expr.is_not_null() & col_expr.is_not_nan()

    valid_count = int(df.select(valid_mask.sum()).item() or 0)
    return MetricResult(value=valid_count, effective_sample=total)


def gap_ratio(df: pl.DataFrame, interval: str, session_profile: str) -> MetricResult:
    """估计交易时段内的缺口比例：缺失 bar / 期望 bar，返回值 clamp 到 [0, 1]
    （Requirement 3.2）。

    估计方法（用相邻时间戳中位间隔估计缺口）：
    1. 取相邻 bar 时间戳的差值序列，以其**中位数** m 作为"正常步长"。中位数对少量
       隔夜 / 周末等大间隔具有稳健性，故能反映时段内的常规 bar 间隔。
    2. 对每个相邻间隔 d，估计其代表的步数 steps = round(d / m)：
       - steps <= 1：正常相邻，无缺失；
       - steps >= 2 且 d 落在 m 的 `_SESSION_GAP_FACTOR` 倍以内：视为**时段内缺失**，
         缺失 bar 数 += steps - 1；
       - 对分钟级周期，若相邻两 bar 跨自然日，则视为隔夜 / 周末等**时段边界**，按一步计、
         不计入缺失（避免大间隔污染缺口比例）；
       - d 超过 `_SESSION_GAP_FACTOR` 倍中位步长（如午休、半日休市等周期性时段切换），
         同样视为时段边界，不计缺失。
    3. 期望 bar 数 = 实际 bar 数 + 缺失 bar 数；缺口比例 = 缺失 / 期望，clamp 到 [0, 1]。

    限制说明：本估计不依赖完整交易日历，而以"跨日 + 大间隔"剔除隔夜 / 午休 / 周末等
    周期性时段边界，使干净数据的缺口比例接近 0；代价是仅能稳健识别"约一个 bar"的
    时段内缺失，对连续多 bar 的长缺口会保守低估。精确的会话感知缺口需引入 session 日历，
    超出本只读统计的范围。纯函数：仅做只读统计，不回填 / 不重采样（Requirement 3.5）。

    :param df: 已裁剪到 as_of 的窗口行情 frame（含 datetime 列）
    :param interval: 周期标识（如 "30m" / "d"），用于判定是否分钟级以剔除跨日时段边界
    :param session_profile: 交易时段画像标识（如 "cn_equity"），保留以兼容会话感知扩展
    :return: MetricResult，value=缺口比例 [0,1]，effective_sample=参与估计的有效 bar 数
    """
    if _DATETIME_COLUMN not in df.columns:
        return MetricResult(value=0.0, effective_sample=0)

    n = df.height
    # 少于两个 bar 无法估计相邻间隔，缺口比例按 0 处理（充足性由上层置信度判定）
    if n < 2:
        return MetricResult(value=0.0, effective_sample=n)

    # 升序时间戳序列（裁剪阶段通常已排序，这里再保证一次以确保纯函数自洽）
    ts = df.select(pl.col(_DATETIME_COLUMN)).to_series().sort()

    # 相邻间隔（秒），并行计算前后两序列之差
    seconds = ts.dt.epoch(time_unit="s").to_list()
    diffs = [seconds[i] - seconds[i - 1] for i in range(1, len(seconds))]
    # 仅保留正间隔参与中位数估计（重复时间戳的 0 间隔不代表正常步长）
    positive_diffs = [d for d in diffs if d > 0]
    if not positive_diffs:
        return MetricResult(value=0.0, effective_sample=n)

    median_step = _median(positive_diffs)
    if median_step <= 0:
        return MetricResult(value=0.0, effective_sample=n)

    minute_level = _is_minute_interval(interval)
    dates = ts.dt.date().to_list()

    missing = 0
    for i in range(1, len(seconds)):
        d = seconds[i] - seconds[i - 1]
        if d <= 0:
            continue
        steps = round(d / median_step)
        if steps <= 1:
            continue
        # 分钟级周期下跨自然日视为时段边界（隔夜 / 周末），不计入缺失
        if minute_level and dates[i] != dates[i - 1]:
            continue
        # 间隔过大视为时段边界，不计入缺失
        if d > _SESSION_GAP_FACTOR * median_step:
            continue
        missing += steps - 1

    expected = n + missing
    ratio = missing / expected if expected > 0 else 0.0
    return MetricResult(value=_clamp01(ratio), effective_sample=n)


def zero_volume_ratio(df: pl.DataFrame) -> MetricResult:
    """统计零成交 bar 占比，返回值落在 [0, 1]（Requirement 3.2 / 4.1）。

    零成交 bar 定义为 volume 为 0 或缺失（null / NaN）的行。
    纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param df: 已裁剪到 as_of 的窗口行情 frame
    :return: MetricResult，value=零成交占比 [0,1]，effective_sample=窗口内总行数
    """
    total = df.height
    if total == 0:
        return MetricResult(value=0.0, effective_sample=0)

    # 缺少 volume 列时，无法判定成交，保守视为全部零成交
    if _VOLUME_COLUMN not in df.columns:
        return MetricResult(value=1.0, effective_sample=total)

    vol = pl.col(_VOLUME_COLUMN)
    # 零成交：null / NaN / 等于 0
    zero_mask = vol.is_null() | vol.is_nan() | (vol == 0)
    zero_count = int(df.select(zero_mask.sum()).item() or 0)

    ratio = zero_count / total
    return MetricResult(value=_clamp01(ratio), effective_sample=total)


def alignment_coverage(target: pl.DataFrame, others: list[pl.DataFrame]) -> MetricResult:
    """计算目标与各观测标的按公共时间轴对齐后的覆盖率，返回 [0, 1]
    （Requirement 3.3 / 13.1）。

    覆盖率 = 对齐后保留行数 / 目标标的行数。其中"对齐后保留"指目标的某个时间戳
    同时存在于**所有**观测标的中（公共时间轴交集）。

    边界与单调性（与 Property 4 一致）：
    - 目标为空：返回 0.0（无可对齐基准）。
    - 观测列表为空：所有时间戳"被全部观测覆盖"为真空真，返回 1.0。
    - 所有观测覆盖目标全部时间戳：返回 1.0。
    - 向观测集合追加任一 frame：覆盖率单调非增（交集只会收缩）。

    纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param target: 目标标的窗口 frame（含 datetime 列）
    :param others: 观测标的窗口 frame 列表（各含 datetime 列）
    :return: MetricResult，value=覆盖率 [0,1]，effective_sample=目标行数
    """
    if _DATETIME_COLUMN not in target.columns:
        return MetricResult(value=0.0, effective_sample=0)

    target_n = target.height
    # 目标为空：无可对齐基准，按约定返回 0.0
    if target_n == 0:
        return MetricResult(value=0.0, effective_sample=0)

    target_dts = target.select(pl.col(_DATETIME_COLUMN)).to_series().to_list()

    # 公共时间轴 = 目标时间戳 ∩ 每个观测标的的时间戳集合
    common: set = set(target_dts)
    for other in others:
        if _DATETIME_COLUMN not in other.columns or other.height == 0:
            # 任一观测无可对齐时间戳，则公共交集为空，覆盖率为 0
            common = set()
            break
        other_dts = set(other.select(pl.col(_DATETIME_COLUMN)).to_series().to_list())
        common &= other_dts

    # 按"保留行数"统计：目标中 datetime 落在公共时间轴的行数（兼容潜在重复时间戳）
    if common:
        retained = sum(1 for dt in target_dts if dt in common)
    else:
        retained = 0

    coverage = retained / target_n
    return MetricResult(value=_clamp01(coverage), effective_sample=target_n)


# ============================ 流动性指标 ============================


def avg_turnover(df: pl.DataFrame) -> MetricResult:
    """计算窗口内日均成交额（或等效成交额代理）（Requirement 4.1）。

    成交额取值优先级：
    1. 存在 `turnover` 列：直接使用其值；
    2. 否则若存在 `close` 与 `volume` 列：用 `close * volume` 作为**等效成交额代理**
       （注释说明：分钟 / 日线行情若缺 turnover，可用价格×成交量近似当期成交额）。

    "日均"口径：按自然日（datetime 的日期部分）汇总每日成交额，再对参与的天数取平均，
    使不同 interval（分钟 / 日线）下口径一致，避免单纯按 bar 平均时分钟级被稀释。
    纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param df: 已裁剪到 as_of 的窗口行情 frame（需含 datetime 列与成交额来源列）
    :return: MetricResult，value=日均成交额（float，>=0），effective_sample=参与计算的有效 bar 数
    """
    if df.height == 0 or _DATETIME_COLUMN not in df.columns:
        return MetricResult(value=0.0, effective_sample=0)

    # 选取成交额表达式：优先 turnover，缺失时退化为 close*volume 代理
    if _TURNOVER_COLUMN in df.columns:
        turnover_expr = pl.col(_TURNOVER_COLUMN)
    elif "close" in df.columns and _VOLUME_COLUMN in df.columns:
        turnover_expr = pl.col("close") * pl.col(_VOLUME_COLUMN)
    else:
        # 无任何可用成交额来源，无法计算
        return MetricResult(value=0.0, effective_sample=0)

    work = df.select(
        [
            pl.col(_DATETIME_COLUMN).dt.date().alias("_date"),
            turnover_expr.cast(pl.Float64).alias("_turnover"),
        ]
    )
    # 仅保留非空、有限、非负的成交额（成交额本应非负）
    work = work.filter(
        pl.col("_turnover").is_not_null()
        & pl.col("_turnover").is_not_nan()
        & pl.col("_turnover").is_finite()
        & (pl.col("_turnover") >= 0)
    )

    valid_bars = work.height
    if valid_bars == 0:
        return MetricResult(value=0.0, effective_sample=0)

    # 按自然日汇总每日成交额，再对天数取平均得日均成交额
    daily = work.group_by("_date").agg(pl.col("_turnover").sum().alias("_daily"))
    num_days = daily.height
    total = float(daily.select(pl.col("_daily").sum()).item() or 0.0)
    avg = total / num_days if num_days > 0 else 0.0
    # 防御性 max(0,)：成交额日均恒非负
    return MetricResult(value=max(0.0, avg), effective_sample=valid_bars)


def intraday_concentration(df: pl.DataFrame, interval: str) -> MetricResult:
    """计算分钟级行情的日内成交分布集中度代理（开收盘集中程度）（Requirement 4.2）。

    仅对**分钟级** interval 有意义（用 `_is_minute_interval` 判定）；非分钟级（如日线）
    没有日内分布概念，返回 `value=None` 表示该指标不适用。

    集中度代理口径：对每个自然日，取该日**开盘 bar** 与**收盘 bar**（按时间排序后的
    首尾 bar）的成交量之和占当日总成交量的比例，再对各日取平均。值越接近 1 表示成交越
    集中于开收盘，越接近 0 表示日内分布越均匀。结果 clamp 到 [0, 1]。
    纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param df: 已裁剪到 as_of 的窗口行情 frame（需含 datetime 与 volume 列）
    :param interval: 周期标识（如 "1m" / "30m" / "d"），用于判定是否分钟级
    :return: MetricResult；分钟级时 value=集中度 [0,1]，非分钟级或不可用时 value=None；
        effective_sample=参与计算的有效 bar 数
    """
    n = df.height
    # 非分钟级周期无日内分布概念，按约定返回 value=None
    if not _is_minute_interval(interval):
        return MetricResult(value=None, effective_sample=n)

    if _DATETIME_COLUMN not in df.columns or _VOLUME_COLUMN not in df.columns:
        return MetricResult(value=None, effective_sample=0)

    work = df.select(
        [
            pl.col(_DATETIME_COLUMN).alias("_dt"),
            pl.col(_VOLUME_COLUMN).cast(pl.Float64).alias("_vol"),
        ]
    ).filter(
        pl.col("_vol").is_not_null()
        & pl.col("_vol").is_not_nan()
        & pl.col("_vol").is_finite()
        & (pl.col("_vol") >= 0)
    )
    if work.height == 0:
        return MetricResult(value=None, effective_sample=0)

    # 按时间升序后分日聚合，保证 first()/last() 对应开盘 / 收盘 bar
    work = work.sort("_dt").with_columns(pl.col("_dt").dt.date().alias("_date"))
    daily = work.group_by("_date", maintain_order=True).agg(
        [
            pl.col("_vol").sum().alias("_total"),
            pl.col("_vol").first().alias("_open"),
            pl.col("_vol").last().alias("_close"),
        ]
    )
    # 仅对当日有正成交量的日子计算占比，避免除零
    daily = daily.filter(pl.col("_total") > 0)
    if daily.height == 0:
        return MetricResult(value=None, effective_sample=0)

    # 开收盘集中度 = (开盘 bar 成交量 + 收盘 bar 成交量) / 当日总成交量；按日取均值
    daily = daily.with_columns(
        ((pl.col("_open") + pl.col("_close")) / pl.col("_total")).alias("_conc")
    )
    conc = float(daily.select(pl.col("_conc").mean()).item() or 0.0)
    return MetricResult(value=_clamp01(conc), effective_sample=work.height)


# ============================ 波动性指标 ============================


def realized_volatility(df: pl.DataFrame) -> MetricResult:
    """计算窗口内已实现波动率，返回值非负（Requirement 5.1）。

    口径：基于收盘价的相邻对数收益 r_t = ln(close_t / close_{t-1})，已实现波动率定义为
    各期对数收益平方和的平方根 RV = sqrt(Σ r_t^2)。该定义由构造保证非负。
    遇到缺失 / 非正收盘价时，在该处中断收益序列（不跨越无效价格计算收益），
    其余相邻有效价格段照常贡献收益。纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param df: 已裁剪到 as_of 的窗口行情 frame（需含 close 列）
    :return: MetricResult，value=已实现波动率（float，>=0），effective_sample=参与计算的收益个数
    """
    if "close" not in df.columns or df.height == 0:
        return MetricResult(value=0.0, effective_sample=0)

    closes = df.select(pl.col("close")).to_series().to_list()
    returns: list[float] = []
    prev: float | None = None
    for c in closes:
        # 无效价格（空 / NaN / 非正）处中断收益序列
        if c is None or (isinstance(c, float) and math.isnan(c)) or c <= 0:
            prev = None
            continue
        if prev is not None:
            returns.append(math.log(c / prev))
        prev = c

    if not returns:
        return MetricResult(value=0.0, effective_sample=0)

    rv = math.sqrt(sum(r * r for r in returns))
    # 构造上非负，防御性 max(0,)
    return MetricResult(value=max(0.0, rv), effective_sample=len(returns))


def atr_ratio(df: pl.DataFrame, window: int) -> MetricResult:
    """计算 ATR（平均真实波幅）相对价格的比率，返回值非负（Requirement 5.1）。

    真实波幅 TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)，
    首个有效 bar 无前收盘价时取 TR = high - low。ATR 取最近 `window` 个 TR 的简单平均
    （窗口非正或样本不足时退化为对全部可用 TR 取平均）。比率 = ATR / 参考价格，
    参考价格取同窗口收盘价均值（正数）。比率由构造非负。
    纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param df: 已裁剪到 as_of 的窗口行情 frame（需含 high / low / close 列）
    :param window: ATR 平滑窗口（bar 数）；非正时对全部可用 TR 取平均
    :return: MetricResult，value=ATR/价格 比率（float，>=0），effective_sample=参与计算的 TR 个数
    """
    if any(col not in df.columns for col in ("high", "low", "close")) or df.height == 0:
        return MetricResult(value=0.0, effective_sample=0)

    highs = df.select(pl.col("high")).to_series().to_list()
    lows = df.select(pl.col("low")).to_series().to_list()
    closes = df.select(pl.col("close")).to_series().to_list()

    def _bad(x: float | None) -> bool:
        return x is None or (isinstance(x, float) and math.isnan(x))

    true_ranges: list[float] = []
    ref_closes: list[float] = []
    prev_close: float | None = None
    for h, l, c in zip(highs, lows, closes):
        if _bad(h) or _bad(l) or _bad(c):
            # 缺失行不贡献 TR；若收盘价有效则更新前收盘以供下一根使用
            if not _bad(c):
                prev_close = c
            continue
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        if tr >= 0:
            true_ranges.append(tr)
            ref_closes.append(c)
        prev_close = c

    if not true_ranges:
        return MetricResult(value=0.0, effective_sample=0)

    w = window if (isinstance(window, int) and window > 0) else len(true_ranges)
    recent_tr = true_ranges[-w:]
    recent_close = ref_closes[-w:]

    atr = sum(recent_tr) / len(recent_tr)
    ref_price = sum(recent_close) / len(recent_close)
    if ref_price <= 0:
        return MetricResult(value=0.0, effective_sample=len(recent_tr))

    ratio = atr / ref_price
    # 构造上非负，防御性 max(0,)
    return MetricResult(value=max(0.0, ratio), effective_sample=len(recent_tr))


def amplitude_quantiles(df: pl.DataFrame, qs: list[float]) -> MetricResult:
    """计算周期振幅分布的分位数，value 为 {分位: 振幅值} 的 dict（Requirement 5.1）。

    周期振幅定义为相对振幅 amp_t = (high_t - low_t) / close_t（相对化便于跨标的 / 跨价位
    比较），由构造非负。对每个请求分位 q（取值落在 [0, 1]）计算振幅分布的分位数
    （线性插值）。无有效振幅时返回空 dict。
    纯函数：仅做只读统计，不修改入参、不做 I/O。

    :param df: 已裁剪到 as_of 的窗口行情 frame（需含 high / low / close 列）
    :param qs: 请求的分位列表（如 [0.5, 0.9]），区间外的值被忽略
    :return: MetricResult，value=dict[分位 -> 振幅值(>=0)]，effective_sample=参与计算的有效振幅个数
    """
    if any(col not in df.columns for col in ("high", "low", "close")) or df.height == 0:
        return MetricResult(value={}, effective_sample=0)

    work = df.select(
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("_amp")
    ).filter(
        pl.col("_amp").is_not_null()
        & pl.col("_amp").is_not_nan()
        & pl.col("_amp").is_finite()
    )

    amp_series = work.select(pl.col("_amp")).to_series()
    effective = amp_series.len()
    if effective == 0:
        return MetricResult(value={}, effective_sample=0)

    result: dict[float, float] = {}
    for q in qs:
        # 忽略非法分位（None 或落在 [0,1] 之外）
        if q is None or q < 0.0 or q > 1.0:
            continue
        qv = amp_series.quantile(q, interpolation="linear")
        # 振幅由构造非负，防御性 max(0,)
        result[q] = max(0.0, float(qv)) if qv is not None else 0.0

    return MetricResult(value=result, effective_sample=effective)


# ============================ 可预测性指标 ============================
#
# 以下指标刻画收益 / 价格序列的可预测性结构（趋势 vs 均值回复、厚尾等），均为纯函数，
# 输入为 numpy.ndarray（收益序列或价格序列），输出 MetricResult。
# 依赖说明：numpy 与 scipy 均为项目既有依赖（见 pyproject.toml）；statsmodels 未安装，
# 因此 adf_pvalue 自行实现一个轻量的 Dickey-Fuller 近似（见该函数注释），不引入重依赖。


# 单项可预测性指标所需的最小样本量（低于此值视为退化，返回 NaN）。
# 这是为避免在极小样本上输出误导性数值的结构性下限，真正的置信度分档由 rules.py 负责。
_MIN_AUTOCORR_SAMPLE = 3      # 至少需 lag+2 个点，逐 lag 再细判
_MIN_HURST_SAMPLE = 8         # 标度回归至少需若干个 lag 点
_MIN_VR_SAMPLE = 4            # 需 T > k 且有多个重叠窗口
_MIN_ADF_SAMPLE = 4           # 需可解的回归（含常数项 + 滞后水平）
_MIN_SKEW_SAMPLE = 3          # 偏度至少需 3 个点
_MIN_KURT_SAMPLE = 4          # 峰度至少需 4 个点


def _clean_1d(arr: np.ndarray) -> np.ndarray:
    """将输入转为一维 float64 数组并剔除非有限值（NaN / ±Inf）。

    纯函数：返回新数组，不修改入参。用于让各可预测性指标在脏输入下行为定义良好。
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def return_autocorr(returns: np.ndarray, lags: list[int]) -> MetricResult:
    """计算收益序列在指定滞后阶上的自相关概要（Requirement 6.1）。

    对每个滞后阶 lag，自相关采用标准样本估计：
        acf(lag) = Σ_{t}(r_t - μ)(r_{t-lag} - μ) / Σ_t (r_t - μ)^2
    分母为收益的总离差平方和（以全样本均值 μ 为中心）。该估计落在 [-1, 1]。

    退化处理：
    - 有效样本（剔除非有限值后）少于 `_MIN_AUTOCORR_SAMPLE`、或收益方差为 0（常数序列）时，
      所有 lag 的自相关返回 `NaN`（自相关无定义，避免误导）。
    - 单个 lag 满足 lag <= 0 或 n - lag < 2（无足够配对）时，该 lag 返回 `NaN`。

    纯函数：无 I/O、不修改入参。

    :param returns: 收益序列（一维 np.ndarray）
    :param lags: 滞后阶列表（如 [1, 5, 10]）
    :return: MetricResult，value=dict[lag -> autocorr]（含退化时的 NaN），
        effective_sample=有效收益个数
    """
    r = _clean_1d(returns)
    n = r.size
    result: dict[int, float] = {}

    if n < _MIN_AUTOCORR_SAMPLE:
        # 样本过少：所有请求 lag 标记为 NaN
        for lag in lags:
            result[int(lag)] = float("nan")
        return MetricResult(value=result, effective_sample=n)

    mu = float(r.mean())
    denom = float(np.sum((r - mu) ** 2))
    if denom <= 0.0:
        # 常数序列：自相关无定义
        for lag in lags:
            result[int(lag)] = float("nan")
        return MetricResult(value=result, effective_sample=n)

    for lag in lags:
        k = int(lag)
        if k <= 0 or n - k < 2:
            result[k] = float("nan")
            continue
        num = float(np.sum((r[k:] - mu) * (r[:-k] - mu)))
        result[k] = num / denom

    return MetricResult(value=result, effective_sample=n)


def hurst_exponent(returns: np.ndarray) -> MetricResult:
    """估计收益序列隐含价格路径的 Hurst 指数（Requirement 6.1）。

    方法（标度 / 结构函数法）：先由收益累加重建隐含价格路径 `p = cumsum(returns)`，
    再对一组滞后尺度 lag 计算价格增量的标准差 τ(lag) = std(p[lag:] - p[:-lag])。
    在 τ(lag) ∝ lag^H 的标度关系下，对 (log lag, log τ) 做一元线性回归，斜率即为 H。

    方向性（与需求一致）：
    - 收益近似独立同分布（随机游走价格）时，τ ∝ sqrt(lag)，H ≈ 0.5；
    - 收益正持续（趋势性）时，价格增量随尺度增长更快，H > 0.5；
    - 收益均值回复时，H < 0.5。
    注意：纯漂移（常数收益 + 噪声）的漂移项在做差时被抵消，不会抬高 H —— H 反映的是
    增量的持续性而非漂移，这与 Hurst 的统计含义一致。

    退化处理：有效样本少于 `_MIN_HURST_SAMPLE`、可用尺度点不足 2、或所有 τ 退化为 0
    （如严格常数序列）时返回 `NaN`。结果 clamp 到 [0, 1]（Hurst 的理论取值范围）。
    纯函数：无 I/O、不修改入参。

    :param returns: 收益序列（一维 np.ndarray）
    :return: MetricResult，value=Hurst 指数 [0,1] 或 NaN，effective_sample=有效收益个数
    """
    r = _clean_1d(returns)
    n = r.size
    if n < _MIN_HURST_SAMPLE:
        return MetricResult(value=float("nan"), effective_sample=n)

    # 重建隐含价格路径（对数价位的累积，常数项不影响差分）
    path = np.cumsum(r)

    # 滞后尺度从 2 取到 n//2，限制上限以保证每个尺度仍有足够配对样本
    max_lag = max(2, min(n // 2, 20))
    lags = list(range(2, max_lag + 1))
    if len(lags) < 2:
        return MetricResult(value=float("nan"), effective_sample=n)

    log_lags: list[float] = []
    log_tau: list[float] = []
    for lag in lags:
        diff = path[lag:] - path[:-lag]
        tau = float(np.std(diff))
        if tau > 0.0:
            log_lags.append(math.log(lag))
            log_tau.append(math.log(tau))

    # 至少需要两个有效尺度点才能拟合斜率
    if len(log_lags) < 2:
        return MetricResult(value=float("nan"), effective_sample=n)

    slope = float(np.polyfit(np.asarray(log_lags), np.asarray(log_tau), 1)[0])
    return MetricResult(value=_clamp01(slope), effective_sample=n)


def variance_ratio(returns: np.ndarray, k: int) -> MetricResult:
    """计算 k 阶方差比 VR(k)（趋势 / 均值回复判据，Requirement 6.1）。

    采用 Lo-MacKinlay 的重叠样本无偏估计：
        VR(k) = Var_k / Var_1
        Var_1 = Σ_t (r_t - μ)^2 / (T - 1)
        Var_k = Σ_{t=k}^{T} (P_t - P_{t-k} - kμ)^2 / m,  m = k(T-k+1)(1 - k/T)
    其中 P 为收益累积路径、μ 为单期收益均值、T 为收益个数。

    含义：
    - 随机游走：VR ≈ 1；
    - 趋势 / 正持续：VR > 1；
    - 均值回复：VR < 1。

    退化处理：k 非正、有效样本少于 `_MIN_VR_SAMPLE`、T <= k、归一化因子 m <= 0、或
    单期方差为 0（常数序列，比值无定义）时返回 `NaN`。结果非负（方差比恒 >= 0）。
    纯函数：无 I/O、不修改入参。

    :param returns: 收益序列（一维 np.ndarray）
    :param k: 聚合阶数（>= 2 才有意义）
    :return: MetricResult，value=方差比（>=0）或 NaN，effective_sample=有效收益个数
    """
    r = _clean_1d(returns)
    T = r.size
    if not isinstance(k, int) or k <= 0 or T < _MIN_VR_SAMPLE or T <= k:
        return MetricResult(value=float("nan"), effective_sample=T)

    mu = float(r.mean())
    var_1 = float(np.sum((r - mu) ** 2)) / (T - 1)
    if var_1 <= 0.0:
        # 常数序列：方差比无定义
        return MetricResult(value=float("nan"), effective_sample=T)

    # 累积路径，前置 0 使 P_t - P_{t-k} 表示第 t 个 k 期收益（t = k..T）
    path = np.concatenate(([0.0], np.cumsum(r)))
    diffs = path[k:] - path[:-k] - k * mu  # 长度 T - k + 1
    m = k * (T - k + 1) * (1.0 - k / T)
    if m <= 0.0:
        return MetricResult(value=float("nan"), effective_sample=T)

    var_k = float(np.sum(diffs ** 2)) / m
    vr = var_k / var_1
    # 方差比由构造非负，防御性 max(0,)
    return MetricResult(value=max(0.0, vr), effective_sample=T)


# Dickey-Fuller（含常数项）检验统计量到 p 值的近似插值表。
# 说明：项目未安装 statsmodels，这里以一维线性插值在 DF 统计量与 p 值之间建立单调映射，
# 作为轻量 ADF 近似（lag=0 的 DF 检验）。表中较负的 t 统计量对应更小的 p 值（更强地拒绝
# "存在单位根 / 非平稳"原假设）。左端临界值（1%/5%/10%）取自 Fuller 的大样本 DF 分布
# （含常数、无趋势），中间与右端为保证单调、连续而设的近似锚点。这是有意为之的近似，
# 不追求与 statsmodels 完全一致，仅用于给出平稳性的相对强弱信号。
_DF_TAU_GRID = (-4.5, -3.43, -3.12, -2.86, -2.57, -2.26, -1.95, -1.62, -1.0, 0.0, 1.0, 2.0)
_DF_PVALUE_GRID = (0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 0.97, 0.99)


def _df_tstat_to_pvalue(tstat: float) -> float:
    """把 Dickey-Fuller t 统计量近似映射为 p 值（单调一维插值，clamp 到 [0,1]）。"""
    if math.isnan(tstat):
        return float("nan")
    # numpy.interp 在区间外自动取端点值，天然完成 clamp
    return float(np.interp(tstat, _DF_TAU_GRID, _DF_PVALUE_GRID))


def adf_pvalue(prices: np.ndarray) -> MetricResult:
    """ADF 平稳性检验 p 值的轻量近似（Requirement 6.1）。

    实现策略（无 statsmodels 依赖）：执行含常数项的 Dickey-Fuller 回归
        Δy_t = α + β · y_{t-1} + ε_t
    用最小二乘估计 β 并计算其 t 统计量 t = β / se(β)，再经 `_df_tstat_to_pvalue`
    近似映射为 p 值。p 值越小，越倾向于拒绝"存在单位根"的原假设，即序列越平稳
    （越偏均值回复）；p 值大则更接近随机游走 / 非平稳。

    退化处理：有效样本少于 `_MIN_ADF_SAMPLE`、回归设计矩阵奇异（如常数价格序列）、
    或残差自由度不足 / 方差为 0 时返回 `NaN`。p 值落在 [0, 1]。
    纯函数：无 I/O、不修改入参。

    :param prices: 价格序列（一维 np.ndarray）
    :return: MetricResult，value=ADF 近似 p 值 [0,1] 或 NaN，effective_sample=参与回归的样本数
    """
    y = _clean_1d(prices)
    n = y.size
    if n < _MIN_ADF_SAMPLE:
        return MetricResult(value=float("nan"), effective_sample=n)

    # 构造 Δy_t 与 y_{t-1}
    dy = np.diff(y)                 # 长度 n-1
    y_lag = y[:-1]                  # 长度 n-1
    obs = dy.size                   # 回归样本数
    # 设计矩阵 [常数, y_{t-1}]，参数 k=2，需 obs > k 才有正自由度
    k_params = 2
    if obs <= k_params:
        return MetricResult(value=float("nan"), effective_sample=obs)

    X = np.column_stack((np.ones(obs), y_lag))
    # 价格恒定 -> y_lag 为常数列 -> X 退化，直接判退化
    if float(np.std(y_lag)) == 0.0:
        return MetricResult(value=float("nan"), effective_sample=obs)

    try:
        # 最小二乘解 β = (X'X)^{-1} X'dy
        xtx = X.T @ X
        xtx_inv = np.linalg.inv(xtx)
        beta = xtx_inv @ (X.T @ dy)
    except np.linalg.LinAlgError:
        return MetricResult(value=float("nan"), effective_sample=obs)

    resid = dy - X @ beta
    dof = obs - k_params
    sigma2 = float(resid @ resid) / dof
    if sigma2 <= 0.0:
        # 完美拟合 / 零残差：t 统计量无定义
        return MetricResult(value=float("nan"), effective_sample=obs)

    # β（y_{t-1} 的系数，索引 1）的标准误
    var_beta = sigma2 * float(xtx_inv[1, 1])
    if var_beta <= 0.0:
        return MetricResult(value=float("nan"), effective_sample=obs)

    tstat = float(beta[1]) / math.sqrt(var_beta)
    pvalue = _df_tstat_to_pvalue(tstat)
    return MetricResult(value=pvalue, effective_sample=obs)


def skewness(returns: np.ndarray) -> MetricResult:
    """计算收益分布的偏度（Requirement 6.1）。

    使用 scipy.stats.skew（`bias=True`，即标准样本偏度）。偏度刻画分布的不对称性：
    正偏表示右尾更长，负偏表示左尾更长，对称分布约为 0。

    退化处理：有效样本少于 `_MIN_SKEW_SAMPLE` 或收益方差为 0（常数序列，偏度无定义）时
    返回 `NaN`，避免输出误导性数值。纯函数：无 I/O、不修改入参。

    :param returns: 收益序列（一维 np.ndarray）
    :return: MetricResult，value=偏度或 NaN，effective_sample=有效收益个数
    """
    r = _clean_1d(returns)
    n = r.size
    if n < _MIN_SKEW_SAMPLE or float(np.var(r)) == 0.0:
        return MetricResult(value=float("nan"), effective_sample=n)
    value = float(stats.skew(r, bias=True))
    return MetricResult(value=value, effective_sample=n)


def kurtosis(returns: np.ndarray) -> MetricResult:
    """计算收益分布的峰度（超额峰度，Requirement 6.1）。

    使用 scipy.stats.kurtosis（`fisher=True`，即超额峰度：正态分布为 0；`bias=True`）。
    峰度刻画尾部厚薄：正值表示比正态更厚尾（leptokurtic），负值表示更瘦尾。

    退化处理：有效样本少于 `_MIN_KURT_SAMPLE` 或收益方差为 0（常数序列，峰度无定义）时
    返回 `NaN`。纯函数：无 I/O、不修改入参。

    :param returns: 收益序列（一维 np.ndarray）
    :return: MetricResult，value=超额峰度或 NaN，effective_sample=有效收益个数
    """
    r = _clean_1d(returns)
    n = r.size
    if n < _MIN_KURT_SAMPLE or float(np.var(r)) == 0.0:
        return MetricResult(value=float("nan"), effective_sample=n)
    value = float(stats.kurtosis(r, fisher=True, bias=True))
    return MetricResult(value=value, effective_sample=n)


def _median(values: list[float]) -> float:
    """计算列表中位数（仅供本模块内部使用的纯函数辅助）。"""
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _clamp01(value: float) -> float:
    """将数值 clamp 到闭区间 [0, 1]；对 NaN 返回 0.0（防御性处理）。"""
    if math.isnan(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
