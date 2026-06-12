"""Symbol Profiling 协调器：加载本地行情、计算画像指标、可选生成建议与持久化。

主入口为 Profiler.profile()，输出 SymbolProfile 对象（见 types.py）。
本模块依赖 AlphaLab 的只读读取接口（不写入 lab 数据）；
唯一允许的写入操作是通过 ProfileStore 持久化画像产物（需 persist=True）。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from aitrade.alpha.lab_utils import normalize_vt_symbol
from aitrade.profiling.loader import _load_window_frame, effective_right_bound, load_local_range
from aitrade.profiling.metrics import (
    MetricResult,
    adf_pvalue,
    alignment_coverage,
    amplitude_quantiles,
    atr_ratio,
    avg_turnover,
    count_valid_bars,
    gap_ratio,
    hurst_exponent,
    intraday_concentration,
    kurtosis,
    realized_volatility,
    return_autocorr,
    skewness,
    variance_ratio,
    zero_volume_ratio,
)
from aitrade.profiling.recommender import build_scheme_suggestion
from aitrade.profiling.rules import (
    DEFAULT_RULES,
    ProfilingRules,
    confidence_for,
    liquidity_level,
    overall_confidence,
    structure_judgement,
    volatility_level,
)
from aitrade.profiling.store import ProfileStore
from aitrade.profiling.types import (
    GroupProfile,
    MetricBlock,
    MetricValue,
    ProfileInput,
    SymbolProfile,
)


class Profiler:
    """标的画像协调器：加载本地行情、驱动指标计算、组装 SymbolProfile。

    Profiler 封装了从数据加载到画像结果的完整流程，对外只暴露 profile() 方法。
    内部严格遵守：只读数据（通过 _load_window_frame）、时间窗口隔离（as_of 右边界）、
    无 AlphaLab 写入（写入只发生在 ProfileStore.save()，由 persist=True 控制）。
    """

    def __init__(
        self,
        lab,
        *,
        rules: ProfilingRules | None = None,
        store: ProfileStore | None = None,
    ) -> None:
        """初始化 Profiler，绑定 AlphaLab 实例与规则/存储配置。

        Args:
            lab: AlphaLab 实例，用于只读读取本地 K 线数据。
            rules: 画像规则配置（阈值/分档）；None 时使用 DEFAULT_RULES（builtin-v1）。
            store: 画像产物存储；None 时使用默认 ProfileStore（写入 PROFILE_PATH）。
        """
        self.lab = lab
        self.rules = rules or DEFAULT_RULES
        self.store = store or ProfileStore()

    def profile(
        self,
        *,
        vt_symbol: str,
        interval: str,
        as_of: datetime,
        lookback_days: int,
        observation_symbols: list[str] | None = None,
        with_suggestion: bool = True,
        persist: bool = False,
    ) -> SymbolProfile:
        """计算标的画像，返回 SymbolProfile 对象。

        计算流程：
        1. 加载 [as_of - lookback_days, as_of] 窗口行情并物理裁剪；
        2. 若无可用数据，返回 available=False 的结构化结果；
        3. 计算四个 MetricBlock（数据质量/流动性/波动性/可预测性）；
        4. 可选计算多标的关联性（observation_symbols）；
        5. 可选生成 SchemeSuggestion 草稿（with_suggestion=True）；
        6. 可选持久化到 PROFILE_PATH（persist=True）。

        Args:
            vt_symbol: 目标标的合约代码，如 ``"000001.SZSE"``。
            interval: K 线周期，如 ``"d"``/``"30m"``。
            as_of: 截止时间（含义为"站在该时刻回看"），必须显式传入，无默认全量。
            lookback_days: 回看日历天数，如 365（日线约 250 bar）。
            observation_symbols: 用于多标的关联性画像的观测标的列表；None 不计算。
            with_suggestion: True 时生成 SchemeSuggestion 草稿（只建议，不写方案）。
            persist: True 时将画像产物 JSON 写入 PROFILE_PATH（唯一写入操作）。

        Returns:
            SymbolProfile 对象；数据不可用时 available=False，blocks=[]。
        """
        if as_of.tzinfo is not None:
            as_of = as_of.replace(tzinfo=None)
        normalized_symbol = normalize_vt_symbol(vt_symbol)

        df = _load_window_frame(self.lab, normalized_symbol, interval, as_of, lookback_days)
        input_meta = ProfileInput(
            vt_symbol=normalized_symbol,
            interval=interval,
            as_of=as_of,
            lookback_days=lookback_days,
            rules_id=self.rules.rules_id,
        )

        if df is None or df.is_empty():
            start, end = load_local_range(self.lab, normalized_symbol, interval)
            reason = "本地无可用行情"
            if start is not None and end is not None:
                reason = f"窗口内无可用行情；本地数据区间为 {start} 至 {end}"
            profile = SymbolProfile(
                input=input_meta,
                available=False,
                unavailable_reason=reason,
                blocks=[],
                overall_confidence="insufficient",
            )
            self._try_persist(profile, persist)
            return profile

        input_meta.effective_right_bound = effective_right_bound(df, as_of)
        input_meta.effective_bar_count = df.height

        blocks = self._build_blocks(df, interval)
        confidence_values = [
            metric.confidence
            for block in blocks
            for metric in block.metrics
            if metric.note != "not_applicable"
        ]
        overall = overall_confidence(confidence_values)
        group = self._build_group_profile(
            normalized_symbol,
            df,
            interval,
            as_of,
            lookback_days,
            observation_symbols or [],
        )
        suggestion = None
        if with_suggestion:
            suggestion = build_scheme_suggestion(
                vt_symbol=normalized_symbol,
                interval=interval,
                blocks=blocks,
                overall_confidence=overall,
                rules=self.rules,
            )

        profile = SymbolProfile(
            input=input_meta,
            available=True,
            blocks=blocks,
            group_profile=group,
            suggestion=suggestion,
            overall_confidence=overall,
        )
        self._try_persist(profile, persist)
        return profile

    def _try_persist(self, profile: SymbolProfile, persist: bool) -> None:
        """尝试将画像产物写入 ProfileStore，忽略任何写入异常（不影响主流程）。

        Args:
            profile: 待持久化的 SymbolProfile 对象。
            persist: False 时直接返回，不写入。
        """
        if not persist:
            return
        try:
            self.store.save(profile)
        except Exception:
            pass

    def _metric(self, key: str, result: MetricResult, *, note: str | None = None) -> MetricValue:
        """将 MetricResult 封装为 MetricValue，应用置信度分档并处理 insufficient 抑制。

        样本不足（insufficient）时将 value 置 None（避免输出误导性数值，Requirement 7.2）；
        value 为 None 且 note 也为 None 时补 ``"not_applicable"`` 标记。

        Args:
            key: 指标键名（如 ``"gap_ratio"``），用于查询 rules.confidence 分档。
            result: 指标计算纯函数返回的 MetricResult。
            note: 额外说明；insufficient 时会被覆盖为 ``"insufficient_sample"``。

        Returns:
            填充了置信度与 note 的 MetricValue 对象。
        """
        confidence = confidence_for(key, result.effective_sample, self.rules)
        value = self._json_safe(result.value)
        if confidence == "insufficient":
            value = None
            note = note or "insufficient_sample"
        if value is None and note is None:
            note = "not_applicable"
        return MetricValue(
            key=key,
            value=value,
            effective_sample=result.effective_sample,
            confidence=confidence,
            note=note,
        )

    def _build_blocks(self, df: pl.DataFrame, interval: str) -> list[MetricBlock]:
        """计算四个指标块（数据质量/流动性/波动性/可预测性）并返回列表。

        Args:
            df: 已裁剪到 as_of 的窗口行情 frame。
            interval: K 线周期（影响分钟指标与缺口估计）。

        Returns:
            [data_quality, liquidity, volatility, predictability] 四个 MetricBlock。
        """
        data_quality = MetricBlock(
            block="data_quality",
            metrics=[
                self._metric("count_valid_bars", count_valid_bars(df)),
                self._metric("gap_ratio", gap_ratio(df, interval, "cn_equity")),
                self._metric("zero_volume_ratio", zero_volume_ratio(df)),
            ],
        )

        turnover = avg_turnover(df)
        liq = liquidity_level(float(turnover.value or 0.0), self.rules)
        concentration = intraday_concentration(df, interval)
        liq_metrics = [
            self._metric("avg_turnover", turnover),
            self._metric(
                "intraday_concentration",
                concentration,
                note="not_applicable" if concentration.value is None else None,
            ),
        ]
        liquidity = MetricBlock(block="liquidity", metrics=liq_metrics, level=liq)

        rv = realized_volatility(df)
        vol = volatility_level(float(rv.value or 0.0), self.rules)
        volatility = MetricBlock(
            block="volatility",
            metrics=[
                self._metric("realized_volatility", rv),
                self._metric("atr_ratio", atr_ratio(df, 14)),
                self._metric("amplitude_quantiles", amplitude_quantiles(df, [0.5, 0.9])),
            ],
            level=vol,
        )

        prices = self._close_array(df)
        returns = self._log_returns(prices)
        hurst = hurst_exponent(returns)
        vr = variance_ratio(returns, 5)
        adf = adf_pvalue(prices)
        pred_conf = overall_confidence(
            [
                confidence_for("hurst_exponent", hurst.effective_sample, self.rules),
                confidence_for("variance_ratio", vr.effective_sample, self.rules),
                confidence_for("adf_pvalue", adf.effective_sample, self.rules),
            ]
        )
        structure = structure_judgement(
            self._finite_or_none(hurst.value),
            self._finite_or_none(vr.value),
            self._finite_or_none(adf.value),
            pred_conf,
        )
        predictability = MetricBlock(
            block="predictability",
            metrics=[
                self._metric("return_autocorr", return_autocorr(returns, [1, 5, 10])),
                self._metric("hurst_exponent", hurst),
                self._metric("variance_ratio", vr),
                self._metric("adf_pvalue", adf),
                self._metric("skewness", skewness(returns)),
                self._metric("kurtosis", kurtosis(returns)),
            ],
            level=structure,
        )

        return [data_quality, liquidity, volatility, predictability]

    def _build_group_profile(
        self,
        vt_symbol: str,
        target: pl.DataFrame,
        interval: str,
        as_of: datetime,
        lookback_days: int,
        observation_symbols: list[str],
    ) -> GroupProfile | None:
        """计算目标标的与观测标的的多标的关联性画像。

        对每个 observation_symbol 加载同窗口行情，计算对数收益相关系数；
        汇总公共时间轴对齐覆盖率。

        Args:
            vt_symbol: 目标标的合约代码（已规范化）。
            target: 目标标的的窗口 frame。
            interval: K 线周期。
            as_of: 截止时间。
            lookback_days: 回看天数。
            observation_symbols: 观测标的列表；为空时直接返回 None。

        Returns:
            GroupProfile 对象；observation_symbols 为空时返回 None。
        """
        if not observation_symbols:
            return None
        frames: list[pl.DataFrame] = []
        members: list[str] = []
        correlations: dict[str, float] = {}
        for symbol in observation_symbols:
            normalized = normalize_vt_symbol(symbol)
            other = _load_window_frame(self.lab, normalized, interval, as_of, lookback_days)
            if other is None or other.is_empty():
                continue
            frames.append(other)
            members.append(normalized)
            corr = self._close_correlation(target, other)
            if corr is not None:
                correlations[normalized] = corr
        coverage = alignment_coverage(target, frames).value if frames else 0.0
        return GroupProfile(
            target=vt_symbol,
            members=members,
            alignment_coverage=float(coverage or 0.0),
            correlation_summary=correlations,
        )

    def _close_array(self, df: pl.DataFrame) -> np.ndarray:
        """从行情 frame 提取收盘价数组（缺列时返回空数组）。

        Args:
            df: 含 close 列的 polars DataFrame。

        Returns:
            float64 类型的 numpy 一维数组；缺 close 列时返回空数组。
        """
        if "close" not in df.columns:
            return np.asarray([], dtype=np.float64)
        return np.asarray(df["close"].to_list(), dtype=np.float64)

    def _log_returns(self, prices: np.ndarray) -> np.ndarray:
        """由价格序列计算对数收益序列（剔除非有限/非正价格后差分 log）。

        Args:
            prices: 价格序列（一维 numpy 数组）。

        Returns:
            对数收益序列；有效价格少于 2 时返回空数组。
        """
        finite = prices[np.isfinite(prices) & (prices > 0)]
        if finite.size < 2:
            return np.asarray([], dtype=np.float64)
        return np.diff(np.log(finite))

    def _close_correlation(self, target: pl.DataFrame, other: pl.DataFrame) -> float | None:
        """计算目标与观测标的在公共时间轴上的对数收益相关系数。

        Args:
            target: 目标标的行情 frame（需含 datetime / close 列）。
            other: 观测标的行情 frame（需含 datetime / close 列）。

        Returns:
            Pearson 相关系数（-1 到 1）；公共对齐样本 < 3、任一序列方差为 0、
            或相关系数为 NaN 时返回 None。
        """
        if "datetime" not in target.columns or "close" not in target.columns:
            return None
        if "datetime" not in other.columns or "close" not in other.columns:
            return None
        joined = target.select(["datetime", "close"]).rename({"close": "target_close"}).join(
            other.select(["datetime", "close"]).rename({"close": "other_close"}),
            on="datetime",
            how="inner",
        )
        if joined.height < 3:
            return None
        a = self._log_returns(np.asarray(joined["target_close"].to_list(), dtype=np.float64))
        b = self._log_returns(np.asarray(joined["other_close"].to_list(), dtype=np.float64))
        n = min(a.size, b.size)
        if n < 2:
            return None
        a = a[:n]
        b = b[:n]
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return None
        corr = float(np.corrcoef(a, b)[0, 1])
        return corr if math.isfinite(corr) else None

    def _finite_or_none(self, value: Any) -> float | None:
        """若 value 为有限实数则转为 float，否则返回 None（用于过滤 NaN/Inf）。

        Args:
            value: 待转换值，可为 int/float 或其他类型。

        Returns:
            有限浮点数；NaN/Inf/非数值类型返回 None。
        """
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def _json_safe(self, value: Any) -> Any:
        """将值规整为 JSON 可序列化形式：Inf/NaN 置 None，dict 递归处理。

        Args:
            value: 待规整的值（可为数值、dict 或其他）。

        Returns:
            JSON 可序列化的值；NaN/Inf 变为 None，None 的键会被 dict 清理掉。
        """
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for k, v in value.items():
                safe = self._json_safe(v)
                if safe is not None:
                    cleaned[str(k)] = safe
            return cleaned
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                return None
            return float(value)
        return value
