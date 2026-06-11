"""Symbol Profiling 协调器。"""

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
    """加载本地行情，计算画像指标，并可选生成建议和持久化。"""

    def __init__(
        self,
        lab,
        *,
        rules: ProfilingRules | None = None,
        store: ProfileStore | None = None,
    ) -> None:
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
        if not persist:
            return
        try:
            self.store.save(profile)
        except Exception:
            pass

    def _metric(self, key: str, result: MetricResult, *, note: str | None = None) -> MetricValue:
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
        if "close" not in df.columns:
            return np.asarray([], dtype=np.float64)
        return np.asarray(df["close"].to_list(), dtype=np.float64)

    def _log_returns(self, prices: np.ndarray) -> np.ndarray:
        finite = prices[np.isfinite(prices) & (prices > 0)]
        if finite.size < 2:
            return np.asarray([], dtype=np.float64)
        return np.diff(np.log(finite))

    def _close_correlation(self, target: pl.DataFrame, other: pl.DataFrame) -> float | None:
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
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def _json_safe(self, value: Any) -> Any:
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
