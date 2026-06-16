"""
proxy_metrics.py 的属性测试与例证测试。

覆盖范围：
1. Property 5（样本不足 / 退化降级）：短数组返回 value=None；常数数组返回 None 或
   定义良好边界（绝不输出 NaN / inf）。
2. 值域有界性：value 不为 None 时落在各指标文档化的 [0,1] 范围内。
3. 确定性：相同输入 → 相同输出（无 RNG）。
4. effective_sample 合法性：非负整数，与输入有效点数一致。
5. 例证语义测试：强结构序列（ARCH 残差 / 重复形态 / 稳定收益）比白噪声分值更高。

# Feature: cnn-stock-screening, Property 5: 代理指标样本不足降级
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ────────────────────────────────────────────────────────────────────────────
# 被测函数导入
# ────────────────────────────────────────────────────────────────────────────

from aitrade.screening.proxy_metrics import (
    _MIN_NONLINEARITY_SAMPLE,
    _MIN_PATTERN_SAMPLE_ABS,
    _MIN_TEMPORAL_SAMPLE,
    nonlinearity,
    pattern_recurrence,
    temporal_stability,
)


# ────────────────────────────────────────────────────────────────────────────
# 辅助策略
# ────────────────────────────────────────────────────────────────────────────

# 有限浮点元素策略（避免 NaN/inf 作为"正常输入"参与合理测试）
_finite_float = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)

# 合理长度的返回序列（最多 300 个点，避免测试太慢）
_returns_array = st.lists(_finite_float, min_size=1, max_size=300).map(np.array)

# 合理长度的价格序列（正数）
_price_array = st.lists(
    st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=300,
).map(np.array)


def _is_valid_value(v) -> bool:
    """判断指标值是否合法（None 或有限浮点）。"""
    if v is None:
        return True
    return isinstance(v, float) and math.isfinite(v)


# ════════════════════════════════════════════════════════════════════════════
# nonlinearity 测试
# ════════════════════════════════════════════════════════════════════════════


class TestNonlinearity:
    """nonlinearity() 的属性测试与例证测试。"""

    # ── Property 5: 短数组降级 ──────────────────────────────────────────────

    @settings(max_examples=100)
    @given(
        # 默认 ar_order=1，真实阈值为 _MIN_NONLINEARITY_SAMPLE + ar_order = 21。
        # max_size 须等于阈值 - 1（即 20），覆盖所有低于阈值的尺寸（含 size=20）。
        arr=st.lists(
            _finite_float, min_size=0, max_size=_MIN_NONLINEARITY_SAMPLE
        ).map(np.array)
    )
    def test_short_array_returns_none(self, arr: np.ndarray) -> None:
        # Feature: cnn-stock-screening, Property 5: 代理指标样本不足降级
        """有效样本不足 _MIN_NONLINEARITY_SAMPLE + ar_order（默认 21）时，value 必须为 None。"""
        result = nonlinearity(arr)
        assert result.value is None, (
            f"期望 value=None，实际={result.value}（len={len(arr)}，"
            f"需 >= {_MIN_NONLINEARITY_SAMPLE + 1}）"
        )

    # ── Property 5: 常数序列降级 ────────────────────────────────────────────

    @settings(max_examples=100)
    @given(length=st.integers(min_value=_MIN_NONLINEARITY_SAMPLE + 2, max_value=300))
    def test_constant_array_returns_none(self, length: int) -> None:
        # Feature: cnn-stock-screening, Property 5: 代理指标样本不足降级
        """常数序列（方差为 0）退化输入，nonlinearity value 必须为 None。"""
        arr = np.full(length, 0.01)
        result = nonlinearity(arr)
        assert result.value is None, f"常数序列应返回 None，实际={result.value}"

    # ── 值域有界性 ──────────────────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_returns_array)
    def test_value_in_range_or_none(self, arr: np.ndarray) -> None:
        """value 不为 None 时必须落在 [0,1]；绝不出现 NaN / inf。"""
        result = nonlinearity(arr)
        assert _is_valid_value(result.value), f"value={result.value} 不是有限值或 None"
        if result.value is not None:
            assert 0.0 <= result.value <= 1.0, f"value={result.value} 超出 [0,1]"

    # ── 确定性 ──────────────────────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_returns_array)
    def test_determinism(self, arr: np.ndarray) -> None:
        """相同输入 → 相同输出（无 RNG）。"""
        r1 = nonlinearity(arr)
        r2 = nonlinearity(arr)
        assert r1.value == r2.value
        assert r1.effective_sample == r2.effective_sample

    # ── effective_sample 合法性 ─────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_returns_array)
    def test_effective_sample_nonneg_int(self, arr: np.ndarray) -> None:
        """effective_sample 为非负整数，且不超过输入数组长度。"""
        result = nonlinearity(arr)
        assert isinstance(result.effective_sample, int)
        assert result.effective_sample >= 0
        assert result.effective_sample <= len(arr)

    # ── ar_order <= 0 时降级 ────────────────────────────────────────────────

    def test_invalid_ar_order_returns_none(self) -> None:
        """ar_order <= 0 时返回 value=None（参数校验降级）。"""
        arr = np.random.default_rng(42).standard_normal(100)
        assert nonlinearity(arr, ar_order=0).value is None
        assert nonlinearity(arr, ar_order=-1).value is None

    # ── 例证语义：ARCH 序列应比白噪声有更高非线性分 ─────────────────────────

    def test_arch_series_higher_than_white_noise(self) -> None:
        """具有 ARCH(1) 效应的序列比标准白噪声有更高的非线性分。"""
        rng = np.random.default_rng(2024)
        n = 500

        # 纯白噪声
        wn = rng.standard_normal(n)

        # 简单 ARCH(1) 样本：σ_t² = 0.1 + 0.8 * ε_{t-1}²
        arch = np.zeros(n)
        sig2 = 1.0
        for i in range(n):
            arch[i] = math.sqrt(max(sig2, 1e-6)) * rng.standard_normal()
            sig2 = 0.1 + 0.8 * arch[i] ** 2

        r_wn = nonlinearity(wn)
        r_arch = nonlinearity(arch)

        assert r_wn.value is not None and r_arch.value is not None, (
            "两者均应有足够样本"
        )
        assert r_arch.value > r_wn.value, (
            f"ARCH 序列({r_arch.value:.4f}) 应 > 白噪声({r_wn.value:.4f})"
        )


# ════════════════════════════════════════════════════════════════════════════
# pattern_recurrence 测试
# ════════════════════════════════════════════════════════════════════════════


class TestPatternRecurrence:
    """pattern_recurrence() 的属性测试与例证测试。"""

    # ── Property 5: 短数组降级 ──────────────────────────────────────────────

    @settings(max_examples=100)
    @given(
        # 使用 flatmap 让 arr 的长度上界依赖 window，确保每个生成样例严格满足
        # len(arr) < min_needed，无虚空断言（vacuous assertion）。
        window_and_arr=st.integers(min_value=2, max_value=32).flatmap(
            lambda w: st.tuples(
                st.just(w),
                st.lists(
                    st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
                    min_size=0,
                    max_size=max(_MIN_PATTERN_SAMPLE_ABS, w * 2) - 1,
                ).map(np.array),
            )
        )
    )
    def test_short_array_returns_none(self, window_and_arr: tuple) -> None:
        # Feature: cnn-stock-screening, Property 5: 代理指标样本不足降级
        """有效样本不足 max(_MIN_PATTERN_SAMPLE_ABS, window * 2) 时，value 为 None。

        使用 flatmap 使数组长度上界严格依赖 window，保证每个生成样例都真正触发降级路径，
        无虚空断言（vacuous assertion）。
        """
        window, arr = window_and_arr
        min_needed = max(_MIN_PATTERN_SAMPLE_ABS, window * 2)
        result = pattern_recurrence(arr, window=window)
        assert result.value is None, (
            f"期望 None，实际={result.value}（len={len(arr)} < {min_needed}，window={window}）"
        )

    # ── Property 5: window <= 0 时降级 ──────────────────────────────────────

    @settings(max_examples=50)
    @given(arr=_price_array)
    def test_invalid_window_returns_none(self, arr: np.ndarray) -> None:
        # Feature: cnn-stock-screening, Property 5: 代理指标样本不足降级
        """window <= 0 时 value 为 None（参数校验降级）。"""
        result_zero = pattern_recurrence(arr, window=0)
        result_neg = pattern_recurrence(arr, window=-1)
        assert result_zero.value is None
        assert result_neg.value is None

    # ── 值域有界性 ──────────────────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_price_array)
    def test_value_in_range_or_none(self, arr: np.ndarray) -> None:
        """value 不为 None 时必须落在 [0,1]；绝不出现 NaN / inf。"""
        result = pattern_recurrence(arr)
        assert _is_valid_value(result.value), f"value={result.value}"
        if result.value is not None:
            assert 0.0 <= result.value <= 1.0, f"value={result.value} 超出 [0,1]"

    # ── 确定性 ──────────────────────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_price_array)
    def test_determinism(self, arr: np.ndarray) -> None:
        """相同输入 → 相同输出。"""
        r1 = pattern_recurrence(arr)
        r2 = pattern_recurrence(arr)
        assert r1.value == r2.value
        assert r1.effective_sample == r2.effective_sample

    # ── effective_sample 合法性 ─────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_price_array)
    def test_effective_sample_nonneg_int(self, arr: np.ndarray) -> None:
        """effective_sample 为非负整数，且不超过输入数组长度。"""
        result = pattern_recurrence(arr)
        assert isinstance(result.effective_sample, int)
        assert result.effective_sample >= 0
        assert result.effective_sample <= len(arr)

    # ── 例证语义：正弦波（重复形态）> 白噪声 ───────────────────────────────

    def test_periodic_series_higher_than_random(self) -> None:
        """正弦波（高度重复的局部形态）应比随机游走有更高的形态复现分。"""
        n = 256
        # 正弦波价格序列（高度重复）
        t = np.linspace(0, 8 * math.pi, n)
        sine_prices = 100.0 + 10.0 * np.sin(t)

        # 随机游走价格序列
        rng = np.random.default_rng(1234)
        rw_prices = np.cumsum(rng.standard_normal(n)) + 100.0

        r_sine = pattern_recurrence(sine_prices, window=16)
        r_rw = pattern_recurrence(rw_prices, window=16)

        assert r_sine.value is not None and r_rw.value is not None
        assert r_sine.value > r_rw.value, (
            f"正弦波({r_sine.value:.4f}) 应 > 随机游走({r_rw.value:.4f})"
        )

    # ── 含 NaN/inf 的价格序列：有效点计算正确 ───────────────────────────────

    def test_handles_nan_in_prices(self) -> None:
        """含 NaN 的价格序列能正确处理，effective_sample 只计非 NaN 正值点数。"""
        arr = np.array([100.0, np.nan, 101.0, 102.0, np.inf, 103.0] * 20)
        result = pattern_recurrence(arr)
        # 有效点应为非 NaN 且正值的点，不含 nan 和 inf
        expected_valid = int(np.sum(np.isfinite(arr) & (arr > 0)))
        assert result.effective_sample == expected_valid


# ════════════════════════════════════════════════════════════════════════════
# temporal_stability 测试
# ════════════════════════════════════════════════════════════════════════════


class TestTemporalStability:
    """temporal_stability() 的属性测试与例证测试。"""

    # ── Property 5: 短数组降级 ──────────────────────────────────────────────

    @settings(max_examples=100)
    @given(
        arr=st.lists(
            _finite_float, min_size=0, max_size=_MIN_TEMPORAL_SAMPLE - 1
        ).map(np.array)
    )
    def test_short_array_returns_none(self, arr: np.ndarray) -> None:
        # Feature: cnn-stock-screening, Property 5: 代理指标样本不足降级
        """有效样本不足 _MIN_TEMPORAL_SAMPLE 时，value 为 None。"""
        result = temporal_stability(arr)
        assert result.value is None, (
            f"期望 None，实际={result.value}（len={len(arr)} < {_MIN_TEMPORAL_SAMPLE}）"
        )

    # ── Property 5: 常数序列降级 ────────────────────────────────────────────

    @settings(max_examples=100)
    @given(length=st.integers(min_value=_MIN_TEMPORAL_SAMPLE, max_value=300))
    def test_constant_array_returns_none(self, length: int) -> None:
        # Feature: cnn-stock-screening, Property 5: 代理指标样本不足降级
        """常数序列（两半段标准差均为 0）退化，value 为 None。"""
        arr = np.full(length, 0.5)
        result = temporal_stability(arr)
        assert result.value is None, f"常数序列应返回 None，实际={result.value}"

    # ── 值域有界性 ──────────────────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_returns_array)
    def test_value_in_range_or_none(self, arr: np.ndarray) -> None:
        """value 不为 None 时必须落在 [0,1]；绝不出现 NaN / inf。"""
        result = temporal_stability(arr)
        assert _is_valid_value(result.value), f"value={result.value}"
        if result.value is not None:
            assert 0.0 <= result.value <= 1.0, f"value={result.value} 超出 [0,1]"

    # ── 确定性 ──────────────────────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_returns_array)
    def test_determinism(self, arr: np.ndarray) -> None:
        """相同输入 → 相同输出。"""
        r1 = temporal_stability(arr)
        r2 = temporal_stability(arr)
        assert r1.value == r2.value
        assert r1.effective_sample == r2.effective_sample

    # ── effective_sample 合法性 ─────────────────────────────────────────────

    @settings(max_examples=100)
    @given(arr=_returns_array)
    def test_effective_sample_nonneg_int(self, arr: np.ndarray) -> None:
        """effective_sample 为非负整数，且不超过输入数组长度。"""
        result = temporal_stability(arr)
        assert isinstance(result.effective_sample, int)
        assert result.effective_sample >= 0
        assert result.effective_sample <= len(arr)

    # ── 例证语义：稳定收益序列 > 分段漂移序列 ─────────────────────────────

    def test_stable_series_higher_than_drifting(self) -> None:
        """统计特征稳定的序列应比前后段剧烈漂移的序列有更高的稳定度。"""
        rng = np.random.default_rng(42)
        n = 200

        # 稳定序列：前后段均来自相同分布
        stable = rng.normal(0.0, 0.01, size=n)

        # 剧烈漂移：前半段均值 0、波动 0.01；后半段均值 0.1、波动 0.1
        half = n // 2
        drifting = np.concatenate([
            rng.normal(0.0, 0.01, size=half),
            rng.normal(0.1, 0.1, size=n - half),
        ])

        r_stable = temporal_stability(stable)
        r_drifting = temporal_stability(drifting)

        assert r_stable.value is not None and r_drifting.value is not None
        assert r_stable.value > r_drifting.value, (
            f"稳定({r_stable.value:.4f}) 应 > 漂移({r_drifting.value:.4f})"
        )

    # ── Fix I1：极端制度切换（两半各自常数但均值不同）→ 低稳定分，非 None ────

    def test_regime_shift_constant_halves_returns_low_score(self) -> None:
        """两个常数半段但均值不同时，value 不为 None 且接近 0（极端漂移）。

        这是 temporal_stability 的制度切换盲点修复验证：
        - 前半全 0.0、后半全 1.0 → 最极端的均值漂移，应得到接近 0 的低分。
        - 全常数序列（前后均值相同）→ 无漂移信息，返回 None。
        """
        # 两半段各自常数但均值不同：极端制度切换
        regime_shift = np.concatenate([np.zeros(50), np.ones(50)])
        result = temporal_stability(regime_shift)
        assert result.value is not None, (
            "极端制度切换（A 全 0、B 全 1）应返回低稳定分，而非 None"
        )
        assert result.value < 0.3, (
            f"极端漂移应产生接近 0 的低稳定分，实际={result.value:.4f}"
        )

        # 对比：整个序列为常数（无漂移信息），应返回 None
        constant = np.full(100, 0.5)
        result_const = temporal_stability(constant)
        assert result_const.value is None, (
            f"全常数序列应返回 None，实际={result_const.value}"
        )

    # ── 边界：稳定序列接近 1 ────────────────────────────────────────────────

    def test_perfectly_stable_approaches_one(self) -> None:
        """完全相同分布的前后半段应产生接近 1 的稳定度。"""
        # 使用固定种子的完全相同的随机序列两份拼接（前后分布完全一致）
        rng = np.random.default_rng(999)
        half = np.random.default_rng(999).normal(0, 0.01, 100)
        # 前后段均来自同分布（用 copy 保证数值完全一致）
        arr = np.concatenate([half, half])
        result = temporal_stability(arr)
        assert result.value is not None
        assert result.value > 0.8, f"完全相同分布应有高稳定度，实际={result.value:.4f}"


# ════════════════════════════════════════════════════════════════════════════
# 跨指标：MetricResult 结构契约
# ════════════════════════════════════════════════════════════════════════════


class TestMetricResultContract:
    """验证三个指标都遵守 MetricResult 结构契约。"""

    @settings(max_examples=50)
    @given(arr=_returns_array)
    def test_returns_metric_result_tuple(self, arr: np.ndarray) -> None:
        """三个指标都返回有 .value 和 .effective_sample 属性的结构。"""
        # 价格序列转正值
        prices = np.abs(arr) + 1.0

        for fn_name, fn, arg in [
            ("nonlinearity", nonlinearity, arr),
            ("pattern_recurrence", pattern_recurrence, prices),
            ("temporal_stability", temporal_stability, arr),
        ]:
            result = fn(arg)
            assert hasattr(result, "value"), f"{fn_name} 缺少 .value 属性"
            assert hasattr(result, "effective_sample"), f"{fn_name} 缺少 .effective_sample 属性"

    @settings(max_examples=50)
    @given(arr=_returns_array)
    def test_no_nan_or_inf_leaks(self, arr: np.ndarray) -> None:
        """任何情况下 value 都不泄漏 NaN / inf（返回 None 或有限数）。"""
        prices = np.abs(arr) + 1.0

        for fn_name, fn, arg in [
            ("nonlinearity", nonlinearity, arr),
            ("pattern_recurrence", pattern_recurrence, prices),
            ("temporal_stability", temporal_stability, arr),
        ]:
            result = fn(arg)
            v = result.value
            if v is not None:
                assert math.isfinite(v), f"{fn_name} 泄漏了非有限值：{v}"

    def test_all_functions_with_single_element(self) -> None:
        """单元素数组：三个指标均返回 value=None（样本不足）。"""
        single = np.array([0.01])
        assert nonlinearity(single).value is None
        assert pattern_recurrence(np.array([100.0])).value is None
        assert temporal_stability(single).value is None

    def test_all_functions_with_empty_array(self) -> None:
        """空数组：三个指标均返回 value=None，effective_sample=0。"""
        empty = np.array([], dtype=np.float64)
        r_nl = nonlinearity(empty)
        r_pr = pattern_recurrence(empty)
        r_ts = temporal_stability(empty)
        assert r_nl.value is None and r_nl.effective_sample == 0
        assert r_pr.value is None and r_pr.effective_sample == 0
        assert r_ts.value is None and r_ts.effective_sample == 0

    def test_nan_only_array(self) -> None:
        """全 NaN 输入等效于空数组，均返回 value=None，effective_sample=0。"""
        nan_arr = np.full(50, float("nan"))
        r_nl = nonlinearity(nan_arr)
        r_ts = temporal_stability(nan_arr)
        assert r_nl.value is None
        assert r_ts.value is None
        assert r_nl.effective_sample == 0
        assert r_ts.effective_sample == 0
