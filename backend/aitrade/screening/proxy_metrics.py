"""
CNN 适配代理指标（CNN Proxy Metrics）纯函数。

本文件实现三个 CNN 适配代理指标，作为 Tier-1 选股的廉价排名信号，捕捉线性可预测性
指标无法反映的 CNN 增益来源（Requirement 3）。

**指标列表**：
- `nonlinearity`：线性 AR 残差中残余的非线性结构强度；
- `pattern_recurrence`：平移不变局部窗口形态的复现程度；
- `temporal_stability`：前后子窗统计画像的稳定度。

**设计约束**（全部对齐 profiling/metrics.py 风格）：
- 纯函数、无 I/O、无全局状态：每个函数只对内存中的 numpy.ndarray 做只读统计。
- 返回 `MetricResult(value, effective_sample)`（与 profiling 保持统一契约）。
- **样本不足降级（Property 5）**：当 effective_sample 低于有效性下限时，value 返回
  None（而非 NaN/inf），上层 scoring.py 据此判为 insufficient 并排除该维度。
- 常数序列等退化输入 → value=None 或定义良好的边界，绝不泄漏 NaN/inf。
- 确定性：无 RNG，相同输入恒产生相同输出。

指标的设计立场是"高召回排名信号"而非"CNN 可学习性的严格证明"，最终结论由 Tier-2 给出
（Requirement 3.6）。
"""

from __future__ import annotations

import math

import numpy as np

from aitrade.profiling.metrics import MetricResult

# ---------------------------------------------------------------------------
# 各指标的最小有效样本量下限（低于此值视为退化，返回 value=None）。
# 这些是结构性下限，真正的置信度分档由 scoring.py / rules.py 负责。
# ---------------------------------------------------------------------------

#: nonlinearity：需要拟合 AR(p) 并计算残差，ar_order+1 个参数 + 残差自相关所需样本。
#: 取 ar_order（默认 1）+ 20，确保残差序列有足够点用于统计；调用方以实际 ar_order 细判。
_MIN_NONLINEARITY_SAMPLE = 20

#: pattern_recurrence：至少需要 2 个完整的滑动窗口；取 window * 2 为基准，
#: 函数内依 window 参数动态检查（floor = window * 2），此常量为绝对最低值。
_MIN_PATTERN_SAMPLE_ABS = 10

#: temporal_stability：切前后两半，每半段需要至少 4 个点以估计均值/方差。
_MIN_TEMPORAL_SAMPLE = 8


# ---------------------------------------------------------------------------
# 内部辅助函数（模块私有）
# ---------------------------------------------------------------------------


def _clean_1d(arr: np.ndarray) -> np.ndarray:
    """将输入转为一维 float64 数组并剔除非有限值（NaN / ±Inf）。

    纯函数：返回新数组，不修改入参。与 profiling/metrics.py 的同名函数保持风格一致。

    Args:
        arr: 任意形状的数组，将被 ravel 为一维并转为 float64。

    Returns:
        剔除非有限值后的一维 float64 ndarray；可能为空数组。
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def _clamp01(value: float) -> float:
    """将数值 clamp 到闭区间 [0, 1]；对 NaN 返回 0.0（防御性处理）。

    与 profiling/metrics.py 保持一致的辅助函数。

    Args:
        value: 待 clamp 的浮点数。

    Returns:
        clamp 到 [0, 1] 的结果；NaN 输入返回 0.0。
    """
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# 代理指标 1：nonlinearity（非线性结构强度）
# ---------------------------------------------------------------------------


def nonlinearity(returns: np.ndarray, *, ar_order: int = 1) -> MetricResult:
    """测量收益序列在去除低阶线性自相关后的残余非线性结构强度（Requirement 3.1）。

    **方法**：
    1. 对收益序列拟合 AR(ar_order)（最小二乘），取残差序列 ε̂；
    2. 计算 |ε̂|（绝对残差）在滞后 1..min(5, n//4) 上的自相关（ARCH-style），
       取各滞后绝对自相关的均值作为非线性强度 s；
    3. 用 tanh(5s) 把 [0,∞) 映射到 [0,1)（tanh 压缩，保持单调性），使更强的非线性
       结构产生更高分值，同时将值域限制在 [0,1)。

    **直觉**：若线性 AR 模型能充分解释序列的动态，残差应接近白噪声，|ε̂| 的自相关应
    接近 0（低非线性分）；若残差平方/绝对值仍有显著自相关（ARCH 效应、阈值效应等），
    则 CNN 有机会利用这些结构（高非线性分）。

    **样本不足降级（Property 5）**：有效样本 < _MIN_NONLINEARITY_SAMPLE + ar_order
    或 ar_order <= 0 时返回 value=None；常数序列（方差为 0）亦返回 value=None。

    Args:
        returns: 收益序列，一维 np.ndarray；非有限值将被剔除。
        ar_order: AR 模型阶数，默认 1；必须 >= 1。

    Returns:
        MetricResult，value 为 [0,1) 的非线性强度（越高越强），或样本不足时为 None；
        effective_sample 为有效收益点数（剔除非有限值后）。

    Example:
        >>> import numpy as np
        >>> r = np.random.default_rng(0).standard_normal(200)
        >>> result = nonlinearity(r)
        >>> result.value is not None  # True
        True
    """
    r = _clean_1d(returns)
    n = int(r.size)

    # --- 参数校验与样本下限 ---
    if not isinstance(ar_order, int) or ar_order <= 0:
        return MetricResult(value=None, effective_sample=n)

    min_needed = _MIN_NONLINEARITY_SAMPLE + ar_order
    if n < min_needed:
        return MetricResult(value=None, effective_sample=n)

    # --- 常数序列退化：AR 拟合无意义（用 ptp 替代 var == 0 以避免浮点精度问题）---
    if float(np.ptp(r)) == 0.0:
        return MetricResult(value=None, effective_sample=n)

    # --- 构造 AR(p) 设计矩阵并最小二乘拟合 ---
    # Y = r[ar_order:], X = [r[p-1:n-1], r[p-2:n-2], ..., r[0:n-p], 1]
    p = ar_order
    Y = r[p:]          # 长度 n - p
    obs = Y.size       # 有效回归观测数

    # 滞后列：X[:, i] = r[p-1-i : n-1-i]（i = 0..p-1）
    lag_cols = [r[p - 1 - i : n - 1 - i] for i in range(p)]
    X = np.column_stack(lag_cols + [np.ones(obs)])  # 含截距，形状 (obs, p+1)

    # 防御：设计矩阵奇异时退化（极端情况）
    try:
        coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return MetricResult(value=None, effective_sample=n)

    resid = Y - X @ coef  # 残差序列

    # --- 计算 |残差| 的滞后自相关（ARCH-style 检测） ---
    abs_resid = np.abs(resid)
    m = abs_resid.size

    # 需要足够点才能估计自相关
    if m < 4:
        return MetricResult(value=None, effective_sample=n)

    mu_abs = float(np.mean(abs_resid))
    denom = float(np.sum((abs_resid - mu_abs) ** 2))

    if denom <= 0.0:
        # 残差全为常数（极罕见，如全零收益），视为无非线性结构
        return MetricResult(value=0.0, effective_sample=n)

    max_lag = max(1, min(5, m // 4))
    acf_vals: list[float] = []
    for lag in range(1, max_lag + 1):
        if m - lag < 2:
            break
        num = float(np.sum((abs_resid[lag:] - mu_abs) * (abs_resid[:-lag] - mu_abs)))
        acf_vals.append(abs(num / denom))  # 取绝对值使正/负相关都算作结构

    if not acf_vals:
        return MetricResult(value=0.0, effective_sample=n)

    avg_acf = float(np.mean(acf_vals))

    # tanh(5s) 映射到 [0,1)：s=0 → 0，s=0.5 → tanh(2.5) ≈ 0.99
    score = float(math.tanh(5.0 * avg_acf))
    return MetricResult(value=_clamp01(score), effective_sample=n)


# ---------------------------------------------------------------------------
# 代理指标 2：pattern_recurrence（局部形态复现性）
# ---------------------------------------------------------------------------


def pattern_recurrence(close: np.ndarray, *, window: int = 16) -> MetricResult:
    """测量价格序列中平移不变局部窗口形态的复现程度（Requirement 3.2）。

    **方法**：
    1. 取对数价格序列 log(close)，用长度 `window` 的滑动窗口提取所有子序列；
    2. 对每个窗口做 z-normalization（零均值单位标准差），得平移/尺度不变的"形态向量"；
    3. 对每个窗口，在其余窗口中寻找**最大绝对余弦相似度（最近邻）**，取绝对值使
       "正向/镜像"形态都算作相似（abs-cosine）；
    4. 对所有窗口的最近邻绝对相似度取均值作为复现分，取值 [0, 1]；
    5. 当窗口数 > 500 时均匀下采样至 500 个查询窗口以控制 O(N²) 矩阵乘法成本。

    **直觉**：若价格序列包含反复出现的 K 线形态（头肩顶、双底、楔形等），每个窗口都能在
    其他时间段找到高度相似的副本（最近邻相似度接近 1）；纯随机游走的窗口在其他时段很难
    找到高度近似的副本（最近邻相似度较低）。CNN 的卷积核正是靠捕捉这类重复局部形态工作，
    故高复现分标的对 CNN 更友好。

    **样本不足降级（Property 5）**：
    - 有效点数 < max(_MIN_PATTERN_SAMPLE_ABS, window * 2) 时返回 value=None；
    - window <= 0 时返回 value=None；
    - 常数价格序列（std ≈ 0）或窗口内方差为 0（无法 z-normalize）统一降级到 0（定义良好边界）。

    Args:
        close: 价格序列（收盘价），一维 np.ndarray；须为正值（非正值视为缺失被剔除）。
        window: 滑动窗口长度（bar 数），默认 16；须 >= 2。

    Returns:
        MetricResult，value 为 [0,1] 的形态复现分（越高形态越重复）或不足时为 None；
        effective_sample 为有效价格点数。

    Example:
        >>> import numpy as np
        >>> # 正弦波形态高度重复
        >>> t = np.linspace(0, 4 * np.pi, 200)
        >>> prices = 100 + 10 * np.sin(t)
        >>> result = pattern_recurrence(prices, window=16)
        >>> result.value > 0.6  # True（重复性强）
        True
    """
    # 剔除非有限值，同时要求价格为正（价格序列约定）
    raw = np.asarray(close, dtype=np.float64).ravel()
    valid_mask = np.isfinite(raw) & (raw > 0.0)
    c = raw[valid_mask]
    n = int(c.size)

    # --- 参数校验与样本下限 ---
    if not isinstance(window, int) or window < 2:
        return MetricResult(value=None, effective_sample=n)

    min_needed = max(_MIN_PATTERN_SAMPLE_ABS, window * 2)
    if n < min_needed:
        return MetricResult(value=None, effective_sample=n)

    # --- 取对数价格 ---
    log_c = np.log(c)

    # --- 提取所有 z-normalized 滑动窗口 ---
    num_windows = n - window + 1
    # 构造 (num_windows, window) 的形状矩阵（视图，不复制）
    shape = (num_windows, window)
    strides = (log_c.strides[0], log_c.strides[0])
    try:
        patches = np.lib.stride_tricks.as_strided(log_c, shape=shape, strides=strides)
    except (ValueError, OverflowError):
        return MetricResult(value=None, effective_sample=n)

    # 计算每个窗口的均值和标准差（保持副本以便 z-normalize）
    patches = np.array(patches, dtype=np.float64)  # 强制复制，避免 stride 共享
    means = patches.mean(axis=1, keepdims=True)
    stds = patches.std(axis=1, keepdims=True)

    # 过滤掉标准差为 0 的窗口（常数片段，z-normalize 无定义）
    valid_win = (stds > 0.0).ravel()
    if not valid_win.any():
        # 所有窗口均为常数（极退化情况），无法度量形态差异，返回 0
        return MetricResult(value=0.0, effective_sample=n)

    z_patches = (patches[valid_win] - means[valid_win]) / stds[valid_win]  # (k, window)
    k = z_patches.shape[0]

    if k < 2:
        return MetricResult(value=None, effective_sample=n)

    # --- 计算每个窗口的最近邻余弦相似度（NN-based 形态复现）---
    #
    # 度量方式：对每个 z-normalized 窗口，找它与至少 window 步距之外的其它窗口中
    # 最相似的一个，取最大**绝对**余弦相似度（绝对值使"正向/镜像"形态均算作相似），
    # 再对所有窗口取均值作为形态复现分。
    #
    # 关键细节：排除"对角线 ±window 以内"的近邻（重叠窗口），避免两个几乎完全共享数据
    # 的窗口（overlap=window-1）互为最近邻（这在任何序列中都会发生，无法区分形态真实
    # 复现与纯粹的窗口滑动连续性）。
    #
    # 直觉：周期性序列（正弦波、重复 K 线形态等）的窗口在其他时间段（>window 步距）
    # 均有高度近似的副本，NN 相似度接近 1；纯随机游走在时间上无周期性，远处窗口
    # 难以找到高度近似的副本，NN 相似度较低（通常 0.6~0.9）。
    #
    # 为避免 O(k²) 内存消耗，当 k 较大时均匀下采样至最多 _MAX_WINDOWS 个查询窗口，
    # 但仍对全量 k 个窗口做参照集，确保可以找到真正的远处近邻。
    _MAX_WINDOWS = 300  # 查询行数上限（gram 矩阵约 300×k，控制内存/速度）

    if k > _MAX_WINDOWS:
        # 均匀步长下采样（确定性，不使用 RNG）
        step = k // _MAX_WINDOWS
        query_idxs = np.arange(0, k, step)
        query = z_patches[query_idxs]
    else:
        query_idxs = np.arange(k)
        query = z_patches

    q = query.shape[0]

    # gram[i, j] = <query[i], z_patches[j]> / window，即余弦相似度（z-normalize 后）
    gram = query @ z_patches.T / window  # (q, k)，理论取值 [-1, 1]

    # 用 NaN 掩盖"过近"的窗口（索引距离 < window 即有重叠），避免近邻为相邻滑动窗口
    rows_abs = query_idxs[:, None]        # (q, 1) — 查询窗口在原 z_patches 中的全局索引
    cols_abs = np.arange(k)[None, :]     # (1, k) — 参照窗口的全局索引
    near_mask = np.abs(rows_abs - cols_abs) < window  # (q, k)，True=过近
    gram_masked = np.where(near_mask, np.nan, gram)   # (q, k)，遮掩近邻

    # 取每行最大绝对相似度
    # 将 NaN 替换为 0.0 后取 max（NaN 行的 max 结果为 0.0，即无可用远处近邻时复现分为 0）
    abs_masked = np.abs(np.where(np.isnan(gram_masked), 0.0, gram_masked))
    nn_abs_sims = abs_masked.max(axis=1)  # (q,)

    if nn_abs_sims.size == 0:
        return MetricResult(value=0.0, effective_sample=n)

    score = float(nn_abs_sims.mean())
    return MetricResult(value=_clamp01(score), effective_sample=n)


# ---------------------------------------------------------------------------
# 代理指标 3：temporal_stability（时间稳定性）
# ---------------------------------------------------------------------------


def temporal_stability(returns: np.ndarray) -> MetricResult:
    """测量收益序列前后半段统计画像的稳定程度（Requirement 3.3）。

    **方法**：
    1. 把有效收益序列均分为前半段 A 和后半段 B（若长度为奇数则 B 多一点）；
    2. 计算两段的统计特征差异：
       - 相对波动差 Δσ = |σ_B - σ_A| / (σ_A + σ_B + ε)，取值 [0,1)；
       - 相对均值差 Δμ = |μ_B - μ_A| / (σ_total + ε)，用整体标准差归一；
       - 综合漂移 drift = (Δσ + Δμ) / 2，取值 [0, ~1]；
    3. 稳定度 stability = exp(-3 * drift)，drift=0 → 1（完全稳定），drift 大 → → 0。

    **直觉**：若价格/收益的统计特征（波动、均值）在时间上保持稳定，CNN 在早期数据上
    学到的形态特征在后期数据上仍然有效，OOS 泛化更可靠（高稳定分）；若统计特征剧变，
    训练集和测试集实质上来自不同分布，CNN 难以泛化（低稳定分）。

    **样本不足降级（Property 5）**：有效样本 < _MIN_TEMPORAL_SAMPLE 时返回 value=None；
    半段标准差均为 0（常数序列）时返回 value=None。

    Args:
        returns: 收益序列，一维 np.ndarray；非有限值将被剔除。

    Returns:
        MetricResult，value 为 [0,1] 的稳定度（越高越稳定）或样本不足时为 None；
        effective_sample 为有效收益点数。

    Example:
        >>> import numpy as np
        >>> # 均匀白噪声：前后半段统计特征相似，稳定性高
        >>> r = np.full(100, 0.01)  # 常数序列退化
        >>> res = temporal_stability(r)
        >>> res.value is None  # True（常数序列，方差为 0）
        True
    """
    r = _clean_1d(returns)
    n = int(r.size)

    # --- 样本下限 ---
    if n < _MIN_TEMPORAL_SAMPLE:
        return MetricResult(value=None, effective_sample=n)

    # --- 切分前后两半 ---
    mid = n // 2
    half_a = r[:mid]
    half_b = r[mid:]

    # --- 计算各半段统计量 ---
    mu_a = float(np.mean(half_a))
    mu_b = float(np.mean(half_b))
    sigma_a = float(np.std(half_a))
    sigma_b = float(np.std(half_b))

    # 两半段标准差均为 0：常数序列，稳定度无意义（退化）
    if sigma_a == 0.0 and sigma_b == 0.0:
        return MetricResult(value=None, effective_sample=n)

    # --- 相对波动差（归一化到 [0,1)）---
    eps = 1e-10
    delta_sigma = abs(sigma_b - sigma_a) / (sigma_a + sigma_b + eps)

    # --- 相对均值差（用整体标准差归一化）---
    sigma_total = float(np.std(r))
    if sigma_total <= 0.0:
        # 整体方差为 0 但某半段不为 0 不可能（上方已排除两半均 0），防御性保留
        sigma_total = eps
    delta_mu = abs(mu_b - mu_a) / (sigma_total + eps)
    # delta_mu 理论上无上界（极端漂移），clamp 到 [0,1] 以稳定综合指数
    delta_mu = min(1.0, delta_mu)

    # --- 综合漂移 & 指数衰减稳定度 ---
    drift = (delta_sigma + delta_mu) / 2.0
    # exp(-3 * drift)：drift=0 → 1.0，drift=1 → exp(-3) ≈ 0.05
    stability = math.exp(-3.0 * drift)

    return MetricResult(value=_clamp01(stability), effective_sample=n)
