"""品种化合约属性推断模块。

当前实现：根据 vt_symbol 的代码前缀与交易所后缀推断涨跌停比例与 T+1 约束，
供回测引擎在缺少显式合约配置时自动选取正确的封板阈值与 T+0/T+1 制度。

规则（A 股现行制度，2024）：
  - 300/301 开头（创业板） + 688/689 开头（科创板） → 20%
  - .BSE 结尾（北交所）                             → 30%
  - 110/111/113/118 开头 .SSE（沪市可转债）
    123/127/128 开头 .SZSE（深市可转债）             → None（无涨跌停）
  - 其余（主板、ETF 等）                            → 10%（默认）

T+1 推断规则（infer_t_plus1）：
  - 可转债前缀（同上列 _CB_SSE_PREFIXES / _CB_SZSE_PREFIXES）→ False（T+0）
  - 其余品种                                        → True（T+1）

注意：ST/*ST 股保留原 6 位代码，无法由代码前缀区分，代码层面无从推断其
±5% 限幅。若需正确模拟 ST 品种涨跌停，须在 contract.json 中为该合约显式
设置 limit_ratio=0.05；否则回测将按默认 10% 封板处理。

注意（infer_t_plus1）：部分 T+0 品种（如特定债券、跨境 ETF）不在推断表内，
须经 contract.json 的 t_plus1=false 显式覆盖，引擎配置路径优先于推断结果。
"""

from __future__ import annotations

# 各前缀与对应限幅（None = 转债，无封板限制）
_GEM_SCI_PREFIXES: frozenset[str] = frozenset({
    "300", "301",  # 创业板
    "688", "689",  # 科创板
})

_CB_SSE_PREFIXES: frozenset[str] = frozenset({
    "110", "111", "113", "118",  # 沪市可转债
})

_CB_SZSE_PREFIXES: frozenset[str] = frozenset({
    "123", "127", "128",  # 深市可转债
})


def infer_t_plus1(vt_symbol: str) -> bool:
    """根据 vt_symbol 推断该品种是否受 T+1 卖出限制。

    复用文件内已定义的转债前缀常量（_CB_SSE_PREFIXES / _CB_SZSE_PREFIXES），
    转债前缀对应 T+0；其余品种（主板、创业板、ETF 等）返回 True（T+1）。

    Parameters
    ----------
    vt_symbol:
        格式为 ``<代码>.<交易所>``，例如 ``113050.SSE``。

    Returns
    -------
    bool
        - False — 可转债（T+0，当日买入当日可卖）
        - True  — 其余品种（T+1，当日买入次日才能卖）

    注意：
        部分 T+0 品种（如特定债券、跨境 ETF 等）不在推断表内，须经
        contract.json 的 ``t_plus1: false`` 显式覆盖；引擎 ``set_parameters``
        中合约配置优先级高于本推断函数。
    """
    if "." not in vt_symbol:
        return True

    code, exchange = vt_symbol.rsplit(".", 1)
    exchange = exchange.upper()
    prefix3 = code[:3]

    # 沪市可转债：T+0
    if exchange == "SSE" and prefix3 in _CB_SSE_PREFIXES:
        return False
    # 深市可转债：T+0
    if exchange == "SZSE" and prefix3 in _CB_SZSE_PREFIXES:
        return False

    # 其余品种（主板、创业板、科创板、北交所、ETF 等）：T+1
    return True


def infer_limit_ratio(vt_symbol: str) -> float | None:
    """根据 vt_symbol 推断该品种的单边涨跌停比例。

    Parameters
    ----------
    vt_symbol:
        格式为 ``<代码>.<交易所>``，例如 ``300001.SZSE``。

    Returns
    -------
    float | None
        - 0.2 — 创业板 / 科创板（±20%）
        - 0.3 — 北交所（±30%）
        - None — 可转债（无涨跌停限制）
        - 0.1 — 默认（主板、ETF 等，±10%）
    """
    if "." not in vt_symbol:
        return 0.1

    code, exchange = vt_symbol.rsplit(".", 1)
    exchange = exchange.upper()

    # 北交所：代码以 .BSE 结尾
    if exchange == "BSE":
        return 0.3

    prefix3 = code[:3]

    # 创业板 / 科创板：±20%
    if prefix3 in _GEM_SCI_PREFIXES:
        return 0.2

    # 可转债：无涨跌停
    if exchange == "SSE" and prefix3 in _CB_SSE_PREFIXES:
        return None
    if exchange == "SZSE" and prefix3 in _CB_SZSE_PREFIXES:
        return None

    # 默认：主板、ETF、LOF 等 ±10%
    return 0.1
