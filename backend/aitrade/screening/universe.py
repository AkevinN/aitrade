"""
CNN 选股 Universe 自动发现与过滤模块（只读）。

调用 AlphaLab.list_data_resources() 扫描本地可用行情，按交易所、最小历史
bar 数、显式清单过滤出候选股票池，同时记录每只被排除标的的排除原因。

设计要点（Requirement 1 全条）：
- **只读**：仅调用 list_data_resources() 与 lab_utils 纯函数，绝不写盘（Requirement 10.1）。
- **容错**：单只缺数据只排除该标的，不中断整批（R1.4）。
- **空池结构化**：过滤后为空时返回 ([], excluded)，不抛异常（R1.5 / Property 10）。
- **规范化**：include_symbols 中的原始代码经 normalize_vt_symbol 规范化再匹配（R1.3）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aitrade.alpha.lab_utils import _parse_vt_symbol, normalize_vt_symbol

if TYPE_CHECKING:
    pass


def discover_universe(
    lab: Any,
    *,
    interval: str,
    min_bar_count: int,
    exchange: str | None = None,
    include_symbols: list[str] | None = None,
    exclude_symbols: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """扫描本地行情库，按过滤条件构建候选股票池（只读）。

    两种输入模式（R1.1 / R1.3）：
    - **默认**（``include_symbols`` 未提供）：候选集 = 本地 ``interval`` 下有数据的全部标的。
    - **显式清单**（``include_symbols`` 非空）：候选集 = 清单中的标的（经规范化），
      本地无此 interval 数据的条目被记入 ``excluded``（原因"无本地数据"）。

    此后按以下顺序逐条过滤，每条被排除的标的均记入 ``excluded``：
    1. ``exchange`` 交易所过滤（经 ``_parse_vt_symbol`` 解析后缀）。
    2. ``min_bar_count`` 最小历史 bar 数过滤。
    3. ``exclude_symbols`` 显式排除清单。

    Args:
        lab: AlphaLab 实例；唯一使用其 ``list_data_resources()`` 方法（只读）。
        interval: K 线周期，如 ``"d"``、``"1m"``，用于从 ``raw_bars`` 与
            ``derived_bars`` 中筛选对应数据。
        min_bar_count: 候选标的需达到的最小历史 bar 数；``row_count < min_bar_count``
            的标的被排除，原因为 ``"历史不足 (N < min_bar_count)"``。
        exchange: 交易所过滤，如 ``"SZSE"``、``"SSE"``、``"BSE"``；
            ``None`` 表示不按交易所过滤。
        include_symbols: 显式候选清单；``None`` 或空列表表示使用全量本地标的。
            每个代码均经 ``normalize_vt_symbol`` 规范化后再与本地数据匹配。
        exclude_symbols: 显式排除清单；列表中出现的标的无论通过其他过滤与否均被排除，
            原因为 ``"用户显式排除"``。``None`` 或空列表等效为不排除任何标的。

    Returns:
        ``(universe, excluded)`` 二元组：

        - ``universe``：经过所有过滤后的 vt_symbol 列表，按字典序升序排列
          （确定性顺序，便于测试与断言）。
        - ``excluded``：被排除标的的字典列表，每条格式为
          ``{"vt_symbol": str, "reason": str}``；与 ``ScreeningResult.excluded``
          字段契约一致。

    Note:
        - 完全只读：只调用 ``lab.list_data_resources()``，不调用任何写盘/聚合方法
          （Requirement 10.1）。
        - 过滤后为空时返回 ``([], excluded)``，不抛异常（R1.5 / Property 10）。
        - ``interval`` 不做规范化，与 ``raw_bars``/``derived_bars`` 的 ``interval``
          字段**精确匹配**；如需模糊匹配，调用方在传入前用
          ``_canonical_bar_interval`` 处理。

    Example:
        >>> universe, excluded = discover_universe(
        ...     lab, interval="d", min_bar_count=250, exchange="SZSE"
        ... )
        >>> print(len(universe), "candidates,", len(excluded), "excluded")
    """
    # 1. 拉取本地数据资源摘要，构造 interval 下的 vt_symbol → row_count 映射
    resources: dict[str, Any] = lab.list_data_resources()
    available_map: dict[str, int] = {}  # vt_symbol → row_count（去重取最大）

    for section in ("raw_bars", "derived_bars"):
        for item in resources.get(section, []):
            if item.get("interval") != interval:
                continue
            vt_sym = item.get("vt_symbol", "")
            if not vt_sym:
                continue
            row_count = int(item.get("row_count", 0))
            # 同一标的可能在 raw_bars 与 derived_bars 中均有记录；取较大值
            if vt_sym not in available_map or row_count > available_map[vt_sym]:
                available_map[vt_sym] = row_count

    excluded: list[dict] = []

    # 2. 确定初始候选集
    if include_symbols:
        # 显式清单模式（R1.3）：规范化后匹配本地数据
        normalized_exclude: set[str] = {
            normalize_vt_symbol(s) for s in (exclude_symbols or []) if s
        }
        candidates: dict[str, int] = {}  # vt_symbol → row_count

        for raw_sym in include_symbols:
            normalized = normalize_vt_symbol(raw_sym)
            if not normalized:
                continue
            if normalized in available_map:
                candidates[normalized] = available_map[normalized]
            else:
                excluded.append({"vt_symbol": normalized, "reason": "无本地数据"})
    else:
        # 默认模式（R1.1）：全量本地标的
        normalized_exclude = {
            normalize_vt_symbol(s) for s in (exclude_symbols or []) if s
        }
        candidates = dict(available_map)

    # 3. 交易所过滤（R1.2）
    if exchange is not None:
        passing: dict[str, int] = {}
        for vt_sym, row_count in candidates.items():
            _, sym_exchange = _parse_vt_symbol(vt_sym)
            if sym_exchange == exchange:
                passing[vt_sym] = row_count
            else:
                excluded.append(
                    {
                        "vt_symbol": vt_sym,
                        "reason": f"交易所不匹配 (期望 {exchange}，实际 {sym_exchange or '未知'})",
                    }
                )
        candidates = passing

    # 4. 最小历史 bar 数过滤（R1.2）
    passing = {}
    for vt_sym, row_count in candidates.items():
        if row_count >= min_bar_count:
            passing[vt_sym] = row_count
        else:
            excluded.append(
                {
                    "vt_symbol": vt_sym,
                    "reason": f"历史不足 ({row_count} < {min_bar_count})",
                }
            )
    candidates = passing

    # 5. 显式排除清单（R1.3 末尾；include_symbols 模式下 normalized_exclude 已设置）
    if normalized_exclude:
        passing = {}
        for vt_sym, row_count in candidates.items():
            if vt_sym in normalized_exclude:
                excluded.append({"vt_symbol": vt_sym, "reason": "用户显式排除"})
            else:
                passing[vt_sym] = row_count
        candidates = passing

    # 6. 排序后返回（确定性顺序，R1.5 / Property 10：空池不抛异常）
    universe = sorted(candidates.keys())
    return universe, excluded
