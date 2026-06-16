"""threshold_scale_check：按 objective 校验信号阈值的尺度合法性。

当回归模型误用了概率阈值（如 buy_threshold=0.6）、或概率模型的阈值越出
[0,1] 时，此模块负责识别并返回人类可读的违规原因列表，供回测端点（400
报错）、实盘 service（拒绝下单）和策略 on_init（软告警）三处共用。

纯函数，无 I/O，无副作用，可安全地在任何线程或进程中调用。
"""

from __future__ import annotations

_PROB_OBJECTIVES = {"classification", "path_class"}
"""概率输出型 objective 集合：buy/sell 必须落在 [0, 1] 内。"""


def threshold_scale_check(
    objective: str | None,
    buy_threshold: float,
    sell_threshold: float | None = None,
) -> list[str]:
    """按 objective 校验阈值尺度，返回违规原因列表（空列表 = 通过）。

    三种校验逻辑：

    - ``objective`` 为 ``None`` 或空字符串（legacy/无 objective 列）→
      跳过校验，返回 ``[]``（向后兼容）。
    - 概率型（``classification`` / ``path_class``）：
      ``buy_threshold`` 与 ``sell_threshold``（非 None）均须在 ``[0, 1]``，
      否则追加一条违规原因。
    - 收益型（``regression``）：``buy_threshold >= 0.5`` 视为
      "误用概率默认值"（收益口径下 +50% 不可达），追加一条违规原因。
      ``sell_threshold`` 在 regression 下不做限制（负数收益阈值合法）。
    - 未知 objective → 无规则，返回 ``[]``。

    Args:
        objective: 模型的输出类型，取 ``"classification"``、
            ``"path_class"``、``"regression"`` 或 ``None``（旧版无该字段）。
        buy_threshold: 触发买入信号所需的阈值。
        sell_threshold: 触发卖出信号所需的阈值；``None`` 表示不设卖出阈值，
            跳过 sell 方向的校验。

    Returns:
        违规原因的字符串列表，每条说明哪个字段、实际值、应满足什么约束。
        空列表表示所有校验通过。

    Example:
        >>> threshold_scale_check("classification", 0.6, 0.4)
        []
        >>> threshold_scale_check("classification", 1.5)
        ['classification 模型 buy_threshold=1.5 应在 [0,1]（概率口径）']
        >>> threshold_scale_check("regression", 0.005, -0.005)
        []
        >>> threshold_scale_check("regression", 0.6)
        ['regression 模型 buy_threshold=0.6 在收益口径下意为 +60% 收益，疑似误用概率阈值']
        >>> threshold_scale_check(None, 999.0)
        []
    """
    if not objective:
        return []

    reasons: list[str] = []

    if objective in _PROB_OBJECTIVES:
        if not (0.0 <= buy_threshold <= 1.0):
            reasons.append(
                f"{objective} 模型 buy_threshold={buy_threshold} 应在 [0,1]（概率口径）"
            )
        if sell_threshold is not None and not (0.0 <= sell_threshold <= 1.0):
            reasons.append(
                f"{objective} 模型 sell_threshold={sell_threshold} 应在 [0,1]（概率口径）"
            )
    elif objective == "regression":
        if buy_threshold >= 0.5:
            reasons.append(
                f"regression 模型 buy_threshold={buy_threshold} "
                f"在收益口径下意为 +{buy_threshold:.0%} 收益，疑似误用概率阈值"
            )

    return reasons
