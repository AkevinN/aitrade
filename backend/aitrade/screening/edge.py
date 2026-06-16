"""
CNN 选股 Tier-2 绝对 edge 门禁（edge.py）。

本文件只做一件事：从 ``run_walk_forward_evaluate`` 返回的 WF 报告中，
**纯由各折的 candidate_score（跨种子均值）派生** 绝对 edge 结论，
写入 ``Tier2Verdict``，完全不读取相对晋级门禁字段
``summary.passed`` / ``summary.avg_score_delta``（Requirement 5.2 / 5.3）。

设计说明
---------
- ``summary.passed`` 是"新候选 vs 生产模型"的相对门禁；选股场景每只股票
  都没有对应生产模型，因此 ``passed`` 恒为 False，不可用于绝对判断。
- 绝对判据：平均 candidate_score > 0 **且** 正分折占比 ≥ 阈值
  （``ScreeningRules.min_positive_fold_ratio``，默认 0.5）。
- 纯函数：无 I/O、无副作用，可直接被属性测试覆盖（Property 7）。
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from .rules import ScreeningRules
from .types import Tier2Verdict


def derive_edge(wf_report: dict[str, Any], rules: ScreeningRules) -> Tier2Verdict:
    """从 WF/OOS 报告派生绝对 edge 结论，不依赖相对晋级门禁 summary.passed。

    绝对判据（Requirement 5.2）：
      - ``avg = mean(candidate_scores)``（各折跨种子均值的平均）
      - ``pos_ratio = 折中 candidate_score > 0 的占比``
      - ``edge_ok = (avg > 0) and (pos_ratio >= rules.min_positive_fold_ratio)``

    折数为空时返回 ``Tier2Verdict(evaluable=False)``，不输出任何数值字段
    （Requirement 5.4）。

    ``avg_cross_seed_std`` 优先取 ``summary.avg_cross_seed_std``（如存在），
    其次对各折 ``cross_seed.std`` 取均值，均无则为 ``None``。

    Args:
        wf_report: ``run_walk_forward_evaluate`` 的返回字典，含 ``folds``、
            ``summary``、``report_id``、``request`` 等键。缺失键均以 ``.get``
            安全访问，不会抛出 KeyError。
        rules: 选股规则；读取 ``min_positive_fold_ratio`` 阈值。

    Returns:
        ``Tier2Verdict``，字段含义：

        - ``vt_symbol``: 从 ``wf_report["request"]["target_symbol"]`` 读取；
          ``request`` 缺失时为空字符串。
        - ``evaluable``: ``folds`` 非空且每折 ``candidate_score`` 均有效时为 ``True``。
        - ``edge_ok``: 绝对 edge 结论；``evaluable=False`` 时恒 ``False``。
        - ``avg_score``: 各折 ``candidate_score`` 的算术平均；``evaluable=False`` 时为 ``None``。
        - ``pos_fold_ratio``: 正分折占比 ``∈ [0, 1]``；``evaluable=False`` 时为 ``None``。
        - ``avg_cross_seed_std``: 跨种子得分标准差均值；无多种子数据时为 ``None``。
        - ``report_id``: ``wf_report.get("report_id")``，用于回读完整报告。
        - ``note``: 不可评估时的原因说明；``evaluable=True`` 时为 ``None``。

    Example:
        >>> from aitrade.screening.rules import DEFAULT_SCREENING_RULES
        >>> report = {
        ...     "report_id": "wf_test_001",
        ...     "request": {"target_symbol": "000001.SZSE"},
        ...     "folds": [
        ...         {"candidate_score": 2.5, "cross_seed": {"std": 0.1}},
        ...         {"candidate_score": 1.0, "cross_seed": {"std": 0.2}},
        ...     ],
        ...     "summary": {"avg_cross_seed_std": 0.15},
        ... }
        >>> verdict = derive_edge(report, DEFAULT_SCREENING_RULES)
        >>> assert verdict.evaluable is True
        >>> assert verdict.edge_ok is True
    """
    report_id: str | None = wf_report.get("report_id")
    request: dict[str, Any] = wf_report.get("request") or {}
    vt_symbol: str = str(request.get("target_symbol") or "")

    folds: list[dict[str, Any]] = wf_report.get("folds") or []

    # --- 提取各折 candidate_score（跨种子均值），过滤掉 None ---
    raw_scores: list[float] = []
    for fold in folds:
        score = fold.get("candidate_score")
        if score is not None:
            raw_scores.append(float(score))

    # --- 折数为空（或全部 None）→ 不可评估 ---
    if not raw_scores:
        return Tier2Verdict(
            vt_symbol=vt_symbol,
            evaluable=False,
            edge_ok=False,
            report_id=report_id,
            note="无可用折，无法评估",
        )

    # --- 派生绝对 edge 判据 ---
    avg: float = mean(raw_scores)
    pos_ratio: float = sum(1 for s in raw_scores if s > 0) / len(raw_scores)
    edge_ok: bool = (avg > 0) and (pos_ratio >= rules.min_positive_fold_ratio)

    # --- avg_cross_seed_std：优先读 summary，其次折内均值 ---
    avg_cross_seed_std: float | None = None
    summary: dict[str, Any] = wf_report.get("summary") or {}
    summary_std = summary.get("avg_cross_seed_std")
    if summary_std is not None:
        avg_cross_seed_std = float(summary_std)
    else:
        fold_stds: list[float] = []
        for fold in folds:
            cross_seed = fold.get("cross_seed") or {}
            std_val = cross_seed.get("std")
            if std_val is not None:
                fold_stds.append(float(std_val))
        if fold_stds:
            avg_cross_seed_std = mean(fold_stds)

    return Tier2Verdict(
        vt_symbol=vt_symbol,
        evaluable=True,
        edge_ok=edge_ok,
        avg_score=avg,
        pos_fold_ratio=pos_ratio,
        avg_cross_seed_std=avg_cross_seed_std,
        report_id=report_id,
        note=None,
    )
