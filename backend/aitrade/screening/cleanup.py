"""
选股运行的产物级联清理。

一次 CNN 选股的 Tier-2 会在隔离区落下不会自动清理的产物——每只入围标的一份 WF 报告
（``SCREENING_GOVERNANCE_PATH/reports/{report_id}.json``）、每折每种子一个临时候选模型
（``CNN_MODEL_DIR/{name}.pt`` + ``_history.json``，单次可达数十个），外加 ``persist=True``
时的 ``ScreeningResult``（``SCREENING_PATH/{run_id}.json``）。删一条历史选股运行时若只删任务
记录，这些产物会不断堆积占盘。本模块据 ``ScreeningResult`` 把它们一并清掉。

安全红线：只动**隔离的**选股区——选股治理 store 的 reports、选股产物区、以及报告里**逐字**
记录的 uuid 临时候选模型名；**绝不触碰生产治理产物**（``CNN_GOVERNANCE_PATH``）或用户自训
模型（临时模型名形如 ``screening_*_wf*_s*_<uuid>_<range>``，按精确名删除，不会误伤）。
全程 best-effort，单项失败仅 warning，不影响其余清理与任务记录删除。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _collect_model_names(report: dict[str, Any]) -> set[str]:
    """从一份 WF 报告收集其全部临时候选模型名（用于删 .pt 与 _history.json）。

    Args:
        report: ``run_walk_forward_evaluate`` 落盘的报告字典。

    Returns:
        去重后的模型名集合（不含 .pt 后缀）。
    """
    names: set[str] = set()
    for fold in report.get("folds", []) or []:
        for m in fold.get("candidate_models", []) or []:
            if m:
                names.add(str(m))
        rep = fold.get("candidate_model")
        if rep:
            names.add(str(rep))
    return names


def purge_screening_run(result: dict[str, Any]) -> dict[str, int]:
    """级联清理一次选股运行的隔离产物：WF 报告 + 其临时模型 + 落盘 ScreeningResult。

    Args:
        result: 该选股运行的 ``ScreeningResult`` 字典（来自 ``task.result``），含 ``run_id``
            与 ``leaderboard[].tier2.report_id``。非 dict 时直接返回零计数。

    Returns:
        计数字典 ``{"reports", "models", "result_files"}``，各为实际删除的数量。
    """
    from aitrade.cnn import delete_cnn_model
    from aitrade.screening.store import ScreeningStore, build_screening_governance_store

    counts = {"reports": 0, "models": 0, "result_files": 0}
    if not isinstance(result, dict):
        return counts

    gov_store = build_screening_governance_store()

    # 1) 收集本次运行的全部 WF 报告 id（来自榜单各行的 tier2.report_id）
    report_ids: list[str] = []
    for row in result.get("leaderboard", []) or []:
        if not isinstance(row, dict):
            continue
        t2 = row.get("tier2")
        rid = t2.get("report_id") if isinstance(t2, dict) else None
        if rid:
            report_ids.append(str(rid))

    # 2) 逐报告：先读出其临时候选模型并删（.pt + _history.json），再删报告文件本身
    for rid in report_ids:
        try:
            report = gov_store.get_report(rid)
            if report:
                for name in _collect_model_names(report):
                    try:
                        if delete_cnn_model(name):
                            counts["models"] += 1
                    except Exception as exc:  # noqa: BLE001 - 单模型清理失败不阻断
                        logger.warning("清理临时模型 %s 失败: %s", name, exc)
            if gov_store.delete_report(rid):
                counts["reports"] += 1
        except Exception as exc:  # noqa: BLE001 - 单报告清理失败不阻断
            logger.warning("清理 WF 报告 %s 失败: %s", rid, exc)

    # 3) 删落盘的 ScreeningResult（仅 persist=True 时存在）
    run_id = result.get("run_id")
    if run_id:
        try:
            if ScreeningStore().delete(str(run_id)):
                counts["result_files"] += 1
        except Exception as exc:  # noqa: BLE001 - 产物清理失败不阻断
            logger.warning("清理选股产物 %s 失败: %s", run_id, exc)

    return counts
