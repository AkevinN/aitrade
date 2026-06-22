"""选股产物 JSON 持久化与隔离治理 store 工厂。

``ScreeningStore`` 负责将 :class:`ScreeningResult` 对象序列化为 JSON 文件，
写入 ``SCREENING_PATH``；文件名以 ``run_id`` 的净化版本命名，规则与 ``ProfileStore``
保持一致（仅保留 ``[A-Za-z0-9._-]``）。

``build_screening_governance_store`` 构造一个根目录指向 ``SCREENING_GOVERNANCE_PATH``
的 :class:`CNNGovernanceStore`，使 Tier-2 WF/OOS 评估产物（报告/历史/候选）
落入专属目录，与生产 ``CNN_GOVERNANCE_PATH`` 完全隔离（Requirement 10.2）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from aitrade import config
from aitrade.cnn.governance import CNNGovernanceStore
from aitrade.screening.types import ScreeningResult


class ScreeningStore:
    """选股产物 JSON 持久化：保存/读取/列举 SCREENING_PATH 下的 ScreeningResult 文件。

    文件名格式为 ``{sanitized_run_id}.json``，非法字符经 :meth:`_sanitize`
    替换为下划线，保证跨平台文件名安全（防路径注入）。
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """初始化存储，并确保根目录存在。

        Args:
            base_path: 选股产物根目录；``None`` 时使用 ``config.SCREENING_PATH``。
        """
        self.base_path = Path(base_path) if base_path is not None else config.SCREENING_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, value: str) -> str:
        """将字符串中的非法文件名字符替换为下划线（防止路径注入）。

        Args:
            value: 原始字符串（如 ``run_id``）。

        Returns:
            仅含 ``[A-Za-z0-9._-]`` 的安全文件名片段；空结果回退为 ``"result"``。

        Example:
            >>> store._sanitize("run/2025-01:01")
            'run_2025-01_01'
        """
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "result"

    def _path(self, run_id: str) -> Path:
        """返回 run_id 对应的 JSON 文件路径。

        Args:
            run_id: 选股运行唯一 ID（未净化）。

        Returns:
            完整文件路径 ``{base_path}/{sanitized_run_id}.json``。
        """
        return self.base_path / f"{self._sanitize(run_id)}.json"

    def save(self, result: ScreeningResult) -> str:
        """将 ScreeningResult 序列化为 JSON 并写入磁盘。

        使用 ``model_dump(mode="json")`` 确保 datetime 等类型均序列化为
        JSON 原生类型（字符串），与 Pydantic v2 惯例一致。

        Args:
            result: 待持久化的 :class:`ScreeningResult` 对象。
                    ``result.run_id`` 为空时为调用方错误；方法对空值做防御性降级，
                    将文件命名为 ``"result.json"``，但不抛出异常。

        Returns:
            写入对应文件时所用的 ``run_id``（即 ``result.run_id``）。

        Raises:
            OSError: 目标路径不可写入时抛出（由底层文件系统抛出）。

        Example:
            >>> from datetime import datetime
            >>> r = ScreeningResult(run_id="run_001", created_at=datetime.now(),
            ...                     rules_id="v1", universe_size=10)
            >>> store.save(r)
            'run_001'
        """
        path = self._path(result.run_id)
        payload = result.model_dump(mode="json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return result.run_id

    def load(self, run_id: str) -> ScreeningResult:
        """从磁盘加载并反序列化 ScreeningResult。

        Args:
            run_id: 目标选股运行 ID（与 ``save`` 时传入的 ``result.run_id`` 相同）。

        Returns:
            反序列化后的 :class:`ScreeningResult` 对象。

        Raises:
            FileNotFoundError: 对应 JSON 文件不存在时抛出，错误信息包含 run_id。

        Example:
            >>> result = store.load("run_001")
            >>> result.status
            'draft'
        """
        path = self._path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"选股产物不存在：run_id={run_id!r}，路径={path}")
        return ScreeningResult.model_validate_json(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        """返回 base_path 下所有选股产物 run_id 的升序列表。

        列举 ``*.json`` 文件的 stem（不含后缀）并升序排序；目录为空时返回 ``[]``。

        Returns:
            所有 ``.json`` 文件的 stem 升序列表（即净化后的 run_id）。

        Example:
            >>> store.list_ids()
            ['run_001', 'run_002']
        """
        return sorted(path.stem for path in self.base_path.glob("*.json"))

    def delete(self, run_id: str) -> bool:
        """删除指定 run_id 的选股产物 JSON（用于"运行历史"级联清理）。

        Args:
            run_id: 目标选股运行 ID（与 ``save`` 时的 ``result.run_id`` 相同）。

        Returns:
            True 表示文件存在并已删除；False 表示本不存在（幂等）。
        """
        path = self._path(run_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def build_screening_governance_store() -> CNNGovernanceStore:
    """构造写入 SCREENING_GOVERNANCE_PATH 的隔离治理 store。

    返回的 :class:`CNNGovernanceStore` 根目录指向 ``config.SCREENING_GOVERNANCE_PATH``，
    使 Tier-2 WF/OOS 产物（报告、候选历史、回放报告）均落入专属的 screening 隔离目录，
    绝不污染生产 ``CNN_GOVERNANCE_PATH``（Requirement 10.2/10.5）。

    每次调用均构造新实例（轻量，不持共享状态），调用方可在选股 Runner 启动时获取
    并通过 ``run_walk_forward_evaluate(store=...)`` 注入（design.md §5）。

    Returns:
        根目录为 ``SCREENING_GOVERNANCE_PATH`` 的 :class:`CNNGovernanceStore`。

    Example:
        >>> gov_store = build_screening_governance_store()
        >>> gov_store.root == config.SCREENING_GOVERNANCE_PATH
        True
    """
    return CNNGovernanceStore(root=config.SCREENING_GOVERNANCE_PATH)
