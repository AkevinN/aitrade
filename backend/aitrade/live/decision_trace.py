"""
决策过程可观测性（Requirement 8）：逐段累积六段 Trace_Section 并驱动标准 logging。

两层可观测性：
- 标准 ``logging``：六段里程碑 INFO / 明细 DEBUG，每行带 ``run_id`` 前缀，用于实时排障。
- 可持久化、可重启回溯的 ``Decision_Trace``（``{signal_id}.trace.json``，sibling 于既有
  ``{signal_id}.json``）。

本模块的 logger 约定为 ``logging.getLogger("aitrade.live.orchestrator")``；
``TraceBuilder`` 通过构造函数注入 logger，因此此处仅存储引用。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class TraceBuilder:
    """逐段累积六段 Trace_Section，并驱动标准 logging（INFO 里程碑 / DEBUG 明细）。"""

    SECTIONS = ("run_header", "inference", "pricing",
                "decision_logic", "risk", "result")

    def __init__(self, run_id: str, signal_id: str, logger) -> None:
        """初始化 TraceBuilder。

        Args:
            run_id:    本次运行的短码（8 位 hex），用于 logger 前缀与 trace 键。
            signal_id: 与本次决策对应的幂等键（允许在实际 Decision_Bar 确定后更新）。
            logger:    注入的 logger（约定为 logging.getLogger("aitrade.live.orchestrator")）。
        """
        self.run_id = run_id
        self.signal_id = signal_id
        self._logger = logger
        self._sections: dict[str, Any] = {}
        self._completed: list[str] = []      # 已完成段（顺序）

    def set_section(self, name: str, payload: dict, *, debug_detail: dict | None = None) -> None:
        """记录一个已完成的 Trace_Section 并写日志。

        name 必须在 SECTIONS 中；每段只应调用一次（重复调用会覆盖已有值）。

        Args:
            name:         段名，须在 SECTIONS 元组中。
            payload:      段内容 dict（脱敏，不含凭证）。
            debug_detail: 可选的重负载明细（如逐点信号序列），以 DEBUG 级别输出，
                          不写入持久化 trace（减少磁盘占用）。
        """
        assert name in self.SECTIONS
        self._sections[name] = payload
        self._completed.append(name)
        self._logger.info("[%s] 段完成: %s", self.run_id, name)          # 里程碑 INFO
        if debug_detail is not None:
            self._logger.debug("[%s] %s 明细: %s", self.run_id, name, debug_detail)  # 重负载 DEBUG

    def to_trace(self, *, schema_version: int = 1,
                 trace_persisted: bool = True,
                 trace_persist_error: str | None = None) -> dict:
        """将累积的 trace 内容序列化为可持久化 dict。

        Args:
            schema_version:     trace schema 版本，供后续字段演进，默认 1。
            trace_persisted:    是否持久化成功（result 段回填用，已废弃在此传入，保留参数兼容）。
            trace_persist_error: 持久化错误信息，None 表示无错误。

        Returns:
            含 schema_version / run_id / signal_id / completed_sections / sections 的 dict。
        """
        return {
            "schema_version": schema_version,
            "run_id": self.run_id,
            "signal_id": self.signal_id,
            "completed_sections": list(self._completed),
            "sections": dict(self._sections),
        }


class DecisionTraceStore:
    """Decision_Trace 的 JSON 持久化（每 signal_id 一文件，sibling 于 {signal_id}.json）。

    写入 ``{safe}.trace.json``，与既有 ``DecisionStore`` 共用相同的 signal_id 安全化规则，
    但绝不触碰 ``{signal_id}.json``（满足 8.3）。``save_if_absent`` 幂等：已存在则不重写（满足 8.9）。
    """

    def __init__(self, base_path: Path | str) -> None:
        """初始化 DecisionTraceStore。

        Args:
            base_path: trace 文件存放目录（与 DecisionStore 共用同一目录）；不存在时自动创建。
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, signal_id: str) -> Path:
        """将 signal_id 安全化后返回对应 .trace.json 文件路径。

        Args:
            signal_id: 幂等键，"/" 与 ":" 替换为 "_"。

        Returns:
            .trace.json 文件路径（不保证文件存在）。
        """
        safe = signal_id.replace("/", "_").replace(":", "_")
        return self.base_path / f"{safe}.trace.json"

    def exists(self, signal_id: str) -> bool:
        """判断该 signal_id 是否已有 trace 文件。

        Args:
            signal_id: 幂等键。

        Returns:
            True 表示 .trace.json 存在。
        """
        return self._path(signal_id).exists()

    def get(self, signal_id: str) -> Optional[dict]:
        """读取指定 signal_id 的 trace dict；不存在返回 None。

        Args:
            signal_id: 幂等键。

        Returns:
            trace dict；文件不存在时返回 None。
        """
        path = self._path(signal_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_if_absent(self, signal_id: str, trace: dict) -> bool:
        path = self._path(signal_id)
        if path.exists():
            return False
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def archive(self, signal_id: str) -> Optional[Path]:
        """归档式删除：trace 文件移入 archive/ 子目录（文件名追加时间戳）。

        必须与 `DecisionStore.archive` 成对调用——只删决策不删 trace 会使重新决策后
        `save_if_absent` 因旧文件存在而不写，造成「决策是新的、档案是旧的」错位。
        归档文件保留审计痕迹。不存在则返回 None。
        """
        path = self._path(signal_id)
        if not path.exists():
            return None
        archive_dir = self.base_path / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        target = archive_dir / f"{path.stem}.{stamp}{path.suffix}"
        path.rename(target)
        return target
