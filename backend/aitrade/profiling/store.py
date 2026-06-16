"""Profile_Artifact JSON 持久化：将 SymbolProfile 对象存储到 PROFILE_PATH。

每个画像产物以 ``{vt_symbol}__{interval}__{as_of}.json`` 命名，
特殊字符被替换为下划线以保证文件名安全（_sanitize）。
本模块是画像模块中唯一允许写入文件系统的位置（Requirement 8.4）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from aitrade.config import PROFILE_PATH
from aitrade.profiling.types import SymbolProfile


class ProfileStore:
    """画像产物 JSON 持久化：保存/读取/列举 PROFILE_PATH 下的 SymbolProfile 文件。"""

    def __init__(self, base_path: Path | None = None) -> None:
        """初始化存储，并确保根目录存在。

        Args:
            base_path: 画像产物根目录；None 时使用 config.PROFILE_PATH。
        """
        self.base_path = Path(base_path) if base_path else PROFILE_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, value: str) -> str:
        """将字符串中的非法文件名字符替换为下划线（防止路径注入）。

        Args:
            value: 原始字符串（如 vt_symbol 或 interval）。

        Returns:
            仅含 ``[A-Za-z0-9._-]`` 的安全文件名片段；空结果回退为 ``"profile"``。
        """
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "profile"

    def artifact_id_for(self, profile: SymbolProfile) -> str:
        """生成画像产物的唯一 artifact_id。

        格式：``{vt_symbol}__{interval}__{as_of_yyyymmddTHHMMSS}``，
        特殊字符经 _sanitize 处理以保证文件名安全。

        Args:
            profile: SymbolProfile 对象（读取其 input 字段）。

        Returns:
            artifact_id 字符串，如 ``"000001_SZSE__d__20250101T000000"``。
        """
        as_of = profile.input.as_of.strftime("%Y%m%dT%H%M%S")
        symbol = self._sanitize(profile.input.vt_symbol)
        interval = self._sanitize(profile.input.interval)
        return f"{symbol}__{interval}__{as_of}"

    def _path(self, artifact_id: str) -> Path:
        """返回 artifact_id 对应的 JSON 文件路径。

        Args:
            artifact_id: 画像产物唯一 ID。

        Returns:
            完整文件路径 ``{base_path}/{sanitized_id}.json``。
        """
        safe_id = self._sanitize(artifact_id)
        return self.base_path / f"{safe_id}.json"

    def save(self, profile: SymbolProfile) -> str:
        """将 SymbolProfile 序列化为 JSON 并写入磁盘，同时写回 artifact_id。

        Args:
            profile: 待持久化的 SymbolProfile 对象（会原地写入 artifact_id 属性）。

        Returns:
            写入后的 artifact_id 字符串。
        """
        artifact_id = self.artifact_id_for(profile)
        profile.artifact_id = artifact_id
        path = self._path(artifact_id)
        payload = profile.model_dump(mode="json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return artifact_id

    def load(self, artifact_id: str) -> SymbolProfile:
        """从磁盘加载并反序列化 SymbolProfile。

        Args:
            artifact_id: 目标画像产物 ID。

        Returns:
            反序列化后的 SymbolProfile 对象。

        Raises:
            FileNotFoundError: 对应 JSON 文件不存在时抛出。
        """
        path = self._path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"画像产物不存在：{artifact_id}")
        return SymbolProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        """返回 base_path 下所有画像产物 ID 的升序列表。

        Returns:
            所有 .json 文件的 stem（不含后缀）升序列表。
        """
        return sorted(path.stem for path in self.base_path.glob("*.json"))
