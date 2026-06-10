"""Profile_Artifact JSON 持久化。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from aitrade.config import PROFILE_PATH
from aitrade.profiling.types import SymbolProfile


class ProfileStore:
    """仅在 PROFILE_PATH 下保存 / 读取画像产物。"""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path = Path(base_path) if base_path else PROFILE_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "profile"

    def artifact_id_for(self, profile: SymbolProfile) -> str:
        as_of = profile.input.as_of.strftime("%Y%m%dT%H%M%S")
        symbol = self._sanitize(profile.input.vt_symbol)
        interval = self._sanitize(profile.input.interval)
        return f"{symbol}__{interval}__{as_of}"

    def _path(self, artifact_id: str) -> Path:
        safe_id = self._sanitize(artifact_id)
        return self.base_path / f"{safe_id}.json"

    def save(self, profile: SymbolProfile) -> str:
        artifact_id = self.artifact_id_for(profile)
        profile.artifact_id = artifact_id
        path = self._path(artifact_id)
        payload = profile.model_dump(mode="json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return artifact_id

    def load(self, artifact_id: str) -> SymbolProfile:
        path = self._path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"画像产物不存在：{artifact_id}")
        return SymbolProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        return sorted(path.stem for path in self.base_path.glob("*.json"))
