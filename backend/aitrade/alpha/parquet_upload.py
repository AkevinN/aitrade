"""Parquet 上传暂存会话：流式落盘、会话目录管理、TTL 清理。

数据准备页「文件夹批量上传 parquet」分两步：先把上传的文件**流式分块落盘**到
``base_dir/<session_id>/``（内存恒定，与文件大小无关），返回逐文件元信息供预览；
用户确认后，异步任务就同一会话逐文件导入为待合并批次，成功后 ``discard`` 删除会话目录。
新会话创建前由调用方触发 ``cleanup_expired`` 回收超时的历史会话，避免暂存无限增长。

本模块只做本地文件 I/O，不依赖 FastAPI，便于直接单元/属性测试。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class StagedFile:
    """暂存目录中的单个已落盘文件。

    Attributes:
        file_name: 去除任何路径成分后的纯文件名（含扩展名）。
        path: 落盘后的绝对/相对路径。
        size_bytes: 实际写入字节数。
        is_parquet: 文件名是否以 ``.parquet`` 结尾（大小写不敏感）。
    """

    file_name: str
    path: Path
    size_bytes: int
    is_parquet: bool


class ParquetUploadStaging:
    """管理 parquet 上传暂存会话的本地文件存储。

    每个上传会话对应 ``base_dir/<session_id>/`` 一个子目录，存放本次上传的原始文件。
    """

    def __init__(self, base_dir: Path | str) -> None:
        """初始化暂存存储。

        Args:
            base_dir: 暂存根目录；不存在时自动创建。
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        """返回某会话的暂存子目录路径（不创建）。

        Args:
            session_id: 会话标识。

        Returns:
            ``base_dir/<session_id>`` 路径。
        """
        return self.base_dir / session_id

    def stage_stream(
        self,
        session_id: str,
        file_name: str,
        stream: BinaryIO,
        *,
        max_file_bytes: int,
        chunk_bytes: int = 1024 * 1024,
    ) -> StagedFile:
        """把上传流分块写入会话目录，内存占用恒定（仅 ``chunk_bytes``）。

        逐块从 ``stream`` 读出写入目标文件；累计写入超过 ``max_file_bytes`` 时抛
        ``ValueError`` 并删除半截文件，避免坏数据残留与内存溢出。文件名中的任何路径
        成分会被剥离（防目录穿越）。

        Args:
            session_id: 会话标识；目录不存在时自动创建。
            file_name: 原始文件名（仅保留 basename）。
            stream: 可读二进制流（如 ``UploadFile.file``）。
            max_file_bytes: 单文件大小上限（字节），超出即抛错。
            chunk_bytes: 每次读取/写入的块大小，决定常驻内存上限，默认 1MB。

        Returns:
            落盘结果 ``StagedFile``。

        Raises:
            ValueError: 累计写入超过 ``max_file_bytes`` 时抛出。
        """
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file_name).name
        dest = session_dir / safe_name
        is_parquet = safe_name.lower().endswith(".parquet")

        size = 0
        try:
            with open(dest, "wb") as out:
                while True:
                    chunk = stream.read(chunk_bytes)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_file_bytes:
                        raise ValueError(f"文件 {safe_name} 超过单文件大小上限 {max_file_bytes} 字节")
                    out.write(chunk)
        except Exception:
            dest.unlink(missing_ok=True)
            raise

        return StagedFile(file_name=safe_name, path=dest, size_bytes=size, is_parquet=is_parquet)

    def list_files(self, session_id: str) -> list[StagedFile]:
        """列出某会话已暂存的全部文件（按文件名排序）。

        Args:
            session_id: 会话标识。

        Returns:
            ``StagedFile`` 列表；会话目录不存在时返回空列表。
        """
        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            return []
        files: list[StagedFile] = []
        for path in sorted(session_dir.iterdir()):
            if path.is_file():
                files.append(
                    StagedFile(
                        file_name=path.name,
                        path=path,
                        size_bytes=path.stat().st_size,
                        is_parquet=path.suffix.lower() == ".parquet",
                    )
                )
        return files

    def discard(self, session_id: str) -> None:
        """删除整个会话目录（导入成功或用户取消时调用）。

        Args:
            session_id: 会话标识；目录不存在时静默跳过。
        """
        shutil.rmtree(self._session_dir(session_id), ignore_errors=True)

    def cleanup_expired(self, ttl_seconds: int, *, now: float | None = None) -> int:
        """回收 ``base_dir`` 下最近修改时间超过 TTL 的历史会话目录。

        Args:
            ttl_seconds: 会话存活时长（秒）；目录 mtime 距 ``now`` 超过该值即回收。
            now: 当前时间戳（秒）；``None`` 时取 ``time.time()``。供测试注入以保证确定性。

        Returns:
            被回收的会话目录数。
        """
        if not self.base_dir.is_dir():
            return 0
        current = time.time() if now is None else now
        removed = 0
        for entry in self.base_dir.iterdir():
            if entry.is_dir() and (current - entry.stat().st_mtime) > ttl_seconds:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed
