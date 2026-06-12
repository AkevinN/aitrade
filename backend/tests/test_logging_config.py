"""
日志基建测试（task-scheduler-observability 任务 7）。

覆盖（R7.1 / R7.2）：
- AITRADE_LOG_FILE 设置时 root 有 RotatingFileHandler，maxBytes/backupCount 正确
- AITRADE_LOG_FILE 未设置时无 RotatingFileHandler
- 重复调用 _configure_logging() 不重复加同路径 handler（幂等性）

注：dev.sh 的 50MB 超限归档改动标注「需用户手动」验证，
    不在此写自动化测试（shell 脚本行为，bash -n 语法检查已通过）。
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler


# ---------------------------------------------------------------------------
# Fixture：每个测试清理 root logger 的 RotatingFileHandler
# ---------------------------------------------------------------------------

def _remove_rotating_handlers() -> None:
    """清理 root logger 上所有 RotatingFileHandler，避免测试间污染。"""
    root = logging.getLogger()
    to_remove = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    for h in to_remove:
        h.close()
        root.removeHandler(h)


# ---------------------------------------------------------------------------
# 1. AITRADE_LOG_FILE 设置时 root 有 RotatingFileHandler，规格正确
# ---------------------------------------------------------------------------

def test_log_file_handler_added(tmp_path, monkeypatch):
    """AITRADE_LOG_FILE 设置时，root logger 有 RotatingFileHandler，规格正确。"""
    _remove_rotating_handlers()
    log_path = str(tmp_path / "test.log")
    monkeypatch.setenv("AITRADE_LOG_FILE", log_path)
    monkeypatch.delenv("AITRADE_LOG_LEVEL", raising=False)

    # 重新导入并调用（规避模块级已调用的缓存）
    from aitrade.main import _configure_logging
    _configure_logging()

    root = logging.getLogger()
    rfh_list = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert rfh_list, "AITRADE_LOG_FILE 设置后，root 应有 RotatingFileHandler"
    rfh = rfh_list[0]
    assert rfh.maxBytes == 20 * 1024 * 1024, f"maxBytes 应为 20MB，实际: {rfh.maxBytes}"
    assert rfh.backupCount == 5, f"backupCount 应为 5，实际: {rfh.backupCount}"

    _remove_rotating_handlers()


# ---------------------------------------------------------------------------
# 2. AITRADE_LOG_FILE 未设置时无 RotatingFileHandler
# ---------------------------------------------------------------------------

def test_no_file_handler_without_env(monkeypatch):
    """AITRADE_LOG_FILE 未设置时，root logger 无 RotatingFileHandler。"""
    _remove_rotating_handlers()
    monkeypatch.delenv("AITRADE_LOG_FILE", raising=False)

    from aitrade.main import _configure_logging
    _configure_logging()

    root = logging.getLogger()
    rfh_list = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert not rfh_list, f"AITRADE_LOG_FILE 未设置时不应有 RotatingFileHandler，实际: {rfh_list}"

    _remove_rotating_handlers()


# ---------------------------------------------------------------------------
# 3. 重复调用不重复加同路径 handler（幂等性）
# ---------------------------------------------------------------------------

def test_idempotent_repeated_calls(tmp_path, monkeypatch):
    """重复调用 _configure_logging() 同路径 handler 不重复添加。"""
    _remove_rotating_handlers()
    log_path = str(tmp_path / "idempotent.log")
    monkeypatch.setenv("AITRADE_LOG_FILE", log_path)

    from aitrade.main import _configure_logging
    _configure_logging()
    _configure_logging()
    _configure_logging()

    root = logging.getLogger()
    rfh_list = [
        h for h in root.handlers
        if isinstance(h, RotatingFileHandler)
        and os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(log_path)
    ]
    assert len(rfh_list) == 1, (
        f"重复调用后同路径 RotatingFileHandler 应恰好一个，实际: {len(rfh_list)}"
    )

    _remove_rotating_handlers()


# ---------------------------------------------------------------------------
# 4. AITRADE_LOG_LEVEL 影响级别（快速烟雾测试）
# ---------------------------------------------------------------------------

def test_log_level_from_env(monkeypatch):
    """AITRADE_LOG_LEVEL=DEBUG 时 root logger level 为 DEBUG（basicConfig 已无 handler 时生效）。"""
    _remove_rotating_handlers()
    monkeypatch.delenv("AITRADE_LOG_FILE", raising=False)
    monkeypatch.setenv("AITRADE_LOG_LEVEL", "DEBUG")

    # 移除所有 root handler 使 basicConfig 生效
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    for h in old_handlers:
        root.removeHandler(h)
    old_level = root.level
    root.setLevel(logging.NOTSET)  # 重置，让 basicConfig 可以设置

    try:
        from aitrade.main import _configure_logging
        _configure_logging()
        assert root.level == logging.DEBUG, (
            f"AITRADE_LOG_LEVEL=DEBUG 时 root level 应为 DEBUG，实际: {root.level}"
        )
    finally:
        # 恢复（避免污染其它测试）
        _remove_rotating_handlers()
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)
        monkeypatch.delenv("AITRADE_LOG_LEVEL", raising=False)
