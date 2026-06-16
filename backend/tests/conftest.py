"""backend/tests 全局存储隔离：把 AITRADE_HOME 重定向到临时目录。

aitrade/config.py 的所有存储路径（TASK_HISTORY_PATH、CNN_MODEL_PATH、
ALPHA_LAB_PATH 等十余个）都是模块级常量，在 import 时从 AITRADE_HOME
派生求值并立即 mkdir；下游模块又以 ``from ..config import X`` 的形式在
各自 import 时把这些常量固化成本模块副本。

因此隔离必须发生在任何 aitrade 模块被 import 之前——而 pytest 在收集
阶段就会 import 全部测试模块（连带 import aitrade）。任何 fixture
（包括 session 级 autouse）都在收集之后才执行，届时改环境变量已经无效。

conftest.py 是 pytest 在收集前最先 import 的文件：在本模块顶层改写
``AITRADE_HOME`` 环境变量，aitrade.config 首次求值时即拿到隔离目录，
全部派生常量与下游模块的 import 时副本随之指向隔离目录，无需逐模块
monkeypatch。隔离效果由 tests/test_storage_isolation.py 持续校验，
本文件的 session 级看门狗 fixture 兜底检测任何真实落盘。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 模块顶层：必须先于任何 aitrade import 执行（理由见模块 docstring）
# ---------------------------------------------------------------------------

if "aitrade.config" in sys.modules:
    raise RuntimeError(
        "aitrade.config 在 conftest.py 之前已被 import，存储隔离失效："
        "config 的路径常量已按真实 AITRADE_HOME 求值并创建目录。"
        "请检查是否有 pytest 插件或更早的 conftest 提前 import 了 aitrade。"
    )

_ISOLATED_HOME = tempfile.mkdtemp(prefix="aitrade-pytest-home-")
os.environ["AITRADE_HOME"] = _ISOLATED_HOME


def _snapshot(root: Path) -> set[tuple[str, int]]:
    """对目录做 (文件路径, 文件大小) 快照，用于检测新增与追加写入。

    记录大小而非仅路径，是为了捕获对已有文件的追加（如往已存在的
    task_history/*.jsonl 里 append 记录——这正是历史上真实发生过的
    污染形态，仅对比文件名会漏检）。

    Args:
        root: 要快照的目录；不存在时视为空目录。

    Returns:
        {(绝对路径字符串, 字节大小)} 集合；root 不存在时返回空集合。
    """
    if not root.is_dir():
        return set()
    return {
        (str(p), p.stat().st_size)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def _real_storage_watchdog():
    """session 级看门狗：整个测试会话结束后断言真实数据目录零变化。

    在会话开始时对两处真实数据目录（项目根 .aitrade 与用户主目录
    .aitrade）做快照，会话结束时复查；若出现新增文件或已有文件被
    追加写入，直接判会话失败。这是对环境变量重定向的兜底——即使
    将来有代码绕过 config 硬编码真实路径，也会在 CI 里立刻暴露。

    Yields:
        None。本 fixture 只做前后置校验，不向测试提供值。
    """
    from aitrade.config import PROJECT_ROOT

    real_dirs = [PROJECT_ROOT / ".aitrade", Path.home() / ".aitrade"]
    before = {d: _snapshot(d) for d in real_dirs}
    yield
    for d, old in before.items():
        leaked = _snapshot(d) - old
        assert not leaked, (
            f"测试套件向真实数据目录 {d} 落盘了 {len(leaked)} 处变化"
            f"（新增或追加），样例: {sorted(leaked)[:5]}"
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """测试会话收尾：删除本次会话创建的隔离存储目录。

    Args:
        session: pytest 会话对象（未使用，钩子签名要求）。
        exitstatus: 会话退出码（未使用，钩子签名要求）。
    """
    shutil.rmtree(_ISOLATED_HOME, ignore_errors=True)
