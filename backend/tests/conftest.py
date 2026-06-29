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
本文件的 session 级看门狗 fixture 兜底检测**本进程**对真实目录的任何写入
（基于 ``open`` 审计钩子，详见 ``_real_storage_watchdog``）。
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

# ---------------------------------------------------------------------------
# 真实数据目录写入看门狗（进程内 ``open`` 审计钩子）
# ---------------------------------------------------------------------------
# 真实（未隔离）数据目录：项目根 .aitrade 与用户主目录 .aitrade。
# PROJECT_ROOT 直接由 conftest 自身位置推导（backend/tests/conftest.py →
# 上三级即项目根），避免在此处 import aitrade.config——只需路径常量。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REAL_DATA_DIRS: tuple[str, ...] = (
    os.path.normpath(_PROJECT_ROOT / ".aitrade"),
    os.path.normpath(Path.home() / ".aitrade"),
)

# 本进程内捕获到的真实目录写入路径集合（审计钩子填充，看门狗 fixture 校验）。
_REAL_WRITES: set[str] = set()

# os.open 形态下（mode 为 None）据 flags 判定写意图的标志位掩码。
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _is_write_open(mode: object, flags: object) -> bool:
    """判断一次 ``open`` 审计事件是否带写入意图。

    CPython 的 ``open`` 审计事件对 builtin ``open``/``io.open`` 给出字符串
    ``mode``（"r"/"a"/"w"/"r+"…），对 ``os.open`` 给出 ``mode=None`` 并把
    打开标志放在整数 ``flags`` 里。两种形态分别判定，纯读取不算写入。

    Args:
        mode: 审计事件第二参数；字符串模式（builtin open）或 None（os.open）。
        flags: 审计事件第三参数；os.open 的整数打开标志，其它情况忽略。

    Returns:
        写入意图为真返回 True；纯读取（"r" / O_RDONLY）返回 False。
    """
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    return bool((int(flags) if isinstance(flags, int) else 0) & _WRITE_FLAGS)


def _audit_open(event: str, args: tuple) -> None:
    """``sys.addaudithook`` 回调：记录**本进程**对真实数据目录的写入式 open。

    只关心写意图的 ``open`` 事件；命中真实数据目录前缀时把规整后的绝对
    路径收入模块级 ``_REAL_WRITES``。审计钩子内抛出的异常会污染被审计的
    ``open`` 调用本身，故全程吞异常、绝不向外传播。

    与旧版"会话前后对真实目录做文件快照差分"相比，本钩子只看本进程发起
    的 open，天然忽略同机并发进程（如本地 ``uvicorn`` 调度线程持续往
    ``.aitrade/live/scheduler_runs`` 追加日志）的落盘，从根上消除误报。

    Args:
        event: 审计事件名；只处理 "open"。
        args: 事件参数元组，对 "open" 为 ``(path, mode, flags)``。
    """
    if event != "open":
        return
    try:
        path = args[0]
        if not isinstance(path, (str, bytes)):  # 经 fd 打开等情况，path 非路径
            return
        if not _is_write_open(args[1], args[2]):
            return
        abspath = os.path.normpath(os.path.abspath(os.fsdecode(path)))
        for real in _REAL_DATA_DIRS:
            if abspath == real or abspath.startswith(real + os.sep):
                _REAL_WRITES.add(abspath)
                return
    except Exception:  # noqa: BLE001 审计钩子绝不能抛出
        return


sys.addaudithook(_audit_open)


@pytest.fixture(scope="session", autouse=True)
def _real_storage_watchdog():
    """session 级看门狗：断言**本测试进程**从未向真实数据目录写入。

    依赖模块顶层安装的 ``open`` 审计钩子 ``_audit_open``：它只捕获本进程
    发起的写入式 open，因此与同机并发运行的真实服务完全隔离——历史上正
    是这种并发进程（本地 ``uvicorn`` 后台调度线程往 ``.aitrade/live/
    scheduler_runs`` 每 30s 追加一行）的落盘，被旧版"目录前后快照差分"
    误判为测试污染，并把 session 级 teardown 失败归因到最后一个被收集的
    用例（如 test_list_data_resources_uses_canonical_symbol）。

    覆盖范围：本进程经 Python 层 ``open`` 的写入/追加/重写（即历史真实泄漏
    形态——JSONL/JSON 存储的 append 与 delete_where 重写均走 ``open``）。
    已知局限：经 Rust/polars 等绕过 CPython ``open`` 的写入不被本钩子捕获；
    config 派生路径的静态隔离保证由 test_storage_isolation.py 兜底。

    Yields:
        None。本 fixture 只做会话末校验，不向测试提供值。
    """
    yield
    assert not _REAL_WRITES, (
        f"测试进程向真实数据目录写入了 {len(_REAL_WRITES)} 个文件"
        f"（存储隔离被绕过，非并发进程所致），样例: {sorted(_REAL_WRITES)[:5]}"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """测试会话收尾：删除本次会话创建的隔离存储目录。

    Args:
        session: pytest 会话对象（未使用，钩子签名要求）。
        exitstatus: 会话退出码（未使用，钩子签名要求）。
    """
    shutil.rmtree(_ISOLATED_HOME, ignore_errors=True)
