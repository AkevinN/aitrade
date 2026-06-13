"""全局存储隔离不变量测试。

校验 conftest.py 的全局存储隔离机制生效：pytest 进程内
aitrade.config.AITRADE_HOME 及其全部派生路径常量都指向系统临时目录，
而非项目根目录下的真实 `.aitrade/`（或用户主目录），保证跑测试零真实落盘。

若本文件失败，说明隔离被破坏（如 conftest 被绕过、config 派生关系被改动、
或某模块在 conftest 之前提前 import 了 aitrade.config）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from aitrade import config


def _all_derived_path_constants() -> dict[str, Path]:
    """收集 config 模块中全部从 AITRADE_HOME 派生的路径常量。

    通过扫描 config 模块的全局命名空间，找出所有类型为 Path 且不等于
    PROJECT_ROOT / AITRADE_HOME 本身的常量。新增派生路径常量时无需修改
    本测试，会被自动纳入校验。

    Returns:
        {常量名: Path 值} 字典，仅含派生路径（不含 PROJECT_ROOT 与
        AITRADE_HOME 自身）。
    """
    return {
        name: value
        for name, value in vars(config).items()
        if isinstance(value, Path)
        and not name.startswith("_")
        and name not in ("PROJECT_ROOT", "AITRADE_HOME")
    }


def test_aitrade_home_redirected_to_tmp():
    """AITRADE_HOME 在 pytest 进程内必须指向系统临时目录。

    防回归点：若 conftest 的隔离失效，AITRADE_HOME 会落回真实的
    PROJECT_ROOT/.aitrade（或用户主目录下的 .aitrade），测试套件
    会把任务历史、模型产物等垃圾写进真实数据目录。
    """
    tmp_root = Path(tempfile.gettempdir()).resolve()
    home = config.AITRADE_HOME.resolve()

    assert home != (config.PROJECT_ROOT / ".aitrade").resolve(), (
        "AITRADE_HOME 指向了项目真实数据目录，存储隔离未生效"
    )
    assert home != (Path.home() / ".aitrade").resolve(), (
        "AITRADE_HOME 指向了用户主目录下的真实数据目录，存储隔离未生效"
    )
    assert home.is_relative_to(tmp_root), (
        f"AITRADE_HOME ({home}) 不在系统临时目录 ({tmp_root}) 下"
    )


def test_all_derived_paths_under_isolated_home():
    """config 中所有派生路径常量都必须落在隔离后的 AITRADE_HOME 下。

    防回归点：若有人在 config.py 里新增了不从 AITRADE_HOME 派生的
    绝对路径常量，本测试会立即暴露。
    """
    derived = _all_derived_path_constants()
    assert derived, "未在 config 中找到任何派生路径常量，扫描逻辑可能失效"

    offenders = {
        name: str(path)
        for name, path in derived.items()
        if not path.resolve().is_relative_to(config.AITRADE_HOME.resolve())
    }
    assert not offenders, f"以下路径常量逃逸出隔离目录: {offenders}"


def test_import_time_bound_constants_redirected():
    """在 import 时固化 config 常量的模块，其副本也必须已被重定向。

    防回归点：`from ..config import TASK_HISTORY_PATH` 这类绑定发生在
    模块 import 时；只有环境变量在任何 aitrade import 之前就被改写，
    这些副本才会指向隔离目录。task.manager 是真实污染的主要源头
    （测试经 TaskManager 落任务历史归档），故单独点名校验。
    """
    from aitrade.cnn import storage as cnn_storage
    from aitrade.task import manager

    home = config.AITRADE_HOME.resolve()
    assert manager.TASK_HISTORY_PATH.resolve().is_relative_to(home), (
        "task.manager.TASK_HISTORY_PATH 未被重定向，任务历史会写进真实目录"
    )
    assert cnn_storage.CNN_MODEL_PATH.resolve().is_relative_to(home), (
        "cnn.storage.CNN_MODEL_PATH 未被重定向，模型产物会写进真实目录"
    )
