"""
screening 测试包级 fixture。

Feature: cnn-screening-tier2-speedup

强制 Tier-2 auto 并行度回退为串行：既有 Tier-2 测试用 ``monkeypatch`` 在**进程内**
桩化 ``run_walk_forward_evaluate``，但 ProcessPoolExecutor spawn 出的子进程会重新
import 模块、看不到 monkeypatch——若 auto 并行度选了多进程，桩化失效会触发真实训练
而挂起/失败。这里把 ``os.cpu_count`` 视作 1，使 ``_resolve_tier2_max_workers`` 的
auto 分支（``min(cpu_count, 任务数)``）恒为 1 → 走串行 inline 路径（桩化进程内生效）。

注意：本 fixture 只影响 **auto** 路径。显式设置 ``tier2_max_workers>1`` 或直接给
``_run_tier2_tasks`` 传 ``max_workers>1`` 的测试（如 test_tier2_parallel.py）不受影响
——它们另行注入 ThreadPoolExecutor 替身在进程内验证并行编排。
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _force_tier2_serial_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 os.cpu_count 视作 1，使 Tier-2 auto 并行度回退串行（见模块 docstring）。"""
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
