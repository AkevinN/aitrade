"""
CPU 训练单线程守卫（_cpu_single_thread 装饰器）测试。

覆盖 cnn-screening-tier2-speedup 特性的 Property 1：
- CPU 机：被装饰函数执行期间 torch intra-op 线程数为 1，退出后还原为进入前的值；
  正常返回与异常路径都还原（不泄漏给后续 torch 使用方）。
- CUDA 机：零干预（不改线程数）。

测试不做真实训练：用一个记录"执行期间线程数"的假函数验证守卫行为，快且确定。

Feature: cnn-screening-tier2-speedup
"""

from __future__ import annotations

import torch

from aitrade.cnn.trainer import _cpu_single_thread


def test_cpu_single_thread_sets_one_and_restores() -> None:
    # Feature: cnn-screening-tier2-speedup, Property 1: 线程守卫还原（CPU 正常路径）
    """CPU 下：执行期间线程数=1，退出后还原为进入前的值。"""
    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(3)  # 设一个可辨识的基线
        seen: dict[str, int] = {}

        @_cpu_single_thread
        def _fake_train() -> str:
            seen["inside"] = torch.get_num_threads()
            return "ok"

        result = _fake_train()

        assert result == "ok"
        assert seen["inside"] == 1, f"执行期间应单线程，实际 {seen['inside']}"
        assert torch.get_num_threads() == 3, "退出后应还原为进入前的 3"
    finally:
        torch.set_num_threads(prev)


def test_cpu_single_thread_restores_on_exception() -> None:
    # Feature: cnn-screening-tier2-speedup, Property 1: 线程守卫还原（CPU 异常路径）
    """CPU 下：被装饰函数抛异常时仍还原线程数。"""
    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(2)

        @_cpu_single_thread
        def _boom() -> None:
            assert torch.get_num_threads() == 1
            raise RuntimeError("boom")

        raised = False
        try:
            _boom()
        except RuntimeError:
            raised = True

        assert raised, "异常应原样向上抛出"
        assert torch.get_num_threads() == 2, "异常路径也必须还原线程数"
    finally:
        torch.set_num_threads(prev)


def test_cpu_single_thread_no_op_on_cuda(monkeypatch) -> None:
    # Feature: cnn-screening-tier2-speedup, Property 1: CUDA 机零干预
    """模拟 CUDA 可用：装饰器不改线程数（GPU 路径零干预）。"""
    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        seen: dict[str, int] = {}

        @_cpu_single_thread
        def _fake_train() -> None:
            seen["inside"] = torch.get_num_threads()

        _fake_train()

        assert seen["inside"] == 4, "CUDA 路径不应把线程数改为 1"
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(prev)
