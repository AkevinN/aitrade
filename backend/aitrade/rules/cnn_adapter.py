"""
CNN 推理适配器：把 predict_cnn_signals 包装为 SignalProvider 并注册为 "cnn"。

设计要点
--------
- 延迟 import：predict_cnn_signals 在 predict() **调用时**才 import，
  不在模块级 import。cnn 包依赖 torch，是懒加载设计；registry/rules 的
  模块级 import 不应把 torch 拉进进程。
- model_name 缺失时在构造期（_build_cnn_source）立即抛 ValueError（fail fast）。
- on_meta 不向外透传：adapter 的 SignalProvider 接口只暴露 on_progress；
  CNN 内部的 on_meta 不属于 SignalProvider 协议范畴。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from ..backtest.registry import SignalProvider, register_signal_source


class _CNNSignalSource:
    """把 predict_cnn_signals 包装为 SignalProvider 的私有实现。"""

    def __init__(self, model_name: str, extra_kwargs: dict[str, Any]) -> None:
        self._model_name = model_name
        self._extra_kwargs = extra_kwargs

    def predict(
        self,
        start: date,
        end: date,
        on_progress: object | None = None,
    ) -> pl.DataFrame:
        """调用 CNN 推理，返回 [datetime, vt_symbol, signal] DataFrame。"""
        # 延迟 import：torch 懒加载，不能在模块级触发
        from ..cnn.predictor import predict_cnn_signals  # noqa: PLC0415

        return predict_cnn_signals(
            model_name=self._model_name,
            start=start,
            end=end,
            on_progress=on_progress,  # type: ignore[arg-type]
            **self._extra_kwargs,
        )


def _build_cnn_source(params: dict) -> SignalProvider:
    """工厂函数：从 params 构造 _CNNSignalSource。

    Args:
        params: 必须包含 ``model_name``（str）；
                其余键作为 kwargs 透传给 predict_cnn_signals（如 on_meta 等可选参数）。

    Raises:
        ValueError: model_name 缺失或为空。
    """
    model_name = params.get("model_name")
    if not model_name:
        raise ValueError("构造 CNN 信号源时 model_name 必填且不能为空")

    # 除 model_name 外的其余参数透传给 predict_cnn_signals
    extra = {k: v for k, v in params.items() if k != "model_name"}
    return _CNNSignalSource(model_name=str(model_name), extra_kwargs=extra)


# 自注册到共享信号源注册表（模块被 import 时执行，模式同 cnn/strategy.py）
register_signal_source(
    "cnn",
    _build_cnn_source,
    description="CNN 模型推理信号",
    param_spec={
        "model_name": {
            "type": "str",
            "required": True,
            "label": "CNN 模型名称",
            "description": "对应 cnn_models/ 目录下的 .pt 文件名（不含扩展名）",
        },
    },
)
