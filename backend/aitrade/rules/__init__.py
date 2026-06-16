"""
规则型免训练信号源域（rules 包）。

本包收录所有**无需训练**即可生产信号的适配器，包括：
- CNN 推理适配器（cnn_adapter）：把训练好的 CNN 模型包装为 SignalProvider
  并注册为 "cnn" 信号源。
- （后续）纯规则/指标型信号源。

装配约定
--------
模块级 import 会触发各子模块的自注册副作用（副作用写在子模块末尾，
与 cnn/strategy.py 的 register_strategy 模式一致）。
但 **本包不能被 backtest/registry.py 反向 import**（会造成循环依赖）。
正确的装配时机：
  - 测试：测试文件显式 ``import aitrade.rules``
  - API 层：``api/strategy.py`` 等入口模块在启动时 import 本包（Phase 2 接入）
"""

from . import cnn_adapter  # noqa: F401  触发 "cnn" 信号源自注册
from . import etf_momentum  # noqa: F401  触发 "etf_momentum" 信号源自注册
from . import strategy  # noqa: F401  触发 "rebalancing_topk" 策略自注册
from . import cb_double_low  # noqa: F401  触发 "cb_double_low" 信号源自注册
from . import small_cap  # noqa: F401  触发 "small_cap" 信号源自注册

__all__ = ["cnn_adapter", "etf_momentum", "strategy", "cb_double_low", "small_cap"]
