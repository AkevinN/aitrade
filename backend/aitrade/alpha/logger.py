"""
Alpha 模块统一日志实例。

基于 loguru 构建，移除默认输出后重新注册带颜色格式的 stdout sink，
供 alpha 包内所有模块通过 ``from .logger import logger`` 引用。
外部模块亦可直接 ``from aitrade.alpha import logger`` 使用。
"""

import sys

from loguru import logger as _logger

# Remove default output
_logger.remove()

# Add terminal output
_fmt: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
_logger.add(sys.stdout, colorize=True, format=_fmt)

# Export logger for external use
logger = _logger
