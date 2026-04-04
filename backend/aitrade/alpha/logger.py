"""
Alpha logger using loguru.
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
