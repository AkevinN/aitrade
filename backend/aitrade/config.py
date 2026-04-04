"""
Global configuration for aitrade backend.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

# =============================================================================
# Storage paths
# =============================================================================

# User home data directory (~/.aitrade/)
AITRADE_HOME: Path = Path.home() / ".aitrade"

# Alpha lab data storage
ALPHA_LAB_PATH: Path = AITRADE_HOME / "alpha_lab"
CNN_MODEL_PATH: Path = AITRADE_HOME / "cnn_models"

# Task state persistence
TASK_DB_PATH: Path = AITRADE_HOME / "tasks.db"

# =============================================================================
# API server
# =============================================================================

API_HOST: str = os.getenv("AITRADE_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("AITRADE_PORT", "8000"))
API_CORS_ORIGINS: list[str] = ["*"]

# =============================================================================
# Tushare data source
# =============================================================================

TUSHARE_TOKEN: str = os.getenv("TUSHARE_TOKEN", "")
TUSHARE_BACKEND: str = os.getenv("TUSHARE_BACKEND", "tushare")  # tushare or tinyshare

# =============================================================================
# Alpha research
# =============================================================================

# Parallel feature computation
MAX_WORKERS: int = int(os.getenv("AITRADE_MAX_WORKERS", "4"))

# Task polling
TASK_POLL_INTERVAL: int = 2  # seconds

# =============================================================================
# CSV Import field mapping
# =============================================================================

CSV_FIELD_MAPPING: dict[str, list[str]] = {
    "datetime": ["datetime", "date", "trade_date", "time", "tradedate", "时间", "交易日期"],
    "symbol": ["symbol", "code", "stock_code", "stockcode", "代码", "股票代码"],
    "exchange": ["exchange", "market", "board", "交易所"],
    "vt_symbol": ["vt_symbol", "vtsymbol"],
    "open": ["open", "open_price", "开盘价"],
    "high": ["high", "high_price", "最高价"],
    "low": ["low", "low_price", "最低价"],
    "close": ["close", "close_price", "收盘价"],
    "volume": ["volume", "vol", "成交量"],
    "turnover": ["turnover", "amount", "成交额", "成交金额"],
    "open_interest": ["open_interest", "oi", "持仓量"],
    "change_pct": ["change_pct", "涨跌幅", "pct_change", "change", "涨幅", "涨跌额%"],
    "amplitude": ["amplitude", "振幅", "amplitude_pct", "波幅"],
}

CSV_REQUIRED_FIELDS: list[str] = ["datetime", "open", "high", "low", "close"]

# =============================================================================
# Initialize storage directories
# =============================================================================

for _dir in [AITRADE_HOME, ALPHA_LAB_PATH, CNN_MODEL_PATH]:
    _dir.mkdir(parents=True, exist_ok=True)
