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

# 项目数据目录（默认：项目根目录下的 `./.aitrade/`）
# 可通过环境变量 `AITRADE_HOME` 覆盖
AITRADE_HOME: Path = Path(
    os.getenv("AITRADE_HOME", str(PROJECT_ROOT / ".aitrade"))
).expanduser()

# Alpha lab data storage
ALPHA_LAB_PATH: Path = AITRADE_HOME / "alpha_lab"
CNN_MODEL_PATH: Path = AITRADE_HOME / "cnn_models"

# CNN 模型治理（滚动评估、候选晋级、回滚、治理回放）持久化目录
CNN_GOVERNANCE_PATH: Path = AITRADE_HOME / "cnn_governance"

# 量化方案（Scheme）配置持久化目录
SCHEME_PATH: Path = AITRADE_HOME / "schemes"

# 标的画像（Symbol Profiling）只读诊断产物持久化目录
# 该目录是 profiling 模块唯一允许写入的位置（只读模块的唯一副作用出口）
PROFILE_PATH: Path = AITRADE_HOME / "profiles"

# 实盘决策持久化目录（交易操作台，每 signal_id 一个 JSON 文件）
# DecisionStore(LIVE_DECISION_PATH) 首次实例化时会经 __init__ 的
# mkdir(parents=True, exist_ok=True) 自动创建该目录
LIVE_DECISION_PATH: Path = AITRADE_HOME / "live" / "decisions"

# 交易计划自动化（Trading Plan Automation）持久化与调度接线
# 交易计划存储目录（每 plan_id 一个 JSON 文件，TradingPlanStore 首次实例化自动创建）
TRADING_PLAN_PATH: Path = AITRADE_HOME / "live" / "plans"

# Phase 3 M2：规则策略调仓决策持久化目录（RebalanceStore 首次实例化自动创建）
LIVE_REBALANCE_PATH: Path = AITRADE_HOME / "live" / "rebalances"
# Phase 3 M2：持仓账本持久化目录（PositionBook 首次实例化自动创建）
LIVE_PORTFOLIO_PATH: Path = AITRADE_HOME / "live" / "portfolios"
# 调度器运行时轻状态（Last_Triggered_Map：{plan_id: "YYYY-MM-DD"}），重启可恢复
LIVE_RUNTIME_STATE_PATH: Path = AITRADE_HOME / "live" / "runtime_state.json"
# 调度器单实例互斥锁（防同机多进程并发触发）
SCHEDULER_LOCK_PATH: Path = AITRADE_HOME / "live" / "scheduler.lock"

# 任务历史持久化（task-scheduler-observability R2.5：以 TASK_HISTORY_PATH 取代已声明未用的 TASK_DB_PATH）
TASK_HISTORY_PATH: Path = AITRADE_HOME / "task_history"

# 调度运行日志（task-scheduler-observability R3：SchedulerRunLog 按日 JSONL 落盘）
SCHEDULER_RUN_LOG_PATH: Path = AITRADE_HOME / "live" / "scheduler_runs"

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
# AKShare data source
# =============================================================================

# 是否启用 AKShare 数据源（开源免费、无需 token）。默认开启。
AKSHARE_ENABLED: bool = os.getenv("AKSHARE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
# AKShare 网络请求重试次数与间隔（秒）
AKSHARE_MAX_RETRIES: int = max(int(os.getenv("AKSHARE_MAX_RETRIES", "3")), 1)
AKSHARE_RETRY_DELAY_SEC: float = max(float(os.getenv("AKSHARE_RETRY_DELAY_SEC", "0.8")), 0.0)

# =============================================================================
# Alpha research
# =============================================================================

# Parallel feature computation
MAX_WORKERS: int = int(os.getenv("AITRADE_MAX_WORKERS", "4"))

# Task polling
TASK_POLL_INTERVAL: int = 2  # seconds

# =============================================================================
# 交易计划自动调度（Plan Scheduler）
# =============================================================================

# 是否启用进程内自动调度器（默认开启；测试/CI/多开发实例可经环境变量关闭）
SCHEDULER_ENABLED: bool = os.getenv("AITRADE_SCHEDULER_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
# 调度器轮询周期（秒），下限 1 秒
SCHEDULER_TICK_SECONDS: float = max(float(os.getenv("AITRADE_SCHEDULER_TICK_SECONDS", "30")), 1.0)

# 注：通知通道凭证（AITRADE_NOTIFY_* webhook/secret）**不在此暴露为模块常量**，
# 由 live/notifier_channels.py:build_notifier 在运行时按需 os.getenv 读取，
# 避免被打印/序列化（脱敏红线，需求 9.4）。

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

# 规则型信号数据目录（转债条款快照 + 历史溢价率，CBTermsStore 使用）
RULES_DATA_PATH: Path = AITRADE_HOME / "rules"

for _dir in [AITRADE_HOME, ALPHA_LAB_PATH, CNN_MODEL_PATH, CNN_GOVERNANCE_PATH, SCHEME_PATH, PROFILE_PATH, RULES_DATA_PATH, TASK_HISTORY_PATH, SCHEDULER_RUN_LOG_PATH]:
    _dir.mkdir(parents=True, exist_ok=True)
