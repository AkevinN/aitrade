"""
规则策略回测/扫参/WalkForward REST API（Phase 2）。

路由前缀：/api/strategy
所有写操作均为异步任务，通过共享 task_manager 注册，前端可经
/api/alpha/tasks/{id} 轮询终态。

装配约定
--------
模块级 ``from .. import rules`` 是 rules 包的正式装配点，import 本模块即触发
etf_momentum / cnn_adapter / rebalancing_topk / cb_double_low 四个注册副作用。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import polars as pl
from fastapi import APIRouter
from pydantic import BaseModel

from .. import rules  # noqa: F401  触发信号源与策略的自注册（rules/__init__.py 约定）
from ..alpha.lab import AlphaLab
from ..backtest.registry import build_signal_source, list_signal_sources
from ..backtest.scheme import (
    CostConfig,
    PredictorConfig,
    Scheme,
    StrategyConfig,
    run_scheme_backtest,
)
from ..backtest.sweep import param_sweep_table
from ..backtest.validation import walk_forward_windows
from ..config import ALPHA_LAB_PATH
from ..datasource.manager import DataSourceManager
from ..models import TaskType
from ..models.strategy import (
    StrategyBacktestRequest,
    StrategySweepRequest,
    StrategyWalkForwardRequest,
)
from ..rules.store import CBTermsStore, FundamentalStore
from ..rules.universe import compute_coverage
from ..task import task_manager

logger = logging.getLogger(__name__)


# =============================================================================
# 模块级 akshare 包装函数（可被测试 monkeypatch）
# =============================================================================


def _ak_bond_zh_cov():
    """调用 akshare.bond_zh_cov 拉取全市场转债列表快照。

    集中在此函数以便测试 monkeypatch（不在业务代码中散落 akshare.xxx 调用）。

    Returns:
        akshare 返回的 pandas DataFrame，每行一只转债，含债券代码、债券简称等字段。
    """
    import akshare as ak  # type: ignore  延迟 import，仅在实际调用时加载
    return ak.bond_zh_cov()


def _ak_bond_zh_cov_value_analysis(symbol: str):
    """调用 akshare.bond_zh_cov_value_analysis 拉取单只转债的转股溢价率历史。

    集中在此函数以便测试 monkeypatch（不在业务代码中散落 akshare.xxx 调用）。

    Args:
        symbol: 6 位纯数字转债代码（不含交易所后缀，如 "113050"）。

    Returns:
        akshare 返回的 pandas DataFrame，含转股溢价率等历史时序列。
    """
    import akshare as ak  # type: ignore
    return ak.bond_zh_cov_value_analysis(symbol=symbol, indicator="转股溢价率")


# -----------------------------------------------------------------------------
# 基本面数据获取函数（可被测试 monkeypatch）
# -----------------------------------------------------------------------------

# 模块级 DataSourceManager 单例（测试可通过 monkeypatch _get_datasource_manager 替换）
_datasource_manager: DataSourceManager | None = None


def _get_datasource_manager() -> DataSourceManager:
    """获取全局 DataSourceManager 单例（延迟初始化）。

    首次调用时实例化 DataSourceManager 并缓存到模块级变量，后续调用复用同一实例；
    测试可通过 monkeypatch 此函数注入 mock，绕过真实数据源。

    Returns:
        进程内共享的 DataSourceManager 单例，恒为非 None：尚未初始化时当场创建后返回，
        之后每次返回的都是同一对象（缓存于模块级 _datasource_manager）。
    """
    global _datasource_manager  # noqa: PLW0603
    if _datasource_manager is None:
        _datasource_manager = DataSourceManager()
    return _datasource_manager


def _fetch_fundamental(
    symbol: str,
    exchange: str,
    start_str: str,
    end_str: str,
) -> list:
    """经 datasource_manager 拉取单标的基本面日度数据。

    集中在此函数便于测试 monkeypatch（与 _ak_* 系列桩点约定一致）。

    Args:
        symbol: 不含交易所后缀的标的代码（如 "600519"）。
        exchange: 交易所代码（如 "SSE"/"SZSE"）。
        start_str: 起始日期，格式 "YYYYMMDD"（含）。
        end_str: 结束日期，格式 "YYYYMMDD"（含）。

    Returns:
        FundamentalRecord 列表，逐交易日一条；区间内无数据时为空列表。
    """
    manager = _get_datasource_manager()
    return manager.get_fundamental(symbol, exchange, start_str, end_str)


# -----------------------------------------------------------------------------
# 基本面下载请求体
# -----------------------------------------------------------------------------


class FundamentalRefreshRequest(BaseModel):
    """POST /fundamental/refresh 的请求体：指定要刷新基本面的标的与日期区间。

    Attributes:
        vt_symbols: 待刷新的标的列表，元素为带交易所后缀的 vt_symbol（如 "600519.SSE"）。
        start: 数据起始日期（含）。
        end: 数据结束日期（含）。
    """

    vt_symbols: list[str]
    start: date
    end: date


router = APIRouter(
    prefix="/api/strategy",
    tags=["规则策略"],
)


# =============================================================================
# Helper：AlphaLab 获取（每次新建，测试可 monkeypatch）
# =============================================================================


def _get_lab() -> AlphaLab:
    """构造一个绑定到 ALPHA_LAB_PATH 的 AlphaLab 实例。

    AlphaLab 为轻量对象、无跨调用的状态副作用，故每次调用都新建一个、不做缓存；
    测试可通过 monkeypatch 此函数注入指向 tmp_path 的 lab，隔离真实数据目录。

    Returns:
        新建的 AlphaLab 实例，恒为非 None；每次调用返回相互独立的对象，
        互不共享内部状态，路径固定为模块常量 ALPHA_LAB_PATH。
    """
    return AlphaLab(ALPHA_LAB_PATH)


# =============================================================================
# POST /cb-terms/refresh  —  转债条款快照 + 溢价率历史下载
# =============================================================================


@router.post(
    "/cb-terms/refresh",
    description=(
        "下载转债列表快照 + 逐债历史溢价率，保存到本地 parquet。\n\n"
        "**注意**：全量 1000+ 只转债耗时 20+ 分钟（受限频约 1.2s/只）。"
        "可通过 `symbols` 参数指定子集加速调试（如 ['113050.SSE', '128093.SZSE']）。"
    ),
)
async def refresh_cb_terms(
    symbols: list[str] | None = None,
) -> dict:
    """启动异步任务：拉取转债列表快照 + 逐债溢价率历史（带进度上报）。

    任务注册到共享 task_manager 后立即返回，实际下载在后台线程进行；
    前端经 /api/alpha/tasks/{id} 轮询终态与进度。

    Args:
        symbols: 可选 vt_symbol 子集（如 ``["113050.SSE"]``），为 None 或空则全量拉取。
                 全量 1000+ 只耗时 20+ 分钟，建议在维护窗口执行。

    Returns:
        dict，含 ``task_id``（轮询用的任务 id）与 ``message``（中文提示文案）。
    """
    task_id = task_manager.create_task(
        TaskType.STRATEGY_BACKTEST,  # 复用已有任务类型，无需新增枚举
        params={"symbols": symbols or []},
        title="转债条款刷新" + (f" ({len(symbols)} 只)" if symbols else " (全量)"),
        entity_type="cb_terms_refresh",
        entity_name="cb_terms",
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """task_manager 调度的任务体：转发到 _refresh_cb_terms 执行实际下载。

        Args:
            on_progress: task_manager 注入的进度回调（进度 0~100，中文描述）。

        Returns:
            _refresh_cb_terms 的统计 dict（snapshot_count/success/failed/...）。
        """
        return _refresh_cb_terms(symbols=symbols or [], on_progress=on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {
        "task_id": task_id,
        "message": (
            "转债条款刷新任务已启动"
            + ("" if symbols else "（全量 1000+ 只，耗时 20+ 分钟）")
        ),
    }


# =============================================================================
# POST /fundamental/refresh  —  基本面数据下载
# =============================================================================


@router.post(
    "/fundamental/refresh",
    description=(
        "异步任务：逐标的拉取日度基本面（pe/pb/total_mv/circ_mv 等），保存到本地 parquet。\n\n"
        "**注意**：total_mv / circ_mv 单位为**万元**（tushare daily_basic 原始值）。\n\n"
        "body: {vt_symbols: [...], start: 'YYYY-MM-DD', end: 'YYYY-MM-DD'}"
    ),
)
async def refresh_fundamental(req: FundamentalRefreshRequest) -> dict:
    """启动异步任务：批量下载基本面数据并落盘。

    任务注册后立即返回，实际逐标的下载在后台线程进行；
    前端经 /api/alpha/tasks/{id} 轮询终态与进度。

    Args:
        req: 请求体，含待刷新的 vt_symbols 与日期区间，见 FundamentalRefreshRequest。

    Returns:
        dict，含 ``task_id``（轮询用任务 id）与 ``message``（中文提示文案）。
    """
    task_id = task_manager.create_task(
        TaskType.STRATEGY_BACKTEST,
        params={"vt_symbols": req.vt_symbols, "start": req.start.isoformat(), "end": req.end.isoformat()},
        title=f"基本面刷新 ({len(req.vt_symbols)} 只)",
        entity_type="fundamental_refresh",
        entity_name="fundamental",
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """task_manager 调度的任务体：转发到 _refresh_fundamental 执行实际下载。

        Args:
            on_progress: task_manager 注入的进度回调（进度 0~100，中文描述）。

        Returns:
            _refresh_fundamental 的统计 dict（success/failed/failed_symbols）。
        """
        return _refresh_fundamental(
            vt_symbols=req.vt_symbols,
            start=req.start,
            end=req.end,
            on_progress=on_progress,
        )

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": f"基本面刷新任务已启动（{len(req.vt_symbols)} 只标的）"}


# =============================================================================
# GET /sources
# =============================================================================


@router.get(
    "/sources",
    description="列出所有已注册信号源（name / description / param_spec）。",
)
async def list_sources() -> list[dict]:
    """列出所有已注册信号源的元信息（同步，无 I/O）。

    Returns:
        每个信号源一个 dict，含 ``name`` / ``description`` / ``param_spec``；
        无任何注册时返回空列表。
    """
    return list_signal_sources()


# =============================================================================
# POST /backtest/run
# =============================================================================


@router.post(
    "/backtest/run",
    description="提交单次规则策略回测异步任务，返回 task_id。前端经 /api/alpha/tasks/{id} 轮询结果。",
)
async def run_backtest(req: StrategyBacktestRequest) -> dict:
    """创建并启动单次规则策略回测异步任务。

    任务注册后立即返回，回测在后台线程执行（_run_strategy_backtest）；
    前端经 /api/alpha/tasks/{id} 轮询结果。

    Args:
        req: 回测请求体，含信号源/策略名/参数/日期区间/成本配置等，见 StrategyBacktestRequest。

    Returns:
        dict，含 ``task_id``（轮询用任务 id）与 ``message``（中文提示文案）。
    """
    task_id = task_manager.create_task(
        TaskType.STRATEGY_BACKTEST,
        params={
            "signal_source": req.signal_source,
            "strategy_name": req.strategy_name,
        },
        title=f"规则策略回测: {req.signal_source}/{req.strategy_name}",
        entity_type="strategy_backtest",
        entity_name=req.signal_source,
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """task_manager 调度的任务体：转发到 _run_strategy_backtest 执行回测。

        Args:
            on_progress: task_manager 注入的进度回调（进度 0~100，中文描述）。

        Returns:
            _run_strategy_backtest 的回测结果 dict（statistics/trades/universe_coverage 等）。
        """
        return _run_strategy_backtest(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "规则策略回测任务已启动"}


# =============================================================================
# POST /sweep/run
# =============================================================================


@router.post(
    "/sweep/run",
    description="提交参数网格扫描异步任务，返回 task_id。",
)
async def run_sweep(req: StrategySweepRequest) -> dict:
    """创建并启动参数网格扫描（grid sweep）异步任务。

    任务注册后立即返回，扫参在后台线程执行（_run_sweep）；
    前端经 /api/alpha/tasks/{id} 轮询结果。

    Args:
        req: 扫参请求体，含信号源、基准参数与待扫描网格 ``grid``，见 StrategySweepRequest。

    Returns:
        dict，含 ``task_id``（轮询用任务 id）与 ``message``（中文提示文案）。
    """
    task_id = task_manager.create_task(
        TaskType.STRATEGY_SWEEP,
        params={
            "signal_source": req.signal_source,
            "grid_size": len(req.grid),
        },
        title=f"规则策略扫参: {req.signal_source} ({len(req.grid)} 组)",
        entity_type="strategy_sweep",
        entity_name=req.signal_source,
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """task_manager 调度的任务体：转发到 _run_sweep 执行网格扫描。

        Args:
            on_progress: task_manager 注入的进度回调（进度 0~100，中文描述）。

        Returns:
            _run_sweep 的结果 dict（含 ``rows`` 逐网格点结果与 ``base_params``）。
        """
        return _run_sweep(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "参数扫描任务已启动"}


# =============================================================================
# POST /walkforward/run
# =============================================================================


@router.post(
    "/walkforward/run",
    description="提交 Walk-Forward 验证异步任务，返回 task_id。",
)
async def run_walkforward(req: StrategyWalkForwardRequest) -> dict:
    """创建并启动 Walk-Forward 滚动验证异步任务。

    任务注册后立即返回，验证在后台线程执行（_run_walkforward）；
    前端经 /api/alpha/tasks/{id} 轮询结果。

    Args:
        req: 验证请求体，含信号源、训练/测试窗长 ``train_days``/``test_days`` 等，
             见 StrategyWalkForwardRequest。

    Returns:
        dict，含 ``task_id``（轮询用任务 id）与 ``message``（中文提示文案）。
    """
    task_id = task_manager.create_task(
        TaskType.STRATEGY_WALKFORWARD,
        params={
            "signal_source": req.signal_source,
            "train_days": req.train_days,
            "test_days": req.test_days,
        },
        title=f"规则策略 Walk-Forward: {req.signal_source}",
        entity_type="strategy_walkforward",
        entity_name=req.signal_source,
    )

    def execute(on_progress: Callable[[float, str], None] | None = None) -> dict:
        """task_manager 调度的任务体：转发到 _run_walkforward 执行滚动验证。

        Args:
            on_progress: task_manager 注入的进度回调（进度 0~100，中文描述）。

        Returns:
            _run_walkforward 的结果 dict（含 ``windows`` 逐窗结果与 ``aggregate`` 聚合统计）。
        """
        return _run_walkforward(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "Walk-Forward 验证任务已启动"}


# =============================================================================
# 任务体：内部实现
# =============================================================================


def _run_strategy_backtest(
    req: StrategyBacktestRequest,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """完整一次规则策略回测的任务体（在后台线程内同步执行）。

    流程：
    1. 信号生成（进度 0-40%）
    2. scheme 组装 + 撮合引擎（进度 40-95%）
    3. 拼装宇宙覆盖率统计与前端契约字段（进度 100%）

    Args:
        req: 回测请求体，提供信号源/策略/参数/区间/成本等配置。
        on_progress: 进度回调（进度 0~100 浮点 + 中文描述）；为 None 时不上报。

    Returns:
        回测结果 dict：除 run_scheme_backtest 原有字段（statistics/trades/equity_curve 等）外，
        额外附加 ``signal_source`` / ``strategy_name`` / ``vt_symbols`` /
        ``universe_coverage``（宇宙覆盖率明细）。

    Raises:
        RuntimeError: 区间内未产生任何信号（数据未下载或参数不当）时抛出。
    """
    lab = _get_lab()

    # 步骤 1：生成信号（把 _lab 注入信号源，保证信号与撮合使用同一数据仓库）
    signal_params = {**req.signal_params, "_lab": lab}
    provider = build_signal_source(req.signal_source, signal_params)

    def _sig_progress(p: float, msg: str = "") -> None:
        """把信号生成阶段的进度（0~100）压缩映射到整体进度的前 40% 区间后上报。

        Args:
            p: 信号生成阶段自身进度（0~100）。
            msg: 中文进度描述，原样透传给外层回调。
        """
        if on_progress:
            on_progress(p * 0.40, msg)

    signal_df = provider.predict(req.start, req.end, on_progress=_sig_progress)

    if signal_df is None or signal_df.is_empty():
        raise RuntimeError(
            "区间内未产生任何信号，请检查数据是否已下载或调整参数"
        )

    if on_progress:
        on_progress(40.0, "信号生成完成，开始撮合回测")

    # 步骤 2：组装 Scheme 并运行回测
    vt_symbols = signal_df["vt_symbol"].unique().to_list()

    scheme = Scheme(
        name=f"rule_{req.signal_source}",
        vt_symbols=vt_symbols,
        interval=req.interval,
        capital=req.capital,
        predictor=PredictorConfig(type=req.signal_source, params=req.signal_params),
        strategy=StrategyConfig(name=req.strategy_name, params=req.strategy_params),
        cost=CostConfig(
            commission_rate=req.cost.commission_rate,
            stamp_duty=req.cost.stamp_duty,
            slippage=req.cost.slippage,
            t_plus1=req.cost.t_plus1,
        ),
        label_spec={},
    )

    start_dt = datetime.combine(req.start, datetime.min.time())
    end_dt = datetime.combine(req.end, datetime.min.time())

    result = run_scheme_backtest(
        scheme,
        data_loader=lab,
        signal_df=signal_df,
        start=start_dt,
        end=end_dt,
    )

    if on_progress:
        on_progress(100.0, "回测完成")

    # ---- 宇宙覆盖率统计 ----
    # 信号实际产出的标的集合即为"有行情数据"的标的（信号源从 lab 读取，能产出信号 = 有数据）
    symbols_with_bars = set(vt_symbols)

    # 无时点过滤（v1 降级：list/delist 日期来源需 contracts 缓存，此处留空并记 warning）
    coverage_warnings: list[str] = [
        "上市/退市日期数据不可用，时点过滤未生效（v1 降级）"
    ]
    coverage = compute_coverage(
        requested=vt_symbols,
        symbols_with_bars=symbols_with_bars,
        symbols_with_fundamental=None,  # 本策略不依赖基本面
        excluded=[],
        coverage_warnings=coverage_warnings,
    )

    # 附加前端契约字段
    result["signal_source"] = req.signal_source
    result["strategy_name"] = req.strategy_name
    result["vt_symbols"] = vt_symbols
    result["universe_coverage"] = {
        "requested": coverage.requested,
        "with_bars": coverage.with_bars,
        "with_fundamental": coverage.with_fundamental,
        "excluded_not_listed": coverage.excluded_not_listed,
        "coverage_ratio": coverage.coverage_ratio,
        "warnings": coverage.warnings,
    }
    return result


def _run_sweep(
    req: StrategySweepRequest,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """参数网格扫描（grid sweep）任务体（在后台线程内同步执行）。

    优化：只有当某 grid 项改变了 signal_params 时才重新生成信号；
    仅动 strategy_params 的项复用基准 signal_df（节省 I/O 与计算）。

    Args:
        req: 扫参请求体，提供信号源、基准参数与待扫描网格 ``grid``。
        on_progress: 进度回调（进度 0~100 浮点 + 中文描述）；为 None 时不上报。

    Returns:
        dict，含 ``rows``（param_sweep_table 产出的逐网格点结果行）与
        ``base_params``（基准 signal_params / strategy_params，供前端展示）。
    """
    lab = _get_lab()
    base_signal_params = {**req.signal_params, "_lab": lab}

    # 预先为「基准信号参数」生成信号（用于纯 strategy_params 变动项复用）
    base_provider = build_signal_source(req.signal_source, base_signal_params)
    base_signal_df = base_provider.predict(req.start, req.end)

    total = len(req.grid)

    def _run_one(override: dict) -> dict[str, Any]:
        """对单个网格覆盖项跑完整回测。

        按需重建信号源：仅当 ``override`` 含非空 ``signal_params`` 时才重新生成信号，
        否则复用闭包内的基准 ``base_signal_df`` 以省去重复 I/O。

        Args:
            override: 单个网格点的参数覆盖，形如
                ``{"signal_params": {...}, "strategy_params": {...}}``，两键均可缺省。

        Returns:
            该网格点的回测结果 dict；信号为空时返回零成交占位结果
            （statistics={}、trades=[]、equity_curve=[]、trade_count=0），不抛错。
        """
        sp_override: dict = override.get("signal_params", {})
        str_override: dict = override.get("strategy_params", {})

        # 信号复用优化：只有信号参数变了才重新生成（注释标明复用策略）
        if sp_override:
            # 信号参数有变动 → 重新构造信号源并生成信号
            merged_sp = {**req.signal_params, **sp_override, "_lab": lab}
            provider = build_signal_source(req.signal_source, merged_sp)
            signal_df = provider.predict(req.start, req.end)
        else:
            # 信号参数未变 → 复用基准 signal_df，避免重复 I/O
            signal_df = base_signal_df

        if signal_df is None or signal_df.is_empty():
            # 空信号：返回零成交结果，不抛错（sweep 允许部分网格点无信号）
            return {
                "statistics": {},
                "trades": [],
                "equity_curve": [],
                "trade_count": 0,
            }

        vt_symbols = signal_df["vt_symbol"].unique().to_list()
        merged_strategy = {**req.strategy_params, **str_override}

        scheme = Scheme(
            name=f"rule_{req.signal_source}_sweep",
            vt_symbols=vt_symbols,
            interval=req.interval,
            capital=req.capital,
            predictor=PredictorConfig(type=req.signal_source, params=req.signal_params),
            strategy=StrategyConfig(name=req.strategy_name, params=merged_strategy),
            cost=CostConfig(
                commission_rate=req.cost.commission_rate,
                stamp_duty=req.cost.stamp_duty,
                slippage=req.cost.slippage,
                t_plus1=req.cost.t_plus1,
            ),
            label_spec={},
        )

        start_dt = datetime.combine(req.start, datetime.min.time())
        end_dt = datetime.combine(req.end, datetime.min.time())

        return run_scheme_backtest(scheme, data_loader=lab, signal_df=signal_df, start=start_dt, end=end_dt)

    def _run_with_progress(override: dict) -> dict[str, Any]:
        """包装 _run_one，在每次完成后递增计数器并上报进度。

        完成数记在函数属性 ``_run_with_progress._done`` 上（闭包外初始化为 0），
        进度按 已完成数/总网格数 线性映射到 0~95% 区间。

        Args:
            override: 透传给 _run_one 的单个网格点参数覆盖。

        Returns:
            _run_one 的回测结果 dict。
        """
        result = _run_one(override)
        _run_with_progress._done += 1  # type: ignore[attr-defined]
        if on_progress:
            on_progress(_run_with_progress._done / total * 95.0, f"已完成 {_run_with_progress._done}/{total} 组")  # type: ignore[attr-defined]
        return result

    _run_with_progress._done = 0  # type: ignore[attr-defined]

    rows = param_sweep_table(_run_with_progress, req.grid)

    if on_progress:
        on_progress(100.0, "参数扫描完成")

    return {
        "rows": rows,
        "base_params": {
            "signal_params": req.signal_params,
            "strategy_params": req.strategy_params,
        },
    }


def _run_walkforward(
    req: StrategyWalkForwardRequest,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Walk-Forward 滚动验证任务体：逐 test 窗跑完整回测（信号也按窗生成）。

    用 walk_forward_windows 切出训练/测试窗后，仅在各测试窗上生成信号并回测；
    单窗信号生成失败或为空时记空结果继续（不中断整体），最后聚合各窗收益与夏普。

    Args:
        req: 验证请求体，提供信号源/策略/参数/区间，以及窗长 ``train_days``/``test_days``。
        on_progress: 进度回调（进度 0~100 浮点 + 中文描述）；为 None 时不上报。

    Returns:
        dict，含 ``windows``（逐窗结果：训练/测试起止日期、statistics、trade_count）与
        ``aggregate``（聚合统计：avg_return / avg_sharpe / positive_window_ratio /
        total_windows，无有效窗时各均值字段为 None）。
    """
    lab = _get_lab()

    windows = walk_forward_windows(
        req.start,
        req.end,
        train_days=req.train_days,
        test_days=req.test_days,
    )

    total = len(windows)
    window_results: list[dict[str, Any]] = []

    for idx, w in enumerate(windows):
        train_start, train_end = w["train"]
        test_start, test_end = w["test"]

        # 每窗独立生成信号（_lab 注入保证同一仓库）
        signal_params = {**req.signal_params, "_lab": lab}
        provider = build_signal_source(req.signal_source, signal_params)
        try:
            signal_df = provider.predict(test_start, test_end)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "walk-forward 窗口 %s~%s 信号生成失败，按空信号处理: %s",
                test_start,
                test_end,
                exc,
            )
            signal_df = None

        if signal_df is None or signal_df.is_empty():
            window_results.append({
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "statistics": {},
                "trade_count": 0,
            })
        else:
            vt_symbols = signal_df["vt_symbol"].unique().to_list()
            scheme = Scheme(
                name=f"rule_{req.signal_source}_wf",
                vt_symbols=vt_symbols,
                interval=req.interval,
                capital=req.capital,
                predictor=PredictorConfig(type=req.signal_source, params=req.signal_params),
                strategy=StrategyConfig(name=req.strategy_name, params=req.strategy_params),
                cost=CostConfig(
                    commission_rate=req.cost.commission_rate,
                    stamp_duty=req.cost.stamp_duty,
                    slippage=req.cost.slippage,
                    t_plus1=req.cost.t_plus1,
                ),
                label_spec={},
            )

            start_dt = datetime.combine(test_start, datetime.min.time())
            end_dt = datetime.combine(test_end, datetime.min.time())

            res = run_scheme_backtest(
                scheme,
                data_loader=lab,
                signal_df=signal_df,
                start=start_dt,
                end=end_dt,
            )

            window_results.append({
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "statistics": res.get("statistics", {}),
                "trade_count": res.get("trade_count", 0),
            })

        if on_progress:
            on_progress((idx + 1) / total * 95.0, f"已完成窗口 {idx + 1}/{total}")

    # 聚合统计
    returns = [
        float(w["statistics"].get("total_return", 0.0))
        for w in window_results
        if w["statistics"]
    ]
    sharpes = [
        float(w["statistics"].get("sharpe_ratio", 0.0))
        for w in window_results
        if w["statistics"]
    ]
    positive_count = sum(1 for r in returns if r > 0)
    aggregate: dict[str, Any] = {
        "avg_return": sum(returns) / len(returns) if returns else None,
        "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else None,
        "positive_window_ratio": positive_count / len(returns) if returns else None,
        "total_windows": total,
    }

    if on_progress:
        on_progress(100.0, "Walk-Forward 验证完成")

    return {"windows": window_results, "aggregate": aggregate}


# =============================================================================
# 任务体：基本面刷新
# =============================================================================


def _refresh_fundamental(
    vt_symbols: list[str],
    start: date,
    end: date,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """基本面刷新任务体。

    流程：
    1. 逐标的经 _fetch_fundamental 拉取基本面数据（FundamentalRecord 列表）
    2. 转为 polars DataFrame，经 FundamentalStore.save 增量落盘
    3. 个别标的失败：warning 日志 + 记录失败名单，继续处理后续标的
    4. 全部失败：任务 FAILED（抛出 RuntimeError，中文错误信息）

    Args:
        vt_symbols: 标的列表（如 ["600519.SSE", "000001.SZSE"]）。
        start: 数据开始日期。
        end: 数据结束日期。
        on_progress: 进度回调（0~100，中文描述）。

    Returns:
        dict 含 success / failed / failed_symbols 统计。
    """
    store = FundamentalStore()
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    total = len(vt_symbols)
    success_count = 0
    failed: list[str] = []

    for idx, vt_symbol in enumerate(vt_symbols):
        # 解析 symbol 与 exchange（格式：symbol.EXCHANGE）
        parts = vt_symbol.rsplit(".", 1)
        if len(parts) != 2:  # noqa: PLR2004
            logger.warning("基本面刷新：无法解析 vt_symbol=%s，已跳过", vt_symbol)
            failed.append(f"{vt_symbol}: 格式无法解析")
            continue

        symbol, exchange = parts[0], parts[1]

        try:
            records = _fetch_fundamental(symbol, exchange, start_str, end_str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("基本面刷新：%s 数据拉取失败: %s", vt_symbol, exc)
            failed.append(f"{vt_symbol}: {exc}")
            continue

        if not records:
            logger.warning("基本面刷新：%s 区间内无数据，已跳过", vt_symbol)
            failed.append(f"{vt_symbol}: 区间内无数据")
            continue

        # FundamentalRecord 列表转 polars DataFrame
        rows = [
            {
                "datetime": rec.trade_date,  # str "YYYYMMDD"
                "pe": rec.pe,
                "pe_ttm": rec.pe_ttm,
                "pb": rec.pb,
                "total_mv": rec.total_mv,
                "circ_mv": rec.circ_mv,
                "turnover_rate": rec.turnover_rate,
            }
            for rec in records
        ]
        df = pl.DataFrame(rows)

        try:
            store.save(vt_symbol, df)
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("基本面刷新：%s 落盘失败: %s", vt_symbol, exc)
            failed.append(f"{vt_symbol}: 落盘失败 {exc}")
            continue

        if on_progress:
            progress = (idx + 1) / total * 95.0
            on_progress(progress, f"基本面：{idx + 1}/{total}，{vt_symbol}")

    if on_progress:
        on_progress(100.0, f"基本面刷新完成：成功 {success_count}/{total}")

    # 全部失败 → 任务 FAILED
    if total > 0 and success_count == 0:
        raise RuntimeError(
            f"基本面刷新全部失败（{len(failed)} 只），请检查数据源配置或网络连接。"
            f"失败标的：{failed[:5]}{'...' if len(failed) > 5 else ''}"
        )

    return {
        "success": success_count,
        "failed": len(failed),
        "failed_symbols": failed,
    }


# =============================================================================
# 任务体：转债条款刷新
# =============================================================================


def _refresh_cb_terms(
    symbols: list[str],
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """转债条款刷新任务体。

    流程：
    1. 拉取 bond_zh_cov() 列表快照并落盘（进度 0~10%）
    2. 过滤要处理的转债（symbols 子集或全量）
    3. 逐只拉取 bond_zh_cov_value_analysis 溢价率历史，带 1.2s 限频（进度 10~95%）
    4. 完成（进度 100%）

    Args:
        symbols: vt_symbol 子集（空列表 = 全量）。
        on_progress: 进度回调（0~100 浮点，中文进度描述）。

    Returns:
        dict 含 snapshot_count / success / failed / failed_symbols 统计。
    """
    import polars as pl  # noqa: PLC0415  延迟 import

    store = CBTermsStore()

    # ---- 步骤 1：拉快照 ----
    if on_progress:
        on_progress(2.0, "正在拉取转债列表快照...")

    try:
        raw_df = _ak_bond_zh_cov()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"拉取转债列表失败: {exc}") from exc

    # 将 pandas DataFrame 转为 polars（兼容不同 akshare 版本）
    try:
        snapshot_df = pl.from_pandas(raw_df)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"转债列表格式解析失败: {exc}") from exc

    store.save_snapshot(snapshot_df)
    snapshot_count = snapshot_df.height

    if on_progress:
        on_progress(10.0, f"快照保存完成，共 {snapshot_count} 只转债")

    # ---- 步骤 2：确定待处理子集 ----
    # 从快照提取 symbol 列（字段名 "债券代码"），拼接 vt_symbol（.SSE/.SZSE）
    def _to_vt_symbol(code: str) -> str:
        """根据转债代码前 3 位推断所属交易所，拼成带后缀的 vt_symbol。

        110/111/113/118 归上交所（.SSE），123/127/128 归深交所（.SZSE）。

        Args:
            code: 6 位纯数字转债代码（如 "113050"）。

        Returns:
            形如 "113050.SSE" 的 vt_symbol；前缀无法判断（含长度不足）时返回空串。
        """
        prefix3 = code[:3] if len(code) >= 3 else ""
        if prefix3 in ("110", "111", "113", "118"):
            return f"{code}.SSE"
        if prefix3 in ("123", "127", "128"):
            return f"{code}.SZSE"
        # 无法判断的代码跳过
        return ""

    # 尝试读取代码列（字段名可能为 "债券代码" 或 "symbol"）
    code_col = None
    for candidate in ("债券代码", "symbol", "代码"):
        if candidate in snapshot_df.columns:
            code_col = candidate
            break

    if code_col is None:
        logger.warning("CBTermsStore: 快照无法找到代码列，跳过溢价率历史下载")
        return {
            "snapshot_count": snapshot_count,
            "success": 0,
            "failed": 0,
            "failed_symbols": [],
        }

    all_codes = snapshot_df[code_col].cast(pl.Utf8).to_list()
    all_vt_symbols = [_to_vt_symbol(str(c).strip()) for c in all_codes]
    all_vt_symbols = [v for v in all_vt_symbols if v]  # 过滤空串

    if symbols:
        # 取交集（用户指定的子集 ∩ 快照中存在的）
        target_vt_symbols = [v for v in all_vt_symbols if v in set(symbols)]
        if not target_vt_symbols:
            logger.warning("CBTermsStore: 指定子集与快照无交集，symbols=%s", symbols)
    else:
        target_vt_symbols = all_vt_symbols

    total = len(target_vt_symbols)
    success_count = 0
    failed: list[str] = []

    # ---- 步骤 3：逐只拉溢价率历史 ----
    for idx, vt_symbol in enumerate(target_vt_symbols):
        code = vt_symbol.rsplit(".", 1)[0]  # "113050.SSE" -> "113050"
        try:
            raw_hist = _ak_bond_zh_cov_value_analysis(symbol=code)
            hist_df = pl.from_pandas(raw_hist)
            store.save_premium_history(vt_symbol, hist_df)
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("CBTermsStore: %s 溢价率历史拉取失败: %s", vt_symbol, exc)
            failed.append(f"{vt_symbol}: {exc}")

        # 限频：1.2s 间隔（避免接口封禁）
        if idx < total - 1:
            time.sleep(1.2)

        if on_progress:
            progress = 10.0 + (idx + 1) / total * 85.0
            on_progress(progress, f"溢价率历史：{idx + 1}/{total}，{vt_symbol}")

    if on_progress:
        on_progress(100.0, f"转债条款刷新完成：快照 {snapshot_count} 只，成功 {success_count}/{total}")

    return {
        "snapshot_count": snapshot_count,
        "success": success_count,
        "failed": len(failed),
        "failed_symbols": failed,
    }
