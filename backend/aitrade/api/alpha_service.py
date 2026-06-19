"""
Alpha 投研业务逻辑层（service）。

从 api/alpha.py 抽出，承载数据下载、聚合、数据集构建、模型训练、信号生成、
策略回测等耗时业务逻辑（通常由 TaskManager 异步执行）。路由层只做参数校验与转发。

说明：`_get_alpha_lab` / `_normalize_market_interval` / `ALPHA_LAB_PATH` 仍保留在
api/alpha.py（CSV 导入测试会 monkeypatch 其 ALPHA_LAB_PATH），本模块在函数内部
延迟导入它们，既复用同一份 lab 访问入口，又避免与路由模块形成顶层循环依赖。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

import polars as pl

from ..alpha.lab_utils import normalize_vt_symbol
from ..config import MAX_WORKERS
from ..datasource import datasource_manager
from ..datasource.types import DataCategory, ProviderStatus
from ..models import (
    DataAggregateRequest,
    DataDownloadRequest,
    DatasetCreateRequest,
    ModelTrainRequest,
    SignalGenerateRequest,
    BacktestRunRequest,
)


# =============================================================================
# 辅助函数
# =============================================================================

def _build_feature_dataset(
    df: pl.DataFrame,
    train_period: tuple[str, str],
    valid_period: tuple[str, str],
    test_period: tuple[str, str],
    feature_names: list[str],
):
    """按特征库列表构建合并特征数据集。

    逐个加载 alpha101 / alpha158 等特征库，校验特征名不冲突后合并到
    一个 ``AlphaDataset`` 实例中，供下游数据预处理与模型训练使用。

    Args:
        df:            K 线原始 DataFrame（含 datetime / vt_symbol / open / high / low / close / volume 等列）。
        train_period:  训练期 ``(start_str, end_str)``，格式 "YYYY-MM-DD"。
        valid_period:  验证期 ``(start_str, end_str)``。
        test_period:   测试期 ``(start_str, end_str)``。
        feature_names: 特征库名列表，支持 "alpha101" / "alpha158"。

    Returns:
        合并了所有指定特征库表达式与结果的 ``AlphaDataset`` 实例（未调用 ``prepare_data``）。

    Raises:
        ValueError: 特征库名不支持、特征名冲突，或特征表达式预校验失败时抛出。
    """
    from ..alpha.dataset import AlphaDataset
    from ..alpha.dataset.datasets.alpha_101 import Alpha101
    from ..alpha.dataset.datasets.alpha_158 import Alpha158
    from ..alpha.dataset.utility import calculate_by_expression, calculate_by_polars

    builders = {
        "alpha101": Alpha101,
        "alpha158": Alpha158,
    }

    dataset = AlphaDataset(
        df=df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
    )
    sample_df = df.sort(["datetime", "vt_symbol"]).head(min(df.height, 1500))

    for feature_name in feature_names:
        builder = builders.get(feature_name)
        if builder is None:
            raise ValueError(f"不支持的特征库: {feature_name}")

        feature_dataset = builder(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
        )

        for name, expression in feature_dataset.feature_expressions.items():
            if name in dataset.feature_expressions:
                raise ValueError(f"特征名冲突: {name}")
            try:
                if isinstance(expression, pl.Expr):
                    calculate_by_polars(sample_df, expression)
                else:
                    calculate_by_expression(sample_df, expression)
            except Exception as exc:
                raise ValueError(f"{feature_name} 特征校验失败: {name}: {exc}") from exc
            dataset.add_feature(name, expression=expression)

        for name, result in feature_dataset.feature_results.items():
            if name in dataset.feature_results:
                raise ValueError(f"特征结果名冲突: {name}")
            dataset.add_feature(name, result=result)

    return dataset


def _normalize_symbol_list(vt_symbols: list[str]) -> list[str]:
    """归一化合约代码列表：去空串、标准化格式、去重保序。

    Args:
        vt_symbols: 原始合约代码列表（可含空串或格式不规范的代码）。

    Returns:
        归一化后的合约代码列表（已去空、格式对齐、去重保序）。
    """
    return list(dict.fromkeys(normalize_vt_symbol(item) for item in vt_symbols if item))


def _apply_default_preprocessing(dataset: Any) -> None:
    """对数据集应用默认 Alpha 预处理流水线（Robust Z-Score 标准化 + 缺失值填充）。

    流水线步骤：
    1. 以训练期的有效（无空标签）样本拟合 Robust Z-Score 统计量；
    2. 将标准化后的特征缺失值填 0.0；
    3. 结果写入 ``dataset.infer_df``（全样本）和 ``dataset.learn_df``（过滤空标签）；
    4. 统计量保存到 ``dataset.preprocess_stats``（供信号生成阶段复用）。

    Args:
        dataset: 已调用 ``prepare_data`` 的 ``AlphaDataset`` 实例；
                 ``raw_df`` 须含特征列（列索引 2:-1）与 label 列。

    Returns:
        None。结果通过原地写回入参 ``dataset`` 的 ``infer_df`` /
        ``learn_df`` / ``preprocess_stats`` 三个属性返回。

    Raises:
        ValueError: 数据集无特征列，或过滤空标签后训练数据为空时抛出。
    """
    from ..alpha.dataset import Segment
    from ..alpha.dataset.processor import (
        apply_robust_zscore_stats,
        fill_feature_nan,
        fit_robust_zscore_stats,
    )

    raw_df = dataset.raw_df.sort(["datetime", "vt_symbol"])
    feature_columns = raw_df.columns[2:-1]
    if not feature_columns:
        raise ValueError("数据集未生成任何特征")

    learn_df = raw_df.with_columns(pl.col("label").fill_nan(None)).drop_nulls(subset=["label"])
    if learn_df.is_empty():
        raise ValueError("训练数据为空，请检查样本区间和标签周期设置")

    train_start, train_end = dataset.data_periods[Segment.TRAIN]
    stats = fit_robust_zscore_stats(
        learn_df,
        feature_columns,
        fit_start_time=train_start,
        fit_end_time=train_end,
    )

    dataset.infer_df = fill_feature_nan(
        apply_robust_zscore_stats(raw_df, stats, feature_columns),
        feature_columns,
        fill_value=0.0,
    )
    dataset.learn_df = fill_feature_nan(
        apply_robust_zscore_stats(learn_df, stats, feature_columns),
        feature_columns,
        fill_value=0.0,
    )
    dataset.preprocess_stats = stats


def _is_usable_bar_provider(name: str) -> bool:
    """判断指定数据源是否可用且支持历史 K 线（拒绝 mock）。

    Args:
        name: 数据源名称（如 "tushare" / "akshare" / "mock"）。

    Returns:
        True 表示可用且支持 ``DataCategory.BAR_HISTORY``；"mock" 固定返回 False。
    """
    if name == "mock":
        return False
    provider = datasource_manager.get_provider(name)
    if not provider or provider.get_info(0).status != ProviderStatus.AVAILABLE:
        return False
    return DataCategory.BAR_HISTORY in provider.get_supported_categories()


def _pick_bar_provider(preferred: str = "", asset_class: str = "stock") -> str:
    """为 Alpha 研究数据选择最佳可用数据源（拒绝 mock）。

    品种限制：
    - ETF：跳过 akshare（其 ``_fetch_daily`` 固定调股票接口，对 ETF 必然返回空数据）；
    - 可转债（cbond）：只允许 akshare（tushare ``pro_bar`` 会把 11/12 开头代码误判为股票资产）；
      显式指定 tushare 时抛 RuntimeError，自动选源时跳过 tushare/gateway。

    Args:
        preferred:   优先指定的数据源名（如 "tushare" / "akshare"）；空串为自动选择。
        asset_class: 资产类型，支持 "stock" / "etf" / "cbond"，影响可用源过滤规则。

    Returns:
        选中的数据源名（如 "tushare"）；无可用源时返回空串。

    Raises:
        RuntimeError: preferred 与品种限制不兼容时（如 akshare + etf / tushare + cbond）。
    """
    if preferred:
        if preferred == "akshare" and asset_class == "etf":
            raise RuntimeError(
                "AKShare 数据源不支持 ETF 行情，请改用 Tushare"
            )
        if asset_class == "cbond" and preferred == "tushare":
            raise RuntimeError(
                "Tushare 数据源不支持可转债行情（pro_bar 会误判代码为股票）；"
                "请改用 AKShare 或选择自动模式"
            )
        if _is_usable_bar_provider(preferred):
            return preferred
    if asset_class == "cbond":
        # 可转债只走 akshare
        if _is_usable_bar_provider("akshare"):
            return "akshare"
        return ""
    for name in ("qmt", "tushare", "akshare", "gateway"):
        if name == "akshare" and asset_class == "etf":
            continue
        if _is_usable_bar_provider(name):
            return name
    return ""


def _load_required_local_bar_df(
    lab: Any,
    vt_symbols: list[str],
    interval: str,
    start: date | datetime,
    end: date | datetime,
    extended_days: int = 0,
) -> pl.DataFrame:
    """加载本地 K 线数据并要求全量合约覆盖，任意缺失则抛错。

    Args:
        lab:           AlphaLab 实例，提供 ``load_bar_df`` 方法。
        vt_symbols:    目标合约代码列表（已归一化）。
        interval:      K 线周期（"d" / "1m" / "30m" 等）。
        start:         加载起始日（含），date 或 datetime 均可。
        end:           加载截止日（含），date 或 datetime 均可。
        extended_days: 向前额外加载天数（供特征计算的 look-back 窗口），默认 0。

    Returns:
        合并后的 polars DataFrame，包含所有 vt_symbols 的 K 线数据。

    Raises:
        ValueError: 本地无数据，或指定合约中有任意一只缺失本地 K 线时抛出，
                    错误信息列出所有缺失合约，指引用户先下载/导入。
    """
    vt_symbols = _normalize_symbol_list(vt_symbols)
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=interval,
        start=start,
        end=end,
        extended_days=extended_days,
    )

    if df is None or df.is_empty():
        raise ValueError("无法加载本地K线数据，请先在 Data Prepare 中下载或导入日线数据")

    available_symbols = set(df["vt_symbol"].unique().to_list())
    missing_symbols = [vt_symbol for vt_symbol in vt_symbols if vt_symbol not in available_symbols]
    if missing_symbols:
        raise ValueError(
            "本地缺少以下合约的日线数据，请先准备完整数据后再继续："
            + ", ".join(missing_symbols)
        )

    return df


# =============================================================================
# 业务执行函数（异步任务）
# =============================================================================

def _download_bar_data(
    req: DataDownloadRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """从数据源下载 K 线数据并存为待合并批次（import_batch）。

    下载结果不直接写入正式 K 线存储，而是经 ``save_bars_as_import_batch`` 暂存为
    待合并批次，用户须在「数据资源」界面做连续性/一致性校验后再并入正式资源。
    按合约逐只下载，部分失败不中断整体（best-effort）。

    AKShare 1 分钟线限制：仅支持近 5 个交易日，时间跨度 > 7 天时服务端直接拒绝并报错。

    Args:
        req:         ``DataDownloadRequest``，含合约列表、周期、时间范围、数据源偏好等。
        on_progress: 可选进度回调 ``(progress: float, message: str)``，
                     每只合约完成后回调一次（0–100）。

    Returns:
        包含 total / success / failed / failed_symbols / provider / batches / saved_as 的 dict。

    Raises:
        RuntimeError: 数据源不支持指定品种/周期，或无任何可用真实数据源时抛出。
    """
    from .alpha import _get_alpha_lab, _normalize_market_interval

    lab = _get_alpha_lab()
    vt_symbols = _normalize_symbol_list(req.vt_symbols)

    if req.data_kind != "bar":
        raise RuntimeError("当前版本仅支持原始 K 线下载，历史 Tick 请通过导入功能准备")

    requested_interval = _normalize_market_interval(req.source_interval or req.interval or "d")
    provider_interval_map = {
        "d": "d",
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "w": "w",
    }
    provider_interval = provider_interval_map.get(requested_interval)
    if not provider_interval:
        raise RuntimeError(f"当前不支持下载周期 {requested_interval}")

    requested_provider = (req.provider or "").strip()
    asset_class = getattr(req, "asset_class", "stock") or "stock"
    provider_name = _pick_bar_provider(requested_provider, asset_class=asset_class)
    if not provider_name:
        if asset_class == "etf":
            raise RuntimeError(
                "ETF 下载当前需要 Tushare 数据源（请配置 TUSHARE_TOKEN）"
            )
        if asset_class == "cbond":
            raise RuntimeError(
                "可转债下载需要 AKShare 数据源（请确认已安装 akshare 依赖且 AKSHARE_ENABLED=true）"
            )
        raise RuntimeError(
            "没有可用的真实数据源。请配置 Tushare token 或启用 AKShare。"
            "Alpha 研究需要真实历史数据，不支持 Mock 数据。"
        )

    # AKShare 的 1 分钟线仅提供近 5 个交易日，跨度过长会静默截断；服务端直接拦截并指引。
    if provider_name == "akshare" and requested_interval == "1m":
        def _as_date(value):
            """把 datetime 归一化为 date；非 datetime 值原样返回。"""
            return value.date() if isinstance(value, datetime) else value

        try:
            span_days = (_as_date(req.end) - _as_date(req.start)).days
        except Exception:
            span_days = 0
        if span_days > 7:
            raise RuntimeError(
                "AKShare 的 1 分钟线仅支持近 5 个交易日，所选时间范围过长会被截断。"
                "请将范围缩短到最近 5 个交易日内，或改用 5m 及以上周期 / Tushare 数据源。"
            )

    total = len(vt_symbols)
    success_count = 0
    failed_symbols = []
    batches: list[dict[str, Any]] = []

    for i, vt_symbol in enumerate(vt_symbols):
        try:
            symbol, exchange_str = vt_symbol.rsplit(".", 1)

            records = datasource_manager.get_bar_history(
                symbol=symbol,
                exchange=exchange_str,
                interval=provider_interval,
                start=datetime.combine(req.start, datetime.min.time()),
                end=datetime.combine(req.end, datetime.max.time()),
                provider_name=provider_name,
            )

            if records:
                from ..alpha.lab import BarData
                bars = []
                for r in records:
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange_str,
                        datetime=r.datetime,
                        interval=requested_interval,
                        open_price=r.open_price,
                        high_price=r.high_price,
                        low_price=r.low_price,
                        close_price=r.close_price,
                        volume=r.volume,
                        turnover=r.turnover,
                        open_interest=r.open_interest,
                    )
                    bars.append(bar)

                # 下载数据先入待合并批次，不直接写正式 K 线；记录复权口径与来源，
                # 由用户在合并环节做连续性/一致性校验后再并入正式资源。
                adjust_type = getattr(records[0], "adjust_type", "none") or "none"
                batch = lab.save_bars_as_import_batch(
                    bars,
                    adjust_type=adjust_type,
                    source="download",
                    file_name=f"{provider_name}_{requested_interval}_{req.start}_{req.end}",
                    extra_meta={"asset_class": asset_class},
                )
                batches.append(batch)
                success_count += 1
            else:
                failed_symbols.append(f"{vt_symbol}: 数据源无数据")
        except Exception as e:
            failed_symbols.append(f"{vt_symbol}: {str(e)}")

        if on_progress:
            progress = (i + 1) / total * 100
            on_progress(progress, f"已下载 {vt_symbol} ({i + 1}/{total})")

    # 末条进度显式汇总成功/失败，避免「部分失败仍显示完成」误导用户。
    if on_progress:
        suffix = "，数据已存为待合并批次，请在「数据资源」中合并到正式 K 线"
        if failed_symbols:
            on_progress(
                100,
                f"下载完成：成功 {success_count}/{total}，失败 {len(failed_symbols)} 个 —— "
                + "；".join(failed_symbols)
                + (suffix if success_count else ""),
            )
        else:
            on_progress(100, f"下载完成：成功 {success_count}/{total}{suffix}")

    return {
        "total": total,
        "success": success_count,
        "failed": len(failed_symbols),
        "failed_symbols": failed_symbols,
        "provider": provider_name,
        "batches": batches,
        "saved_as": "import_batch",
    }


def _aggregate_data(
    req: DataAggregateRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """将本地原始 K 线聚合为派生周期 K 线（如 1m → 5m / 30m）。

    调用 ``AlphaLab.aggregate_market_data`` 完成聚合，结果写入 AlphaLab 存储。

    Args:
        req:         ``DataAggregateRequest``，含合约列表、源/目标周期、时间范围、
                     盘中会话配置等。
        on_progress: 可选进度回调 ``(progress: float, message: str)``。

    Returns:
        聚合结果 dict，含 success / total / target_interval 等字段（由 AlphaLab 返回）。
    """
    from .alpha import _get_alpha_lab

    lab = _get_alpha_lab()
    vt_symbols = _normalize_symbol_list(req.vt_symbols)
    total = max(len(vt_symbols), 1)

    if on_progress:
        on_progress(10, "开始本地聚合...")

    result = lab.aggregate_market_data(
        vt_symbols=vt_symbols,
        source_kind=req.source_kind,
        source_interval=req.source_interval,
        target_interval=req.target_interval,
        start=req.start,
        end=req.end,
        session_profile=req.session_profile,
    )

    if on_progress:
        on_progress(
            100,
            f"聚合完成: 成功 {result['success']}/{total}, 目标周期 {result['target_interval']}",
        )

    return result


def _import_parquet_session(
    session_id: str,
    *,
    data_kind: str,
    interval: str,
    import_mode: str = "merge",
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """逐个导入某上传会话的暂存 parquet 文件为待合并批次，并报进度。

    遍历 ``_get_staging().list_files(session_id)``，对每个文件调
    ``AlphaLab.import_parquet_path`` 落为待合并批次；单文件失败（非 parquet / 坏文件 /
    缺列 / 代码无法识别）被捕获并计入 ``failed_files``，不影响其余文件。进度照
    ``_download_bar_data`` 的 ``(i+1)/total*100`` 范式上报。全部处理完后删除该会话暂存目录。

    Args:
        session_id: 上传会话标识（由 stage 端点生成）。
        data_kind: ``"bar"`` 或 ``"tick"``。
        interval: K 线周期；``data_kind="tick"`` 时由底层归一为 ``"tick"``。
        import_mode: 导入阶段忽略（批次一律 pending）；合并/替换语义在后续「批次合并」环节再选择。
        on_progress: 可选进度回调 ``(progress: float, message: str)``。

    Returns:
        dict，含 ``total``/``success``/``failed``/``failed_files``/``batches``/``saved_as``。
    """
    from .alpha import _get_alpha_lab, _get_staging

    lab = _get_alpha_lab()
    staging = _get_staging()
    files = staging.list_files(session_id)
    total = max(len(files), 1)

    success = 0
    batches: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for i, staged in enumerate(files):
        try:
            if not staged.is_parquet:
                raise ValueError("非 parquet 文件")
            result = lab.import_parquet_path(
                staged.path, data_kind=data_kind, interval=interval, file_name=staged.file_name
            )
            if result.get("success"):
                success += 1
                batches.extend(result.get("batches", []))
            else:
                failed_files.append({"file": staged.file_name, "reason": result.get("error", "导入失败")})
        except Exception as exc:
            failed_files.append({"file": staged.file_name, "reason": str(exc)})

        if on_progress:
            on_progress((i + 1) / total * 100, f"已导入 {staged.file_name} ({i + 1}/{len(files)})")

    staging.discard(session_id)

    if on_progress:
        on_progress(
            100,
            f"导入完成：成功 {success}/{len(files)}，失败 {len(failed_files)} —— "
            "数据已存为待合并批次，请在「批次」中合并到正式资源",
        )

    return {
        "total": len(files),
        "success": success,
        "failed": len(failed_files),
        "failed_files": failed_files,
        "batches": batches,
        "saved_as": "import_batch",
    }


def _create_dataset(
    req: DatasetCreateRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """创建 AlphaDataset：加载 K 线 → 构建特征 → 设置标签 → 预处理 → 持久化。

    完整流水线（含 ``prepare_data`` 特征计算与默认预处理）。数据集存储到
    AlphaLab，可直接用于后续模型训练。

    Args:
        req:         ``DatasetCreateRequest``，含合约列表、特征库、时间三段
                     （start/train_end/end）、标签周期（label_period）、数据集名等。
        on_progress: 可选进度回调 ``(progress: float, message: str)``，
                     逐阶段（10→90）回调。

    Returns:
        ``{"name": str, "feature_count": int, "sample_count": int}``。

    Raises:
        ValueError: 日期段顺序非法、本地缺少 K 线数据，或特征库/标签计算失败时抛出。
    """
    from .alpha import _get_alpha_lab

    lab = _get_alpha_lab()
    vt_symbols = _normalize_symbol_list(req.vt_symbols)

    if req.start >= req.train_end:
        raise ValueError(f"数据起始日期({req.start})必须早于训练截止日期({req.train_end})")
    if req.train_end >= req.end:
        raise ValueError(f"训练截止日期({req.train_end})必须早于数据结束日期({req.end})")

    if on_progress:
        on_progress(10, "加载K线数据...")

    valid_end = req.valid_end or (req.train_end + timedelta(days=90))
    if valid_end >= req.end:
        valid_end = req.train_end + (req.end - req.train_end) // 2

    df = _load_required_local_bar_df(
        lab=lab,
        vt_symbols=vt_symbols,
        interval="d",
        start=req.start,
        end=req.end,
        extended_days=100,
    )

    if on_progress:
        on_progress(30, "初始化数据集...")

    train_period = (str(req.start), str(req.train_end))
    valid_period = (str(req.train_end + timedelta(days=1)), str(valid_end))
    test_period = (str(valid_end + timedelta(days=1)), str(req.end))

    dataset = _build_feature_dataset(
        df=df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        feature_names=req.features,
    )
    dataset.feature_libraries = list(req.features)

    if on_progress:
        on_progress(40, "设置标签...")

    dataset.set_label(f"ts_delay(close, -{req.label_period}) / ts_delay(close, -1) - 1")

    if on_progress:
        on_progress(50, "计算特征（可能需要几分钟）...")

    dataset.prepare_data(max_workers=max(1, MAX_WORKERS))

    if on_progress:
        on_progress(80, "数据预处理...")

    _apply_default_preprocessing(dataset)

    if on_progress:
        on_progress(90, "保存数据集...")

    lab.save_dataset(req.name, dataset)

    return {
        "name": req.name,
        "feature_count": len(dataset.feature_expressions),
        "sample_count": len(dataset.learn_df) if hasattr(dataset, "learn_df") else 0
    }


def _train_model(
    req: ModelTrainRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """训练机器学习模型并持久化到 AlphaLab。

    支持 lgb / mlp / lasso 三种模型类型；对 mlp 自动处理 hidden_sizes 类型转换
    与 device 选择（CUDA 优先）；对 lgb 将 n_estimators 重命名为 num_boost_round。
    训练后将 dataset_name / preprocess_stats / feature_libraries 作为元信息附加到模型，
    供信号生成阶段复用。

    Args:
        req:         ``ModelTrainRequest``，含数据集名、模型名、模型类型、超参数等。
        on_progress: 可选进度回调 ``(progress: float, message: str)``。

    Returns:
        ``{"name": str, "model_type": str}``。

    Raises:
        ValueError: 数据集不存在、模型类型不支持，或训练数据为空时抛出（含中文指引）。
    """
    from ..alpha.model.models.lgb_model import LgbModel
    from ..alpha.model.models.mlp_model import MlpModel
    from ..alpha.model.models.lasso_model import LassoModel

    from .alpha import _get_alpha_lab

    lab = _get_alpha_lab()

    if on_progress:
        on_progress(10, "加载数据集...")

    dataset = lab.load_dataset(req.dataset)
    if dataset is None:
        raise ValueError(f"数据集 {req.dataset} 不存在")

    if on_progress:
        on_progress(20, "初始化模型...")

    model_classes = {
        "lgb": LgbModel,
        "mlp": MlpModel,
        "lasso": LassoModel
    }

    model_class = model_classes.get(req.model_type)
    if model_class is None:
        raise ValueError(f"不支持的模型类型: {req.model_type}")

    params = dict(req.params)
    if req.model_type == "lgb":
        if "n_estimators" in params and "num_boost_round" not in params:
            params["num_boost_round"] = params.pop("n_estimators")
    elif req.model_type == "mlp":
        hidden_sizes = params.get("hidden_sizes", params.get("hidden_layer_sizes"))
        if isinstance(hidden_sizes, str):
            params["hidden_sizes"] = tuple(
                int(part.strip()) for part in hidden_sizes.split(",") if part.strip()
            )
        elif isinstance(hidden_sizes, list):
            params["hidden_sizes"] = tuple(int(part) for part in hidden_sizes)

        params.pop("hidden_layer_sizes", None)
        params.setdefault("input_size", len(dataset.learn_df.columns[2:-1]))
        try:
            import torch
            params.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            params.setdefault("device", "cpu")

    model = model_class(**params)

    if on_progress:
        on_progress(30, "开始训练...")

    try:
        model.fit(dataset)
    except Exception as e:
        error_msg = str(e)
        if "non empty" in error_msg or "empty" in error_msg.lower():
            raise ValueError(
                "训练数据为空。可能原因：\n"
                "1. 数据集的时间范围设置不合理（start >= train_end）\n"
                "2. 下载的K线数据量不足\n"
                "3. 特征计算后有效样本被过滤掉\n"
                "请检查数据集创建时的日期参数。"
            ) from e
        raise

    if on_progress:
        on_progress(90, "保存模型...")

    setattr(model, "dataset_name", req.dataset)
    setattr(model, "preprocess_stats", getattr(dataset, "preprocess_stats", {}))
    setattr(model, "feature_libraries", getattr(dataset, "feature_libraries", []))

    lab.save_model(req.name, model)

    return {
        "name": req.name,
        "model_type": req.model_type
    }


def _generate_signal(
    req: SignalGenerateRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """使用已训练模型对指定区间产生逐日交易信号，并持久化到 AlphaLab。

    信号生成时复用模型随附的 preprocess_stats 和 feature_expressions，
    保证信号与训练期的标准化口径一致（不重新拟合统计量）。
    预测结果对齐到 infer_df（TEST 段），逐行与 (datetime, vt_symbol) 索引拼接。

    Args:
        req:         ``SignalGenerateRequest``，含模型名、合约列表、信号名、时间范围等。
        on_progress: 可选进度回调 ``(progress: float, message: str)``。

    Returns:
        ``{"name": str, "row_count": int}``。

    Raises:
        ValueError: 模型/模板数据集不存在、缺少预处理统计量、本地 K 线缺失，
                    或预测与索引长度不一致时抛出。
    """
    from ..alpha.dataset import AlphaDataset, Segment
    from ..alpha.dataset.processor import apply_robust_zscore_stats, fill_feature_nan

    from .alpha import _get_alpha_lab

    lab = _get_alpha_lab()
    vt_symbols = _normalize_symbol_list(req.vt_symbols)

    if on_progress:
        on_progress(10, "加载模型...")

    model = lab.load_model(req.model)
    if model is None:
        raise ValueError(f"模型 {req.model} 不存在")

    template_dataset_name = getattr(model, "dataset_name", "")
    if not template_dataset_name:
        raise ValueError("模型缺少训练数据集信息，请重新训练模型")

    template_dataset = lab.load_dataset(template_dataset_name)
    if template_dataset is None:
        raise ValueError(f"训练数据集 {template_dataset_name} 不存在")

    if on_progress:
        on_progress(20, "加载K线数据...")

    df = _load_required_local_bar_df(
        lab=lab,
        vt_symbols=vt_symbols,
        interval="d",
        start=req.start,
        end=req.end,
        extended_days=100,
    )

    if on_progress:
        on_progress(40, "准备数据集...")

    train_period = (str(req.start), str(req.start))
    valid_period = (str(req.start), str(req.start))
    test_period = (str(req.start), str(req.end))

    dataset = AlphaDataset(
        df=df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period
    )
    dataset.feature_expressions = dict(template_dataset.feature_expressions)
    dataset.feature_results = dict(getattr(template_dataset, "feature_results", {}))
    dataset.set_label(template_dataset.label_expression)

    if on_progress:
        on_progress(60, "计算特征...")

    dataset.prepare_data(max_workers=max(1, MAX_WORKERS))
    dataset.raw_df = dataset.raw_df.sort(["datetime", "vt_symbol"])
    feature_columns = dataset.raw_df.columns[2:-1]
    preprocess_stats = getattr(model, "preprocess_stats", {}) or getattr(template_dataset, "preprocess_stats", {})
    if not preprocess_stats:
        raise ValueError("模型缺少预处理统计量，请重新训练模型")

    dataset.infer_df = fill_feature_nan(
        apply_robust_zscore_stats(dataset.raw_df, preprocess_stats, feature_columns),
        feature_columns,
        fill_value=0.0,
    )
    dataset.learn_df = dataset.infer_df.with_columns(pl.col("label").fill_nan(None)).drop_nulls(subset=["label"])

    if on_progress:
        on_progress(80, "生成预测...")

    predictions = model.predict(dataset, Segment.TEST)

    index_df = dataset.fetch_infer(Segment.TEST).sort(["datetime", "vt_symbol"]).select(["datetime", "vt_symbol"])
    if len(index_df) != len(predictions):
        raise ValueError("预测结果与信号索引长度不一致")

    signal_df = index_df.with_columns(
        pl.Series(name="signal", values=predictions)
    )

    if on_progress:
        on_progress(90, "保存信号...")

    lab.save_signal(req.name, signal_df)

    return {
        "name": req.name,
        "row_count": len(signal_df)
    }


def _run_backtest(
    req: BacktestRunRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """基于已有信号运行策略回测，返回统计指标、成交明细与逐日净值曲线。

    使用 ``BacktestingEngine + EquityDemoStrategy``；基准（买入持有）优先取
    ``req.benchmark``，单标的回测时其次取该标的本身。无成交记录时返回零值统计
    与空 trades/equity_curve（不报错），供前端正常展示。

    Args:
        req:         ``BacktestRunRequest``，含信号名、回测区间、资金、基准标的等。
        on_progress: 可选进度回调 ``(progress: float, message: str)``。

    Returns:
        包含 name / target_symbol / statistics / trades / equity_curve 的 dict。
        ``statistics`` 包含夏普、年化收益、最大回撤、基准对比等指标。

    Raises:
        ValueError: 信号不存在、指定区间内无信号，或本地缺少回测所需历史 K 线时抛出。
    """
    from ..alpha.strategy import BacktestingEngine
    from ..alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy
    from ..backtest.artifacts import (
        serialize_trades,
        serialize_equity_curve,
        extract_benchmark_prices,
        attach_benchmark_returns,
        summarize_benchmark,
    )

    from .alpha import _get_alpha_lab

    lab = _get_alpha_lab()

    if on_progress:
        on_progress(10, "加载信号...")

    signal_df = lab.load_signal(req.signal)
    if signal_df is None:
        raise ValueError(f"信号 {req.signal} 不存在")
    signal_df = signal_df.filter(
        (pl.col("datetime") >= datetime.combine(req.start, datetime.min.time())) &
        (pl.col("datetime") <= datetime.combine(req.end, datetime.max.time()))
    )
    if signal_df.is_empty():
        raise ValueError("指定回测区间内没有可用信号")
    signal_df = signal_df.with_columns(
        pl.col("vt_symbol").map_elements(normalize_vt_symbol, return_dtype=pl.Utf8)
    )

    vt_symbols = _normalize_symbol_list(signal_df["vt_symbol"].unique().to_list())

    if on_progress:
        on_progress(20, "初始化回测引擎...")

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval="d",
        start=datetime.combine(req.start, datetime.min.time()),
        end=datetime.combine(req.end, datetime.max.time()),
        capital=req.capital
    )

    for vt_symbol in vt_symbols:
        if vt_symbol not in engine.sizes:
            engine.sizes[vt_symbol] = 1
            engine.priceticks[vt_symbol] = 0.01
            engine.long_rates[vt_symbol] = 0.0003
            engine.short_rates[vt_symbol] = 0.0003

    if on_progress:
        on_progress(40, "加载历史数据...")

    engine.add_strategy(EquityDemoStrategy, {}, signal_df)
    engine.load_data()
    loaded_symbols = {vt_symbol for _, vt_symbol in engine.history_data.keys()}
    missing_history = [vt_symbol for vt_symbol in vt_symbols if vt_symbol not in loaded_symbols]
    if missing_history:
        raise ValueError(
            "回测缺少以下合约的历史日线数据，请先准备数据："
            + ", ".join(missing_history)
        )

    if on_progress:
        on_progress(60, "运行回测...")

    engine.run_backtesting()

    if on_progress:
        on_progress(80, "计算结果...")

    result = engine.calculate_result()

    if result is None:
        return {
            "name": req.name,
            "statistics": {
                "start_date": str(req.start),
                "end_date": str(req.end),
                "total_days": 0,
                "profit_days": 0,
                "loss_days": 0,
                "capital": req.capital,
                "end_balance": req.capital,
                "total_return": 0,
                "annual_return": 0,
                "max_drawdown": 0,
                "max_ddpercent": 0,
                "sharpe_ratio": 0,
                "total_trade_count": 0,
                "total_net_pnl": 0,
                "total_commission": 0,
                "error": "回测期间无成交记录，策略未产生交易信号",
            },
            # 字段恒在：无成交即空成交列表与空净值曲线
            "trades": [],
            "equity_curve": [],
        }

    statistics = engine.calculate_statistics()

    # 基准（买入持有标的）：优先用请求指定的 benchmark，其次单标的回测取该标的本身。
    benchmark_symbol = normalize_vt_symbol(req.benchmark) if req.benchmark else (vt_symbols[0] if len(vt_symbols) == 1 else None)
    equity_curve = serialize_equity_curve(engine.daily_df)
    benchmark_prices = extract_benchmark_prices(engine.daily_results, benchmark_symbol)
    attach_benchmark_returns(equity_curve, benchmark_prices, req.capital)
    statistics.update(summarize_benchmark(equity_curve, benchmark_symbol))

    return {
        "name": req.name,
        "target_symbol": benchmark_symbol,
        "statistics": statistics,
        # 成交明细与逐日净值序列：equity_curve 必须在 calculate_statistics() 之后取
        # （此时 engine.daily_df 才补入 balance/drawdown 列）
        "trades": serialize_trades(engine.trades),
        "equity_curve": equity_curve,
    }
