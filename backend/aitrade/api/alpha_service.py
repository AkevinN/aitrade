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
    """Build a combined feature dataset using selected feature libraries."""
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
    return list(dict.fromkeys(normalize_vt_symbol(item) for item in vt_symbols if item))


def _apply_default_preprocessing(dataset: Any) -> None:
    """Apply the default Alpha preprocessing pipeline."""
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
    """判断指定数据源是否可用且支持历史K线（拒绝 mock）。"""
    if name == "mock":
        return False
    provider = datasource_manager.get_provider(name)
    if not provider or provider.get_info(0).status != ProviderStatus.AVAILABLE:
        return False
    return DataCategory.BAR_HISTORY in provider.get_supported_categories()


def _pick_bar_provider(preferred: str = "") -> str:
    """为 Alpha 研究数据选择最佳数据源（拒绝 mock）。

    指定 preferred 且其可用时优先返回；否则按优先级自动选择。
    """
    if preferred and _is_usable_bar_provider(preferred):
        return preferred
    for name in ("tushare", "akshare", "gateway"):
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
    """Load local bar data and require full symbol coverage."""
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
    """从数据源下载K线数据，保存到 AlphaLab。"""
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
    provider_name = _pick_bar_provider(requested_provider)
    if not provider_name:
        raise RuntimeError(
            "没有可用的真实数据源。请配置 Tushare token 或启用 AKShare。"
            "Alpha 研究需要真实历史数据，不支持 Mock 数据。"
        )

    # AKShare 的 1 分钟线仅提供近 5 个交易日，跨度过长会静默截断；服务端直接拦截并指引。
    if provider_name == "akshare" and requested_interval == "1m":
        def _as_date(value):
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
    """聚合本地原始数据为派生K线。"""
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


def _create_dataset(
    req: DatasetCreateRequest,
    on_progress: Optional[Callable[[float, str], None]] = None
) -> dict[str, Any]:
    """创建数据集（含特征计算）。"""
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
    """训练机器学习模型。"""
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
    """使用训练好的模型生成交易信号。"""
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
    """基于信号运行策略回测。"""
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
