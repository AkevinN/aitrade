"""
CNN 卷积神经网络量化预测 API 路由。

提供 CNN 模型的训练、查询、删除功能，
复用现有 TaskManager 实现异步任务管理。
"""

from typing import Any, Callable, Optional

import polars as pl
from fastapi import APIRouter, HTTPException

from ..alpha.lab_utils import normalize_vt_symbol
from ..models import CNNTrainRequest, CNNBacktestRequest, CNNPredictRequest, TaskType
from ..models.governance import (
    CNNCandidateTrainRequest,
    CNNGovernanceConfig,
    CNNGovernanceReplayRequest,
    CNNPromotionRequest,
    CNNRollbackRequest,
    CNNWalkForwardRequest,
)
from ..task import task_manager

router = APIRouter(prefix="/api/cnn", tags=["CNN量化预测"])


def _normalize_symbol_list(vt_symbols: list[str]) -> list[str]:
    """规范化证券代码列表并去重（保持首次出现顺序）。

    Args:
        vt_symbols: 原始证券代码列表，允许含空项。

    Returns:
        去重后的规范化代码列表。
    """
    return list(dict.fromkeys(normalize_vt_symbol(item) for item in vt_symbols if item))


# =============================================================================
# 模型治理（WF/OOS、候选、晋级、回滚、治理回放）
# =============================================================================

@router.get("/governance/config")
async def get_governance_config() -> dict:
    """读取 CNN 治理配置（晋级门禁参数、调度策略等）。"""
    from ..cnn.governance import store
    return store.get_config()


@router.put("/governance/config")
async def update_governance_config(req: CNNGovernanceConfig) -> dict:
    """更新 CNN 治理配置并追加 config_updated 历史事件。"""
    from ..cnn.governance import store
    return store.save_config(req)


@router.get("/governance/production")
async def get_governance_production() -> dict:
    """读取当前生产模型信息（model_name/version/promoted_at 等）。"""
    from ..cnn.governance import store
    return store.get_production()


@router.get("/governance/candidates")
async def list_governance_candidates() -> list[dict]:
    """列出所有候选模型，按创建时间降序排列。"""
    from ..cnn.governance import store
    return store.list_candidates()


@router.get("/governance/candidates/{candidate_id}")
async def get_governance_candidate(candidate_id: str) -> dict:
    """读取指定候选模型的元数据（含 WF 报告 ID 与训练结果摘要）。"""
    from ..cnn.governance import store
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(404, f"候选不存在: {candidate_id}")
    return candidate


@router.post("/governance/evaluate")
async def start_governance_evaluate(req: CNNWalkForwardRequest) -> dict:
    """启动 WF/OOS 评估任务，异步执行并立即返回 task_id。"""
    task_id = task_manager.create_task(
        TaskType.CNN_WF_EVALUATE,
        params=req.model_dump(mode="json"),
        title="CNN WF/OOS 评估",
        entity_type="cnn_governance_report",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        from ..cnn.governance import run_walk_forward_evaluate
        return run_walk_forward_evaluate(req, on_progress=on_progress)

    task_manager.run_async(task_id, _run, enable_progress=True)
    return {"task_id": task_id, "name": req.name}


@router.post("/governance/candidates/train")
async def start_governance_candidate_train(req: CNNCandidateTrainRequest) -> dict:
    """启动候选模型训练任务（WF/OOS 评估 + 最终模型训练），异步执行并立即返回 task_id。"""
    task_id = task_manager.create_task(
        TaskType.CNN_CANDIDATE_TRAIN,
        params=req.model_dump(mode="json"),
        title="CNN 候选模型训练",
        entity_type="cnn_candidate",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        from ..cnn.governance import train_candidate
        return train_candidate(req, on_progress=on_progress)

    task_manager.run_async(task_id, _run, enable_progress=True)
    return {"task_id": task_id, "name": req.name}


@router.post("/governance/candidates/{candidate_id}/promote")
async def promote_governance_candidate(candidate_id: str, req: CNNPromotionRequest) -> dict:
    """将指定候选模型晋级为生产模型（同步执行）。"""
    from ..cnn.governance import promote_candidate
    try:
        return promote_candidate(candidate_id, req)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/governance/candidates/{candidate_id}/reject")
async def reject_governance_candidate(candidate_id: str, req: CNNPromotionRequest) -> dict:
    """拒绝指定候选模型，更新其状态为 rejected（同步执行）。"""
    from ..cnn.governance import reject_candidate
    try:
        return reject_candidate(candidate_id, note=req.note)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/governance/rollback")
async def rollback_governance_production(req: CNNRollbackRequest) -> dict:
    """将生产模型回滚到上一版本（或指定模型），同步执行。"""
    from ..cnn.governance import rollback_production
    try:
        return rollback_production(req)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/governance/history")
async def get_governance_history() -> list[dict]:
    """读取全量治理历史事件列表（JSONL，按写入时间升序）。"""
    from ..cnn.governance import store
    return store.history()


@router.get("/governance/reports/{report_id}")
async def get_governance_report(report_id: str) -> dict:
    """读取指定 WF/OOS 评估报告（含各折统计与晋级门禁结果）。"""
    from ..cnn.governance import store
    report = store.get_report(report_id)
    if report is None:
        raise HTTPException(404, f"报告不存在: {report_id}")
    return report


@router.post("/governance/replay/run")
async def start_governance_replay(req: CNNGovernanceReplayRequest) -> dict:
    """启动治理回放回测任务，在历史区间对比三条基线策略，异步执行并立即返回 task_id。"""
    task_id = task_manager.create_task(
        TaskType.CNN_GOVERNANCE_REPLAY,
        params=req.model_dump(mode="json"),
        title="CNN 治理回放回测",
        entity_type="cnn_governance_replay",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        from ..cnn.governance import run_governance_replay
        return run_governance_replay(req, on_progress=on_progress)

    task_manager.run_async(task_id, _run, enable_progress=True)
    return {"task_id": task_id, "name": req.name}


@router.get("/governance/replay")
async def list_governance_replays() -> list[dict]:
    """列出所有治理回放报告，按创建时间降序排列。"""
    from ..cnn.governance import store
    return store.list_replay_reports()


@router.get("/governance/replay/{replay_id}")
async def get_governance_replay(replay_id: str) -> dict:
    """读取指定治理回放报告（含三条基线对比、晋级事件与结论）。"""
    from ..cnn.governance import store
    replay = store.get_replay_report(replay_id)
    if replay is None:
        raise HTTPException(404, f"治理回放报告不存在: {replay_id}")
    return replay


# =============================================================================
# Torch check helpers
# =============================================================================

def _check_torch() -> bool:
    """检查 PyTorch 是否可用。"""
    try:
        import torch
        return True
    except ImportError:
        return False


def _get_device() -> str:
    """返回 PyTorch 可用的设备（cuda/cpu）。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "N/A"


# =============================================================================
# 状态检查
# =============================================================================

@router.get("/status")
async def get_cnn_status() -> dict:
    """检查 CNN 功能是否可用（PyTorch 是否安装）。"""
    torch_ok = _check_torch()
    return {
        "torch_installed": torch_ok,
        "device": _get_device() if torch_ok else "N/A",
    }


# =============================================================================
# 训练管理
# =============================================================================

@router.post("/train")
async def start_cnn_train(req: CNNTrainRequest) -> dict:
    """
    启动 CNN 训练任务。

    通过 TaskManager 在后台线程执行，立即返回 task_id。
    前端可通过 /api/alpha/tasks/{task_id} 轮询进度。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，请先执行: pip install torch")

    if req.start >= req.end:
        raise HTTPException(400, "开始日期必须早于结束日期")
    if not 0 < req.train_ratio < 1:
        raise HTTPException(400, "train_ratio 必须在 0 和 1 之间")
    if req.input_data_kind not in {"bar", "tick"}:
        raise HTTPException(400, "input_data_kind 必须是 bar 或 tick")

    symbols_from_groups = [
        normalize_vt_symbol(symbol)
        for group in req.observation_groups
        for symbol in group.symbols
    ]
    target_symbol = normalize_vt_symbol(req.target_symbol) if req.target_symbol else None
    effective_symbols = _normalize_symbol_list([*(req.vt_symbols or []), *symbols_from_groups])
    if target_symbol and target_symbol not in effective_symbols:
        effective_symbols = [target_symbol, *effective_symbols]
    if not effective_symbols:
        raise HTTPException(400, "至少需要提供 target_symbol 或观测证券列表")

    input_interval = str(req.input_interval).lower()
    if input_interval not in {"d", "1m", "5m", "10m", "15m", "30m", "60m"}:
        raise HTTPException(400, "input_interval 仅支持 d/1m/5m/10m/15m/30m/60m")
    if req.input_data_kind == "tick" and input_interval == "d":
        raise HTTPException(400, "Tick 输入目前仅支持分钟周期，请选择 1m/5m/10m/15m/30m/60m")

    label_spec = req.label_spec.model_dump() if hasattr(req.label_spec, "model_dump") else dict(req.label_spec)
    if label_spec.get("mode") == "horizon_bars" and not label_spec.get("horizon"):
        raise HTTPException(400, "horizon_bars 模式必须提供 horizon")
    if label_spec.get("mode") == "oco" and (
        not label_spec.get("take_profit") or not label_spec.get("stop_loss")
    ):
        raise HTTPException(400, "oco 模式必须提供正的 take_profit 与 stop_loss（如 0.03 表示 3%）")

    # Property 6 API 侧：path_class objective 必须配合 oco label，否则路径标签无法构建四类标注
    if req.objective == "path_class" and label_spec.get("mode") != "oco":  # 与上方 "oco" 字符串风格一致（LabelMode.OCO.value == "oco"，等价）
        raise HTTPException(
            400,
            "objective=path_class 需要 label_spec.mode=oco（路径标签依赖三重障碍判定）",
        )

    task_id = task_manager.create_task(
        TaskType.CNN_TRAIN,
        params={"name": req.name, "symbols": effective_symbols},
        title="训练 CNN 模型",
        entity_type="cnn_model",
        entity_name=req.name,
    )

    def _run(on_progress: Optional[Callable[[float, str], None]] = None) -> dict[str, Any]:
        from ..cnn import train_cnn_model
        return train_cnn_model(
            name=req.name,
            vt_symbols=effective_symbols,
            start=req.start,
            end=req.end,
            target_symbol=target_symbol,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            lookback=req.lookback,
            dropout=req.dropout,
            train_ratio=req.train_ratio,
            observation_groups=[
                {
                    **group.model_dump(),
                    "symbols": _normalize_symbol_list(group.symbols),
                }
                for group in req.observation_groups
            ],
            input_data_kind=req.input_data_kind,
            input_interval=input_interval,
            label_spec=label_spec,
            loss_weighting=req.loss_weighting,
            objective=req.objective,
            on_progress=on_progress,
        )

    task_manager.run_async(task_id, _run, enable_progress=True)

    return {"task_id": task_id, "name": req.name}


# =============================================================================
# 模型管理
# =============================================================================

@router.get("/models")
async def list_models() -> list[dict]:
    """列出已保存的 CNN 模型。"""
    from ..cnn import list_cnn_models
    return list_cnn_models()


@router.get("/models/{name}")
async def get_model_detail(name: str) -> dict:
    """获取模型详情（含训练历史）。"""
    from ..cnn import get_cnn_model_detail
    try:
        return get_cnn_model_detail(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/models/{name}/architecture")
async def get_model_architecture(name: str) -> dict:
    """探查模型的真实网络结构（重建实例 + 加载权重 + 逐层形状）。"""
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，无法探查模型结构")
    from ..cnn import describe_cnn_architecture
    try:
        return describe_cnn_architecture(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@router.delete("/models/{name}")
async def delete_model(name: str) -> dict:
    """删除 CNN 模型。"""
    from ..cnn import delete_cnn_model
    ok = delete_cnn_model(name)
    if not ok:
        raise HTTPException(404, f"模型不存在: {name}")
    return {"deleted": name}


# =============================================================================
# 回测
# =============================================================================

def _run_cnn_backtest(
    req: CNNBacktestRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """执行 CNN 模型回测：推理生成信号 → 共享回测引擎 → 返回统计+成交+净值。

    核心步骤：
    1. 校验模型文件存在；
    2. 调用 predict_cnn_signals 生成 [datetime, vt_symbol, signal] 信号表；
    3. 解析出场配置（auto=按 label 自动推导），校验 label↔策略一致性；
    4. 初始化 BacktestingEngine，设置成本参数与 fill_price_mode；
    5. 运行回测并计算统计，附买入持有基准收益。

    隐性失败守护：信号与行情 datetime 无交集时直接抛 ValueError 而非返回空结果。

    Args:
        req: CNN 回测请求，含 model/start/end/capital/exit_mode/buy_threshold 等参数。
        on_progress: 进度回调 ``(percent, message)``，可为 None。

    Returns:
        字典，含 name/model/target_symbol/statistics/trades/equity_curve；
        无信号或无成交时 statistics 含 error 键，trades/equity_curve 为空列表。

    Raises:
        ValueError: CNN 模型不存在、信号与行情无交集、label↔策略硬性不一致时抛出。
    """
    import logging
    from datetime import datetime

    from ..alpha import AlphaLab
    from ..config import ALPHA_LAB_PATH
    from ..cnn.predictor import predict_cnn_signals
    from ..cnn.strategy import CNNSignalStrategy
    from ..cnn.storage import CNN_MODEL_DIR
    from ..backtest.engine import BacktestingEngine
    from ..backtest.artifacts import (
        serialize_trades,
        serialize_equity_curve,
        extract_benchmark_prices,
        attach_benchmark_returns,
        summarize_benchmark,
    )

    logger = logging.getLogger(__name__)

    # 1. Validate model exists
    model_path = CNN_MODEL_DIR / f"{req.model}.pt"
    if not model_path.exists():
        raise ValueError(f"CNN 模型不存在: {req.model}")

    if on_progress:
        on_progress(5, f"开始 CNN 回测: {req.name}")

    # 2. Generate signals via CNN inference
    if on_progress:
        on_progress(10, "正在进行 CNN 推理...")

    def inference_progress(pct: float, msg: str) -> None:
        if on_progress:
            # Map inference progress (0-100) to overall (10-50)
            on_progress(10 + pct * 0.4, f"[推理] {msg}")

    signal_df = predict_cnn_signals(
        model_name=req.model,
        start=req.start,
        end=req.end,
        on_progress=inference_progress,
    )

    if signal_df.is_empty():
        return {
            "name": req.name,
            "statistics": {"error": "CNN 推理未产生任何信号"},
            # 字段恒在：无信号即无成交、无净值曲线
            "trades": [],
            "equity_curve": [],
        }

    # 3. Extract target symbol and setup engine
    import torch
    from ..cnn.consistency import (
        check_label_strategy_consistency,
        derive_strategy_exit_from_label,
    )

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    target_symbol = normalize_vt_symbol(checkpoint["train_config"]["target_symbol"])
    # 回测周期必须与训练/推理使用的 K 线周期一致，否则信号与行情对不齐
    input_interval = checkpoint["train_config"].get("input_interval", "d")
    # 训练 label 定义（用于 label↔策略出场 一致性自检，迭代 2 红线）
    label_spec = checkpoint["train_config"].get("label_spec", {}) or {}
    # 预测目标：分类 signal 为概率(0~1)，回归 signal 为预测收益(无界)；阈值口径随之不同
    objective = checkpoint["train_config"].get("objective", "classification")
    vt_symbols = [target_symbol]
    signal_df = signal_df.with_columns(
        pl.col("vt_symbol").map_elements(normalize_vt_symbol, return_dtype=pl.Utf8)
    )

    # 解析出场配置：auto=按 label 自动推导对齐的出场（固定持有 或 OCO 止盈止损）
    if req.exit_mode == "auto":
        exit_cfg = derive_strategy_exit_from_label(label_spec, input_interval)
        exit_mode = exit_cfg["exit_mode"]
        hold_days = exit_cfg["hold_days"]
        # OCO label 自动对齐时，止盈/止损以 label 为准，确保 label 出场 ≡ 策略出场
        take_profit = exit_cfg.get("take_profit", req.take_profit)
        stop_loss = exit_cfg.get("stop_loss", req.stop_loss)
    else:
        exit_mode = req.exit_mode
        hold_days = req.hold_days
        take_profit = req.take_profit
        stop_loss = req.stop_loss

    # label↔策略出场 一致性自检：硬性不一致直接抛错（回测失败），软性问题收集为告警
    consistency_warnings = check_label_strategy_consistency(
        label_spec, exit_mode, hold_days, input_interval
    )

    if on_progress:
        on_progress(55, f"初始化回测引擎, 目标: {target_symbol}, 周期: {input_interval}")

    lab = AlphaLab(ALPHA_LAB_PATH)
    engine = BacktestingEngine(data_loader=lab)

    start_dt = datetime.combine(req.start, datetime.min.time())
    end_dt = datetime.combine(req.end, datetime.max.time())

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=input_interval,
        start=start_dt,
        end=end_dt,
        capital=int(req.capital),
    )

    # Set defaults for symbols missing contract config，并按请求统一成本参数
    for vt_symbol in vt_symbols:
        if vt_symbol not in engine.sizes:
            engine.sizes[vt_symbol] = 1
            engine.priceticks[vt_symbol] = 0.01
        # 成本以回测请求为准：佣金、卖出印花税、每笔不利滑点
        engine.long_rates[vt_symbol] = req.commission_rate
        engine.short_rates[vt_symbol] = req.commission_rate
        engine.stamp_duties[vt_symbol] = req.stamp_duty
        engine.slippages[vt_symbol] = req.slippage

    # T+1 卖出限制（当日买入不可当日卖出）
    engine.t_plus1 = req.t_plus1

    # 撮合成交价口径与训练 label 的 price_ref 一一对齐，杜绝回测↔label 背离：
    #   next_open→open、next_close→close、next_vwap→vwap；close(研究口径)仍按开盘近似。
    _PRICE_REF_TO_FILL_MODE = {
        "next_open": "open",
        "next_close": "close",
        "next_vwap": "vwap",
        "close": "open",
    }
    engine.fill_price_mode = _PRICE_REF_TO_FILL_MODE.get(
        str(label_spec.get("price_ref") or "close"), "open"
    )

    # 4. Add strategy（出场配置经一致性自检/自动对齐后注入）
    strategy_setting = {
        "buy_threshold": req.buy_threshold,
        "sell_threshold": req.sell_threshold,
        "price_add": req.price_add,
        "exit_mode": exit_mode,
        "hold_days": hold_days,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        # path_class 专用：否决阈值透传，非 path_class 场景下保持默认 1.0（等效关闭）
        "veto_threshold": req.veto_threshold,
    }
    engine.add_strategy(CNNSignalStrategy, strategy_setting, signal_df)

    # 5. Load data
    if on_progress:
        on_progress(60, "加载历史数据...")
    engine.load_data()

    # 隐性失败守护：信号与行情 datetime 必须有交集，否则全程无成交（疑似周期/对齐错误）
    bar_dts = {dt.replace(tzinfo=None) for dt in engine.dts}
    sig_dts = {
        (d.replace(tzinfo=None) if isinstance(d, datetime) else d)
        for d in signal_df["datetime"].to_list()
    }
    if bar_dts.isdisjoint(sig_dts):
        raise ValueError(
            f"信号与行情 datetime 无交集（信号 {len(sig_dts)} 个时点、行情 {len(bar_dts)} 个时点），"
            f"疑似回测周期({input_interval})与信号周期不一致或数据缺失，回测判定失败。"
        )

    # 6. Run backtest
    if on_progress:
        on_progress(70, "运行回测...")
    engine.run_backtesting()

    # 7. Calculate results
    if on_progress:
        on_progress(85, "计算结果...")
    daily_df = engine.calculate_result()

    if daily_df is None or engine.trade_count == 0:
        logger.warning("CNN 回测期间无成交记录")
        # veto_count 在零成交时尤其有意义：激进 veto 导致全程否决恰是最需要解释的场景
        return {
            "name": req.name,
            "statistics": {
                "error": "回测期间无成交，请调整阈值或检查数据",
                "capital": req.capital,
                "total_trade_count": 0,
                "veto_count": int(getattr(engine.strategy, "_veto_count", 0)),
            },
            # 字段恒在：无成交即空成交列表与空净值曲线
            "trades": [],
            "equity_curve": [],
        }

    statistics = engine.calculate_statistics()
    # path_class 否决计数：「本要买入却被否决的次数」，供前端显示策略激进程度
    # 防御式读取：非 path_class 场景下策略无 _veto_count 属性时回退 0
    statistics["veto_count"] = int(getattr(engine.strategy, "_veto_count", 0))
    # 回传本次回测使用的成本参数，便于前端展示「成本假设」
    statistics["commission_rate"] = req.commission_rate
    statistics["stamp_duty"] = req.stamp_duty
    statistics["slippage"] = req.slippage
    statistics["price_add"] = req.price_add
    statistics["exit_mode"] = exit_mode
    statistics["hold_days"] = hold_days
    statistics["t_plus1"] = req.t_plus1
    statistics["objective"] = objective
    # 回测撮合成交价口径（与训练 label price_ref 对齐）
    statistics["fill_price_mode"] = engine.fill_price_mode
    # label↔策略一致性：回传训练 label 口径与软性告警，便于前端提示「回测可信度」
    statistics["label_spec"] = label_spec
    statistics["consistency_warnings"] = consistency_warnings

    if on_progress:
        on_progress(100, "CNN 回测完成")

    # 基准（买入持有标的）：CNN 回测的基准即目标标的本身
    equity_curve = serialize_equity_curve(engine.daily_df)
    benchmark_prices = extract_benchmark_prices(engine.daily_results, target_symbol)
    attach_benchmark_returns(equity_curve, benchmark_prices, req.capital)
    statistics.update(summarize_benchmark(equity_curve, target_symbol))

    return {
        "name": req.name,
        "model": req.model,
        "target_symbol": target_symbol,
        "statistics": statistics,
        # 成交明细与逐日净值序列：equity_curve 必须在 calculate_statistics() 之后取
        # （此时 engine.daily_df 才补入 balance/drawdown 列）
        "trades": serialize_trades(engine.trades),
        "equity_curve": equity_curve,
    }


@router.post("/backtest/run")
async def start_cnn_backtest(req: CNNBacktestRequest) -> dict:
    """
    启动 CNN 模型回测任务。

    通过 CNN 推理生成信号后，使用共享回测引擎执行回测。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，请先执行: pip install torch")

    if req.start >= req.end:
        raise HTTPException(400, "开始日期必须早于结束日期")
    # 阈值范围放宽到 (-1, 1)：分类用概率尺度(如 0.6/0.4)，回归用收益尺度(如 0.005/-0.005，可为负)
    if not (-1 < req.buy_threshold < 1):
        raise HTTPException(400, "buy_threshold 必须在 (-1, 1) 之间")
    if not (-1 < req.sell_threshold < 1):
        raise HTTPException(400, "sell_threshold 必须在 (-1, 1) 之间")
    if req.buy_threshold <= req.sell_threshold:
        raise HTTPException(400, "buy_threshold 必须大于 sell_threshold")

    task_id = task_manager.create_task(
        TaskType.CNN_BACKTEST,
        params={"name": req.name, "model": req.model},
        title=f"CNN 回测: {req.model}",
        entity_type="cnn_backtest",
        entity_name=req.name,
    )

    def execute(on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
        return _run_cnn_backtest(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "CNN 回测任务已启动"}


# =============================================================================
# 推理（生成信号）
# =============================================================================

def _run_cnn_predict(
    req: CNNPredictRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """执行 CNN 推理，将概率信号保存到信号库（供 Alpha 回测复用）。"""
    from ..alpha import AlphaLab
    from ..config import ALPHA_LAB_PATH
    from ..cnn.predictor import predict_cnn_signals
    from ..cnn.storage import CNN_MODEL_DIR

    # 1. 校验模型存在
    model_path = CNN_MODEL_DIR / f"{req.model}.pt"
    if not model_path.exists():
        raise ValueError(f"CNN 模型不存在: {req.model}")

    if on_progress:
        on_progress(5, f"开始 CNN 推理: {req.name}")

    # 2. CNN 推理生成信号
    def inference_progress(pct: float, msg: str) -> None:
        if on_progress:
            # 将推理进度 (0-100) 映射到整体 (5-90)
            on_progress(5 + pct * 0.85, f"[推理] {msg}")

    signal_df = predict_cnn_signals(
        model_name=req.model,
        start=req.start,
        end=req.end,
        on_progress=inference_progress,
    )

    if signal_df.is_empty():
        return {
            "name": req.name,
            "signal_count": 0,
            "message": "CNN 推理未产生任何信号",
        }

    # 3. 保存信号到信号库
    if on_progress:
        on_progress(92, "保存信号...")

    lab = AlphaLab(ALPHA_LAB_PATH)
    lab.save_signal(req.name, signal_df)

    if on_progress:
        on_progress(100, f"信号已保存: {req.name} ({signal_df.height} 条)")

    return {
        "name": req.name,
        "model": req.model,
        "signal_count": signal_df.height,
        "signal_mean": float(signal_df["signal"].mean()),
        "start": str(req.start),
        "end": str(req.end),
    }


@router.post("/predict")
async def start_cnn_predict(req: CNNPredictRequest) -> dict:
    """
    启动 CNN 推理任务，生成概率信号并保存到信号库。

    保存后的信号可在「Alpha 因子回测」中作为普通信号直接复用。
    """
    if not _check_torch():
        raise HTTPException(400, "PyTorch 未安装，请先执行: pip install torch")

    if req.start >= req.end:
        raise HTTPException(400, "开始日期必须早于结束日期")

    task_id = task_manager.create_task(
        TaskType.CNN_PREDICT,
        params={"name": req.name, "model": req.model},
        title=f"CNN 推理: {req.model}",
        entity_type="cnn_signal",
        entity_name=req.name,
    )

    def execute(on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
        return _run_cnn_predict(req, on_progress)

    task_manager.run_async(task_id, execute, enable_progress=True)
    return {"task_id": task_id, "message": "CNN 推理任务已启动"}
