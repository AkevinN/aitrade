"""CNN 模型治理：WF/OOS 评估、候选晋级、回滚与治理回放。"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from ..alpha import AlphaLab
from ..backtest.engine import BacktestingEngine
from ..backtest.validation import walk_forward_windows
from ..config import ALPHA_LAB_PATH, CNN_GOVERNANCE_PATH
from ..models.alpha import CNNBacktestRequest
from ..models.governance import (
    CNNBacktestParams,
    CNNCandidateTrainRequest,
    CNNGovernanceConfig,
    CNNGovernanceHistoryEvent,
    CNNGovernanceReplayRequest,
    CNNProductionModel,
    CNNPromotionGate,
    CNNPromotionRequest,
    CNNRollbackRequest,
    CNNWalkForwardRequest,
)
from .consistency import check_label_strategy_consistency, derive_strategy_exit_from_label
from .predictor import predict_cnn_signals
from .storage import CNN_MODEL_DIR
from .strategy import CNNSignalStrategy
from .trainer import train_cnn_model


# 多种子治理的基准种子。第 seed_index 个重复试验使用 seed=BASE_SEED+seed_index，
# 保证 seed_index=0 退化为单种子时与历史硬编码默认（train_cnn_model seed=42）一致。
BASE_SEED = 42


def _now_id(prefix: str) -> str:
    """生成带时间戳的唯一 ID，格式为 ``{prefix}_{YYYYmmddHHMMSS}_{6位hex}``。

    Args:
        prefix: ID 前缀，如 "wf"、"cand"、"replay"。

    Returns:
        可用作文件名或 JSON 主键的唯一字符串。
    """
    return f"{prefix}_{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}"


def _json_default(value: Any) -> Any:
    """json.dumps default 序列化器，处理 date/datetime 与 Pydantic 模型。

    Args:
        value: 不可直接 JSON 序列化的对象。

    Returns:
        可 JSON 序列化的等价值：date/datetime → ISO 字符串，Pydantic 模型 → dict，其余原样返回。
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _read_json(path: Path, default: Any) -> Any:
    """读取 JSON 文件；文件不存在或内容损坏时返回 default。

    Args:
        path: JSON 文件路径。
        default: 文件不存在或解析失败时的返回值。

    Returns:
        解析后的 Python 对象；失败时返回 default。
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, value: Any) -> None:
    """将对象序列化为缩进 JSON 并写入文件，自动创建父目录。

    Args:
        path: 目标文件路径；父目录不存在时自动创建。
        value: 可 JSON 序列化的对象（date/datetime/Pydantic 模型由 _json_default 处理）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


class CNNGovernanceStore:
    """CNN 治理产物的 JSON 文件存储。

    管理 config/production/candidates/reports/replay_reports 及历史事件日志（JSONL），
    均以 JSON/JSONL 格式持久化到 root 目录下，无数据库依赖。
    """

    def __init__(self, root: Path | str = CNN_GOVERNANCE_PATH) -> None:
        """初始化存储，若目录不存在则自动创建。

        Args:
            root: 治理产物根目录；默认取配置中的 CNN_GOVERNANCE_PATH。
        """
        self.root = Path(root)
        self.candidates_dir = self.root / "candidates"
        self.reports_dir = self.root / "reports"
        self.replay_reports_dir = self.root / "replay_reports"
        for path in [self.root, self.candidates_dir, self.reports_dir, self.replay_reports_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> Path:
        """治理配置文件路径（root/config.json）。"""
        return self.root / "config.json"

    @property
    def production_path(self) -> Path:
        """当前生产模型信息文件路径（root/production.json）。"""
        return self.root / "production.json"

    @property
    def history_path(self) -> Path:
        """治理历史事件日志路径（root/history.jsonl，逐行一条 JSON 事件）。"""
        return self.root / "history.jsonl"

    @property
    def scheduler_state_path(self) -> Path:
        """调度器状态文件路径（root/scheduler_state.json）。"""
        return self.root / "scheduler_state.json"

    def get_config(self) -> dict[str, Any]:
        """读取治理配置；文件不存在时返回 Pydantic 默认值对应的字典。"""
        return CNNGovernanceConfig.model_validate(_read_json(self.config_path, {})).model_dump()

    def save_config(self, config: CNNGovernanceConfig) -> dict[str, Any]:
        """持久化治理配置并追加 config_updated 历史事件。

        Args:
            config: 新的治理配置对象。

        Returns:
            序列化后的配置字典。
        """
        data = config.model_dump()
        _write_json(self.config_path, data)
        self.append_history("config_updated", data)
        return data

    def get_production(self) -> dict[str, Any]:
        """读取当前生产模型信息；文件不存在时返回 Pydantic 默认值对应的字典。"""
        return CNNProductionModel.model_validate(_read_json(self.production_path, {})).model_dump()

    def save_production(self, production: dict[str, Any]) -> dict[str, Any]:
        """持久化生产模型信息（经 Pydantic 校验后写入）。

        Args:
            production: 生产模型信息字典，须符合 CNNProductionModel 结构。

        Returns:
            经 Pydantic 校验后的序列化字典。
        """
        data = CNNProductionModel.model_validate(production).model_dump()
        _write_json(self.production_path, data)
        return data

    def candidate_path(self, candidate_id: str) -> Path:
        """返回候选模型 JSON 文件的完整路径（不保证文件已存在）。

        Args:
            candidate_id: 候选 ID，直接作为文件名主干拼为 candidates/{candidate_id}.json。

        Returns:
            候选文件的完整 Path，仅做路径拼接、不触盘。
        """
        return self.candidates_dir / f"{candidate_id}.json"

    def report_path(self, report_id: str) -> Path:
        """返回 WF/OOS 评估报告 JSON 文件的完整路径（不保证文件已存在）。

        Args:
            report_id: 报告 ID，直接作为文件名主干拼为 reports/{report_id}.json。

        Returns:
            报告文件的完整 Path，仅做路径拼接、不触盘。
        """
        return self.reports_dir / f"{report_id}.json"

    def replay_path(self, replay_id: str) -> Path:
        """返回治理回放报告 JSON 文件的完整路径（不保证文件已存在）。

        Args:
            replay_id: 回放报告 ID，直接作为文件名主干拼为 replay_reports/{replay_id}.json。

        Returns:
            回放报告文件的完整 Path，仅做路径拼接、不触盘。
        """
        return self.replay_reports_dir / f"{replay_id}.json"

    def save_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """持久化候选模型元数据。

        Args:
            candidate: 候选信息字典，须含 candidate_id 键。

        Returns:
            原样返回 candidate 字典。
        """
        _write_json(self.candidate_path(str(candidate["candidate_id"])), candidate)
        return candidate

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """读取候选模型元数据；不存在时返回 None。

        Args:
            candidate_id: 候选 ID，对应 candidates/{candidate_id}.json。

        Returns:
            候选信息字典；文件不存在时返回 None。
        """
        path = self.candidate_path(candidate_id)
        if not path.exists():
            return None
        return _read_json(path, {})

    def list_candidates(self) -> list[dict[str, Any]]:
        """列出所有候选模型，按 created_at 降序排列。

        Returns:
            候选信息字典列表；candidates 目录为空时返回 []。
        """
        return sorted(
            [_read_json(path, {}) for path in self.candidates_dir.glob("*.json")],
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )

    def save_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """持久化 WF/OOS 评估报告。

        Args:
            report: 报告字典，须含 report_id 键。

        Returns:
            原样返回 report 字典。
        """
        _write_json(self.report_path(str(report["report_id"])), report)
        return report

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """读取 WF/OOS 评估报告；不存在时返回 None。

        Args:
            report_id: 报告 ID，对应 reports/{report_id}.json。

        Returns:
            报告字典；文件不存在时返回 None。
        """
        path = self.report_path(report_id)
        if not path.exists():
            return None
        return _read_json(path, {})

    def save_replay_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """持久化治理回放报告。

        Args:
            report: 回放报告字典，须含 replay_id 键。

        Returns:
            原样返回 report 字典。
        """
        _write_json(self.replay_path(str(report["replay_id"])), report)
        return report

    def get_replay_report(self, replay_id: str) -> dict[str, Any] | None:
        """读取治理回放报告；不存在时返回 None。

        Args:
            replay_id: 回放报告 ID，对应 replay_reports/{replay_id}.json。

        Returns:
            回放报告字典；文件不存在时返回 None。
        """
        path = self.replay_path(replay_id)
        if not path.exists():
            return None
        return _read_json(path, {})

    def list_replay_reports(self) -> list[dict[str, Any]]:
        """列出所有治理回放报告，按 created_at 降序排列。

        Returns:
            回放报告字典列表；replay_reports 目录为空时返回 []。
        """
        return sorted(
            [_read_json(path, {}) for path in self.replay_reports_dir.glob("*.json")],
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )

    def append_history(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """追加一条治理历史事件到 JSONL 日志文件。

        Args:
            event_type: 事件类型字符串，如 "candidate_trained"、"production_rollback"。
            payload: 事件附带的结构化数据字典。

        Returns:
            序列化后的事件字典（含 event_type/payload/created_at）。
        """
        event = CNNGovernanceHistoryEvent(event_type=event_type, payload=payload).model_dump()
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
        return event

    def history(self) -> list[dict[str, Any]]:
        """读取全量治理历史事件列表（按写入时间升序）。

        Returns:
            事件字典列表；历史文件不存在时返回 []。
        """
        if not self.history_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


store = CNNGovernanceStore()


def _serialize_groups(groups: list[Any]) -> list[dict[str, Any]]:
    """将观测分组列表序列化为纯字典列表（兼容 Pydantic 模型和普通字典）。

    Args:
        groups: Pydantic 模型对象或字典的混合列表。

    Returns:
        纯字典列表，顺序与输入一致。
    """
    return [g.model_dump() if hasattr(g, "model_dump") else dict(g) for g in groups]


def _label_spec_dict(value: Any) -> dict[str, Any]:
    """将 label 配置对象转为纯字典（兼容 Pydantic 模型和 None）。

    Args:
        value: Pydantic LabelSpec 模型、普通字典或 None。

    Returns:
        纯字典；value 为 None 时返回空字典。
    """
    return value.model_dump() if hasattr(value, "model_dump") else dict(value or {})


def _core_score(statistics: dict[str, Any], objective: str) -> float:
    """计算用于 WF 折对比的核心综合得分，越高越好。

    公式：``total_return + sharpe * 5 - max_dd * 0.2 - trade_penalty``，
    附加训练质量项（由 ``_merge_training_metrics`` 注入）：
    - regression：``+best_val_rank_ic * 10``（键缺失时中性按 0.0）
    - path_class：``+(tp_auc + sl_auc - 1) * 10``（键缺失时各按 0.5 中性；
      注意 0.5-0.5=0 恰好中性——此处用 ``None`` 判断而非 ``or``，
      防止 0.0 被错误升为 0.5）

    statistics 含 error 字段时返回 -1e9（标记为失败/无效）。

    Args:
        statistics: 回测统计字典，含 total_return/sharpe_ratio/max_ddpercent/
            total_trade_count 等键（缺失或 None 按 0 处理）；
            regression 时可含 best_val_rank_ic；
            path_class 时可含 best_val_tp_auc / best_val_sl_auc。
        objective: 训练目标，"classification"、"regression" 或 "path_class"。

    Returns:
        综合得分浮点值，保留 6 位有效小数。
    """
    if statistics.get("error"):
        return -1e9
    total_return = float(statistics.get("total_return", 0.0) or 0.0)
    sharpe = float(statistics.get("sharpe_ratio", 0.0) or 0.0)
    max_dd = abs(float(statistics.get("max_ddpercent", 0.0) or 0.0))
    trade_count = float(statistics.get("total_trade_count", 0.0) or 0.0)
    trade_penalty = 0.0 if trade_count > 0 else 5.0
    score = total_return + sharpe * 5.0 - max_dd * 0.2 - trade_penalty
    if objective == "regression":
        score += float(statistics.get("best_val_rank_ic", 0.0) or 0.0) * 10.0
    elif objective == "path_class":
        # 用 None 判断而非 or：AUC=0.0 是有效值，不可被 or 吞掉升为 0.5
        raw_tp = statistics.get("best_val_tp_auc")
        raw_sl = statistics.get("best_val_sl_auc")
        tp_auc = 0.5 if raw_tp is None else float(raw_tp)
        sl_auc = 0.5 if raw_sl is None else float(raw_sl)
        score += (tp_auc + sl_auc - 1.0) * 10.0
    return round(score, 6)


def _merge_training_metrics(
    statistics: dict[str, Any],
    model_name: str,
    checkpoint: dict[str, Any],
) -> None:
    """将 checkpoint 对应的训练期验证指标并入 statistics（原地修改）。

    从 ``{model_name}_history.json`` 读取 ``best_epoch`` 处的 epoch 指标，
    将目标键写入 statistics 供 ``_core_score`` 读取：
    - regression：``best_val_rank_ic``（Spearman 相关，范围 -1~1）
    - path_class：``best_val_tp_auc``、``best_val_sl_auc``（AUC，范围 0~1）

    设计说明：这些指标存储于训练历史 JSON，而非 checkpoint 本体，因此回测统计
    中默认不含这些键。本函数是唯一将训练质量信息接入治理评分的桥梁；若历史文件
    缺失或 best_epoch 超出范围，静默跳过（由 _core_score 的缺失回退 0.5/0.0 兜底）。

    Args:
        statistics: 回测统计字典，原地写入目标键。
        model_name: 模型名称（不含 .pt 后缀），用于定位 _history.json 文件。
        checkpoint: 已加载的 checkpoint 字典，须含 best_epoch 键。
    """
    history_path = CNN_MODEL_DIR / f"{model_name}_history.json"
    if not history_path.exists():
        return
    try:
        with history_path.open(encoding="utf-8") as file:
            history: list[dict[str, Any]] = json.load(file)
    except (OSError, ValueError):
        return

    best_epoch: int = int(checkpoint.get("best_epoch") or 0)
    if best_epoch <= 0 or best_epoch > len(history):
        return
    best_metrics = history[best_epoch - 1]

    objective = checkpoint.get("train_config", {}).get("objective", "classification")
    if objective == "regression":
        val = best_metrics.get("val_rank_ic")
        if val is not None:
            statistics["best_val_rank_ic"] = float(val)
    elif objective == "path_class":
        for stat_key, hist_key in (
            ("best_val_tp_auc", "val_tp_auc"),
            ("best_val_sl_auc", "val_sl_auc"),
        ):
            val = best_metrics.get(hist_key)
            if val is not None:
                statistics[stat_key] = float(val)


def _backtest_model(
    *,
    model_name: str,
    name: str,
    start: date,
    end: date,
    capital: float,
    params: CNNBacktestParams,
) -> dict[str, Any]:
    """直接在治理流程内回测一个 CNN 模型，绕过 API 层。

    先用 predict_cnn_signals 生成信号，再读取 checkpoint 的训练配置确定标的、
    输入周期与离场方式（exit_mode="auto" 时由标签规格反推持有/止盈止损），
    配好交易成本/滑点/T+1/成交价模式后跑回测，最终把训练期验证指标
    （rank_ic / tp_auc / sl_auc）并入统计供 _core_score 评分。仅供治理模块内部调用。

    Args:
        model_name: 待回测的模型名（不含 .pt 后缀），对应 CNN_MODEL_DIR 下的权重文件。
        name: 本次回测的展示名，原样写入返回结果的 name 字段，用于区分候选/生产/各折。
        start: 回测起始日期（含）。
        end: 回测结束日期（含）。
        capital: 初始资金（元），传入引擎时取整。
        params: 回测参数（买卖阈值、离场方式、成本、滑点、否决阈值等）；
            exit_mode="auto" 时离场参数由标签规格自动反推。

    Returns:
        字典，含 name/model/target_symbol/statistics/trades/equity_curve。
        异常路径下 statistics 仅含 error 键（取值："CNN 推理未产生任何信号"、
        "信号与行情 datetime 无交集"、"回测期间无成交"），trades/equity_curve 为空列表。

    Raises:
        FileNotFoundError: 模型权重文件 {model_name}.pt 不存在时抛出。
    """
    import torch

    from .storage import CNN_MODEL_DIR

    model_path = CNN_MODEL_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"CNN 模型不存在: {model_name}")

    signal_df = predict_cnn_signals(model_name=model_name, start=start, end=end)
    if signal_df.is_empty():
        return {"statistics": {"error": "CNN 推理未产生任何信号"}, "trades": [], "equity_curve": []}

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    target_symbol = checkpoint["train_config"]["target_symbol"]
    input_interval = checkpoint["train_config"].get("input_interval", "d")
    label_spec = checkpoint["train_config"].get("label_spec", {}) or {}
    objective = checkpoint["train_config"].get("objective", "classification")

    if params.exit_mode == "auto":
        exit_cfg = derive_strategy_exit_from_label(label_spec, input_interval)
        exit_mode = exit_cfg["exit_mode"]
        hold_days = exit_cfg["hold_days"]
        take_profit = exit_cfg.get("take_profit", params.take_profit)
        stop_loss = exit_cfg.get("stop_loss", params.stop_loss)
    else:
        exit_mode = params.exit_mode
        hold_days = params.hold_days
        take_profit = params.take_profit
        stop_loss = params.stop_loss

    consistency_warnings = check_label_strategy_consistency(
        label_spec, exit_mode, hold_days, input_interval
    )

    lab = AlphaLab(ALPHA_LAB_PATH)
    engine = BacktestingEngine(data_loader=lab)
    engine.set_parameters(
        vt_symbols=[target_symbol],
        interval=input_interval,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.max.time()),
        capital=int(capital),
    )
    for vt_symbol in [target_symbol]:
        if vt_symbol not in engine.sizes:
            engine.sizes[vt_symbol] = 1
            engine.priceticks[vt_symbol] = 0.01
        engine.long_rates[vt_symbol] = params.commission_rate
        engine.short_rates[vt_symbol] = params.commission_rate
        engine.stamp_duties[vt_symbol] = params.stamp_duty
        engine.slippages[vt_symbol] = params.slippage
    engine.t_plus1 = params.t_plus1
    engine.fill_price_mode = {
        "next_open": "open",
        "next_close": "close",
        "next_vwap": "vwap",
        "close": "open",
    }.get(str(label_spec.get("price_ref") or "close"), "open")
    engine.add_strategy(
        CNNSignalStrategy,
        {
            "buy_threshold": params.buy_threshold,
            "sell_threshold": params.sell_threshold,
            "price_add": params.price_add,
            "exit_mode": exit_mode,
            "hold_days": hold_days,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            # path_class 专用：否决阈值透传，非 path_class 下保持默认 1.0（等效关闭）
            "veto_threshold": params.veto_threshold,
        },
        signal_df,
    )
    engine.load_data()
    bar_dts = {dt.replace(tzinfo=None) for dt in engine.dts}
    sig_dts = {
        (d.replace(tzinfo=None) if isinstance(d, datetime) else d)
        for d in signal_df["datetime"].to_list()
    }
    if bar_dts.isdisjoint(sig_dts):
        return {
            "name": name,
            "model": model_name,
            "target_symbol": target_symbol,
            "statistics": {"error": "信号与行情 datetime 无交集", "total_trade_count": 0},
            "trades": [],
            "equity_curve": [],
        }
    engine.run_backtesting()
    daily_df = engine.calculate_result()
    if daily_df is None or engine.trade_count == 0:
        return {
            "name": name,
            "model": model_name,
            "target_symbol": target_symbol,
            "statistics": {"error": "回测期间无成交", "capital": capital, "total_trade_count": 0},
            "trades": [],
            "equity_curve": [],
        }
    statistics = engine.calculate_statistics()
    # 将训练期验证指标（best_val_rank_ic / best_val_tp_auc / best_val_sl_auc）并入统计，
    # 供 _core_score 读取。这些指标存储在 _history.json 的 best_epoch 行，而非 checkpoint 本体。
    # 对称接线：regression 与 path_class 均从 history 读取，结构完全一致。
    _merge_training_metrics(statistics, model_name, checkpoint)
    statistics.update({
        "objective": objective,
        "label_spec": label_spec,
        "consistency_warnings": consistency_warnings,
        "fill_price_mode": engine.fill_price_mode,
        "commission_rate": params.commission_rate,
        "stamp_duty": params.stamp_duty,
        "slippage": params.slippage,
        "price_add": params.price_add,
    })
    from ..backtest.artifacts import serialize_equity_curve, serialize_trades

    return {
        "name": name,
        "model": model_name,
        "target_symbol": target_symbol,
        "statistics": statistics,
        "trades": serialize_trades(engine.trades),
        "equity_curve": serialize_equity_curve(engine.daily_df),
    }


def _train_governance_model(
    req: CNNWalkForwardRequest,
    *,
    model_name: str,
    start: date,
    end: date,
    seed_index: int = 0,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """在指定时间区间训练一个治理专用 CNN 模型（按 seed_index 选定随机种子）。

    从 CNNWalkForwardRequest 中提取全量证券列表，将 ``seed_index`` 映射为
    ``seed = BASE_SEED + seed_index`` 下传 train_cnn_model：seed_index 不再只是
    返回字典里的标记，而是真正驱动权重初始化与 DataLoader shuffle，使不同
    seed_index 训出可分辨的不同模型（多种子治理的基础）。seed_index=0 退化为
    单种子，等价于历史的硬编码默认种子。

    Args:
        req: WF/OOS 评估请求，含目标证券、观测分组、训练参数等。
        model_name: 本次训练保存的模型名称（含日期范围后缀由 train_cnn_model 自动追加）。
        start: 训练数据起始日期（含）。
        end: 训练数据结束日期（含）。
        seed_index: 重复试验序号（从 0 起）；映射为实际种子 BASE_SEED+seed_index
            下传训练，并写入返回字典供溯源。
        on_progress: 进度回调 ``(percent, message)``，可为 None。

    Returns:
        train_cnn_model 的返回字典附加 seed_index 键（实际所用种子为 BASE_SEED+seed_index）。
    """
    target = req.target_symbol
    symbols = [target]
    for group in req.observation_groups:
        for symbol in group.symbols:
            if symbol not in symbols:
                symbols.append(symbol)
    return train_cnn_model(
        name=model_name,
        vt_symbols=symbols,
        start=start,
        end=end,
        target_symbol=target,
        epochs=req.training_params.epochs,
        batch_size=req.training_params.batch_size,
        learning_rate=req.training_params.learning_rate,
        lookback=req.training_params.lookback,
        dropout=req.training_params.dropout,
        train_ratio=req.training_params.train_ratio,
        observation_groups=_serialize_groups(req.observation_groups),
        input_data_kind=req.input_data_kind,
        input_interval=req.input_interval,
        label_spec=_label_spec_dict(req.label_spec),
        loss_weighting=req.training_params.loss_weighting,
        objective=req.objective,
        seed=BASE_SEED + seed_index,
        on_progress=on_progress,
    ) | {"seed_index": seed_index}


def _cross_seed_dispersion(scores: list[float]) -> dict[str, Any]:
    """汇总同一折内多个随机种子的得分，给出均值/标准差/样本数。

    用于衡量候选模型对随机种子的敏感度：std 越大说明该折结果越不稳定、
    越依赖具体种子，治理决策应更谨慎。门禁与生产对比统一消费这里的 mean，
    避免单一幸运种子蒙混过关。

    Args:
        scores: 同一折内各种子的核心得分列表（_core_score 输出）。n=1 即单种子。

    Returns:
        字典 ``{"mean": float, "std": float, "n": int}``：
        - n: 种子数（len(scores)）；
        - mean: 各种子得分均值，空列表时为 0.0；
        - std: 总体标准差（ddof=0），n<2 或空列表时恒为 0.0。

    Example:
        >>> _cross_seed_dispersion([0.2, 0.4, 0.6])
        {'mean': 0.4, 'std': 0.163..., 'n': 3}
        >>> _cross_seed_dispersion([0.5])
        {'mean': 0.5, 'std': 0.0, 'n': 1}
    """
    n = len(scores)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n}


def run_walk_forward_evaluate(
    req: CNNWalkForwardRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """执行 Walk-Forward/OOS 多折评估并持久化评估报告（支持折内多种子）。

    按 (train_days, test_days, step_days) 生成滚动窗口。每折对 ``req.n_seeds`` 个
    随机种子（seed=BASE_SEED+seed_index）各训一个独立模型并在测试区间回测，候选
    得分取这些种子核心得分的均值（``cross_seed.mean``），并记录跨种子标准差
    （``cross_seed.std``）衡量结果对种子的敏感度。门禁与生产对比均消费跨种子均值，
    避免单一幸运种子蒙混过关。生产模型已固定，每折只回测一次。最终汇总折胜率、
    平均得分提升与平均跨种子波动，并应用 promotion_gate 判断是否通过晋级门禁。

    Args:
        req: WF/OOS 评估请求，含时间区间、训练参数、回测参数、晋级门禁与
            ``n_seeds``（每折种子数，<1 时按 1 兜底）。
        on_progress: 进度回调 ``(percent, message)``，可为 None。

    Returns:
        评估报告字典，含 report_id/type/folds/summary 等字段；每折附
        ``cross_seed`` ``{mean,std,n}`` 与 ``candidate_seed_scores``，summary 附
        ``n_seeds`` 与 ``avg_cross_seed_std``；已持久化到 store。

    Raises:
        ValueError: 无法生成 walk-forward 窗口（日期范围或窗口参数不合理）时抛出。
    """
    report_id = _now_id("wf")
    windows = walk_forward_windows(req.start, req.end, req.train_days, req.test_days, req.step_days)
    if not windows:
        raise ValueError("无法生成 walk-forward 窗口，请扩大日期范围或缩短 train/test days")

    production = store.get_production()
    production_model = req.production_model or production.get("model_name") or ""
    n_seeds = max(1, req.n_seeds)
    folds: list[dict[str, Any]] = []
    candidate_wins = 0
    score_deltas: list[float] = []

    total_steps = len(windows)
    for idx, window in enumerate(windows, start=1):
        train_start, train_end = window["train"]
        test_start, test_end = window["test"]
        if on_progress:
            on_progress(
                5 + 85 * (idx - 1) / total_steps,
                f"WF {idx}/{total_steps}: 训练 {train_start}~{train_end}（{n_seeds} 个种子）",
            )

        # 折内对 n_seeds 个种子各训一个模型并回测，收集核心得分；
        # 候选得分取跨种子均值，避免单个幸运种子主导晋级判定。
        seed_scores: list[float] = []
        seed_models: list[str] = []
        seed_statistics: list[dict[str, Any]] = []
        for seed_index in range(n_seeds):
            model_name = f"{req.name}_wf{idx}_s{seed_index}_{uuid.uuid4().hex[:4]}"
            train_result = _train_governance_model(
                req,
                model_name=model_name,
                start=train_start,
                end=train_end,
                seed_index=seed_index,
            )
            actual_model = str(train_result["name"])
            candidate_bt = _backtest_model(
                model_name=actual_model,
                name=f"{req.name}_wf{idx}_s{seed_index}_candidate",
                start=test_start,
                end=test_end,
                capital=1_000_000,
                params=req.backtest_params,
            )
            seed_models.append(actual_model)
            seed_statistics.append(candidate_bt.get("statistics", {}))
            seed_scores.append(_core_score(candidate_bt.get("statistics", {}), req.objective))

        cross_seed = _cross_seed_dispersion(seed_scores)
        candidate_score = cross_seed["mean"]

        production_bt: dict[str, Any] | None = None
        production_score: float | None = None
        if production_model and (CNN_MODEL_DIR / f"{production_model}.pt").exists():
            # 生产模型已固定，无需多种子重复，单次回测即可。
            production_bt = _backtest_model(
                model_name=production_model,
                name=f"{req.name}_wf{idx}_production",
                start=test_start,
                end=test_end,
                capital=1_000_000,
                params=req.backtest_params,
            )
            production_score = _core_score(production_bt.get("statistics", {}), req.objective)
            delta = candidate_score - production_score
            score_deltas.append(delta)
            if delta > 0:
                candidate_wins += 1

        folds.append({
            "fold": idx,
            "train": {"start": train_start.isoformat(), "end": train_end.isoformat()},
            "test": {"start": test_start.isoformat(), "end": test_end.isoformat()},
            # candidate_model / candidate_statistics 取第 0 个种子作"代表"展示用；
            # 晋级判定真正消费的 candidate_score 是跨种子均值（cross_seed.mean），
            # 二者在多种子分歧时不会逐项对账一致——完整的逐种子数据见
            # candidate_seed_statistics / candidate_seed_scores / cross_seed。
            "candidate_model": seed_models[0],
            "candidate_models": seed_models,
            "candidate_statistics": seed_statistics[0],
            "candidate_seed_statistics": seed_statistics,
            "candidate_seed_scores": seed_scores,
            "candidate_score": candidate_score,
            "cross_seed": cross_seed,
            "production_model": production_model,
            "production_statistics": production_bt.get("statistics", {}) if production_bt else None,
            "production_score": production_score,
            "score_delta": candidate_score - production_score if production_score is not None else None,
        })

    pass_result = _gate_result(folds, req.promotion_gate, req.objective, has_production=bool(production_model))
    report = {
        "report_id": report_id,
        "type": "walk_forward",
        "name": req.name,
        "created_at": datetime.now().isoformat(),
        "request": req.model_dump(),
        "production_model": production_model,
        "folds": folds,
        "summary": {
            "fold_count": len(folds),
            "candidate_win_count": candidate_wins,
            "candidate_win_rate": round(candidate_wins / len(folds), 4) if folds else 0.0,
            "avg_score_delta": round(sum(score_deltas) / len(score_deltas), 6) if score_deltas else None,
            "n_seeds": n_seeds,
            "avg_cross_seed_std": (
                round(sum(f["cross_seed"]["std"] for f in folds) / len(folds), 6) if folds else None
            ),
            "passed": pass_result["passed"],
            "reasons": pass_result["reasons"],
        },
    }
    store.save_report(report)
    store.append_history("wf_evaluate_completed", {"report_id": report_id, "passed": pass_result["passed"]})
    if on_progress:
        on_progress(100, "WF/OOS 评估完成")
    return report


def _gate_result(
    folds: list[dict[str, Any]],
    gate: CNNPromotionGate,
    objective: str,
    *,
    has_production: bool,
) -> dict[str, Any]:
    """判断候选模型是否通过晋级门禁，返回结果与失败原因列表。

    无生产模型时，直接返回 passed=False 并附提示（首个模型需人工确认）。
    有生产模型时，依次校验：
    - require_positive_oos：候选平均 OOS 核心分数须为正。
    - min_win_rate：候选折胜率须达标。
    - min_core_score_delta：候选平均分数提升须达标。

    Args:
        folds: WF 各折结果列表，每项含 candidate_score/score_delta 等键。
        gate: 晋级门禁配置（CNNPromotionGate）。
        objective: 训练目标，"classification" | "regression" | "path_class"；
            门禁判定本身不区分目标（目标间的评分差异已在 _core_score 中体现）。
        has_production: 当前是否存在生产模型；False 时跳过对比类门禁。

    Returns:
        字典 ``{"passed": bool, "reasons": list[str]}``；通过时 reasons 含成功说明，
        未通过时 reasons 列出各失败原因。
    """
    reasons: list[str] = []
    if not folds:
        return {"passed": False, "reasons": ["无 WF 折结果"]}

    candidate_scores = [float(f.get("candidate_score", -1e9)) for f in folds]
    if gate.require_positive_oos and sum(candidate_scores) / len(candidate_scores) <= 0:
        reasons.append("候选模型平均 OOS 核心分数未为正")

    if has_production:
        deltas = [float(f["score_delta"]) for f in folds if f.get("score_delta") is not None]
        win_rate = sum(1 for delta in deltas if delta > 0) / len(deltas) if deltas else 0.0
        avg_delta = sum(deltas) / len(deltas) if deltas else -1e9
        if win_rate < gate.min_win_rate:
            reasons.append(f"候选胜出折数比例 {win_rate:.2f} 低于门禁 {gate.min_win_rate:.2f}")
        if avg_delta < gate.min_core_score_delta:
            reasons.append(f"候选平均核心分数提升 {avg_delta:.4f} 低于门禁 {gate.min_core_score_delta:.4f}")
    else:
        reasons.append("无生产模型，报告仅用于首个生产模型人工确认")

    return {"passed": len(reasons) == 0, "reasons": reasons or ["通过相对胜出门禁"]}


def train_candidate(
    req: CNNCandidateTrainRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """执行 WF/OOS 评估、训练最终候选模型并持久化候选元数据。

    流程：
    1. 运行 WF/OOS 评估（占总进度 55%）；
    2. 用 final_train_start~final_train_end 训练最终候选模型（占 25%）；
    3. 将候选信息（含 WF 报告 ID、通过/失败状态）写入候选库。

    Args:
        req: 候选训练请求，继承 CNNWalkForwardRequest 并补充 final_train_start/end。
        on_progress: 进度回调 ``(percent, message)``，可为 None。

    Returns:
        候选元数据字典，含 candidate_id/status/model_name/report_id/summary 等键；
        已持久化到 store，并追加 candidate_trained 历史事件。
    """
    if on_progress:
        on_progress(5, "开始候选模型 WF/OOS 评估...")
    wf_report = run_walk_forward_evaluate(
        CNNWalkForwardRequest.model_validate(req.model_dump()),
        on_progress=lambda p, m: on_progress(5 + p * 0.55, m) if on_progress else None,
    )
    candidate_id = _now_id("cand")
    final_start = req.final_train_start or req.start
    final_end = req.final_train_end or date.fromisoformat(wf_report["folds"][-1]["train"]["end"])
    if on_progress:
        on_progress(70, f"训练最终候选模型 {final_start}~{final_end}...")
    final_name = f"{req.name}_candidate_{candidate_id}"
    train_result = _train_governance_model(
        req,
        model_name=final_name,
        start=final_start,
        end=final_end,
        on_progress=lambda p, m: on_progress(70 + p * 0.25, m) if on_progress else None,
    )
    passed = bool(wf_report.get("summary", {}).get("passed"))
    candidate = {
        "candidate_id": candidate_id,
        "created_at": datetime.now().isoformat(),
        "status": "passed" if passed else "failed",
        "model_name": train_result["name"],
        "report_id": wf_report["report_id"],
        "target_symbol": req.target_symbol,
        "input_interval": req.input_interval,
        "objective": req.objective,
        "baseline_model": wf_report.get("production_model", ""),
        "summary": wf_report.get("summary", {}),
        "request": req.model_dump(),
    }
    store.save_candidate(candidate)
    store.append_history("candidate_trained", candidate)
    if on_progress:
        on_progress(100, "候选模型训练完成")
    return candidate


def promote_candidate(candidate_id: str, req: CNNPromotionRequest) -> dict[str, Any]:
    """将候选模型晋级为生产模型并记录历史事件。

    校验候选存在且模型文件存在后，将候选信息写入 production.json，
    更新候选状态为 "promoted"，追加 candidate_promoted 历史事件。

    Args:
        candidate_id: 待晋级的候选 ID。
        req: 晋级请求，含 promoted_by（操作人）和 note（备注）。

    Returns:
        更新后的生产模型信息字典。

    Raises:
        FileNotFoundError: 候选不存在或候选模型文件缺失时抛出。
    """
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        raise FileNotFoundError(f"候选不存在: {candidate_id}")
    if not (CNN_MODEL_DIR / f"{candidate['model_name']}.pt").exists():
        raise FileNotFoundError(f"候选模型文件不存在: {candidate['model_name']}")
    previous = store.get_production()
    version = f"v{datetime.now():%Y%m%d%H%M%S}"
    production = {
        "model_name": candidate["model_name"],
        "model_version": version,
        "target_symbol": candidate.get("target_symbol", ""),
        "input_interval": candidate.get("input_interval", "d"),
        "objective": candidate.get("objective", "classification"),
        "promoted_at": datetime.now(),
        "promoted_by": req.promoted_by,
        "report_id": candidate.get("report_id", ""),
        "previous_model_name": previous.get("model_name", ""),
        "previous_model_version": previous.get("model_version", ""),
    }
    saved = store.save_production(production)
    candidate["status"] = "promoted"
    candidate["promoted_at"] = datetime.now().isoformat()
    candidate["promotion_note"] = req.note
    store.save_candidate(candidate)
    store.append_history("candidate_promoted", {"candidate_id": candidate_id, "production": saved, "note": req.note})
    return saved


def reject_candidate(candidate_id: str, note: str = "") -> dict[str, Any]:
    """拒绝候选模型并更新其状态为 "rejected"。

    Args:
        candidate_id: 待拒绝的候选 ID。
        note: 拒绝原因备注（可为空）。

    Returns:
        更新后的候选元数据字典。

    Raises:
        FileNotFoundError: 候选不存在时抛出。
    """
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        raise FileNotFoundError(f"候选不存在: {candidate_id}")
    candidate["status"] = "rejected"
    candidate["rejected_at"] = datetime.now().isoformat()
    candidate["reject_note"] = note
    store.save_candidate(candidate)
    store.append_history("candidate_rejected", {"candidate_id": candidate_id, "note": note})
    return candidate


def rollback_production(req: CNNRollbackRequest) -> dict[str, Any]:
    """将生产模型回滚到上一版本（或指定模型）并记录历史事件。

    取当前生产信息的 previous_model_name 作为回滚目标（或使用 req.rollback_to 显式指定），
    将目标模型升格为新生产模型，并在版本号中嵌入 "rollback_" 前缀以示区分。

    Args:
        req: 回滚请求，含可选的 rollback_to（指定回滚目标）、requested_by 和 note。

    Returns:
        更新后的生产模型信息字典。

    Raises:
        ValueError: 当前生产信息中无上一版本可回滚时抛出。
        FileNotFoundError: 回滚目标模型文件不存在时抛出。
    """
    current = store.get_production()
    target_model = req.rollback_to or current.get("previous_model_name", "")
    if not target_model:
        raise ValueError("没有可回滚的上一生产模型")
    if not (CNN_MODEL_DIR / f"{target_model}.pt").exists():
        raise FileNotFoundError(f"回滚目标模型不存在: {target_model}")
    production = {
        **current,
        "model_name": target_model,
        "model_version": f"rollback_{datetime.now():%Y%m%d%H%M%S}",
        "promoted_at": datetime.now(),
        "promoted_by": req.requested_by,
        "previous_model_name": current.get("model_name", ""),
        "previous_model_version": current.get("model_version", ""),
    }
    saved = store.save_production(production)
    store.append_history("production_rollback", {"from": current, "to": saved, "note": req.note})
    return saved


def _buy_and_hold(vt_symbol: str, start: date, end: date, capital: float, interval: str) -> dict[str, Any]:
    """计算买入持有基准策略的净值曲线与汇总统计。

    以 start 日收盘价买入，持有到 end，仅计算持有期内每日相对首日收盘价的资产价值变化，
    不考虑交易成本。用于治理回放中与主动管理策略对比。

    Args:
        vt_symbol: 标的证券代码。
        start: 持仓起始日期（含）。
        end: 持仓结束日期（含）。
        capital: 初始资金（元）。
        interval: K 线周期，如 "d"。

    Returns:
        字典，含 statistics（total_return/end_balance 等汇总指标）
        和 equity_curve（逐日 {date, balance} 列表）；无行情时 statistics 含 error 键。
    """
    lab = AlphaLab(ALPHA_LAB_PATH)
    rows = lab.load_bar_frame(
        vt_symbol,
        interval,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.max.time()),
        include_derived=True,
    )
    if rows is None or rows.is_empty():
        return {"statistics": {"error": "买入持有基准无行情"}, "equity_curve": []}
    rows = rows.sort("datetime")
    first = float(rows["close"][0])
    equity_curve = []
    for item in rows.select(["datetime", "close"]).iter_rows(named=True):
        close = float(item["close"])
        balance = capital * close / first if first > 0 else capital
        equity_curve.append({"date": item["datetime"].date().isoformat(), "balance": balance})
    end_balance = equity_curve[-1]["balance"] if equity_curve else capital
    return {
        "statistics": {
            "total_return": round((end_balance / capital - 1) * 100, 4),
            "end_balance": end_balance,
            "total_trade_count": 1,
            "sharpe_ratio": 0.0,
            "max_ddpercent": 0.0,
            "total_turnover": capital,
            "total_commission": 0.0,
        },
        "equity_curve": equity_curve,
    }


def run_governance_replay(
    req: CNNGovernanceReplayRequest,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """在历史区间回放模型治理决策，对比三种基线策略。

    按 evaluation_period_days 将 [start, end] 切分为多个评估周期，每周期：
    1. 训练新候选；
    2. 在本周期回测候选模型与当前治理模型；
    3. 若候选得分超出 promotion_gate.min_core_score_delta 则自动晋级（"governed_promotion"）。

    同步维护三条基线以供比较：
    - fixed_initial_model：全程使用初始模型，不重训不换；
    - always_retrain：每周期无脑重训并使用最新模型；
    - governed_promotion：治理决策驱动的晋级策略（本方法的核心）；
    - buy_and_hold：买入持有目标标的基准。

    Args:
        req: 治理回放请求，含时间区间、评估周期、初训天数、晋级门禁等配置。
        on_progress: 进度回调 ``(percent, message)``，可为 None。

    Returns:
        回放报告字典，含 replay_id/baselines/promotion_events/rejected_events/conclusion 等；
        已持久化到 store，并追加 governance_replay_completed 历史事件。

    Raises:
        ValueError: 回放区间为空（end <= start）时抛出。
    """
    replay_id = _now_id("replay")
    cycle_starts: list[date] = []
    cursor = req.start
    while cursor < req.end:
        cycle_starts.append(cursor)
        cursor = cursor + timedelta(days=req.evaluation_period_days)
    if not cycle_starts:
        raise ValueError("回放区间为空")

    def train_for_cycle(prefix: str, train_end: date) -> str:
        """为回放某一周期训练单个治理模型并返回落盘后的模型名。

        回放刻意使用单种子（seed_index=0，即 seed=BASE_SEED），保证回放结果
        确定可复现，且不让"每周期×多种子"的开销在长区间回放中爆炸；多种子的
        鲁棒性评估留给 run_walk_forward_evaluate 的折内循环。

        Args:
            prefix: 本周期模型名前缀（已含唯一后缀，避免覆盖）。
            train_end: 该周期训练窗结束日期；起点回退 initial_train_days 天。

        Returns:
            train_cnn_model 落盘后的实际模型名（含日期范围后缀）。
        """
        train_start = train_end - timedelta(days=req.initial_train_days)
        train_req = CNNWalkForwardRequest(
            name=prefix,
            target_symbol=req.target_symbol,
            input_data_kind=req.input_data_kind,
            input_interval=req.input_interval,
            start=train_start,
            end=train_end,
            train_days=max(30, req.initial_train_days - req.test_period_days),
            test_days=req.test_period_days,
            objective=req.objective,
            label_spec=req.label_spec,
            observation_groups=req.observation_groups,
            training_params=req.training_params,
            backtest_params=req.backtest_params,
            promotion_gate=req.promotion_gate,
        )
        return str(
            _train_governance_model(
                train_req, model_name=prefix, start=train_start, end=train_end, seed_index=0
            )["name"]
        )

    if on_progress:
        on_progress(5, "训练回放初始模型...")
    initial_model = train_for_cycle(f"{req.name}_initial_{uuid.uuid4().hex[:4]}", req.start)
    fixed_model = initial_model
    governed_model = initial_model
    always_model = initial_model

    baseline_results: dict[str, dict[str, Any]] = {
        "fixed_initial_model": {"periods": [], "events": []},
        "always_retrain": {"periods": [], "events": []},
        "governed_promotion": {"periods": [], "events": []},
        "buy_and_hold": _buy_and_hold(req.target_symbol, req.start, req.end, req.capital, req.input_interval),
    }
    promotion_events: list[dict[str, Any]] = []
    rejected_events: list[dict[str, Any]] = []

    for index, period_start in enumerate(cycle_starts, start=1):
        period_end = min(period_start + timedelta(days=req.evaluation_period_days), req.end)
        progress_base = 10 + 80 * (index - 1) / len(cycle_starts)
        if on_progress:
            on_progress(progress_base, f"回放周期 {index}/{len(cycle_starts)}: {period_start}~{period_end}")

        fixed_bt = _backtest_model(
            model_name=fixed_model,
            name=f"{req.name}_fixed_{index}",
            start=period_start,
            end=period_end,
            capital=req.capital,
            params=req.backtest_params,
        )
        baseline_results["fixed_initial_model"]["periods"].append(_period_result(period_start, period_end, fixed_model, fixed_bt, req.objective))

        if index > 1:
            always_model = train_for_cycle(f"{req.name}_always_{index}_{uuid.uuid4().hex[:4]}", period_start)
        always_bt = _backtest_model(
            model_name=always_model,
            name=f"{req.name}_always_{index}",
            start=period_start,
            end=period_end,
            capital=req.capital,
            params=req.backtest_params,
        )
        baseline_results["always_retrain"]["periods"].append(_period_result(period_start, period_end, always_model, always_bt, req.objective))

        candidate_model = train_for_cycle(f"{req.name}_governed_candidate_{index}_{uuid.uuid4().hex[:4]}", period_start)
        candidate_bt = _backtest_model(
            model_name=candidate_model,
            name=f"{req.name}_candidate_oos_{index}",
            start=period_start,
            end=period_end,
            capital=req.capital,
            params=req.backtest_params,
        )
        current_bt = _backtest_model(
            model_name=governed_model,
            name=f"{req.name}_governed_{index}",
            start=period_start,
            end=period_end,
            capital=req.capital,
            params=req.backtest_params,
        )
        candidate_score = _core_score(candidate_bt.get("statistics", {}), req.objective)
        current_score = _core_score(current_bt.get("statistics", {}), req.objective)
        if candidate_score > current_score + req.promotion_gate.min_core_score_delta:
            event = {
                "date": period_start.isoformat(),
                "old_model": governed_model,
                "new_model": candidate_model,
                "reason": f"候选核心分数 {candidate_score:.4f} > 当前 {current_score:.4f}",
            }
            promotion_events.append(event)
            governed_model = candidate_model
            governed_bt = candidate_bt
        else:
            event = {
                "date": period_start.isoformat(),
                "candidate_model": candidate_model,
                "kept_model": governed_model,
                "reason": f"候选核心分数 {candidate_score:.4f} 未胜出现有 {current_score:.4f}",
                "counterfactual_statistics": candidate_bt.get("statistics", {}),
            }
            rejected_events.append(event)
            governed_bt = current_bt
        baseline_results["governed_promotion"]["periods"].append(_period_result(period_start, period_end, governed_model, governed_bt, req.objective))

    for key in ["fixed_initial_model", "always_retrain", "governed_promotion"]:
        baseline_results[key]["statistics"] = _aggregate_periods(baseline_results[key]["periods"], req.capital)

    replay = {
        "replay_id": replay_id,
        "name": req.name,
        "target_symbol": req.target_symbol,
        "start": req.start.isoformat(),
        "end": req.end.isoformat(),
        "created_at": datetime.now().isoformat(),
        "request": req.model_dump(),
        "baselines": baseline_results,
        "process": {
            "evaluation_period_count": len(cycle_starts),
            "candidate_train_count": len(cycle_starts),
            "passed_count": len(promotion_events),
            "rejected_count": len(rejected_events),
            "promotion_count": len(promotion_events),
            "rollback_count": 0,
        },
        "promotion_events": promotion_events,
        "rejected_events": rejected_events,
        "diagnostics": {
            "bad_model_block_count": len(rejected_events),
            "rejected_counterfactuals": rejected_events,
        },
        "conclusion": _replay_conclusion(baseline_results),
    }
    store.save_replay_report(replay)
    store.append_history("governance_replay_completed", {"replay_id": replay_id, "conclusion": replay["conclusion"]})
    if on_progress:
        on_progress(100, "治理回放回测完成")
    return replay


def _period_result(start: date, end: date, model: str, backtest: dict[str, Any], objective: str) -> dict[str, Any]:
    """将单个评估周期的回测结果汇整为标准化字典。

    Args:
        start: 评估周期起始日期。
        end: 评估周期结束日期。
        model: 本周期使用的模型名称。
        backtest: _backtest_model 返回的完整回测结果字典。
        objective: 训练目标，用于 _core_score 的计算。

    Returns:
        字典，含 start/end/model/statistics/score/equity_curve/trades 键。
    """
    stats = backtest.get("statistics", {})
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "model": model,
        "statistics": stats,
        "score": _core_score(stats, objective),
        "equity_curve": backtest.get("equity_curve", []),
        "trades": backtest.get("trades", []),
    }


def _aggregate_periods(periods: list[dict[str, Any]], capital: float) -> dict[str, Any]:
    """汇总多个评估周期的统计指标为整体绩效摘要。

    total_return 为各周期之和（非复利），annual_return 按周期数折算为年化，
    sharpe_ratio 为各周期均值，max_ddpercent 取各周期最大值。

    Args:
        periods: _period_result 返回的字典列表，含 statistics 键。
        capital: 初始资金（元），写入 summary 供前端显示。

    Returns:
        绩效摘要字典，含 total_return/annual_return/sharpe_ratio/max_ddpercent/
        total_trade_count/turnover/total_cost/empty_position_ratio/capital。
    """
    returns = [float(p.get("statistics", {}).get("total_return", 0.0) or 0.0) for p in periods]
    trade_counts = [float(p.get("statistics", {}).get("total_trade_count", 0.0) or 0.0) for p in periods]
    turnovers = [float(p.get("statistics", {}).get("total_turnover", 0.0) or 0.0) for p in periods]
    commissions = [float(p.get("statistics", {}).get("total_commission", 0.0) or 0.0) for p in periods]
    max_dds = [abs(float(p.get("statistics", {}).get("max_ddpercent", 0.0) or 0.0)) for p in periods]
    total_return = sum(returns)
    return {
        "total_return": round(total_return, 4),
        "annual_return": round(total_return / max(len(periods), 1) * 12, 4),
        "sharpe_ratio": round(sum(float(p.get("statistics", {}).get("sharpe_ratio", 0.0) or 0.0) for p in periods) / max(len(periods), 1), 4),
        "max_ddpercent": round(max(max_dds) if max_dds else 0.0, 4),
        "total_trade_count": int(sum(trade_counts)),
        "turnover": round(sum(turnovers), 4),
        "total_cost": round(sum(commissions), 4),
        "empty_position_ratio": round(sum(1 for c in trade_counts if c == 0) / max(len(trade_counts), 1), 4),
        "capital": capital,
    }


def _replay_conclusion(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """根据三条基线的总收益率生成治理回放结论与建议。

    比较 governed_promotion 与 fixed_initial_model、always_retrain 的 total_return，
    给出四种结论之一并附使用建议。

    Args:
        results: 回放报告的 baselines 字典，键为基线名，值含 statistics 子字典。

    Returns:
        字典，含 better_than_fixed_initial_model/better_than_always_retrain/
        recommend_enable_promotion/verdict 四个键。
    """
    fixed = results.get("fixed_initial_model", {}).get("statistics", {})
    always = results.get("always_retrain", {}).get("statistics", {})
    governed = results.get("governed_promotion", {}).get("statistics", {})
    governed_return = float(governed.get("total_return", 0.0) or 0.0)
    fixed_return = float(fixed.get("total_return", 0.0) or 0.0)
    always_return = float(always.get("total_return", 0.0) or 0.0)
    better_fixed = governed_return >= fixed_return
    better_always = governed_return >= always_return
    if better_fixed and better_always:
        verdict = "治理优于固定模型与无脑重训，建议启用半自动晋级"
    elif better_fixed:
        verdict = "治理优于固定模型，但未优于无脑重训，建议继续观察"
    elif better_always:
        verdict = "治理优于无脑重训，但未优于固定模型，建议只保留评估报告"
    else:
        verdict = "治理无明显优势，暂不建议启用生产晋级"
    return {
        "better_than_fixed_initial_model": better_fixed,
        "better_than_always_retrain": better_always,
        "recommend_enable_promotion": better_fixed and better_always,
        "verdict": verdict,
    }

