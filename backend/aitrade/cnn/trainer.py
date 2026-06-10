"""
CNN trainer — grouped market observation training pipeline.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Callable, Optional

import numpy as np

from .model import build_dataset, create_market_cnn, normalize_observation_groups
from .storage import save_cnn_model

logger = logging.getLogger(__name__)


def _broadcast_group_mask(group_mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Expand a [1,1,1,S,G] mask to [N,1,T,S,G] for normalization."""
    return np.broadcast_to(
        group_mask,
        (x.shape[0], 1, 1, x.shape[3], x.shape[4]),
    ).repeat(x.shape[2], axis=2)


def _normalize_grouped_tensor(
    train_x: np.ndarray,
    full_x: np.ndarray,
    group_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit channel-wise stats on train slice only and apply to all samples."""
    train_valid = _broadcast_group_mask(group_mask, train_x)
    full_valid = _broadcast_group_mask(group_mask, full_x)

    channel_count = np.maximum(train_valid.sum(axis=(0, 2, 3, 4), keepdims=True), 1.0)
    channel_mean = (train_x * train_valid).sum(axis=(0, 2, 3, 4), keepdims=True) / channel_count
    channel_var = (((train_x - channel_mean) * train_valid) ** 2).sum(
        axis=(0, 2, 3, 4),
        keepdims=True,
    ) / channel_count
    channel_std = np.sqrt(channel_var) + 1e-8

    normalized = ((full_x - channel_mean) / channel_std) * full_valid
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return normalized, {
        "channel_mean": channel_mean.reshape(-1).tolist(),
        "channel_std": channel_std.reshape(-1).tolist(),
        "group_mask": group_mask.reshape(group_mask.shape[3], group_mask.shape[4]).tolist(),
    }


def _rank_auc(
    y_true: "list[float] | np.ndarray",
    y_score: "list[float] | np.ndarray",
) -> float | None:
    """二分类 AUC（Mann-Whitney U 秩和法，避免引入 sklearn 依赖）。

    单一类别（全涨或全跌）时 AUC 无定义，返回 None。并列分数取平均秩。
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = float((y_true_arr == 1).sum())
    n_neg = float((y_true_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score_arr, kind="mergesort")
    sorted_scores = y_score_arr[order]
    ranks = np.empty(len(y_score_arr), dtype=np.float64)
    i = 0
    total = len(sorted_scores)
    while i < total:
        j = i
        while j < total and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0  # 并列取平均秩（1-based）
        i = j

    rank_sum_pos = ranks[y_true_arr == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _classification_metrics(
    y_true: "list[float] | np.ndarray",
    y_prob: "list[float] | np.ndarray",
    positive_ratio: float,
) -> dict[str, Any]:
    """方向分类评估指标。

    重点是 baseline_acc（多数类基线）与 excess_acc（超额准确率），用于戳破
    「涨跌近 50/50 时 accuracy 接近随机却看起来还行」的假象；另含 AUC 与
    正类的查准/查全/F1。
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y_true_arr) == 0:
        return {
            "acc": 0.0, "baseline_acc": 0.0, "excess_acc": 0.0,
            "auc": None, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    y_pred = (y_prob_arr > 0.5).astype(np.float64)
    acc = float((y_pred == y_true_arr).mean())
    baseline = max(positive_ratio, 1.0 - positive_ratio)  # 始终预测占比更高的方向

    tp = float(((y_pred == 1) & (y_true_arr == 1)).sum())
    fp = float(((y_pred == 1) & (y_true_arr == 0)).sum())
    fn = float(((y_pred == 0) & (y_true_arr == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    auc = _rank_auc(y_true_arr, y_prob_arr)
    return {
        "acc": round(acc, 4),
        "baseline_acc": round(baseline, 4),
        "excess_acc": round(acc - baseline, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _rankdata(x: "list[float] | np.ndarray") -> np.ndarray:
    """并列取平均秩（1-based），供 RankIC 使用。"""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    sorted_arr = arr[order]
    i = 0
    total = len(arr)
    while i < total:
        j = i
        while j < total and sorted_arr[j] == sorted_arr[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _pearson(a: "list[float] | np.ndarray", b: "list[float] | np.ndarray") -> float | None:
    """皮尔逊相关；任一方差为 0 或样本不足时返回 None。"""
    arr_a = np.asarray(a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(arr_a) < 2 or arr_a.std() < 1e-12 or arr_b.std() < 1e-12:
        return None
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def _regression_metrics(
    y_true: "list[float] | np.ndarray",
    y_pred: "list[float] | np.ndarray",
    up_ratio: float,
) -> dict[str, Any]:
    """回归评估指标：IC/RankIC（预测收益与真实收益的相关性，量化最关心）、
    MAE/RMSE（误差幅度），以及方向准确率 + 多数类基线（与分类口径衔接）。"""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(yt) == 0:
        return {
            "ic": None, "rank_ic": None, "mae": 0.0, "rmse": 0.0,
            "dir_acc": 0.0, "baseline_acc": 0.0, "excess_acc": 0.0,
        }
    ic = _pearson(yp, yt)
    rank_ic = _pearson(_rankdata(yp), _rankdata(yt))
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    dir_acc = float(np.mean((yp > 0) == (yt > 0)))
    baseline = max(up_ratio, 1.0 - up_ratio)
    return {
        "ic": round(ic, 4) if ic is not None else None,
        "rank_ic": round(rank_ic, 4) if rank_ic is not None else None,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "dir_acc": round(dir_acc, 4),
        "baseline_acc": round(baseline, 4),
        "excess_acc": round(dir_acc - baseline, 4),
    }


# 选优/早停的业务指标权重（仅回归用）：
# rank_ic 衡量排序能力，excess_acc 衡量相对基线的方向超额准确率，二者越大越好。
_SEL_RANK_IC_WEIGHT = 1.0
_SEL_EXCESS_ACC_WEIGHT = 1.0


def _selection_score(epoch_row: dict[str, Any], is_regression: bool) -> float:
    """计算用于选最佳 epoch / 早停的业务指标分数，越大越好。

    - 回归：综合 RankIC（排序能力）与 excess_acc（方向超额准确率），
      避免单看 val_loss（MSE）导致选中"loss 低但方向差"的模型。
    - 分类：优先用 AUC，缺失时回退到 excess_acc。
    None 值按 0 处理，保证可比较。
    """
    if is_regression:
        rank_ic = epoch_row.get("val_rank_ic")
        excess = epoch_row.get("val_excess_acc")
        rank_ic = float(rank_ic) if rank_ic is not None else 0.0
        excess = float(excess) if excess is not None else 0.0
        return _SEL_RANK_IC_WEIGHT * rank_ic + _SEL_EXCESS_ACC_WEIGHT * excess
    auc = epoch_row.get("val_auc")
    if auc is not None:
        return float(auc)
    excess = epoch_row.get("val_excess_acc")
    return float(excess) if excess is not None else 0.0


def train_cnn_model(
    name: str,
    vt_symbols: list[str],
    start: date,
    end: date,
    target_symbol: str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lookback: int = 30,
    dropout: float = 0.5,
    train_ratio: float = 0.7,
    on_progress: Optional[Callable[[float, str], None]] = None,
    observation_groups: list[dict[str, Any]] | None = None,
    input_data_kind: str = "bar",
    input_interval: str = "d",
    label_spec: dict[str, Any] | None = None,
    loss_weighting: str = "none",
    objective: str = "classification",
) -> dict[str, Any]:
    """Train a grouped market CNN and persist checkpoint + training history.

    objective:
    - "classification"：方向二分类，Sigmoid + BCELoss，输出上涨概率。
    - "regression"：直接预测未来收益，线性输出 + Huber 损失，配 IC/MAE 等指标。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        logger.info("CUDA 不可用，将使用 CPU 训练")

    # 在模型名后追加训练数据范围（_YYYYMMDD-YYYYMMDD），便于后续回测识别并避开训练区间。
    # 若名称已包含该范围后缀（如重新训练同名模型），则不重复追加。
    range_suffix = f"_{start:%Y%m%d}-{end:%Y%m%d}"
    if not name.endswith(range_suffix):
        name = f"{name}{range_suffix}"

    target_symbol, normalized_groups = normalize_observation_groups(
        target_symbol=target_symbol,
        observation_groups=observation_groups,
        vt_symbols=vt_symbols,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    X, y, group_mask, info = build_dataset(
        vt_symbols=vt_symbols,
        start=start,
        end=end,
        lookback=lookback,
        target_symbol=target_symbol,
        on_progress=on_progress,
        observation_groups=normalized_groups,
        input_data_kind=input_data_kind,
        input_interval=input_interval,
        label_spec=label_spec,
        objective=objective,
    )
    is_regression = objective == "regression"
    n = len(X)

    if n < 50:
        raise ValueError(f"样本数不足: {n}，需至少 50 个样本，请扩大日期范围或增加观测证券")

    n_train = int(n * train_ratio)
    n_val = n - n_train
    if n_train <= 0 or n_val <= 0:
        raise ValueError("训练集或验证集为空，请调整 train_ratio 或日期范围")

    X, normalization = _normalize_grouped_tensor(X[:n_train], X, group_mask)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    # 上涨样本占比（分类: y∈{0,1}; 回归: y>0），用于方向基线，戳破「接近随机却看起来还行」
    train_pos_ratio = float(np.mean(y_train > 0)) if len(y_train) else 0.0
    val_pos_ratio = float(np.mean(y_val > 0)) if len(y_val) else 0.0

    # #1 幅度加权：让 +5% 比 +0.01% 在损失里权重更大；none 时权重恒为 1（等价旧 BCE）。
    # 回归目标本身即幅度，无需再加权，权重恒为 1。
    weight_cap = 10.0  # 防止单根涨跌停样本独占梯度
    sample_returns = np.asarray(info.get("sample_returns", []), dtype=np.float32)
    if not is_regression and loss_weighting == "magnitude" and len(sample_returns) == n:
        abs_returns = np.abs(sample_returns)
        train_scale = float(abs_returns[:n_train].mean())
        scale = train_scale if train_scale > 1e-12 else 1.0
        sample_weights = np.clip(abs_returns / scale, 0.0, weight_cap).astype(np.float32)
    else:
        if loss_weighting == "magnitude" and not is_regression:
            logger.warning("幅度加权不可用（缺少样本收益），回退为普通 BCE")
            loss_weighting = "none"
        elif is_regression:
            loss_weighting = "none"
        sample_weights = np.ones(n, dtype=np.float32)
    w_train = sample_weights[:n_train]
    w_val = sample_weights[n_train:]
    group_mask_train = np.repeat(group_mask, len(X_train), axis=0)
    group_mask_val = np.repeat(group_mask, len(X_val), axis=0)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(group_mask_train),
            torch.FloatTensor(w_train),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.FloatTensor(group_mask_val),
            torch.FloatTensor(w_val),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    if on_progress:
        on_progress(
            60,
            f"数据划分: 训练={n_train}, 验证={n_val}, 设备={device}, 张量={X.shape[1]}x{X.shape[2]}x{X.shape[3]}x{X.shape[4]}",
        )

    C, T, S, G = X.shape[1], X.shape[2], X.shape[3], X.shape[4]
    model = create_market_cnn(C, T, S, G, dropout, objective=objective).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 分类: BCE（reduction=none 以便幅度加权）；回归: Huber（对涨跌停等离群更稳健）
    if is_regression:
        criterion = nn.HuberLoss(reduction="none", delta=0.03)
    else:
        criterion = nn.BCELoss(reduction="none")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    if on_progress:
        on_progress(62, f"模型参数: {total_params:,}, 开始训练...")

    best_val_loss = float("inf")
    best_score = float("-inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    patience_counter = 0
    patience = max(10, epochs // 5)
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb, mb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            mb = mb.to(device)
            wb = wb.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(xb, mb)
            loss = (criterion(pred, yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(xb)
            if is_regression:
                train_correct += ((pred > 0) == (yb > 0)).sum().item()
            else:
                train_correct += ((pred > 0.5).float() == yb).sum().item()
            train_total += len(xb)

        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs: list[float] = []
        val_labels: list[float] = []
        with torch.no_grad():
            for xb, yb, mb, _wb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                mb = mb.to(device)

                pred = model(xb, mb)
                # 验证损失保持不加权，便于早停/选模与历史口径一致可比
                loss = criterion(pred, yb).mean()
                val_loss += loss.item() * len(xb)
                if is_regression:
                    val_correct += ((pred > 0) == (yb > 0)).sum().item()
                else:
                    val_correct += ((pred > 0.5).float() == yb).sum().item()
                val_total += len(xb)
                val_probs.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
                val_labels.extend(yb.detach().cpu().numpy().reshape(-1).tolist())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(current_lr, 8),
        }
        if is_regression:
            reg_metrics = _regression_metrics(val_labels, val_probs, val_pos_ratio)
            epoch_row.update({
                "val_ic": reg_metrics["ic"],
                "val_rank_ic": reg_metrics["rank_ic"],
                "val_mae": reg_metrics["mae"],
                "val_rmse": reg_metrics["rmse"],
                "val_dir_acc": reg_metrics["dir_acc"],
                "val_baseline_acc": reg_metrics["baseline_acc"],
                "val_excess_acc": reg_metrics["excess_acc"],
            })
        else:
            cls_metrics = _classification_metrics(val_labels, val_probs, val_pos_ratio)
            epoch_row.update({
                "val_baseline_acc": cls_metrics["baseline_acc"],
                "val_excess_acc": cls_metrics["excess_acc"],
                "val_auc": cls_metrics["auc"],
                "val_precision": cls_metrics["precision"],
                "val_recall": cls_metrics["recall"],
                "val_f1": cls_metrics["f1"],
            })
        history.append(epoch_row)

        # 用业务指标（回归: RankIC+超额方向准确率 / 分类: AUC）选最佳 epoch，
        # 而非单看 val_loss，避免选中"loss 低但方向/排序差"的模型。
        current_score = _selection_score(epoch_row, is_regression)
        if current_score > best_score + 0.0001:
            best_score = current_score
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1

        if on_progress:
            pct = 62 + 35 * epoch / epochs
            if is_regression:
                ic_text = f"{epoch_row['val_ic']:.3f}" if epoch_row["val_ic"] is not None else "N/A"
                on_progress(
                    pct,
                    f"Epoch {epoch}/{epochs} | val_loss={val_loss:.5f} | "
                    f"IC={ic_text} RankIC={epoch_row['val_rank_ic'] if epoch_row['val_rank_ic'] is not None else 'N/A'} | "
                    f"方向准确率={epoch_row['val_dir_acc']:.1%} (基线 {epoch_row['val_baseline_acc']:.1%})",
                )
            else:
                auc_text = f"{epoch_row['val_auc']:.3f}" if epoch_row["val_auc"] is not None else "N/A"
                on_progress(
                    pct,
                    f"Epoch {epoch}/{epochs} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.1%} (基线 {epoch_row['val_baseline_acc']:.1%}, "
                    f"超额 {epoch_row['val_excess_acc']:+.1%}) | AUC={auc_text}",
                )

        if patience_counter >= patience:
            if on_progress:
                on_progress(pct, f"早停触发: {patience_counter} 轮无改善, 最佳 epoch={best_epoch}")
            break

    elapsed = time.time() - start_time

    if best_state:
        model.load_state_dict(best_state)

    save_data = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": C,
            "time_steps": T,
            "max_group_width": S,
            "group_count": G,
            "dropout": dropout,
        },
        "train_config": {
            "symbols": info["symbols"],
            "target_symbol": info["target_symbol"],
            "observation_groups": info["groups"],
            "input_data_kind": info["input_data_kind"],
            "input_interval": info["input_interval"],
            "label_spec": info["label_spec"],
            "start": str(start),
            "end": str(end),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "lookback": lookback,
            "train_ratio": train_ratio,
            "loss_weighting": loss_weighting,
            "objective": objective,
        },
        "normalization": normalization,
        "dataset_info": {
            "feature_names": info["feature_names"],
            "feature_channels": info["feature_channels"],
            "group_count": info["group_count"],
            "max_group_width": info["max_group_width"],
            "sample_anchor_dates": info["sample_anchor_dates"],
            "skipped_for_label": info["skipped_for_label"],
            "skipped_for_neutral": info.get("skipped_for_neutral", 0),
            "label_threshold": info.get("label_threshold", 0.0),
            "train_pos_ratio": round(train_pos_ratio, 4),
            "val_pos_ratio": round(val_pos_ratio, 4),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_score": round(best_score, 6),
        "selection_metric": "rank_ic+excess_acc" if is_regression else "auc",
    }

    model_path, history_path = save_cnn_model(name, save_data, history)

    best_metrics: dict[str, Any] = history[best_epoch - 1] if best_epoch > 0 else {}
    best_val_acc = round(best_metrics.get("val_acc", 0.0), 4)
    best_excess_acc = best_metrics.get("val_excess_acc", 0.0)
    # 跑赢基线 = 方向准确率超过「始终预测多数类」的水平，是模型是否有方向 edge 的底线判据
    beats_baseline = bool(best_excess_acc is not None and best_excess_acc > 0)

    if on_progress:
        verdict = "✅ 跑赢基线" if beats_baseline else "⚠️ 未跑赢多数类基线"
        if is_regression:
            ic_val = best_metrics.get("val_ic")
            ic_text = f"{ic_val:.3f}" if ic_val is not None else "N/A"
            on_progress(
                100,
                f"训练完成 | 最佳 Epoch={best_epoch} | IC={ic_text} | "
                f"方向准确率={best_metrics.get('val_dir_acc', 0):.1%} (基线 {best_metrics.get('val_baseline_acc', 0):.1%}) | "
                f"{verdict} | 耗时={elapsed:.0f}s",
            )
        else:
            on_progress(
                100,
                f"训练完成 | 最佳 Epoch={best_epoch} | val_acc={best_val_acc:.1%} "
                f"(基线 {best_metrics.get('val_baseline_acc', 0):.1%}) | {verdict} | 耗时={elapsed:.0f}s",
            )

    result = {
        "name": name,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "objective": objective,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_acc": best_val_acc,
        "val_baseline_acc": best_metrics.get("val_baseline_acc"),
        "best_val_excess_acc": best_excess_acc,
        "beats_baseline": beats_baseline,
        "train_pos_ratio": round(train_pos_ratio, 4),
        "val_pos_ratio": round(val_pos_ratio, 4),
        "total_params": total_params,
        "train_samples": n_train,
        "val_samples": n_val,
        "elapsed_seconds": round(elapsed, 1),
        "history": history,
        "tensor_shape": [int(C), int(T), int(S), int(G)],
    }
    if is_regression:
        result.update({
            "best_val_ic": best_metrics.get("val_ic"),
            "best_val_rank_ic": best_metrics.get("val_rank_ic"),
            "best_val_mae": best_metrics.get("val_mae"),
            "best_val_dir_acc": best_metrics.get("val_dir_acc"),
        })
    else:
        result.update({
            "best_val_auc": best_metrics.get("val_auc"),
            "best_val_f1": best_metrics.get("val_f1"),
        })
    return result
