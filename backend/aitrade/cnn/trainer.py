"""分组行情 CNN 训练流水线。

提供 train_cnn_model 主入口：构建样本集、在训练集上拟合逐通道归一化、
按训练目标（classification/regression/path_class）选择损失并迭代训练，
依业务指标（AUC/RankIC/tp_auc 等）选最佳 epoch 与早停，最后持久化
checkpoint 与训练历史。配套若干评估指标工具函数（AUC、IC/RankIC、
方向准确率与多数类基线等），用于戳破「看似不错实则接近随机」的假象。
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Callable, Optional

import numpy as np

from .dataset import (
    PATH_CLASS_NAMES,
    PATH_SL_FIRST,
    PATH_TIME_DOWN,
    PATH_TIME_UP,
    PATH_TP_FIRST,
)
from .features import ALIGN_DROP_WARN_THRESHOLD
from .model import build_dataset, create_market_cnn, normalize_observation_groups
from .storage import save_cnn_model

logger = logging.getLogger(__name__)


def _broadcast_group_mask(group_mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """将 [1,1,1,S,G] 掩码广播为 [N,1,T,S,G]，供逐样本归一化使用。

    Args:
        group_mask: 分组掩码，形状 [1, 1, 1, S, G]，float32。
        x: 特征张量，形状 [N, C, T, S, G]；仅使用其 shape 计算广播目标。

    Returns:
        广播后的掩码，形状 [N, 1, T, S, G]；N 为样本数，T 为时间步数。
    """
    return np.broadcast_to(
        group_mask,
        (x.shape[0], 1, 1, x.shape[3], x.shape[4]),
    ).repeat(x.shape[2], axis=2)


def _normalize_grouped_tensor(
    train_x: np.ndarray,
    full_x: np.ndarray,
    group_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """在训练集上拟合逐通道均值/标准差，并对全集（训练+验证）做标准化。

    只使用训练集的有效样本（group_mask=1 的位置）计算统计量，防止验证集数据泄漏；
    归一化后对无效位置（掩码=0）保持 0，NaN/Inf 替换为 0。

    Args:
        train_x: 训练集特征张量，形状 [N_train, C, T, S, G]，float32。
        full_x: 全集特征张量，形状 [N, C, T, S, G]，float32（N >= N_train）。
        group_mask: 分组掩码，形状 [1, 1, 1, S, G]，float32；有效位为 1.0。

    Returns:
        (normalized_full_x, normalization_info)：
        - normalized_full_x: 标准化后的全集张量，形状同 full_x，float32。
        - normalization_info: 含 channel_mean/channel_std/group_mask 的字典，
          供推理时复现同一标准化（保存到 checkpoint）。
    """
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

    等价于「随机取一个正样本、一个负样本，正样本得分更高的概率」。
    并列分数取平均秩，因此完全无区分度时为 0.5。

    Args:
        y_true: 真值标签，取值 {0, 1}（正类为 1）；list 或一维 ndarray，内部展平。
        y_score: 模型给出的正类得分/概率，越大越倾向正类；与 y_true 等长。

    Returns:
        AUC 值（float，0~1）；当样本里只有单一类别（全为正或全为负，
        正负样本数任一为 0）AUC 无定义时返回 None。
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
    正类的查准/查全/F1。预测以 0.5 为阈值二值化（prob > 0.5 判为上涨）。

    Args:
        y_true: 真值标签，取值 {0, 1}（1 为上涨）；list 或一维 ndarray，内部展平。
        y_prob: 模型预测的上涨概率，值域 [0, 1]；与 y_true 等长。
        positive_ratio: 正样本（上涨）占比，用于推算多数类基线
            baseline_acc = max(positive_ratio, 1 - positive_ratio)。

    Returns:
        指标字典，键含 acc/baseline_acc/excess_acc/auc/precision/recall/f1，
        数值均保留 4 位小数；auc 在单一类别时为 None。
        y_true 为空时返回各项为 0.0、auc 为 None 的占位字典。
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
    """把一组数值转成秩次（1-based），并列值取平均秩，供 RankIC 计算使用。

    Args:
        x: 待排秩的数值序列；list 或一维 ndarray，内部展平为一维。

    Returns:
        与输入等长的 ndarray，元素为对应位置的秩次（最小值秩为 1）；
        相等的数值共享其名次的平均值（如两个并列第 2、3 名均记 2.5）。
    """
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
    """皮尔逊线性相关系数。

    用于计算 IC（预测值对真值）/RankIC（两者秩次的相关）。

    Args:
        a: 第一组数值；list 或一维 ndarray，内部展平。
        b: 第二组数值；与 a 等长。

    Returns:
        相关系数（float，-1~1）；当样本数 < 2，或任一序列标准差近 0
        （< 1e-12，相关无定义）时返回 None。
    """
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
    MAE/RMSE（误差幅度），以及方向准确率 + 多数类基线（与分类口径衔接）。

    方向准确率按符号判定（预测与真值同为正/同为非正即算命中）。

    Args:
        y_true: 真实收益序列；list 或一维 ndarray，内部展平。
        y_pred: 模型预测的收益序列；与 y_true 等长。
        up_ratio: 上涨样本（真值 > 0）占比，用于推算方向基线
            baseline_acc = max(up_ratio, 1 - up_ratio)。

    Returns:
        指标字典，键含 ic/rank_ic/mae/rmse/dir_acc/baseline_acc/excess_acc；
        ic/rank_ic 在方差为 0 或样本不足时为 None，其余为 float；
        y_true 为空时返回 ic/rank_ic 为 None、其余为 0.0 的占位字典。
    """
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


def _path_class_metrics(
    y_true: list[float] | np.ndarray,
    y_logits: list[list[float]] | np.ndarray,
) -> dict[str, Any]:
    """path_class 四分类评估指标：逐类 precision/recall、macro_f1、tp_auc/sl_auc。

    基于 raw logits 矩阵计算：先做 softmax（numpy 实现，减 max 防溢出）得到四类概率，
    然后用 argmax 得到预测类别，再分别计算各类指标。
    argmax 并列时取最小下标，即偏向 tp_first（类 0）。

    类别对应关系（与 dataset.py 常量一致）：
    - 0 = tp_first（止盈先触发）
    - 1 = sl_first（止损先触发）
    - 2 = time_up（时间止损 + 方向向上）
    - 3 = time_down（时间止损 + 方向向下）

    macro_f1 的均值只统计 support > 0 的类，support=0 的类不参与分母，
    避免「预测了 0 个样本的类别」人为拉低整体 F1。

    tp_auc / sl_auc 均为 one-vs-rest AUC（复用 _rank_auc），单类缺失时返回 None。

    Args:
        y_true: 形状 [N] 的真值类别数组，float32，值域 {0.0, 1.0, 2.0, 3.0}。
        y_logits: 形状 [N, 4] 的 raw logits 矩阵；支持 list[list[float]] 或 np.ndarray。

    Returns:
        指标字典，含：
        - "tp_auc": float | None，TP First 类（类 0）的 one-vs-rest AUC。
        - "sl_auc": float | None，SL First 类（类 1）的 one-vs-rest AUC。
        - "macro_f1": float，各有效类 F1 的均值（无有效类时为 0.0）。
        - "class_report": 嵌套字典，键为类别名，值含 precision/recall/support。

    Example:
        >>> y_t = np.array([0, 1, 2, 3], dtype=np.float32)
        >>> logits = np.eye(4) * 10  # 完美预测
        >>> m = _path_class_metrics(y_t, logits)
        >>> m["macro_f1"]  # 1.0
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lgt = np.asarray(y_logits, dtype=np.float64)
    if lgt.ndim != 2 or lgt.shape[1] != 4:
        lgt = lgt.reshape(-1, 4)

    # softmax：减 row-max 防数值溢出
    lgt_shifted = lgt - lgt.max(axis=1, keepdims=True)
    exp_lgt = np.exp(lgt_shifted)
    probs = exp_lgt / exp_lgt.sum(axis=1, keepdims=True)  # [N, 4]

    y_pred = np.argmax(probs, axis=1).astype(np.float64)  # [N]

    # PATH_CLASS_NAMES 与常量编码一一对应，消除硬编码
    class_labels = (PATH_TP_FIRST, PATH_SL_FIRST, PATH_TIME_UP, PATH_TIME_DOWN)

    class_report: dict[str, dict[str, Any]] = {}
    f1_list: list[float] = []

    for name, label in zip(PATH_CLASS_NAMES, class_labels, strict=True):
        support = int(np.sum(yt == label))
        tp = float(np.sum((y_pred == label) & (yt == label)))
        fp = float(np.sum((y_pred == label) & (yt != label)))
        fn = float(np.sum((y_pred != label) & (yt == label)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_report[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": support,
        }
        if support > 0:
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # one-vs-rest AUC（复用 _rank_auc）：单类缺失（全为正/负）时自然返回 None
    tp_auc_raw = _rank_auc((yt == 0.0).astype(np.float64), probs[:, 0])
    sl_auc_raw = _rank_auc((yt == 1.0).astype(np.float64), probs[:, 1])
    tp_auc = round(float(tp_auc_raw), 4) if tp_auc_raw is not None else None
    sl_auc = round(float(sl_auc_raw), 4) if sl_auc_raw is not None else None

    return {
        "tp_auc": tp_auc,
        "sl_auc": sl_auc,
        "macro_f1": round(macro_f1, 4),
        "class_report": class_report,
    }


# 选优/早停的业务指标权重（仅回归用）：
# rank_ic 衡量排序能力，excess_acc 衡量相对基线的方向超额准确率，二者越大越好。
_SEL_RANK_IC_WEIGHT = 1.0
_SEL_EXCESS_ACC_WEIGHT = 1.0


def _selection_score(epoch_row: dict[str, Any], objective: str) -> float:
    """计算用于选最佳 epoch / 早停的业务指标分数，越大越好。

    按 objective 分三条路径：
    - "regression"：综合 RankIC（排序能力）与 excess_acc（方向超额准确率），
      避免单看 val_loss（MSE）导致选中"loss 低但方向差"的模型；
      缺失值按 0.0 处理。
    - "path_class"：tp_auc + sl_auc（止盈/止损判别能力之和）；
      任一 AUC 为 None 时按 0.5（随机基线）代入，保证可比较。
    - "classification"（默认）：优先用 AUC，缺失时回退到 excess_acc；
      缺失值按 0.0 处理。

    Args:
        epoch_row: 单个 epoch 的指标字典（history 中的一行）。
        objective: 训练目标字符串，"classification"/"regression"/"path_class"。

    Returns:
        业务指标分数（float，越大越好）。
    """
    if objective == "regression":
        rank_ic = epoch_row.get("val_rank_ic")
        excess = epoch_row.get("val_excess_acc")
        rank_ic = float(rank_ic) if rank_ic is not None else 0.0
        excess = float(excess) if excess is not None else 0.0
        return _SEL_RANK_IC_WEIGHT * rank_ic + _SEL_EXCESS_ACC_WEIGHT * excess
    if objective == "path_class":
        tp_auc = epoch_row.get("val_tp_auc")
        sl_auc = epoch_row.get("val_sl_auc")
        tp = float(tp_auc) if tp_auc is not None else 0.5
        sl = float(sl_auc) if sl_auc is not None else 0.5
        return tp + sl
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
    seed: int = 42,
) -> dict[str, Any]:
    """训练分组感知的多尺度行情 CNN 并持久化 checkpoint 与训练历史。

    流程：
    1. 构建样本集（build_dataset）；
    2. 在训练集上拟合逐通道归一化参数（_normalize_grouped_tensor）；
    3. AdamW + CosineAnnealingLR 迭代训练，按业务指标（AUC/RankIC）选最佳 epoch；
    4. 早停：连续 patience 轮无改善则提前终止；
    5. 将最佳权重、训练配置、归一化统计量等保存到 checkpoint（save_cnn_model）。

    模型名后自动追加 _{start:%Y%m%d}-{end:%Y%m%d} 后缀，以便回测侧识别并跳过训练区间。

    Args:
        name: 模型基础名称；最终保存名为 {name}_{start}-{end}（如 name 已含后缀则不重复）。
        vt_symbols: 参与训练的证券代码列表（observation_groups 为空时构造默认分组）。
        start: 训练数据起始日期（含）。
        end: 训练数据结束日期（含）。
        target_symbol: 预测目标证券代码；None 时取 vt_symbols[0]。
        epochs: 最大训练轮数；默认 50。
        batch_size: 每批样本数；默认 32。
        learning_rate: AdamW 初始学习率；默认 0.001。
        lookback: 每个样本的回看 bar 数（时间窗口 T）；默认 30。
        dropout: 全连接头 Dropout 概率；默认 0.5。
        train_ratio: 训练集比例（时间顺序切分，非随机）；默认 0.7。
        on_progress: 进度回调 ``(percent, message)``，可为 None。
        observation_groups: 语义分组配置；None 时退化为旧版兼容逻辑。
        input_data_kind: 数据种类，"bar" 或 "tick"。
        input_interval: K 线周期，"d" | "1m" | "5m" | "10m" | "15m" | "30m" | "60m"。
        label_spec: label 配置字典；None → 默认 next_bar。
        loss_weighting: 损失加权策略，"none"（均匀）或 "magnitude"（幅度加权，仅 classification）；
            path_class 与 regression 下强制回退为 "none"（权重恒 1）。
        objective: 训练目标：
            - "classification"：BCELoss，输出上涨概率 [B,1]。
            - "regression"：HuberLoss，输出无界预测收益 [B,1]。
            - "path_class"：CrossEntropyLoss，输出四分类路径 logits [B,4]；
              需配合 label_spec.mode="oco"。
        seed: 随机种子，用于 torch.manual_seed 与 np.random.seed，保证同 seed 训练可复现；
            不同 seed 产生不同初始化与 DataLoader shuffle，可用于多种子集成评估；默认 42。

    Returns:
        训练结果字典，含 name/model_path/history_path/best_epoch/best_val_loss/
        best_val_acc/beats_baseline/total_params/train_samples/val_samples/
        elapsed_seconds/history/tensor_shape 等键；
        回归模式下额外含 best_val_ic/best_val_rank_ic/best_val_mae/best_val_dir_acc；
        分类模式下额外含 best_val_auc/best_val_f1；
        path_class 模式下额外含 num_classes=4/best_val_tp_auc/best_val_sl_auc/
        best_val_macro_f1/class_distribution（四类样本数字典）。

    Raises:
        ValueError: 样本数不足（< 50）、训练集或验证集为空时抛出。
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
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    is_path_class = objective == "path_class"
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
    # 上涨/TP 样本占比，用于方向基线，戳破「接近随机却看起来还行」的假象。
    # - 分类: y∈{0,1}；y>0 即为上涨正样本。
    # - 回归: y 为连续收益；y>0 为方向向上。
    # - path_class: y∈{0,1,2,3}；用 TP_FIRST 类（y==0）占比作为代表性正样本比率，
    #   语义：模型多大概率可预测到「止盈先触发」路径。y>0 在此无正确语义（会把 sl/time 混入）。
    if is_path_class:
        train_pos_ratio = float(np.mean(y_train == PATH_TP_FIRST)) if len(y_train) else 0.0
        val_pos_ratio = float(np.mean(y_val == PATH_TP_FIRST)) if len(y_val) else 0.0
    else:
        train_pos_ratio = float(np.mean(y_train > 0)) if len(y_train) else 0.0
        val_pos_ratio = float(np.mean(y_val > 0)) if len(y_val) else 0.0

    # #1 幅度加权：让 +5% 比 +0.01% 在损失里权重更大；none 时权重恒为 1（等价旧 BCE）。
    # 回归目标本身即幅度，无需再加权，权重恒为 1。
    # path_class 使用 CrossEntropyLoss，类别 index 无幅度语义，强制回退 none。
    weight_cap = 10.0  # 防止单根涨跌停样本独占梯度
    sample_returns = np.asarray(info.get("sample_returns", []), dtype=np.float32)
    if not is_regression and not is_path_class and loss_weighting == "magnitude" and len(sample_returns) == n:
        abs_returns = np.abs(sample_returns)
        train_scale = float(abs_returns[:n_train].mean())
        scale = train_scale if train_scale > 1e-12 else 1.0
        sample_weights = np.clip(abs_returns / scale, 0.0, weight_cap).astype(np.float32)
    else:
        if loss_weighting == "magnitude" and not is_regression and not is_path_class:
            logger.warning("幅度加权不可用（缺少样本收益），回退为普通 BCE")
            loss_weighting = "none"
        elif is_regression or is_path_class:
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
    # 三路损失选择：
    # - path_class: CrossEntropyLoss（pred [B,4] vs target long [B]，reduction=none 保留样本维度）
    # - regression: HuberLoss（对涨跌停等离群收益更稳健）
    # - classification: BCELoss（reduction=none 以便幅度加权）
    if is_path_class:
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif is_regression:
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
            mb = mb.to(device)

            optimizer.zero_grad()
            pred = model(xb, mb)

            if is_path_class:
                # CrossEntropyLoss 要求 target 为 long [B]，pred 为 [B, 4]
                target = yb.to(device).long()  # [B]
                wb_dev = wb.to(device)          # [B]，path 分支权重无需 squeeze
                loss = (criterion(pred, target) * wb_dev).mean()
                train_correct += (pred.argmax(dim=1) == target).sum().item()
            else:
                yb = yb.to(device).unsqueeze(1)   # [B, 1]
                wb_squeezed = wb.to(device).unsqueeze(1)  # [B, 1]
                loss = (criterion(pred, yb) * wb_squeezed).mean()
                if is_regression:
                    train_correct += ((pred > 0) == (yb > 0)).sum().item()
                else:
                    train_correct += ((pred > 0.5).float() == yb).sum().item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(xb)
            train_total += len(xb)

        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs: list[float] = []
        val_labels: list[float] = []
        val_logits_rows: list[list[float]] = []   # [N, 4]，仅 path_class 使用
        val_class_labels: list[float] = []        # [N]，仅 path_class 使用
        with torch.no_grad():
            for xb, yb, mb, _wb in val_loader:
                xb = xb.to(device)
                mb = mb.to(device)

                pred = model(xb, mb)

                # 验证损失保持不加权，便于早停/选模与历史口径一致可比
                if is_path_class:
                    target = yb.to(device).long()  # [B]
                    loss = criterion(pred, target).mean()
                    val_correct += (pred.argmax(dim=1) == target).sum().item()
                    val_logits_rows.extend(pred.detach().cpu().numpy().tolist())
                    val_class_labels.extend(yb.numpy().reshape(-1).tolist())
                else:
                    yb = yb.to(device).unsqueeze(1)
                    loss = criterion(pred, yb).mean()
                    if is_regression:
                        val_correct += ((pred > 0) == (yb > 0)).sum().item()
                    else:
                        val_correct += ((pred > 0.5).float() == yb).sum().item()
                    val_probs.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
                    val_labels.extend(yb.detach().cpu().numpy().reshape(-1).tolist())

                val_loss += loss.item() * len(xb)
                val_total += len(xb)

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
        if is_path_class:
            pc_metrics = _path_class_metrics(
                np.asarray(val_class_labels, dtype=np.float32),
                np.asarray(val_logits_rows, dtype=np.float64),
            )
            epoch_row.update({
                "val_tp_auc": pc_metrics["tp_auc"],
                "val_sl_auc": pc_metrics["sl_auc"],
                "val_macro_f1": pc_metrics["macro_f1"],
                "val_class_report": pc_metrics["class_report"],
            })
        elif is_regression:
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

        # 用业务指标（回归: RankIC+超额方向准确率 / 分类: AUC / path_class: tp_auc+sl_auc）
        # 选最佳 epoch，而非单看 val_loss，避免选中"loss 低但方向/排序差"的模型。
        current_score = _selection_score(epoch_row, objective=objective)
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
            if is_path_class:
                tp_text = f"{epoch_row['val_tp_auc']:.3f}" if epoch_row["val_tp_auc"] is not None else "N/A"
                sl_text = f"{epoch_row['val_sl_auc']:.3f}" if epoch_row["val_sl_auc"] is not None else "N/A"
                on_progress(
                    pct,
                    f"Epoch {epoch}/{epochs} | val_loss={val_loss:.5f} | "
                    f"tp_auc={tp_text} sl_auc={sl_text} macro_f1={epoch_row['val_macro_f1']:.3f}",
                )
            elif is_regression:
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

    model_config: dict[str, Any] = {
        "in_channels": C,
        "time_steps": T,
        "max_group_width": S,
        "group_count": G,
        "dropout": dropout,
    }
    if is_path_class:
        model_config["num_classes"] = 4

    dataset_info: dict[str, Any] = {
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
        # 对齐丢弃率：训练时因停牌/数据缺失导致 inner join 后流失的 bar 比例
        "alignment_drop_rate": info.get("alignment_drop_rate", 0.0),
    }
    if is_path_class and info.get("class_distribution"):
        dataset_info["class_distribution"] = info["class_distribution"]

    if is_path_class:
        selection_metric = "tp_auc+sl_auc"
    elif is_regression:
        selection_metric = "rank_ic+excess_acc"
    else:
        selection_metric = "auc"

    save_data = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
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
            "seed": seed,
        },
        "normalization": normalization,
        "dataset_info": dataset_info,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_score": round(best_score, 6),
        "selection_metric": selection_metric,
    }

    model_path, history_path = save_cnn_model(name, save_data, history)

    best_metrics: dict[str, Any] = history[best_epoch - 1] if best_epoch > 0 else {}
    best_val_acc = round(best_metrics.get("val_acc", 0.0), 4)
    best_excess_acc = best_metrics.get("val_excess_acc", 0.0)

    # 跑赢基线判据：
    # - path_class: val_tp_auc > 0.5（止盈路径的判别优于随机）
    # - 其余: 方向准确率超过多数类基线（excess_acc > 0）
    if is_path_class:
        best_tp_auc = best_metrics.get("val_tp_auc")
        beats_baseline = bool(best_tp_auc is not None and best_tp_auc > 0.5)
    else:
        beats_baseline = bool(best_excess_acc is not None and best_excess_acc > 0)

    if on_progress:
        verdict = "✅ 跑赢基线" if beats_baseline else "⚠️ 未跑赢多数类基线"
        if is_path_class:
            tp_auc_val = best_metrics.get("val_tp_auc")
            tp_text = f"{tp_auc_val:.3f}" if tp_auc_val is not None else "N/A"
            on_progress(
                100,
                f"训练完成 | 最佳 Epoch={best_epoch} | tp_auc={tp_text} | "
                f"macro_f1={best_metrics.get('val_macro_f1', 0):.3f} | {verdict} | 耗时={elapsed:.0f}s",
            )
        elif is_regression:
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

    # 组装训练告警：超阈值丢弃率加入 warnings 列表（仿 beats_baseline 等既有结果键）
    _train_warnings: list[str] = []
    _align_drop = info.get("alignment_drop_rate", 0.0)
    if _align_drop > ALIGN_DROP_WARN_THRESHOLD:
        _train_warnings.append(
            f"对齐丢弃率 {_align_drop:.1%} 超过阈值 {ALIGN_DROP_WARN_THRESHOLD:.0%}，"
            f"请检查各标的数据完整性"
        )
    if info.get("alignment_warning"):
        # dataset 侧已有更详细告警，去重只加入若 _train_warnings 尚未覆盖
        if not _train_warnings:
            _train_warnings.append(info["alignment_warning"])

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
        # 对齐丢弃率直接暴露在 result 顶层，便于调用方快速判断
        "alignment_drop_rate": _align_drop,
    }
    if _train_warnings:
        result["warnings"] = _train_warnings
    if is_path_class:
        result.update({
            "num_classes": 4,
            "best_val_tp_auc": best_metrics.get("val_tp_auc"),
            "best_val_sl_auc": best_metrics.get("val_sl_auc"),
            "best_val_macro_f1": best_metrics.get("val_macro_f1"),
            "class_distribution": info.get("class_distribution"),
        })
    elif is_regression:
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
