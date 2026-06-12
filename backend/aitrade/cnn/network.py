"""
CNN 网络工厂 —— 分组感知的多尺度行情 CNN。

刻意保留「工厂函数 + 函数内延迟 import torch」的形式：
这样在未安装 torch 的环境下也能 import aitrade.cnn 做数据集构建/状态检查，
torch 仅在真正创建模型（训练/推理）时才被导入。
"""

from __future__ import annotations


def create_market_cnn(
    in_channels: int,
    time_steps: int,
    width: int,
    group_count: int = 1,
    dropout: float = 0.5,
    objective: str = "classification",
):
    """创建分组感知的多尺度行情 CNN 实例（GroupAwareMarketCNN）。

    网络结构：
    - 三路并行多尺度 2D 卷积（kernel_time = 1/3/5，kernel_width = 1 或 3），
      在时间轴上捕获短/中/长期模式，输出各 16 通道，拼接为 48 通道。
    - 分组掩码加权池化（symbol 维度），将多证券信息聚合为单向量。
    - 1D 时序卷积 + AdaptiveAvgPool1d(8) 提取时间特征。
    - 全连接融合头：按 group 数线性展开后做分类/回归预测。

    输入张量规约（forward 的 x 参数）：``[B, C, T, S, G]``
    - B: batch size
    - C: 特征通道数（in_channels，对应 FEATURE_NAMES 长度，通常为 6）
    - T: 时间步数（time_steps，即 lookback）
    - S: 每组最大证券数（width，即 max_group_width）
    - G: 分组数（group_count）

    group_mask 参数形状：``[B, 1, 1, S, G]``，有效证券位置为 1.0，占位为 0.0。

    Args:
        in_channels: 输入特征通道数 C（通常等于 len(FEATURE_NAMES) = 6）。
        time_steps: 时间窗口长度 T（即训练 lookback）。
        width: 每组最大证券数 S（即 max_group_width）。
        group_count: 观测分组数 G；默认 1（单分组）。
        dropout: 全连接头前的 Dropout 概率；默认 0.5。
        objective: 训练目标，"classification"（输出经 Sigmoid 的上涨概率 0~1）
            或 "regression"（输出无界的预测收益，无 Sigmoid）。

    Returns:
        初始化后的 GroupAwareMarketCNN 实例（torch.nn.Module 子类）。
    """
    import torch
    import torch.nn as nn

    class GroupAwareMarketCNN(nn.Module):
        def __init__(self, C: int, T: int, S: int, G: int, drop: float = 0.5, task: str = "classification") -> None:
            """初始化网络层。

            Args:
                C: 输入特征通道数（即 in_channels）。
                T: 时间步数（即 lookback，决定 temporal_conv 输入长度）。
                S: 每组最大证券数（即 max_group_width）。
                G: 分组数（group_count）。
                drop: Dropout 概率。
                task: "classification" 或 "regression"，决定融合头是否附 Sigmoid。
            """
            super().__init__()
            # Use odd kernel widths only so width stays stable after padding.
            kernel_width = 3 if S >= 3 else 1
            padding_width = 1 if kernel_width == 3 else 0
            self.time_steps = T
            self.max_group_width = S
            self.group_count = G

            self.conv_s = nn.Sequential(
                nn.Conv2d(C, 16, (1, kernel_width), padding=(0, padding_width)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(C, 16, (3, kernel_width), padding=(1, padding_width)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.conv_l = nn.Sequential(
                nn.Conv2d(C, 16, (5, kernel_width), padding=(2, padding_width)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )

            self.temporal_conv = nn.Sequential(
                nn.Conv1d(48, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )

            fusion_hidden = max(96, 32 * G)
            head_layers = [
                nn.Flatten(),
                nn.Linear(32 * 8 * G, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(fusion_hidden, 1),
            ]
            # 回归头不加 Sigmoid，输出无界的预测收益；分类头保留 Sigmoid 输出概率
            if task != "regression":
                head_layers.append(nn.Sigmoid())
            self.group_fusion = nn.Sequential(*head_layers)

        def forward(self, x, group_mask):
            """前向传播：多尺度卷积 → 掩码池化 → 时序提取 → 分组融合 → 预测。

            Args:
                x: 输入特征张量，形状 ``[B, C, T, S, G]``。
                group_mask: 分组有效性掩码，形状 ``[B, 1, 1, S, G]``；
                    有效证券位为 1.0，无效（占位）位为 0.0。

            Returns:
                预测结果张量，形状 ``[B, 1]``；
                分类任务为上涨概率（0~1），回归任务为预测收益（无界）。
            """
            batch_size, _, _, _, group_count = x.shape
            x = x.permute(0, 4, 1, 2, 3).reshape(batch_size * group_count, x.shape[1], x.shape[2], x.shape[3])

            mask = group_mask.permute(0, 4, 1, 2, 3).reshape(
                batch_size * group_count,
                1,
                1,
                self.max_group_width,
            )
            mask = mask.expand(-1, 1, self.time_steps, -1)

            features = torch.cat([self.conv_s(x), self.conv_m(x), self.conv_l(x)], dim=1)
            masked = features * mask
            denom = mask.sum(dim=3, keepdim=True).clamp_min(1.0)
            pooled = masked.sum(dim=3, keepdim=True) / denom
            temporal = self.temporal_conv(pooled.squeeze(-1))
            temporal = temporal.reshape(batch_size, group_count, temporal.shape[1], temporal.shape[2])
            return self.group_fusion(temporal)

    return GroupAwareMarketCNN(in_channels, time_steps, width, group_count, dropout, objective)


def _create_model(
    in_channels: int,
    time_steps: int,
    width: int,
    group_count: int = 1,
    dropout: float = 0.5,
):
    """create_market_cnn 的向后兼容别名（默认分类目标）。

    Args:
        in_channels: 输入特征通道数 C。
        time_steps: 时间窗口长度 T。
        width: 每组最大证券数 S（即 max_group_width）。
        group_count: 观测分组数 G；默认 1。
        dropout: Dropout 概率；默认 0.5。

    Returns:
        GroupAwareMarketCNN 实例（objective 固定为 "classification"）。
    """
    return create_market_cnn(in_channels, time_steps, width, group_count, dropout)
