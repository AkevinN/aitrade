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
    """Create a group-aware market CNN.

    objective:
    - "classification"：输出经 Sigmoid 的上涨概率（0~1），配 BCELoss。
    - "regression"：输出线性预测的未来收益（无界），配 Huber/MSE。
    """
    import torch
    import torch.nn as nn

    class GroupAwareMarketCNN(nn.Module):
        def __init__(self, C: int, T: int, S: int, G: int, drop: float = 0.5, task: str = "classification") -> None:
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
    """Compatibility alias."""
    return create_market_cnn(in_channels, time_steps, width, group_count, dropout)
