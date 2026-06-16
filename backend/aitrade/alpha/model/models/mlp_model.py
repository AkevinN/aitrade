"""多层感知机模型——基于 PyTorch 的 Alpha 因子预测深度学习模型。"""

import copy
from collections import defaultdict
from typing import Literal, cast

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error      # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim

from ..template import AlphaModel
from ...dataset import AlphaDataset, Segment
from ...logger import logger


class MlpModel(AlphaModel):
    """多层感知机（MLP）Alpha 因子预测模型。

    基于 PyTorch 实现，主要特性：
    1. 可配置隐藏层数量与宽度的全连接网络（含 BatchNorm + Dropout）；
    2. 支持 Adam 与 SGD 优化器及 ReduceLROnPlateau 学习率调度；
    3. 内置早停机制，验证集损失连续 early_stop_rounds 步不改善时终止；
    4. MSE 损失函数；
    5. 通过扰动法（perturbation-based）估算特征重要性。

    Example:
        >>> model = MlpModel(input_size=158, hidden_sizes=(256, 128), n_epochs=200)
        >>> model.fit(dataset)
        >>> predictions = model.predict(dataset, Segment.TEST)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int] = (256,),
        lr: float = 0.001,
        n_epochs: int = 300,
        batch_size: int = 2000,
        early_stop_rounds: int = 50,
        eval_steps: int = 20,
        optimizer: Literal["sgd", "adam"] = "adam",
        weight_decay: float = 0.0,
        device: str = "cpu",
        seed: int | None = None
    ) -> None:
        """初始化 MLP 模型结构、优化器与学习率调度器。

        Args:
            input_size: 输入特征维度，应等于数据集特征列数。
            hidden_sizes: 各隐藏层神经元数量元组，如 (256, 128) 表示两层。默认 (256,)。
            lr: 初始学习率，默认 0.001。
            n_epochs: 最大训练 epoch 数，默认 300。
            batch_size: 每个 mini-batch 的样本数，默认 2000。
            early_stop_rounds: 早停轮数：验证集损失连续此轮数不改善则终止，默认 50。
            eval_steps: 每隔多少 epoch 在验证集上评估一次，默认 20。
            optimizer: 优化器类型，"adam" 或 "sgd"（不区分大小写），默认 "adam"。
            weight_decay: L2 正则化系数，默认 0.0。
            device: 训练设备字符串，如 "cpu"、"cuda"、"mps"，默认 "cpu"。
            seed: 随机种子；为 None 时不固定随机状态。

        Raises:
            NotImplementedError: optimizer 不为 "adam" 或 "sgd" 时抛出。
        """
        self.input_size: int = input_size
        self.hidden_sizes: tuple[int] = hidden_sizes
        self.lr: float = lr
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.early_stop_rounds: int = early_stop_rounds
        self.eval_steps: int = eval_steps
        self.device: str = device
        self.fitted: bool = False
        self.feature_names: list[str] = []
        self.best_step: int | None = None

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._scorer = mean_squared_error

        self.model: nn.Module = MlpNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
        )

        self.model = self.model.to(device)

        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            self.optimizer: optim.Optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.00001,
            eps=1e-08,
        )

    def fit(
        self,
        dataset: AlphaDataset,
        evaluation_results: dict | None = None,
    ) -> None:
        """训练 MLP 模型，含早停与最优参数保存。

        将训练集/验证集数据转为 Tensor 后按 mini-batch 随机采样训练；
        每隔 eval_steps 步在验证集上计算 MSE 损失并更新学习率调度器；
        验证集损失改善时保存最优 state_dict，训练结束后恢复最优参数。

        Args:
            dataset: 已完成 prepare_data 与 process_data 的 AlphaDataset 实例。
            evaluation_results: 可选的空字典，训练过程中会在此填入
                {Segment.TRAIN: [loss_list], Segment.VALID: [loss_list]}，
                用于外部绘制学习曲线。
        """
        if evaluation_results is None:
            evaluation_results = {}

        train_valid_data: dict[str, dict] = defaultdict(dict)

        for segment in [Segment.TRAIN, Segment.VALID]:
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            features = df.select(df.columns[2: -1]).to_numpy()
            labels = np.array(df["label"])

            train_valid_data["x"][segment] = torch.from_numpy(features).float().to(self.device)
            train_valid_data["y"][segment] = torch.from_numpy(labels).float().to(self.device)

            evaluation_results[segment] = []

        df = dataset.fetch_learn(Segment.TRAIN)
        self.feature_names = df.columns[2:-1]

        early_stop_count: int = 0
        train_loss: float = 0
        best_valid_score: float = np.inf
        best_params = None

        train_samples: int = train_valid_data["y"][Segment.TRAIN].shape[0]

        for step in range(1, self.n_epochs + 1):
            if early_stop_count >= self.early_stop_rounds:
                logger.info("达到早停条件,训练结束")
                break

            batch_loss = self._train_step(train_valid_data, train_samples)
            train_loss += batch_loss

            if step % self.eval_steps == 0 or step == self.n_epochs:
                early_stop_count, best_valid_score, best_params = self._evaluate_step(
                    train_valid_data,
                    evaluation_results,
                    step,
                    train_loss,
                    early_stop_count,
                    best_valid_score
                )
                train_loss = 0

        self.fitted = True

        if best_params:
            self.model.load_state_dict(best_params)

    def _train_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        train_samples: int
    ) -> float:
        """执行一步 mini-batch 训练并返回本步损失值。

        随机采样 batch_size 个样本，前向传播计算 MSE 损失，反向传播更新参数。

        Args:
            train_valid_data: 包含 "x" 与 "y" 键的嵌套字典，
                各键下存放 {Segment: Tensor} 映射。
            train_samples: 训练集总样本数，用于随机采样索引范围。

        Returns:
            本步 mini-batch 的 MSE 损失值（float）。
        """
        batch_loss = AverageMeter()
        self.model.train()
        self.optimizer.zero_grad()

        batch_indices = np.random.choice(train_samples, self.batch_size)
        batch_features = train_valid_data["x"][Segment.TRAIN][batch_indices]
        batch_labels = train_valid_data["y"][Segment.TRAIN][batch_indices]

        predictions = self.model(batch_features)
        cur_loss = self._loss_fn(predictions, batch_labels)
        cur_loss.backward()

        self.optimizer.step()
        batch_loss.update(cur_loss.item())

        return batch_loss.val

    def _evaluate_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        evaluation_results: dict[Segment, list[float]],
        step: int,
        train_loss: float,
        early_stop_count: int,
        best_valid_score: float
    ) -> tuple[int, float, dict[str, torch.Tensor] | None]:
        """在验证集上评估当前模型，更新早停计数器与最优参数。

        Args:
            train_valid_data: 包含训练集与验证集 Tensor 的嵌套字典。
            evaluation_results: 记录各 epoch 损失的字典，由 fit 传入并原地更新。
            step: 当前训练步数，用于日志输出。
            train_loss: 本评估周期内训练损失的累积值（将除以 eval_steps 求均值）。
            early_stop_count: 当前早停计数器值。
            best_valid_score: 当前最优验证集损失。

        Returns:
            三元组 (新 early_stop_count, 新 best_valid_score, best_params_or_None)；
            当验证损失改善时 best_params 为最新 state_dict，否则为 None。
        """
        early_stop_count += 1
        train_loss /= self.eval_steps

        with torch.no_grad():
            self.model.eval()

            data: torch.Tensor = train_valid_data["x"][Segment.VALID]
            pred: torch.Tensor = cast(torch.Tensor, self._predict_batch(data, return_cpu=False))
            valid_loss = self._loss_fn(pred, train_valid_data["y"][Segment.VALID])

            loss_val = valid_loss.item()

        logger.info(f"[Step {step}]: train_loss {train_loss:.6f}, valid_loss {loss_val:.6f}")
        evaluation_results[Segment.TRAIN].append(train_loss)
        evaluation_results[Segment.VALID].append(loss_val)

        best_params = None
        if loss_val < best_valid_score:
            logger.info(f"\t验证集损失从 {best_valid_score:.6f} 降低到 {loss_val:.6f}")
            best_valid_score = loss_val
            self.best_step = step
            early_stop_count = 0
            best_params = copy.deepcopy(self.model.state_dict())

        if self.scheduler is not None:
            self.scheduler.step(metrics=valid_loss, epoch=step)

        return early_stop_count, best_valid_score, best_params

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算预测值与目标值之间的 MSE 损失。

        Args:
            pred: 模型输出 Tensor，任意形状（内部会 reshape 为一维）。
            target: 标签 Tensor，形状与 pred 兼容。

        Returns:
            标量 MSE 损失 Tensor。
        """
        pred, target = pred.reshape(-1), target.reshape(-1)
        loss: torch.Tensor = nn.MSELoss()(pred, target)
        return loss

    def _predict_batch(self, data: torch.Tensor, return_cpu: bool = True) -> np.ndarray | torch.Tensor:
        """对输入 Tensor 做批量推断（每批 8096 条，防止 OOM）。

        Args:
            data: 输入特征 Tensor，shape (N, input_size)。
            return_cpu: 为 True 时将结果转为 numpy 数组；
                为 False 时返回 GPU/CPU Tensor（用于训练阶段的损失计算）。

        Returns:
            return_cpu=True 时返回 shape (N,) 的 numpy 数组；
            否则返回合并后的 Tensor。
        """
        data = data.to(self.device)

        predictions: list[torch.Tensor] = []

        self.model.eval()

        with torch.no_grad():
            batch_size: int = 8096
            for i in range(0, len(data), batch_size):
                x: torch.Tensor = data[i: i + batch_size]
                predictions.append(self.model(x.to(self.device)).detach().reshape(-1))

        if return_cpu:
            return cast(np.ndarray, np.concatenate([pr.cpu().numpy() for pr in predictions]))
        else:
            return torch.cat(predictions, dim=0)

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """对指定区间的推断数据进行批量预测。

        Args:
            dataset: 已拟合完成的数据集，特征列须与训练时一致。
            segment: 目标区间，Segment.TRAIN / VALID / TEST。

        Returns:
            一维 numpy 数组，长度等于目标区间样本数，为模型预测值。

        Raises:
            ValueError: fit 尚未调用（self.fitted=False）时抛出。
        """
        if not self.fitted:
            raise ValueError("Model has not been trained yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        data: np.ndarray = df.select(df.columns[2: -1]).to_numpy()

        return cast(np.ndarray, self._predict_batch(torch.Tensor(data)))

    def _check_tensor_nan(self, tensor: torch.Tensor, name: str) -> None:
        """检查 Tensor 中是否含有 NaN 值，有则打印警告。

        Args:
            tensor: 待检查的 Tensor。
            name: 用于日志标识的名称字符串。
        """
        if torch.isnan(tensor).any():
            print(f"NaN values detected: {name}")

    def detail(self) -> pd.DataFrame | None:
        """输出 MLP 模型诊断信息并返回特征重要性 DataFrame。

        通过 logger 打印输入维度、隐藏层大小、参数量、设备、学习率、批大小；
        调用 _calculate_feature_importance 计算并返回基于扰动法的特征重要性表。

        Returns:
            以特征名为索引、"Importance" 为列的 pandas DataFrame，
            按重要性降序排列；模型未训练时返回 None。
        """
        if not self.fitted:
            logger.info("模型尚未训练，无法显示详细信息")
            return None

        logger.info(f"输入特征维度: {self.input_size}")
        logger.info(f"隐藏层大小: {self.hidden_sizes}")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型总参数量: {total_params:,}")

        logger.info(f"训练设备: {self.device}")
        logger.info(f"当前学习率: {self.lr}")
        logger.info(f"批次大小: {self.batch_size}")

        importance_df = self._calculate_feature_importance()
        return importance_df

    def _calculate_feature_importance(self) -> pd.DataFrame:
        """用扰动法估算各特征对模型输出的影响（特征重要性）。

        对 1000 条随机输入，逐特征加入高斯噪声（noise_level=0.1），
        计算输出变化的标准差作为该特征的重要性分数。

        Returns:
            以特征名为索引、"Importance" 为列的 pandas DataFrame，
            按重要性降序排列。
        """
        self.model.eval()
        importance_dict = {}

        test_data = torch.randn(1000, self.input_size).to(self.device)
        base_pred = self.model(test_data).detach()

        noise_level = 0.1
        for i, feature_name in enumerate(self.feature_names):
            perturbed_data = test_data.clone()
            perturbed_data[:, i] += torch.randn(1000).to(self.device) * noise_level

            with torch.no_grad():
                new_pred = self.model(perturbed_data)
                importance = torch.std(torch.abs(new_pred - base_pred)).item()
                importance_dict[feature_name] = importance

        df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        df = df.sort_values('Importance', ascending=False)
        df = df.set_index('Feature')

        return df


class AverageMeter:
    """累计均值计算器，用于追踪训练过程中的损失滑动均值。"""

    def __init__(self) -> None:
        """初始化并重置所有统计量。"""
        self.reset()

    def reset(self) -> None:
        """将 val、avg、sum、count 全部归零。"""
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """更新统计量。

        Args:
            val: 当前步的值（如 mini-batch 损失）。
            n: 当前步的样本权重（通常为 1）。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MlpNetwork(nn.Module):
    """MLP 前向网络结构：输入 Dropout → [Linear → BatchNorm → Activation] × N → Dropout → Linear。

    每个隐藏层由 Linear + BatchNorm1d + 激活函数组成；
    首尾各有一个 Dropout(0.05) 层；权重使用 Kaiming 正态初始化（leaky_relu 模式）。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_sizes: tuple[int] = (256,),
        activation: str = "LeakyReLU"
    ) -> None:
        """初始化 MLP 网络层。

        Args:
            input_size: 输入特征维度。
            output_size: 输出维度，默认 1（回归任务）。
            hidden_sizes: 各隐藏层神经元数量元组，默认 (256,)。
            activation: 激活函数名称，支持 "LeakyReLU"（negative_slope=0.1）
                或 "SiLU"；其他值将在 _get_activation 中抛出 ValueError。
        """
        super().__init__()

        layers: list[nn.Module] = []
        layer_sizes = [input_size] + list(hidden_sizes)

        layers.append(nn.Dropout(0.05))

        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                self._get_activation(activation)
            ])

        layers.extend([
            nn.Dropout(0.05),
            nn.Linear(hidden_sizes[-1], output_size)
        ])

        self.network = nn.ModuleList(layers)

        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """根据名称返回对应的激活函数实例。

        Args:
            name: 激活函数名称，"LeakyReLU" 或 "SiLU"。

        Returns:
            对应的 nn.Module 激活函数实例。

        Raises:
            ValueError: name 不在支持列表时抛出。
        """
        if name == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.1)
        elif name == "SiLU":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function type: {name}")

    def _initialize_weights(self) -> None:
        """使用 Kaiming 正态分布初始化所有 Linear 层权重（leaky_relu 模式，a=0.1）。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.1,
                    mode="fan_in",
                    nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：依次通过 network 中的所有层。

        Args:
            x: 输入 Tensor，shape (batch_size, input_size)。

        Returns:
            输出 Tensor，shape (batch_size, output_size)。
        """
        for layer in self.network:
            x = layer(x)
        return x
