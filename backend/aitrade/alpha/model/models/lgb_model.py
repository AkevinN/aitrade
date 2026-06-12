"""LightGBM 集成学习模型——基于 GBDT 的 Alpha 因子预测模型。"""

from typing import cast

import numpy as np
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt

from ..template import AlphaModel
from ...dataset import AlphaDataset, Segment


class LgbModel(AlphaModel):
    """LightGBM（GBDT）集成学习因子预测模型。

    使用 MSE 作为目标函数，支持早停与验证集监控。
    训练完成后可通过 detail() 生成特征重要性可视化图。

    Example:
        >>> model = LgbModel(learning_rate=0.05, num_leaves=63)
        >>> model.fit(dataset)
        >>> predictions = model.predict(dataset, Segment.TEST)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        log_evaluation_period: int = 1,
        seed: int | None = None
    ):
        """初始化 LightGBM 模型超参数。

        Args:
            learning_rate: 学习率，默认 0.1。
            num_leaves: 每棵树的最大叶子节点数，默认 31。
            num_boost_round: 最大训练轮数，默认 1000。
            early_stopping_rounds: 验证集损失连续多少轮不改善时提前终止，默认 50。
            log_evaluation_period: 每隔多少轮打印一次训练日志，默认 1。
            seed: 随机种子；为 None 时不固定。
        """
        self.params: dict = {
            "objective": "mse",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "seed": seed
        }

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.log_evaluation_period: int = log_evaluation_period

        self.model: lgb.Booster | None = None

    def _prepare_data(self, dataset: AlphaDataset) -> list[lgb.Dataset]:
        """从数据集中准备训练集与验证集的 LightGBM Dataset 列表。

        Args:
            dataset: 已完成预处理的 AlphaDataset 实例。

        Returns:
            长度为 2 的列表 [train_dataset, valid_dataset]，可直接传入 lgb.train。
        """
        ds: list[lgb.Dataset] = []

        for segment in [Segment.TRAIN, Segment.VALID]:
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            data = df.select(df.columns[2: -1]).to_pandas()
            label = np.array(df["label"])

            ds.append(lgb.Dataset(data, label=label))

        return ds

    def fit(self, dataset: AlphaDataset) -> None:
        """训练 LightGBM 模型，含早停机制。

        Args:
            dataset: 已完成 prepare_data 与 process_data 的 AlphaDataset 实例。
        """
        ds: list[lgb.Dataset] = self._prepare_data(dataset)

        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round,
            valid_sets=ds,
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(self.log_evaluation_period)
            ]
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """对指定区间的推断数据进行预测。

        Args:
            dataset: 已拟合完成的数据集，特征列须与训练时一致。
            segment: 目标区间，Segment.TRAIN / VALID / TEST。

        Returns:
            一维 numpy 数组，长度等于目标区间样本数，为模型预测值。

        Raises:
            ValueError: fit 尚未调用（self.model 为 None）时抛出。
        """
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        data: np.ndarray = df.select(df.columns[2: -1]).to_numpy()

        result: np.ndarray = cast(np.ndarray, self.model.predict(data))
        return result

    def detail(self) -> None:
        """展示特征重要性图（split 与 gain 两种口径，仅 Jupyter 环境）。

        调用 lightgbm.plot_importance 生成 matplotlib 图表，
        展示 Top-50 特征。模型未训练时静默返回。
        """
        if not self.model:
            return

        for importance_type in ["split", "gain"]:
            ax: plt.Axes = lgb.plot_importance(
                self.model,
                max_num_features=50,
                importance_type=importance_type,
                figsize=(10, 20)
            )
            ax.set_title(f"Feature Importance ({importance_type})")
