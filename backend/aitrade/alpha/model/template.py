"""Alpha 模型抽象基类——定义所有机器学习算法模型的统一接口。"""

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

from ..dataset import AlphaDataset, Segment


class AlphaModel(metaclass=ABCMeta):
    """Alpha 因子预测模型抽象基类。

    所有具体模型（LassoModel、LgbModel、MlpModel 等）均继承此类，
    并实现 fit 与 predict 方法。detail 方法为可选实现，用于输出模型诊断信息。
    """

    @abstractmethod
    def fit(self, dataset: AlphaDataset) -> None:
        """在数据集上拟合（训练）模型。

        通常使用 dataset.fetch_learn(Segment.TRAIN) 与
        dataset.fetch_learn(Segment.VALID) 获取训练和验证数据。

        Args:
            dataset: 已调用 prepare_data 与 process_data 的 AlphaDataset 实例。
        """
        pass

    @abstractmethod
    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """对指定区间的数据进行预测，返回信号数组。

        通常使用 dataset.fetch_infer(segment) 获取推断数据。

        Args:
            dataset: 已拟合完成的数据集，需与训练时特征列一致。
            segment: 目标区间，Segment.TRAIN / VALID / TEST。

        Returns:
            一维 numpy 数组，长度等于目标区间样本数，表示预测的因子值/信号强度。

        Raises:
            ValueError: 模型尚未训练时应抛出。
        """
        pass

    def detail(self) -> Any:
        """输出模型详细诊断信息（可选实现）。

        子类可重写此方法，输出特征重要性、权重分布、训练曲线等信息。
        默认实现不输出任何内容。

        Returns:
            依子类而定，可为 None、DataFrame 等；基类返回 None。
        """
        return
