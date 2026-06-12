"""LASSO 回归模型——基于 sklearn 的 L1 正则化线性因子预测模型。"""

import numpy as np
import polars as pl
from sklearn.linear_model import Lasso      # type: ignore

from ..template import AlphaModel
from ...dataset import AlphaDataset, Segment
from ...logger import logger


class LassoModel(AlphaModel):
    """LASSO（L1 正则化）线性回归因子预测模型。

    将训练集与验证集合并后拟合 LASSO 回归，输出各特征的稀疏线性权重。
    适合特征数较多但期望稀疏解的场景，可通过 detail() 查看非零特征及权重。

    Example:
        >>> model = LassoModel(alpha=0.001)
        >>> model.fit(dataset)
        >>> predictions = model.predict(dataset, Segment.TEST)
    """

    def __init__(
        self,
        alpha: float = 0.0005,
        max_iter: int = 1000,
        random_state: int | None = None,
    ) -> None:
        """初始化 LASSO 模型超参数。

        Args:
            alpha: L1 正则化系数，越大稀疏性越强，默认 0.0005。
            max_iter: 坐标下降最大迭代次数，默认 1000。
            random_state: 随机种子；为 None 时不固定随机状态。
        """
        self.alpha: float = alpha
        self.max_iter: int = max_iter
        self.random_state: int | None = random_state

        self.model: Lasso = None

        self.feature_names: list[str] = []

    def fit(self, dataset: AlphaDataset) -> None:
        """合并训练集与验证集后拟合 LASSO 回归模型。

        去重后按 datetime/vt_symbol 排序，不拟合截距项（fit_intercept=False）。
        训练完成后 self.model 和 self.feature_names 可用。

        Args:
            dataset: 已完成 prepare_data 与 process_data 的 AlphaDataset 实例。
        """
        df_train: pl.DataFrame = dataset.fetch_learn(Segment.TRAIN)
        df_valid: pl.DataFrame = dataset.fetch_learn(Segment.VALID)

        df_train = pl.concat([df_train, df_valid])
        df_train = df_train.unique(subset=["datetime", "vt_symbol"])
        df_train = df_train.sort(["datetime", "vt_symbol"])

        self.feature_names = df_train.columns[2:-1]

        X: np.ndarray = df_train.select(self.feature_names).to_numpy()
        y: np.ndarray = np.array(df_train["label"])

        self.model = Lasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            fit_intercept=False,
            copy_X=False
        )
        self.model.fit(X, y)

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """对指定区间的推断数据做线性预测。

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

        result: np.ndarray = self.model.predict(data)

        return result

    def detail(self) -> None:
        """打印非零特征权重，按绝对值降序排列。

        过滤掉权重为零的特征后，通过 logger.info 逐行输出特征名与权重值（保留 6 位小数）。
        需在 fit 之后调用。
        """
        coef: np.ndarray = self.model.coef_

        data: list[tuple[str, float]] = list(zip(self.feature_names, coef, strict=False))

        data = [x for x in data if x[1]]

        data.sort(key=lambda x: abs(x[1]), reverse=True)

        data = [x for x in data if round(x[1], 6) != 0]

        logger.info(f"LASSO模型特征总数量: {len(data)}")

        for name, importance in data:
            logger.info(f"{name}: {importance:.6f}")
