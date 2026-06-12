"""Alpha 数据集模板——AlphaDataset 主类及其配套工具函数。

AlphaDataset 负责：注册表达式/Polars 因子特征与标签、并行计算特征、
按训练/验证/测试区间切分数据，以及调用预处理管道。
"""

import time
from datetime import datetime
from typing import cast
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import BaseContext

import polars as pl
import pandas as pd
from tqdm import tqdm

try:
    from alphalens.utils import get_clean_factor_and_forward_returns    # type: ignore
    from alphalens.tears import create_full_tear_sheet                  # type: ignore
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
    get_clean_factor_and_forward_returns = None
    create_full_tear_sheet = None

from ..logger import logger
from .utility import (
    to_datetime,
    Segment,
    calculate_by_expression,
    calculate_by_polars
)


class AlphaDataset:
    """Alpha 因子数据集基类，管理特征/标签注册、计算、区间切分与预处理。

    典型用法：
    1. 构造时传入原始行情 DataFrame 及三段时间区间；
    2. 调用 add_feature / set_label 注册因子表达式；
    3. 调用 prepare_data 并行计算所有特征；
    4. 调用 process_data 执行预处理管道；
    5. 通过 fetch_raw / fetch_infer / fetch_learn 按区间取数。

    Example:
        >>> ds = AlphaDataset(df, ("20200101", "20211231"), ("20220101", "20221231"), ("20230101", "20231231"))
        >>> ds.add_feature("alpha1", "cs_rank(close)")
        >>> ds.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        >>> ds.prepare_data()
        >>> ds.process_data()
        >>> train_df = ds.fetch_learn(Segment.TRAIN)
    """

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        process_type: str = "append"
    ) -> None:
        """初始化数据集并划分训练/验证/测试区间。

        Args:
            df: 原始行情 DataFrame，必须包含 datetime、vt_symbol 及 OHLCV 等列。
            train_period: 训练区间 (start, end)，日期格式 "YYYY-MM-DD" 或 "YYYYMMDD"。
            valid_period: 验证区间，格式同上。
            test_period: 测试区间，格式同上。
            process_type: 预处理模式。"append" 表示 learn_df 在 infer 处理后再追加
                learn_processors；其他值表示先从 raw_df 独立应用 learn_processors。
        """
        self.df: pl.DataFrame = df

        self.result_df: pl.DataFrame
        self.raw_df: pl.DataFrame
        self.infer_df: pl.DataFrame
        self.learn_df: pl.DataFrame

        self.data_periods: dict[Segment, tuple[str, str]] = {
            Segment.TRAIN: train_period,
            Segment.VALID: valid_period,
            Segment.TEST: test_period
        }

        self.feature_expressions: dict[str, str | pl.expr.expr.Expr] = {}
        self.feature_results: dict[str, pl.DataFrame] = {}
        self.label_expression: str = ""

        self.process_type: str = process_type
        self.infer_processors: list = []
        self.learn_processors: list = []

    def add_feature(
        self,
        name: str,
        expression: str | pl.expr.expr.Expr | None = None,
        result: pl.DataFrame | None = None
    ) -> None:
        """注册一个特征。

        expression 和 result 只能传其中一个：expression 为字符串/Polars 表达式，
        在 prepare_data 阶段按表达式计算；result 为已算好的 DataFrame（含
        datetime、vt_symbol、data 三列），直接 join 到结果表。

        Args:
            name: 特征名称，将成为结果 DataFrame 的列名。
            expression: 字符串表达式（如 "cs_rank(close)"）或 Polars Expr 对象，
                与 result 互斥。
            result: 已预计算的特征 DataFrame（列为 datetime、vt_symbol、data），
                与 expression 互斥。

        Raises:
            ValueError: 同时提供了 expression 和 result 时抛出。
        """
        if expression is not None and result is not None:
            raise ValueError("Only one of 'expression' or 'result' can be provided")

        if expression is not None:
            self.feature_expressions[name] = expression
        elif result is not None:
            self.feature_results[name] = result

    def set_label(self, expression: str) -> None:
        """设置标签表达式。

        标签列名固定为 "label"，始终排在结果 DataFrame 的最后一列。
        prepare_data 时与特征表达式一并计算。

        Args:
            expression: 字符串表达式，如 "ts_delay(close, -3) / ts_delay(close, -1) - 1"。
        """
        self.label_expression = expression

    def add_processor(self, task: str, processor: Callable[[pl.DataFrame], None]) -> None:
        """注册预处理器。

        预处理器在 process_data 阶段按注册顺序依次应用。infer 处理器作用于
        infer_df（推断数据）；learn 处理器作用于 learn_df（学习/训练数据）。
        当 process_type == "append" 时，learn_df 先继承 infer 处理结果，
        再追加 learn_processors。

        Args:
            task: 处理器类型，"infer" 或其他任意字符串（视为 "learn"）。
            processor: 可调用对象，签名为 (df: pl.DataFrame) -> pl.DataFrame，
                接收当前 DataFrame 并返回处理后的 DataFrame。
        """
        if task == "infer":
            self.infer_processors.append(processor)
        else:
            self.learn_processors.append(processor)

    def prepare_data(self, filters: dict | None = None, max_workers: int | None = None) -> None:
        """并行计算所有注册的特征与标签，构建 raw_df。

        计算完成后将结果列 join 到原始行情 DataFrame，生成 self.raw_df；
        并初始化 self.infer_df 与 self.learn_df 均指向 raw_df（未经预处理）。

        Args:
            filters: 可选的成分股过滤字典，格式为
                {vt_symbol: [(start_date, end_date), ...]}；
                不为 None 时仅保留各标的在指定区间内的行。
            max_workers: 并行进程数；为 None 或 1 时在主进程串行计算（调试友好）；
                大于 1 时使用 spawn 模式多进程池。
        """
        results: list = []
        max_workers = max(1, max_workers or 1)

        expressions: list[tuple[str, str | pl.expr.expr.Expr]] = list(self.feature_expressions.items())

        if self.label_expression:
            expressions.append(("label", self.label_expression))

        logger.info("开始计算表达式因子特征")

        args: list[tuple] = [(self.df, name, expression) for name, expression in expressions]

        if max_workers == 1:
            for arg in tqdm(args, total=len(args)):
                results.append(calculate_feature(arg))
        else:
            context: BaseContext = get_context("spawn")

            with context.Pool(processes=max_workers) as pool:
                it = pool.imap(calculate_feature, args)

                for result in tqdm(it, total=len(args)):
                    results.append(result)

        self.result_df = self.df.with_columns(results)

        logger.info("开始合并结果数据因子特征")

        label_exist: bool = "label" in self.result_df
        for name, feature_result in tqdm(self.feature_results.items()):
            feature_result = feature_result.rename({"data": name})
            self.result_df = self.result_df.join(feature_result, on=["datetime", "vt_symbol"], how="left")

        if label_exist:
            cols: list = [col for col in self.result_df.columns if col != "label"] + ["label"]
            self.result_df = self.result_df.select(cols).sort(["datetime", "vt_symbol"])

        raw_df = self.result_df.fill_null(float("nan"))

        if filters:
            logger.info("开始筛选成分股数据")

            dfs: list[pl.DataFrame] = []

            for vt_symbol, ranges in tqdm(filters.items(), total=len(filters)):
                for start, end in ranges:
                    temp_df = raw_df.filter(
                        (pl.col("vt_symbol") == vt_symbol)
                        & (pl.col("datetime") >= pl.lit(start))
                        & (pl.col("datetime") <= pl.lit(end))
                    )
                    dfs.append(temp_df)

            raw_df = pl.concat(dfs)

        select_columns: list[str] = ["datetime", "vt_symbol"] + raw_df.columns[self.df.width:]
        self.raw_df = raw_df.select(select_columns).sort(["datetime", "vt_symbol"])

        self.infer_df = self.raw_df
        self.learn_df = self.raw_df

    def process_data(self) -> None:
        """按序执行已注册的预处理管道，更新 infer_df 与 learn_df。

        先对 infer_df 依次应用 infer_processors；
        若 process_type == "append"，learn_df 继承处理后的 infer_df 再应用
        learn_processors；否则 learn_df 从 raw_df 独立执行 learn_processors。
        需在 prepare_data 之后调用。
        """
        for processor in self.infer_processors:
            self.infer_df = processor(df=self.infer_df)

        if self.process_type == "append":
            self.learn_df = self.infer_df

        for processor in self.learn_processors:
            self.learn_df = processor(df=self.learn_df)

    def fetch_raw(self, segment: Segment) -> pl.DataFrame:
        """取指定区间的原始（未预处理）特征数据。

        Args:
            segment: 区间枚举，Segment.TRAIN / VALID / TEST。

        Returns:
            按 [datetime, vt_symbol] 排序的 raw_df 切片，包含特征列与标签列。
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.raw_df, start, end)

    def fetch_infer(self, segment: Segment) -> pl.DataFrame:
        """取指定区间的推断数据（经过 infer 预处理管道）。

        用于模型推断/预测阶段，特征已经过 infer_processors 标准化。

        Args:
            segment: 区间枚举，Segment.TRAIN / VALID / TEST。

        Returns:
            按 [datetime, vt_symbol] 排序的 infer_df 切片。
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.infer_df, start, end)

    def fetch_learn(self, segment: Segment) -> pl.DataFrame:
        """取指定区间的学习数据（经过完整预处理管道）。

        用于模型训练阶段，特征已经过 infer_processors + learn_processors 处理。

        Args:
            segment: 区间枚举，Segment.TRAIN / VALID / TEST。

        Returns:
            按 [datetime, vt_symbol] 排序的 learn_df 切片。
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.learn_df, start, end)

    def show_feature_performance(self, name: str) -> None:
        """用 Alphalens 展示单因子绩效分析图（仅 Jupyter 环境）。

        自动跨越三段区间取全量数据，将 learn_df 中的目标因子与 close 价格合并后，
        调用 get_clean_factor_and_forward_returns 并输出完整 tear sheet。

        Args:
            name: 要分析的特征列名，必须已存在于 learn_df 中。

        Raises:
            ImportError: 若未安装 alphalens 则运行时报错（ALPHALENS_AVAILABLE=False）。
        """
        starts: list[datetime] = []
        ends: list[datetime] = []

        for period in self.data_periods.values():
            starts.append(to_datetime(period[0]))
            ends.append(to_datetime(period[1]))

        start: datetime = min(starts)
        end: datetime = max(ends)

        result_df: pl.DataFrame = query_by_time(self.result_df, start, end)
        learn_df: pl.DataFrame = query_by_time(self.learn_df, start, end)

        merged_df = (
            result_df
            .select(["datetime", "vt_symbol", "close"])
            .join(
                learn_df.select(["datetime", "vt_symbol", name]),
                on=["datetime", "vt_symbol"],
                how="inner"
            )
        )

        merged_df = merged_df.fill_nan(None).drop_nulls()

        feature_df: pd.DataFrame = merged_df.select(["datetime", "vt_symbol", name]).to_pandas()
        feature_df.set_index(["datetime", "vt_symbol"], inplace=True)

        feature_s: pd.Series = feature_df[name]

        price_df: pd.DataFrame = merged_df.select(["datetime", "vt_symbol", "close"]).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")

        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(feature_s, price_df, quantiles=10)  # type: ignore

        if ALPHALENS_AVAILABLE and create_full_tear_sheet:
            create_full_tear_sheet(clean_data)  # type: ignore

    def show_signal_performance(self, signal: pl.DataFrame) -> None:
        """用 Alphalens 展示模型预测信号的绩效分析图（仅 Jupyter 环境）。

        将外部传入的信号 DataFrame 与 result_df 中的 close 价格对齐，
        调用 Alphalens 生成完整 tear sheet，max_loss=1.0 以保留全量样本。

        Args:
            signal: 包含 datetime、vt_symbol、signal 三列的 Polars DataFrame；
                时间范围应与 result_df 有交集。

        Raises:
            ImportError: 若未安装 alphalens 则运行时报错。
        """
        start: datetime = cast(datetime, signal["datetime"].min())
        end: datetime = cast(datetime, signal["datetime"].max())

        df: pl.DataFrame = query_by_time(self.result_df, start, end)

        signal_df: pd.DataFrame = signal.to_pandas()
        signal_df.set_index(["datetime", "vt_symbol"], inplace=True)
        signal_s: pd.Series = signal_df["signal"]

        price_df: pd.DataFrame = df.select(["datetime", "vt_symbol", "close"]).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")

        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(  # type: ignore
            signal_s,
            price_df,
            max_loss=1.0,
            quantiles=10
        )

        if ALPHALENS_AVAILABLE and create_full_tear_sheet:
            create_full_tear_sheet(clean_data)  # type: ignore


def query_by_time(df: pl.DataFrame, start: datetime | str = "", end: datetime | str = "") -> pl.DataFrame:
    """按时间范围过滤 DataFrame 并按 [datetime, vt_symbol] 排序。

    start/end 支持 datetime 对象或字符串（"YYYY-MM-DD"/"YYYYMMDD"），
    均为闭区间；传空字符串则不过滤对应边界。

    Args:
        df: 包含 datetime 列的 Polars DataFrame。
        start: 起始时间（含）；空字符串表示不限下界。
        end: 结束时间（含）；空字符串表示不限上界。

    Returns:
        过滤后按 [datetime, vt_symbol] 升序排列的 DataFrame。
    """
    if start:
        start = to_datetime(start)
        df = df.filter(pl.col("datetime") >= start)

    if end:
        end = to_datetime(end)
        df = df.filter(pl.col("datetime") <= end)

    return df.sort(["datetime", "vt_symbol"])


def calculate_feature(args: tuple[pl.DataFrame, str, str | pl.expr.expr.Expr]) -> pl.Series:
    """计算单个特征并返回命名 Series（进程池 worker 函数）。

    接收打包参数以兼容 multiprocessing.Pool.imap，根据 expression 类型
    分别调用 calculate_by_polars 或 calculate_by_expression，并打印耗时。

    Args:
        args: 三元组 (df, name, expression)：
            - df: 原始行情 DataFrame。
            - name: 特征名，用于为输出 Series 命名。
            - expression: 字符串表达式或 Polars Expr 对象。

    Returns:
        名称为 name 的 Polars Series，长度与 df 行数一致。
    """
    start = time.time()

    df, name, expression = args

    if isinstance(expression, pl.expr.expr.Expr):
        result = calculate_by_polars(df, expression)["data"].alias(name)
    else:
        result = calculate_by_expression(df, expression)["data"].alias(name)

    end = time.time()
    print(f"Feature calculation {name} took: {end - start} seconds | {expression}")

    return result
