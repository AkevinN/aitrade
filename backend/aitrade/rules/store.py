"""
转债条款快照 + 历史溢价率存储（CBTermsStore）、日度基本面存储（FundamentalStore）。

存储位置：AITRADE_HOME/rules/（由 config.RULES_DATA_PATH 配置）
写入策略：原子写（tmp 文件 + os.replace），避免写一半的脏数据。

文件布局
--------
- rules/cb_snapshot.parquet        : bond_zh_cov 列表快照（含 rating/conv_price/issue_scale 等）
- rules/cb_premium/<vt_symbol>.parquet : 逐债溢价率历史（value_analysis 数据）
- rules/fundamental/<vt_symbol>.parquet : 日度基本面历史（tushare daily_basic 数据）

读取返回 polars.DataFrame；文件不存在时返回 None。

单位说明
--------
FundamentalStore 中 total_mv / circ_mv 存储单位为**万元**（tushare daily_basic 原始值），
与 tushare 官方文档保持一致，**未作任何单位换算**。
使用方在消费这两列时务必注意：1亿元 = 10000万元。
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from ..config import RULES_DATA_PATH

logger = logging.getLogger(__name__)

# 快照文件名
_SNAPSHOT_FILE = "cb_snapshot.parquet"
# 溢价率历史子目录
_PREMIUM_DIR = "cb_premium"


class CBTermsStore:
    """转债条款快照 + 历史溢价率，parquet 落 RULES_DATA_PATH/，原子写（tmp+os.replace）。"""

    def __init__(self, base_path: Path | None = None) -> None:
        self._base = Path(base_path) if base_path else RULES_DATA_PATH
        self._base.mkdir(parents=True, exist_ok=True)
        # 溢价率历史子目录
        self._premium_dir = self._base / _PREMIUM_DIR
        self._premium_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 条款列表快照（bond_zh_cov 返回的整表）
    # -------------------------------------------------------------------------

    def save_snapshot(self, df: pl.DataFrame) -> None:
        """保存转债列表快照（原子写）。

        Args:
            df: polars DataFrame，建议含 symbol/name/conv_price/price/premium_rate/
                issue_scale/rating/list_date 等字段。
        """
        target = self._base / _SNAPSHOT_FILE
        self._atomic_write_parquet(df, target)
        logger.info("CBTermsStore: 快照已保存，共 %d 条记录 -> %s", df.height, target)

    def load_snapshot(self) -> pl.DataFrame | None:
        """读取转债列表快照。文件不存在时返回 None。"""
        path = self._base / _SNAPSHOT_FILE
        if not path.exists():
            return None
        try:
            return pl.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CBTermsStore: 快照读取失败: %s", exc)
            return None

    # -------------------------------------------------------------------------
    # 逐债溢价率历史（bond_zh_cov_value_analysis 返回的逐债数据）
    # -------------------------------------------------------------------------

    def save_premium_history(self, vt_symbol: str, df: pl.DataFrame) -> None:
        """保存单只转债的溢价率历史（原子写）。

        文件名：<vt_symbol>.parquet（vt_symbol 中的 "." 替换为 "_"，保证文件名合法）。

        Args:
            vt_symbol: 如 "113050.SSE" 或 "128093.SZSE"。
            df: polars DataFrame，建议含 date/close/premium_rate 等字段。
        """
        safe_name = vt_symbol.replace(".", "_")
        target = self._premium_dir / f"{safe_name}.parquet"
        self._atomic_write_parquet(df, target)
        logger.debug("CBTermsStore: %s 溢价率历史已保存 -> %s", vt_symbol, target)

    def load_premium_history(self, vt_symbol: str) -> pl.DataFrame | None:
        """读取单只转债的溢价率历史。文件不存在时返回 None。"""
        safe_name = vt_symbol.replace(".", "_")
        path = self._premium_dir / f"{safe_name}.parquet"
        if not path.exists():
            return None
        try:
            return pl.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CBTermsStore: %s 溢价率历史读取失败: %s", vt_symbol, exc)
            return None

    # -------------------------------------------------------------------------
    # 原子写辅助
    # -------------------------------------------------------------------------

    @staticmethod
    def _atomic_write_parquet(df: pl.DataFrame, target: Path) -> None:
        """将 DataFrame 原子写入 target（先写 tmp，再 os.replace）。"""
        # 写入同目录的临时文件，确保 rename 是同一文件系统的原子操作
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=target.parent,
            suffix=".tmp.parquet",
        )
        try:
            os.close(tmp_fd)
            df.write_parquet(tmp_path)
            os.replace(tmp_path, target)
        except Exception:
            # 清理临时文件，然后重新抛出
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# =============================================================================
# FundamentalStore
# =============================================================================

# 基本面历史子目录
_FUNDAMENTAL_DIR = "fundamental"

# 存储列：datetime（date 型）+ 指标列
# 注意：total_mv / circ_mv 单位为**万元**（tushare daily_basic 原始值），未换算
# 此列表是 FundamentalRecord 字段的有意子集：ps（市销率）等字段未在此持久化，
# 如需扩展请同步更新下游消费方及迁移脚本。
_FUNDAMENTAL_COLUMNS = ["datetime", "pe", "pe_ttm", "pb", "total_mv", "circ_mv", "turnover_rate"]


class FundamentalStore:
    """日度基本面（total_mv 等）parquet 落盘。

    存储路径：RULES_DATA_PATH/fundamental/<safe_name>.parquet
    写入策略：原子写（tmp + os.replace）；增量追加时按 datetime 去重合并。

    列约定
    ------
    datetime     : polars.Date（YYYY-MM-DD）
    pe           : Float64 | Null
    pe_ttm       : Float64 | Null
    pb           : Float64 | Null
    total_mv     : Float64 | Null  （**万元**，tushare daily_basic 原始单位，1亿=10000万）
    circ_mv      : Float64 | Null  （**万元**，同上）
    turnover_rate: Float64 | Null

    ⚠ total_mv / circ_mv 单位为万元，消费方需自行换算（÷10000 得亿元）。
    """

    def __init__(self, base_path: Path | None = None) -> None:
        self._base = Path(base_path) if base_path else RULES_DATA_PATH
        self._base.mkdir(parents=True, exist_ok=True)
        self._fund_dir = self._base / _FUNDAMENTAL_DIR
        self._fund_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------------

    def _path_for(self, vt_symbol: str) -> Path:
        """返回标的对应的 parquet 文件路径。"""
        safe_name = vt_symbol.replace(".", "_")
        return self._fund_dir / f"{safe_name}.parquet"

    # -------------------------------------------------------------------------
    # save
    # -------------------------------------------------------------------------

    def save(self, vt_symbol: str, df: pl.DataFrame) -> None:
        """保存（增量追加合并）基本面数据，按 datetime 去重。

        若文件已存在，将已存数据与新数据合并，以 datetime 去重（新数据优先），
        然后原子写回文件。

        Args:
            vt_symbol: 如 "600519.SSE"。
            df: polars DataFrame，必须含 datetime 列（Date 或 Utf8 均可）。
                其余列若缺失将以 Null 填充（保证 schema 稳定）。
        """
        # 规整输入 schema：统一转为标准列
        df = self._normalize_df(df)

        target = self._path_for(vt_symbol)
        existing = self._load_raw(target)

        if existing is not None:
            # 合并：将旧数据与新数据拼接，以 datetime 去重（新数据优先，keep="last"
            # 按拼接顺序，旧在前新在后，last = 新数据）
            merged = pl.concat([existing, df], how="diagonal")
            merged = merged.unique(subset=["datetime"], keep="last")
            merged = merged.sort("datetime")
        else:
            merged = df.sort("datetime")

        CBTermsStore._atomic_write_parquet(merged, target)
        logger.debug("FundamentalStore: %s 已保存，共 %d 条 -> %s", vt_symbol, merged.height, target)

    # -------------------------------------------------------------------------
    # load
    # -------------------------------------------------------------------------

    def load(
        self,
        vt_symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame | None:
        """读取基本面历史，可按日期区间过滤。

        Args:
            vt_symbol: 标的 vt_symbol。
            start: 开始日期（含），None 表示不限。
            end: 结束日期（含），None 表示不限。

        Returns:
            polars DataFrame 或 None（文件不存在）。
        """
        target = self._path_for(vt_symbol)
        df = self._load_raw(target)
        if df is None:
            return None

        if start is not None:
            df = df.filter(pl.col("datetime") >= pl.lit(start))
        if end is not None:
            df = df.filter(pl.col("datetime") <= pl.lit(end))
        return df

    # -------------------------------------------------------------------------
    # list_symbols
    # -------------------------------------------------------------------------

    def list_symbols(self) -> list[str]:
        """返回已落盘的所有标的 vt_symbol 列表（按字母顺序）。"""
        result = []
        for p in sorted(self._fund_dir.glob("*.parquet")):
            # 文件名格式：<safe_name>.parquet，safe_name = vt_symbol.replace(".", "_")
            # 反推：将最后一个 "_" 替换回 "."（交易所后缀固定为 3~4 位大写字母）
            stem = p.stem  # 如 "600519_SSE"
            # 交易所后缀：SSE / SZSE / BSE 等
            idx = stem.rfind("_")
            if idx != -1:
                vt_symbol = stem[:idx] + "." + stem[idx + 1:]
            else:
                vt_symbol = stem
            result.append(vt_symbol)
        return result

    # -------------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_df(df: pl.DataFrame) -> pl.DataFrame:
        """规整输入 DataFrame：确保含标准列，转换 datetime 类型。"""
        # 如果 datetime 列是字符串，转为 Date（严格模式：非 YYYYMMDD 格式立即抛异常，
        # 而非静默置 null 后被 unique() 坍缩成单行导致数据无声消失）
        if "datetime" in df.columns and df["datetime"].dtype in (pl.Utf8, pl.String):
            # 先用非严格模式找出非法样例，再用严格模式触发 fail-fast
            probed = df["datetime"].str.to_date("%Y%m%d", strict=False)
            bad_mask = df["datetime"].is_not_null() & probed.is_null()
            if bad_mask.any():
                samples = df["datetime"].filter(bad_mask).head(3).to_list()
                raise ValueError(
                    f"基本面数据 datetime 列格式非法（期望 YYYYMMDD），样例: {samples}"
                )
            df = df.with_columns(probed.alias("datetime"))

        # 补充缺失的指标列为 Null
        for col in _FUNDAMENTAL_COLUMNS:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # 只保留标准列（多余列丢弃），保持 schema 稳定
        return df.select(_FUNDAMENTAL_COLUMNS)

    @staticmethod
    def _load_raw(target: Path) -> pl.DataFrame | None:
        """读取 parquet 文件，不存在或损坏时返回 None。"""
        if not target.exists():
            return None
        try:
            return pl.read_parquet(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FundamentalStore: 读取失败 %s: %s", target, exc)
            return None
