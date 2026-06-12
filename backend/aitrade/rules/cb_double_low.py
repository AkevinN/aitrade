"""
可转债双低轮动信号源（cb_double_low）。

设计要点
--------
- 双低分 = 收盘价 + 溢价率×100（越低越好）。
  输出 signal = -(收盘价 + 溢价率×100)，越高对应"越双低"，对齐 TopK 降序消费。
- 行情：从 AlphaLab 读转债日线。
- 溢价率：优先从 CBTermsStore 逐债历史（bond_zh_cov_value_analysis 数据）取；
  若该转债某日缺失则用快照中的当前溢价率回退，并 logger.warning 说明（v1 容忍）。
- 过滤：
  - price > max_price（绝对价格过高，安全垫低）
  - rating < min_rating（信用评级不达标）
  - issue_scale < min_issue_scale（规模太小，流动性差）
  - 上市天数 < min_list_days（刚上市的新券，行情不稳定）
- 全空 → RuntimeError 中文（提示用户先跑 cb-terms/refresh 与行情下载）
- 不引入 torch（规则信号源红线）
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import polars as pl

from ..backtest.registry import SignalProvider, register_signal_source

logger = logging.getLogger(__name__)

# 评级序列（从高到低），AA- 及以上才允许进入
# 显式列表比较，避免字符串排序的歧义
_RATING_ORDER: list[str] = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC",
    "CC",
    "C",
    "D",
]


def _rating_rank(rating: str) -> int:
    """返回评级排名（数值越小 = 越好）；未知评级返回极大值（最差）。"""
    try:
        return _RATING_ORDER.index(rating.strip())
    except ValueError:
        return len(_RATING_ORDER)


def _rating_ge(rating: str, min_rating: str) -> bool:
    """判断 rating >= min_rating（评级足够好）。"""
    return _rating_rank(rating) <= _rating_rank(min_rating)


class _CbDoubleLowSource:
    """可转债双低轮动信号源的私有实现，实现 SignalProvider 协议。"""

    def __init__(
        self,
        top_k: int,
        max_price: float,
        min_rating: str,
        min_issue_scale: float,
        min_list_days: int,
        interval: str,
        lab: Any,
        terms_store: Any,
    ) -> None:
        self._top_k = top_k  # 信息用：真实持仓数在策略层（TopK rule）配置，此处不作截断
        self._max_price = max_price
        self._min_rating = min_rating
        self._min_issue_scale = min_issue_scale
        self._min_list_days = min_list_days
        self._interval = interval
        self._lab = lab
        self._terms_store = terms_store

    def predict(
        self,
        start: date,
        end: date,
        on_progress: object | None = None,
    ) -> pl.DataFrame:
        """计算双低信号，返回 [datetime, vt_symbol, signal] DataFrame。

        signal = -(close + premium_rate * 100)，越高 = 越双低（对齐 TopK 降序选股）。

        Args:
            start: 信号区间起始日（含）。
            end: 信号区间截止日（含）。
            on_progress: 可选进度回调。

        Returns:
            Polars DataFrame，列为 datetime(Datetime) / vt_symbol(Utf8) / signal(Float64)，
            按 (datetime, vt_symbol) 升序排序。

        Raises:
            RuntimeError: 无任何输出时抛出中文错误（提示先刷新转债数据）。
        """
        # ---- 1. 加载快照，确定 universe ----
        snapshot = self._terms_store.load_snapshot()
        if snapshot is None or snapshot.is_empty():
            raise RuntimeError(
                "转债条款快照不存在，请先调用 POST /api/strategy/cb-terms/refresh 刷新数据"
            )

        # 快照字段映射（bond_zh_cov 返回的中文列名）
        col_code = self._detect_column(snapshot, ["债券代码", "symbol", "代码"])
        col_rating = self._detect_column(snapshot, ["信用评级", "rating"])
        col_scale = self._detect_column(snapshot, ["发行规模", "issue_scale"])
        col_premium = self._detect_column(snapshot, ["转股溢价率", "premium_rate"])
        col_list_date = self._detect_column(snapshot, ["上市时间", "list_date"])

        # 构建代码 → 元信息字典（快速查阅）
        terms_map: dict[str, dict] = {}
        for row in snapshot.to_dicts():
            code = str(row.get(col_code, "") or "").strip()
            if not code:
                continue
            terms_map[code] = {
                "rating": str(row.get(col_rating, "") or "").strip() if col_rating else "",
                "issue_scale": self._safe_float(row.get(col_scale) if col_scale else None),
                "snapshot_premium": self._safe_float(row.get(col_premium) if col_premium else None),
                "list_date": str(row.get(col_list_date, "") or "").strip() if col_list_date else "",
            }

        # 将 code → vt_symbol（仅转债代码段）
        def _to_vt(code: str) -> str:
            prefix3 = code[:3] if len(code) >= 3 else ""
            if prefix3 in ("110", "111", "113", "118"):
                return f"{code}.SSE"
            if prefix3 in ("123", "127", "128"):
                return f"{code}.SZSE"
            return ""

        # universe = 快照中能映射到 vt_symbol 的代码集
        universe: dict[str, str] = {}  # vt_symbol -> code
        for code in terms_map:
            vt = _to_vt(code)
            if vt:
                universe[vt] = code

        # ---- 2. 逐标的加载日线并计算双低信号 ----
        all_parts: list[pl.DataFrame] = []
        skipped: list[str] = []

        for vt_symbol, code in universe.items():
            meta = terms_map.get(code, {})

            # 条款过滤（非行情过滤，可在加载行情前提前剔除）
            rating = meta.get("rating", "")
            issue_scale = meta.get("issue_scale", 0.0)
            list_date_str = meta.get("list_date", "")

            # 评级过滤
            if rating and not _rating_ge(rating, self._min_rating):
                continue
            # 规模过滤
            if issue_scale and issue_scale < self._min_issue_scale:
                continue
            # 上市天数过滤
            if list_date_str:
                try:
                    ld = date.fromisoformat(list_date_str[:10])
                    if (start - ld).days < self._min_list_days:
                        continue
                except (ValueError, TypeError):
                    pass  # 日期格式不合法时不过滤

            # 加载日线（无需预热，双低信号是截面信号，不依赖历史回看）
            df = self._lab.load_bar_frame(
                vt_symbol,
                self._interval,
                start,
                end,
                include_derived=True,
            )

            if df is None or df.is_empty():
                skipped.append(vt_symbol)
                logger.warning("cb_double_low：%s 无本地行情数据，已跳过", vt_symbol)
                continue

            # 裁剪到区间
            start_dt = pl.lit(start).cast(pl.Date)
            end_dt = pl.lit(end).cast(pl.Date)
            df = df.filter(
                (pl.col("datetime").cast(pl.Date) >= start_dt)
                & (pl.col("datetime").cast(pl.Date) <= end_dt)
            )

            if df.is_empty():
                skipped.append(vt_symbol)
                continue

            # 价格过滤（使用 close 列）
            df = df.filter(pl.col("close") <= self._max_price)
            if df.is_empty():
                continue

            # ---- 溢价率获取 ----
            premium_hist = self._terms_store.load_premium_history(vt_symbol)
            snapshot_premium = meta.get("snapshot_premium")

            # 构造每行的溢价率（优先历史，其次快照回退）
            part_rows: list[dict] = []
            for row in df.sort("datetime").to_dicts():
                row_date = row["datetime"]
                if isinstance(row_date, str):
                    row_date_str = row_date[:10]
                else:
                    row_date_str = str(row_date)[:10]

                premium_rate = self._lookup_premium(
                    premium_hist,
                    row_date_str,
                    snapshot_premium,
                    vt_symbol,
                )
                if premium_rate is None:
                    # 无溢价率可用，跳过该行
                    continue

                close_price = float(row["close"])
                signal_val = -(close_price + premium_rate * 100.0)

                part_rows.append(
                    {
                        "datetime": row["datetime"],
                        "vt_symbol": vt_symbol,
                        "signal": signal_val,
                    }
                )

            if part_rows:
                part = pl.DataFrame(part_rows).select(
                    [
                        pl.col("datetime").cast(pl.Datetime),
                        pl.col("vt_symbol").cast(pl.Utf8),
                        pl.col("signal").cast(pl.Float64),
                    ]
                )
                all_parts.append(part)

        # ---- 3. 合并 + 排序 ----
        if not all_parts:
            raise RuntimeError(
                "cb_double_low：区间内无任何双低信号输出。"
                "请先运行 POST /api/strategy/cb-terms/refresh 刷新转债条款，"
                "并在数据准备页下载转债日线行情。"
            )

        result = pl.concat(all_parts)
        result = result.sort(["datetime", "vt_symbol"])

        if on_progress is not None and callable(on_progress):
            on_progress(1.0, "cb_double_low 信号计算完成")  # type: ignore[operator]

        return result

    # -------------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_column(df: pl.DataFrame, candidates: list[str]) -> str | None:
        """从 candidates 中找第一个存在于 df.columns 的列名；均不存在则返回 None。"""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _safe_float(value: object) -> float:
        """安全转 float，失败时返回 0.0。"""
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    def _lookup_premium(
        self,
        premium_hist: pl.DataFrame | None,
        date_str: str,
        snapshot_premium: float | None,
        vt_symbol: str,
    ) -> float | None:
        """在历史溢价率中按日期查找；失败时回退到快照值。

        Returns:
            溢价率（小数形式，如 0.12 表示 12%）；无任何数据时返回 None。
        """
        if premium_hist is not None and not premium_hist.is_empty():
            # value_analysis 返回的字段：日期/收盘价/纯债价值/转股价值/纯债溢价率/转股溢价率
            date_col = self._detect_column(premium_hist, ["日期", "date"])
            premium_col = self._detect_column(premium_hist, ["转股溢价率", "premium_rate"])

            if date_col and premium_col:
                matched = premium_hist.filter(
                    pl.col(date_col).cast(pl.Utf8).str.slice(0, 10) == date_str
                )
                if not matched.is_empty():
                    raw_val = matched[premium_col][0]
                    try:
                        pct = float(raw_val)
                        # value_analysis 返回的溢价率通常是百分比形式（如 12.34）
                        # 此处统一规范为小数（/100）
                        return pct / 100.0
                    except (TypeError, ValueError):
                        pass

        # 回退到快照值
        if snapshot_premium is not None:
            logger.warning(
                "cb_double_low：%s 在 %s 无历史溢价率，回退使用快照值 %.2f%%",
                vt_symbol,
                date_str,
                snapshot_premium,
            )
            # 快照的 premium_rate 字段（bond_zh_cov）与历史路径一致，均为百分比形式
            # （如 3.5 表示 3.5%），统一 /100 规范为小数
            return snapshot_premium / 100.0

        return None


def _build_cb_double_low_source(params: dict) -> SignalProvider:
    """工厂函数：从 params 构造 _CbDoubleLowSource。

    Args:
        params: 参数字典。可通过 ``params["_lab"]`` 注入 AlphaLab 实例，
                ``params["_terms_store"]`` 注入 CBTermsStore（测试用途，
                下划线前缀表示内部参数，param_spec 不展示）。
    """
    top_k = int(params.get("top_k", 15))
    max_price = float(params.get("max_price", 130.0))
    min_rating = str(params.get("min_rating", "AA-"))
    min_issue_scale = float(params.get("min_issue_scale", 3.0))
    min_list_days = int(params.get("min_list_days", 5))
    interval = str(params.get("interval", "d"))

    # AlphaLab 依赖注入
    lab = params.get("_lab")
    if lab is None:
        from ..alpha.lab import AlphaLab  # noqa: PLC0415
        from ..config import ALPHA_LAB_PATH  # noqa: PLC0415

        lab = AlphaLab(ALPHA_LAB_PATH)

    # CBTermsStore 依赖注入
    terms_store = params.get("_terms_store")
    if terms_store is None:
        from .store import CBTermsStore  # noqa: PLC0415

        terms_store = CBTermsStore()

    return _CbDoubleLowSource(
        top_k=top_k,
        max_price=max_price,
        min_rating=min_rating,
        min_issue_scale=min_issue_scale,
        min_list_days=min_list_days,
        interval=interval,
        lab=lab,
        terms_store=terms_store,
    )


# 自注册到共享信号源注册表
register_signal_source(
    "cb_double_low",
    _build_cb_double_low_source,
    description="可转债双低轮动（低价+低溢价，T+0）",
    param_spec={
        "top_k": {
            "type": "int",
            "required": False,
            "label": "持仓数量",
            "description": "信息用参数（真实持仓数在策略层配置），用于描述组合目标规模",
            "default": 15,
        },
        "max_price": {
            "type": "float",
            "required": False,
            "label": "最高收盘价",
            "description": "超过此价格的转债不纳入（安全垫不足）",
            "default": 130.0,
        },
        "min_rating": {
            "type": "str",
            "required": False,
            "label": "最低信用评级",
            "description": "低于此评级的转债不纳入（评级序：AAA > AA+ > AA > AA- > ...）",
            "default": "AA-",
        },
        "min_issue_scale": {
            "type": "float",
            "required": False,
            "label": "最小发行规模（亿元）",
            "description": "低于此规模的转债不纳入（流动性保障）",
            "default": 3.0,
        },
        "min_list_days": {
            "type": "int",
            "required": False,
            "label": "最小上市天数",
            "description": "上市天数不足此值的新券不纳入（价格稳定期）",
            "default": 5,
        },
        "interval": {
            "type": "str",
            "required": False,
            "label": "行情周期",
            "description": "数据加载周期（默认日线 'd'）",
            "default": "d",
        },
    },
)
