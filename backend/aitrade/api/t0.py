"""半仓做 T 回测 API（同步）。

把 backtest/t0 的 T0BacktestRunner 暴露为一个同步端点：前端传标的/区间/档位/成交假设网格，
后端跑引擎@1m 出"成交敏感性区间 + 逐年/逐月超额 + 命中分布"并直接返回。做 T 单次扫描通常
数秒内完成，故采用同步返回（区间过长时前端会有 loading 态）。
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..backtest.t0.profiler import T0Profiler, load_daily_from_1m
from ..backtest.t0.runner import T0BacktestRunner
from ..backtest.t0.tick_policy import FixedTick
from ..backtest.types import FillPolicy
from ..config import ALPHA_LAB_PATH

router = APIRouter(prefix="/api/t0", tags=["t0"])


class FillCfg(BaseModel):
    """单个成交假设：穿越阈值（元）+ 单根成交比例。"""

    penetration: float = Field(default=0.0, ge=0.0, description="穿越阈值 ε（元）")
    ratio: float = Field(default=1.0, gt=0.0, le=1.0, description="单根触价成交比例")


class T0BacktestRequest(BaseModel):
    """半仓做 T 回测请求。"""

    symbol: str = Field(default="000415.SZSE", description="标的 vt_symbol")
    start: date = Field(description="评估窗起（含）")
    end: date = Field(description="评估窗止（含）")
    sell_tick: float = Field(default=0.02, gt=0, description="卖单挂高价差（元）")
    buy_tick: float = Field(default=0.02, gt=0, description="买单挂低价差（元）")
    swing_frac: float = Field(default=1.0, gt=0, le=1.0, description="做T摆动占半仓比例")
    base_weight: float = Field(default=0.5, gt=0, lt=1, description="半仓锚权重")
    capital: int = Field(default=1_000_000, gt=0, description="初始资金")
    commission_rate: float = Field(default=0.0003, ge=0, description="单边佣金率")
    stamp_duty: float = Field(default=0.0005, ge=0, description="卖出印花税率")
    fill_grid: list[FillCfg] = Field(
        default_factory=lambda: [FillCfg(penetration=0.0, ratio=1.0),
                                 FillCfg(penetration=0.01, ratio=1.0),
                                 FillCfg(penetration=0.0, ratio=0.5)],
        description="成交假设网格（默认含理想/穿越1分/部分成交）")


@router.post("/backtest", summary="运行半仓做T回测，返回成交敏感性区间")
async def run_t0_backtest(req: T0BacktestRequest) -> dict:
    """跑半仓做 T 回测并返回区间报告。

    Args:
        req: 回测请求（标的/区间/档位/摆动/成交网格）。

    Returns:
        T0Report.to_dict()：含 symbol、eval_window、fill_sensitivity（区间）、results（逐年/逐月超额、命中分布）。

    Raises:
        HTTPException: 标的 1m 数据缺失（404）、区间非法或回测失败（400）。
    """
    if req.start >= req.end:
        raise HTTPException(status_code=400, detail="起始日期须早于结束日期")

    parquet = ALPHA_LAB_PATH / "bars" / "1m" / f"{req.symbol}.parquet"
    if not parquet.exists():
        raise HTTPException(status_code=404, detail=f"未找到 {req.symbol} 的 1 分钟数据：{parquet}")

    try:
        daily = load_daily_from_1m(str(parquet), req.start.year, req.end.year)
        import polars as pl
        daily = daily.filter((pl.col("d") >= req.start) & (pl.col("d") <= req.end))
        if daily.height < 2:
            raise HTTPException(status_code=400, detail="评估窗内有效交易日不足")

        runner = T0BacktestRunner()
        report = runner.run(
            req.symbol, req.start, req.end, daily,
            tick_policies=[(f"fixed_{int(req.sell_tick*100)}/{int(req.buy_tick*100)}fen",
                            FixedTick(req.sell_tick, req.buy_tick))],
            fill_grid=[FillPolicy(f.penetration, f.ratio) for f in req.fill_grid],
            capital=req.capital, commission_rate=req.commission_rate,
            stamp_duty=req.stamp_duty, swing_frac=req.swing_frac, base_weight=req.base_weight,
        )
        return report.to_dict()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"做T回测失败：{exc}") from exc


class T0ProfileRequest(BaseModel):
    """标的做 T 画像请求。"""

    symbol: str = Field(default="000415.SZSE", description="标的 vt_symbol")
    start: date = Field(description="标定窗起（含）")
    end: date = Field(description="标定窗止（含）")
    x_max_fen: int = Field(default=15, ge=2, le=50, description="档位网格上限（分）")
    commission_rate: float = Field(default=0.0003, ge=0, description="单边佣金率")
    stamp_duty: float = Field(default=0.0005, ge=0, description="卖出印花税率")


@router.post("/profile", summary="标的做T画像：偏离-回归边际曲线 + 建议档位")
async def run_t0_profile(req: T0ProfileRequest) -> dict:
    """统计某标的"按偏离开盘 x 分挂单、单腿做 T 的每笔边际收益（理想撮合）"曲线。

    按偏离档位网格、分买/卖腿算成交率与净于成本的条件回归边际收益，并给出建议档位（天然非对称）。

    Args:
        req: 画像请求（标的/标定窗/档位网格上限/成本）。

    Returns:
        T0Profile.to_dict()：rows（逐档位逐腿曲线）、suggested_sell_tick/buy_tick、note。

    Raises:
        HTTPException: 数据缺失（404）或区间非法/统计失败（400）。
    """
    if req.start >= req.end:
        raise HTTPException(status_code=400, detail="起始日期须早于结束日期")
    parquet = ALPHA_LAB_PATH / "bars" / "1m" / f"{req.symbol}.parquet"
    if not parquet.exists():
        raise HTTPException(status_code=404, detail=f"未找到 {req.symbol} 的 1 分钟数据：{parquet}")
    try:
        import polars as pl
        daily = load_daily_from_1m(str(parquet), req.start.year, req.end.year)
        daily = daily.filter((pl.col("d") >= req.start) & (pl.col("d") <= req.end))
        if daily.height < 5:
            raise HTTPException(status_code=400, detail="标定窗内有效交易日不足（至少 5 日）")
        profile = T0Profiler().profile(
            req.symbol, daily, x_grid_fen=range(1, req.x_max_fen + 1),
            commission_rate=req.commission_rate, stamp_duty=req.stamp_duty)
        return profile.to_dict()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"做T画像失败：{exc}") from exc
