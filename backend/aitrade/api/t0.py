"""半仓做 T 回测 API（同步）。

把 backtest/t0 的 T0BacktestRunner 暴露为一个同步端点：前端传标的/区间/档位/成交假设网格，
后端跑引擎@1m 出"成交敏感性区间 + 逐年/逐月超额 + 命中分布"并直接返回。做 T 单次扫描通常
数秒内完成，故采用同步返回（区间过长时前端会有 loading 态）。
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..backtest.t0.policy_spec import TickPolicyCfg, compile_tick_policy
from ..backtest.t0.profiler import T0Profiler, load_daily_from_1m, profile_by_gap
from ..backtest.t0.runner import T0BacktestRunner
from ..backtest.t0.signals import LabSignalProvider, SignalProvider
from ..backtest.t0.tick_policy import FixedTick, TickPolicy
from ..backtest.types import FillPolicy
from ..config import ALPHA_LAB_PATH

router = APIRouter(prefix="/api/t0", tags=["t0"])


def _compile_tick_policies(cfgs: list) -> tuple[list[tuple[str, TickPolicy]], list[str]]:
    """把声明式档位策略列表编译成 ``[(label, TickPolicy)]`` 并汇总引用的信号名。

    Args:
        cfgs: 已被 Pydantic 解析的 ``TickPolicyCfg`` 列表（kind 已是白名单）。

    Returns:
        ``(tick_policies, signal_names)``：前者交 runner 扫网格，后者（升序去重）用于按需加载信号源。

    Raises:
        HTTPException: 策略 ``label`` 不唯一（400）。
    """
    compiled = [compile_tick_policy(c) for c in cfgs]
    labels = [lbl for lbl, _, _ in compiled]
    if len(set(labels)) != len(labels):
        raise HTTPException(status_code=400, detail="档位策略 label 必须唯一")
    policies = [(lbl, pol) for lbl, pol, _ in compiled]
    names = sorted({n for _, _, ns in compiled for n in ns})
    return policies, names


def _load_signal_provider(symbol: str, names: list[str]) -> SignalProvider | None:
    """按信号名加载持久化模型信号，建该标的的 point-in-time 提供器；无可用信号返回 None。

    仅加载**已登记**（``list_all_signals()`` 白名单内）且用到的信号，并预过滤到该标的以省内存。
    白名单杜绝了客户端用伪造 ``signal_name``（如 ``../../x``、绝对路径）穿越出信号目录读任意
    ``.parquet`` 的风险：不在白名单的名字直接跳过（规则恒不命中），不会触达 ``load_signal``。
    Alpha 模块不可用或信号缺失/脏帧时优雅降级（跳过该信号），不报致命错误。

    Args:
        symbol: 标的 vt_symbol。
        names: 规则引用的信号名集合（可能含客户端伪造名）。

    Returns:
        ``LabSignalProvider`` 或 None（无引用/不可用）。
    """
    if not names:
        return None
    try:
        import polars as pl

        from ..alpha import AlphaLab
    except ImportError:
        return None
    lab = AlphaLab(ALPHA_LAB_PATH)
    available = set(lab.list_all_signals())   # 白名单：只认已登记信号，杜绝路径穿越
    frames = {}
    for name in names:
        if name not in available:
            continue                          # 伪造/未知名直接跳过，绝不触达文件系统
        try:
            f = lab.load_signal(name)
        except Exception:  # noqa: BLE001  脏信号帧不致命，跳过该信号
            continue
        if f is not None and "vt_symbol" in f.columns:
            sub = f.filter(pl.col("vt_symbol") == symbol)
            if sub.height > 0:
                frames[name] = sub
    return LabSignalProvider.from_frames(frames) if frames else None


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
        max_length=20,
        description="成交假设网格（默认含理想/穿越1分/部分成交）")
    tick_policies: list[TickPolicyCfg] | None = Field(
        default=None, max_length=20,
        description="多档位策略声明（fixed/vol_scaled/trend_tilt/conditional）；"
                    "省略则回退为单 FixedTick(sell_tick, buy_tick)（向后兼容）")


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

        if req.tick_policies:
            tick_policies, names = _compile_tick_policies(req.tick_policies)
            # 防滥用：策略 × 成交假设 的网格规模封顶（每格都是一次完整 1m 回测）
            if len(tick_policies) * len(req.fill_grid) > 120:
                raise HTTPException(status_code=400, detail="策略 × 成交假设 组合过多（上限 120 格）")
            signal_provider = _load_signal_provider(req.symbol, names)
        else:  # 向后兼容：无 tick_policies 时沿用单 FixedTick
            tick_policies = [(f"fixed_{int(req.sell_tick*100)}/{int(req.buy_tick*100)}fen",
                              FixedTick(req.sell_tick, req.buy_tick))]
            signal_provider = None

        runner = T0BacktestRunner()
        report = runner.run(
            req.symbol, req.start, req.end, daily,
            tick_policies=tick_policies,
            fill_grid=[FillPolicy(f.penetration, f.ratio) for f in req.fill_grid],
            signal_provider=signal_provider,
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


class T0ProfileSegmentedRequest(T0ProfileRequest):
    """分场景（高/低/平开）做 T 画像请求：在全窗画像请求上加跳空阈值。"""

    gap_thresh: float = Field(default=0.003, gt=0, description="高/低开判定阈值（小数，如 0.003=0.3%）")


@router.post("/profile_segmented", summary="按高/低/平开分场景的做T画像（供条件策略逐规则标定）")
async def run_t0_profile_segmented(req: T0ProfileSegmentedRequest) -> dict:
    """把标定窗按跳空切高/低/平开三组，各自做 T 画像，供条件(跳空)策略逐规则建议档位。

    每组各自给建议 (sell, buy) 档与样本天数 ``n_days``；调用方据 ``n_days`` 对小样本场景告警。
    无前视（gap 用昨收、画像只读标定窗），建议仍为理想撮合上限、须经回测 FillPolicy 网格验证。

    Args:
        req: 分场景画像请求（标的/标定窗/档位上限/成本 + gap_thresh）。

    Returns:
        ``{"symbol", "thresh", "segments": [{regime,label,n_days,profile}, ...]}``，
        ``segments`` 固定顺序 高开/低开/平开。

    Raises:
        HTTPException: 数据缺失（404）或区间非法/有效日不足/统计失败（400）。
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
        if daily.height < 6:   # 分场景需更多样本（首日剔除 + 三组各需若干天）
            raise HTTPException(status_code=400, detail="标定窗内有效交易日不足（分场景至少 6 日）")
        segs = profile_by_gap(
            req.symbol, daily, thresh=req.gap_thresh, x_grid_fen=range(1, req.x_max_fen + 1),
            commission_rate=req.commission_rate, stamp_duty=req.stamp_duty)
        return {"symbol": req.symbol, "thresh": req.gap_thresh, "segments": [s.to_dict() for s in segs]}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"分场景画像失败：{exc}") from exc


@router.get("/signals", summary="列出可用于条件规则的信号名")
async def list_t0_signals() -> dict:
    """列出可作为条件规则 ``lhs=signal`` 来源的持久化模型信号名（升序）。

    供前端条件规则编辑器填充信号下拉。Alpha 模块不可用时返回空列表（不报错）。

    Returns:
        ``{"names": [...]}``：信号名升序列表。
    """
    try:
        from ..alpha import AlphaLab
    except ImportError:
        return {"names": []}
    lab = AlphaLab(ALPHA_LAB_PATH)
    return {"names": sorted(lab.list_all_signals())}
