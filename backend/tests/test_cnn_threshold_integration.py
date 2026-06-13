"""任务5 集成测试：阈值尺度校验三处接入（cnn-eval-honesty-fixes）。

三处接入共用 :func:`aitrade.cnn.thresholds.threshold_scale_check`（回测实盘一致红线）：

  5.1  回测 API 端点 ``POST /api/cnn/backtest/run``：读 checkpoint objective 后按其口径
       校验 buy/sell 阈值，不匹配 → 400（regression+0.6 当场拒，合法配置放行）。
  5.2  回测策略 ``CNNSignalStrategy``：on_init 从信号帧读 objective，违规设 _threshold_invalid
       并拒绝三种入场分支的开仓（防御纵深，缺列/合法时零影响）。
  5.3  实盘 ``SignalService``：拿到信号帧 objective 后同款校验，违规则该次决策标记拒绝、不买入。

Feature: cnn-eval-honesty-fixes
Requirements: 4.1–4.5, 6.2（Property 5 的"回测 API 与实盘 service 对同一输入同判定"落地）。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import polars as pl
import pytest

torch = pytest.importorskip("torch")  # API 400 测试需构造真实 checkpoint

from fastapi.testclient import TestClient  # noqa: E402

from aitrade.backtest.engine import BacktestingEngine  # noqa: E402
from aitrade.backtest.types import BarData, Direction  # noqa: E402
from aitrade.cnn.strategy import CNNSignalStrategy  # noqa: E402
from aitrade.cnn.thresholds import threshold_scale_check  # noqa: E402
from aitrade.live.decision import DecisionStore  # noqa: E402
from aitrade.live.decision_instant import DecisionInstant  # noqa: E402
from aitrade.live.notifier import LogNotifier  # noqa: E402
from aitrade.live.risk import RiskConfig, RiskManager  # noqa: E402
from aitrade.live.signal_service import PortfolioSnapshot, SignalService  # noqa: E402
from aitrade.main import create_app  # noqa: E402

SYMBOL = "TEST.SZSE"
START = datetime(2026, 1, 5)


# =============================================================================
# 共享工具
# =============================================================================


class _FakeLoader:
    """实现 BarDataLoader 协议的合成数据源，无外部依赖。"""

    def __init__(self, bars: list[BarData]) -> None:
        self._bars = bars

    def load_bar_data(self, vt_symbol: str, interval: str, start, end) -> list[BarData]:
        return list(self._bars)

    def load_contract_settings(self) -> dict:
        return {
            SYMBOL: {
                "long_rate": 0.0003,
                "short_rate": 0.0003,
                "stamp_duty": 0.0,
                "slippage": 0.0,
                "size": 1,
                "pricetick": 0.01,
            }
        }


def _build_bars(closes: list[float]) -> tuple[list[BarData], list[datetime]]:
    """构造满足撮合条件的合成日线（low 低于前收，保证限价单可成交）。

    Args:
        closes: 各根 bar 的收盘价列表。

    Returns:
        (bars 列表, datetime 列表) 元组。
    """
    days = [START + timedelta(days=i) for i in range(len(closes))]
    bars: list[BarData] = []
    for i, close in enumerate(closes):
        prev = closes[i - 1] if i > 0 else close
        bars.append(
            BarData(
                symbol="TEST",
                exchange="SZSE",
                datetime=days[i],
                interval="d",
                open_price=prev,
                high_price=max(prev, close) + 1.0,
                low_price=min(prev, close) - 1.0,
                close_price=close,
                volume=1_000_000,
            )
        )
    return bars, days


def _signal_df(days: list[datetime], signals: list[float], objective: str | None) -> pl.DataFrame:
    """构造三列信号帧（可选追加 objective 常量列）。

    Args:
        days: 各行 datetime。
        signals: 各行 signal 值。
        objective: 末列 objective 常量；None 表示构造不含该列的 legacy 帧。

    Returns:
        含 [datetime, vt_symbol, signal(, objective)] 的 DataFrame。
    """
    n = len(days)
    data: dict[str, Any] = {
        "datetime": days,
        "vt_symbol": [SYMBOL] * n,
        "signal": signals,
    }
    if objective is not None:
        data["objective"] = [objective] * n
    return pl.DataFrame(data)


def _run_engine(
    bars: list[BarData],
    days: list[datetime],
    signal_df: pl.DataFrame,
    setting: dict,
) -> BacktestingEngine:
    """组装并运行一次回测，返回引擎实例供断言。

    Args:
        bars: 合成 BarData 列表。
        days: 对应 datetime 列表。
        signal_df: 信号 DataFrame。
        setting: CNNSignalStrategy 参数字典。

    Returns:
        完成回测后的 BacktestingEngine 实例。
    """
    engine = BacktestingEngine(data_loader=_FakeLoader(bars))
    engine.set_parameters(
        vt_symbols=[SYMBOL],
        interval="d",
        start=days[0],
        end=days[-1] + timedelta(days=1),
        capital=1_000_000,
    )
    engine.add_strategy(CNNSignalStrategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    return engine


# =============================================================================
# 5.1  回测 API 400（regression+0.6 当场拒；合法配置放行）
# =============================================================================


def _make_checkpoint(model_dir, name: str, objective: str) -> None:
    """落盘一个仅含 train_config.objective 的最小 checkpoint（无需真实权重）。

    阈值校验只读 ``checkpoint["train_config"]["objective"]``，故无需真实模型权重；
    torch.load(weights_only=False) 能读回任意 dict。

    Args:
        model_dir: 目标目录（隔离的 CNN_MODEL_DIR）。
        name: 模型名（不含 .pt）。
        objective: 写入 train_config 的 objective 字符串。
    """
    save_data = {"train_config": {"objective": objective, "target_symbol": SYMBOL}}
    torch.save(save_data, str(model_dir / f"{name}.pt"))


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    """隔离 CNN_MODEL_DIR 的 TestClient；run_async 置为 no-op 避免真实回测线程。"""
    from unittest.mock import patch

    import aitrade.cnn.storage as cnn_storage

    monkeypatch.setattr(cnn_storage, "CNN_MODEL_DIR", tmp_path)

    app = create_app()
    with patch("aitrade.api.cnn.task_manager.run_async", return_value=None):
        with TestClient(app) as c:
            c._model_dir = tmp_path  # type: ignore[attr-defined]
            yield c


class TestBacktestApiThresholdCheck:
    """5.1：回测端点按 checkpoint objective 校验阈值尺度，违规 400、合法放行。"""

    BASE = {
        "name": "bt_thresh",
        "start": "2024-01-01",
        "end": "2025-01-01",
    }

    def test_regression_buy_06_returns_400(self, api_client: TestClient) -> None:
        """regression 模型 + buy_threshold=0.6 → 400，detail 含 regression/收益口径提示。"""
        _make_checkpoint(api_client._model_dir, "reg_model", "regression")
        payload = {**self.BASE, "model": "reg_model", "buy_threshold": 0.6, "sell_threshold": -0.005}
        resp = api_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code == 400, f"期望 400，实际 {resp.status_code} {resp.text}"
        assert "regression" in resp.text
        assert "收益" in resp.text or "概率" in resp.text

    def test_regression_buy_0005_passes(self, api_client: TestClient) -> None:
        """regression 模型 + buy_threshold=0.005（合法收益阈值）→ 放行到任务创建（非 400/422）。"""
        _make_checkpoint(api_client._model_dir, "reg_model", "regression")
        payload = {**self.BASE, "model": "reg_model", "buy_threshold": 0.005, "sell_threshold": -0.005}
        resp = api_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code not in (400, 422), (
            f"合法 regression 阈值不应被拦，实际 {resp.status_code} {resp.text}"
        )
        assert "task_id" in resp.json()

    def test_classification_buy_06_passes(self, api_client: TestClient) -> None:
        """classification 模型 + buy=0.6/sell=0.4（合法概率阈值）→ 放行到任务创建。"""
        _make_checkpoint(api_client._model_dir, "cls_model", "classification")
        payload = {**self.BASE, "model": "cls_model", "buy_threshold": 0.6, "sell_threshold": 0.4}
        resp = api_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code not in (400, 422), (
            f"合法 classification 阈值不应被拦，实际 {resp.status_code} {resp.text}"
        )
        assert "task_id" in resp.json()

    def test_classification_negative_buy_returns_400(self, api_client: TestClient) -> None:
        """classification 模型 + buy=-0.5（越出 [0,1] 但落在端点 (-1,1) 范围内）→ 400。

        端点自身 (-1,1) 范围校验放行 -0.5，但 threshold_scale_check 检出概率口径越界。
        """
        _make_checkpoint(api_client._model_dir, "cls_model", "classification")
        payload = {**self.BASE, "model": "cls_model", "buy_threshold": -0.5, "sell_threshold": -0.6}
        resp = api_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code == 400, f"期望 400，实际 {resp.status_code} {resp.text}"
        assert "classification" in resp.text

    def test_path_class_buy_06_passes(self, api_client: TestClient) -> None:
        """path_class 模型 + buy=0.6（合法概率阈值）→ 放行。"""
        _make_checkpoint(api_client._model_dir, "pc_model", "path_class")
        payload = {**self.BASE, "model": "pc_model", "buy_threshold": 0.6, "sell_threshold": 0.4, "veto_threshold": 0.7}
        resp = api_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code not in (400, 422), (
            f"合法 path_class 阈值不应被拦，实际 {resp.status_code} {resp.text}"
        )

    def test_missing_model_not_blocked_by_threshold_check(self, api_client: TestClient) -> None:
        """模型不存在 → 阈值校验跳过（不 400/422），交由任务内抛"模型不存在"（保持既有行为）。"""
        payload = {**self.BASE, "model": "no_such_model", "buy_threshold": 0.6, "sell_threshold": 0.4}
        resp = api_client.post("/api/cnn/backtest/run", json=payload)
        assert resp.status_code not in (400, 422), (
            f"缺失模型不应被阈值校验提前拦截，实际 {resp.status_code} {resp.text}"
        )


# =============================================================================
# 5.2  策略防御纵深（直接构造非法组合 → _threshold_invalid 且无买入）
# =============================================================================


class TestStrategyDefenseInDepth:
    """5.2：CNNSignalStrategy 从信号帧 objective 自检阈值，违规则拒绝开仓。"""

    SETTING = {
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "exit_mode": "threshold",
        "price_add": 0.0,
    }

    def test_regression_objective_with_prob_threshold_no_buy(self) -> None:
        """信号帧 objective=regression + buy=0.6（非法）→ _threshold_invalid=True 且无买入成交。"""
        closes = [10.0] * 8
        bars, days = _build_bars(closes)
        # signal 全 0.9 > buy_threshold；若不拦会买入。objective=regression+0.6 应拦。
        signal_df = _signal_df(days, [0.9] * len(days), objective="regression")
        engine = _run_engine(bars, days, signal_df, dict(self.SETTING))

        assert engine.strategy._threshold_invalid is True, "regression+0.6 应标记阈值非法"
        buy_trades = [t for t in engine.get_all_trades() if t.direction == Direction.LONG]
        assert len(buy_trades) == 0, "阈值非法时不应有任何买入成交"

    def test_classification_objective_legal_buys_normally(self) -> None:
        """信号帧 objective=classification + buy=0.6（合法）→ 正常买卖（_threshold_invalid=False）。"""
        closes = [10.0] * 8
        bars, days = _build_bars(closes)
        # 前段高概率买入、后段低概率卖出
        signals = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9]
        signal_df = _signal_df(days, signals, objective="classification")
        engine = _run_engine(bars, days, signal_df, dict(self.SETTING))

        assert engine.strategy._threshold_invalid is False, "合法配置不应标记非法"
        buy_trades = [t for t in engine.get_all_trades() if t.direction == Direction.LONG]
        assert len(buy_trades) >= 1, "合法 classification 应正常触发买入"

    def test_legacy_frame_no_objective_behaves_as_before(self) -> None:
        """legacy 帧（无 objective 列）+ 任意阈值 → 不拦（_threshold_invalid=False），行为与改造前一致。"""
        closes = [10.0] * 8
        bars, days = _build_bars(closes)
        # 故意用 buy=0.6（若按 regression 解读会被拦），但帧无 objective 列 → 跳过校验
        signal_df = _signal_df(days, [0.9] * len(days), objective=None)
        engine = _run_engine(bars, days, signal_df, dict(self.SETTING))

        assert engine.strategy._threshold_invalid is False, "legacy 帧应跳过阈值校验"
        buy_trades = [t for t in engine.get_all_trades() if t.direction == Direction.LONG]
        assert len(buy_trades) >= 1, "legacy 帧应正常触发买入（向后兼容）"

    @pytest.mark.parametrize("exit_mode", ["threshold", "fixed_hold", "oco"])
    def test_regression_blocked_in_all_exit_modes(self, exit_mode: str) -> None:
        """三种出场模式下，regression+0.6 非法组合均拒绝开仓（防御纵深覆盖全部入场分支）。"""
        closes = [10.0] * 8
        bars, days = _build_bars(closes)
        signal_df = _signal_df(days, [0.9] * len(days), objective="regression")
        setting = {
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "exit_mode": exit_mode,
            "hold_days": 2,
            "take_profit": 0.05,
            "stop_loss": 0.03,
            "price_add": 0.0,
        }
        engine = _run_engine(bars, days, signal_df, setting)
        assert engine.strategy._threshold_invalid is True
        buy_trades = [t for t in engine.get_all_trades() if t.direction == Direction.LONG]
        assert len(buy_trades) == 0, f"exit_mode={exit_mode} 阈值非法时不应买入"


# =============================================================================
# 5.3  实盘 SignalService 拒绝（regression+0.6 → 决策拒绝；合法 → 正常）
# =============================================================================


def _run_service(svc: SignalService, **kw):
    """以 1d 收盘后 as_of 触发一次决策（Decision_Bar=当日）。"""
    d = date(2026, 6, 9)
    return svc.run_for_instant(
        DecisionInstant(datetime.combine(d, time(15, 5)), "1d"),
        decision_bar_dt=datetime.combine(d, time(15, 0)),
        **kw,
    )


def _make_service(tmp_path, buy_threshold: float) -> SignalService:
    """构造一个可单测的 SignalService（LogNotifier + 宽松风控）。"""
    notifier = LogNotifier()
    store = DecisionStore(tmp_path)
    risk = RiskManager(RiskConfig(max_total_position_ratio=0.95, max_single_position_ratio=1.0))
    return SignalService("thresh_test", buy_threshold=buy_threshold, risk=risk,
                         store=store, notifier=notifier, model_version="v1")


class TestSignalServiceThresholdCheck:
    """5.3：SignalService 按 objective 自检阈值，违规则该次决策拒绝（hold，不买入）。"""

    def test_regression_objective_buy_06_rejected(self, tmp_path) -> None:
        """objective=regression + buy_threshold=0.6 → 决策拒绝（action=hold，reason 含拒绝说明）。"""
        svc = _make_service(tmp_path, buy_threshold=0.6)
        pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
        d = _run_service(svc, signal=0.9, price=10.0, portfolio=pf,
                         vt_symbol=SYMBOL, objective="regression")
        assert d.action == "hold", f"非法阈值应拒绝买入，实际 action={d.action}"
        assert "拒绝" in d.reason and "objective" in d.reason

    def test_classification_objective_buy_06_buys(self, tmp_path) -> None:
        """objective=classification + buy=0.6（合法）→ 正常买入。"""
        svc = _make_service(tmp_path, buy_threshold=0.6)
        pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
        d = _run_service(svc, signal=0.9, price=10.0, portfolio=pf,
                         vt_symbol=SYMBOL, objective="classification")
        assert d.action == "buy", f"合法 classification 应买入，实际 {d.action}: {d.reason}"

    def test_legacy_no_objective_buys(self, tmp_path) -> None:
        """objective=None（legacy 信号帧）+ buy=0.6 → 跳过校验，正常买入（向后兼容）。"""
        svc = _make_service(tmp_path, buy_threshold=0.6)
        pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
        d = _run_service(svc, signal=0.9, price=10.0, portfolio=pf,
                         vt_symbol=SYMBOL, objective=None)
        assert d.action == "buy", f"legacy 帧应正常买入，实际 {d.action}: {d.reason}"

    def test_regression_objective_legal_buys(self, tmp_path) -> None:
        """objective=regression + buy=0.005（合法收益阈值）→ 正常买入。"""
        svc = _make_service(tmp_path, buy_threshold=0.005)
        pf = PortfolioSnapshot(portfolio_value=100000, current_position=0)
        d = _run_service(svc, signal=0.02, price=10.0, portfolio=pf,
                         vt_symbol=SYMBOL, objective="regression")
        assert d.action == "buy", f"合法 regression 阈值应买入，实际 {d.action}: {d.reason}"

    def test_rejection_does_not_block_exit(self, tmp_path) -> None:
        """阈值非法但已有持仓 + should_exit → 仍正常出场（拒绝仅拦买入，出场零影响）。"""
        svc = _make_service(tmp_path, buy_threshold=0.6)
        pf = PortfolioSnapshot(portfolio_value=100000, current_position=900)
        d = _run_service(svc, signal=0.1, price=10.0, portfolio=pf,
                         vt_symbol=SYMBOL, should_exit=True, objective="regression")
        assert d.action == "sell" and d.volume == 900, "阈值校验不应阻断出场"


# =============================================================================
# 一致性：回测 API 与实盘 service 对同一 (objective, buy, sell) 同判定（Property 5 子句）
# =============================================================================


class TestBacktestLiveConsistency:
    """回测端点与实盘 service 经同一 threshold_scale_check，对同一输入给出一致判定。"""

    @pytest.mark.parametrize(
        "objective,buy,sell,expect_violation",
        [
            ("regression", 0.6, -0.005, True),
            ("regression", 0.005, -0.005, False),
            ("classification", 0.6, 0.4, False),
            ("classification", -0.5, -0.6, True),
            ("path_class", 0.6, 0.4, False),
            (None, 0.6, 0.4, False),
        ],
    )
    def test_same_input_same_verdict(
        self, objective, buy, sell, expect_violation
    ) -> None:
        """同一 (objective, buy, sell) 在两侧共用的纯函数判定一致。

        回测端点对 (buy, sell) 调 threshold_scale_check；实盘 service 对 (buy,) 调。
        此处校验"是否违规"在两侧的判定来源一致——同一函数同一输入同一输出。
        """
        api_reasons = threshold_scale_check(objective, buy, sell)
        live_reasons = threshold_scale_check(objective, buy)
        # buy 方向的违规两侧必然一致（sell 仅影响 API 侧的额外 sell 违规）
        buy_violation_api = any("buy_threshold" in r for r in api_reasons)
        buy_violation_live = any("buy_threshold" in r for r in live_reasons)
        assert buy_violation_api == buy_violation_live, (
            f"buy 方向判定不一致: api={api_reasons} live={live_reasons}"
        )
        assert bool(api_reasons) == expect_violation or bool(live_reasons) == expect_violation


# =============================================================================
# 既有合法配置全部放行（5.4 回归守护）
# =============================================================================


@pytest.mark.parametrize(
    "objective,buy,sell",
    [
        ("classification", 0.6, 0.4),
        ("regression", 0.005, -0.005),
        ("path_class", 0.6, 0.4),
        (None, 0.6, 0.4),  # legacy
    ],
)
def test_existing_legal_configs_all_pass(objective, buy, sell) -> None:
    """既有合法配置（classification 0.6/0.4、regression 0.005/-0.005、path_class 0.6、legacy）零误拦。"""
    assert threshold_scale_check(objective, buy, sell) == []
