"""Tier-2 WF 报告折级保留第 0 种子净值/成交曲线（cnn-screening-tier2-detail 第三波）。

验证 run_walk_forward_evaluate 在 fold dict 增加 candidate_equity_curve / candidate_trades：
- 多种子时只保留第 0 种子（代表）的曲线/成交，与 candidate_statistics 同源（体积治理）；
- 回测无曲线（早退/异常路径返回空）时折级曲线为 []，不抛错。

桩化重型训练/回测，使 WF 评估秒级跑通；全局生产 store 重定向 tmp 防污染。
"""

from __future__ import annotations

from datetime import date

import pytest

from aitrade.cnn import governance as gov
from aitrade.cnn.governance import CNNGovernanceStore, run_walk_forward_evaluate
from aitrade.models.governance import CNNWalkForwardRequest


def _make_request(n_seeds: int) -> CNNWalkForwardRequest:
    """构造能生成 ≥1 个 walk-forward 窗口的最小请求（n_seeds 可调）。"""
    return CNNWalkForwardRequest(
        name="curve_test",
        target_symbol="000001.SZSE",
        start=date(2023, 1, 1),
        end=date(2025, 6, 1),
        train_days=720,
        test_days=90,
        n_seeds=n_seeds,
    )


@pytest.fixture
def _redirect_global_store(tmp_path, monkeypatch):
    """把全局生产 store 重定向到 tmp，避免污染真实治理目录。返回隔离 screening store。"""
    global_store = CNNGovernanceStore(tmp_path / "global_gov")
    monkeypatch.setattr(gov, "store", global_store)
    monkeypatch.setattr(gov, "_GLOBAL_STORE", global_store)
    return CNNGovernanceStore(tmp_path / "screening_gov")


def test_fold_keeps_seed0_equity_curve_and_trades(_redirect_global_store, monkeypatch):
    """多种子时，fold 的 candidate_equity_curve/candidate_trades 取第 0 种子（代表）。"""

    def fake_train(req, *, model_name, start, end, seed_index=0, on_progress=None):
        return {"name": model_name}

    def fake_backtest(*, model_name, name, start, end, capital, params):
        # 第 0 种子与其余种子返回不同曲线/成交，验证保留的是第 0 种子。
        is_seed0 = "_s0_" in name
        balance = 10_000.0 if is_seed0 else 20_000.0
        price = 1.0 if is_seed0 else 2.0
        return {
            "statistics": {"total_return": 5.0, "sharpe_ratio": 1.2, "win_rate": 0.6},
            "equity_curve": [
                {"date": "2025-01-02", "balance": balance, "drawdown": 0.0, "ddpercent": 0.0, "net_pnl": 0.0}
            ],
            "trades": [
                {
                    "datetime": "2025-01-02T09:30:00",
                    "vt_symbol": "000001.SZSE",
                    "direction": "long",
                    "offset": "open",
                    "price": price,
                    "volume": 100.0,
                }
            ],
        }

    monkeypatch.setattr(gov, "_train_governance_model", fake_train)
    monkeypatch.setattr(gov, "_backtest_model", fake_backtest)

    report = run_walk_forward_evaluate(_make_request(n_seeds=2), store=_redirect_global_store)
    fold = report["folds"][0]

    # 两键存在，且取的是第 0 种子（balance=10000 / price=1.0），不是第 1 种子（20000 / 2.0）。
    assert fold["candidate_equity_curve"][0]["balance"] == 10_000.0
    assert fold["candidate_trades"][0]["price"] == 1.0
    # 与 candidate_statistics 同源（第 0 种子代表）。
    assert fold["candidate_statistics"]["total_return"] == 5.0


def test_fold_curves_empty_when_backtest_returns_none(_redirect_global_store, monkeypatch):
    """回测不含 equity_curve/trades（早退路径）时，折级曲线安全回落为 []。"""

    def fake_train(req, *, model_name, start, end, seed_index=0, on_progress=None):
        return {"name": model_name}

    def fake_backtest(*, model_name, name, start, end, capital, params):
        # 只给 statistics，模拟早退/异常路径（无 equity_curve/trades 键）。
        return {"statistics": {"total_return": 1.0, "sharpe_ratio": 0.5, "win_rate": 0.5}}

    monkeypatch.setattr(gov, "_train_governance_model", fake_train)
    monkeypatch.setattr(gov, "_backtest_model", fake_backtest)

    report = run_walk_forward_evaluate(_make_request(n_seeds=1), store=_redirect_global_store)
    fold = report["folds"][0]

    assert fold["candidate_equity_curve"] == []
    assert fold["candidate_trades"] == []
