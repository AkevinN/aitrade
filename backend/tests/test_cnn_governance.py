from datetime import datetime

from aitrade.cnn.governance import (
    CNNGovernanceStore,
    _aggregate_periods,
    _replay_conclusion,
)
from aitrade.models.governance import CNNGovernanceConfig


def test_governance_store_roundtrip(tmp_path):
    store = CNNGovernanceStore(tmp_path)

    config = store.save_config(CNNGovernanceConfig(evaluation_period_days=14))
    assert config["evaluation_period_days"] == 14
    assert store.get_config()["evaluation_period_days"] == 14

    production = store.save_production({
        "model_name": "cnn_prod",
        "model_version": "v1",
        "target_symbol": "000001.SZSE",
        "input_interval": "d",
        "objective": "classification",
        "promoted_at": datetime.now(),
        "promoted_by": "test",
        "report_id": "wf_1",
        "previous_model_name": "cnn_old",
        "previous_model_version": "v0",
    })
    assert production["model_name"] == "cnn_prod"
    assert store.get_production()["previous_model_name"] == "cnn_old"

    candidate = store.save_candidate({
        "candidate_id": "cand_1",
        "model_name": "cnn_candidate",
        "status": "passed",
        "created_at": "2026-06-08T00:00:00",
    })
    assert store.get_candidate("cand_1") == candidate
    assert len(store.list_candidates()) == 1

    report = store.save_report({"report_id": "wf_1", "summary": {"passed": True}})
    assert store.get_report("wf_1") == report

    replay = store.save_replay_report({"replay_id": "replay_1", "conclusion": {"verdict": "ok"}})
    assert store.get_replay_report("replay_1") == replay
    assert len(store.list_replay_reports()) == 1

    store.append_history("unit_event", {"ok": True})
    assert store.history()[-1]["event_type"] == "unit_event"


def test_replay_aggregate_and_conclusion():
    periods = [
        {"statistics": {"total_return": 3.0, "sharpe_ratio": 1.0, "max_ddpercent": -2.0, "total_trade_count": 2, "total_turnover": 1000, "total_commission": 1}},
        {"statistics": {"total_return": -1.0, "sharpe_ratio": 0.5, "max_ddpercent": -4.0, "total_trade_count": 0, "total_turnover": 0, "total_commission": 0}},
    ]
    stats = _aggregate_periods(periods, 1_000_000)
    assert stats["total_return"] == 2.0
    assert stats["max_ddpercent"] == 4.0
    assert stats["total_trade_count"] == 2
    assert stats["empty_position_ratio"] == 0.5

    conclusion = _replay_conclusion({
        "fixed_initial_model": {"statistics": {"total_return": 1.0}},
        "always_retrain": {"statistics": {"total_return": 1.5}},
        "governed_promotion": {"statistics": {"total_return": 2.0}},
    })
    assert conclusion["better_than_fixed_initial_model"] is True
    assert conclusion["better_than_always_retrain"] is True
    assert conclusion["recommend_enable_promotion"] is True
