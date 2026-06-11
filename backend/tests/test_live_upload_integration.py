"""
数据源——上传（upload）分支端到端集成验证（交易操作台特性，任务 11）。

任务 11 是「数据源拉取（pull）集成验证 [需用户手动]」：

- pull 分支依赖真实 Tushare token（`TUSHARE_TOKEN`）拉取决策日行情，**无法在
  无 token 环境自动验证**，其手动验证步骤记录于
  `.kiro/specs/trading-console/manual-verification-pull.md`。
- upload 分支**可在仓库内自动验证**：本测试用样例 CSV 经既有
  `/api/alpha/bar-data/import` 把决策日行情写入 AlphaLab，再触发决策，
  让编排器用**真实**的 `_load_close_price`（不打桩）从已准备好的 AlphaLab
  bar 数据读取决策日收盘价。只对 CNN 推理 `predict_cnn_signals` 打桩
  （需要真实 .pt 模型权重，超出本集成范围）。

覆盖需求：
- 5.1：upload 数据源复用 CSV_Import 接收行情写入数据目录。
- 5.4：行情就绪后用已就绪的目标标的行情进行推理 / 取价（与来源无关）。
- 5.3：决策日目标标的行情在数据源中不可得时返回错误并说明行情缺失。

不依赖任何真实网络 / Tushare token / CNN 模型权重；行情来自仓库内样例 CSV，
AlphaLab 与决策存储、模型库目录均隔离在 `tmp_path`。
"""

from __future__ import annotations

import time
from datetime import date, datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

from aitrade import config as app_config
from aitrade.api import alpha as alpha_api
from aitrade.api import live as live_api
from aitrade.live import orchestrator
from aitrade.live.decision import DecisionStore
from aitrade.main import create_app


VT_SYMBOL = "000001.SZSE"
SCHEME = "eod_buy_v1"
MODEL = "测试"

# 样例日线 CSV：决策日 2024-01-03 收盘 10.9（最后一行）。
SAMPLE_BAR_CSV = b"""trade_date,symbol,open,high,low,close,volume
2024-01-02,000001,10,11,9,10.5,1000
2024-01-03,000001,10.5,11.2,10.2,10.9,1200
"""

IMPORTED_TRADE_DATE = date(2024, 1, 3)
IMPORTED_CLOSE = 10.9
# as_of 取导入决策日收盘后；Decision_Bar = 当日（与导入日线一致）。
AS_OF = datetime(2024, 1, 3, 15, 5)


def _signal_frame(signal: float, *, trade_date: date) -> pl.DataFrame:
    """构造 predict_cnn_signals 同 schema 的桩输出：[datetime, vt_symbol, signal]。"""
    return pl.DataFrame(
        {
            "datetime": [datetime.combine(trade_date, datetime.min.time())],
            "vt_symbol": [VT_SYMBOL],
            "signal": [float(signal)],
        }
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离的 TestClient：

    - AlphaLab 路径（CSV 导入写入 + `_load_close_price` 读取）共用同一 tmp 目录，
      使「上传 → 行情就绪 → 决策取价」形成真实端到端链路；
    - 决策存储 / 模型库目录隔离到 tmp_path，并预置一个桩模型文件以通过 404 校验。
    """
    alpha_lab_path = tmp_path / "alpha_lab"
    alpha_lab_path.mkdir(parents=True, exist_ok=True)
    # alpha API 模块级 ALPHA_LAB_PATH（CSV 导入写入处）。
    monkeypatch.setattr(alpha_api, "ALPHA_LAB_PATH", alpha_lab_path)
    # 编排器 _load_close_price 内部 `from ..config import ALPHA_LAB_PATH`（读取处）。
    monkeypatch.setattr(app_config, "ALPHA_LAB_PATH", alpha_lab_path)

    store = DecisionStore(tmp_path / "decisions")
    monkeypatch.setattr(live_api, "_store", store)

    model_dir = tmp_path / "cnn_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{MODEL}.pt").write_bytes(b"stub-model")
    monkeypatch.setattr(live_api, "CNN_MODEL_PATH", model_dir)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, store


def _import_sample_csv(test_client: TestClient) -> None:
    """经既有 /api/alpha/bar-data/import 把样例日线行情写入 AlphaLab（upload 分支前置步骤）。"""
    resp = test_client.post(
        "/api/alpha/bar-data/import",
        # save_mode=official：直接写入正式资源（默认 batch 仅暂存待合并批次，
        # 编排器 _load_close_price 读取的是正式资源，故集成验证需写正式资源）。
        data={"interval": "d", "import_mode": "merge", "save_mode": "official"},
        files={"file": ("bars.csv", SAMPLE_BAR_CSV, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    assert payload["imported_count"] == 2


def _request_body(**overrides) -> dict:
    body = {
        "model": MODEL,
        "vt_symbol": VT_SYMBOL,
        "scheme": SCHEME,
        "as_of": AS_OF.isoformat(),
        "bar_freq": "1d",
        "data_source": "upload",
        "portfolio": {"portfolio_value": 100000, "current_position": 0},
        "buy_threshold": 0.6,
        "model_version": "v3",
        # 放宽风控，使达标信号产出 buy（默认单票上限 0.30 会拦截满仓买入）。
        "risk": {"max_total_position_ratio": 0.95, "max_single_position_ratio": 0.95},
    }
    body.update(overrides)
    return body


def _poll_task(test_client: TestClient, task_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = test_client.get(f"/api/alpha/tasks/{task_id}")
        assert resp.status_code == 200
        task = resp.json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内完成")


# ---------------------------------------------------------------------------
# 需求 5.1 / 5.4：upload 端到端——CSV 导入 → 行情就绪 → 决策用真实取价读 AlphaLab
# ---------------------------------------------------------------------------
def test_upload_branch_end_to_end_reads_imported_close(client, monkeypatch) -> None:
    """上传样例 CSV 后触发决策，编排器用真实 `_load_close_price` 读决策日收盘价。

    只对 CNN 推理打桩；取价路径不打桩，真实命中已导入的 AlphaLab 日线收盘价。
    """
    test_client, _ = client

    # 1) upload 分支前置：CSV 导入写入 AlphaLab。
    _import_sample_csv(test_client)

    # 2) 仅桩化 CNN 推理（需要真实 .pt 权重，超出集成范围）；取价保持真实。
    monkeypatch.setattr(
        orchestrator,
        "predict_cnn_signals",
        lambda **kwargs: _signal_frame(0.72, trade_date=IMPORTED_TRADE_DATE),
    )

    # 3) 触发决策。
    resp = test_client.post("/api/live/decision", json=_request_body())
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll_task(test_client, task_id)
    assert task["status"] == "completed", f"任务消息: {task.get('message')}"

    decision = task["result"]["decision"]
    # 关键断言：建议价位 == 样例 CSV 决策日收盘价（证明取价真实读自上传行情）。
    assert decision["price"] == IMPORTED_CLOSE
    assert decision["signal"] == 0.72
    assert decision["action"] in ("buy", "sell", "hold")
    assert decision["action"] == "buy"
    # 风控明细完整（5 项）。
    assert len(task["result"]["risk_detail"]) == 5


# ---------------------------------------------------------------------------
# 需求 5.3：决策日行情在数据源中不可得 → 任务失败且消息含「行情缺失」
# ---------------------------------------------------------------------------
def test_upload_branch_missing_quote_fails_task(client, monkeypatch) -> None:
    """导入了 2024-01-03 行情，但对未导入的决策日 2099-12-31 触发 → 取价缺失。"""
    test_client, _ = client
    _import_sample_csv(test_client)

    missing_date = date(2099, 12, 31)
    monkeypatch.setattr(
        orchestrator,
        "predict_cnn_signals",
        lambda **kwargs: _signal_frame(0.72, trade_date=missing_date),
    )

    resp = test_client.post(
        "/api/live/decision",
        json=_request_body(as_of=datetime(2099, 12, 31, 15, 5).isoformat()),
    )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll_task(test_client, task_id)
    assert task["status"] == "failed"
    assert "行情缺失" in task["message"]


# ---------------------------------------------------------------------------
# 需求 5.3：完全未上传任何行情（数据源为空）→ 取价缺失，任务失败
# ---------------------------------------------------------------------------
def test_upload_branch_no_data_imported_fails_task(client, monkeypatch) -> None:
    """不导入任何 CSV，直接触发决策 → AlphaLab 无行情 → 行情缺失。"""
    test_client, _ = client

    monkeypatch.setattr(
        orchestrator,
        "predict_cnn_signals",
        lambda **kwargs: _signal_frame(0.72, trade_date=IMPORTED_TRADE_DATE),
    )

    resp = test_client.post("/api/live/decision", json=_request_body())
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    task = _poll_task(test_client, task_id)
    assert task["status"] == "failed"
    assert "行情缺失" in task["message"]
