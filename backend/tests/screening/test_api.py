"""
CNN 选股 API 端点集成测试。

覆盖 POST /api/cnn/screening/batch（Task 9.1）：
- 合法请求 → 200 + {task_id, name}，task_manager.create_task 以 CNN_SCREENING 被调用。
- 缺失 as_of → 422（FastAPI/Pydantic 校验拦截）。
- lookback_days <= 0 → 422（Field(gt=0) 拦截）。

测试策略：
- monkeypatch ``task_manager.run_async`` 为 no-op，避免触发真实选股任务
  （与 test_cnn_path_api.py 保持同一范式）。
- 用 ``unittest.mock.patch`` 拦截 ``create_task``，断言它以正确 TaskType 被调用，
  并控制返回的 task_id，使端点返回值可验证。
- 整个测试不访问行情数据、不训练模型，是纯 API 层的接线测试。

Feature: cnn-stock-screening, Task 9.1
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aitrade.main import create_app
from aitrade.models.alpha import TaskType

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_VALID_PAYLOAD: dict = {
    "name": "screening_test",
    "as_of": "2025-06-01T00:00:00",
    "lookback_days": 30,
}

_ENDPOINT = "/api/cnn/screening/batch"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """模块级 TestClient。

    同时将 task_manager.run_async patch 为 no-op，保证测试不向真实线程池提交
    选股任务，防止污染本机任务历史归档（.aitrade/task_history）。
    """
    app = create_app()
    with patch("aitrade.api.cnn.task_manager.run_async", return_value=None):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# 正向测试
# ---------------------------------------------------------------------------


class TestStartCnnScreeningSuccess:
    """合法请求应立即返回 200 + {task_id, name}，并以 CNN_SCREENING 创建任务。"""

    def test_returns_200_with_task_id_and_name(self, client: TestClient) -> None:
        """合法 body → 200，响应含 task_id（字符串）与 name（与请求一致）。

        同时桩化 create_task，断言以 TaskType.CNN_SCREENING 调用（接线正确）。
        """
        fake_task_id = "scr-test-0001"
        with patch(
            "aitrade.api.cnn.task_manager.create_task",
            return_value=fake_task_id,
        ) as mock_create:
            resp = client.post(_ENDPOINT, json=_VALID_PAYLOAD)

        assert resp.status_code == 200, f"期望 200，实际: {resp.status_code} {resp.text}"
        body = resp.json()
        assert body["task_id"] == fake_task_id
        assert body["name"] == _VALID_PAYLOAD["name"]

        # 验证 create_task 以 CNN_SCREENING 被调用（接线的关键属性）。
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args.args[0] == TaskType.CNN_SCREENING, (
            f"期望 TaskType.CNN_SCREENING，实际: {call_args.args[0]}"
        )

    def test_returns_name_matching_request(self, client: TestClient) -> None:
        """响应中的 name 应与请求体 name 字段完全一致（不做大小写转换）。"""
        payload = {**_VALID_PAYLOAD, "name": "my_special_screening"}
        with patch("aitrade.api.cnn.task_manager.create_task", return_value="tid-x"):
            resp = client.post(_ENDPOINT, json=payload)

        assert resp.status_code == 200
        assert resp.json()["name"] == "my_special_screening"


# ---------------------------------------------------------------------------
# 参数校验（422）
# ---------------------------------------------------------------------------


class TestStartCnnScreeningValidation:
    """非法 body 应被 FastAPI/Pydantic 以 422 拒绝，不到达任务创建层。"""

    def test_missing_as_of_returns_422(self, client: TestClient) -> None:
        """as_of 为必填字段；缺失时应返回 422。"""
        payload = {"name": "no_as_of", "lookback_days": 30}
        resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 422, (
            f"期望 422（缺 as_of），实际: {resp.status_code} {resp.text}"
        )

    def test_missing_lookback_days_returns_422(self, client: TestClient) -> None:
        """lookback_days 为必填字段；缺失时应返回 422。"""
        payload = {"name": "no_lookback", "as_of": "2025-06-01T00:00:00"}
        resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 422, (
            f"期望 422（缺 lookback_days），实际: {resp.status_code} {resp.text}"
        )

    def test_lookback_days_zero_returns_422(self, client: TestClient) -> None:
        """lookback_days=0 违反 Field(gt=0) 约束，应返回 422。"""
        payload = {**_VALID_PAYLOAD, "lookback_days": 0}
        resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 422, (
            f"期望 422（lookback_days=0），实际: {resp.status_code} {resp.text}"
        )

    def test_lookback_days_negative_returns_422(self, client: TestClient) -> None:
        """lookback_days=-1 违反 Field(gt=0) 约束，应返回 422。"""
        payload = {**_VALID_PAYLOAD, "lookback_days": -1}
        resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 422, (
            f"期望 422（lookback_days=-1），实际: {resp.status_code} {resp.text}"
        )

    def test_missing_name_returns_422(self, client: TestClient) -> None:
        """name 为必填字段；缺失时应返回 422。"""
        payload = {"as_of": "2025-06-01T00:00:00", "lookback_days": 30}
        resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 422, (
            f"期望 422（缺 name），实际: {resp.status_code} {resp.text}"
        )


# ---------------------------------------------------------------------------
# path_class ↔ oco 入口校验（400）
# Feature: cnn-screening-path-class, Property 2（Requirements 2.1/2.2/2.3/2.4）
# ---------------------------------------------------------------------------


class TestPathClassOcoValidation:
    """objective="path_class" 必须配 oco 标签，否则入口快速失败返回 400。

    这层校验与数据集层 (dataset.py) 的同口径 ValueError 构成双层守护：入口做
    友好提示、不创建任务；数据集层做最终兜底。文案与直训接口 /api/cnn/train 一致。
    """

    def test_path_class_without_label_spec_returns_400(self, client: TestClient) -> None:
        """path_class 未提供 label_spec → 400（路径标签依赖三重障碍）。"""
        payload = {**_VALID_PAYLOAD, "objective": "path_class"}
        with patch("aitrade.api.cnn.task_manager.create_task", return_value="tid") as mc:
            resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 400, f"期望 400，实际: {resp.status_code} {resp.text}"
        assert "label_spec.mode=oco" in resp.json()["detail"]
        mc.assert_not_called()  # 校验失败时不应创建任务

    def test_path_class_with_non_oco_label_returns_400(self, client: TestClient) -> None:
        """path_class 配了非 oco 标签（next_bar）→ 400。"""
        payload = {
            **_VALID_PAYLOAD,
            "objective": "path_class",
            "label_spec": {"mode": "next_bar"},
        }
        with patch("aitrade.api.cnn.task_manager.create_task", return_value="tid") as mc:
            resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 400, f"期望 400，实际: {resp.status_code} {resp.text}"
        assert "label_spec.mode=oco" in resp.json()["detail"]
        mc.assert_not_called()

    def test_path_class_oco_missing_take_profit_returns_400(self, client: TestClient) -> None:
        """path_class + oco 但缺 take_profit → 400（要求正的止盈/止损）。"""
        payload = {
            **_VALID_PAYLOAD,
            "objective": "path_class",
            "label_spec": {"mode": "oco", "stop_loss": 0.02},
        }
        with patch("aitrade.api.cnn.task_manager.create_task", return_value="tid") as mc:
            resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 400, f"期望 400，实际: {resp.status_code} {resp.text}"
        assert "take_profit" in resp.json()["detail"]
        mc.assert_not_called()

    def test_path_class_with_valid_oco_passes(self, client: TestClient) -> None:
        """path_class + 合法 oco（正的 tp/sl）→ 放行，照常创建任务返回 200。"""
        payload = {
            **_VALID_PAYLOAD,
            "objective": "path_class",
            "label_spec": {
                "mode": "oco",
                "take_profit": 0.03,
                "stop_loss": 0.02,
                "max_hold": 10,
            },
        }
        with patch("aitrade.api.cnn.task_manager.create_task", return_value="tid") as mc:
            resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 200, f"期望 200，实际: {resp.status_code} {resp.text}"
        mc.assert_called_once()

    def test_classification_without_label_spec_passes(self, client: TestClient) -> None:
        """classification 不携带 label_spec → 不被 path_class 校验拦截，照常 200。"""
        payload = {**_VALID_PAYLOAD, "objective": "classification"}
        with patch("aitrade.api.cnn.task_manager.create_task", return_value="tid") as mc:
            resp = client.post(_ENDPOINT, json=payload)
        assert resp.status_code == 200, f"期望 200，实际: {resp.status_code} {resp.text}"
        mc.assert_called_once()
