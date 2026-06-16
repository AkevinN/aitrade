"""
任务执行记录增强测试套件（task-scheduler-observability Wave 1，任务 2）

覆盖：
- 计时数学（started<=finished、duration>=0）
- 失败任务 traceback 非空 ≤8000 且 logger.exception 被调（caplog）
- params 深拷贝隔离（调用方修改不影响存储记录）
- TSO-2 属性测试（Hypothesis：含凭证键的 params 存储后凭证值不出现在 model_dump JSON 文本）
- 嵌套 dict 凭证脱敏（审查 Minor #2）
- list 内嵌 dict 凭证脱敏（TSO-8 补充）
- TSO-2 属性测试扩展：生成器支持嵌套 dict 与 list[dict]，断言整个 JSON 不含凭证原值
- 失败任务 error_traceback 不含 params 凭证值（审查 Minor #3）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from aitrade.models import TaskStatus, TaskType
from aitrade.task import task_manager


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def wait_terminal(task_id: str, timeout: float = 5.0) -> Any:
    """等待任务到达终态，返回最新 TaskModel 或 None（超时）。"""
    deadline = time.time() + timeout
    task = task_manager.get_task(task_id)
    while task is not None and task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        if time.time() > deadline:
            return None
        time.sleep(0.02)
        task = task_manager.get_task(task_id)
    return task


# ---------------------------------------------------------------------------
# 计时数学
# ---------------------------------------------------------------------------

def test_timing_started_le_finished() -> None:
    """完成任务：started_at <= finished_at。"""
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD)

    def _work():
        time.sleep(0.01)
        return {}

    task_manager.run_async(task_id, _work)
    task = wait_terminal(task_id)
    assert task is not None
    assert task.started_at is not None
    assert task.finished_at is not None
    assert task.started_at <= task.finished_at


def test_timing_duration_nonnegative() -> None:
    """完成任务：duration_ms >= 0。"""
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD)
    task_manager.run_async(task_id, lambda: {})
    task = wait_terminal(task_id)
    assert task is not None
    assert task.duration_ms is not None
    assert task.duration_ms >= 0


def test_timing_failed_task() -> None:
    """失败任务：started_at/finished_at/duration_ms 均有值。"""
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD)

    def _fail():
        raise RuntimeError("故意失败")

    task_manager.run_async(task_id, _fail)
    task = wait_terminal(task_id)
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.started_at is not None
    assert task.finished_at is not None
    assert task.duration_ms is not None
    assert task.duration_ms >= 0
    assert task.started_at <= task.finished_at


# ---------------------------------------------------------------------------
# 失败任务 traceback
# ---------------------------------------------------------------------------

def test_failed_task_traceback_nonempty(caplog) -> None:
    """失败任务 error_traceback 非空且 <= 8000 字符，logger.exception 被调。"""
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD)

    def _fail():
        raise ValueError("traceback_test_sentinel")

    with caplog.at_level(logging.ERROR, logger="aitrade.task.manager"):
        task_manager.run_async(task_id, _fail)
        task = wait_terminal(task_id)

    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.error_traceback != ""
    assert len(task.error_traceback) <= 8000
    assert "traceback_test_sentinel" in task.error_traceback

    # logger.exception 应被调用
    assert any(
        "traceback_test_sentinel" in r.message or task_id in r.message
        for r in caplog.records
        if r.levelname in ("ERROR", "CRITICAL")
    ), "期望 logger.exception 记录含任务 ID 或异常信息的日志"


def test_traceback_truncated_to_8000() -> None:
    """traceback 截断不超过 8000 字符。"""
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD)

    def _fail():
        # 构造递归深栈
        def _deep(n):
            if n == 0:
                raise RuntimeError("deep")
            _deep(n - 1)
        _deep(200)

    task_manager.run_async(task_id, _fail)
    task = wait_terminal(task_id)
    assert task is not None
    assert len(task.error_traceback) <= 8000


# ---------------------------------------------------------------------------
# params 深拷贝隔离
# ---------------------------------------------------------------------------

def test_params_deep_copy_isolation() -> None:
    """调用方在 create_task 后修改原 dict，不影响存储的 params。"""
    original_params = {"name": "test-model", "lr": 0.001}
    task_id = task_manager.create_task(TaskType.MODEL_TRAIN, original_params)
    task = task_manager.get_task(task_id)
    assert task is not None
    assert task.params["name"] == "test-model"

    # 修改原始 dict
    original_params["name"] = "mutated"
    original_params["extra"] = "injected"

    # 存储中的 params 不应被影响
    task_after = task_manager.get_task(task_id)
    assert task_after is not None
    assert task_after.params["name"] == "test-model"
    assert "extra" not in task_after.params


def test_params_none_stores_empty_dict() -> None:
    """create_task(params=None) 时存储空 dict 而非 None。"""
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, None)
    task = task_manager.get_task(task_id)
    assert task is not None
    assert task.params == {}


# ---------------------------------------------------------------------------
# TSO-2 属性测试：params 脱敏
# Feature: task-scheduler-observability, Property TSO-2: 对任意含凭证键的 params dict，
# 存储后凭证值不出现在 model_dump 的 JSON 文本中。
# ---------------------------------------------------------------------------

_CRED_KEY_NAMES = [
    "token", "Token", "TOKEN",
    "secret", "Secret", "SECRET",
    "webhook", "Webhook", "WEBHOOK_URL",
    "password", "Password", "PASSWORD",
    "api_token", "secret_key", "webhook_url",
]

# Hypothesis 策略：含凭证键的 params
_cred_params_st = st.fixed_dictionaries(
    {
        "name": st.text(min_size=1, max_size=20),
        "value": st.floats(min_value=0, max_value=1, allow_nan=False),
    }
).flatmap(
    lambda base: st.lists(
        st.sampled_from(_CRED_KEY_NAMES), min_size=1, max_size=3, unique=True
    ).flatmap(
        lambda cred_keys: st.fixed_dictionaries(
            {k: st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=["C"])) for k in cred_keys}
        ).map(lambda creds: {**base, **creds})
    )
)


@settings(max_examples=100)
@given(params=_cred_params_st)
def test_tso2_credentials_sanitized(params: dict[str, Any]) -> None:
    """
    # Feature: task-scheduler-observability, Property TSO-2: 任意含凭证键的 params dict，
    # 存储后凭证值不出现在 model_dump 的 JSON 文本中。
    """
    # 找出所有疑似凭证键及其原始值
    cred_keywords = {"token", "secret", "webhook", "password"}
    cred_values: list[str] = [
        v for k, v in params.items()
        if any(kw in k.lower() for kw in cred_keywords) and isinstance(v, str) and v != "***"
    ]
    if not cred_values:
        return  # 无凭证值则跳过

    task_id = task_manager.create_task(TaskType.MODEL_TRAIN, params)
    task = task_manager.get_task(task_id)
    assert task is not None

    stored_params: dict[str, Any] = task.model_dump(mode="json")["params"]

    # 对每个凭证键：存储值必须是 "***"，而非原始凭证值
    for k, orig_v in params.items():
        if any(kw in k.lower() for kw in cred_keywords) and isinstance(orig_v, str):
            assert stored_params.get(k) == "***", (
                f"凭证键 {k!r} 的存储值应为 '***'，实际为 {stored_params.get(k)!r}"
            )
            if orig_v != "***":
                assert stored_params.get(k) != orig_v, (
                    f"凭证键 {k!r} 的原始值 {orig_v!r} 不应原样存储"
                )


# ---------------------------------------------------------------------------
# 嵌套 dict 凭证脱敏（审查 Minor #2：递归 dict 分支覆盖）
# ---------------------------------------------------------------------------

def test_sanitize_nested_dict_credentials() -> None:
    """嵌套 dict 中的凭证键值被脱敏。"""
    params = {
        "name": "test",
        "auth": {
            "token": "super-secret-token",
            "user": "alice",
        },
    }
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, params)
    task = task_manager.get_task(task_id)
    assert task is not None
    auth = task.params["auth"]
    assert isinstance(auth, dict)
    assert auth["token"] == "***", f"嵌套 token 应被脱敏，实际: {auth['token']!r}"
    assert auth["user"] == "alice"  # 非凭证键保持原值


def test_sanitize_deeply_nested_dict_credentials() -> None:
    """三层嵌套 dict 中的凭证键值也被脱敏。"""
    params = {
        "level1": {
            "level2": {
                "password": "deep-secret",
                "safe": "keep-me",
            }
        }
    }
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, params)
    task = task_manager.get_task(task_id)
    assert task is not None
    deep = task.params["level1"]["level2"]
    assert deep["password"] == "***", f"深层嵌套 password 应被脱敏，实际: {deep['password']!r}"
    assert deep["safe"] == "keep-me"


# ---------------------------------------------------------------------------
# list 内嵌 dict 凭证脱敏（TSO-8 补充）
# ---------------------------------------------------------------------------

def test_sanitize_list_with_dict_credentials() -> None:
    """list 元素中的 dict 凭证键值被脱敏（TSO-8 一般形态修复验证）。"""
    params = {
        "items": [
            {"token": "leak-me", "name": "item-1"},
            {"name": "item-2", "value": 42},
        ]
    }
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, params)
    task = task_manager.get_task(task_id)
    assert task is not None
    items = task.params["items"]
    assert isinstance(items, list)
    assert items[0]["token"] == "***", f"list 内 token 应被脱敏，实际: {items[0]['token']!r}"
    assert items[0]["name"] == "item-1"  # 非凭证键保持原值
    assert items[1]["name"] == "item-2"
    assert items[1]["value"] == 42


def test_sanitize_tuple_with_dict_credentials() -> None:
    """tuple 元素中的 dict 凭证键值被脱敏。"""
    params = {
        "configs": (
            {"secret": "tuple-secret", "id": 1},
        )
    }
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, params)
    task = task_manager.get_task(task_id)
    assert task is not None
    configs = task.params["configs"]
    assert configs[0]["secret"] == "***", f"tuple 内 secret 应被脱敏，实际: {configs[0]['secret']!r}"
    assert configs[0]["id"] == 1


# ---------------------------------------------------------------------------
# TSO-2 属性测试扩展：嵌套结构，断言整个 model_dump JSON 不含凭证原值
# ---------------------------------------------------------------------------

# 非凭证标量策略
_safe_scalar_st = st.one_of(
    st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_categories=["C"])),
    st.integers(min_value=0, max_value=9999),
    st.floats(min_value=0, max_value=1, allow_nan=False),
)

# 含凭证键的叶 dict（1~2 个凭证键 + 1 个安全键）
_leaf_cred_dict_st = st.lists(
    st.sampled_from(_CRED_KEY_NAMES), min_size=1, max_size=2, unique=True
).flatmap(
    lambda cred_keys: st.fixed_dictionaries(
        {
            **{k: st.text(min_size=1, max_size=40, alphabet=st.characters(blacklist_categories=["C"])) for k in cred_keys},
            "safe_key": _safe_scalar_st,
        }
    )
)


@settings(max_examples=80)
@given(
    flat_cred=st.fixed_dictionaries(
        {k: st.text(min_size=1, max_size=40, alphabet=st.characters(blacklist_categories=["C"]))
         for k in ["token", "secret"]}
    ),
    nested_dict=_leaf_cred_dict_st,
    list_item=_leaf_cred_dict_st,
)
def test_tso2_extended_full_json_no_cred_leakage(
    flat_cred: dict[str, Any],
    nested_dict: dict[str, Any],
    list_item: dict[str, Any],
) -> None:
    """
    # Feature: task-scheduler-observability, Property TSO-2 扩展：
    # 含凭证键的 params（包括嵌套 dict 与 list[dict] 形态），
    # 存储后整个 model_dump JSON 序列化文本中所有凭证键对应值均为 '***'。
    # 断言方式：遍历 model_dump JSON 中所有凭证键，其值必须为 '***'（不依赖原值是否出现，
    # 避免原值恰好是普通子串的误报）。
    """
    cred_keywords = {"token", "secret", "webhook", "password"}

    # 构造含三类形态的 params
    params: dict[str, Any] = {
        **flat_cred,           # 顶层凭证键
        "auth": nested_dict,   # 嵌套 dict 凭证
        "items": [list_item],  # list 内嵌 dict 凭证
    }

    task_id = task_manager.create_task(TaskType.MODEL_TRAIN, params)
    task = task_manager.get_task(task_id)
    assert task is not None

    # 遍历 model_dump JSON 结构，断言所有凭证键的值均为 "***"
    def _assert_all_creds_masked(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                child_path = f"{path}.{k}"
                if any(kw in k.lower() for kw in cred_keywords):
                    assert v == "***", (
                        f"凭证键 {child_path!r} 的值应为 '***'，实际为 {v!r}（TSO-2 属性违反）"
                    )
                else:
                    _assert_all_creds_masked(v, child_path)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _assert_all_creds_masked(item, f"{path}[{i}]")

    stored = task.model_dump(mode="json")["params"]
    _assert_all_creds_masked(stored, "params")


# ---------------------------------------------------------------------------
# 失败任务 error_traceback 不含 params 凭证值（审查 Minor #3）
# ---------------------------------------------------------------------------

def test_failed_task_traceback_no_credential_leakage() -> None:
    """
    # 审查 Minor #3：失败任务的 error_traceback 不包含 params 中的凭证原值。
    # 构造函数抛异常且 params 带凭证键，断言 error_traceback 字段无凭证明文。
    """
    secret_value = "SUPER_SECRET_TOKEN_12345_XYZ"
    params = {
        "token": secret_value,
        "name": "leak-test",
    }
    task_id = task_manager.create_task(TaskType.DATA_DOWNLOAD, params)

    def _fail_with_cred_in_scope():
        # 即使本地有凭证变量，traceback 也不应含 params 原值
        _local_secret = secret_value  # noqa: F841
        raise RuntimeError("任务失败，不应泄露凭证")

    task_manager.run_async(task_id, _fail_with_cred_in_scope)
    task = wait_terminal(task_id)

    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.error_traceback is not None

    # params 存储层凭证已脱敏
    assert task.params.get("token") == "***", "params token 应被脱敏"

    # traceback 文本本身不由 params 生成，但确认凭证值不出现在任务记录的 JSON 中
    full_json = json.dumps(task.model_dump(mode="json"), ensure_ascii=False)
    assert secret_value not in full_json, (
        f"凭证值 {secret_value!r} 不应出现在任务记录的完整 JSON 中（含 error_traceback 字段）"
    )
