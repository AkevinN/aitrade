"""ScreeningStore 持久化与 build_screening_governance_store 的单元测试。

覆盖：
- save → load 往返（run_id / status / leaderboard 等核心字段保真）
- list_ids 返回已存 run_id 的升序列表；空目录时返回 []
- 文件名净化：含非法字符的 run_id 被净化，文件可创建，往返不丢数据
- build_screening_governance_store 返回 CNNGovernanceStore，
  根目录为 SCREENING_GOVERNANCE_PATH，与生产 CNN_GOVERNANCE_PATH 不同

Feature: cnn-stock-screening, Task 7.1: 选股产物持久化 + 隔离治理 store
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from aitrade import config
from aitrade.cnn.governance import CNNGovernanceStore
from aitrade.screening.store import ScreeningStore, build_screening_governance_store
from aitrade.screening.types import ScreeningResult


# ---------------------------------------------------------------------------
# 辅助：构造最小有效的 ScreeningResult
# ---------------------------------------------------------------------------

def _make_result(
    run_id: str = "run_test_001",
    *,
    universe_size: int = 5,
    rules_id: str = "v1",
) -> ScreeningResult:
    """构造最小有效的 ScreeningResult 以供测试使用。

    Args:
        run_id: 选股运行 ID。
        universe_size: 候选池大小（不影响核心断言）。
        rules_id: 本次所用规则版本标识。

    Returns:
        字段均合法的 :class:`ScreeningResult` 实例。
    """
    return ScreeningResult(
        run_id=run_id,
        status="draft",
        created_at=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        input={"name": "test_screening", "as_of": "2025-06-01T00:00:00"},
        rules_id=rules_id,
        universe_size=universe_size,
        excluded=[],
        leaderboard=[],
        effective_right_bound=None,
        eval_window=None,
    )


# ---------------------------------------------------------------------------
# save → load 往返测试
# ---------------------------------------------------------------------------

class TestScreeningStoreSaveLoad:
    """验证 ScreeningStore 的 save/load 往返正确性。"""

    def test_save_load_roundtrip_basic_fields(self, tmp_path: Path) -> None:
        """save 后 load 所得对象的核心字段与原始对象完全一致。"""
        store = ScreeningStore(base_path=tmp_path)
        result = _make_result("run_roundtrip_001", universe_size=42, rules_id="v2")

        returned_id = store.save(result)
        loaded = store.load(returned_id)

        assert loaded.run_id == result.run_id
        assert loaded.status == "draft"
        assert loaded.universe_size == 42
        assert loaded.rules_id == "v2"
        assert loaded.leaderboard == []
        assert loaded.excluded == []
        assert loaded.effective_right_bound is None
        assert loaded.eval_window is None

    def test_save_returns_run_id(self, tmp_path: Path) -> None:
        """save 方法返回值为原始 result 的 run_id。"""
        store = ScreeningStore(base_path=tmp_path)
        result = _make_result("run_return_check")

        returned = store.save(result)

        assert returned == "run_return_check"

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        """save 在 base_path 下创建 .json 文件。"""
        store = ScreeningStore(base_path=tmp_path)
        result = _make_result("run_file_check")

        store.save(result)

        assert (tmp_path / "run_file_check.json").exists()

    def test_load_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        """load 不存在的 run_id 时抛出 FileNotFoundError，错误信息包含 run_id。"""
        store = ScreeningStore(base_path=tmp_path)

        with pytest.raises(FileNotFoundError, match="nonexistent_run"):
            store.load("nonexistent_run")

    def test_roundtrip_created_at_preserved(self, tmp_path: Path) -> None:
        """往返后 created_at 时间戳被精确保留（datetime 字段）。"""
        store = ScreeningStore(base_path=tmp_path)
        result = _make_result("run_ts_check")
        original_ts = result.created_at

        store.save(result)
        loaded = store.load("run_ts_check")

        # Pydantic v2 序列化后 UTC offset-aware 与 naive 在 model_dump(mode="json") 均
        # 转字符串再解析；只比较等价性（ISO 格式往返不损失精度）
        assert loaded.created_at == original_ts or str(loaded.created_at) == str(original_ts)

    def test_roundtrip_input_dict_preserved(self, tmp_path: Path) -> None:
        """往返后 input 字典内容完整保留。"""
        store = ScreeningStore(base_path=tmp_path)
        result = _make_result("run_input_check")
        result.input["extra_key"] = "extra_value"

        store.save(result)
        loaded = store.load("run_input_check")

        assert loaded.input["extra_key"] == "extra_value"


# ---------------------------------------------------------------------------
# list_ids 测试
# ---------------------------------------------------------------------------

class TestScreeningStoreListIds:
    """验证 list_ids 方法行为。"""

    def test_list_ids_empty_when_no_files(self, tmp_path: Path) -> None:
        """目录为空时 list_ids 返回空列表。"""
        store = ScreeningStore(base_path=tmp_path)

        assert store.list_ids() == []

    def test_list_ids_returns_sorted_run_ids(self, tmp_path: Path) -> None:
        """list_ids 返回的 run_id 列表为升序排列。"""
        store = ScreeningStore(base_path=tmp_path)
        ids_to_save = ["run_c", "run_a", "run_b"]
        for run_id in ids_to_save:
            store.save(_make_result(run_id))

        ids = store.list_ids()

        assert ids == sorted(ids_to_save)

    def test_list_ids_count_matches_saved(self, tmp_path: Path) -> None:
        """list_ids 的长度等于实际保存的文件数。"""
        store = ScreeningStore(base_path=tmp_path)
        for i in range(5):
            store.save(_make_result(f"run_{i:03d}"))

        assert len(store.list_ids()) == 5


# ---------------------------------------------------------------------------
# 文件名净化测试
# ---------------------------------------------------------------------------

class TestScreeningStoreFilenamesSanitization:
    """验证含非法字符的 run_id 被正确净化且往返正常。"""

    def test_sanitize_unsafe_chars_creates_file(self, tmp_path: Path) -> None:
        """含路径分隔符与冒号的 run_id 被净化后可以创建文件。"""
        store = ScreeningStore(base_path=tmp_path)
        unsafe_run_id = "run/2025-06:01 test"
        result = _make_result(unsafe_run_id)

        store.save(result)

        # 至少存在一个 .json 文件（净化后的文件名）
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        # 净化后的文件名不含非法字符
        stem = json_files[0].stem
        import re
        assert re.fullmatch(r"[A-Za-z0-9._-]+", stem), f"净化后的文件名含非法字符：{stem!r}"

    def test_sanitize_roundtrip_via_sanitized_id(self, tmp_path: Path) -> None:
        """含非法字符的 run_id：save 后用净化后的 stem 作为 run_id 可以 load 回来。"""
        store = ScreeningStore(base_path=tmp_path)
        unsafe_run_id = "run/unsafe:id"
        result = _make_result(unsafe_run_id)

        store.save(result)

        # 列出后得到净化的 stem，用其 load
        ids = store.list_ids()
        assert len(ids) == 1
        loaded = store.load(ids[0])
        # 原始 run_id 保留在 JSON 的 run_id 字段中
        assert loaded.run_id == unsafe_run_id

    def test_sanitize_empty_after_strip_falls_back_to_result(self, tmp_path: Path) -> None:
        """净化后为空的 run_id 回退为 'result'，文件可正常创建。"""
        store = ScreeningStore(base_path=tmp_path)
        # 全部是非法字符：净化后 strip 结果为空字符串
        edge_run_id = "////:::"
        result = _make_result(edge_run_id)

        store.save(result)

        assert (tmp_path / "result.json").exists()

    def test_sanitize_only_legal_chars_unchanged(self, tmp_path: Path) -> None:
        """仅含合法字符的 run_id 保持不变。"""
        store = ScreeningStore(base_path=tmp_path)
        safe_id = "run-2025_06.001"
        result = _make_result(safe_id)

        store.save(result)

        assert (tmp_path / f"{safe_id}.json").exists()


# ---------------------------------------------------------------------------
# build_screening_governance_store 测试
# ---------------------------------------------------------------------------

class TestBuildScreeningGovernanceStore:
    """验证 build_screening_governance_store 工厂函数行为。"""

    def test_returns_cnn_governance_store_instance(self) -> None:
        """工厂函数返回的对象是 CNNGovernanceStore 实例。"""
        store = build_screening_governance_store()

        assert isinstance(store, CNNGovernanceStore)

    def test_root_is_screening_governance_path(self) -> None:
        """返回的 store 根目录为 config.SCREENING_GOVERNANCE_PATH。"""
        store = build_screening_governance_store()

        assert store.root == config.SCREENING_GOVERNANCE_PATH

    def test_root_is_distinct_from_production_governance_path(self) -> None:
        """screening governance store 的根目录与生产 CNN_GOVERNANCE_PATH 不同。"""
        store = build_screening_governance_store()

        assert store.root != config.CNN_GOVERNANCE_PATH

    def test_each_call_returns_independent_instance(self) -> None:
        """多次调用返回不同的对象实例（无共享单例）。"""
        store_a = build_screening_governance_store()
        store_b = build_screening_governance_store()

        assert store_a is not store_b
        assert store_a.root == store_b.root  # 根目录相同

    def test_store_root_directory_exists(self) -> None:
        """工厂函数调用后，根目录已被创建。"""
        store = build_screening_governance_store()

        assert store.root.exists()
        assert store.root.is_dir()


# ---------------------------------------------------------------------------
# ScreeningStore 默认路径测试
# ---------------------------------------------------------------------------

class TestScreeningStoreDefaultPath:
    """验证 ScreeningStore 的默认路径行为。"""

    def test_default_base_path_is_screening_path(self) -> None:
        """不传 base_path 时，base_path 默认为 config.SCREENING_PATH。"""
        store = ScreeningStore()

        assert store.base_path == config.SCREENING_PATH

    def test_custom_base_path_accepted(self, tmp_path: Path) -> None:
        """传入 tmp_path 时，store 使用该路径而非默认。"""
        store = ScreeningStore(base_path=tmp_path)

        assert store.base_path == tmp_path
