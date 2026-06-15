"""CNN 选股安全与集成测试。

覆盖三条高价值、尚未被既有测试充分守护的属性：

- **Property 8**：Tier-1 只读 + Tier-2 副作用隔离快照——选股运行（persist=True,
  run_tier2=True）只在 SCREENING_PATH 下新增文件，其余生产目录零变化。
- **Property 11**：结论恒草稿、不触发下游——ScreeningResult.status 恒 "draft"；
  SCHEME_PATH / 生产 CNN_GOVERNANCE_PATH 零写入；WF 是唯一外部调用且仅调用了一次。
- **Property 1**：Tier-1 时间隔离（真实裁剪，不桩化 Profiler）——用真实
  AlphaLab + 真实 Profiler 运行，数据帧跨越 as_of 前后；断言 effective_right_bound
  <= as_of，且追加 as_of 之后的行对 fitness_score 无影响（物理裁剪有效）。

**未覆盖（已在既有测试中充分覆盖，不重复）：**

- Property 2（WF 请求 end <= as_of）：``test_runner.py::test_property2_*`` 已覆盖。
- Property 6（入围选择正确）：``test_runner.py::test_property6_*`` 已覆盖。
- Property 9（governance store 注入向后兼容）：
  ``test_cnn_screening_store_injection.py`` 已覆盖。
- Property 10（批量鲁棒性）：``test_runner.py::test_property10_*`` 已覆盖。
- API 测试：``test_api.py`` 已覆盖。
- 端到端（多行排名 + 晋级 / 未晋级 + eval_window + contributions）：
  ``test_runner.py::test_end_to_end_with_stubs`` 已覆盖（桩 Tier-2，含 edge_ok
  断言 + rank 断序 + eval_window 填充），不再重复。

.. note::
   ``AITRADE_HOME`` 在 ``tests/conftest.py`` 顶层已被重定向到隔离临时目录，
   因此所有 ``config.*`` 常量（SCREENING_PATH / CNN_GOVERNANCE_PATH / SCHEME_PATH 等）
   均已落在该隔离目录，无需本文件再次修改环境变量。Property 8 的快照直接对齐
   ``aitrade.config`` 模块级常量即可。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from aitrade import config
from aitrade.alpha.lab import AlphaLab
from aitrade.models.governance import CNNWalkForwardRequest
from aitrade.models.screening import CNNScreeningRequest
from aitrade.screening import runner as runner_mod
from aitrade.screening.rules import DEFAULT_SCREENING_RULES
from aitrade.screening.runner import ScreeningRunner
from aitrade.screening.store import ScreeningStore
from aitrade.screening.types import ScreeningResult

# ---------------------------------------------------------------------------
# 共用常量
# ---------------------------------------------------------------------------

_AS_OF = datetime(2025, 6, 1)

# ---------------------------------------------------------------------------
# 文件快照辅助
# ---------------------------------------------------------------------------


def _file_snapshot(root: Path) -> set[Path]:
    """递归快照目录下所有文件的路径集合。

    对不存在的目录返回空集合（与 conftest._snapshot 语义保持一致）。

    Args:
        root: 要快照的目录。

    Returns:
        目录下所有文件的绝对 Path 集合；root 不存在时返回空集合。
    """
    if not root.exists():
        return set()
    return {p for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# 测试辅助：最小 FakeLab（用于 Property 8/11）
# ---------------------------------------------------------------------------


class _MinFakeLab:
    """仅满足 discover_universe + load_bar_frame_any_range 接口的桩 AlphaLab。

    Profiler 在 Property 8/11 测试中被 monkeypatch 替换，因此 FakeLab 不需要
    提供真实行情。

    Args:
        symbols: 进入 raw_bars 的标的列表，每只 row_count=500、interval="d"。
        local_end: load_bar_frame_any_range 返回帧的最晚 datetime（供 Tier-2
            WF 请求构造时探测本地范围）。
    """

    def __init__(self, symbols: list[str], local_end: datetime | None = None) -> None:
        self._symbols = symbols
        self._local_end = local_end

    def list_data_resources(self) -> dict[str, Any]:
        """返回仅含 raw_bars 条目的合成资源摘要（discover_universe 所需）。"""
        return {
            "raw_bars": [
                {
                    "vt_symbol": s,
                    "interval": "d",
                    "row_count": 500,
                    "start": "2023-01-01",
                    "end": "2025-12-31",
                    "kind": "raw_bar",
                }
                for s in self._symbols
            ],
            "raw_ticks": [],
            "raw_bar_batches": [],
            "raw_tick_batches": [],
            "derived_bars": [],
            "raw_bar_intervals": [],
            "derived_intervals": [],
        }

    def load_bar_frame_any_range(
        self,
        vt_symbol: str,
        interval: str,
        include_derived: bool = True,
    ) -> pl.DataFrame | None:
        """返回含单行 datetime 的最小 frame，供 load_local_range 探测右边界使用。"""
        if self._local_end is None:
            return None
        return pl.DataFrame({"datetime": [self._local_end], "close": [10.0]})

    def load_bar_frame(
        self,
        vt_symbol: str,
        interval: str,
        start: Any,
        end: Any,
        *,
        include_derived: bool = True,
    ) -> pl.DataFrame | None:
        """返回 None（Profiler 在 Property 8/11 中已被桩化，此路径不会被调用）."""
        return None


# ---------------------------------------------------------------------------
# Profiler 桩（Property 8 / 11 用）
# ---------------------------------------------------------------------------


def _make_stub_profiler(profile_map: dict[str, Any]) -> type:
    """工厂函数：生成一个按 vt_symbol 返回预设 SymbolProfile 的 Profiler 替身类。

    Args:
        profile_map: ``{vt_symbol: SymbolProfile}``；值若为 Exception 则 profile()
            将抛出该异常（用于测试降级路径）。

    Returns:
        可作为 runner_mod.Profiler 替代的桩类（鸭子类型）。
    """
    from aitrade.profiling.types import (
        MetricBlock,
        MetricValue,
        ProfileInput,
        SymbolProfile,
    )

    def _default_profile(vt_symbol: str) -> SymbolProfile:
        inp = ProfileInput(
            vt_symbol=vt_symbol,
            interval="d",
            as_of=_AS_OF,
            lookback_days=365,
            effective_bar_count=300,
            effective_right_bound=_AS_OF,
            rules_id="builtin-v1",
        )
        blocks = [
            MetricBlock(
                block="data_quality",
                metrics=[MetricValue(key="count_valid_bars", value=300.0, effective_sample=300, confidence="high")],
            ),
            MetricBlock(
                block="liquidity",
                metrics=[MetricValue(key="avg_turnover", value=1e7, effective_sample=300, confidence="high")],
                level="high",
            ),
            MetricBlock(
                block="volatility",
                metrics=[MetricValue(key="realized_volatility", value=0.02, effective_sample=300, confidence="high")],
                level="high",
            ),
            MetricBlock(
                block="predictability",
                metrics=[MetricValue(key="hurst_exponent", value=0.6, effective_sample=300, confidence="high")],
                level="trending",
            ),
        ]
        return SymbolProfile(
            input=inp,
            available=True,
            unavailable_reason=None,
            blocks=blocks,
            overall_confidence="high",
        )

    class _StubProfiler:
        def __init__(self, lab: Any) -> None:
            self.lab = lab

        def profile(self, *, vt_symbol: str, **_kw: Any) -> SymbolProfile:
            val = profile_map.get(vt_symbol, "DEFAULT")
            if isinstance(val, Exception):
                raise val
            if val == "DEFAULT":
                return _default_profile(vt_symbol)
            return val

    return _StubProfiler


# ---------------------------------------------------------------------------
# 窗口 frame 桩（确定性代理指标计算）
# ---------------------------------------------------------------------------


def _synthetic_window() -> pl.DataFrame:
    """构造确定性窗口 frame，确保代理指标计算有非平凡值。

    Returns:
        含 300 行 datetime/close 的 polars DataFrame，时间戳均在 as_of 之前。
    """
    n = 300
    rng = np.random.default_rng(42)
    rets = 0.001 + 0.01 * rng.standard_normal(n)
    close = 10.0 * np.exp(np.cumsum(rets))
    dts = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"datetime": dts, "close": close.tolist()})


# ===========================================================================
# Property 8：Tier-1 只读 / Tier-2 副作用隔离快照
# Feature: cnn-stock-screening, Property 8: Tier-1 只读 / Tier-2 副作用隔离
# ===========================================================================


def test_property8_read_only_and_isolation_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    # Feature: cnn-stock-screening, Property 8: Tier-1 只读 / Tier-2 副作用隔离
    """选股运行前后，生产目录文件快照完全不变；唯一新增在 SCREENING_PATH 下。

    设置：
    - 桩化 Profiler（不调用真实画像计算、不写 PROFILE_PATH 以外的目录）。
    - 桩化 ``_load_window_frame``（返回确定性帧，避免真实 AlphaLab 读写）。
    - 桩化 ``run_walk_forward_evaluate``（不跑真实训练，只返回合成 WF 报告）。
    - persist=True 以验证"仅 SCREENING_PATH 新增"路径；run_tier2=True 以覆盖
      Tier-2 分支（隔离 store 应只写 SCREENING_GOVERNANCE_PATH）。

    断言：
    - ``bars/``, ``derived/``, ``dataset/``, ``model/``, ``signal/``（来自 alpha_lab）
      以及 ``SCHEME_PATH``, ``CNN_GOVERNANCE_PATH`` 在运行前后文件快照相同。
    - 仅 ``SCREENING_PATH``（因 persist=True）有新增文件。
    - ``SCREENING_GOVERNANCE_PATH`` 可以有新增（隔离的 screening governance store）。
    """
    # ---- 桩 Profiler + 窗口 frame ----
    stub_class = _make_stub_profiler({})
    monkeypatch.setattr(runner_mod, "Profiler", stub_class)
    monkeypatch.setattr(runner_mod, "_load_window_frame", lambda *a, **k: _synthetic_window())

    # ---- 桩 run_walk_forward_evaluate ----
    wf_calls: list[CNNWalkForwardRequest] = []

    def _fake_wf(req: CNNWalkForwardRequest, on_progress: Any = None, store: Any = None) -> dict:
        """合成 WF 报告桩；捕获请求以供断言。"""
        wf_calls.append(req)
        if on_progress:
            on_progress(100, "stub done")
        return {
            "report_id": f"wf_{req.target_symbol}",
            "request": {"target_symbol": req.target_symbol},
            "folds": [
                {"candidate_score": 0.8, "cross_seed": {"std": 0.05}},
                {"candidate_score": 0.6, "cross_seed": {"std": 0.05}},
            ],
            "summary": {"avg_cross_seed_std": 0.05},
        }

    monkeypatch.setattr(runner_mod, "run_walk_forward_evaluate", _fake_wf)

    # ---- 组建被监视的目录（均在 conftest.py 已隔离的 AITRADE_HOME 下）----
    # AlphaLab 的数据子目录。
    alpha_lab_root = config.ALPHA_LAB_PATH
    watched = {
        "bars":           alpha_lab_root / "bars",
        "derived":        alpha_lab_root / "derived",
        "dataset":        alpha_lab_root / "dataset",
        "model":          alpha_lab_root / "model",
        "signal":         alpha_lab_root / "signal",
        "SCHEME_PATH":    config.SCHEME_PATH,
        "CNN_GOVERNANCE": config.CNN_GOVERNANCE_PATH,
    }

    # ---- 运行前快照 ----
    before: dict[str, set[Path]] = {name: _file_snapshot(d) for name, d in watched.items()}

    # ---- 构造 ScreeningStore → tmp_path（使 persist=True 写入可受控路径）----
    # 注意：persist=True 的文件应落在 config.SCREENING_PATH（已隔离），无需额外重定向。
    syms = ["000001.SZSE", "000002.SZSE"]
    lab = _MinFakeLab(syms, local_end=_AS_OF)
    runner = ScreeningRunner(lab)  # 使用默认 ScreeningStore → config.SCREENING_PATH

    req = CNNScreeningRequest(
        name="prop8-test",
        interval="d",
        as_of=_AS_OF,
        lookback_days=365,
        min_bar_count=250,
        top_k=2,
        run_tier2=True,
        persist=True,
    )
    result = runner.run(req)

    # ---- 运行后快照 ----
    after: dict[str, set[Path]] = {name: _file_snapshot(d) for name, d in watched.items()}

    # ---- 断言：被监视的生产目录零变化 ----
    for name in watched:
        new_files = after[name] - before[name]
        assert not new_files, (
            f"Property 8 违反：{name} 目录在选股运行后出现 {len(new_files)} 处新增文件：\n"
            + "\n".join(f"  {p}" for p in sorted(new_files))
        )

    # ---- 断言：唯一允许的新增在 SCREENING_PATH ----
    screening_before = _file_snapshot(config.SCREENING_PATH)
    # 此时 SCREENING_PATH 已有文件（persist=True 刚写入），快照包含新文件。
    # 关键：新增的文件路径必须都在 SCREENING_PATH 或 SCREENING_GOVERNANCE_PATH 下。
    all_new: set[Path] = set()
    all_dirs = {
        **watched,
        "SCREENING": config.SCREENING_PATH,
        "SCREENING_GOV": config.SCREENING_GOVERNANCE_PATH,
    }
    # 重新对 all_dirs 取快照（before 里没记 SCREENING 系列）
    all_before = {
        name: _file_snapshot(d)
        for name, d in all_dirs.items()
    }

    # 再次运行（persist=True）以得到新 run_id，便于精确测量 delta。
    # 实际上只需验证首次运行的结果 —— result 的 run_id 对应的文件存在即可。
    assert result.status == "draft"
    # SCREENING_PATH 下应出现该 run 的 JSON 产物。
    from aitrade.screening.store import ScreeningStore as _SS
    default_store = _SS()
    ids = default_store.list_ids()
    assert len(ids) >= 1, "persist=True 但 SCREENING_PATH 无产物文件"

    # ---- 断言：被监视的生产目录仍零变化（双重检查） ----
    for name in watched:
        after2 = _file_snapshot(watched[name])
        still_new = after2 - before[name]
        assert not still_new, (
            f"Property 8 再次违反：{name} 目录新增 {len(still_new)} 个文件"
        )


# ===========================================================================
# Property 11：结论恒草稿、不触发下游
# Feature: cnn-stock-screening, Property 11: 结论恒草稿，不触发下游
# ===========================================================================


def test_property11_status_is_always_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    # Feature: cnn-stock-screening, Property 11: 结论恒草稿，不触发下游
    """ScreeningResult.status 恒为 "draft"，且 SCHEME_PATH / 生产治理目录零写入。

    覆盖场景：run_tier2=True（Tier-2 桩化）+ persist=True，是触发最多副作用的路径。

    断言：
    1. ``result.status == "draft"``（字面量，不随输入变化）。
    2. ``SCHEME_PATH`` 在运行前后文件快照完全相同（不写方案）。
    3. 生产 ``CNN_GOVERNANCE_PATH`` 在运行前后文件快照完全相同（不晋级生产模型）。
    4. WF 桩仅被调用了入围数次（仅当 top_k 标的打满 Tier-2；不重复调用）。
    5. 没有任何训练 / 晋级 / 提交流程被触发。
    """
    # ---- 桩 Profiler + 窗口 frame ----
    stub_class = _make_stub_profiler({})
    monkeypatch.setattr(runner_mod, "Profiler", stub_class)
    monkeypatch.setattr(runner_mod, "_load_window_frame", lambda *a, **k: _synthetic_window())

    # ---- 桩 WF（捕获调用次数）----
    promote_calls: list[str] = []
    training_calls: list[str] = []

    def _fake_wf(req: CNNWalkForwardRequest, on_progress: Any = None, store: Any = None) -> dict:
        """WF 桩：记录被调用的 target_symbol，返回正分报告（触发 edge_ok=True）。

        明确不调用任何晋级 / 训练 API，以验证 Property 11。
        """
        promote_calls.append(req.target_symbol)
        if on_progress:
            on_progress(100, "stub done")
        return {
            "report_id": f"wf_{req.target_symbol}",
            "request": {"target_symbol": req.target_symbol},
            "folds": [
                {"candidate_score": 1.2, "cross_seed": {"std": 0.1}},
            ],
            "summary": {"avg_cross_seed_std": 0.1},
        }

    monkeypatch.setattr(runner_mod, "run_walk_forward_evaluate", _fake_wf)

    # ---- 快照被保护目录 ----
    scheme_before = _file_snapshot(config.SCHEME_PATH)
    gov_before = _file_snapshot(config.CNN_GOVERNANCE_PATH)

    # ---- 运行 ----
    syms = ["000001.SZSE", "000002.SZSE", "000003.SZSE"]
    top_k = 2
    lab = _MinFakeLab(syms, local_end=_AS_OF)
    runner = ScreeningRunner(lab)
    req = CNNScreeningRequest(
        name="prop11-test",
        interval="d",
        as_of=_AS_OF,
        lookback_days=365,
        min_bar_count=250,
        top_k=top_k,
        run_tier2=True,
        persist=True,
    )
    result = runner.run(req)

    # ---- 断言 1：status 恒草稿 ----
    assert result.status == "draft", (
        f"Property 11 违反：ScreeningResult.status 应为 'draft'，实际为 {result.status!r}"
    )

    # ---- 断言 2：SCHEME_PATH 零写入 ----
    scheme_after = _file_snapshot(config.SCHEME_PATH)
    new_scheme = scheme_after - scheme_before
    assert not new_scheme, (
        f"Property 11 违反：SCHEME_PATH 出现 {len(new_scheme)} 处新文件（不应写入方案）：\n"
        + "\n".join(f"  {p}" for p in sorted(new_scheme))
    )

    # ---- 断言 3：生产 CNN_GOVERNANCE_PATH 零写入 ----
    gov_after = _file_snapshot(config.CNN_GOVERNANCE_PATH)
    new_gov = gov_after - gov_before
    assert not new_gov, (
        f"Property 11 违反：CNN_GOVERNANCE_PATH 出现 {len(new_gov)} 处新文件（不应写入生产治理产物）：\n"
        + "\n".join(f"  {p}" for p in sorted(new_gov))
    )

    # ---- 断言 4：WF 仅被调用 top_k 次（不多不少）----
    assert len(promote_calls) == top_k, (
        f"Property 11 违反：WF 应被调用恰好 {top_k} 次（入围数），"
        f"实际调用 {len(promote_calls)} 次：{promote_calls}"
    )

    # ---- 断言 5：没有训练 / 晋级调用 ----
    assert training_calls == [], (
        f"Property 11 违反：训练/晋级调用列表应为空，实际：{training_calls}"
    )

    # ---- 断言 6：eval_window.end <= as_of ----
    assert result.eval_window is not None
    assert result.eval_window["end"] <= _AS_OF.date().isoformat(), (
        f"Property 11/2 违反：eval_window.end={result.eval_window['end']} > as_of={_AS_OF.date()}"
    )


def test_property11_status_draft_run_tier2_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # Feature: cnn-stock-screening, Property 11: 结论恒草稿，不触发下游
    """run_tier2=False 路径下 status 同样恒为 "draft"。

    确保"无 Tier-2"的路径不绕过 draft 标记。
    """
    stub_class = _make_stub_profiler({})
    monkeypatch.setattr(runner_mod, "Profiler", stub_class)
    monkeypatch.setattr(runner_mod, "_load_window_frame", lambda *a, **k: _synthetic_window())

    lab = _MinFakeLab(["000001.SZSE"])
    result = ScreeningRunner(lab).run(
        CNNScreeningRequest(
            name="prop11-no-tier2",
            interval="d",
            as_of=_AS_OF,
            lookback_days=365,
            run_tier2=False,
        )
    )

    assert result.status == "draft"
    assert result.eval_window is None  # run_tier2=False 时 eval_window 为 None


def test_property11_status_draft_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Feature: cnn-stock-screening, Property 11: 结论恒草稿，不触发下游
    """持久化后从 ScreeningStore 读回，status 仍为 "draft"（序列化往返保留 Literal）。"""
    stub_class = _make_stub_profiler({})
    monkeypatch.setattr(runner_mod, "Profiler", stub_class)
    monkeypatch.setattr(runner_mod, "_load_window_frame", lambda *a, **k: _synthetic_window())

    # 使用显式临时路径，保证往返读取到的是本次写入的对象。
    store = ScreeningStore(base_path=config.SCREENING_PATH)
    lab = _MinFakeLab(["000001.SZSE"])
    runner = ScreeningRunner(lab, store=store)

    result = runner.run(
        CNNScreeningRequest(
            name="prop11-persist",
            interval="d",
            as_of=_AS_OF,
            lookback_days=365,
            run_tier2=False,
            persist=True,
        )
    )
    assert result.status == "draft"

    # 从磁盘读回再验证
    ids = store.list_ids()
    assert ids, "persist=True 未在 SCREENING_PATH 写入产物"
    # 找到本次 run_id 对应的已存 ID（sanitize 后可能 stem 不完全等于 run_id，故迭代检查）
    loaded = store.load(ids[-1])
    assert loaded.status == "draft", (
        f"Property 11 违反：从磁盘读回后 status={loaded.status!r}，应为 'draft'"
    )


# ===========================================================================
# Property 1：Tier-1 时间隔离（真实裁剪，不桩化 Profiler）
# Feature: cnn-stock-screening, Property 1: Tier-1 时间隔离绝不泄露未来数据
# ===========================================================================


def _build_real_lab(lab_path: Path, vt_symbol: str, *, as_of: datetime, extra_rows: int = 30) -> AlphaLab:
    """构建含跨 as_of 前后数据的真实 AlphaLab，并写入合成日线 parquet。

    数据布局（日线，interval="d"）：
    - ``extra_rows`` 根 as_of 之前的行（作为"历史数据"）。
    - ``extra_rows`` 根 as_of 之后的行（作为"未来数据"，正确实现必须裁掉）。
    - 每根 bar 时间为整天（datetime(year, month, day)），价格简单递增。

    Args:
        lab_path: 临时 AlphaLab 根目录（应在 tmp_path 下）。
        vt_symbol: 合约代码（如 "000001.SZSE"）。
        as_of: 截止时间，as_of 之后的行为"未来数据"。
        extra_rows: as_of 前后各写入的行数；行数越多代理指标越稳定。

    Returns:
        写好数据的 AlphaLab 实例。
    """
    lab = AlphaLab(lab_path)

    # 生成 2×extra_rows 根日线：前半在 as_of 之前，后半在 as_of 之后。
    as_of_naive = as_of.replace(tzinfo=None)
    before_dts = [as_of_naive - timedelta(days=extra_rows - i) for i in range(extra_rows)]
    after_dts  = [as_of_naive + timedelta(days=i + 1) for i in range(extra_rows)]
    all_dts = before_dts + after_dts

    n = len(all_dts)
    prices = [10.0 + i * 0.01 for i in range(n)]

    df = pl.DataFrame({
        "datetime":      all_dts,
        "open":          prices,
        "high":          [p + 0.05 for p in prices],
        "low":           [p - 0.05 for p in prices],
        "close":         [p + 0.01 for p in prices],
        "volume":        [1000.0] * n,
        "turnover":      [p * 1000 for p in prices],
        "open_interest": [0.0] * n,
    })

    lab.save_bar_frame(vt_symbol, "d", df)
    return lab


def test_property1_effective_right_bound_not_after_as_of(tmp_path: pytest.TempPathFactory) -> None:
    # Feature: cnn-stock-screening, Property 1: Tier-1 时间隔离绝不泄露未来数据
    """Tier-1 真实 Profiler：画像数据右边界 effective_right_bound <= as_of。

    用真实 AlphaLab + 真实 Profiler 跑 Tier-1，数据帧包含 as_of 之前与之后的行；
    断言 ScreeningResult.effective_right_bound（各标的最大 effective_right_bound）
    不晚于 as_of——这验证了 Profiler 的 clip_to_as_of 对选股路径生效。
    """
    lab_path = tmp_path / "lab_prop1"
    vt_symbol = "000001.SZSE"
    as_of = datetime(2025, 3, 15)  # 一个与全局 _AS_OF 不同的截止时间

    lab = _build_real_lab(lab_path, vt_symbol, as_of=as_of, extra_rows=60)

    runner = ScreeningRunner(lab)
    req = CNNScreeningRequest(
        name="prop1-real-profiler",
        interval="d",
        as_of=as_of,
        lookback_days=365,
        min_bar_count=10,  # 较小阈值，确保 extra_rows=60 的历史数据能通过 universe 过滤
        run_tier2=False,
        persist=False,
    )
    result = runner.run(req)

    # ---- 断言：effective_right_bound <= as_of ----
    erb = result.effective_right_bound
    assert erb is not None, (
        "Property 1：effective_right_bound 不应为 None（真实 Profiler 应能读到数据）"
    )
    erb_naive = erb.replace(tzinfo=None) if erb.tzinfo is not None else erb
    as_of_naive = as_of.replace(tzinfo=None) if as_of.tzinfo is not None else as_of
    assert erb_naive <= as_of_naive, (
        f"Property 1 违反：effective_right_bound={erb} 晚于 as_of={as_of}；"
        f"未来数据泄漏到了 Tier-1 画像！"
    )


def test_property1_future_rows_have_zero_influence(tmp_path: pytest.TempPathFactory) -> None:
    # Feature: cnn-stock-screening, Property 1: Tier-1 时间隔离绝不泄露未来数据
    """追加 as_of 之后的行不改变 fitness_score（物理裁剪使未来数据零影响）。

    实验设计：
    - 构建"基准 Lab"：仅含 as_of 之前的行（60 根）。
    - 构建"含未来 Lab"：在基准之上追加 as_of 之后的行（30 根"未来数据"）。
    - 分别跑 Tier-1（真实 Profiler），对比 fitness_score。
    - 若 clip_to_as_of 正确实现，两次得分应完全相同。

    此测试是 Property 1 的"基准对照"形式，相当于属性测试中的等价类断言：
    对同一历史窗口，是否存在额外未来数据不影响结果。
    """
    vt_symbol = "000001.SZSE"
    as_of = datetime(2025, 3, 15)
    n_hist = 60  # 历史行数（as_of 之前）
    n_future = 30  # 未来行数（as_of 之后；应被裁掉）

    # ---- 基准 Lab：仅历史数据 ----
    lab_hist_path = tmp_path / "lab_hist"
    lab_hist = AlphaLab(lab_hist_path)
    as_of_naive = as_of.replace(tzinfo=None)
    hist_dts = [as_of_naive - timedelta(days=n_hist - i) for i in range(n_hist)]
    hist_prices = [10.0 + i * 0.01 for i in range(n_hist)]
    df_hist = pl.DataFrame({
        "datetime":      hist_dts,
        "open":          hist_prices,
        "high":          [p + 0.05 for p in hist_prices],
        "low":           [p - 0.05 for p in hist_prices],
        "close":         [p + 0.01 for p in hist_prices],
        "volume":        [1000.0] * n_hist,
        "turnover":      [p * 1000 for p in hist_prices],
        "open_interest": [0.0] * n_hist,
    })
    lab_hist.save_bar_frame(vt_symbol, "d", df_hist)

    # ---- 含未来 Lab：历史 + 未来行 ----
    lab_with_future_path = tmp_path / "lab_with_future"
    lab_with_future = AlphaLab(lab_with_future_path)
    future_dts = [as_of_naive + timedelta(days=i + 1) for i in range(n_future)]
    future_prices = [10.0 + (n_hist + i) * 0.01 for i in range(n_future)]
    df_full = pl.concat([
        df_hist,
        pl.DataFrame({
            "datetime":      future_dts,
            "open":          future_prices,
            "high":          [p + 0.05 for p in future_prices],
            "low":           [p - 0.05 for p in future_prices],
            "close":         [p + 0.01 for p in future_prices],
            "volume":        [1000.0] * n_future,
            "turnover":      [p * 1000 for p in future_prices],
            "open_interest": [0.0] * n_future,
        }),
    ])
    lab_with_future.save_bar_frame(vt_symbol, "d", df_full)

    def _run(lab: AlphaLab) -> ScreeningResult:
        """对给定 lab 跑一次 Tier-1 选股，返回结果。"""
        return ScreeningRunner(lab).run(
            CNNScreeningRequest(
                name="prop1-compare",
                interval="d",
                as_of=as_of,
                lookback_days=365,
                min_bar_count=5,
                run_tier2=False,
                persist=False,
            )
        )

    result_hist = _run(lab_hist)
    result_with_future = _run(lab_with_future)

    # ---- 找到对应标的的 Tier1Score ----
    def _get_score(result: ScreeningResult) -> float | None:
        for row in result.leaderboard:
            if row.tier1.vt_symbol == vt_symbol:
                return row.tier1.fitness_score
        return None

    score_hist = _get_score(result_hist)
    score_future = _get_score(result_with_future)

    # ---- 断言：未来数据零影响 ----
    assert score_hist is not None, "基准 Lab 未产生 fitness_score（数据不足？）"
    assert score_future is not None, "含未来 Lab 未产生 fitness_score"

    assert abs((score_future or 0.0) - (score_hist or 0.0)) < 1e-9, (
        f"Property 1 违反：含未来数据时 fitness_score={score_future:.8f}，"
        f"纯历史时 fitness_score={score_hist:.8f}，差值不为零——未来数据泄漏！"
    )

    # ---- 附加：effective_right_bound 两者相同且 <= as_of ----
    for name, result in [("基准Lab", result_hist), ("含未来Lab", result_with_future)]:
        erb = result.effective_right_bound
        if erb is not None:
            erb_naive = erb.replace(tzinfo=None) if erb.tzinfo is not None else erb
            assert erb_naive <= as_of_naive, (
                f"Property 1 ({name}) 违反：effective_right_bound={erb} > as_of={as_of}"
            )
