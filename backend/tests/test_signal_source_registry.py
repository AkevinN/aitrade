"""
Task 1.1/1.2 测试：信号源注册表 + rules 包 + CNN 适配器。

覆盖范围
--------
1. register_signal_source / build_signal_source / list_signal_sources 基本功能
2. 同名覆盖（热更新/测试场景）
3. 未注册时 build_signal_source 抛 KeyError 并列出已注册名
4. name 非空校验（ValueError）
5. factory 非可调用校验（TypeError）
6. import aitrade.rules 不触发 torch 加载（lazy import 保证）
7. "cnn" adapter 转调：monkeypatch predict_cnn_signals 断言 kwargs 透传与 DataFrame 原样返回
8. "cnn" adapter model_name 缺失时构造期抛 ValueError（fail fast）
9. list_signal_sources 返回 name / description / param_spec 结构
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date

import polars as pl
import pytest

# ──────────────────────────────────────────────
# 工具：隔离注册表状态，防止测试之间互相污染
# ──────────────────────────────────────────────

import aitrade.backtest.registry as _reg


# ──────────────────────────────────────────────
# 1. 基本注册 + 构造 + 列表
# ──────────────────────────────────────────────

@pytest.fixture()
def clean_registry():
    """每个使用此 fixture 的测试独享一份空注册表，测试后恢复。"""
    saved = dict(_reg._SIGNAL_SOURCE_REGISTRY)
    _reg._SIGNAL_SOURCE_REGISTRY.clear()
    yield
    _reg._SIGNAL_SOURCE_REGISTRY.clear()
    _reg._SIGNAL_SOURCE_REGISTRY.update(saved)


def _dummy_df() -> pl.DataFrame:
    return pl.DataFrame({"datetime": [], "vt_symbol": [], "signal": []})


class _DummyProvider:
    """用于测试的最小 SignalProvider 实现。"""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def predict(self, start: date, end: date, on_progress=None) -> pl.DataFrame:
        return self._df


def test_register_and_build(clean_registry):
    """注册工厂后 build_signal_source 应返回工厂产出的对象。"""
    df = _dummy_df()
    factory = lambda params: _DummyProvider(df)  # noqa: E731

    _reg.register_signal_source("dummy", factory, description="测试源", param_spec={"k": {}})
    provider = _reg.build_signal_source("dummy", {})

    assert isinstance(provider, _DummyProvider)
    result = provider.predict(date(2024, 1, 1), date(2024, 12, 31))
    assert result is df


def test_list_signal_sources(clean_registry):
    """list_signal_sources 应按名字排序并包含 name/description/param_spec 字段。"""
    spec_a = {"x": {"type": "int"}}
    _reg.register_signal_source("b_source", lambda p: _DummyProvider(_dummy_df()), description="B", param_spec=None)
    _reg.register_signal_source("a_source", lambda p: _DummyProvider(_dummy_df()), description="A", param_spec=spec_a)

    sources = _reg.list_signal_sources()
    assert len(sources) == 2
    # 排序：a_source < b_source
    assert sources[0]["name"] == "a_source"
    assert sources[0]["description"] == "A"
    assert sources[0]["param_spec"] == spec_a
    assert sources[1]["name"] == "b_source"
    assert sources[1]["param_spec"] is None


# ──────────────────────────────────────────────
# 2. 同名覆盖
# ──────────────────────────────────────────────

def test_overwrite_same_name(clean_registry):
    """同名注册应覆盖旧工厂（便于热更新/测试），不抛异常。"""
    df1 = pl.DataFrame({"datetime": [], "vt_symbol": [], "signal": []})
    df2 = pl.DataFrame({"datetime": [1], "vt_symbol": ["X.SZ"], "signal": [0.8]})

    _reg.register_signal_source("overwrite_me", lambda p: _DummyProvider(df1))
    _reg.register_signal_source("overwrite_me", lambda p: _DummyProvider(df2))

    provider = _reg.build_signal_source("overwrite_me", {})
    result = provider.predict(date(2024, 1, 1), date(2024, 12, 31))
    # 应返回第二次注册的工厂产出
    assert len(result) == 1
    assert float(result["signal"][0]) == pytest.approx(0.8)


# ──────────────────────────────────────────────
# 3. 未注册抛 KeyError 并列出已注册名
# ──────────────────────────────────────────────

def test_build_unknown_key_error(clean_registry):
    """build_signal_source 未注册名应抛 KeyError，错误信息包含该名字。"""
    _reg.register_signal_source("real_source", lambda p: _DummyProvider(_dummy_df()))

    with pytest.raises(KeyError, match="not_exist"):
        _reg.build_signal_source("not_exist", {})


def test_build_unknown_lists_registered(clean_registry):
    """KeyError 信息中应列出已注册的名字，方便排查。"""
    _reg.register_signal_source("alpha", lambda p: _DummyProvider(_dummy_df()))

    with pytest.raises(KeyError) as exc_info:
        _reg.build_signal_source("ghost", {})

    assert "alpha" in str(exc_info.value)


# ──────────────────────────────────────────────
# 4. 入参校验
# ──────────────────────────────────────────────

def test_register_empty_name_raises(clean_registry):
    """name 为空字符串时应抛 ValueError。"""
    with pytest.raises(ValueError, match="非空字符串"):
        _reg.register_signal_source("", lambda p: _DummyProvider(_dummy_df()))


def test_register_non_str_name_raises(clean_registry):
    """name 非 str 时应抛 ValueError。"""
    with pytest.raises(ValueError, match="非空字符串"):
        _reg.register_signal_source(123, lambda p: _DummyProvider(_dummy_df()))  # type: ignore[arg-type]


def test_register_non_callable_factory_raises(clean_registry):
    """factory 非可调用时应抛 TypeError。"""
    with pytest.raises(TypeError):
        _reg.register_signal_source("bad_factory", "not_a_callable")  # type: ignore[arg-type]


# ──────────────────────────────────────────────
# 5. import rules 不触发 torch 加载
# ──────────────────────────────────────────────

def test_import_rules_no_torch():
    """import aitrade.rules（及其子模块）不应触发 torch 加载（懒加载保证）。

    策略：在子进程中 import aitrade.rules，检查 'torch' 是否出现在 sys.modules。
    用子进程而非操纵 sys.modules，彻底消除对父进程模块缓存的副作用，
    避免与其他测试产生执行顺序依赖（不同顺序下 sys.modules 状态不同）。
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import aitrade.rules, sys; sys.exit(1 if 'torch' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "import aitrade.rules 不应触发 torch 加载；"
        "请检查 rules/__init__.py 或 cnn_adapter.py 是否有模块级 torch import。"
        f"\nstderr: {result.stderr}"
    )


# ──────────────────────────────────────────────
# 6. "cnn" adapter：转调与 kwargs 透传（monkeypatch）
# ──────────────────────────────────────────────

def test_cnn_adapter_registered():
    """import aitrade.rules 后 'cnn' 应出现在信号源注册表中。"""
    import aitrade.rules  # noqa: F401  触发自注册

    names = [s["name"] for s in _reg.list_signal_sources()]
    assert "cnn" in names


def test_cnn_adapter_calls_predict_cnn_signals(monkeypatch):
    """build_signal_source('cnn', ...) 产出的 provider.predict() 应转调
    predict_cnn_signals 并把 model_name + start/end/on_progress 透传过去。"""
    import aitrade.rules  # noqa: F401

    expected_df = pl.DataFrame({
        "datetime": ["2024-01-02"],
        "vt_symbol": ["000001.SZSE"],
        "signal": [0.75],
    })

    captured: dict = {}

    def fake_predict_cnn_signals(
        model_name,
        start,
        end,
        on_progress=None,
        **kwargs,
    ) -> pl.DataFrame:
        captured["model_name"] = model_name
        captured["start"] = start
        captured["end"] = end
        captured["on_progress"] = on_progress
        captured["extra_kwargs"] = kwargs
        return expected_df

    # monkeypatch 替换 cnn_adapter 内的延迟 import 路径
    import aitrade.cnn.predictor as _predictor
    monkeypatch.setattr(_predictor, "predict_cnn_signals", fake_predict_cnn_signals)

    # 同时 patch rules.cnn_adapter 模块内引用（延迟 import 用 from ..cnn.predictor import 会重新查找）
    # 实际延迟 import 在函数体内执行 `from ..cnn.predictor import predict_cnn_signals`
    # 因此替换模块属性即可覆盖
    import aitrade.rules.cnn_adapter as _adapter_mod  # noqa: F401

    provider = _reg.build_signal_source("cnn", {"model_name": "my_model", "extra_arg": 42})

    start = date(2024, 1, 1)
    end = date(2024, 6, 30)
    sentinel_progress = object()

    result = provider.predict(start, end, on_progress=sentinel_progress)

    assert captured["model_name"] == "my_model"
    assert captured["start"] == start
    assert captured["end"] == end
    assert captured["on_progress"] is sentinel_progress
    assert captured["extra_kwargs"] == {"extra_arg": 42}
    assert result is expected_df


def test_cnn_adapter_returns_dataframe_unchanged(monkeypatch):
    """predict() 的返回值应与 predict_cnn_signals 返回值完全相同（原样透传）。"""
    import aitrade.rules  # noqa: F401

    marker_df = pl.DataFrame({"datetime": ["2024-03-01"], "vt_symbol": ["X"], "signal": [0.5]})

    import aitrade.cnn.predictor as _predictor
    monkeypatch.setattr(
        _predictor,
        "predict_cnn_signals",
        lambda **kwargs: marker_df,
    )

    provider = _reg.build_signal_source("cnn", {"model_name": "test_model"})
    result = provider.predict(date(2024, 1, 1), date(2024, 12, 31))
    assert result is marker_df


# ──────────────────────────────────────────────
# 7. model_name 缺失 → 构造期 ValueError（fail fast）
# ──────────────────────────────────────────────

def test_cnn_adapter_missing_model_name():
    """build_signal_source('cnn', {}) 不含 model_name 时应在构造期抛 ValueError。"""
    import aitrade.rules  # noqa: F401

    with pytest.raises(ValueError, match="model_name"):
        _reg.build_signal_source("cnn", {})


def test_cnn_adapter_empty_model_name():
    """build_signal_source('cnn', {'model_name': ''}) 空字符串也应抛 ValueError。"""
    import aitrade.rules  # noqa: F401

    with pytest.raises(ValueError, match="model_name"):
        _reg.build_signal_source("cnn", {"model_name": ""})


# ──────────────────────────────────────────────
# 8. list_signal_sources 的 "cnn" 条目结构
# ──────────────────────────────────────────────

def test_cnn_source_metadata():
    """list_signal_sources() 中 'cnn' 条目应包含 description 与 param_spec。"""
    import aitrade.rules  # noqa: F401

    sources = {s["name"]: s for s in _reg.list_signal_sources()}
    assert "cnn" in sources

    cnn = sources["cnn"]
    assert cnn["description"]  # 非空
    assert isinstance(cnn["param_spec"], dict)
    assert "model_name" in cnn["param_spec"]
    assert cnn["param_spec"]["model_name"].get("required") is True
