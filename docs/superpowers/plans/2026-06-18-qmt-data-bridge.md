# QMT 数据桥 实现计划（方案 A · remote-only）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Mac 上的 aitrade 研究系统，经一个跑在 Windows 上的瘦 REST 服务，用上只有 Windows 能取的 QMT（xtquant/xtdata）历史数据。

**Architecture:** 两个交付物。① Windows `qmt-bridge`：独立 FastAPI 服务，内部"薄路由 + 可测 xtdata 封装模块"，封装模块懒加载 `from xtquant import xtdata`、把数据归一化为 `BarRecord` 形状。② Mac `QmtBridgeProvider(BaseProvider)`：永远经 REST 调 ①，排进现有数据源优先级链最前，复用现有"下载→落 Parquet"管线；QMT 给不了的按类目自动落回 tushare。Mac 端永不 import xtquant。

**Tech Stack:** Python 3.10+、FastAPI、Polars、PyArrow（Arrow IPC）、httpx、pytest。设计依据见 `docs/superpowers/specs/2026-06-18-qmt-data-bridge-design.md`。

**开发/测试边界（关键）：**
- **Wave A（Windows 服务）与 Wave B（Mac Provider）都在 Mac 上用"假 xtdata / mock REST"开发并通过单测**。封装模块的 xtquant 是懒加载 + 可注入的，测试注入假对象（照 `tests/test_akshare_provider.py` 的 `_FakeAk` 范式）。
- **Wave C（真机验证）必须在你的 Windows 机器上跑**（连真 QMT），含 7 条 MEDIUM 置信项固化、复权金标对拍、端到端。
- 解释器：`backend/.venv/bin/python`（已含 pytest/polars/pyarrow/fastapi/httpx/pandas）。`qmt-bridge` 以 editable 方式装进该 venv 供 Mac 端开发：`backend/.venv/bin/pip install -e qmt-bridge`。
- 仓库放置：`qmt-bridge/` 放在 aitrade 仓库根（与 `backend/`、`frontend/` 同级），自带 `pyproject.toml`，**自包含、可独立拷到 Windows 部署**；版本与 aitrade 一起管理。

**契约常量（两端共用，逐字一致）：**
- `BARS_COLUMNS = ["symbol","exchange","datetime","interval","open_price","high_price","low_price","close_price","volume","turnover","open_interest","adjust_type"]`（== `BarRecord` 字段，Mac 端可 `BarRecord(**row)`）
- `EXCHANGE_TO_QMT = {"SSE":"SH","SZSE":"SZ","BSE":"BJ"}`，`QMT_TO_EXCHANGE` 为其反转
- `PERIOD_TO_QMT = {"d":"1d","1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","60m":"1h","w":"1w"}`
- `ADJUST_TO_DIVIDEND = {"none":"none","qfq":"front","hfq":"back"}`（等比口径变体 `front_ratio/back_ratio` 由配置开关切换）
- xtdata K 线请求字段：`["time","open","high","low","close","volume","amount","openInterest"]`；映射 `turnover←amount`、`open_interest←openInterest`、`datetime←time/1000`

---

## Wave A — Windows `qmt-bridge` 服务（Mac 上用假 xtdata 开发+单测）

### Task A0: 项目脚手架

**Files:**
- Create: `qmt-bridge/pyproject.toml`
- Create: `qmt-bridge/qmt_bridge/__init__.py`
- Create: `qmt-bridge/tests/__init__.py`
- Create: `qmt-bridge/README.md`

- [ ] **Step 1: 写 pyproject.toml**

```toml
[project]
name = "qmt-bridge"
version = "0.1.0"
description = "Windows-side thin REST bridge exposing QMT/xtdata to remote clients"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "polars>=1.26",
    "pyarrow>=19.0",
    "pandas>=2.0",
]
# 注意：xtquant 不在 deps —— Windows 部署时随 QMT 安装提供；Mac 开发时用假对象注入。

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 建空包文件**

`qmt-bridge/qmt_bridge/__init__.py`：

```python
"""qmt-bridge：跑在 Windows 上、把 QMT/xtdata 暴露成 REST 的瘦服务。"""

__version__ = "0.1.0"
```

`qmt-bridge/tests/__init__.py`：空文件。

- [ ] **Step 3: 装进 dev venv**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/pip install -e qmt-bridge`
Expected: `Successfully installed qmt-bridge-0.1.0`

- [ ] **Step 4: 冒烟**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -c "import qmt_bridge; print(qmt_bridge.__version__)"`
Expected: `0.1.0`

- [ ] **Step 5: README 写部署说明**

`qmt-bridge/README.md`：

```markdown
# qmt-bridge

跑在 **Windows** 上、连 QMT/miniQMT、把行情/基本面数据经 REST 暴露给 Mac 上的 aitrade。

## 部署（Windows）
1. 安装券商 QMT/miniQMT 客户端并登录（默认连接模式①），或配置独立 xtdatacenter（模式②，需迅投 VIP token）。
2. 把本目录拷到 Windows，确保 `xtquant` 在 PYTHONPATH（随 QMT 安装目录提供，或 `pip install xtquant`）。
3. `pip install -e .`，设置环境变量（见 qmt_bridge/config.py），`uvicorn qmt_bridge.app:app --host 0.0.0.0 --port 58610`。
4. **只在内网 / Tailscale 暴露，绝不开公网。**

## 开发（Mac）
不连真 QMT，用假 xtdata 单测：`backend/.venv/bin/python -m pytest qmt-bridge/tests -v`
```

- [ ] **Step 6: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/pyproject.toml qmt-bridge/qmt_bridge/__init__.py qmt-bridge/tests/__init__.py qmt-bridge/README.md
git commit -m "QMT 桥：Windows 瘦服务项目脚手架（Task A0）"
```

---

### Task A1: 契约常量与映射纯函数

**Files:**
- Create: `qmt-bridge/qmt_bridge/contract.py`
- Test: `qmt-bridge/tests/test_contract.py`

- [ ] **Step 1: 写失败测试**

`qmt-bridge/tests/test_contract.py`：

```python
"""契约常量与代码/周期/复权映射的单元测试。"""

import pytest

from qmt_bridge.contract import (
    BARS_COLUMNS,
    to_qmt_code,
    from_qmt_code,
    to_qmt_period,
    to_dividend_type,
)


def test_bars_columns_match_barrecord_fields():
    assert BARS_COLUMNS == [
        "symbol", "exchange", "datetime", "interval",
        "open_price", "high_price", "low_price", "close_price",
        "volume", "turnover", "open_interest", "adjust_type",
    ]


@pytest.mark.parametrize("symbol,exchange,expected", [
    ("600000", "SSE", "600000.SH"),
    ("000001", "SZSE", "000001.SZ"),
    ("430047", "BSE", "430047.BJ"),
])
def test_to_qmt_code(symbol, exchange, expected):
    assert to_qmt_code(symbol, exchange) == expected


@pytest.mark.parametrize("code,expected", [
    ("600000.SH", ("600000", "SSE")),
    ("000001.SZ", ("000001", "SZSE")),
    ("430047.BJ", ("430047", "BSE")),
])
def test_from_qmt_code(code, expected):
    assert from_qmt_code(code) == expected


@pytest.mark.parametrize("interval,expected", [
    ("d", "1d"), ("1m", "1m"), ("30m", "30m"),
    ("1h", "1h"), ("60m", "1h"), ("w", "1w"),
])
def test_to_qmt_period(interval, expected):
    assert to_qmt_period(interval) == expected


def test_to_qmt_period_rejects_unknown():
    with pytest.raises(ValueError):
        to_qmt_period("3m")


def test_to_dividend_type_default():
    assert to_dividend_type("none") == "none"
    assert to_dividend_type("qfq") == "front"
    assert to_dividend_type("hfq") == "back"


def test_to_dividend_type_ratio_mode():
    assert to_dividend_type("hfq", ratio=True) == "back_ratio"
    assert to_dividend_type("qfq", ratio=True) == "front_ratio"
    assert to_dividend_type("none", ratio=True) == "none"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_contract.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'qmt_bridge.contract'`）

- [ ] **Step 3: 实现 contract.py**

```python
"""两端共用的契约：列名、代码/周期/复权口径映射。

这些常量与 aitrade 后端 BarRecord 字段、交易所/周期约定逐字对齐，
是 Mac Provider 与 Windows 服务之间的接口契约。
"""

from __future__ import annotations

# 与 aitrade BarRecord 字段顺序一致，Mac 端可直接 BarRecord(**row)
BARS_COLUMNS = [
    "symbol", "exchange", "datetime", "interval",
    "open_price", "high_price", "low_price", "close_price",
    "volume", "turnover", "open_interest", "adjust_type",
]

# xtdata K 线请求字段
XTDATA_BAR_FIELDS = ["time", "open", "high", "low", "close", "volume", "amount", "openInterest"]

_EXCHANGE_TO_QMT = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}
_QMT_TO_EXCHANGE = {v: k for k, v in _EXCHANGE_TO_QMT.items()}

_PERIOD_TO_QMT = {
    "d": "1d", "1m": "1m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "60m": "1h", "w": "1w",
}

_ADJUST_TO_DIVIDEND = {"none": "none", "qfq": "front", "hfq": "back"}
_ADJUST_TO_DIVIDEND_RATIO = {"none": "none", "qfq": "front_ratio", "hfq": "back_ratio"}


def to_qmt_code(symbol: str, exchange: str) -> str:
    """把 aitrade 的 (symbol, exchange) 拼成 xtdata 合约码，如 ('600000','SSE')->'600000.SH'。"""
    suffix = _EXCHANGE_TO_QMT.get(exchange)
    if suffix is None:
        raise ValueError(f"不支持的交易所: {exchange}")
    return f"{symbol}.{suffix}"


def from_qmt_code(code: str) -> tuple[str, str]:
    """把 xtdata 合约码拆回 aitrade 形式，如 '600000.SH'->('600000','SSE')。"""
    sym, _, suffix = code.partition(".")
    exchange = _QMT_TO_EXCHANGE.get(suffix)
    if exchange is None:
        raise ValueError(f"不支持的 QMT 后缀: {code}")
    return sym, exchange


def to_qmt_period(interval: str) -> str:
    """aitrade 内部周期 -> xtdata period，如 'd'->'1d'、'60m'->'1h'。未知周期抛 ValueError。"""
    period = _PERIOD_TO_QMT.get(interval)
    if period is None:
        raise ValueError(f"不支持的周期: {interval}")
    return period


def to_dividend_type(adjust_type: str, *, ratio: bool = False) -> str:
    """adjust_type(none/qfq/hfq) -> xtdata dividend_type。ratio=True 走等比口径。"""
    table = _ADJUST_TO_DIVIDEND_RATIO if ratio else _ADJUST_TO_DIVIDEND
    if adjust_type not in table:
        raise ValueError(f"不支持的复权口径: {adjust_type}")
    return table[adjust_type]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_contract.py -v`
Expected: PASS（全部）

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/qmt_bridge/contract.py qmt-bridge/tests/test_contract.py
git commit -m "QMT 桥：契约常量与代码/周期/复权映射纯函数（Task A1）"
```

---

### Task A2: xtdata 封装模块 — K 线归一化

**Files:**
- Create: `qmt-bridge/qmt_bridge/xtdata_client.py`
- Test: `qmt-bridge/tests/test_xtdata_client_bars.py`

**关键约定：** `XtdataClient` 懒加载 `from xtquant import xtdata`，但支持构造时注入（`XtdataClient(xtdata=fake)`）以便脱机单测。`get_bars` 内部"先 download_history_data2 落地、再 get_market_data_ex 读取"，返回 `BARS_COLUMNS` 形状的 `list[dict]`。

- [ ] **Step 1: 写失败测试**

`qmt-bridge/tests/test_xtdata_client_bars.py`：

```python
"""xtdata 封装模块的 K 线归一化测试（注入假 xtdata，不连真 QMT）。"""

from datetime import datetime

import pandas as pd
import pytest

from qmt_bridge.xtdata_client import XtdataClient


class FakeXtdata:
    """模拟 xtquant.xtdata：记录调用、返回 get_market_data_ex 形状的数据。"""

    def __init__(self) -> None:
        self.download_calls = []

    def download_history_data2(self, stock_list, period, start_time="", end_time="", callback=None, incrementally=None):
        self.download_calls.append((tuple(stock_list), period, start_time, end_time))
        return None

    def get_market_data_ex(self, field_list, stock_list, period="1d", start_time="", end_time="",
                           count=-1, dividend_type="none", fill_data=True):
        code = stock_list[0]
        # index=时间, columns=字段；'time' 为毫秒时间戳
        df = pd.DataFrame(
            {
                "time": [1704211200000, 1704297600000],  # 2024-01-02, 2024-01-03 (UTC+8 当日)
                "open": [10.0, 10.5],
                "high": [11.0, 10.8],
                "low": [9.8, 10.2],
                "close": [10.5, 10.6],
                "volume": [1000.0, 1200.0],
                "amount": [10500.0, 12720.0],
                "openInterest": [0.0, 0.0],
            }
        )
        return {code: df}


def test_get_bars_two_step_and_normalize():
    fake = FakeXtdata()
    client = XtdataClient(xtdata=fake)
    rows = client.get_bars("600000", "SSE", "d", "20240101", "20240131", adjust_type="hfq")

    # 两步：先 download 再 get
    assert fake.download_calls == [(("600000.SH",), "1d", "20240101", "20240131")]

    assert len(rows) == 2
    first = rows[0]
    assert first["symbol"] == "600000"
    assert first["exchange"] == "SSE"
    assert first["interval"] == "d"
    assert first["open_price"] == 10.0
    assert first["close_price"] == 10.5
    assert first["turnover"] == 10500.0          # amount
    assert first["open_interest"] == 0.0          # openInterest
    assert first["adjust_type"] == "hfq"
    assert isinstance(first["datetime"], datetime)
    assert first["datetime"].year == 2024 and first["datetime"].month == 1 and first["datetime"].day == 2


def test_get_bars_passes_back_dividend_for_hfq():
    fake = FakeXtdata()
    captured = {}
    orig = fake.get_market_data_ex

    def spy(field_list, stock_list, **kw):
        captured.update(kw)
        return orig(field_list, stock_list, **kw)

    fake.get_market_data_ex = spy
    XtdataClient(xtdata=fake).get_bars("600000", "SSE", "d", "20240101", "20240131", adjust_type="hfq")
    assert captured["dividend_type"] == "back"


def test_get_bars_empty_returns_empty_list():
    class EmptyXt(FakeXtdata):
        def get_market_data_ex(self, *a, **k):
            return {"600000.SH": pd.DataFrame()}

    rows = XtdataClient(xtdata=EmptyXt()).get_bars("600000", "SSE", "d", "20240101", "20240131", adjust_type="hfq")
    assert rows == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_xtdata_client_bars.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'qmt_bridge.xtdata_client'`）

- [ ] **Step 3: 实现 xtdata_client.py（先到能过 K 线测试）**

```python
"""xtdata 封装模块：把 QMT 原始数据归一化为 aitrade 契约形状。

懒加载 xtquant，支持注入假对象以便脱机单测。所有"调 xtdata + 归一化 +
后缀/复权口径"逻辑集中在此，路由层只做序列化与鉴权。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .contract import (
    XTDATA_BAR_FIELDS,
    to_qmt_code,
    to_qmt_period,
    to_dividend_type,
)

# A 股交易时间按东八区
_CN_TZ = timezone(timedelta(hours=8))


def _ms_to_datetime(ms: int) -> datetime:
    """毫秒时间戳 -> 东八区 naive datetime（交易所本地时间）。"""
    return datetime.fromtimestamp(int(ms) / 1000, tz=_CN_TZ).replace(tzinfo=None)


class XtdataClient:
    """对 xtquant.xtdata 的薄封装。注入 xtdata=None 时懒加载真模块。"""

    def __init__(self, xtdata: Any = None, *, ratio_adjust: bool = False) -> None:
        self._xt = xtdata
        self._ratio = ratio_adjust

    @property
    def xt(self) -> Any:
        """懒加载真 xtquant.xtdata（Windows 上首次调用时）。"""
        if self._xt is None:
            from xtquant import xtdata  # type: ignore
            self._xt = xtdata
        return self._xt

    def get_bars(self, symbol: str, exchange: str, interval: str,
                 start: str, end: str, *, adjust_type: str = "hfq") -> list[dict]:
        """先 download 落地、再 get 读取，归一化为 BARS_COLUMNS 形状的 list[dict]。

        Args:
            symbol/exchange: aitrade 形式，如 ('600000','SSE')。
            interval: aitrade 内部周期，如 'd'/'1m'/'60m'。
            start/end: 'YYYYMMDD'（日线）或 14 位 'YYYYMMDDHHMMSS'（分钟）。
            adjust_type: 'none'/'qfq'/'hfq'，默认 'hfq'。

        Returns:
            list[dict]，每个含 BARS_COLUMNS 全部键，按时间升序。无数据返回 []。
        """
        code = to_qmt_code(symbol, exchange)
        period = to_qmt_period(interval)
        dividend = to_dividend_type(adjust_type, ratio=self._ratio)

        self.xt.download_history_data2([code], period, start, end)
        data = self.xt.get_market_data_ex(
            XTDATA_BAR_FIELDS, [code], period, start, end, -1, dividend, True
        )
        df = data.get(code)
        if df is None or len(df) == 0:
            return []

        rows: list[dict] = []
        for rec in df.to_dict(orient="records"):
            rows.append({
                "symbol": symbol,
                "exchange": exchange,
                "datetime": _ms_to_datetime(rec["time"]),
                "interval": interval,
                "open_price": float(rec["open"]),
                "high_price": float(rec["high"]),
                "low_price": float(rec["low"]),
                "close_price": float(rec["close"]),
                "volume": float(rec.get("volume", 0.0)),
                "turnover": float(rec.get("amount", 0.0)),
                "open_interest": float(rec.get("openInterest", 0.0)),
                "adjust_type": adjust_type,
            })
        rows.sort(key=lambda r: r["datetime"])
        return rows
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_xtdata_client_bars.py -v`
Expected: PASS（3 个）

> ⚠️ **真机待核（Wave C）**：`get_market_data_ex` 的实际签名/返回维度、`'time'` 时区是 MEDIUM 置信项。本任务按"index=时间、含 'time' 列、东八区毫秒"实现；C1 在 Windows 上 `print` 确认后如有偏差回这里改归一化。

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/qmt_bridge/xtdata_client.py qmt-bridge/tests/test_xtdata_client_bars.py
git commit -m "QMT 桥：xtdata 封装模块 K 线归一化（两步取数+复权口径，Task A2）"
```

---

### Task A3: 复权因子 — get_divid_factors 的 dr 累乘

**Files:**
- Modify: `qmt-bridge/qmt_bridge/xtdata_client.py`
- Test: `qmt-bridge/tests/test_xtdata_client_adj.py`

- [ ] **Step 1: 写失败测试**

`qmt-bridge/tests/test_xtdata_client_adj.py`：

```python
"""复权因子累乘测试：get_divid_factors 的 dr -> 后复权累积因子。"""

import pandas as pd

from qmt_bridge.xtdata_client import XtdataClient


class FakeXtAdj:
    def get_divid_factors(self, stock_code, start_time="", end_time=""):
        # 两个除权日，dr 分别 1.10、1.05；index 为除权日
        return pd.DataFrame(
            {"dr": [1.10, 1.05]},
            index=["20240110", "20240620"],
        )


def test_adj_factor_cumulative_product():
    rows = XtdataClient(xtdata=FakeXtAdj()).get_adj_factor("600000", "SSE", "20240101", "20241231")
    # 后复权累积因子：从 1.0 起，遇除权日乘 dr
    assert rows == [
        {"trade_date": "20240110", "adj_factor": 1.10},
        {"trade_date": "20240620", "adj_factor": round(1.10 * 1.05, 6)},
    ]


def test_adj_factor_no_dividends_returns_empty():
    class Empty:
        def get_divid_factors(self, *a, **k):
            return pd.DataFrame()

    assert XtdataClient(xtdata=Empty()).get_adj_factor("600000", "SSE", "", "") == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_xtdata_client_adj.py -v`
Expected: FAIL（`AttributeError: 'XtdataClient' object has no attribute 'get_adj_factor'`）

- [ ] **Step 3: 加 get_adj_factor 到 xtdata_client.py**

在 `XtdataClient` 类内追加：

```python
    def get_adj_factor(self, symbol: str, exchange: str,
                       start: str = "", end: str = "") -> list[dict]:
        """用 get_divid_factors 的 dr 累乘出后复权累积因子。

        Returns:
            list[{'trade_date': 'YYYYMMDD', 'adj_factor': float}]，按日升序；
            adj_factor 从 1.0 起、每个除权日乘当日 dr，单调 >=1。无除权返回 []。
        """
        code = to_qmt_code(symbol, exchange)
        df = self.xt.get_divid_factors(code, start, end)
        if df is None or len(df) == 0:
            return []

        out: list[dict] = []
        factor = 1.0
        for trade_date, rec in df.sort_index().iterrows():
            factor = round(factor * float(rec["dr"]), 6)
            out.append({"trade_date": str(trade_date), "adj_factor": factor})
        return out
```

> 注：`to_qmt_code` 已在文件顶部 import；无需重复导入。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_xtdata_client_adj.py -v`
Expected: PASS（2 个）

> ⚠️ **真机待核（Wave C2）**：dr 累乘的方向与边界（除权日用当日 dr 还是次日）是 LOW/MEDIUM 置信。C2 用已知分红+送转个股金标对拍，若边界不符回此修正。

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/qmt_bridge/xtdata_client.py qmt-bridge/tests/test_xtdata_client_adj.py
git commit -m "QMT 桥：复权因子 dr 累乘生成后复权累积因子（Task A3）"
```

---

### Task A4: 合约 / 日历 / 基本面 归一化

**Files:**
- Modify: `qmt-bridge/qmt_bridge/xtdata_client.py`
- Test: `qmt-bridge/tests/test_xtdata_client_meta.py`

- [ ] **Step 1: 写失败测试**

`qmt-bridge/tests/test_xtdata_client_meta.py`：

```python
"""合约/交易日历/基本面归一化测试。"""

import pandas as pd

from qmt_bridge.xtdata_client import XtdataClient


class FakeXtMeta:
    def download_sector_data(self):
        return None

    def get_stock_list_in_sector(self, sector_name, real_timetag=None):
        assert sector_name in ("沪深A股", "沪深京A股")
        return ["600000.SH", "000001.SZ"]

    def get_instrument_detail(self, stock_code, iscomplete=False):
        return {
            "InstrumentName": "浦发银行" if stock_code.startswith("600000") else "平安银行",
            "OpenDate": "19991110",
            "ExpireDate": "",
            "PriceTick": 0.01,
            "VolumeMultiple": 1,
            "InstrumentStatus": 0,
            "IsTrading": True,
        }

    def get_instrument_type(self, stock_code):
        return {"stock": True}

    def download_holiday_data(self):
        return None

    def get_trading_calendar(self, market, start_time="", end_time=""):
        return ["20240102", "20240103"]

    def download_financial_data2(self, stock_list, table_list=None, start_time="", end_time="", callback=None):
        return None

    def get_financial_data(self, stock_list, table_list=None, start_time="", end_time="", report_type="report_time"):
        assert report_type == "announce_time"  # 防未来函数
        df = pd.DataFrame({"m_timetag": ["20231231"], "m_anntime": ["20240328"], "tot_assets": [1.0e12]})
        return {stock_list[0]: {"Balance": df}}


def test_list_contracts_normalizes_suffix_and_fields():
    rows = XtdataClient(xtdata=FakeXtMeta()).get_contracts(include_bse=False)
    assert {r["symbol"] for r in rows} == {"600000", "000001"}
    pf = next(r for r in rows if r["symbol"] == "600000")
    assert pf["exchange"] == "SSE"
    assert pf["name"] == "浦发银行"
    assert pf["list_date"] == "19991110"
    assert pf["pricetick"] == 0.01


def test_trading_calendar():
    rows = XtdataClient(xtdata=FakeXtMeta()).get_trade_calendar("SSE", "20240101", "20240105")
    assert rows == [
        {"date": "20240102", "exchange": "SSE", "is_open": True},
        {"date": "20240103", "exchange": "SSE", "is_open": True},
    ]


def test_fundamental_uses_announce_time():
    rows = XtdataClient(xtdata=FakeXtMeta()).get_fundamental("600000", "SSE", "20230101", "20240401")
    assert rows[0]["symbol"] == "600000"
    assert rows[0]["report_period"] == "20231231"
    assert rows[0]["ann_date"] == "20240328"
    assert rows[0]["table"] == "Balance"
    assert rows[0]["fields"]["tot_assets"] == 1.0e12
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_xtdata_client_meta.py -v`
Expected: FAIL（缺 `get_contracts`/`get_trade_calendar`/`get_fundamental`）

- [ ] **Step 3: 加三个方法到 xtdata_client.py**

文件顶部 import 增加 `from .contract import from_qmt_code, _EXCHANGE_TO_QMT`（若 `_EXCHANGE_TO_QMT` 未导出，改为在 contract.py 暴露一个 `exchange_to_market(exchange)->str` 辅助；这里用它）。先在 `contract.py` 末尾加：

```python
def exchange_to_market(exchange: str) -> str:
    """aitrade 交易所 -> xtdata 日历/市场码，如 'SSE'->'SH'。"""
    m = _EXCHANGE_TO_QMT.get(exchange)
    if m is None:
        raise ValueError(f"不支持的交易所: {exchange}")
    return m
```

再在 `xtdata_client.py` 顶部 import 改为：

```python
from .contract import (
    XTDATA_BAR_FIELDS,
    to_qmt_code,
    from_qmt_code,
    to_qmt_period,
    to_dividend_type,
    exchange_to_market,
)
```

在 `XtdataClient` 内追加：

```python
    def get_contracts(self, *, include_bse: bool = False) -> list[dict]:
        """枚举沪深(可含北交所)A股并补详情，归一为 ContractInfo 形状的 list[dict]。"""
        self.xt.download_sector_data()
        sector = "沪深京A股" if include_bse else "沪深A股"
        codes = self.xt.get_stock_list_in_sector(sector) or []
        out: list[dict] = []
        for code in codes:
            symbol, exchange = from_qmt_code(code)
            detail = self.xt.get_instrument_detail(code, False) or {}
            types = self.xt.get_instrument_type(code) or {}
            product = "股票" if types.get("stock") else (
                "指数" if types.get("index") else ("基金" if types.get("fund") else ""))
            out.append({
                "symbol": symbol,
                "exchange": exchange,
                "name": detail.get("InstrumentName", ""),
                "product_type": product,
                "size": float(detail.get("VolumeMultiple", 1) or 1),
                "pricetick": float(detail.get("PriceTick", 0.01) or 0.01),
                "list_date": str(detail.get("OpenDate", "") or ""),
                "delist_date": str(detail.get("ExpireDate", "") or ""),
                "extra": {
                    "instrument_status": detail.get("InstrumentStatus"),
                    "is_trading": detail.get("IsTrading"),
                },
            })
        return out

    def get_trade_calendar(self, exchange: str, start: str, end: str) -> list[dict]:
        """交易日历，归一为 CalendarDay 形状的 list[dict]（仅含交易日，is_open=True）。"""
        self.xt.download_holiday_data()
        market = exchange_to_market(exchange)
        days = self.xt.get_trading_calendar(market, start, end) or []
        return [{"date": str(d), "exchange": exchange, "is_open": True} for d in days]

    def get_fundamental(self, symbol: str, exchange: str, start: str, end: str) -> list[dict]:
        """财务数据，report_type 固定 announce_time 防未来函数。

        Returns:
            list[dict]，每条形如
            {'symbol','exchange','table','report_period','ann_date','fields': {...}}。
        """
        code = to_qmt_code(symbol, exchange)
        self.xt.download_financial_data2([code], start_time=start, end_time=end)
        data = self.xt.get_financial_data([code], start_time=start, end_time=end,
                                          report_type="announce_time")
        tables = data.get(code, {})
        out: list[dict] = []
        for table_name, df in tables.items():
            if df is None or len(df) == 0:
                continue
            for rec in df.to_dict(orient="records"):
                fields = {k: v for k, v in rec.items() if k not in ("m_timetag", "m_anntime")}
                out.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "table": table_name,
                    "report_period": str(rec.get("m_timetag", "")),
                    "ann_date": str(rec.get("m_anntime", "")),
                    "fields": fields,
                })
        return out

    def is_connected(self) -> bool:
        """QMT/xtdata 连接是否在线（驱动 /health）。"""
        try:
            return bool(self.xt.get_client().is_connected())
        except Exception:
            return False
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_xtdata_client_meta.py -v`
Expected: PASS（3 个）

> ⚠️ **真机待核（Wave C1）**：财务表名大小写、`get_financial_data` 返回结构、`OpenDate/ExpireDate` 格式、`沪深A股` 成分范围、`get_instrument_type` 键全集。

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/qmt_bridge/xtdata_client.py qmt-bridge/qmt_bridge/contract.py qmt-bridge/tests/test_xtdata_client_meta.py
git commit -m "QMT 桥：合约/日历/基本面归一化（announce_time 防未来函数，Task A4）"
```

---

### Task A5: 序列化 — bars→Arrow IPC，元数据→JSON

**Files:**
- Create: `qmt-bridge/qmt_bridge/serialize.py`
- Test: `qmt-bridge/tests/test_serialize.py`

- [ ] **Step 1: 写失败测试**

`qmt-bridge/tests/test_serialize.py`：

```python
"""bars 列式序列化 round-trip 测试。"""

from datetime import datetime

import polars as pl

from qmt_bridge.contract import BARS_COLUMNS
from qmt_bridge.serialize import bars_to_ipc


def test_bars_to_ipc_roundtrip():
    rows = [{
        "symbol": "600000", "exchange": "SSE",
        "datetime": datetime(2024, 1, 2), "interval": "d",
        "open_price": 10.0, "high_price": 11.0, "low_price": 9.8, "close_price": 10.5,
        "volume": 1000.0, "turnover": 10500.0, "open_interest": 0.0, "adjust_type": "hfq",
    }]
    blob = bars_to_ipc(rows)
    assert isinstance(blob, (bytes, bytearray))

    df = pl.read_ipc_stream(blob)
    assert df.columns == BARS_COLUMNS
    assert df.height == 1
    assert df["close_price"][0] == 10.5
    assert df["symbol"][0] == "600000"


def test_bars_to_ipc_empty():
    blob = bars_to_ipc([])
    df = pl.read_ipc_stream(blob)
    assert df.columns == BARS_COLUMNS
    assert df.height == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_serialize.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'qmt_bridge.serialize'`）

- [ ] **Step 3: 实现 serialize.py**

```python
"""把归一化行情序列化为 Arrow IPC stream（zstd）供远端零转码落盘。"""

from __future__ import annotations

import polars as pl

from .contract import BARS_COLUMNS

# 空数据时也要保证 schema 稳定
_EMPTY_SCHEMA = {
    "symbol": pl.Utf8, "exchange": pl.Utf8, "datetime": pl.Datetime, "interval": pl.Utf8,
    "open_price": pl.Float64, "high_price": pl.Float64, "low_price": pl.Float64,
    "close_price": pl.Float64, "volume": pl.Float64, "turnover": pl.Float64,
    "open_interest": pl.Float64, "adjust_type": pl.Utf8,
}


def bars_to_ipc(rows: list[dict]) -> bytes:
    """list[dict]（BARS_COLUMNS 形状）-> Arrow IPC stream 字节（zstd）。"""
    if rows:
        df = pl.DataFrame(rows).select(BARS_COLUMNS)
    else:
        df = pl.DataFrame(schema=_EMPTY_SCHEMA)
    buf = df.write_ipc_stream(None, compression="zstd")
    return buf.getvalue()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_serialize.py -v`
Expected: PASS（2 个）

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/qmt_bridge/serialize.py qmt-bridge/tests/test_serialize.py
git commit -m "QMT 桥：bars 列式 Arrow IPC 序列化（Task A5）"
```

---

### Task A6: 配置 + 鉴权依赖

**Files:**
- Create: `qmt-bridge/qmt_bridge/config.py`
- Create: `qmt-bridge/qmt_bridge/auth.py`
- Test: `qmt-bridge/tests/test_auth.py`

- [ ] **Step 1: 写失败测试**

`qmt-bridge/tests/test_auth.py`：

```python
"""bearer 鉴权依赖测试。"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from qmt_bridge.auth import make_token_guard


def _cred(tok):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok)


def test_correct_token_passes():
    guard = make_token_guard("s3cret")
    guard(_cred("s3cret"))  # 不抛即通过


def test_wrong_token_401():
    guard = make_token_guard("s3cret")
    with pytest.raises(HTTPException) as ei:
        guard(_cred("nope"))
    assert ei.value.status_code == 401
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_auth.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'qmt_bridge.auth'`）

- [ ] **Step 3: 实现 config.py + auth.py**

`qmt-bridge/qmt_bridge/config.py`：

```python
"""qmt-bridge 配置（环境变量）。"""

from __future__ import annotations

import os

# REST 鉴权 token（Mac↔Win）
BRIDGE_TOKEN = os.getenv("QMT_BRIDGE_TOKEN", "")
# 监听
HOST = os.getenv("QMT_BRIDGE_HOST", "0.0.0.0")
PORT = int(os.getenv("QMT_BRIDGE_PORT", "58610"))
# 连接模式：'client'=连券商 QMT 客户端（默认）；'xtdc'=独立 xtdatacenter
CONNECT_MODE = os.getenv("QMT_BRIDGE_CONNECT_MODE", "client")
# 模式②用：迅投 VIP token
XTDC_TOKEN = os.getenv("QMT_XTDC_TOKEN", "")
# 复权口径：等比 or 普通
RATIO_ADJUST = os.getenv("QMT_BRIDGE_RATIO_ADJUST", "false").lower() in ("1", "true")
```

`qmt-bridge/qmt_bridge/auth.py`：

```python
"""bearer token 鉴权依赖（常量时间比对）。"""

from __future__ import annotations

import secrets
from typing import Callable

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


def make_token_guard(expected: str) -> Callable[[HTTPAuthorizationCredentials], None]:
    """生成一个校验函数：token 不匹配抛 401。常量时间比对防时序攻击。"""

    def guard(cred: HTTPAuthorizationCredentials) -> None:
        if not expected or not secrets.compare_digest(cred.credentials, expected):
            raise HTTPException(status_code=401, detail="invalid bridge token")

    return guard
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_auth.py -v`
Expected: PASS（2 个）

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/qmt_bridge/config.py qmt-bridge/qmt_bridge/auth.py qmt-bridge/tests/test_auth.py
git commit -m "QMT 桥：配置 + bearer 鉴权依赖（Task A6）"
```

---

### Task A7: FastAPI 应用 + 路由 + /health（单 worker 串行）

**Files:**
- Create: `qmt-bridge/qmt_bridge/app.py`
- Test: `qmt-bridge/tests/test_app.py`

**约定：** app 通过 `app.state.client`（一个 `XtdataClient`）取数；测试用 `app.dependency_overrides` 或直接替换 `app.state.client` 注入假 client。下载/取数经一个**单线程串行锁**保护（MiniQmt 单连接）。

- [ ] **Step 1: 写失败测试**

`qmt-bridge/tests/test_app.py`：

```python
"""FastAPI 路由集成测试（TestClient + 假 XtdataClient，不连真 QMT）。"""

import polars as pl
from datetime import datetime
from fastapi.testclient import TestClient

from qmt_bridge.app import create_app
from qmt_bridge.contract import BARS_COLUMNS

TOKEN = "t0ken"
H = {"Authorization": f"Bearer {TOKEN}"}


class FakeClient:
    def is_connected(self):
        return True

    def get_bars(self, symbol, exchange, interval, start, end, *, adjust_type="hfq"):
        return [{
            "symbol": symbol, "exchange": exchange, "datetime": datetime(2024, 1, 2),
            "interval": interval, "open_price": 10.0, "high_price": 11.0,
            "low_price": 9.8, "close_price": 10.5, "volume": 1000.0,
            "turnover": 10500.0, "open_interest": 0.0, "adjust_type": adjust_type,
        }]

    def get_contracts(self, *, include_bse=False):
        return [{"symbol": "600000", "exchange": "SSE", "name": "浦发银行",
                 "product_type": "股票", "size": 1.0, "pricetick": 0.01,
                 "list_date": "19991110", "delist_date": "", "extra": {}}]

    def get_trade_calendar(self, exchange, start, end):
        return [{"date": "20240102", "exchange": exchange, "is_open": True}]

    def get_adj_factor(self, symbol, exchange, start="", end=""):
        return [{"trade_date": "20240110", "adj_factor": 1.1}]

    def get_fundamental(self, symbol, exchange, start, end):
        return [{"symbol": symbol, "exchange": exchange, "table": "Balance",
                 "report_period": "20231231", "ann_date": "20240328", "fields": {}}]


def _client():
    app = create_app(client=FakeClient(), token=TOKEN)
    return TestClient(app)


def test_health_no_auth_required():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["connected"] is True


def test_bars_returns_arrow():
    r = _client().post("/bars", json={
        "symbol": "600000", "exchange": "SSE", "interval": "d",
        "start": "20240101", "end": "20240131", "adjust_type": "hfq",
    }, headers=H)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/vnd.apache.arrow.stream"
    df = pl.read_ipc_stream(r.content)
    assert df.columns == BARS_COLUMNS
    assert df["close_price"][0] == 10.5


def test_bars_requires_token():
    r = _client().post("/bars", json={
        "symbol": "600000", "exchange": "SSE", "interval": "d",
        "start": "20240101", "end": "20240131", "adjust_type": "hfq",
    })  # 无 Authorization
    assert r.status_code in (401, 403)


def test_contracts_json():
    r = _client().get("/contracts", headers=H)
    assert r.status_code == 200
    assert r.json()[0]["symbol"] == "600000"


def test_calendar_and_adj_and_fundamental():
    c = _client()
    assert c.get("/trading_calendar", params={"exchange": "SSE", "start": "20240101", "end": "20240105"}, headers=H).json()[0]["date"] == "20240102"
    assert c.get("/adj_factor", params={"symbol": "600000", "exchange": "SSE"}, headers=H).json()[0]["adj_factor"] == 1.1
    assert c.get("/fundamental", params={"symbol": "600000", "exchange": "SSE", "start": "20230101", "end": "20240401"}, headers=H).json()[0]["table"] == "Balance"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_app.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'qmt_bridge.app'`）

- [ ] **Step 3: 实现 app.py**

```python
"""qmt-bridge FastAPI 应用：把 XtdataClient 投影成 REST 端点。

取数经单线程串行锁保护（MiniQmt 单连接，禁并发 download）。
"""

from __future__ import annotations

import threading
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Query, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from . import config
from .auth import make_token_guard
from .serialize import bars_to_ipc
from .xtdata_client import XtdataClient


class BarsRequest(BaseModel):
    symbol: str
    exchange: str
    interval: str
    start: str
    end: str
    adjust_type: str = "hfq"


def create_app(client: Any = None, token: str | None = None) -> FastAPI:
    """构造 app。client/token 不传时从 config + 真 xtquant 装配（Windows 用）。"""
    app = FastAPI(title="qmt-bridge", version="0.1.0")
    app.state.client = client if client is not None else XtdataClient(ratio_adjust=config.RATIO_ADJUST)
    app.state.lock = threading.Lock()

    expected = token if token is not None else config.BRIDGE_TOKEN
    _guard = make_token_guard(expected)
    bearer = HTTPBearer()

    def require_token(cred: Annotated[HTTPAuthorizationCredentials, Depends(bearer)]) -> None:
        _guard(cred)

    Auth = Depends(require_token)

    @app.get("/health")
    def health() -> dict:
        return {
            "connected": bool(app.state.client.is_connected()),
            "version": app.version,
        }

    @app.post("/bars")
    def bars(req: BarsRequest, _: Annotated[None, Auth]) -> Response:
        with app.state.lock:
            rows = app.state.client.get_bars(
                req.symbol, req.exchange, req.interval, req.start, req.end,
                adjust_type=req.adjust_type,
            )
        return Response(content=bars_to_ipc(rows),
                        media_type="application/vnd.apache.arrow.stream")

    @app.get("/contracts")
    def contracts(_: Annotated[None, Auth],
                  include_bse: bool = Query(default=False)) -> list[dict]:
        with app.state.lock:
            return app.state.client.get_contracts(include_bse=include_bse)

    @app.get("/trading_calendar")
    def trading_calendar(_: Annotated[None, Auth],
                         exchange: str = Query(...), start: str = Query(...),
                         end: str = Query(...)) -> list[dict]:
        with app.state.lock:
            return app.state.client.get_trade_calendar(exchange, start, end)

    @app.get("/adj_factor")
    def adj_factor(_: Annotated[None, Auth],
                   symbol: str = Query(...), exchange: str = Query(...),
                   start: str = Query(default=""), end: str = Query(default="")) -> list[dict]:
        with app.state.lock:
            return app.state.client.get_adj_factor(symbol, exchange, start, end)

    @app.get("/fundamental")
    def fundamental(_: Annotated[None, Auth],
                    symbol: str = Query(...), exchange: str = Query(...),
                    start: str = Query(...), end: str = Query(...)) -> list[dict]:
        with app.state.lock:
            return app.state.client.get_fundamental(symbol, exchange, start, end)

    return app


app = create_app()
```

> 注：模块顶层 `app = create_app()` 在 Mac 上会构造一个 `XtdataClient()`（xtdata 懒加载，未触发 import），仅用于 `uvicorn qmt_bridge.app:app`。测试一律用 `create_app(client=Fake, token=...)`，不碰顶层 app。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests/test_app.py -v`
Expected: PASS（6 个）

- [ ] **Step 5: 跑整个 Wave A 回归**

Run: `cd /Users/kevin_1/aitrade && backend/.venv/bin/python -m pytest qmt-bridge/tests -v`
Expected: 全 PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/kevin_1/aitrade
git add qmt-bridge/qmt_bridge/app.py qmt-bridge/tests/test_app.py
git commit -m "QMT 桥：FastAPI 路由 + /health + 单 worker 串行（Task A7）"
```

---

## Wave B — Mac `QmtBridgeProvider`（mock REST 单测）

### Task B0: 配置项

**Files:**
- Modify: `backend/aitrade/config.py`（在 Tushare/AKShare 段落附近）
- Test: `backend/tests/test_qmt_bridge_config.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_qmt_bridge_config.py`：

```python
"""QMT 桥配置项存在性测试。"""

import importlib


def test_qmt_bridge_config_defaults(monkeypatch):
    monkeypatch.delenv("QMT_BRIDGE_URL", raising=False)
    monkeypatch.delenv("QMT_BRIDGE_TOKEN", raising=False)
    import aitrade.config as cfg
    importlib.reload(cfg)
    assert cfg.QMT_BRIDGE_URL == ""
    assert cfg.QMT_BRIDGE_TOKEN == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_config.py -v`
Expected: FAIL（`AttributeError: module 'aitrade.config' has no attribute 'QMT_BRIDGE_URL'`）

- [ ] **Step 3: 加配置（照 TUSHARE_TOKEN 风格）**

在 `backend/aitrade/config.py` 的数据源段落追加：

```python
# QMT 数据桥（Mac↔Windows）
QMT_BRIDGE_URL = os.getenv("QMT_BRIDGE_URL", "")
QMT_BRIDGE_TOKEN = os.getenv("QMT_BRIDGE_TOKEN", "")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add backend/aitrade/config.py backend/tests/test_qmt_bridge_config.py
git commit -m "QMT 桥：Mac 端新增 QMT_BRIDGE_URL/TOKEN 配置（Task B0）"
```

---

### Task B1: QmtBridgeProvider — 状态机 + get_bar_history

**Files:**
- Create: `backend/aitrade/datasource/qmt_bridge_provider.py`
- Test: `backend/tests/test_qmt_bridge_provider.py`

**约定：** `init()` 懒加载 `httpx` 并建一个 `httpx.Client(base_url=..., headers={Authorization})`；测试注入 `provider._http = FakeHttp()`（照 `_FakeAk` 范式），不联网。`get_bar_history` 调 `/bars` 拿 Arrow → `pl.read_ipc_stream` → 逐行 `BarRecord(**row)`；空数据返回 `None`（让 manager 回退），HTTP 错误 `raise`。

- [ ] **Step 1: 写失败测试**

`backend/tests/test_qmt_bridge_provider.py`：

```python
"""QmtBridgeProvider 单元测试（注入假 httpx，不联网）。"""

from datetime import datetime

import polars as pl
import pytest

from aitrade.datasource.qmt_bridge_provider import QmtBridgeProvider
from aitrade.datasource.types import BarRecord, DataCategory, ProviderStatus


BARS_COLUMNS = [
    "symbol", "exchange", "datetime", "interval",
    "open_price", "high_price", "low_price", "close_price",
    "volume", "turnover", "open_interest", "adjust_type",
]


def _arrow_one_bar() -> bytes:
    df = pl.DataFrame([{
        "symbol": "600000", "exchange": "SSE", "datetime": datetime(2024, 1, 2),
        "interval": "d", "open_price": 10.0, "high_price": 11.0, "low_price": 9.8,
        "close_price": 10.5, "volume": 1000.0, "turnover": 10500.0,
        "open_interest": 0.0, "adjust_type": "hfq",
    }]).select(BARS_COLUMNS)
    return df.write_ipc_stream(None, compression="zstd").getvalue()


class _FakeResp:
    def __init__(self, status_code=200, content=b"", json_data=None):
        self.status_code = status_code
        self.content = content
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttp:
    """模拟 httpx.Client：按 path 返回预置响应。"""

    def __init__(self, responses: dict):
        self._responses = responses
        self.calls = []

    def post(self, path, json=None):
        self.calls.append(("POST", path, json))
        return self._responses[path]

    def get(self, path, params=None):
        self.calls.append(("GET", path, params))
        return self._responses[path]


def _provider(responses, url="http://win:58610", token="t"):
    p = QmtBridgeProvider(url=url, token=token)
    p._http = _FakeHttp(responses)
    p._inited = True
    return p


def test_status_not_configured_without_url():
    p = QmtBridgeProvider(url="", token="")
    assert p.get_status() == ProviderStatus.NOT_CONFIGURED


def test_supported_categories():
    p = QmtBridgeProvider(url="http://win", token="t")
    cats = p.get_supported_categories()
    assert DataCategory.BAR_HISTORY in cats
    assert DataCategory.REFERENCE in cats  # adj factor


def test_get_bar_history_decodes_arrow_to_barrecord():
    p = _provider({"/bars": _FakeResp(content=_arrow_one_bar())})
    bars = p.get_bar_history("600000", "SSE", "d", datetime(2024, 1, 1), datetime(2024, 1, 31))
    assert isinstance(bars, list) and len(bars) == 1
    b = bars[0]
    assert isinstance(b, BarRecord)
    assert b.symbol == "600000" and b.exchange == "SSE"
    assert b.close_price == 10.5
    assert b.adjust_type == "hfq"
    # 请求体携带映射后的参数
    assert p._http.calls[0] == ("POST", "/bars", {
        "symbol": "600000", "exchange": "SSE", "interval": "d",
        "start": "20240101", "end": "20240131", "adjust_type": "hfq",
    })


def test_get_bar_history_empty_returns_none():
    empty = pl.DataFrame(schema={c: (pl.Utf8 if c in ("symbol","exchange","interval","adjust_type")
                                      else pl.Datetime if c == "datetime" else pl.Float64)
                                  for c in BARS_COLUMNS}).write_ipc_stream(None, compression="zstd").getvalue()
    p = _provider({"/bars": _FakeResp(content=empty)})
    assert p.get_bar_history("600000", "SSE", "d", datetime(2024, 1, 1)) is None


def test_get_bar_history_http_error_raises():
    p = _provider({"/bars": _FakeResp(status_code=500)})
    with pytest.raises(RuntimeError):
        p.get_bar_history("600000", "SSE", "d", datetime(2024, 1, 1))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_provider.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'aitrade.datasource.qmt_bridge_provider'`）

- [ ] **Step 3: 实现 qmt_bridge_provider.py（先到能过 bars）**

```python
"""QmtBridgeProvider：经 REST 调用 Windows 上的 qmt-bridge 服务取 QMT 数据。

Mac 端永不 import xtquant。懒加载 httpx；无数据返回 None（让 manager 回退），
HTTP/网络错误 raise（provider_name 锁定时原样上抛，区分"真错"vs"无数据"）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

import polars as pl

from .base import BaseProvider
from .types import BarRecord, DataCategory, ProviderStatus

# 与 qmt-bridge 契约一致
_EXCHANGE_TO_QMT = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}
_PERIOD_MAP = {"d": "d", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
               "1h": "1h", "60m": "1h", "w": "w"}


def _fmt_date(d: datetime) -> str:
    """datetime -> 'YYYYMMDD'。"""
    return d.strftime("%Y%m%d")


class QmtBridgeProvider(BaseProvider):
    """通过 REST 桥使用 QMT 数据的数据源。"""

    name = "qmt"
    display_name = "QMT 数据桥"
    description = "经 Windows 上的 qmt-bridge 服务使用 QMT/xtdata 数据"

    def __init__(self, url: str = "", token: str = "") -> None:
        self._url = url
        self._token = token
        self._http: Any = None
        self._inited = False

    def init(self, output: Callable = print) -> bool:
        if not self._url:
            output("[qmt] 未配置 QMT_BRIDGE_URL，跳过")
            return False
        try:
            import httpx  # 懒加载
            self._http = httpx.Client(
                base_url=self._url,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=60.0,
            )
            resp = self._http.get("/health")
            resp.raise_for_status()
            self._inited = bool(resp.json().get("connected"))
            output(f"[qmt] 桥连接 {'正常' if self._inited else '在线但 QMT 未连'}")
            return self._inited
        except Exception as exc:  # 软失败
            output(f"[qmt] 初始化失败: {exc}")
            self._inited = False
            return False

    def get_status(self) -> ProviderStatus:
        if not self._url:
            return ProviderStatus.NOT_CONFIGURED
        return ProviderStatus.AVAILABLE if self._inited else ProviderStatus.UNAVAILABLE

    def get_supported_categories(self) -> list[DataCategory]:
        return [
            DataCategory.BAR_HISTORY,
            DataCategory.CONTRACT,
            DataCategory.TRADE_CALENDAR,
            DataCategory.REFERENCE,
            DataCategory.FUNDAMENTAL,
        ]

    def get_bar_history(self, symbol: str, exchange: str, interval: str,
                        start: datetime, end: Optional[datetime] = None,
                        adjust_type: str = "hfq") -> Optional[list[BarRecord]]:
        body = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": _PERIOD_MAP.get(interval, interval),
            "start": _fmt_date(start),
            "end": _fmt_date(end) if end else "",
            "adjust_type": adjust_type,
        }
        resp = self._http.post("/bars", json=body)
        resp.raise_for_status()
        df = pl.read_ipc_stream(resp.content)
        if df.height == 0:
            return None
        return [BarRecord(**row) for row in df.iter_rows(named=True)]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_provider.py -v`
Expected: PASS（5 个）

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add backend/aitrade/datasource/qmt_bridge_provider.py backend/tests/test_qmt_bridge_provider.py
git commit -m "QMT 桥：Mac 端 Provider 状态机 + get_bar_history（Arrow 解码，None/raise 三态，Task B1）"
```

---

### Task B2: Provider — 合约/日历/复权因子/基本面

**Files:**
- Modify: `backend/aitrade/datasource/qmt_bridge_provider.py`
- Test: `backend/tests/test_qmt_bridge_provider_meta.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_qmt_bridge_provider_meta.py`：

```python
"""QmtBridgeProvider 元数据方法测试（注入假 httpx）。"""

from aitrade.datasource.qmt_bridge_provider import QmtBridgeProvider
from aitrade.datasource.types import ContractInfo, CalendarDay


class _Resp:
    def __init__(self, json_data):
        self._json = json_data
        self.status_code = 200

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


class _Http:
    def __init__(self, responses):
        self._responses = responses

    def get(self, path, params=None):
        return self._responses[path]


def _p(responses):
    p = QmtBridgeProvider(url="http://win", token="t")
    p._http = _Http(responses)
    p._inited = True
    return p


def test_get_contracts():
    p = _p({"/contracts": _Resp([{
        "symbol": "600000", "exchange": "SSE", "name": "浦发银行",
        "product_type": "股票", "size": 1.0, "pricetick": 0.01,
        "list_date": "19991110", "delist_date": "", "extra": {},
    }])})
    cs = p.get_contracts()
    assert isinstance(cs[0], ContractInfo)
    assert cs[0].symbol == "600000" and cs[0].name == "浦发银行"


def test_get_trade_calendar():
    p = _p({"/trading_calendar": _Resp([{"date": "20240102", "exchange": "SSE", "is_open": True}])})
    cal = p.get_trade_calendar("SSE", "20240101", "20240105")
    assert isinstance(cal[0], CalendarDay)
    assert cal[0].date == "20240102" and cal[0].is_open is True


def test_get_adj_factor():
    p = _p({"/adj_factor": _Resp([{"trade_date": "20240110", "adj_factor": 1.1}])})
    af = p.get_adj_factor("600000", "SSE")
    assert af == [{"trade_date": "20240110", "adj_factor": 1.1}]


def test_get_contracts_empty_returns_none():
    p = _p({"/contracts": _Resp([])})
    assert p.get_contracts() is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_provider_meta.py -v`
Expected: FAIL（缺方法）

- [ ] **Step 3: 加方法到 qmt_bridge_provider.py**

文件顶部 import 增加：`from .types import ContractInfo, CalendarDay, FundamentalRecord`（与现有 import 合并）。在类内追加：

```python
    def get_contracts(self, product_type: str = "", exchange: str = "") -> Optional[list[ContractInfo]]:
        resp = self._http.get("/contracts", params={"include_bse": False})
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        return [ContractInfo(
            symbol=r["symbol"], exchange=r["exchange"], name=r.get("name", ""),
            product_type=r.get("product_type", ""), size=r.get("size", 1.0),
            pricetick=r.get("pricetick", 0.01), list_date=r.get("list_date", ""),
            delist_date=r.get("delist_date", ""), extra=r.get("extra", {}),
        ) for r in rows]

    def get_trade_calendar(self, exchange: str, start: str, end: str) -> Optional[list[CalendarDay]]:
        resp = self._http.get("/trading_calendar",
                              params={"exchange": exchange, "start": start, "end": end})
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        return [CalendarDay(date=r["date"], exchange=r["exchange"], is_open=r["is_open"]) for r in rows]

    def get_adj_factor(self, symbol: str, exchange: str,
                       start: str = "", end: str = "") -> Optional[list[dict]]:
        resp = self._http.get("/adj_factor",
                              params={"symbol": symbol, "exchange": exchange, "start": start, "end": end})
        resp.raise_for_status()
        rows = resp.json()
        return rows or None

    def get_fundamental(self, symbol: str, exchange: str, start: str, end: str) -> Optional[list[FundamentalRecord]]:
        resp = self._http.get("/fundamental",
                              params={"symbol": symbol, "exchange": exchange, "start": start, "end": end})
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        # QMT 财务为原始报表，映射进 FundamentalRecord：trade_date 用 ann_date(可见日)，
        # 原始科目放 extra（估值类 PE/PB 等 QMT 给不了，留空让 manager 回退 tushare）
        return [FundamentalRecord(
            symbol=r["symbol"], exchange=r["exchange"], trade_date=r.get("ann_date", ""),
            extra={"table": r.get("table"), "report_period": r.get("report_period"),
                   "fields": r.get("fields", {})},
        ) for r in rows]
```

> ⚠️ 设计取舍：QMT 财务是"原始报表科目"，与 `FundamentalRecord` 的估值字段(PE/PB/...)语义不同。这里把原始科目装进 `extra`，估值字段留 `None`——这样估值类查询走 tushare（按类目降级仍生效）。Wave C 真机校验后若要把每股指标提取成顶层字段，回此调整。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_provider_meta.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: Commit**

```bash
cd /Users/kevin_1/aitrade
git add backend/aitrade/datasource/qmt_bridge_provider.py backend/tests/test_qmt_bridge_provider_meta.py
git commit -m "QMT 桥：Mac 端 Provider 合约/日历/复权因子/基本面（Task B2）"
```

---

### Task B3: 注册 + 接入选源链

**Files:**
- Modify: `backend/aitrade/datasource/__init__.py`（导出）
- Modify: `backend/aitrade/main.py`（lifespan 注册）
- Modify: `backend/aitrade/api/alpha_service.py`（`_pick_bar_provider` 选源元组）
- Test: `backend/tests/test_qmt_bridge_registration.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_qmt_bridge_registration.py`：

```python
"""QMT 桥注册到 manager 与选源链接入测试。"""

from aitrade.datasource.manager import DataSourceManager
from aitrade.datasource.qmt_bridge_provider import QmtBridgeProvider
from aitrade.datasource.types import DataCategory


def test_registered_first_for_bar_history():
    mgr = DataSourceManager()
    # QMT 优先级最小（最先）
    mgr.register(QmtBridgeProvider(url="http://win", token="t"), priority=-10)

    class _Other(QmtBridgeProvider):
        name = "other"

    mgr.register(_Other(url="http://win", token="t"), priority=0)

    providers = mgr._resolve_providers(DataCategory.BAR_HISTORY)
    assert providers[0].name == "qmt"


def test_pick_bar_provider_includes_qmt():
    from aitrade.api import alpha_service
    import inspect
    src = inspect.getsource(alpha_service._pick_bar_provider)
    assert "qmt" in src  # 选源元组已含 'qmt'
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_registration.py -v`
Expected: FAIL（第二个断言失败：`_pick_bar_provider` 源码不含 'qmt'）

- [ ] **Step 3a: 导出 Provider**

在 `backend/aitrade/datasource/__init__.py` 加：

```python
from .qmt_bridge_provider import QmtBridgeProvider  # noqa: F401
```

- [ ] **Step 3b: lifespan 注册**

先看现有注册写法：`grep -n "datasource_manager.register\|register(" backend/aitrade/main.py`。仿照在 `backend/aitrade/main.py` 的 lifespan（约 94-103 行，tushare/akshare 注册附近）加：

```python
    from aitrade.config import QMT_BRIDGE_URL, QMT_BRIDGE_TOKEN
    from aitrade.datasource.qmt_bridge_provider import QmtBridgeProvider
    datasource_manager.register(
        QmtBridgeProvider(url=QMT_BRIDGE_URL, token=QMT_BRIDGE_TOKEN),
        priority=-10,  # manager 升序，负值 => 最优先
    )
```

- [ ] **Step 3c: 选源元组加 'qmt'**

先定位：`grep -n "tushare.*akshare.*gateway\|_pick_bar_provider" backend/aitrade/api/alpha_service.py`。把硬编码的 `("tushare", "akshare", "gateway")` 改为 `("qmt", "tushare", "akshare", "gateway")`（QMT 排首位）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_qmt_bridge_registration.py -v`
Expected: PASS（2 个）

- [ ] **Step 5: 全量回归（确保没碰坏现有数据源逻辑）**

Run: `cd /Users/kevin_1/aitrade/backend && .venv/bin/python -m pytest tests/test_akshare_provider.py tests/test_qmt_bridge_provider.py tests/test_qmt_bridge_provider_meta.py tests/test_qmt_bridge_registration.py -v`
Expected: 全 PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/kevin_1/aitrade
git add backend/aitrade/datasource/__init__.py backend/aitrade/main.py backend/aitrade/api/alpha_service.py backend/tests/test_qmt_bridge_registration.py
git commit -m "QMT 桥：注册到数据源链最前 + 接入自动选源（Task B3）"
```

---

## Wave C — 真机验证（在你的 Windows 机器上执行）

> 这一波**必须在装有 QMT/miniQMT 并登录的 Windows 上跑**，连真 xtquant。Mac 上无法替代。每条若与 Wave A 的假设不符，回对应 A 任务改归一化逻辑、补/改单测、重跑、提交。

### Task C1: 固化 7 条 MEDIUM 置信项

- [ ] 在 Windows 装好 `qmt-bridge`（见 README），`python -c "from xtquant import xtdata; ..."` 逐条 `inspect.signature` / `print` 固化：
  1. `get_market_data_ex` 实际签名/返回维度（确认"按 stock_code 拆表、index/列含 time"）。
  2. 财务表名大小写、`get_financial_data` 返回结构。
  3. `OpenDate/ExpireDate` 实际格式、在市股票 `ExpireDate` 占位含义。
  4. `'time'` 毫秒时间戳时区（反推一根已知 K 线确认东八区当日）。
  5. `沪深A股` 是否含科创/创业、`沪深京A股` 是否含北交所；北交所 `BJ`、日历 `market='BJ'`。
  6. `front` vs `front_ratio` 算法差异（同票取数对比）。
  7. dr 累乘方向/边界。
- [ ] 与假设有偏差的，回 Task A2/A3/A4 改实现 + 改假 xtdata 测试，重跑 `pytest qmt-bridge/tests`，提交。

### Task C2: 复权金标对拍

- [ ] 选一只已知分红+送转个股，分别取 `dividend_type='back'` 与本地 `adj_factor` 还原的 hfq 价对比；校验 dr 累乘方向/边界正确。
- [ ] 决定默认 `back` 还是 `back_ratio`（待决②），写进 `config.py` 默认值 + README，提交。

### Task C3: 端到端联调

- [ ] Windows 起服务：`uvicorn qmt_bridge.app:app --host 0.0.0.0 --port 58610`（设好 `QMT_BRIDGE_TOKEN`、连接模式）。
- [ ] Mac 配 `QMT_BRIDGE_URL`（内网/Tailscale 地址）、`QMT_BRIDGE_TOKEN`，启动 aitrade 后端。
- [ ] 前端"数据下载"页用 `provider='qmt'` 拉一段历史 → 确认落到待合并批次 → 合并入正式 K 线 → `load_bar_df` 读出 → 跑一次回测。
- [ ] 验证降级：QMT 给不了的估值字段查询自动落回 tushare（断开 QMT 时 `/health` 反映 DEGRADED/UNAVAILABLE，下载明确报错而非空盘）。

---

## 自检（spec 覆盖）

- §3 架构 → Wave A（服务）+ Wave B（Provider）✅
- §5 线协议（Arrow/JSON/bearer）→ A5/A6/A7、B1/B2 ✅
- §6 鉴权安全 → A6（token）、README（内网红线）✅
- §7 复权 → A2（dividend 映射）/A3（dr 累乘）/C2（对拍）✅
- §8 健康检查 → A4（is_connected）/A7（/health）/B1（get_status）✅
- §9 错误降级 → B1（None/raise）/B2（按类目）/B3（选源链）✅
- §10 类目归属 → B2（财务装 extra、估值落回 tushare）✅
- §11 测试 → 各任务单测 + C 真机对拍 ✅
- §12 必核清单 → C1 ✅
- §13 字段映射 → A1/A2/A4 + B1/B2 ✅
- §14 待决①部署模式 → A6 config（CONNECT_MODE）；待决②复权 → C2 ✅
