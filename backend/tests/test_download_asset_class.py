"""Task 2.1: 品种（asset_class）字段下载链路单元测试。

覆盖：
- etf + 显式 akshare → RuntimeError，文案含"AKShare 数据源不支持 ETF 行情"
- _pick_bar_provider(asset_class="etf") 自动选源时跳过 akshare，落到 tushare
- 默认 stock 时行为不变（akshare 可被正常选中）
- DataDownloadRequest 默认 asset_class="stock"，ETF 请求能通过 Pydantic 校验
- 批次 metadata 落盘后含 asset_class 字段
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from aitrade.api.alpha_service import _download_bar_data, _pick_bar_provider
from aitrade.models.alpha import DataDownloadRequest
from aitrade.alpha.lab import AlphaLab, BarData
from aitrade.datasource.types import DataCategory, ProviderStatus


# ---------------------------------------------------------------------------
# 辅助：构造一个可用 provider 的 mock
# ---------------------------------------------------------------------------

def _make_provider(name: str, available: bool = True) -> MagicMock:
    p = MagicMock()
    p.get_info.return_value.status = ProviderStatus.AVAILABLE if available else ProviderStatus.UNAVAILABLE
    p.get_supported_categories.return_value = [DataCategory.BAR_HISTORY]
    return p


# ---------------------------------------------------------------------------
# 测试 1: DataDownloadRequest 默认 asset_class = "stock"
# ---------------------------------------------------------------------------

def test_data_download_request_default_asset_class():
    """默认不传 asset_class 时值为 'stock'。"""
    from datetime import date
    req = DataDownloadRequest(
        vt_symbols=["000001.SZSE"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
    )
    assert req.asset_class == "stock"


def test_data_download_request_etf_asset_class():
    """etf 值通过 Pydantic 校验。"""
    from datetime import date
    req = DataDownloadRequest(
        vt_symbols=["510300.SSE"],
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        asset_class="etf",
    )
    assert req.asset_class == "etf"


# ---------------------------------------------------------------------------
# 测试 2: etf + 显式 akshare → 报错，文案匹配
# ---------------------------------------------------------------------------

def test_pick_bar_provider_etf_explicit_akshare_raises():
    """etf 品种下显式指定 akshare 应抛出 RuntimeError，文案含"AKShare 数据源不支持 ETF 行情"。"""
    with pytest.raises(RuntimeError, match="AKShare 数据源不支持 ETF 行情"):
        _pick_bar_provider(preferred="akshare", asset_class="etf")


# ---------------------------------------------------------------------------
# 测试 3: etf 自动选源时跳过 akshare，落到 tushare
# ---------------------------------------------------------------------------

def test_pick_bar_provider_etf_skips_akshare_uses_tushare():
    """etf 品种下自动选源，akshare 可用但应被跳过，最终选中 tushare。"""
    tushare_mock = _make_provider("tushare")
    akshare_mock = _make_provider("akshare")

    def fake_get_provider(name: str):
        if name == "tushare":
            return tushare_mock
        if name == "akshare":
            return akshare_mock
        return None

    with patch("aitrade.api.alpha_service.datasource_manager") as mgr:
        mgr.get_provider.side_effect = fake_get_provider
        result = _pick_bar_provider(preferred="", asset_class="etf")

    assert result == "tushare"


def test_pick_bar_provider_etf_only_akshare_available_returns_empty():
    """etf 品种下只有 akshare 可用时，返回空串（无可用源）。"""
    akshare_mock = _make_provider("akshare")

    def fake_get_provider(name: str):
        if name == "akshare":
            return akshare_mock
        return None

    with patch("aitrade.api.alpha_service.datasource_manager") as mgr:
        mgr.get_provider.side_effect = fake_get_provider
        result = _pick_bar_provider(preferred="", asset_class="etf")

    assert result == ""


# ---------------------------------------------------------------------------
# 测试 4: stock 默认行为不变（akshare 可被选中）
# ---------------------------------------------------------------------------

def test_pick_bar_provider_stock_uses_akshare_when_only_one_available():
    """stock 品种下，akshare 可用时应被正常选中（默认 tushare 不在时）。"""
    akshare_mock = _make_provider("akshare")

    def fake_get_provider(name: str):
        if name == "akshare":
            return akshare_mock
        return None

    with patch("aitrade.api.alpha_service.datasource_manager") as mgr:
        mgr.get_provider.side_effect = fake_get_provider
        result = _pick_bar_provider(preferred="", asset_class="stock")

    assert result == "akshare"


def test_pick_bar_provider_default_asset_class_is_stock():
    """不传 asset_class 时默认为 stock，akshare 可被选中。"""
    akshare_mock = _make_provider("akshare")

    def fake_get_provider(name: str):
        if name == "akshare":
            return akshare_mock
        return None

    with patch("aitrade.api.alpha_service.datasource_manager") as mgr:
        mgr.get_provider.side_effect = fake_get_provider
        result = _pick_bar_provider(preferred="")

    assert result == "akshare"


# ---------------------------------------------------------------------------
# 测试 5: 批次 metadata 落盘后含 asset_class 字段
# ---------------------------------------------------------------------------

def _bar(dt: datetime, *, interval: str = "d", price: float = 10.0) -> BarData:
    return BarData(
        symbol="510300",
        exchange="SSE",
        datetime=dt,
        interval=interval,
        open_price=price,
        high_price=price + 0.2,
        low_price=price - 0.2,
        close_price=price + 0.1,
        volume=100.0,
        turnover=1000.0,
    )


def test_save_bars_as_import_batch_persists_asset_class(tmp_path) -> None:
    """save_bars_as_import_batch 传入 extra_meta 后，批次 metadata JSON 包含 asset_class。"""
    import json

    lab = AlphaLab(tmp_path)
    bars = [_bar(datetime(2024, 1, 2) + timedelta(days=i)) for i in range(3)]

    lab.save_bars_as_import_batch(
        bars,
        adjust_type="none",
        source="download",
        extra_meta={"asset_class": "etf"},
    )

    # 直接读取 JSON 文件验证落盘是否正确
    meta_files = list(tmp_path.rglob("*.meta.json"))
    assert len(meta_files) == 1, f"期望找到 1 个 meta.json，实际找到 {len(meta_files)} 个"

    with open(meta_files[0], encoding="utf-8") as f:
        meta = json.load(f)

    assert meta.get("asset_class") == "etf", (
        f"批次 metadata 应包含 asset_class='etf'，实际: {meta}"
    )


# ---------------------------------------------------------------------------
# 测试 6: 端到端——仅 akshare 可用时调用 _download_bar_data(etf) 抛出用户文案
# ---------------------------------------------------------------------------

def test_download_bar_data_etf_only_akshare_raises_user_message():
    """仅 akshare 可用的环境下，调用 _download_bar_data(asset_class="etf") 应抛出
    RuntimeError，文案含"ETF 下载当前需要 Tushare"。

    守护链路：_pick_bar_provider 返回空串 → _download_bar_data 检测 asset_class="etf"
    → 抛出用户可读的错误提示，而非静默失败。
    """
    from datetime import date

    req = DataDownloadRequest(
        vt_symbols=["510300.SSE"],
        start=date(2024, 1, 2),
        end=date(2024, 1, 31),
        asset_class="etf",
    )

    akshare_mock = _make_provider("akshare")

    def fake_get_provider(name: str):
        if name == "akshare":
            return akshare_mock
        return None  # tushare 不可用

    with patch("aitrade.api.alpha_service.datasource_manager") as mgr:
        mgr.get_provider.side_effect = fake_get_provider
        with pytest.raises(RuntimeError, match="ETF 下载当前需要 Tushare"):
            _download_bar_data(req)
