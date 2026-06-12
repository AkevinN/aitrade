from __future__ import annotations

from datetime import date, datetime

import polars as pl

from aitrade.alpha.lab import AlphaLab, normalize_vt_symbol


def test_normalize_vt_symbol_variants() -> None:
    assert normalize_vt_symbol("sz000415") == "000415.SZSE"
    assert normalize_vt_symbol("SH600000") == "600000.SSE"
    assert normalize_vt_symbol("000415.SZ") == "000415.SZSE"
    assert normalize_vt_symbol("600000.SSE") == "600000.SSE"
    assert normalize_vt_symbol("000415") == "000415.SZSE"


def test_normalize_vt_symbol_etf() -> None:
    """ETF：510xxx→SSE，159xxx→SZSE。"""
    assert normalize_vt_symbol("510300") == "510300.SSE"
    assert normalize_vt_symbol("159915") == "159915.SZSE"


def test_normalize_vt_symbol_convertible_bonds_sse() -> None:
    """沪市转债：110/111/113/118 开头 → SSE（现网 bug：之前会错误归到 SZSE）。"""
    # 110xxx
    assert normalize_vt_symbol("110059") == "110059.SSE"
    # 111xxx
    assert normalize_vt_symbol("111001") == "111001.SSE"
    # 113xxx
    assert normalize_vt_symbol("113001") == "113001.SSE"
    assert normalize_vt_symbol("113050") == "113050.SSE"
    # 118xxx
    assert normalize_vt_symbol("118001") == "118001.SSE"


def test_normalize_vt_symbol_convertible_bonds_szse() -> None:
    """深市转债：123/127/128 开头 → SZSE（原逻辑已正确，验证不回退）。"""
    assert normalize_vt_symbol("123001") == "123001.SZSE"
    assert normalize_vt_symbol("127001") == "127001.SZSE"
    assert normalize_vt_symbol("128001") == "128001.SZSE"


def test_normalize_vt_symbol_convertible_bonds_with_suffix() -> None:
    """带交易所后缀的转债代码：后缀优先，不经数字规则。"""
    assert normalize_vt_symbol("113001.SSE") == "113001.SSE"
    assert normalize_vt_symbol("113001.SH") == "113001.SSE"
    assert normalize_vt_symbol("113001.SZ") == "113001.SZSE"
    assert normalize_vt_symbol("128001.SZSE") == "128001.SZSE"


def test_normalize_vt_symbol_convertible_bonds_prefix() -> None:
    """带 sh/sz 前缀的转债代码：前缀优先。"""
    assert normalize_vt_symbol("sh113001") == "113001.SSE"
    assert normalize_vt_symbol("sz128001") == "128001.SZSE"


def test_load_bar_frame_accepts_prefixed_symbol(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    legacy_path = lab.bars_path / "1m" / "sz000415..parquet"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "datetime": [datetime(2025, 3, 1, 9, 31), datetime(2025, 3, 1, 9, 32)],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [100.0, 120.0],
            "turnover": [1000.0, 1200.0],
            "open_interest": [0.0, 0.0],
        }
    ).write_parquet(legacy_path)

    frame = lab.load_bar_frame(
        "000415.SZSE",
        "1m",
        date(2025, 2, 1),
        date(2026, 1, 1),
    )

    assert frame is not None
    assert len(frame) == 2


def test_list_data_resources_uses_canonical_symbol(tmp_path) -> None:
    lab = AlphaLab(tmp_path)
    legacy_path = lab.bars_path / "1m" / "sz000415..parquet"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "datetime": [datetime(2025, 3, 1, 9, 31)],
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "volume": [100.0],
            "turnover": [1000.0],
            "open_interest": [0.0],
        }
    ).write_parquet(legacy_path)

    resources = lab.list_data_resources()
    assert len(resources["raw_bars"]) == 1
    assert resources["raw_bars"][0]["vt_symbol"] == "000415.SZSE"
    assert resources["raw_bars"][0]["key"] == "1m__000415.SZSE"
