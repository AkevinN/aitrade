from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from aitrade.alpha.dataset.datasets.alpha_101 import Alpha101
from aitrade.alpha.dataset.utility import calculate_by_expression, calculate_by_polars


def _make_sample_df() -> pl.DataFrame:
    rows: list[dict] = []
    symbols = ["000001.SZSE", "000002.SZSE", "600000.SSE"]
    start = datetime(2023, 1, 1)

    for day in range(260):
        current = start + timedelta(days=day)
        for index, vt_symbol in enumerate(symbols):
            base = 100 + index * 10 + day * 0.1
            volume = 1_000 + index * 100 + day
            rows.append(
                {
                    "datetime": current,
                    "vt_symbol": vt_symbol,
                    "open": base,
                    "high": base * 1.01,
                    "low": base * 0.99,
                    "close": base * 1.001,
                    "volume": volume,
                    "turnover": volume * (base * 1.0005),
                    "open_interest": 0.0,
                    "vwap": base * 1.0005,
                }
            )

    return pl.DataFrame(rows)


def test_alpha101_feature_expressions_smoke() -> None:
    sample_df = _make_sample_df()
    dataset = Alpha101(
        df=sample_df,
        train_period=("2023-01-01", "2023-06-01"),
        valid_period=("2023-06-02", "2023-08-01"),
        test_period=("2023-08-02", "2023-09-17"),
    )

    failures: list[str] = []

    for name, expression in dataset.feature_expressions.items():
        try:
            if isinstance(expression, pl.Expr):
                result = calculate_by_polars(sample_df, expression)
            else:
                result = calculate_by_expression(sample_df, expression)
            assert result.height == sample_df.height
        except Exception as exc:  # pragma: no cover - failure path is the assertion target
            failures.append(f"{name}: {exc}")

    assert failures == []
