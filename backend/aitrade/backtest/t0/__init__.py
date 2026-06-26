"""半仓做 T 回测模块（half-position-t0-backtest）。

在既有 BacktestingEngine 之上提供：可插拔档位策略（TickPolicy）、半仓做 T 策略
（HalfPositionT0Strategy）、做 T 标的画像（T0Profiler）与回测编排（T0BacktestRunner）。
撮合/成本/T+1 一律复用引擎单一事实源，本模块只做"按规则挂单 / 算画像 / 扫网格出区间"。
"""
