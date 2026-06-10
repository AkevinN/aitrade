# AKShare 对接参考

> 本文档收录 AITrade 接入 AKShare 数据源时实际使用的接口规格与映射约定，供后续扩展时查阅。
> 官方文档：https://akshare.akfamily.xyz/data/stock/stock.html
> 许可证：MIT（开源、免费、无需 token）；定位为学术研究数据，生产/实盘前需自行校验数据质量。

---

## 一、接入概览

| 项目层 | AKShare 层 | 说明 |
|--------|-----------|------|
| Provider 名称 | `akshare` | 注册优先级 priority=10（tushare=0 → akshare=10 → mock=100） |
| 认证 | 无 | 不需要 token，`pip install akshare` 即可 |
| 首期能力 | `BAR_HISTORY` | A 股日/周/月线 + 分钟线（1/5/15/30/60m） |
| 复权口径 | 默认不复权 `adjust=""` | 与现有 Tushare 下载保持一致 |

### symbol / exchange 映射

- 项目内部使用 `vt_symbol`，如 `600519.SSE`、`000001.SZSE`。
- **AkshareProvider 适配器会自动识别以下写法**（无需用户手动转换）：
  - `000415.SZSE` / `000415.SZ` / `000415.sz`
  - `sz000415` / `SH600000` / `bj830799`
  - `000415` + `exchange=SZSE`（或留空由代码前缀推断）
- 传给 AKShare 接口时统一转为 **6 位纯数字代码**（如 `000415`）。
- 若代码与交易所明显不匹配（如 `000415.SSE`），会直接报错提示正确写法。
- 仅支持 A 股（SSE / SZSE / BSE）；其他市场需后续扩展专用接口。

### 周期映射

| 项目 interval | AKShare 接口 | period 参数 |
|---------------|-------------|-------------|
| `d`  | `stock_zh_a_hist` | `daily` |
| `w`  | `stock_zh_a_hist` | `weekly` |
| `m`  | `stock_zh_a_hist` | `monthly` |
| `1m` | `stock_zh_a_hist_min_em` | `1`（仅近 5 个交易日，且不复权） |
| `5m` | `stock_zh_a_hist_min_em` | `5` |
| `15m`| `stock_zh_a_hist_min_em` | `15` |
| `30m`| `stock_zh_a_hist_min_em` | `30` |
| `1h` / `60m` | `stock_zh_a_hist_min_em` | `60` |

---

## 二、日/周/月线：`stock_zh_a_hist`

- 描述：东方财富-沪深京 A 股日频率数据；历史按日更新，当日收盘价需收盘后获取。
- 调用示例：

```python
import akshare as ak

df = ak.stock_zh_a_hist(
    symbol="000001",
    period="daily",         # daily / weekly / monthly
    start_date="20170301",  # YYYYMMDD
    end_date="20240528",
    adjust="",              # "" 不复权 / qfq 前复权 / hfq 后复权
)
```

### 输入参数

| 名称 | 类型 | 描述 |
|------|------|------|
| symbol | str | 股票代码，如 `603777`（纯代码） |
| period | str | `daily` / `weekly` / `monthly` |
| start_date | str | 开始日期 `YYYYMMDD` |
| end_date | str | 结束日期 `YYYYMMDD` |
| adjust | str | `""` 不复权 / `qfq` 前复权 / `hfq` 后复权 |
| timeout | float | 超时，默认 None |

### 输出字段（中文列）

| 列名 | 含义 | 单位/说明 | 映射到 BarRecord |
|------|------|-----------|------------------|
| 日期 | 交易日 | object | `datetime` |
| 股票代码 | 不带市场后缀 | object | （忽略，使用入参 symbol） |
| 开盘 | 开盘价 | float | `open_price` |
| 收盘 | 收盘价 | float | `close_price` |
| 最高 | 最高价 | float | `high_price` |
| 最低 | 最低价 | float | `low_price` |
| 成交量 | 成交量 | **手** | `volume` |
| 成交额 | 成交额 | 元 | `turnover` |
| 振幅 | 振幅 | % | （忽略） |
| 涨跌幅 | 涨跌幅 | % | （忽略） |
| 涨跌额 | 涨跌额 | 元 | （忽略） |
| 换手率 | 换手率 | % | （忽略） |

> `open_interest` 股票无持仓量，置 0。

### 复权说明

- 不复权：原始价格，可能因除权除息出现缺口。
- 前复权 `qfq`：保持当前价不变，历史价时变；可能为负。
- 后复权 `hfq`：保持历史价不变；量化研究中普遍采用。

---

## 三、分钟线

AKShare 分钟线有**两个接口**，Provider 会自动降级：

| 优先级 | 接口 | 来源 | 特点 |
|--------|------|------|------|
| 1 | `stock_zh_a_hist_min_em` | 东方财富 | 可按日期区间拉取；**网络不稳定，经常断连** |
| 2 | `stock_zh_a_minute` | 新浪财经 | 稳定性更好；历史深度有限（通常仅近 1～2 个月） |

### 3.1 东财：`stock_zh_a_hist_min_em`

- 描述：东方财富-沪深京 A 股每日分时行情；**只能获取近期数据**，注意周期设置。
- 限量：单次返回指定股票、频率、复权与时间区间的分时数据。**1 分钟数据仅返回近 5 个交易日，且不复权。**
- 调用示例：

```python
import akshare as ak

df = ak.stock_zh_a_hist_min_em(
    symbol="000001",
    start_date="2024-03-20 09:30:00",  # 日期时间字符串
    end_date="2024-03-20 15:00:00",
    period="5",                         # 1 / 5 / 15 / 30 / 60
    adjust="",                          # "" / qfq / hfq
)
```

### 输入参数

| 名称 | 类型 | 描述 |
|------|------|------|
| symbol | str | 股票代码（纯代码） |
| start_date | str | `YYYY-MM-DD HH:MM:SS`，默认返回所有 |
| end_date | str | `YYYY-MM-DD HH:MM:SS`，默认返回所有 |
| period | str | `1` / `5` / `15` / `30` / `60` |
| adjust | str | `""` / `qfq` / `hfq`（1 分钟强制不复权） |

### 输出字段

1 分钟数据列：`时间 / 开盘 / 收盘 / 最高 / 最低 / 成交量(手) / 成交额 / 均价`
其他周期列：`时间 / 开盘 / 收盘 / 最高 / 最低 / 涨跌幅 / 涨跌额 / 成交量(手) / 成交额 / 振幅 / 换手率`

映射到 BarRecord：`时间→datetime`（`YYYY-MM-DD HH:MM:SS`）、`开盘/收盘/最高/最低`、`成交量→volume`、`成交额→turnover`。

> 注意：1 分钟接口返回的数据中，只有最近一个交易日有真实开盘价，其余日期开盘价可能为 0。

### 3.2 新浪：`stock_zh_a_minute`（降级备用）

- 描述：新浪财经 A 股分钟线；Provider 在东财失败时自动切换到此接口。
- symbol 格式：`sh600519` / `sz000415` / `bj830799`
- 调用示例：

```python
import akshare as ak

df = ak.stock_zh_a_minute(symbol="sz000415", period="5", adjust="")
```

| 列名 | 映射到 BarRecord |
|------|------------------|
| day | datetime |
| open / high / low / close | OHLC |
| volume | volume |
| amount | turnover |

> 新浪接口不支持自定义起止日期，Provider 会在本地按请求时间范围过滤。

---

## 四、注意事项与风险

- 分钟接口为东财近端数据，历史深度有限；1 分钟仅近 5 个交易日。批量下载需要节流/重试。
- AKShare 依赖上游网页结构，接口可能因目标网站变化失效，需定期升级 `akshare` 版本。
- 统一在 Provider 内 `try/except` 降级，返回 `None`，由 `DataSourceManager` 决定回退到下一数据源。
- 后续可扩展接口（暂未接入）：
  - 交易日历：`tool_trade_date_hist_sina()`
  - 复权因子：`stock_zh_a_daily(adjust="hfq-factor"/"qfq-factor")`
  - 实时快照：`stock_zh_a_spot_em()`
