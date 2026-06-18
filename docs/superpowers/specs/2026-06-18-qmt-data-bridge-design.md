# QMT 数据桥设计（方案 A · remote-only）

> 状态：设计草案（待评审）
> 日期：2026-06-18
> 范围：**数据桥**（不含交易下单），第一期

## 1. 背景与目标

### 1.1 问题
QMT（迅投/券商系量化终端）的行情与基本面数据，只能在 **Windows** 上经 `xtquant`（`xtdata` 模块）取得；`xtquant` 本质是个客户端，必须连上一个正在运行的 QMT/miniQMT 客户端或 `xtdatacenter` 数据服务才能取数。而本项目的研究、回测、选股全部在 **macOS** 上进行。核心目标：**让 Mac 上的研究系统用上只有 Windows 能拿到的 QMT 数据。**

### 1.2 本期范围（已与用户确认）
- **只做数据桥**，不含交易/下单（交易桥作为后续单独一期，复用本期的服务外壳与鉴权层）。
- **只做历史批量同步**，不含实时行情（纯请求-响应 REST，不引入 WebSocket/回调中继）。
- **全量数据类目**：历史 K 线、合约列表、交易日历、复权因子、基本面/财务；QMT 给不了的（估值/行业/一致预期/分红事件）**按类目自动落回 tushare**。
- **架构方案 A（remote-only）**：Windows 跑一个独立的瘦 REST 服务包住 `xtdata`；Mac 端新增一个数据源 Provider，永远经 REST 调它。Mac 端**永不 import xtquant**。

### 1.3 非目标
- 实时行情/tick 推送、交易下单（后续期）。
- Mac 直连 xtquant（已被核查证伪：跨机 `xtdata.connect(ip,port)` 直连取数官方未证实且有反证）。
- "整套 aitrade 后端跑在 Windows 上直连"这种共用代码的能力（YAGNI，用户确认不需要；若将来要，另起一条独立轨道）。

## 2. 关键事实依据（来自专项核查，标注置信度）

| 结论 | 置信度 | 影响 |
|---|---|---|
| `xtquant` 是客户端，必须有 QMT/miniQMT 客户端或 `xtdatacenter` 服务在跑才能取数 | confirmed | 决定"必须有 Windows 后端在跑"，Mac 不能直连 |
| 取历史数据是两步：先 `download_history_data(2)` 落地本地，再 `get_market_data_ex`/`get_local_data` 读 | confirmed | 服务端每个取数端点内部都要"先下载再读" |
| 复权由 `dividend_type` 控制，取值 `none/front/back/front_ratio/back_ratio` | confirmed | 复权口径映射的基础 |
| `get_stock_list_in_sector('沪深A股')` 可枚举几乎全部沪深 A 股（含科创/创业）；`'沪深京A股'` 含北交所 | confirmed | 全市场标的枚举 |
| 跨机 `xtdata.connect(ip,port)` 让另一台机器直连取数 | **uncertain（有反证）** | 故走 REST 桥而非直连 |
| `get_market_data_ex` 精确签名/返回维度、财务表名大小写、`OpenDate/ExpireDate` 格式、时间戳时区、复权 `dr` 累乘边界 | **medium** | 列入"上真机必核清单"，落地第一步现场固化 |

数据契约（aitrade 侧，已逐行核对源码）：`BaseProvider`/`DataSourceManager`/`BarRecord` 等定义见 `backend/aitrade/datasource/{base,types,manager}.py`；下载管线见 `backend/aitrade/api/alpha_service.py` 与 `backend/aitrade/alpha/lab.py`。

## 3. 总体架构

```
   Mac（aitrade 现有仓库）                 Windows（全新独立项目 qmt-bridge）
   ┌─────────────────────────────┐         ┌────────────────────────────────────┐
   │ DataSourceManager 优先级链   │  HTTPS  │ FastAPI 薄路由层                     │
   │  ├ QmtBridgeProvider ────────┼────────▶│   └ 调用 →【xtdata 封装模块(可单测)】│
   │  │   永远 httpx 远程调用      │◀────────┤        └ import xtquant.xtdata       │
   │  │   排在最前(priority 最小)  │ Arrow/  │           归一化 + 复权 + 后缀转换   │
   │  ├ TushareProvider（兜底）    │  JSON   │   单 worker 串行队列(MiniQmt 单连接) │
   │  └ AkshareProvider/Mock      │         └────────────────────────────────────┘
   │ → 复用现有"下载→落 Parquet"   │           连接二选一：
   │   管线；研究/回测离线读缓存   │           ① 连券商 QMT 客户端(默认, 免费行情)
   └─────────────────────────────┘           ② 独立 xtdatacenter(无 GUI, 需 VIP token)
```

要点：
- **Mac 端**：一个 `QmtBridgeProvider`，永远走 REST（remote-only，构造简单，无接口注入抽象）。坐在现有 Provider 链上，**自动获得 tushare 兜底、TTL 缓存、按类目降级**。
- **Windows 端**：一个与 aitrade 主仓**分离**的独立小项目 `qmt-bridge`，内部分两层——薄 FastAPI 路由 + 可单测的 xtdata 封装模块（喂假 xtdata 即可测归一化/复权/后缀逻辑）。
- 系统其余部分（下载页、回测、CNN、选股）**零改动**——它们只认 Provider 抽象，不在乎背后是 QMT 还是 tushare。

## 4. 组件设计

### 4.1 Windows 端：`qmt-bridge`（独立项目）
- **分层**：`xtdata_client`（封装模块：调 `xtdata` + 归一化）↔ `routes`（FastAPI，鉴权 + 序列化）。封装模块不依赖 FastAPI，可脱机单测。
- **连接模式（配置二选一，默认①）**：
  - ① 连券商 QMT/miniQMT 客户端：`xtdata.connect()`（本机 `127.0.0.1:58610`），用券商免费行情；代价是客户端可能需每日登录。
  - ② 独立 `xtdatacenter`：`set_token → init(start_local_service=False) → listen → connect`，无 GUI 可常驻，需迅投 VIP token（新用户 14 天试用）。
- **并发**：所有 `download_*` 同步阻塞、MiniQmt 单连接 → **单 worker 串行队列**，REST 层请求排队而非并发打满。
- **进程保活/自愈**：`xtquant` 不自动重连；轮询 `get_client().is_connected()`，断线则 `reconnect()`/重启 xtdc。可参考开源 `atompilot/qmt-bridge`（同款架构）。

### 4.2 Mac 端：`QmtBridgeProvider(BaseProvider)`
- 新文件 `backend/aitrade/datasource/qmt_bridge_provider.py`，模板照 `AkshareProvider`（延迟导入 + 软失败）+ `TushareProvider`（状态机）。
- 实现：`init`/`get_status`/`get_supported_categories` + `get_bar_history`；可选 `get_contracts`/`get_trade_calendar`/`get_adj_factor`/`get_fundamental`。
- 注册：`main.py` lifespan 中 `datasource_manager.register(qmt, priority=-10)`（manager 按 priority **升序**，越小越先；现有 tushare=0/akshare=10/mock=100，故用负值让 QMT 最优先）。`init_all()` 自动调 `init()`。
- 配置：`config.py` 新增 `QMT_BRIDGE_URL` / `QMT_BRIDGE_TOKEN`（`os.getenv` 风格）。
- **两处必改的既有逻辑**（核查发现的盲点）：
  1. `alpha_service._pick_bar_provider` 自动选源元组硬编码 `('tushare','akshare','gateway')`，需加入 `'qmt'`（否则只能靠用户显式传 `provider='qmt'`）。
  2. `cbond` 只放行 akshare、`etf` 跳过 akshare 的品种过滤规则，若 QMT 要供 ETF/转债需同步放行。

## 5. 线协议（REST 端点）

| 端点 | 方法 | 负载格式 | 对应 Provider 方法 |
|---|---|---|---|
| `/health` | GET | JSON `{connected, server_addr, xtdata_version, init_done}` | `init`/`get_status` |
| `/bars` | POST | **Arrow IPC stream (zstd)**，列名对齐 `BarRecord`，**显式带回 `adjust_type`** | `get_bar_history` |
| `/contracts` | GET | JSON | `get_contracts` |
| `/trading_calendar` | GET | JSON | `get_trade_calendar` |
| `/adj_factor` | GET | JSON `[{trade_date, adj_factor}]` | `get_adj_factor` |
| `/fundamental` | GET | JSON（分页） | `get_fundamental` |

- **K 线走 Arrow IPC stream**：服务端 `df.write_ipc_stream(None, compression='zstd').getvalue()` → `Response(media_type='application/vnd.apache.arrow.stream')`；Mac 端 `pl.read_ipc_stream(resp.content)`。流/文件格式必须配对。元数据小，走 JSON。
- **鉴权**：静态 bearer token + `secrets.compare_digest`（常量时间），`HTTPBearer` 依赖注入；缺/错 token → 401。**绝不挂 GZipMiddleware**（Arrow/Parquet 已内置压缩）。
- **安全红线**：服务**绝不暴露公网**，走内网或 Tailscale/WireGuard。
- **后缀转换集中一处**：xtdata 全程 `.SH/.SZ/.BJ`，aitrade 用 `SSE/SZSE/BSE`；双向转换复用 `akshare_provider.py` 现有对照表，入参转 `.SH`、返回转回——漏一处即静默取空。

## 6. 数据流（download ≠ use）

```
Windows: download_history_data2([code], period, start, end)   ← 串行队列, 断点续传
            → get_market_data_ex([], [code], period, ..., dividend_type=…)
            → 归一化为列式 Arrow → REST 返回
Mac: QmtBridgeProvider.get_bar_history → list[BarRecord]
            → 复用 POST /api/alpha/data/download（provider='qmt'）
            → BarData → save_bars_as_import_batch（待合并批次，现有管线）
            → 研究/回测离线读本地 Parquet
```
- 两步模式已确认：不下载直接读会空/不全。
- Mac 端**按 symbol + 时间窗分页**多次请求，单请求只干一个标的一段时间，避免 HTTP 超时与服务端队列堵塞。
- 首次全 A（~5000+ 只）回补耗时长，之后每日增量。

## 7. 复权处理（正确性最关键）

- `dividend_type` 五值已确认。`adjust_type{none,qfq,hfq}` → `dividend_type` 做**显式映射 dict**，配置项可选"普通/等比"。
- **默认落 `hfq`（`back` 或 `back_ratio`）**：前复权历史价会随"今天"漂移 → 缓存失效 + 未来函数风险。`qfq/none` 在**本地由 `adj_factor` 推导**（qfq 因子 = hfq 因子 / 末值），不向 xtdata 重复取前复权。
- **一次同步只用单一口径**：`lab.py` 有"同资源口径一致性校验"，口径不一致直接 `ValueError`；`BarRecord.adjust_type` 必须如实标注。
- `get_adj_factor` 用 `get_divid_factors`（**单只**代码）的 `dr` 累乘出后复权累积因子（单调 ≥1，非除权日沿前值）。
- **落地前金标对拍**：拿已知分红+送转个股，对比 hfq 价 / adj_factor，校验累乘方向与边界。

## 8. 连接与健康检查

- `/health` 由 `xtdata.get_client().is_connected()`（或缓存 `watch_quote_server_status` 回调的 `connected/disconnected`）驱动。
- **`get_status()` 不发网络请求**（照 `TushareProvider` 模式，仅由 `init` 状态推导）；健康探测只在 `init()` 打一次。
- 状态映射：已连=`AVAILABLE`；服务在但 xtdata 断/QMT 未登录=`DEGRADED`；REST 不可达=`UNAVAILABLE`；未配 url/token=`NOT_CONFIGURED`。
- **xtquant 不自动重连** → Windows 端轮询 + 自愈（见 4.1）。

## 9. 错误处理与降级

- **三态清晰**：无数据 → `None`（manager 回退下一个源）；接口/网络错 → **raise**（`provider_name='qmt'` 锁定时原样上抛，区分"真错"vs"无数据"）；配置缺失 → `NOT_CONFIGURED`（不进 fallback 链）。
- **回补健壮性**：分批 + 超时 + 重试 + 断点续传（`incrementally=True`）；单只失败不拖垮整批。
- **每日重登风险**：定时回补任务可能撞上 QMT 刚好断线/未登录；`/health` 必须如实反映，失败应明确报错而非递空盘。

## 10. 数据类目归属（QMT vs tushare）

| 数据 | 来源 | 说明 |
|---|---|---|
| 历史 K 线（含复权） | **QMT** | 核心，本期主要价值 |
| 合约列表/详情、交易日历、复权因子 | **QMT** | 基础数据 |
| 财报三表 + 每股指标 + 股本 + 股东 | **QMT** | `report_type` 固定 `'announce_time'` 防未来函数 |
| 估值（PE/PB/总市值/换手）、行业分类代码、一致预期、分红送转/解禁/业绩快报事件 | **tushare 兜底** | xtdata 完全给不了；`get_supported_categories` 不声明这些细类，manager 自动落回 tushare |

## 11. 测试策略

- **Mac 端单测（纯离线，mock REST）**：后缀双向转换、interval 映射（尤其 `60m→1h`）、`adjust_type↔dividend_type` 映射、毫秒时间戳→datetime 时区、`None vs []` 语义、HTTP 失败 raise、Arrow round-trip（write↔read 配对）、bearer 鉴权（缺/错 token→401）。
- **Windows 端单测（喂假 xtdata）**：xtdata 封装模块的归一化/复权/后缀逻辑，不依赖真 QMT。
- **真机金标对拍（集成，单独标记按需跑）**：已知分红送转票的 hfq 价 / adj_factor；`announce_time` vs `report_time` 防未来函数；`沪深A股` vs `沪深京A股` 成分数；`get_market_data_ex` 实际签名/返回结构。
- **边界**：`as_of` 早于本地数据（空窗跳过，复用 Tier-2 已有处理）、停牌票、未 download 直接 get（空/None）、download 带毫秒 `.000` 会 hang 的回归保护。

## 12. 上真机必核清单（落地第一步先固化）

研究中以下项官方文档未写死，**Windows 上 `import xtdata` 后用 `inspect.signature`/`print` 现场确认再写映射**：

1. `get_market_data_ex` 精确签名/默认值 + "按 stock_code 拆表"返回维度（与 `get_market_data` 相反，写反全错）。
2. 财务表名大小写（`Top10holder` vs `Top10Holder`）、`get_financial_data` 返回结构（dict-of-dict vs 四 list）。
3. `OpenDate/ExpireDate` 实际格式（int / `YYYYMMDD` / `YYYY-MM-DD`）、在市股票 `ExpireDate` 占位含义。
4. `'time'` 毫秒时间戳的时区（反推一根已知 K 线确认东八区当日）。
5. `沪深A股` 是否含科创/创业、`沪深京A股` 是否含北交所；北交所 `BJ→BSE`、日历 `market='BJ'`。
6. `front` vs `front_ratio` 算法差异（同票取数对比）。
7. 等比/普通复权 `dr` 累乘的方向与边界。

## 13. 字段映射附录（grounded，落地前以真机为准复核 MEDIUM 项）

**K 线**：`BarRecord` ← `get_market_data_ex` 单标的 DataFrame
`open_price←open`、`high_price←high`、`low_price←low`、`close_price←close`、`volume←volume`、`turnover←amount`、`open_interest←openInterest`（股票恒 0）、`datetime←'time'/1000`（毫秒时间戳）、`adjust_type←` 同步口径。

**合约**：`ContractInfo` ← `get_instrument_detail` + `get_instrument_type`
`symbol←InstrumentID`、`exchange←` 后缀映射、`name←InstrumentName`、`product_type←get_instrument_type`、`size←VolumeMultiple`、`pricetick←PriceTick`、`list_date←OpenDate`、`delist_date←ExpireDate`、`extra←{InstrumentStatus, IsTrading, ...}`。

**日历**：`CalendarDay` ← `get_trading_calendar(market, start, end)`（返回 `'YYYYMMDD'` 字符串列表，需先 `download_holiday_data()`）。

**复权因子**：`[{trade_date, adj_factor}]` ← `get_divid_factors` 的 `dr` 累乘。

**基本面**：`FundamentalRecord` ← `get_financial_data(..., report_type='announce_time')`，`report_period←m_timetag`、`ann_date←m_anntime`。

**周期映射**：`d→1d`、`1m→1m`、`5m→5m`、`15m→15m`、`30m→30m`、`60m/1h→1h`、`w→1w`。
**交易所映射**：`SSE↔SH`、`SZSE↔SZ`、`BSE↔BJ`。

## 14. 风险与待决

- **待决①（部署路）**：连券商客户端（默认，免费但需每日登录）vs 独立 xtdatacenter（常驻但需 VIP token）——做成配置开关，默认①，可后续切②。
- **待决②（复权细则）**：默认 `hfq=back` 还是 `back_ratio`（普通 vs 等比，是否含红利复投）——落地金标对拍后定。
- **风险**：全市场首次回补耗时/偶发卡死；QMT 每日重登撞上定时任务；MEDIUM 置信项需真机固化。
