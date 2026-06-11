# AITrade API 接口文档

> 本文档覆盖后端全部 REST API 端点和 WebSocket 协议。

---

## 一、系统状态

### `GET /api/status`
> 系统健康检查，返回运行环境信息。

**响应**:
```json
{
  "version": "1.0.0",
  "torch_available": true,
  "torch_device": "cpu",
  "data_path": "/Users/xxx/.aitrade/alpha_lab",
  "tushare_token_set": true,
  "providers": [
    { "name": "tushare", "priority": 0, "status": "available", "description": "..." }
  ]
}
```

---

## 二、Alpha 研究 API (`/api/alpha`)

### 2.1 模块状态

#### `GET /api/alpha/status`
> 检查 Alpha 模块安装状态。

```json
{ "installed": true, "version": "0.1.0", "lab_path": "...", "lab_exists": true }
```

### 2.2 任务管理

#### `GET /api/alpha/tasks`
> 获取所有任务列表。

```json
[
  {
    "task_id": "a1b2c3d4",
    "type": "data_download",
    "title": "下载 d 原始K线",
    "entity_type": "data",
    "entity_name": "000001.SZSE",
    "status": "running",
    "progress": 45.0,
    "message": "已下载 000001.SZSE (1/3)",
    "result": null,
    "created_at": "2026-06-04T10:00:00",
    "updated_at": "2026-06-04T10:00:05"
  }
]
```

#### `GET /api/alpha/tasks/{task_id}`
> 查询单个任务状态。

### 2.3 数据下载

#### `POST /api/alpha/data/download`
> 从 Tushare 下载原始行情数据，异步执行。

**请求体**:
```json
{
  "vt_symbols": ["000001.SZSE", "600000.SSE"],
  "start": "2024-01-01",
  "end": "2024-12-31",
  "data_kind": "bar",
  "source_interval": "d"
}
```

**参数说明**:

| 字段 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `vt_symbols` | ✅ | string[] | 合约标识列表 |
| `start` | ✅ | date | 开始日期 |
| `end` | ✅ | date | 结束日期 |
| `data_kind` | 否 | string | `bar` (默认) |
| `source_interval` | 否 | string | `d`/`1m`/`5m`/`15m`/`30m`/`60m`/`w` |

**响应**: `{ "task_id": "xxx", "message": "下载任务已启动" }`

### 2.4 数据聚合

#### `POST /api/alpha/data/aggregate`
> 将本地原始数据聚合为派生周期。

**请求体**:
```json
{
  "vt_symbols": ["000001.SZSE"],
  "start": "2024-01-01",
  "end": "2024-12-31",
  "source_kind": "bar",
  "source_interval": "1m",
  "target_interval": "5m",
  "session_profile": "cn_equity"
}
```

### 2.5 数据资源管理

#### `GET /api/alpha/data/resources`
> 获取全部数据资源列表（原始 Bar / 原始 Tick / 派生 Bar）。

```json
{
  "raw_bars": [
    { "key": "000001.SZSE_d", "vt_symbol": "000001.SZSE", "interval": "d", "row_count": 1200, ... }
  ],
  "raw_ticks": [...],
  "derived_bars": [...]
}
```

#### `GET /api/alpha/data/resources/{kind}/{key}`
> 查看单个数据资源的详情和预览数据。

**路径参数**:
- `kind`: `raw_bar` / `raw_tick` / `derived_bar`
- `key`: 资源标识 (如 `000001.SZSE_d`)

**查询参数**: `?limit=100&before=2024-06-01T00:00:00`

#### `DELETE /api/alpha/data/resources/{kind}/{key}`
> 删除数据资源。

### 2.6 数据集管理

#### `GET /api/alpha/datasets`
> 列出所有数据集名称。

**响应**: `["my_dataset_1", "alpha158_test"]`

#### `GET /api/alpha/datasets/{name}`
> 获取数据集详情。

```json
{
  "name": "my_dataset",
  "feature_count": 158,
  "sample_count": 12500,
  "label_expression": "ts_delay(close, -3) / ts_delay(close, -1) - 1"
}
```

#### `POST /api/alpha/datasets/create`
> 创建因子数据集，异步执行。

**请求体**:
```json
{
  "name": "test_ds",
  "vt_symbols": ["000001.SZSE", "600519.SSE"],
  "start": "2022-01-01",
  "end": "2024-12-31",
  "train_end": "2024-06-30",
  "valid_end": "2024-09-30",
  "features": ["alpha158"],
  "label_period": 3
}
```

#### `DELETE /api/alpha/datasets/{name}`

### 2.7 模型管理

#### `GET /api/alpha/models`
> 列出所有 Alpha 模型。

#### `GET /api/alpha/models/{name}`
> 获取模型详情。

```json
{ "name": "lgb_v1", "model_type": "LgbModel" }
```

#### `POST /api/alpha/models/train`
> 启动模型训练，异步执行。

**请求体**:
```json
{
  "name": "lgb_v1",
  "dataset": "test_ds",
  "model_type": "lgb",
  "params": { "num_boost_round": 200, "learning_rate": 0.05 }
}
```

**支持的 `model_type`**: `lgb` (LightGBM), `mlp` (PyTorch MLP), `lasso` (Lasso 回归)

#### `DELETE /api/alpha/models/{name}`

### 2.8 信号管理

#### `GET /api/alpha/signals`
#### `GET /api/alpha/signals/{name}`

```json
{
  "name": "sig_v1",
  "row_count": 5000,
  "columns": ["datetime", "vt_symbol", "signal"],
  "preview": [...]
}
```

#### `POST /api/alpha/signals/generate`
> 使用训练好的模型生成预测信号。

```json
{
  "name": "sig_v1",
  "model": "lgb_v1",
  "vt_symbols": ["000001.SZSE", "600519.SSE"],
  "start": "2024-07-01",
  "end": "2024-12-31"
}
```

#### `DELETE /api/alpha/signals/{name}`

### 2.9 策略回测

#### `POST /api/alpha/backtest/run`
> 基于信号运行策略回测。

```json
{
  "name": "bt_v1",
  "signal": "sig_v1",
  "capital": 1000000,
  "start": "2024-07-01",
  "end": "2024-12-31",
  "benchmark": "000300.SSE"
}
```

**响应** (任务完成后 `result` 字段):
```json
{
  "name": "bt_v1",
  "statistics": {
    "total_return": 0.15,
    "annual_return": 0.28,
    "max_drawdown": -0.08,
    "sharpe_ratio": 2.1,
    "total_trade_count": 120,
    ...
  },
  "trades": [
    {
      "datetime": "2024-07-03T15:00:00",
      "vt_symbol": "000001.SZSE",
      "direction": "LONG",
      "offset": "OPEN",
      "price": 12.34,
      "volume": 1000.0
    }
  ],
  "equity_curve": [
    { "date": "2024-07-03", "balance": 1012300.0, "drawdown": 0.0, "ddpercent": 0.0, "net_pnl": 12300.0 }
  ]
}
```

> **`trades` / `equity_curve`（图表可视化特性新增）**：回测 `task.result` 在既有 `statistics` 基础上额外回传两条过程数据，供前端绘制净值/回撤曲线与 K 线买卖点。**Alpha 回测（`/api/alpha/backtest/run`）与 CNN 回测（`/api/cnn/backtest/run`）两条路径结构一致**，字段恒在（无成交/净值时为 `[]`）。
>
> - `trades`：逐笔成交，由回测引擎 `engine.trades`（`TradeData`）序列化得到，按 `datetime` 升序。
>   - `datetime` string — 成交时间，ISO 字符串（前端可直接解析）
>   - `vt_symbol` string — 标的代码
>   - `direction` string — 多空方向，`LONG` / `SHORT`
>   - `offset` string — 开平方向，`OPEN`(买入) / `CLOSE`(卖出)
>   - `price` number — 成交价
>   - `volume` number — 成交量
> - `equity_curve`：逐日净值，由 `engine.daily_df`（经 `calculate_statistics` 补 `balance/drawdown/ddpercent` 列后）序列化得到。爆仓（净值不可计算）时返回 `[]` 并保留 `statistics.error`。
>   - `date` string — 交易日，`YYYY-MM-DD`
>   - `balance` number — 当日账户净值
>   - `drawdown` number — 当日回撤（绝对额）
>   - `ddpercent` number — 当日回撤百分比
>   - `net_pnl` number — 当日净盈亏

### 2.10 合约配置

#### `GET /api/alpha/contracts`
#### `POST /api/alpha/contracts`

```
?vt_symbol=000001.SZSE&long_rate=0.0003&short_rate=0.0003&size=1&pricetick=0.01
```

### 2.11 CSV 导入

#### `POST /api/alpha/bar-data/import/preview`
> 预览 Bar CSV 文件映射情况。

**Content-Type**: `multipart/form-data`
- `file`: CSV 文件
- `field_mapping`: 可选的 JSON 字段映射

#### `POST /api/alpha/bar-data/import`
> 执行 Bar CSV 导入。

**Form 参数**:
- `file`: CSV 文件
- `interval`: `d`/`1m`/`5m`/`15m`/`30m`/`60m`
- `import_mode`: `merge`(追加) / `replace`(替换)
- `field_mapping`: 可选

#### `POST /api/alpha/ticks/import/preview`
> 预览 Tick CSV 文件。

#### `POST /api/alpha/ticks/import`
> 执行 Tick CSV 导入。

### 2.12 K 线数据 (兼容旧前端)

#### `GET /api/alpha/bar-data`
> 获取日线和 1 分钟原始 K 线列表。

#### `GET /api/alpha/bar-data/{interval}/{vt_symbol}`
> 获取单个合约的 K 线详情，支持游标分页。

#### `DELETE /api/alpha/bar-data/{interval}/{vt_symbol}`

---

## 三、CNN 预测 API (`/api/cnn`)

### 3.1 状态检查

#### `GET /api/cnn/status`
```json
{ "torch_installed": true, "device": "cpu" }
```

### 3.2 训练管理

#### `POST /api/cnn/train`
> 启动 CNN 模型训练，异步执行。

**请求体**:
```json
{
  "name": "cnn_v1",
  "start": "2022-01-01",
  "end": "2024-12-31",
  "target_symbol": "000001.SZSE",
  "input_data_kind": "bar",
  "input_interval": "d",
  "observation_groups": [
    { "role": "market", "name": "大盘", "symbols": ["000300.SSE"] },
    { "role": "sector", "name": "银行板块", "symbols": ["601398.SSE", "601288.SSE"] }
  ],
  "label_spec": { "mode": "next_bar" },
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "lookback": 30,
  "dropout": 0.5,
  "train_ratio": 0.7
}
```

**完整参数**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | 必填 | 模型名称 |
| `start` / `end` | date | 必填 | 训练数据时间范围 |
| `target_symbol` | string | 可选 | 预测目标，默认取第一个 |
| `input_data_kind` | string | `bar` | 输入数据类型 |
| `input_interval` | string | `d` | 输入周期 |
| `observation_groups` | array | `[]` | 语义观测分组 |
| `label_spec` | object | `{mode: "next_bar"}` | 标签定义 |
| `epochs` | int | 50 | 训练轮数 |
| `batch_size` | int | 32 | 批大小 |
| `learning_rate` | float | 0.001 | 学习率 |
| `lookback` | int | 30 | 回看窗口 |
| `dropout` | float | 0.5 | Dropout 率 |
| `train_ratio` | float | 0.7 | 训练集比例 |

### 3.3 模型管理

#### `GET /api/cnn/models`
```json
[
  {
    "name": "cnn_v1",
    "size_mb": 0.45,
    "created_at": "2026-06-04T10:00:00",
    "best_epoch": 35,
    "best_val_loss": 0.6832,
    "target_symbol": "000001.SZSE",
    "group_count": 3,
    "observation_groups": [...]
  }
]
```

#### `GET /api/cnn/models/{name}`
> 获取模型详情（含完整训练历史）。

#### `DELETE /api/cnn/models/{name}`

### 3.4 模型回测

#### `POST /api/cnn/backtest/run`
> 用训练好的 CNN 模型直接驱动策略回测，异步执行，返回 `task_id`；结果写入 `task.result`。

**请求体**:
```json
{
  "name": "cnn_bt_v1",
  "model": "cnn_v1",
  "capital": 1000000,
  "start": "2024-07-01",
  "end": "2024-12-31",
  "buy_threshold": 0.6,
  "sell_threshold": 0.4,
  "commission_rate": 0.0003,
  "stamp_duty": 0.001,
  "slippage": 0.0005,
  "price_add": 0.002,
  "exit_mode": "threshold",
  "hold_days": 1,
  "take_profit": 0.0,
  "stop_loss": 0.0,
  "t_plus1": false
}
```

**主要参数**:

| 字段 | 必填 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `name` | ✅ | string | — | 回测名称 |
| `model` | ✅ | string | — | CNN 模型名 |
| `capital` | 否 | float | 1000000 | 初始资金 |
| `start` / `end` | ✅ | date | — | 回测区间 |
| `buy_threshold` | 否 | float | 0.6 | 买入阈值（0~1） |
| `sell_threshold` | 否 | float | 0.4 | 卖出阈值（0~1） |
| `exit_mode` | 否 | string | `threshold` | 出场模式：`threshold`/`fixed_hold`/`oco`/`auto` |
| `hold_days` | 否 | int | 1 | `fixed_hold`/`oco` 的持有交易日数 |
| `take_profit` / `stop_loss` | 否 | float | 0.0 | `oco` 止盈/止损幅度，0=不启用 |
| `t_plus1` | 否 | bool | false | 是否启用 T+1 卖出限制 |

> 成本类参数 `commission_rate`（佣金率）/ `stamp_duty`（印花税）/ `slippage`（滑点）/ `price_add`（挂单价格缓冲）均有默认值，并随结果回传至 `statistics` 便于前端展示「成本假设」。

**响应**: `{ "task_id": "xxx", "message": "..." }`

**任务完成后 `result` 字段**:
```json
{
  "name": "cnn_bt_v1",
  "model": "cnn_v1",
  "target_symbol": "000001.SZSE",
  "statistics": { "total_return": 0.15, "max_drawdown": -0.08, "sharpe_ratio": 2.1, ... },
  "trades": [
    { "datetime": "2024-07-03T15:00:00", "vt_symbol": "000001.SZSE",
      "direction": "LONG", "offset": "OPEN", "price": 12.34, "volume": 1000.0 }
  ],
  "equity_curve": [
    { "date": "2024-07-03", "balance": 1012300.0, "drawdown": 0.0, "ddpercent": 0.0, "net_pnl": 12300.0 }
  ]
}
```

> **`trades` / `equity_curve`（图表可视化特性新增）**：与 Alpha 回测（`/api/alpha/backtest/run`，见 §2.9）**结构完全一致**，字段含义与序列化口径见 §2.9 的字段说明。CNN 回测结果额外含 `model` 与 `target_symbol`（前端 `BacktestCharts` 优先用 `target_symbol` 确定 K 线标的）。无成交时 `trades`/`equity_curve` 为 `[]` 且 `statistics.error` 提示「回测期间无成交」。

---

## 四、交易操作台 API (`/api/live`)

> 交易操作台（Trading Console）把既有实盘原语（`predict_cnn_signals` → `SignalService` → `RiskManager` → `Notifier` → `DecisionStore`，见 docs/08 迭代 6）经 `LiveSignalOrchestrator` 串联后暴露为 HTTP 接口，让用户在临近收盘时基于 CNN 预测产出「今日是否买入」的决策。
>
> ⚠️ **安全声明**：当前后端处于**无鉴权环境**（CORS=`["*"]`，对应技术债 TD-015）。本组接口**仅产出决策与提醒，不向任何券商网关提交真实订单**（no broker submission）。真实下单能力依赖**鉴权 + kill-switch UI** 前置条件，超出本特性范围（Requirement 7.3）。

### 4.1 触发今日决策

#### `POST /api/live/decision`
> 触发一次今日决策，异步执行，返回 `task_id`；进度经 `/ws` 的 `task` 主题推送。

**请求体**:
```json
{
  "model": "cnn_v1",
  "vt_symbol": "000001.SZSE",
  "scheme": "eod_buy_v1",
  "trade_date": "2026-06-08",
  "data_source": "pull",
  "portfolio": {
    "portfolio_value": 1000000,
    "total_position_value": 0,
    "current_position": 0,
    "current_symbol_value": 0
  },
  "risk": {
    "blacklist": [],
    "max_total_position_ratio": 0.95,
    "max_single_position_ratio": 0.30,
    "allow_when_halted": false
  },
  "buy_threshold": 0.6,
  "position_ratio": 0.95,
  "min_volume": 100,
  "model_version": "",
  "halted": false,
  "should_exit": false
}
```

**参数说明**:

| 字段 | 必填 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `model` | ✅ | string | — | CNN 模型名 |
| `vt_symbol` | ✅ | string | — | 目标标的 |
| `scheme` | ✅ | string | — | 方案名 |
| `trade_date` | 否 | date | 当天 | 决策日，缺省=当天 |
| `data_source` | 否 | string | `pull` | `upload`（先经 `/api/alpha/bar-data/import` 导入）/ `pull`（接口拉取） |
| `portfolio` | ✅ | object | — | 组合快照（总市值/总持仓市值/目标标的持仓股数与市值） |
| `risk` | 否 | object | 默认风控 | 黑名单/单票上限/总仓上限/停牌可否交易 |
| `buy_threshold` | 否 | float | 0.6 | 买入阈值 |
| `position_ratio` | 否 | float | 0.95 | 目标仓位比例 |
| `min_volume` | 否 | int | 100 | 最小成交手数 |
| `model_version` | 否 | string | `""` | 模型版本，参与 `signal_id` |
| `halted` | 否 | bool | false | 目标标的当日是否停牌/封死 |
| `should_exit` | 否 | bool | false | 是否触发出场 |

**响应**: `{ "task_id": "xxx", "message": "今日决策任务已启动" }`

**任务完成后 `result` 字段**:
```json
{
  "decision": {
    "signal_id": "2026-06-08:eod_buy_v1@v3",
    "trade_date": "2026-06-08",
    "scheme": "eod_buy_v1",
    "action": "buy",
    "vt_symbol": "000001.SZSE",
    "volume": 1000,
    "price": 12.34,
    "signal": 0.71,
    "reason": "概率达标且通过风控",
    "created_at": "2026-06-08T14:55:00"
  },
  "risk_detail": [
    { "check": "kill_switch_or_circuit", "passed": true,  "detail": "通过" },
    { "check": "blacklist",              "passed": true,  "detail": "通过" },
    { "check": "halted",                 "passed": true,  "detail": "通过" },
    { "check": "max_total_position",     "passed": true,  "detail": "拟新增后 ... vs 上限 ..." },
    { "check": "max_single_position",    "passed": false, "detail": "... 超单票上限 ..." }
  ],
  "idempotent_hit": false
}
```

**错误**:
- `400`：缺必填字段（`model`/`vt_symbol`/`scheme`）。
- `404`：CNN 模型不存在。
- 决策日行情缺失：任务 `status=FAILED`，`message` 含「行情缺失」（经 WS 推送，非 HTTP 错误）。

### 4.2 决策历史

#### `GET /api/live/decisions`
> 列出 `DecisionStore` 中已持久化决策的标识符集合。

**响应**: `{ "signal_ids": ["2026-06-08:eod_buy_v1@v3", ...] }`

#### `GET /api/live/decisions/{signal_id}`
> 按 `signal_id` 返回单条决策详情。

**响应**: 完整 `decision` 对象（字段同上 `result.decision`）。

**错误**: `404` — 决策不存在。

### 4.3 决策过程档案（决策过程可观测性）

#### `GET /api/live/decisions/{signal_id}/trace`
> 按 `signal_id` 返回该决策的**完整 Decision_Trace**（决策过程档案）。Decision_Trace 与决策 `{signal_id}.json` 并列、独立持久化为 sibling 文件 `{signal_id}.trace.json`，系统重启后仍可回溯（Requirement 8.2 / 8.4）。
>
> 过程档案按**六段 Trace_Section** 组织：运行头（`run_header`）/ 推理段（`inference`）/ 取价段（`pricing`）/ 决策逻辑段（`decision_logic`）/ 风控段（`risk`）/ 结果段（`result`）。
>
> 🔒 **脱敏红线**：过程档案**不含任何凭证与密钥**（含 Tushare token）；数据源仅以**类型 + bar 数量**记录；运行头的风控配置仅存**摘要**（比率 + 黑名单长度，不展开内容）。

**响应**:
```json
{
  "schema_version": 1,
  "run_id": "ab12cd34",
  "signal_id": "2026-06-08:eod_buy_v1@v3",
  "completed_sections": [
    "run_header", "inference", "pricing", "decision_logic", "risk", "result"
  ],
  "sections": {
    "run_header": {
      "run_id": "ab12cd34",
      "model_name": "cnn_v1", "model_version": "v3",
      "vt_symbol": "000001.SZSE", "scheme": "eod_buy_v1",
      "trade_date": "2026-06-08",
      "data_source_type": "pull",
      "buy_threshold": 0.6,
      "portfolio": { "portfolio_value": 1000000, "total_position_value": 0,
                     "current_position": 0, "current_symbol_value": 0 },
      "risk_config_summary": { "max_total_position_ratio": 0.95,
                               "max_single_position_ratio": 0.30,
                               "allow_when_halted": false, "blacklist_size": 0 }
    },
    "inference": {
      "target_symbol": "000001.SZSE", "lookback": 240, "input_interval": "30m",
      "objective": "classification", "observation_symbols": ["000001.SZSE"],
      "observation_group_count": 1, "warmup_start": "2025-06-08",
      "total_steps": 480, "valid_points": 240,
      "per_symbol_bars": { "000001.SZSE": 480 },
      "signal_seq_stats": { "count": 1, "mean": 0.71, "min": 0.71, "max": 0.71 },
      "decision_day_signal": 0.71
    },
    "pricing": { "interval_used": "d", "close_price": 12.34 },
    "decision_logic": {
      "signal": 0.71, "buy_threshold": 0.6, "signal_passed": true,
      "target_value": 950000, "volume": 1000, "intended_value": 12340,
      "should_exit": false, "halted": false
    },
    "risk": {
      "records": [
        { "check": "kill_switch_or_circuit", "passed": true, "detail": "通过" },
        { "check": "blacklist", "passed": true, "detail": "通过" },
        { "check": "halted", "passed": true, "detail": "通过" },
        { "check": "max_total_position", "passed": true, "detail": "..." },
        { "check": "max_single_position", "passed": true, "detail": "..." }
      ],
      "authoritative_ok": true
    },
    "result": {
      "action": "buy", "volume": 1000, "price": 12.34, "reason": "概率达标且通过风控",
      "idempotent_hit": false, "notified": true,
      "signal_id": "2026-06-08:eod_buy_v1@v3",
      "trace_persisted": true, "trace_persist_error": null, "abort_reason": null
    }
  }
}
```

**字段说明**:

| 字段 | 说明 |
|------|------|
| `completed_sections` | 已完成段（顺序）。正常运行为全六段；**产出 Decision 前因错误中止**（行情/信号缺失等）时仅为失败点之前的前缀，结果段记 `abort_reason` 且不含成功决策字段（Requirement 8.11）。 |
| `sections.result.idempotent_hit` | 是否幂等命中（同 `signal_id` 第二次触发）。幂等命中**不写入新的 Decision_Trace**，返回首次已持久化的档案（Requirement 8.9 / 8.10）。 |
| `sections.result.trace_persisted` | 过程档案是否持久化成功。持久化为 **best-effort**：失败仅告警并置 `false`、附 `trace_persist_error`，**绝不影响 Decision 落盘与返回**（Requirement 8.12）。 |
| `sections.result.abort_reason` | 中止原因；正常完成为 `null`（Requirement 8.11）。 |

**错误**: `404` — 决策过程档案不存在，`detail` 为「决策过程档案不存在: {signal_id}」（Requirement 8.5）。

---

## 五、WebSocket (`/ws`)

### 连接
```javascript
const ws = new WebSocket('ws://localhost:8000/ws')
```

### 客户端消息

**订阅主题**:
```json
{ "action": "subscribe", "topics": ["task", "tick"] }
```

**取消订阅**:
```json
{ "action": "unsubscribe", "topics": ["tick"] }
```

**心跳**:
```json
{ "action": "ping" }
// 响应: { "action": "pong" }
```

### 服务端推送

所有推送消息按主题分发。连接建立时默认订阅全部主题：
`tick`, `order`, `trade`, `position`, `account`, `contract`, `log`, `task`

---

## 六、错误响应格式

所有错误统一使用 FastAPI HTTPException：

```json
{
  "detail": "错误描述信息"
}
```

| HTTP 状态码 | 场景 |
|------------|------|
| 400 | 参数校验失败 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |
| 503 | Alpha 模块未安装 |

---

## 七、API 路由速查表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 系统状态 |
| GET | `/api/alpha/status` | Alpha 模块状态 |
| GET | `/api/alpha/tasks` | 任务列表 |
| GET | `/api/alpha/tasks/{id}` | 任务详情 |
| POST | `/api/alpha/data/download` | 数据下载 |
| POST | `/api/alpha/data/aggregate` | 周期聚合 |
| GET | `/api/alpha/data/resources` | 资源列表 |
| GET | `/api/alpha/data/resources/{kind}/{key}` | 资源详情 |
| DELETE | `/api/alpha/data/resources/{kind}/{key}` | 删除资源 |
| GET | `/api/alpha/datasets` | 数据集列表 |
| GET | `/api/alpha/datasets/{name}` | 数据集详情 |
| POST | `/api/alpha/datasets/create` | 创建数据集 |
| DELETE | `/api/alpha/datasets/{name}` | 删除数据集 |
| GET | `/api/alpha/models` | 模型列表 |
| GET | `/api/alpha/models/{name}` | 模型详情 |
| POST | `/api/alpha/models/train` | 训练模型 |
| DELETE | `/api/alpha/models/{name}` | 删除模型 |
| GET | `/api/alpha/signals` | 信号列表 |
| GET | `/api/alpha/signals/{name}` | 信号详情 |
| POST | `/api/alpha/signals/generate` | 生成信号 |
| DELETE | `/api/alpha/signals/{name}` | 删除信号 |
| POST | `/api/alpha/backtest/run` | 运行回测（result 含 trades/equity_curve） |
| GET | `/api/alpha/contracts` | 合约配置 |
| POST | `/api/alpha/contracts` | 添加合约 |
| GET | `/api/alpha/bar-data` | K线列表 |
| GET | `/api/alpha/bar-data/{interval}/{symbol}` | K线详情 |
| DELETE | `/api/alpha/bar-data/{interval}/{symbol}` | 删除K线 |
| POST | `/api/alpha/bar-data/import/preview` | CSV预览 |
| POST | `/api/alpha/bar-data/import` | CSV导入 |
| POST | `/api/alpha/ticks/import/preview` | Tick预览 |
| POST | `/api/alpha/ticks/import` | Tick导入 |
| GET | `/api/cnn/status` | CNN 状态 |
| POST | `/api/cnn/train` | CNN 训练 |
| GET | `/api/cnn/models` | CNN 模型列表 |
| GET | `/api/cnn/models/{name}` | CNN 模型详情 |
| DELETE | `/api/cnn/models/{name}` | 删除 CNN 模型 |
| POST | `/api/cnn/backtest/run` | 运行 CNN 回测（result 含 trades/equity_curve） |
| GET | `/api/cnn/governance/config` | CNN 治理配置 |
| PUT | `/api/cnn/governance/config` | 更新 CNN 治理配置 |
| GET | `/api/cnn/governance/production` | 当前生产 CNN 模型 |
| GET | `/api/cnn/governance/candidates` | CNN 候选模型列表 |
| POST | `/api/cnn/governance/evaluate` | 启动 WF/OOS 评估 |
| POST | `/api/cnn/governance/candidates/train` | 训练候选模型 |
| POST | `/api/cnn/governance/candidates/{id}/promote` | 晋级候选模型 |
| POST | `/api/cnn/governance/candidates/{id}/reject` | 拒绝候选模型 |
| POST | `/api/cnn/governance/rollback` | 回滚生产 CNN 模型 |
| POST | `/api/cnn/governance/replay/run` | 启动治理回放回测 |
| GET | `/api/cnn/governance/replay` | 治理回放报告列表 |
| GET | `/api/cnn/governance/replay/{id}` | 治理回放报告详情 |
| POST | `/api/live/decision` | 触发今日决策（异步，不下单） |
| GET | `/api/live/decisions` | 决策列表 |
| GET | `/api/live/decisions/{signal_id}` | 决策详情 |
| GET | `/api/live/decisions/{signal_id}/trace` | 决策过程档案（六段 Decision_Trace） |
| GET | `/api/live/plans` | 交易计划列表（摘要，含启用状态与最近触发日） |
| POST | `/api/live/plans` | 创建交易计划 |
| GET | `/api/live/plans/{plan_id}` | 交易计划详情 |
| PUT | `/api/live/plans/{plan_id}` | 更新交易计划 |
| DELETE | `/api/live/plans/{plan_id}` | 删除交易计划 |
| PATCH | `/api/live/plans/{plan_id}/enabled` | 启用/停用交易计划 |
| POST | `/api/live/plans/{plan_id}/run` | 按计划立即触发今日决策（异步，不下单） |
| GET | `/api/live/scheduler/status` | 调度器运行状态 |
| WS | `/ws` | WebSocket |

### 交易计划自动化（Trading Plan Automation）

把一次决策所需的完整配置保存为可复用、可编辑、可启用/停用的「交易计划」，
支持手动「按计划触发」与进程内调度器在决策时点自动触发。**仅产出决策与提醒，
不向任何券商网关提交真实订单。**

- **通知通道**：`notify_channels` 仅含通道名（`dingtalk` / `wecom` / `serverchan` /
  `webhook`）。各通道的 webhook/secret/token **只经环境变量读取**
  （`AITRADE_NOTIFY_DINGTALK_WEBHOOK` 等），不入计划、响应、日志、trace。
  无任一通道配置凭证时退回 `LogNotifier`（写日志兜底）。
- **无前视约束**：`data_basis=closed_t` 时 `decision_time` 必须 ≥ `15:00`（收盘后），
  否则创建/更新返回 422。
- **幂等与恢复**：同一计划同一交易日至多自动触发一次；触发记录持久化到
  `RuntimeStateStore`，进程重启当日不重复触发。
- **调度开关**：环境变量 `AITRADE_SCHEDULER_ENABLED`（默认 true）、
  `AITRADE_SCHEDULER_TICK_SECONDS`（默认 30）。单机经 `SingleInstanceLock` 防并发。

创建计划请求体（`POST /api/live/plans`）示例：

```json
{
  "name": "平安银行尾盘买入计划",
  "model": "cnn_demo",
  "vt_symbol": "000001.SZSE",
  "scheme": "eod_buy_v1",
  "decision_time": "15:05",
  "data_basis": "closed_t",
  "notify_channels": ["dingtalk"],
  "enabled": true,
  "buy_threshold": 0.6,
  "position_ratio": 0.95,
  "min_volume": 100,
  "data_source": "pull",
  "portfolio": { "portfolio_value": 1000000 },
  "risk": { "max_total_position_ratio": 0.95, "max_single_position_ratio": 0.3 }
}
```

`GET /api/live/scheduler/status` 响应：

```json
{ "running": true, "tick_seconds": 30, "enabled_plan_count": 2, "last_triggered": { "<plan_id>": "2026-06-09" } }
```
