# CNN Label 可配置化与多方案迭代计划

> 本文档是一份**活文档（living doc）**，用于规划「CNN label 可配置化 + 多量化方案解耦复用」的迭代落地，并在每轮迭代后回来更新状态。
>
> 背景缘起：要做一个能实际应用的「快收盘时判断是否买入」方案。核心结论是——**label 必须随交易方案改变，且要与策略出场、回测撮合严格配对**，因此 label 可配置会牵动整条链路，需要分阶段、成体系地推进。
>
> 关联文档：[02-后端模块设计](02-后端模块设计.md)、[数据回测模块](数据回测模块.md)、[cnn模块](cnn模块.md)、[06-迭代规划](06-迭代规划.md)。
> 关联审查技能：`.cursor/skills/quant-backtest-live-alignment/`（回测-实盘一致性）。

---

## 一、背景与目标

### 1.1 需求

先落地一个**可实际应用**的模式：根据前一段时间的交易数据，在**快收盘时**判断是否买入。后续还会有更多类似的量化方案，需要在系统中**复用公共能力**，同时**解耦、分支独立**各方案。

### 1.2 现状（已具备的地基）

链路其实已经接通：

| 环节 | 实现 | 文件 |
|------|------|------|
| 前一段数据 → 特征 | lookback 滑动窗口 + 6 通道特征 | `cnn/features.py`、`cnn/dataset.py` |
| 涨跌概率 | CNN 推理产出 0~1 概率 | `cnn/predictor.py` |
| 是否买入 | 概率阈值买卖（单标的） | `cnn/strategy.py` |
| 回测 | 共享回测引擎（撮合/盯市/统计） | `backtest/engine.py` |
| 一键跑 | `/api/cnn/predict`、`/api/cnn/backtest/run` | `api/cnn.py` |

统一信号契约：`[datetime, vt_symbol, signal]`（signal 为 0~1 概率）。

### 1.3 现状局限（本计划要解决的）

1. **label 写死为「收盘价 → 收盘价」方向二分类**：`cnn/dataset.py` 中 `base_close` 与 `future_close` 都取 `close`，4 个 `LabelMode` 只改了「未来对齐到哪根 bar」。无法表达「不同入场/出场规则」的方案。
2. **策略出场与 label 未对齐**：`CNNSignalStrategy` 出场是「概率 < sell_threshold 才卖」，与任何 label 定义都不配对 → 回测与实盘必然背离。
3. **成本不完整**：回测扣了佣金，但缺印花税（卖出）、滑点压测、T+1 卖出限制。
4. **缺方案（Scheme）抽象**：新方案容易 fork 引擎或互相污染。

---

## 二、核心原则（务必遵守）

### 2.1 label 语义 = 这笔交易「算赚钱」的定义

由三件事共同决定，而不只是一个枚举：

- **入场时点（entry）**：哪根 bar 的什么价入场。
- **出场规则（exit）**：怎么卖（次日收盘 / 次日开盘 / 持有 N 日 / 止盈止损挂单）。
- **盈利门槛（threshold）**：要把成本（佣金 + 印花税 + 滑点）算进去，**不能用裸 `> 0`**。

### 2.2 label-trade 一致性（黄金法则，红线）

```
label 的出场假设  ≡  策略的真实出场逻辑  ≡  回测撮合的真实成交规则
```

三者任一不一致，回测就会虚高、实盘会翻车。典型反例：label 用「次日最高点卖」（隐含能卖在最高点），实盘做不到 → 严禁。

### 2.3 无前视（look-ahead）红线

- `signal.datetime` ≤ 计算它所用的最后一根 bar 的 datetime。
- 成交 datetime > 对应信号的 datetime。
- label 用未来数据是正常的（监督学习本就如此），但**严禁泄漏进特征 X**。
- 现状已满足：特征取 `aligned[:, anchor-lookback+1 : anchor+1, :, :]`，label 取 `future_index > anchor`，时点干净。

### 2.4 「快收盘买入」的两种时点语义

| 方案 | 数据 | 执行 | 说明 |
|------|------|------|------|
| A | 日线 | T 收盘决策 → 次日开盘买 | 引擎默认（先 cross_order 再 on_bars，T+1 成交），最省事，但不是当天尾盘 |
| B | 分钟线 5m/15m | 14:45 决策 → 14:45~14:55 买 | 真·尾盘买入，label 用 `session_close`，模型已支持，无需改网络 |

> 坑：方案 B 若信号落在当日**最后一根** bar，「下一根成交」会跨到次日变 T+1，信号需比收盘 bar 早一根。

---

## 三、影响地图：哪些流程必须跟着 label 一起配置

| 环节 | 是否要改 | 改什么 | 一致性原因 |
|------|:---:|------|------|
| Label 引擎 `cnn/dataset.py` | ✅ 核心 | entry/exit/threshold 参数化，抽 `LabelBuilder` | 源头 |
| 训练 Schema `models/alpha.py` | ✅ | `LabelSpec` 加字段 + 校验 | 前端要能传配置 |
| Checkpoint `cnn/trainer.py` | ✅ | 持久化完整 label_spec | 推理/回测读回同一定义 |
| 推理 `cnn/predictor.py` | ◻️ 小改 | 信号 datetime 随 entry 时点对齐 | 信号≠出场，prob 够用 |
| 策略 `cnn/strategy.py` | ✅ 重点 | 出场逻辑参数化，对齐 label.exit | label 说怎么卖策略就怎么卖 |
| 回测撮合/成本 `backtest/engine.py` | ✅ | 印花税 + 可配滑点 +（可选）T+1 | 成本不全 = 回测虚高 |
| 回测 Schema `CNNBacktestRequest` | ✅ | 传出场规则 + 成本参数 | 回测要能复现 scheme |
| Scheme 配置层（新增） | ✅ | 绑定 predictor + label + strategy + 成本 | 复用/解耦/分支独立落点 |
| 前端表单 | ✅ | 训练 label 配置 + 回测出场/成本 + scheme 管理 | 让配置化真正可用 |
| 验证/质量 | ✅ | 对齐自检 + 一致性断言 + 成本压测 + 样本外 + 纸面对账 | 防回归、防上线翻车 |

**一句话**：改 label 的同时，必须配套改「策略出场 + 回测撮合成本 + Scheme 绑定 + 验证」。

---

## 四、分阶段迭代计划

> 排序原则：**先立一致性地基，再加可配置功能**。越灵活的 label 越会放大「回测好看实盘翻车」。

### 迭代 0 · 地基对齐（P0，必须先做）

- **目标**：在最简单场景下让「label ↔ 策略出场 ↔ 撮合成交」严格一致，并补全成本。
- **改动**：
  - `cnn/strategy.py`：`CNNSignalStrategy` 增加 `fixed_hold` 固定持有出场模式（`hold_days` 个交易日后强制平仓，出场不看信号）。
  - `backtest/engine.py`：成本补印花税（卖出）、可配滑点；新增 T+1 卖出限制开关。
- **验收标准**：
  - `scripts/cnn_backtest_alignment_check.py` 有成交；
  - 固定持有出场确定性：随机抽交易日核对，策略出场日与 label 固定持有期一致。
- **风险等级**：🔴 高（不做这步，后续全部失真）。
- **状态**：✅ 完成（2026-06-06）

**实现说明（v0.2）**：

- **成本补全**：经核对，印花税（仅卖出）、每笔不利滑点已由前序改动接通全链路——`engine.__init__`/`set_parameters`/`cross_order`/`calculate_result` → `pnl.py(stamp_rate)` → `api/cnn.py`（按 `CNNBacktestRequest` 的 `commission_rate/stamp_duty/slippage` 设置并回传统计）。本次新增 T+1 开关。
- **固定持有出场（fixed_hold）**：`exit_mode="fixed_hold"` 时，概率 > `buy_threshold` 建仓，建仓成交后**固定持有 `hold_days` 个交易日强制平仓，出场不依赖信号**。语义精确定义：持有计数从建仓成交日起算（建仓日记为第 1 日），计满 `hold_days` 当根触发出场决策，按共享引擎「下一根 bar 成交」撮合。即 **出场成交日 = 建仓成交日 + `hold_days` 个交易日**（如 `hold_days=1`：D1 建仓 → D2 平仓）。
- **T+1 开关**：`engine.t_plus1=True` 时，`buy_dates` 记录建仓成交日，`cross_order` 拦截「当日买入当日卖出」的卖单（订单保留待次日撮合）。默认关闭，向后兼容。
- **接线**：`CNNBacktestRequest` 新增 `exit_mode / hold_days / t_plus1`，`_run_cnn_backtest` 注入策略与引擎并回传统计。默认 `exit_mode="threshold"`，旧行为不变。
- **执行近似（遗留，迭代 1 收敛）**：fixed_hold 出场按「下一根 bar 开盘」成交，与 label 的「收盘价」口径存在一根 bar 偏差；精确口径由 `LabelSpec.price_ref`（close / next_open）在迭代 1 对齐。
- **测试**：`backend/tests/test_cnn_strategy_iteration0.py`（5 项）——固定持有往返、hold_days=2、印花税降低净盈亏、T+1 拦截、阈值模式兼容；后端全量 23 passed 无回归。

### 迭代 1 · Label 可配置化（核心）

- **目标**：label 支持「入场时点 + 出场规则 + 盈利门槛」三段配置。
- **改动**：
  - `models/alpha.py`：`LabelSpec` 增加 `price_ref`（close / next_open）计价口径。
  - `cnn/dataset.py`：`_compute_label_return` 实现两种口径；`_label_future_index` 定位出场 bar（next_bar / horizon_bars / session_close / next_session_close）；`_label_from_return` 做去噪 dead-zone。
  - `cnn/trainer.py`：完整 `label_spec` 写进 `train_config`（checkpoint）。
- **验收标准**：
  - 同一份数据、不同口径/出场配置 → 正样本比例不同且可解释；✅（`test_price_ref_changes_labels`：close 全涨、next_open 全跌）
  - checkpoint 读回 label_spec 无损；✅（trainer 持久化 + `test_label_spec_persisted_in_info`）
  - 无前视：特征 X 取 `… : anchor+1`，label 取未来 bar；next_open 越界返回 None。✅（多项纯函数测试）
- **风险等级**：🟡 中。
- **状态**：✅ 完成（2026-06-06）

**实现说明（v0.3）**：

- **计价口径 `price_ref`（核心）**：经核对已由前序改动接通——
  - `close`（旧/研究口径）：`close[anchor] → close[future]`。隐含「按观测到的收盘价成交」，实盘吃不到，仅供研究。
  - `next_open`（可执行口径）：`open[anchor+1] → open[future+1]`，对应「T 收盘出信号、T+1 开盘建仓、目标周期后开盘平仓」，剔除隔夜跳空错配；越界（无次开盘）返回 None 跳过，杜绝前视/越界。
- **出场 bar**：仍由 `mode` 决定（next_bar=anchor+1、horizon_bars=anchor+h、next_session_close=次日、session_close=当日尾，仅分钟）。`mode` 决定「哪根 bar」、`price_ref` 决定「该 bar 用收/开盘价 + 入场口径」，二者正交组合。
- **盈利门槛**：`threshold`（去噪 dead-zone，收益率口径）+ `neutral_policy`（drop / negative）。**注意：threshold 应人工设为 ≥ 单次往返成本**（佣金+印花税+滑点），把成本挡在标签外（自动 cost-aware 留待按需）。
- **本次（CH-3）增量**：新增 `backend/tests/test_cnn_label_iteration1.py`（14 项）——`_compute_label_return`（两口径+越界）、`_label_from_return`（dead-zone/drop/negative）、`_label_future_index`（next_bar/horizon/next_session_close）、`_normalize_label_spec` 兜底、build_dataset 集成（price_ref 改变标签、阈值去噪、label_spec 回传）。后端全量 **37 passed** 无回归。
- **遗留 / 按需扩展**：① 非对称 entry/exit（当前 price_ref 成对配置，够用）；② `LabelBuilder` 类化（当前为内聚函数，迭代 3 Scheme 需要时再抽）；③ 自动 cost-aware threshold。

> ⭐ **label ↔ 执行一致性矩阵（迭代 0+1 的关键产出，防回测实盘背离）**
>
> | 训练 label 配置 | 回测策略配置 | 实际成交口径 | 一致性 |
> |---|---|---|---|
> | `mode=next_bar` + `price_ref=next_open` | `exit_mode=fixed_hold, hold_days=1` | 买 open[T+1] → 卖 open[T+2] | ✅ 精确对齐 |
> | `mode=horizon_bars(h)` + `price_ref=next_open` | `exit_mode=fixed_hold, hold_days=h` | 买 open[T+1] → 卖 open[T+1+h] | ✅ 精确对齐 |
> | `mode=next_bar` + `price_ref=close` | （需按收盘价当根成交） | 实盘吃不到收盘价 | ⚠️ 仅研究，含前视执行假设 |
>
> 结论：**训练用 `price_ref=next_open` + `fixed_hold(hold_days=mode 的持有期)`，label 与执行精确一致**，迭代 0 遗留的「执行近似」已被消除。

### 迭代 2 · 策略与 label 配对 + 一致性自检

- **目标**：策略出场能逐一对齐迭代 1 的每种 exit，并用代码守住一致性。
- **改动**：
  - 新增 `cnn/consistency.py`：`label_holding_horizon` / `derive_strategy_exit_from_label` / `check_label_strategy_consistency`。
  - `CNNBacktestRequest.exit_mode` 增加 `auto`（按模型 label 自动推导对齐的固定持有出场）。
  - `_run_cnn_backtest`：读 checkpoint 的 `label_spec` → auto 推导 / 一致性校验 → 隐性失败守护。
- **验收标准**：
  - 一致性自检通过；✅（13 项单测）
  - 杜绝「无成交=0 还只 warn」的隐性失败（无成交/对不齐直接判失败）；✅（信号∩行情无交集→抛错）
- **风险等级**：🔴 高（回测可信度的总闸）。
- **状态**：✅ 完成（2026-06-06）

**实现说明（v0.4）**：

- **一致性自检模块 `cnn/consistency.py`**：
  - `label_holding_horizon(label_spec, interval)`：解析 label 蕴含的固定持有期（next_bar→1、horizon_bars→h、日线 next_session_close→1、分钟 session_close/next_session_close→None）。
  - `derive_strategy_exit_from_label(...)`：由 label 自动推导 `{exit_mode: fixed_hold, hold_days: horizon}`；持有期非固定时抛错。
  - `check_label_strategy_consistency(label_spec, exit_mode, hold_days, interval)`：**硬性不一致抛错**（fixed_hold 的 hold_days≠label 持有期、或对非固定持有 label 用 fixed_hold）；**软性问题返回告警**（`price_ref=close` 研究口径、`threshold` 信号衰减式出场）。
- **回测接线 `_run_cnn_backtest`**：
  - 从 checkpoint 读 `train_config.label_spec`；
  - `exit_mode="auto"` → 调 `derive_*` 自动对齐；否则用请求值；
  - 调 `check_*`：硬性不一致 → 抛错（任务 FAILED，杜绝「跑出来但口径错」）；软性 → 收集到 `statistics["consistency_warnings"]`；
  - **隐性失败守护**：`load_data` 后校验「信号 datetime ∩ 行情 datetime」非空，无交集（P0 对齐类问题）→ 抛错并说明疑似周期不一致；
  - 统计回传 `label_spec` / `consistency_warnings` / 解析后的 `exit_mode`、`hold_days`。
- **测试**：`backend/tests/test_cnn_consistency_iteration2.py`（13 项）覆盖持有期解析、自动推导、对齐/不一致/告警/抛错全分支。后端全量 **50 passed** 无回归。
- **用法**：训练用 `price_ref=next_open` + `mode=next_bar/horizon_bars`，回测设 `exit_mode="auto"`，即自动得到与 label 精确对齐的 `fixed_hold`，并通过一致性自检。

### 迭代 3 · Scheme 配置层（复用 / 解耦落地）

- **目标**：把「一个量化方案」做成一条可持久化配置，方案间互不污染。
- **改动**：
  - `backtest/registry.py`：`SignalProvider` 协议（`predict()->signal_df`）+ 策略注册表（`register_strategy`/`get_strategy`/`list_strategies`）。
  - `cnn/strategy.py`：`CNNSignalStrategy` 自动注册为 `cnn_signal`。
  - `backtest/scheme.py`：`Scheme`/`PredictorConfig`/`StrategyConfig`/`CostConfig` + `SchemeStore`(JSON 持久化) + `run_scheme_backtest`(按方案驱动共享引擎)。
  - `config.py`：新增 `SCHEME_PATH`。
- **验收标准**：
  - 两个不同 scheme 独立跑通、结果互不影响；✅（`test_two_schemes_run_independently`：不同 hold_days 结果不同且可复现）
  - 新增 scheme **不改引擎代码**；✅（run_scheme_backtest 无任何方案/模型硬编码）
- **风险等级**：🟡 中。
- **状态**：✅ 完成（2026-06-06）

**实现说明（v0.5）**：信号契约腰线落地——预测器(SignalProvider) → `[datetime,vt_symbol,signal]` → 策略(注册表按名取) → 共享引擎(BarDataLoader)。一个 Scheme = 预测器配置 + label 口径 + 策略名与参数 + 标的/周期/成本，存为独立 JSON。共享地基不变，新增方案=新增配置(+可选独立策略子类)，分支独立、互不污染。测试 `backend/tests/test_scheme_iteration3.py`（6 项）。

### 迭代 4 · 进阶 exit：止盈止损 OCO

- **目标**：支持「挂止盈 + 止损」这类路径依赖出场。
- **改动**：
  - `backtest/engine.py`：新增 `sell_to_close_intrabar(vt_symbol, price, volume)`——当根 bar 内按触发价直接平多（含佣金/印花税/现金结算）。
  - `cnn/strategy.py`：`exit_mode="oco"`（`take_profit`/`stop_loss` + `hold_days` 最大持有回退），按当根 high/low 触发，**保守假设止损先到**。
  - `models/alpha.py`/`api/cnn.py`：`CNNBacktestRequest` 增加 `take_profit`/`stop_loss`，并注入策略。
  - `cnn/consistency.py`：识别 `oco` 为路径依赖出场，给出软性告警。
- **验收标准**：止盈/止损/同根双触发(止损优先)/最大持有回退 均符合预期；✅（4 项单测）
- **风险等级**：🟠 中高（日内撮合假设是新乐观假设来源，重点审查）。
- **状态**：✅ 完成（2026-06-06）

**实现说明（v0.6）**：止盈/止损为触发价当根成交（`sell_to_close_intrabar`），区别于限价单的下一根撮合；最大持有期回退沿用 fixed_hold 的下一根成交。**残余风险**：同一根 bar 内 high/low 先后未知，保守取止损先到——仍是近似，需纸面交易复核触发价是否真能成交；T+1 开启时当日买入当日不触发止损（合规优先）。测试 `backend/tests/test_cnn_oco_iteration4.py`（4 项）。

### 迭代 5 · 验证闭环 + 前端

- **目标**：上线前质量闭环 + 让配置真正可用。
- **改动**：
  - `backtest/validation.py`：`cost_sensitivity_table`（基准/佣金×2/滑点+5bp 情景对比）+ `time_series_holdout`/`walk_forward_windows`（按时间顺序切分，样本外不含训练期）。
  - 前端 `pages/Backtest/CNNBacktest.tsx`：接入出场模式（阈值/固定持有/OCO 止盈止损/auto 自动对齐）、T+1 开关，并展示 label↔策略一致性告警；类型 `types/cnn.ts`、`types/alpha.ts` 同步扩展。
- **验收标准**：
  - 成本敏感性（佣金×2 → 成本升、净盈亏降）；✅（集成测试）
  - 样本外切分按时间顺序、无泄漏；✅（holdout/walk-forward 测试）
- **风险等级**：🟡 中。
- **状态**：✅ 完成（2026-06-06，前端基础接入；纸面对账与多 regime 报表为后续增强）

**实现说明（v0.7）**：
- **验证工具**：成本敏感性以「给定 CostConfig → statistics」的回调对多情景跑测对比，可直接套在 `run_scheme_backtest` 上；walk-forward 窗口保证 `test_start==train_end`（样本外不含训练期）。测试 `backend/tests/test_validation_iteration5.py`（5 项）。
- **前端**：CNN 回测页可选 4 种出场模式（OCO 时显示止盈/止损/最大持有；auto 时提示按 label 推导）、T+1 开关；结果区在有软性告警时以黄色 Alert 列出 `consistency_warnings`。
- **后续增强（未完，已标注）**：纸面交易对账（理论 vs 实际成交价/延迟，差异告警）、分牛熊震荡 regime 报表、scheme 管理可视化 CRUD 页面。

> 实盘上线（监控 / 提醒 / 自动交易）的迭代规划见 [08-实盘上线与监控交易迭代计划](08-实盘上线与监控交易迭代计划.md)（迭代 6~10）。

---

## 五、进度跟踪（每轮迭代回来更新）

| 迭代 | 名称 | 优先级 | 风险 | 状态 | 负责人 | 备注 |
|:---:|------|:---:|:---:|:---:|:---:|------|
| 0 | 地基对齐 | P0 | 🔴 高 | ✅ 完成 | CH-3 | fixed_hold 出场 + 成本(印花税/滑点) + T+1；5 单测，全量 23 passed |
| 1 | Label 可配置化 | P0 | 🟡 中 | ✅ 完成 | CH-3 | price_ref(close/next_open) 计价口径 + checkpoint 持久化；14 单测，全量 37 passed；含 label↔执行一致性矩阵 |
| 2 | 策略配对 + 一致性自检 | P0 | 🔴 高 | ✅ 完成 | CH-3 | consistency.py(自检+auto推导) + 隐性失败守护；exit_mode=auto；13 单测，全量 50 passed |
| 3 | Scheme 配置层 | P1 | 🟡 中 | ✅ 完成 | CH-3 | registry + Scheme(JSON) + run_scheme_backtest；6 单测 |
| 4 | 止盈止损 OCO | P1 | 🟠 中高 | ✅ 完成 | CH-3 | sell_to_close_intrabar + oco 出场(止损优先)；4 单测 |
| 5 | 验证闭环 + 前端 | P1 | 🟡 中 | ✅ 完成* | CH-3 | 成本敏感性/样本外工具 + 前端出场配置接入；5 单测（*纸面对账/regime 报表为后续增强） |

状态图例：⬜ 未开始 / 🟦 进行中 / ✅ 完成 / ⏸️ 暂停 / ❌ 放弃

---

## 六、关键红线与残余风险

### 6.1 红线（不可破）

- `label.exit ≡ 策略出场 ≡ 撮合成交`，靠迭代 2 的 CI 断言守住。
- 回测「无成交 / 命中 = 0」必须升级为**显式失败**，不能只 warn（否则配置错了看不出来）。
- 策略预估成本与引擎扣费**共用同一套参数**，禁止两处各算。

### 6.2 残余风险（需持续关注）

- 迭代 4 日内 high/low 先后顺序假设（保守取止损先到，仍是近似）。
- 尾盘流动性能否吃下目标仓位（回测难完全模拟，需纸面验证）。
- 多 scheme 后成本参数分散，需统一来源。
- 分钟线方案的「信号早一根」边界处理（避免退化成 T+1）。

---

## 七、架构设计：复用与解耦

### 7.1 信号契约腰线（已存在，要守住）

```
预测器 Predictor   →   统一信号 [datetime, vt_symbol, signal]   →   策略 Strategy   →   执行环境（回测引擎 / 实盘网关）
 (CNN / Alpha / 规则)            ← 这条腰线解耦上下游 →                (BaseStrategy 子类)      (BarDataLoader Protocol)
```

- 引擎已用 `BarDataLoader` 协议解耦数据源、`BaseStrategy` 解耦策略。
- 策略层不区分回测/实盘，差异只在执行层。

### 7.2 三个增量约定（不动引擎）

1. **Predictor 接口化**：`SignalProvider.predict() -> signal_df`，CNN/Alpha/规则各实现一个，互相可替换。
2. **Strategy 注册表**：`name → class` 登记，参数全部走 `setting` 注入。
3. **方案 / Scheme 配置化**：一个 scheme = 一条配置（含 label_spec）。每个方案 = 配置 + 可选独立 strategy 文件，天然分支独立、互不污染。

**结果**：共享地基（引擎 / 撮合 / 盯市 / 统计 / 数据源 / 信号契约 / label 框架），独立上层（predictor 配置 + label_spec + strategy 子类 + 参数）。新增方案不 fork 引擎，也不污染别的方案。

---

## 八、附：尾盘买入持有到次日的 label 选型

入场固定为「T 日尾盘买入」，区别全在出场：

| 出场规则（实盘可执行） | label 定义 | 可实现性 |
|------|------|:---:|
| 持有到次日收盘卖 | 次日收盘 / 买入价 − 1 > 阈值 | ✅ 最推荐 |
| 次日开盘卖 | 次日开盘 / 买入价 − 1 > 阈值 | ✅ |
| 挂固定止盈 +x%，否则次日收盘平 | 次日 high ≥ 买入价 ×(1+x) | ⚠️ 需处理日内 high/low 先后 |
| 卖在最高点 | 次日 high / 买入价 − 1 > 阈值 | ❌ 不可实现 |

**首选**：先用「次日收盘卖」跑通迭代 0~2，拿到一个**可信**的尾盘买入回测，再考虑 OCO。

---

## 更新记录

| 日期 | 版本 | 变更 | 作者 |
|------|------|------|------|
| 2026-06-06 | v0.1 | 初稿：影响地图 + 6 个迭代 + 进度跟踪 + 架构设计 | QingTian CH-3 |
| 2026-06-06 | v0.2 | 迭代 0 完成：fixed_hold 固定持有出场 + T+1 开关（成本部分已由前序改动接通）；新增 5 项单测，后端全量 23 passed | QingTian CH-3 |
| 2026-06-06 | v0.3 | 迭代 1 完成：price_ref 计价口径（close/next_open）核对接通 + checkpoint 持久化；新增 14 项 label 单测，后端全量 37 passed；补 label↔执行一致性矩阵 | QingTian CH-3 |
| 2026-06-06 | v0.4 | 迭代 2 完成：新增 consistency.py（label↔策略出场自检 + auto 推导）+ exit_mode=auto + 信号∩行情无交集隐性失败守护；新增 13 项单测，后端全量 50 passed | QingTian CH-3 |
| 2026-06-06 | v0.5 | 迭代 3 完成：registry(SignalProvider+策略注册表) + Scheme 配置层(JSON 持久化 + run_scheme_backtest)；6 项单测 | QingTian CH-3 |
| 2026-06-06 | v0.6 | 迭代 4 完成：引擎 sell_to_close_intrabar + 策略 oco 止盈止损(止损优先)；4 项单测 | QingTian CH-3 |
| 2026-06-06 | v0.7 | 迭代 5 完成：validation 成本敏感性/样本外工具 + 前端 CNN 回测页接入出场模式/T+1/一致性告警；5 项单测，后端全量 71 passed（纸面对账/regime 报表列为后续增强） | QingTian CH-3 |
