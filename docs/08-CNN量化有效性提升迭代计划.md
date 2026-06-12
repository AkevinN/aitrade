# CNN 量化有效性提升迭代计划

> 本文档是一份**活文档（living doc）**，用于规划「让 CNN 模块从『工程跑得通』走向『量化有真实 alpha 且可用』」的迭代落地，每轮迭代后回来更新状态。
>
> 背景缘起：[07-CNN-Label可配置化与多方案迭代计划](07-CNN-Label可配置化与多方案迭代计划.md) 已把 **label↔执行一致性、成本、Scheme 复用** 的地基打牢（迭代 0~5 完成）。但一次系统评估显示：**工程架构 8/10，量化有效性 5/10**——骨架扎实，瓶颈在「模型 alpha 本身」。本计划专门解决量化有效性短板。
>
> 关联文档：[07-CNN-Label可配置化与多方案迭代计划](07-CNN-Label可配置化与多方案迭代计划.md)、[cnn模块](cnn模块.md)、[02-后端模块设计](02-后端模块设计.md)、[数据回测模块](数据回测模块.md)。
> 关联审查技能：`.cursor/skills/quant-backtest-live-alignment/`（回测-实盘一致性）。

---

## 一、背景与目标

### 1.1 与 07 的分工

| 文档 | 解决的问题 | 一句话 |
|------|------------|--------|
| 07 | label↔策略出场↔撮合成交 一致性、成本、Scheme 复用解耦 | 「让回测可信、方案可复用」 |
| **08（本文）** | 特征预测力、样本外可信验证、用预测幅度定仓、横截面分散、流动性 | 「让模型真有 alpha，且把 alpha 用对」 |

07 保证「回测不骗人」，08 保证「模型值得跑」。两者正交，08 站在 07 的地基上。

### 1.2 前置已完成的地基（08 直接复用）

- **回归头（`objective=regression`）**：模型可直接预测涨跌幅（线性输出 + Huber），评估含 IC/RankIC/MAE/方向准确率；分类路径保留。涉及 `cnn/network.py`、`cnn/dataset.py`、`cnn/trainer.py`、`cnn/predictor.py`。→ 这是迭代 A「按预测收益定仓」的前提。
- **去噪阈值 + 计价口径**（07 迭代 1）：`LabelSpec.threshold` / `price_ref=next_open`。
- **幅度加权**（分类）：`loss_weighting=magnitude`。
- **成本三件套**：佣金 + 卖出印花税 + 滑点，集中在 `engine._settle_fill`。
- **一致性自检 + Scheme + OCO + 样本外/成本敏感性工具**（07 迭代 2~5）。

### 1.3 本计划要解决的量化短板（来自系统评估）

| 级别 | 短板 | 现状位置 |
|------|------|----------|
| P0 | 特征过于单薄（仅 6 个技术指标），alpha 天花板低 | `cnn/features.py:FEATURE_NAMES` |
| P0 | 过拟合防护不足（单段 holdout、单种子、无 walk-forward 稳定性） | `cnn/trainer.py` |
| P1 | 仓位粗糙（固定全仓，回归预测了幅度却只当开关用） | `cnn/strategy.py:_target_volume` |
| P1 | 无横截面/分散（单标的择时，无组合选股） | 全模块 |
| P1 | 流动性/冲击未建模（除一字板外全量成交） | `backtest/engine.py:_settle_fill` |
| P2 | 复权一致性未显式校验、OCO 跳空穿越偏乐观 | `features.py` / `backtest/oco.py` |
| 工程债 | signal 语义随 objective 漂移、`torch.load(weights_only=False)`、统计零回撤除零、魔法数字 | `predictor/strategy/storage/engine` |

---

## 二、核心原则（务必遵守）

1. **先验证可信度，再堆能力**：没有 walk-forward 样本外，扩特征也无法判断好坏，只会过拟合得更隐蔽。故 **迭代 B（验证）优先级高于迭代 C（特征）**。
2. **宁可悲观，不要乐观**（承接 skill）：任何让回测变好看、实盘做不到的假设，要么写进代码约束，要么在成本里量化。
3. **把幅度信息用对**：回归预测的收益必须真正驱动仓位（迭代 A），否则等于白预测。
4. **金标准是样本外**：判定「有没有 edge」只认 walk-forward / 留出集的 OOS 指标，不认全样本/训练集。
5. **向后兼容**：所有新能力做成可选开关，默认不改变既有行为；旧模型/旧 Scheme 零影响。

---

## 三、影响地图：每个迭代改哪些文件

| 迭代 | 后端核心 | Schema/API | 前端 | 测试 |
|------|----------|------------|------|------|
| A 按预测收益定仓 | `cnn/strategy.py:_target_volume` | `models/alpha.py(CNNBacktestRequest.sizing_mode)` / `api/cnn.py` | `CNNBacktest.tsx` | 分档单调性、sizing 对比 |
| B 样本外验证 | `cnn/trainer.py`（抽 `_train_one_fold` + `walk_forward_train`） | `CNNTrainRequest.validation_scheme/n_folds/n_seeds` | `CNNTrain.tsx`（OOS 报表） | OOS 一致性、泄漏检查 |
| C 特征扩充 | `cnn/features.py`（特征集注册表 + 新因子族） | `CNNTrainRequest.feature_set` | `CNNTrain.tsx`（特征集选择 + 动态通道数） | 因果性、置换重要性 |
| D 横截面组合 | 新增 `cnn/portfolio_strategy.py` | `api/cnn.py`（多标的回测） | 组合回测页 | 分层回测、top/bottom 分化 |
| E 流动性+收口 | `engine._settle_fill`（缩量+冲击）、`backtest/oco.py`（跳空取 open） | `volume_limit_ratio/impact_coef` | 成本面板 | 规模敏感性 |
| 债 工程债 | `predictor/strategy/storage/engine` | — | — | 安全/性能/边界 |

---

## 四、分阶段迭代计划

> 排序原则：**先把已做的回归头用起来（A，闭环最短）→ 立可信验证（B，金标准）→ 再治本提 alpha（C）→ 扩到组合（D）→ 收口悲观假设（E）**。

### 迭代 A · 按预测收益定仓（P1，闭环最短）

- **目标**：让回归模型预测的涨跌幅真正驱动仓位，而非只当阈值开关。
- **改动**：
  - `cnn/strategy.py`：`_target_volume` 增加 `sizing_mode`：
    - `fixed`（现状，默认）：`position_ratio` 全仓。
    - `linear`（推荐）：`weight = clip((signal − buy_threshold) / sizing_scale, 0, 1) × position_ratio`。
    - `tiered`：按预测收益分档（如 >2%→满仓，1~2%→70%，0.5~1%→40%）。
    - `vol_target`：`weight = target_vol / 标的近期已实现波动率`，控单票风险贡献。
  - `models/alpha.py`：`CNNBacktestRequest` 加 `sizing_mode` + `sizing_scale` / 档位参数。
  - `api/cnn.py`：注入策略；统计回传 sizing 配置。
  - 前端 `CNNBacktest.tsx`：回归模型时显示 sizing 选项。
- **验收**：
  - 同模型 `fixed` vs `linear` vs `tiered` 的夏普/回撤/换手对比可复现；
  - **分档单调性检验**：高预测收益档的实际平均收益确实更高（否则模型幅度无效）。
- **风险**：🟡 中（引入超参，易过拟合到回测；默认 `linear` 较稳健）。
- **依赖**：回归头（已完成）。
- **状态**：⬜ 未开始

### 迭代 B · 样本外验证（Walk-forward + 多种子 + 留出集）（P0，金标准）

- **目标**：用滚动样本外判定「模型到底有没有 edge」，并量化稳定性。
- **改动**：
  - `cnn/trainer.py`：把单折逻辑抽成 `_train_one_fold(...)`；新增 `walk_forward_train`：时间轴切 K 折滚动（train 窗 → 紧邻 OOS 窗 → 前移），每折**独立拟合归一化 + 模型**，汇总 OOS 的 IC/方向准确率/超额（均值 ± std）。
  - 多种子：同配置跑 N 个 seed，报告 OOS 指标 std（std 大 = 靠运气）。
  - 留出集：最近 1~2 年完全不参与调参（最终 OOS）。
  - 复用 07 迭代 5 的 `backtest/validation.py:walk_forward_windows`（已保证 `test_start==train_end`，样本外不含训练期）。
  - `models/alpha.py`：`CNNTrainRequest` 加 `validation_scheme(holdout|walk_forward)` / `n_folds` / `n_seeds`。
- **验收**：
  - 报告各折 OOS IC 均值/std；
  - 能识别「holdout 好但 walk-forward 崩」的典型过拟合；
  - OOS IC 均值 > 0 且 std 可控 才算「有 edge」。
- **风险**：🟡 中（训练时间 ×K×N，需进度反馈）。
- **依赖**：无（独立，建议紧随 A）。
- **状态**：⬜ 未开始

### 迭代 C · 特征扩充（特征集注册表 + 因果因子族）（P0，治本）

- **目标**：突破「6 个技术指标」的 alpha 天花板。
- **改动**：
  - `cnn/features.py`：仿 `backtest/registry` 做**特征集注册表** `FEATURE_SETS`；`_compute_features(bars, feature_set)` 动态产出 `[N, len(set)]`；通道数由特征集长度驱动（`dataset.build_dataset` 已读 `len(FEATURE_NAMES)`）。
  - 新增**因果特征族**（全部滚动窗口、只用 ≤t 数据）：
    - 动量：多周期收益(5/10/20)、ROC、动量 rank；
    - 波动率：滚动已实现波动率、ATR、Parkinson/Garman-Klass（用 high/low）；
    - 量价：多周期量比、成交额变化、量价背离、OBV；
    - 趋势：MACD、RSI、布林带位置、MA 斜率；
    - 微观结构：上/下影线占比、跳空、收盘位置 `(close-low)/(high-low)`。
  - 通道数写进 `model_config.in_channels`（已存）；前端 `tensorEstimate.channels` 从写死 6 改为读特征集长度。
  - `models/alpha.py`：`CNNTrainRequest.feature_set`。
- **验收**：
  - 因果性单测（`_compute_features` 第 i 行只依赖 ≤i 的数据）；
  - 置换重要性 / 特征重要性报表；
  - **walk-forward（迭代 B）OOS IC 较 basic6 提升** 才算有效。
- **风险**：🟠 中高（特征越多越易过拟合，必须靠迭代 B 验证 + 正则；建议先加 2 个稳健族）。
- **依赖**：迭代 B（否则无法判断特征是否真有用）。
- **状态**：⬜ 未开始

### 迭代 D · 横截面多标的组合（P1，从择时到选股）

- **目标**：从「单票择时」升级到「组合分散 + 横截面排序」。
- **改动**：
  - **短期（低风险，先做）**：训练仍单标的，回测层支持「一篮子单标的模型组合」——多标的各推理 → 每日按预测收益**横截面排序选 top_k** → 等权/按预测收益加权。复用现有引擎（本就支持多 `vt_symbols`）。新增 `cnn/portfolio_strategy.py:CNNPortfolioStrategy`（吃多标的 `signal_df` → rank → top_k → 目标权重 → `execute_trading`），注册为 `cnn_portfolio`。
  - **长期（范式升级，大改）**：训练改横截面——一次喂多标的、标签用**截面超额收益**、网络共享权重逐标的打分、归一化改**截面标准化**。涉及 `dataset/network/trainer` 大改。
- **验收**：
  - 组合 vs 单票的夏普/回撤；
  - **分层回测**：top_k 与 bottom_k 收益分化、横截面 IC。
- **风险**：🟡 中（短期复用引擎风险低；长期才是大改）。
- **依赖**：迭代 A（定仓）、B（验证）。
- **状态**：⬜ 未开始

### 迭代 E · 流动性/冲击建模 + 悲观假设收口（P1/P2）

- **目标**：让回测更接近实盘可成交性，消除残余乐观假设。
- **改动**：
  - `backtest/engine.py:_settle_fill`：单笔成交 ≤ `bar.volume × volume_limit_ratio`（超出缩量，跨 bar 续单可后置）；滑点基础上加冲击 `impact = k × sqrt(order_value / ADV)`（ADV 用 `turnover` 滚动算）。
  - `backtest/oco.py:check_oco_trigger`：当 `open` 已穿越止损价时，成交价取 `open`（更差）而非触发价。
  - `features._load_market_frame`：显式断言训练/推理/回测同一复权口径。
  - 成本敏感性：复用 07 迭代 5 的 `validation.cost_sensitivity_table`，在回测结果页一键展示「基准 / 佣金×2 / 滑点+5bp」对照。
- **验收**：
  - 大资金 vs 小资金回测（冲击随规模上升）；高换手策略收益明显下降；
  - 成本敏感性表收益下降可接受。
- **风险**：🟠 中高（部分成交 + 续单会让订单状态机复杂，先做缩量 + 冲击）。
- **依赖**：无。
- **状态**：⬜ 未开始

### 工程债 · 代码质量收口（穿插各迭代）

- `predictor.py`/`strategy.py`：signal 携带 `objective` 元信息（避免分类概率/回归收益同叫 `signal` 被误用）。
- `storage.py`：`torch.load(weights_only=False)` 安全评估 + 列表元信息存 sidecar JSON（避免逐文件全量 load 偏慢）。
- `engine.calculate_statistics`：修零回撤 `-total_net_pnl/max_drawdown` 除零。
- 集中魔法数字：Huber `delta=0.03`、`weight_cap=10`、warm-up `lookback*2.5` → 常量/可配置。
- 训练主循环补单测（目前仅 dataset/指标级 + 冲烟）。
- **状态**：⬜ 未开始（建议随对应迭代顺手清）

---

## 五、进度跟踪（每轮迭代回来更新）

| 迭代 | 名称 | 优先级 | 风险 | 状态 | 负责人 | 备注 |
|:---:|------|:---:|:---:|:---:|:---:|------|
| 前置 | 回归头 objective=regression | — | 🟡 | ✅ 完成 | CH-2 | 网络/数据/训练/推理 + 前端；test_cnn_objective.py |
| A | 按预测收益定仓 | P1 | 🟡 中 | ⬜ 未开始 | — | sizing_mode: fixed/linear/tiered/vol_target |
| B | 样本外验证(walk-forward) | P0 | 🟡 中 | ✅ 基础完成 | — | 已由 CNN 治理模块落地 WF/OOS 报告、候选训练与治理回放；多成本场景/多种子深度增强后续补 |
| C | 特征扩充(注册表+因子族) | P0 | 🟠 中高 | ⬜ 未开始 | — | 必须靠 B 验证；先加 2 个稳健族 |
| D | 横截面多标的组合 | P1 | 🟡 中 | ⬜ 未开始 | — | 短期复用引擎；长期改横截面训练 |
| E | 流动性/冲击 + 悲观收口 | P1/P2 | 🟠 中高 | ⬜ 未开始 | — | 缩量+冲击+跳空+复权断言+成本压测 |
| 债 | 工程债收口 | P1/P2 | 🟢 低 | ⬜ 未开始 | — | signal 元信息/storage/除零/魔法数字 |
| 独立特性 | 路径形态多分类头（path_class） | — | 🟢 低 | ✅ 完成 | Claude | 详见 `.kiro/specs/cnn-path-multiclass-head/` 与分支 feat/cnn-path-multiclass-head；Task 1~8 全部闭环 |

状态图例：⬜ 未开始 / 🟦 进行中 / ✅ 完成 / ⏸️ 暂停 / ❌ 放弃

**推荐落地顺序（按 ROI）**：A → B → C → D → E（工程债穿插）。最短见效路径 **A→B**：先让回归幅度驱动仓位，再用 walk-forward 验证它到底有没有 edge。

---

## 六、关键红线与残余风险

### 6.1 红线（不可破，承接 07）

- `label.exit ≡ 策略出场 ≡ 撮合成交`（07 迭代 2 的 CI 断言守住）。
- 判定「有没有 edge」只认**样本外**（迭代 B），不认全样本/训练集。
- 扩特征（C）前必须先有可信验证（B），否则过拟合不可见。
- 回归的预测幅度必须真正用于定仓（A），否则白预测。

### 6.2 残余风险（需持续关注）

- 单标的日线样本量有限，CNN 易过拟合——迭代 C 须配合正则 + 特征筛选。
- 横截面长期方案（D 长期）改动大，需独立分支验证后再合。
- 流动性/冲击系数（E）是经验参数，需用真实成交数据校准。
- 多 regime（牛/熊/震荡）下表现差异——建议 D 之后补分 regime 报表（07 迭代 5 已列为后续增强）。

---

## 七、验证总纲（每个迭代都要过）

1. **单元测试**：纯函数（特征因果、sizing 映射、walk-forward 窗口无泄漏）。
2. **样本外**：walk-forward OOS IC/方向准确率 + 多种子 std。
3. **成本敏感性**：佣金×2、滑点+5bp 压测。
4. **分层/单调性**：预测分位与实际收益单调对应。
5. **回归保护**：全量后端测试不掉绿；前端 `tsc --noEmit` 通过。

---

## 更新记录

| 日期 | 版本 | 变更 | 作者 |
|------|------|------|------|
| 2026-06-06 | v0.1 | 初稿：承接 07，规划量化有效性提升的 5 个迭代（A 定仓 / B 样本外 / C 特征 / D 横截面 / E 流动性）+ 工程债；含影响地图、进度跟踪、红线与验证总纲。前置「回归头」已完成。 | QingTian CH-2 |
| 2026-06-08 | v0.2 | CNN 模型治理基础闭环落地：WF/OOS 评估、候选训练、人工晋级/拒绝/回滚、治理回放回测与前端治理页；详见 `10-CNN模型滚动评估与半自动晋级.md`。 | Codex |
| 2026-06-12 | v0.3 | 新增独立特性：objective=path_class（OCO 路径四分类 + 回测否决条件 veto_threshold），全链路落地；进度跟踪表新增「路径形态多分类头」行。 | Claude |
