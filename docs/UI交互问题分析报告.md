# AITrade 前端 UI 交互问题分析报告

> 分析日期：2026-07-02
> 分析方法：实际启动前后端（后端无 Tushare token，处于空数据/未配置状态），用 Playwright 以 1440×900、800×900、375×812 三种视口遍历全部 11 个路由截图；对每页执行空表单提交、Tab 切换、按钮点击等交互探测并收集控制台错误与 HTTP 失败；同时对 `frontend/src` 全量页面代码做逐文件交互审查。以下所有问题均给出代码位置，标注「实测」的问题在真实浏览器中复现过。

## 总体结论

应用整体信息架构清晰，暗色主题统一，空态引导（工作台数据就绪面板、各页 Empty 文案）做得较好；危险操作（删除决策/计划、确认调仓、复位熔断）大多有 Popconfirm。但存在 **1 个全局性 P0 缺陷（React 19 下 antd 静态 API 完全失效，导致约 8 个页面的全部 toast 反馈静默丢失、2 处二次确认弹窗疑似弹不出）**、1 个数据展示错误（回测扫描收益放大 100 倍），以及一批共性模式问题：**接口错误被伪装成空态、任务完成后列表不刷新、提交防重窗口不完整、图表亮色样式与暗色主题冲突**。

---

## P0 — 必须尽快修复

### 1.（实测）React 19 + antd v5 静态 `message` / `Modal.confirm` 完全失效，8 个页面所有操作反馈静默丢失

- **实测证据**：在模型训练页空表单点击「创建数据集」（`ModelTrain/index.tsx:102` 应触发 `message.warning('请输入数据集名称和证券代码')`），点击后 0.4s / 2.2s 两个时点 DOM 中均无 `.ant-message-notice` 节点——**提示从未渲染**。控制台同时出现两条警告：
  - `Warning: [antd: message] Static function can not consume context like dynamic theme. Please use 'App' component instead.`
  - `Warning: [antd: compatible] antd v5 support React is 16 ~ 18.`
- **根因**：项目使用 React 19.2 + antd 5.29，未安装 `@ant-design/v5-patch-for-react-19` 兼容补丁。antd v5 的静态方法（`message.xxx`、`Modal.confirm`）依赖 React 18 的渲染入口，在 React 19 下静默失败。
- **影响范围**（静态导入 `message` 的文件，其中的校验警告、成功提示、失败报错全部不显示）：
  - `pages/ModelTrain/index.tsx`
  - `pages/Signal/index.tsx`
  - `pages/Backtest/AlphaBacktest.tsx`、`CNNBacktest.tsx`、`RuleBacktest.tsx`
  - `pages/Resource/index.tsx`
  - `pages/CNNScreening/index.tsx`
  - `pages/CNNGovernance/index.tsx`
- **更严重的连带故障**：静态 `Modal.confirm` 同机制，大概率同样弹不出：
  - `pages/DataPrepare/index.tsx:819`「确认更正周期标签」——handler `await` 一个由 `onOk/onCancel` resolve 的 Promise，弹窗不出现则 **Promise 永不 resolve，整个功能死锁**；
  - `pages/CNNGovernance/index.tsx:315、388`「回滚生产模型」「晋级确认」——治理关键操作的二次确认可能整个不可用。
- **对比**：使用 `App.useApp()` 的页面（DataPrepare、CNNTrain、TradingConsole、Portfolio 等 9 个文件）反馈正常（实测 DataPrepare 表单校验 inline 提示正常显示）。
- **修复建议**：
  1. 安装并在 `main.tsx` 顶部引入 `@ant-design/v5-patch-for-react-19`（一行修复，立即恢复全部静态 API）；
  2. 中期统一改为 `const { message, modal } = App.useApp()`，消除主题不一致与上下文问题（`main.tsx` 已包 `<AntdApp>`，具备条件）。

### 2. 回测参数扫描 / Walk-Forward 表格收益、回撤放大 100 倍

- **位置**：`pages/Backtest/RuleBacktest.tsx:320-341`
- **问题**：`sweepColumns` 对 `total_return`、`max_ddpercent` 做 `(v * 100).toFixed(2)%`，但后端 `backend/aitrade/backtest/engine.py:466` 已经按百分比返回（`(end_balance/capital - 1) * 100`），`sweep.py:45` 原样透传。同页上方单次回测结果（`BacktestResults.tsx:74-79`）直接加 `%`（正确口径）。
- **后果**：实际收益 5% 在扫描表中显示 500.00%，与同屏单次结果自相矛盾，直接误导参数选择。
- **修复**：去掉 `* 100`，与 `BacktestResults` 统一口径。

### 3.（实测）CNN 治理页：启动按钮未捕获异常直接抛到全局，关键操作失败全静默

- **实测证据**：治理页点击「训练候选模型」后浏览器捕获到 `[pageerror]`（未处理的 Promise rejection）——`startEvaluate/startCandidate/startReplay`（`CNNGovernance/index.tsx:240-286`）是裸 async onClick，无 try/catch、无 loading/防重。
- **连带问题**（同页）：
  - 晋级/拒绝/回滚/保存配置四个 `useMutation`（`:194-224`）**只有 onSuccess 没有 onError**——治理级不可逆操作失败时 UI 零反馈，用户会误以为已生效；
  - 大量参数只写在 `initialValues` 里而无对应表单控件（`:454-543`、`:558-618`：`label_threshold_pct`、`horizon`、`reg_buy/sell_threshold`、回放的 epochs/lookback/阈值等），`validateFields()` 对未注册字段返回 undefined，**实际提交值与界面暗示的默认值不一致**（如 threshold 恒为 0 而非显示的 0.5）；
  - 任务完成后候选/回放/历史列表不自动刷新（`:149` 与 `:159-170` 无联动），必须整页刷新；
  - `latestReport`（`:229-232`）优先取当前任务结果，一旦本会话有任务完成，点击任何候选的「报告」按钮内容永远不变；
  - 长任务只有一行 `status · message` 文字（`:227、359`），无进度条无失败堆栈——现成的 `TaskStatusPanel` 未复用。

### 4. 组合页把接口错误渲染成「暂无持仓记录」

- **位置**：`pages/Portfolio/index.tsx:171-189、335-372`
- **问题**：`getPortfolio`/`getPortfolioRisk` 404 或失败时（`isError` 完全未读取），卡片落入 Empty 分支显示「暂无持仓记录」「暂无风险状态」。对一个**持仓账本**页面，把「接口挂了」显示成「空仓」是危险的误导。
- **修复**：区分 `isError`（含 404「组合不存在」专用文案）与真实空态，提供重试。

---

## P1 — 高优先级（正常使用路径必然触达）

### 5. 全局共性：接口错误被伪装成空态（约 10+ 处）

`api/client.ts:32-39` 拦截器把后端 `detail` 打到 console 后原样 reject，且无全局 toast；各页 `useQuery` 几乎都只取 `data` 不读 `isError/isLoading`。后端不可用时全站表现为「什么都没有」：

- Dashboard（`pages/Dashboard/index.tsx:41-66`）：统计卡显示 0，并给出「先下载日线」的**错误引导**；更糟的是 `:126-128` 的 `!status?.installed` 在加载中/失败时为 true，**误报红色「Alpha 模块未安装」横幅**（首帧也会闪现）。
- 交易操作台模型下拉（`ConfigForm.tsx:94-97、216-228`）显示「暂无模型」，误导用户去重新训练。
- 计划列表（`PlanManager.tsx:40-43` + `PlanList.tsx:126-128`）显示「暂无交易计划，点击右上角新建」，可能诱发重复创建。
- 资源页 5 个查询（`Resource/index.tsx:46-69`）、CNN 训练模型/资源列表（`CNNTrain/index.tsx:478-486`）、模型/信号/数据集下拉（ModelTrain/Signal/各 Backtest）同模式。
- 表格普遍未传 `loading`（Portfolio 调仓表 `:441-455`、治理页各表、DataPrepare `ResourceTable`），**首屏加载中直接显示"还没有××"空态**，数据到达后闪现。

**修复模式**：拦截器把 `detail` 写回 `error.message`；关键查询区分 isLoading（Skeleton/Table loading）/ isError（Alert + 重试）/ 真空态三态。

### 6. 全局共性：提交防重窗口不完整（双击可重复创建任务）

所有「启动任务」按钮的 loading 都绑定 `task.data?.status === 'running'`，存在两段裸奔窗口：① POST 在途；② 拿到 task_id 后 `useTask` 首次轮询返回前（最多 2 秒）及 `pending` 状态。快速双击会创建重复任务，且第一个任务的 taskId 被覆盖后进度无处展示。

- 涉及：`ModelTrain/index.tsx:257-265、457-465`、`Signal/index.tsx:214-222`、`AlphaBacktest.tsx:137-145`、`CNNBacktest.tsx:407-415`、`RuleBacktest.tsx:516-524、575-582、613-620`、`DataPrepare/index.tsx:1044-1052`（此处 loading 还绑错对象——聚合任务运行时下载按钮也转圈）、`CNNTrain/index.tsx:1377`（已有 submitting 但漏了 pending）。
- **修复模式**：本地 `submitting` state + `loading={submitting || ['pending','running'].includes(task.data?.status ?? '')}`。

### 7. 信号分析页：生成完成后结果不加载、列表不刷新（实测右侧空坐标轴）

- `pages/Signal/index.tsx:42-55、101-115`：任务完成后既不 invalidate `['alpha-signals']` 也不自动加载新信号——进度 100% 后图表/表格仍空（或残留旧数据），用户会以为失败。ModelTrain `:74-84` 有正确范式可抄。
- （实测截图确认）「信号时间线」「信号分布」在无数据时渲染**两块空白坐标轴**而非 Empty 占位（`:275-302`）；且时间线把所有标的按行序连成一条线，多标的时是无意义锯齿。

### 8. K 线图硬编码浅色主题，出图即「暗屏白块」

- `components/charts/KLineChart.tsx:88-98`：`background:'#ffffff'`、`textColor:'#333'`、网格 `#f0f0f0`——全局是 darkAlgorithm 暗色主题，任何回测出图都会在暗色卡片中出现刺眼白色图表。
- recharts 系列同病：`EquityCurveChart.tsx:57`、`ReturnComparisonChart.tsx:66` 的 `CartesianGrid stroke="#f0f0f0"`，Tooltip 默认白底样式，tick 未设 fill（深灰字在深底上对比度不足）。
- **修复**：统一从 antd token（或常量）取暗色系：bg `#1f1f1f`、grid `#303030`、text `#8c8c8c+`，Tooltip `contentStyle` 定制。

### 9. 规则回测：切换信号源后旧参数残留并混入请求

- `RuleBacktest.tsx:48、241-304、352-362`：`signalParamValues` 切换 `selectedSource` 不清空，A 源的参数会展开进 B 源请求的 `signal_params`；动态输入用 `defaultValue` 非受控，同名参数切换后显示旧值。后端行为不可预期且用户不可见。
- 同页 Sweep/Walk-Forward 任务**无进度条、失败零反馈**（`:575-629`，结果表只在有行时渲染），网格执行失败时按钮转一会儿后界面纹丝不动。

### 10. 工作台引导跳转的 `state.focus` 无人消费，引导链路断裂

- 发送方 `Dashboard/index.tsx:95、103、185、194` 传 `{ state: { focus: 'tick-import' | 'aggregate' } }`；接收方 `DataPrepare/index.tsx` 全文无 `useLocation`，两处 Tabs 均无受控 `activeKey`。点「导入 Tick」「做聚合」跳过去永远停在默认第一个 Tab。

### 11. 交易操作台：生产模型异步返回后覆盖用户已填内容

- `ConfigForm.tsx:103-110`：`governance/production` 查询返回后无条件回填 `model` 和 `vt_symbol`，慢网络下会覆盖用户刚手选的模型/标的。应以 `form.isFieldsTouched()` 保护，只预填未触碰的表单。

### 12. 任务与页面生命周期脱节：离开页面即丢失进度视图

- 各页 taskId 只存组件 state（如 `DataPrepare/index.tsx:468`），切页回来后进度面板消失，只能在工作台看到粗略状态；Backtest 的 Tab 切换（`Backtest/index.tsx:25-81`，Tab 不落 URL，刷新回到第一个 Tab）同样导致已完成任务结果无法找回。
- 相关：`hooks/useWebSocket.ts` + `stores/taskStore.ts` 是**死代码**（全仓库无调用），注释宣称的「WS 推送 + 轮询兜底」不成立，实际全靠 2s/5s 轮询；该 hook 一旦按注释启用还有每帧重连缺陷（`topics` 默认参数每次渲染新数组 → effect 反复 close/connect）。建议要么真正接入（顺便解决本条），要么删除死代码修正注释。

---

## P2 — 中优先级

**表单与校验**

- 编辑计划弹窗点遮罩即关闭，十余项长表单输入静默丢失（`PlanManager.tsx:224-233`，建议 `maskClosable={false}`）。
- rule 计划股票池无必填校验，空 universe 计划创建成功但调度时必然产不出信号（`PlanForm.tsx:427-436`）。
- 买入阈值等 InputNumber 可清空为 null 照常提交，后端 422 且错误 detail 不上屏（`ConfigForm.tsx:359-365`、`PlanForm.tsx:617-629`）。
- 日频/日内模型切换时 `as_of` 时间部分残留，决策 bar 会落到前一交易日（`ConfigForm.tsx:283-293`）。
- MLP `hidden_sizes` 非法输入被静默 filter 成空数组提交（`ModelTrain/index.tsx:141-149`）。
- `v || default` 回退模式导致输入框无法清空、合法 0 值被回退（ModelTrain/AlphaBacktest/RuleBacktest 多处，应改 `v ?? default`）。
- 训练集截止日期可选在数据日期范围之外、特征集可全部取消后照常提交（`ModelTrain/index.tsx:29-35、210-246`）。
- 画像结果不随标的/周期切换失效，「填充到训练表单」可能把旧标的的建议填进新标的（`CNNTrain/index.tsx:463-464、1062-1068`、`ProfilingPanel.tsx:356`）。

**反馈与状态同步**

- 后端错误 `detail` 全站丢失：拦截器只 console（`api/client.ts:32-39`），多数页面 `message.error(error.message)` 显示的是 axios 英文泛化文案；仅 `RebalancePlanCard.tsx:154-164`、`ProfilingPanel.tsx:75-78` 各自造了轮子，行为不一致。建议拦截器统一 `error.message = detail`。
- 批次合并确认框 `onOk` 内 merge 失败被 antd 吞掉，无任何提示（`DataPrepare/index.tsx:900-912`）；合并预览 Alert 不随勾选变化清空，展示陈旧结论（`:1216-1225、1275-1284`）。
- 计划启停 Switch 状态回跳闪烁：`finally` 先清 togglingId、invalidate 重拉是异步（`PlanManager.tsx:102-112`）；「立即触发」成功后 last_triggered 不刷新（`:123-143`）。
- 「立即触发」会推送钉钉/企微提醒并占用当日幂等位，却无二次确认（`PlanList.tsx:105-111`）。
- 任务查询失败时 ProgressCard 永久停在「正在获取任务状态...」（`ProgressCard.tsx:47-56`）。
- getPlan 失败时 rule 计划结果被误用 CNN 卡片渲染，丢失「确认执行」入口（`PlanManager.tsx:129-134、211-218`）。
- Parquet 暂存失败后无重试/清空入口，面板死局（`ParquetUploadPanel.tsx:117-121、258-323`）。
- 聚合工作区：任何资源刷新（任务完成/删除/手动刷新）都会**静默重置用户手选的时间范围**为全区间，若未察觉会聚合远超预期的数据量（`AggregationWorkspace.tsx:251-263`）。
- 训练完成无条件覆盖右侧模型详情，正在对比的另一个模型被无提示替换（`CNNTrain/index.tsx:620-629`）。
- BacktestCharts 拉全量历史 K 线而非回测区间，`fitContent` 把回测段压成一小截，且缓存键与实际请求不符（`BacktestCharts.tsx:56-60`）。
- CNN 回测 interval 缺失时 K 线永远显示「暂无行情数据」，实为周期未知（`CNNBacktest.tsx:84-87` + `BacktestCharts.tsx:59、92-115`）。
- 禁用按钮包 Tooltip 不显示，用户不知为何不能「带入训练」（`CNNScreening/index.tsx:323-331`，应包一层 span；CNNTrain `:1046-1057` 已有正确写法）。
- 长任务（训练/选股/评估/回放）一律**无取消入口**，误配大参数只能干等或刷新（刷新后连状态都丢了）。

**视觉走查补充（实测截图）**

- **双标题冗余**：Header 已展示「页面名 + 描述」，Content 内又渲染一遍几乎相同的大标题和描述（信号分析、交易操作台、数据准备、资源管理最明显），浪费一屏约 90px 垂直空间，移动端更突出。建议 Content 内标题降级或移除。
- CNN 选股在本地无数据时跑完显示绿色「选股任务已完成」但榜单区只有「暂无数据」，无「扫描了 0 个标的」类解释，完成态与空结果并列显得矛盾。
- 响应式整体可用：375px 侧边栏正确折叠为图标、内容单列；800px 布局正常。唯 768px 断点只有「展开/折叠」两态，isNarrow 由 window resize 驱动而非 antd 的 breakpoint，规则简单但可用。
- 控制台残留多个 antd 弃用警告（`InputNumber addonAfter`、`Modal destroyOnClose`、`rc-collapse children`）以及一条 `useForm not connected to any Form element`（DataPrepare 有 form 实例未挂载，实测出现在页面加载即触发）。

---

## P3 — 低优先级 / 打磨项

- 头部「运行中任务 N」徽标纯展示不可点击，无处查看是哪些任务（`App.tsx:206-221`）。
- 任务状态 Tag 直接显示英文枚举 running/failed（`TaskStatusPanel.tsx:62`、Dashboard `:243-245`）；完成文案写死「训练结果已收起」不匹配下载/聚合任务（`TaskStatusPanel.tsx:128`）。
- `rgba(0,0,0,0.04)` 亮色系代码块背景在暗色下不可辨（`TaskStatusPanel.tsx:106、144`、`CNNTrain/index.tsx:420`）；空态图标 `#444` 在暗背景上几乎不可见（`CNNScreening/index.tsx:775`）。
- 硬编码色值与主题 token 脱钩：`#1677ff` vs 主题 `#1668dc`、`#52c41a` vs `#49aa19` 等（Dashboard `:85`、AggregationWorkspace `:379`、ConfigForm `:238`、RebalancePlanCard `:217`、Resource `:296` 等），同屏两种蓝/绿。
- 红绿语义冲突：K 线遵循 A 股红涨绿跌、买红卖绿（`KLineChart.tsx:48-53`），同屏统计卡用绿=盈利（`BacktestResults.tsx:79`）；信号分布 `[0,0.2)` 桶染灰与统计卡「>0 即看多」口径不一（`Signal/index.tsx:292-299`）；负超额收益仍用绿色填充（`ReturnComparisonChart.tsx:83-94`）。建议统一约定并在图例标注。
- K 线图无 crosshair OHLC 读数图例，悬停读不到具体价格（`KLineChart.tsx`）。
- 可访问性：行点击详情无键盘可达（`HistoryTable.tsx:230-237`、`Portfolio/index.tsx:450-454`）；图标按钮无 aria-label（DataPrepare `:438`、CNNTrain `:927、929、1390`）；CheckableTag 作单选控件不可 Tab 聚焦（`ModelTrain/index.tsx:221-244`）。
- Portfolio 自定义组合 ID 靠隐蔽的 onKeyDown-Enter hack 录入，`onSearch` 是死代码（`Portfolio/index.tsx:309-319`），建议改 AutoComplete。
- 日期选择无未来日期约束、clamp 静默改写用户输入无提示（`DateRangeSelector.tsx:161-171`）。
- 调度器「上次触发」只显示裸 plan_id（`SchedulerStatusCard.tsx:51-56`）；非法时间戳会渲染 `NaN:NaN:NaN`（`SchedulerRunsCard.tsx:29-39`）；rowKey 同秒撞 key（`SchedulerRunsCard.tsx:174`、`CNNGovernance/index.tsx:422`）。
- 胜率等数值未格式化，会显示 `0.6666666666666666`（`CNNGovernance/index.tsx:650、683`）。
- 删除类 Popconfirm 确认按钮未标 danger、无 pending 防重（`Resource/index.tsx:202、265、290`）。
- 任务默认名挂载时生成一次，同一分钟内二次提交重名（`CNNScreening/index.tsx:451`、CNNGovernance `:460、564`）。
- CSV 导入按钮在预览已提示缺字段时仍可点击（`DataPrepare/index.tsx:1096-1104`）。
- failed 状态候选仍可点「晋级」，与门禁语义相悖（`CNNGovernance/index.tsx:657`）。

---

## 做得好的地方（保持）

- 空态引导链路完整：工作台「数据就绪面板 → 推荐下一步」、各页 Empty 都指向下一步动作。
- 危险操作确认：删除决策/计划、确认调仓、复位熔断均有 Popconfirm 且文案明确；「仅提醒，不自动下单」提示贯穿交易台各结果区。
- DataPrepare/AggregationWorkspace 的四态（加载/错误/空/数据）互斥处理、Parquet 面板的分步反馈是全站最佳实践。
- `chartAdapters.ts` 对脏数据的防御、`KLineChart` 的 ResizeObserver 清理、`CNNBacktest` 用 `result.model` 反查周期避免出图后改选模型的错配。
- ModelTrain 任务完成后 refetch 列表的模式正确（可推广到 Signal/治理页）。

## 建议修复顺序

1. **一行修复**：引入 `@ant-design/v5-patch-for-react-19`（恢复全部静态 message/Modal）→ 随后统一迁移到 `App.useApp()`。
2. `RuleBacktest` 扫描表 ×100 口径错误（一处两列）。
3. `api/client.ts` 拦截器把后端 `detail` 写回 `error.message`（全站错误文案立刻改善）。
4. CNN 治理页：mutation 补 onError、启动按钮补 try/catch + loading、任务完成 refreshGovernance、补齐/删除幽灵表单字段。
5. 全站「错误≠空态」三态改造（优先 Portfolio、Dashboard、ConfigForm、PlanList）。
6. 图表暗色主题适配（KLineChart + recharts 三件）。
7. 提交防重（统一 submitting 模式）、Signal 完成后自动加载、Dashboard focus 断链。
8. 其余 P2/P3 按迭代节奏消化。
