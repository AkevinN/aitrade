/** CNN 模型列表条目（`GET /api/cnn/models` 元素）。 */
export interface CNNModelInfo {
  /** 模型唯一名称 */
  name: string
  /** 模型创建时刻（ISO 时间串） */
  created_at: string
  /** 权重文件最后修改时刻（ISO 时间串） */
  modified?: string
  /** 权重文件大小，单位 MB */
  size_mb?: number
  /** 验证损失最低的 epoch（即采用的最优权重所在轮） */
  best_epoch?: number
  /** 最优 epoch 对应的验证损失 */
  best_val_loss?: number
  /** 训练目标合约代码 */
  target_symbol?: string
  /** 输入数据类型：bar=K 线；tick=逐笔 */
  input_data_kind?: string
  /** 输入数据周期，如 "d"、"30m" */
  input_interval?: string
  /**
   * 预测目标：
   * - `classification`：方向二分类，输出上涨概率；
   * - `regression`：直接预测涨跌幅；
   * - `path_class`：路径形态四分类（先触止盈/先触止损/到期小涨/到期小跌，输出四类概率）。
   */
  objective?: 'classification' | 'regression' | 'path_class'
  /** 观测特征分组数量 */
  group_count?: number
  /** 观测特征分组配置明细，结构由后端决定 */
  observation_groups?: Array<Record<string, unknown>>
}

/** CNN 训练历史单个 Epoch 指标（`CNNModelDetail.history` 元素）。 */
export interface CNNHistoryItem {
  /** 轮序号，从 1 开始 */
  epoch: number
  /** 训练集损失 */
  train_loss: number
  /** 验证集损失 */
  val_loss: number
  /** 训练集准确率（仅分类目标有意义） */
  train_acc: number
  /** 验证集准确率（仅分类目标有意义） */
  val_acc: number
  /** 多数类基线准确率（始终预测占比更高的方向） */
  val_baseline_acc?: number
  /** 超额准确率 = val_acc - 基线；≤0 说明没有方向 edge */
  val_excess_acc?: number
  /** 验证 AUC；单一类别时为 null */
  val_auc?: number | null
  /** 验证集查准率（precision） */
  val_precision?: number
  /** 验证集查全率（recall） */
  val_recall?: number
  /** 验证集 F1（precision 与 recall 的调和平均） */
  val_f1?: number
  /** 回归指标：IC（预测收益与真实收益的皮尔逊相关） */
  val_ic?: number | null
  /** 回归指标：RankIC（秩相关） */
  val_rank_ic?: number | null
  /** 回归指标：平均绝对误差 */
  val_mae?: number
  /** 回归指标：均方根误差 */
  val_rmse?: number
  /** 回归指标：方向准确率（sign(pred)==sign(actual)） */
  val_dir_acc?: number
  /**
   * path_class 指标：先触止盈类 AUC（one-vs-rest）；非 path_class 模型为 null。
   *
   * @see CNNModelInfo.objective
   */
  val_tp_auc?: number | null
  /**
   * path_class 指标：先触止损类 AUC（one-vs-rest）；非 path_class 模型为 null。
   *
   * @see CNNModelInfo.objective
   */
  val_sl_auc?: number | null
  /**
   * path_class 指标：四分类宏平均 F1；非 path_class 模型为 null。
   *
   * @see CNNModelInfo.objective
   */
  val_macro_f1?: number | null
  /** 该轮学习率（启用调度器时随轮变化） */
  lr?: number
}

/** CNN 模型完整详情（`GET /api/cnn/models/{name}` 响应体），在列表条目基础上附训练配置与历史。 */
export interface CNNModelDetail extends CNNModelInfo {
  /** 训练时使用的超参数配置，结构由后端决定 */
  train_config: Record<string, unknown>
  /** 模型结构配置（层数/通道等），结构由后端决定 */
  model_config: Record<string, unknown>
  /** 特征归一化参数（均值/方差等）；未保存时缺省 */
  normalization?: Record<string, unknown>
  /** 数据集信息（样本数/区间/类别分布等）；未记录时缺省 */
  dataset_info?: Record<string, unknown>
  /** 逐 epoch 训练历史指标序列 */
  history: CNNHistoryItem[]
}

/** 单个网络层（叶子模块）的真实信息 */
export interface CNNArchitectureLayer {
  /** 模块路径名，如 conv_s.0 */
  name: string
  /** 模块类型，如 Conv2d / BatchNorm2d / Linear */
  type: string
  /** 该层参数量 */
  params: number
  /** 参数量易读格式，如 49.25K */
  params_h: string
  /** 真实前向计算得到的输出张量形状；null 表示未捕获到 */
  output_shape: number[] | null
}

/** 模型真实结构探查结果（加载权重重建模型并做一次前向，反映真实而非声明的结构）。 */
export interface CNNArchitecture {
  /** 模型名称 */
  name: string
  /** 权重是否与重建结构严格一致（真实性闸门） */
  verified: boolean
  /** verified=false 时的不一致说明 */
  verify_message?: string
  /** 前向探查失败时的说明（逐层形状不可用） */
  forward_error?: string
  /** PyTorch 原生模块树 str(model) */
  module_repr: string
  /** 预测目标（classification / regression / path_class） */
  objective?: string
  /** 探查使用的输入张量形状 */
  input_shapes?: Record<string, number[]>
  /** 整个模型的输出张量形状 */
  output_shape?: number[] | null
  /** 模型总参数量 */
  total_params: number
  /** 总参数量的易读格式，如 "1.2M" */
  total_params_h: string
  /** 可训练参数量（排除冻结参数） */
  trainable_params: number
  /** 可训练参数量的易读格式 */
  trainable_params_h: string
  /** 参数数据类型，如 "float32" */
  param_dtype?: string
  /** 逐叶子层的真实信息列表 */
  layers: CNNArchitectureLayer[]
}

/** CNN 模块状态（`GET /api/cnn/status` 响应体）。 */
export interface CNNStatus {
  /** PyTorch 是否已安装。 */
  torch_installed: boolean
  /** 当前使用的推理设备（`cpu` / `cuda`）。 */
  device: string
}

/** CNN 回测运行请求（`POST /api/cnn/backtest/run`）。 */
export interface CNNBacktestRequest {
  /** 回测任务名称 */
  name: string
  /** 使用的模型名称 */
  model: string
  /** 初始资金，单位元；缺省时由后端取默认值 */
  capital?: number
  /** 回测区间起始日期（YYYY-MM-DD，含） */
  start: string
  /** 回测区间结束日期（YYYY-MM-DD，含） */
  end: string
  /** 买入信号概率阈值；缺省时由后端取默认值 */
  buy_threshold?: number
  /** 卖出信号概率阈值；缺省时由后端取默认值 */
  sell_threshold?: number
  /** 单边佣金率（默认万3） */
  commission_rate?: number
  /** 卖出印花税率（默认千1，A股） */
  stamp_duty?: number
  /** 每笔成交不利滑点率（默认5bp） */
  slippage?: number
  /** 限价单价格缓冲/市价化挂单（默认20bp） */
  price_add?: number
  /** 出场模式：threshold=概率阈值；fixed_hold=固定持有；oco=止盈止损；auto=按 label 自动对齐 */
  exit_mode?: 'threshold' | 'fixed_hold' | 'oco' | 'auto'
  /** fixed_hold/oco 的固定/最大持有交易日数 */
  hold_days?: number
  /** oco 止盈幅度（0.02=+2%），0=不启用 */
  take_profit?: number
  /** oco 止损幅度（0.03=-3%），0=不启用 */
  stop_loss?: number
  /** 是否启用 T+1 卖出限制 */
  t_plus1?: boolean
  /**
   * 否决阈值（仅 path_class 模型有效）：当推理输出中先触止损概率（prob_sl）≥ 该值时，
   * 放弃本次买入信号。范围 (0, 1]，默认 1.0 表示关闭（永不否决）。
   *
   * @example
   * veto_threshold: 0.5  // prob_sl ≥ 0.5 时跳过买入
   * veto_threshold: 1.0  // 关闭否决（默认）
   */
  veto_threshold?: number
}

/** CNN 推理（生成信号）请求（`POST /api/cnn/predict`）。 */
export interface CNNPredictRequest {
  /** 推理任务/输出信号名称 */
  name: string
  /** 使用的模型名称 */
  model: string
  /** 推理区间起始日期（YYYY-MM-DD，含） */
  start: string
  /** 推理区间结束日期（YYYY-MM-DD，含） */
  end: string
}

export type { TaskStartResponse, CNNTrainRequest, LabelSpec, ObservationGroup } from './alpha'
