export interface CNNModelInfo {
  name: string
  created_at: string
  modified?: string
  size_mb?: number
  best_epoch?: number
  best_val_loss?: number
  target_symbol?: string
  input_data_kind?: string
  input_interval?: string
  /** 预测目标：classification=方向概率；regression=预测涨跌幅 */
  objective?: 'classification' | 'regression'
  group_count?: number
  observation_groups?: Array<Record<string, unknown>>
}

export interface CNNHistoryItem {
  epoch: number
  train_loss: number
  val_loss: number
  train_acc: number
  val_acc: number
  /** 多数类基线准确率（始终预测占比更高的方向） */
  val_baseline_acc?: number
  /** 超额准确率 = val_acc - 基线；≤0 说明没有方向 edge */
  val_excess_acc?: number
  /** 验证 AUC；单一类别时为 null */
  val_auc?: number | null
  val_precision?: number
  val_recall?: number
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
  lr?: number
}

export interface CNNModelDetail extends CNNModelInfo {
  train_config: Record<string, unknown>
  model_config: Record<string, unknown>
  normalization?: Record<string, unknown>
  dataset_info?: Record<string, unknown>
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

/** 模型真实结构探查结果 */
export interface CNNArchitecture {
  name: string
  /** 权重是否与重建结构严格一致（真实性闸门） */
  verified: boolean
  /** verified=false 时的不一致说明 */
  verify_message?: string
  /** 前向探查失败时的说明（逐层形状不可用） */
  forward_error?: string
  /** PyTorch 原生模块树 str(model) */
  module_repr: string
  objective?: string
  /** 探查使用的输入张量形状 */
  input_shapes?: Record<string, number[]>
  /** 整个模型的输出张量形状 */
  output_shape?: number[] | null
  total_params: number
  total_params_h: string
  trainable_params: number
  trainable_params_h: string
  param_dtype?: string
  layers: CNNArchitectureLayer[]
}

export interface CNNStatus {
  torch_installed: boolean
  device: string
}

export interface CNNBacktestRequest {
  name: string
  model: string
  capital?: number
  start: string
  end: string
  buy_threshold?: number
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
}

export interface CNNPredictRequest {
  name: string
  model: string
  start: string
  end: string
}

export type { TaskStartResponse, CNNTrainRequest, LabelSpec, ObservationGroup } from './alpha'
