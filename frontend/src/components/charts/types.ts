// 图表组件层的纯数据类型定义
// 这些类型不含任何业务语义，仅描述图表渲染所需的数据结构

/** 图表时间值：秒级 UTC 时间戳（分钟线）或 'YYYY-MM-DD' 字符串（日线），与 lightweight-charts 时间格式兼容 */
export type ChartTime = number | string

/** 单根 K 线数据（开高低收 + 可选成交量） */
export interface OHLCBar {
  /** 秒级 UTC 时间戳 或 'YYYY-MM-DD' */
  time: ChartTime
  open: number
  high: number
  low: number
  close: number
  /** 成交量（可选，用于量副图） */
  volume?: number
}

/** 买卖方向：买入 / 卖出 */
export type TradeSide = 'buy' | 'sell'

/** 画在 K 线上的买卖点标注 */
export interface TradeMarker {
  /** 标注所在时间，需与所属 K 线序列的 time 取值口径一致 */
  time: ChartTime
  /** 买卖方向，决定标注的箭头朝向与默认配色 */
  side: TradeSide
  /** 成交价（可选） */
  price?: number
  /** tooltip / 标签文案，如 "买 1000@12.34" */
  text?: string
}

/** 净值曲线单点 */
export interface EquityPoint {
  /** 'YYYY-MM-DD' */
  date: string
  /** 账户净值 */
  balance: number
  /** 回撤金额 */
  drawdown: number
  /** 回撤百分比 */
  ddpercent: number
  /** 当日净盈亏 */
  netPnl: number
  /** 策略累计收益（%，可缺省） */
  strategyReturn?: number | null
  /** 基准（买入持有标的）累计收益（%，可缺省） */
  benchmarkReturn?: number | null
  /** 超额收益（%）= 策略累计收益 - 基准累计收益（可缺省） */
  excessReturn?: number | null
}

/** 叠加到主图的图层（v1 仅支持价位线，后续可扩展均线序列等） */
export interface ChartOverlay {
  /** v1 仅 'price-line'（阈值/决策价位线）；后续可扩展 'line'（均线序列） */
  type: 'price-line'
  /** 价位线的 y 值，单位与主图价格轴一致 */
  price: number
  /** 线条颜色（CSS 颜色串，可选）；缺省时由图表主题决定 */
  color?: string
  /** 价位线右侧标题文案（可选） */
  title?: string
}

/** K 线配色方案（默认遵循 A 股惯例：涨红跌绿，可覆盖） */
export interface KLineColorScheme {
  /** 上涨色，默认 A 股：涨红 #dc4446 */
  up: string
  /** 下跌色，默认 A 股：跌绿 #49aa19 */
  down: string
  /** 买点标注色 */
  buyMarker: string
  /** 卖点标注色 */
  sellMarker: string
}
