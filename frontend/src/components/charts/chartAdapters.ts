// 图表数据适配层：把后端原始数据（行情行、成交、逐日净值）转换为图表组件所需结构。
// 全部为不依赖 React 的纯函数，可被单元测试独立验证（Req 5.5）。

import type { ChartTime, EquityPoint, OHLCBar, TradeMarker, TradeSide } from './types'

// ============================================================
// 内部工具函数
// ============================================================

/** 把任意值安全转为有限数字；无法解析（null/空串/NaN/Infinity）时返回 null。 */
function num(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const n = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(n) ? n : null
}

/** 取出 row 中第一个存在的列名对应的值（支持列名别名映射）。 */
function pick(row: Record<string, unknown>, keys: string[]): unknown {
  for (const k of keys) {
    const v = row[k]
    if (v !== null && v !== undefined) return v
  }
  return undefined
}

/** 判断时间是否落在可选区间内；区间缺省或类型不一致时不过滤（返回 true）。 */
function inRange(time: ChartTime, range?: { min: ChartTime; max: ChartTime }): boolean {
  if (!range) return true
  const { min, max } = range
  // ChartTime 可能是 number（时间戳）或 string（'YYYY-MM-DD'）。
  // 仅当三者类型一致时才比较，避免 number 与 string 混比产生 NaN 误判。
  if (typeof time !== typeof min || typeof time !== typeof max) return true
  return time >= min && time <= max
}

// ============================================================
// 时间解析
// ============================================================

/**
 * 把后端时间统一成 lightweight-charts 接受的时间值：
 * - `'YYYY-MM-DD'`（纯日期，日线）→ 原样返回字符串
 * - ISO 日期时间（分钟线，如 `'2025-03-04T15:00:00'`）→ 秒级 UTC 时间戳（number）
 * - 已是数字 → 视为秒级时间戳，原样返回
 * - 无法解析 → 返回 null（由调用方跳过该行，Req 5.4）
 *
 * 说明：无时区标识的朴素时间按 UTC 处理，保证行情与成交两条数据走同一转换、跨时区对齐一致（Req 5.6）。
 *
 * @param value - 后端时间值，可为数字（秒级时间戳）、`'YYYY-MM-DD'` 或 ISO 日期时间字符串；
 *   `null`/`undefined`/空串/非有限数/非时间字符串均视为不可解析
 * @returns 归一化后的 {@link ChartTime}（日线为字符串、分钟线为秒级时间戳）；无法解析时返回 `null`
 */
export function parseChartTime(value: unknown): ChartTime | null {
  if (value === null || value === undefined) return null

  // 已是数字：视为秒级时间戳
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }
  if (typeof value !== 'string') return null

  const s = value.trim()
  if (s === '') return null

  // 纯日期 'YYYY-MM-DD'：日线，原样返回字符串
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s

  // ISO 日期时间：转为秒级 UTC 时间戳。
  // 无时区后缀（Z 或 ±HH:MM）的朴素时间补 'Z' 强制按 UTC 解释，保证确定性。
  const hasTz = /([zZ]|[+-]\d{2}:?\d{2})$/.test(s)
  const ms = Date.parse(hasTz ? s : `${s}Z`)
  if (Number.isNaN(ms)) return null
  return Math.floor(ms / 1000)
}

// ============================================================
// 行情行 → OHLCBar[]
// ============================================================

/**
 * 把 Bar_Data_API 返回的行情行转换为 OHLCBar[]（Req 5.1）。
 * - 列名映射：datetime/open/high/low/close/volume（time 作为 datetime 的别名兜底）
 * - 缺必要列（时间或开高低收任一）的行被跳过，不抛异常（Req 5.4）
 * - 输出按 time 升序排列，满足 lightweight-charts 输入约束（Req 7.3、5.6）
 *
 * @param rows - 行情行数组，每行为按列名取值的字典；非数组时按空输入处理
 * @returns 按时间升序的 K 线数组；无可用行时返回空数组（volume 缺失则该根不带成交量字段）
 */
export function toOHLCBars(rows: Record<string, unknown>[]): OHLCBar[] {
  if (!Array.isArray(rows)) return []

  const bars: OHLCBar[] = []
  for (const r of rows) {
    if (r === null || typeof r !== 'object') continue
    const time = parseChartTime(pick(r, ['datetime', 'time']))
    const open = num(r.open)
    const high = num(r.high)
    const low = num(r.low)
    const close = num(r.close)
    // 缺时间或任一价格列 → 跳过该行
    if (time === null || open === null || high === null || low === null || close === null) {
      continue
    }
    const volume = num(r.volume)
    bars.push(
      volume === null
        ? { time, open, high, low, close }
        : { time, open, high, low, close, volume },
    )
  }

  // 升序排序：number 与 string 时间均可用 < / > 直接比较（同类型）
  bars.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0))
  return bars
}

// ============================================================
// 成交 → TradeMarker[]
// ============================================================

/** toTradeMarkers 接收的成交结构（后端 serialize_trades 输出）。 */
interface TradeInput {
  /** 成交时间，ISO 日期时间或 `'YYYY-MM-DD'`；无法解析的成交会被跳过 */
  datetime: string
  /** 开平标志，含 "OPEN" 判为买入、否则判为卖出（大小写不敏感） */
  offset: string
  /** 买卖方向原始字段，当前判向以 offset 为准，此列暂未参与计算 */
  direction: string
  /** 成交价（元） */
  price: number
  /** 成交数量（股） */
  volume: number
}

/**
 * 把回测成交转换为 TradeMarker[]（Req 5.2）。
 * - 买卖语义以 offset 为主判据：OPEN→buy（买入）、CLOSE→sell（卖出），大小写不敏感
 * - 时间无法解析的成交被跳过；时间落在可选 barTimeRange 之外的成交被过滤（Req 7.4）
 * - text 形如 "买 1000@12.34" / "卖 1000@12.34"
 *
 * @param trades - 成交数组；非数组时按空输入处理
 * @param barTimeRange - 可选的 K 线时间窗 `{ min, max }`，用于剔除落在行情区间外的成交；
 *   缺省或与时间类型不一致时不做区间过滤
 * @returns 与成交对应的买卖点标注数组；无可用成交时返回空数组
 */
export function toTradeMarkers(
  trades: TradeInput[],
  barTimeRange?: { min: ChartTime; max: ChartTime },
): TradeMarker[] {
  if (!Array.isArray(trades)) return []

  const markers: TradeMarker[] = []
  for (const t of trades) {
    if (t === null || typeof t !== 'object') continue
    const time = parseChartTime(t.datetime)
    if (time === null) continue
    if (!inRange(time, barTimeRange)) continue

    const offset = typeof t.offset === 'string' ? t.offset.toUpperCase() : ''
    // 以 offset 为主判据（开仓=买、平仓=卖），契合 A 股多头策略
    const side: TradeSide = offset.includes('OPEN') ? 'buy' : 'sell'
    const price = num(t.price)
    const volume = num(t.volume)
    const label = side === 'buy' ? '买' : '卖'
    markers.push({
      time,
      side,
      ...(price === null ? {} : { price }),
      text: `${label} ${volume ?? ''}@${price ?? ''}`.trim(),
    })
  }
  return markers
}

// ============================================================
// 逐日净值 → EquityPoint[]
// ============================================================

/** toEquityPoints 接收的净值行结构（后端 serialize_equity_curve 输出）。 */
interface EquityRowInput {
  /** 交易日，`'YYYY-MM-DD'` */
  date: string
  /** 当日账户净值（元） */
  balance: number
  /** 当日回撤金额（元） */
  drawdown: number
  /** 当日回撤百分比（%） */
  ddpercent: number
  /** 当日净盈亏（元），映射为 EquityPoint.netPnl */
  net_pnl: number
  /** 策略累计收益（%），缺值保留 null */
  strategy_return?: number | null
  /** 基准（买入持有）累计收益（%），缺值保留 null */
  benchmark_return?: number | null
  /** 超额收益（%）= 策略累计 - 基准累计，缺值保留 null */
  excess_return?: number | null
}

/**
 * 把回测逐日净值转换为 EquityPoint[]（Req 5.3）。
 * - 字段映射：net_pnl→netPnl；date/balance/drawdown/ddpercent 透传
 * - 收益列（strategy_return/benchmark_return/excess_return）按 % 透传，缺值保留 null
 * - 空输入返回 []（Req 5.4）
 *
 * @param rows - 逐日净值行数组；非数组时按空输入处理
 * @returns 净值曲线点数组；balance/drawdown/ddpercent/netPnl 不可解析时归零，
 *   三个收益列不可解析时保留 `null`
 */
export function toEquityPoints(rows: EquityRowInput[]): EquityPoint[] {
  if (!Array.isArray(rows)) return []
  return rows.map((r) => ({
    date: r.date,
    balance: num(r.balance) ?? 0,
    drawdown: num(r.drawdown) ?? 0,
    ddpercent: num(r.ddpercent) ?? 0,
    netPnl: num(r.net_pnl) ?? 0,
    strategyReturn: num(r.strategy_return),
    benchmarkReturn: num(r.benchmark_return),
    excessReturn: num(r.excess_return),
  }))
}
