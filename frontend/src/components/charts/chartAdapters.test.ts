// 单元测试：图表数据适配层纯函数（chartAdapters.ts）。
//
// 测试运行器为仓库已配置的 Vitest。覆盖适配函数的正常映射、
// 缺列跳过、空输入、排序与过滤等示例及边界用例。
// _Requirements: 5.4, 5.5, 7.3, 7.4_

import { describe, it, expect } from 'vitest'

import {
  parseChartTime,
  toOHLCBars,
  toTradeMarkers,
  toEquityPoints,
} from './chartAdapters'

// ---------------------------------------------------------------------------
// parseChartTime（两种输入：纯日期字符串 / ISO 日期时间）
// _Requirements: 5.4_
// ---------------------------------------------------------------------------

describe('parseChartTime', () => {
  // 纯日期 'YYYY-MM-DD'：日线，原样返回字符串
  it('返回纯日期字符串原样（日线）', () => {
    expect(parseChartTime('2025-03-04')).toBe('2025-03-04')
  })

  // ISO 日期时间（无时区）：补 'Z' 按 UTC 解释，转为秒级时间戳
  it('把无时区 ISO 日期时间转为秒级 UTC 时间戳', () => {
    const expected = Math.floor(Date.parse('2025-03-04T15:00:00Z') / 1000)
    expect(parseChartTime('2025-03-04T15:00:00')).toBe(expected)
  })

  // 带时区后缀的 ISO：按其声明时区解析
  it('把带 Z 后缀的 ISO 日期时间转为秒级时间戳', () => {
    const expected = Math.floor(Date.parse('2025-03-04T15:00:00Z') / 1000)
    expect(parseChartTime('2025-03-04T15:00:00Z')).toBe(expected)
  })

  // 已是数字：视为秒级时间戳原样返回
  it('数字输入视为秒级时间戳原样返回', () => {
    expect(parseChartTime(1_700_000_000)).toBe(1_700_000_000)
  })

  // 无法解析 → null
  it('无法解析的输入返回 null', () => {
    expect(parseChartTime('not-a-date')).toBeNull()
    expect(parseChartTime('')).toBeNull()
    expect(parseChartTime(null)).toBeNull()
    expect(parseChartTime(undefined)).toBeNull()
    expect(parseChartTime(NaN)).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// toOHLCBars（正常映射、缺 close 行被跳过、空 []、升序）
// _Requirements: 5.4, 7.3_
// ---------------------------------------------------------------------------

describe('toOHLCBars', () => {
  // 正常映射：datetime/open/high/low/close/volume 全部映射
  it('正常映射行情行到 OHLCBar（含 volume）', () => {
    const rows = [
      {
        datetime: '2025-03-04',
        open: 10,
        high: 12,
        low: 9,
        close: 11,
        volume: 1000,
      },
    ]
    expect(toOHLCBars(rows)).toEqual([
      { time: '2025-03-04', open: 10, high: 12, low: 9, close: 11, volume: 1000 },
    ])
  })

  // 缺 volume 时输出不含 volume 字段
  it('缺 volume 列时输出不含 volume 字段', () => {
    const rows = [{ datetime: '2025-03-04', open: 10, high: 12, low: 9, close: 11 }]
    expect(toOHLCBars(rows)).toEqual([
      { time: '2025-03-04', open: 10, high: 12, low: 9, close: 11 },
    ])
  })

  // 缺 close 列的行被跳过，不抛异常
  it('缺 close 列的行被跳过', () => {
    const rows = [
      { datetime: '2025-03-04', open: 10, high: 12, low: 9, close: 11 },
      { datetime: '2025-03-05', open: 11, high: 13, low: 10 }, // 缺 close
    ]
    const bars = toOHLCBars(rows)
    expect(bars).toHaveLength(1)
    expect(bars[0].time).toBe('2025-03-04')
  })

  // 时间无法解析的行被跳过
  it('时间缺失或无法解析的行被跳过', () => {
    const rows = [
      { datetime: 'bad', open: 10, high: 12, low: 9, close: 11 },
      { open: 10, high: 12, low: 9, close: 11 }, // 无 datetime
    ]
    expect(toOHLCBars(rows)).toEqual([])
  })

  // 空输入返回 []
  it('空数组输入返回空数组', () => {
    expect(toOHLCBars([])).toEqual([])
  })

  // 非数组输入返回 []
  it('非数组输入返回空数组', () => {
    // @ts-expect-error 故意传入非法类型验证健壮性
    expect(toOHLCBars(null)).toEqual([])
  })

  // 输出按 time 升序排列
  it('输出按 time 升序排列', () => {
    const rows = [
      { datetime: '2025-03-06', open: 1, high: 2, low: 0.5, close: 1.5 },
      { datetime: '2025-03-04', open: 1, high: 2, low: 0.5, close: 1.5 },
      { datetime: '2025-03-05', open: 1, high: 2, low: 0.5, close: 1.5 },
    ]
    const times = toOHLCBars(rows).map((b) => b.time)
    expect(times).toEqual(['2025-03-04', '2025-03-05', '2025-03-06'])
  })
})

// ---------------------------------------------------------------------------
// toTradeMarkers（OPEN→buy / CLOSE→sell、越界过滤、空 []）
// _Requirements: 5.4, 7.4_
// ---------------------------------------------------------------------------

describe('toTradeMarkers', () => {
  // OPEN→buy、CLOSE→sell（大小写不敏感）
  it('按 offset 映射买卖方向：OPEN→buy、CLOSE→sell', () => {
    const trades = [
      { datetime: '2025-03-04', offset: 'OPEN', direction: 'LONG', price: 12.34, volume: 1000 },
      { datetime: '2025-03-05', offset: 'close', direction: 'LONG', price: 13.5, volume: 1000 },
    ]
    const markers = toTradeMarkers(trades)
    expect(markers).toHaveLength(2)
    expect(markers[0].side).toBe('buy')
    expect(markers[0].text).toBe('买 1000@12.34')
    expect(markers[1].side).toBe('sell')
    expect(markers[1].text).toBe('卖 1000@13.5')
  })

  // 越界过滤：落在 barTimeRange 之外的成交被过滤
  it('过滤掉 barTimeRange 区间之外的成交', () => {
    const trades = [
      { datetime: '2025-03-03', offset: 'OPEN', direction: 'LONG', price: 10, volume: 100 },
      { datetime: '2025-03-05', offset: 'OPEN', direction: 'LONG', price: 11, volume: 100 },
      { datetime: '2025-03-09', offset: 'CLOSE', direction: 'LONG', price: 12, volume: 100 },
    ]
    const markers = toTradeMarkers(trades, { min: '2025-03-04', max: '2025-03-06' })
    expect(markers).toHaveLength(1)
    expect(markers[0].time).toBe('2025-03-05')
  })

  // 时间无法解析的成交被跳过
  it('时间无法解析的成交被跳过', () => {
    const trades = [
      { datetime: 'bad', offset: 'OPEN', direction: 'LONG', price: 10, volume: 100 },
    ]
    expect(toTradeMarkers(trades)).toEqual([])
  })

  // 空输入返回 []
  it('空数组输入返回空数组', () => {
    expect(toTradeMarkers([])).toEqual([])
  })

  // 非数组输入返回 []
  it('非数组输入返回空数组', () => {
    // @ts-expect-error 故意传入非法类型验证健壮性
    expect(toTradeMarkers(null)).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// toEquityPoints（字段映射：net_pnl→netPnl，其余透传）
// _Requirements: 5.4, 5.5_
// ---------------------------------------------------------------------------

describe('toEquityPoints', () => {
  // 字段映射：net_pnl→netPnl，date/balance/drawdown/ddpercent 透传；缺收益列时收益字段为 null
  it('映射净值行字段（net_pnl→netPnl）', () => {
    const rows = [
      { date: '2025-03-04', balance: 100000, drawdown: -500, ddpercent: -0.5, net_pnl: 1200 },
    ]
    expect(toEquityPoints(rows)).toEqual([
      {
        date: '2025-03-04',
        balance: 100000,
        drawdown: -500,
        ddpercent: -0.5,
        netPnl: 1200,
        strategyReturn: null,
        benchmarkReturn: null,
        excessReturn: null,
      },
    ])
  })

  // 收益列（strategy_return/benchmark_return/excess_return）按 % 透传
  it('透传策略/基准/超额收益（百分比）', () => {
    const rows = [
      {
        date: '2025-03-04',
        balance: 110000,
        drawdown: 0,
        ddpercent: 0,
        net_pnl: 0,
        strategy_return: 10,
        benchmark_return: 5,
        excess_return: 5,
      },
    ]
    expect(toEquityPoints(rows)).toEqual([
      {
        date: '2025-03-04',
        balance: 110000,
        drawdown: 0,
        ddpercent: 0,
        netPnl: 0,
        strategyReturn: 10,
        benchmarkReturn: 5,
        excessReturn: 5,
      },
    ])
  })

  // 空输入返回 []
  it('空数组输入返回空数组', () => {
    expect(toEquityPoints([])).toEqual([])
  })

  // 非数组输入返回 []
  it('非数组输入返回空数组', () => {
    // @ts-expect-error 故意传入非法类型验证健壮性
    expect(toEquityPoints(null)).toEqual([])
  })
})
