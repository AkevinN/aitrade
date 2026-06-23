import { describe, it, expect } from 'vitest'

import {
  barsPerDay,
  barsToDays,
  daysToBars,
  inferIntervalFromDatetimes,
} from './barInterval'

describe('barInterval', () => {
  it('barsPerDay 已知周期 / 未知周期返回 null', () => {
    expect(barsPerDay('d')).toBe(1)
    expect(barsPerDay('30m')).toBe(8)
    expect(barsPerDay('1m')).toBe(240)
    expect(barsPerDay('3m')).toBeNull()
  })

  it('daysToBars = days × 每日 bar 数；日线恒等；未知周期 null', () => {
    expect(daysToBars(30, '30m')).toBe(240)
    expect(daysToBars(5, 'd')).toBe(5)
    expect(daysToBars(1, '1m')).toBe(240)
    expect(daysToBars(10, '3m')).toBeNull()
  })

  it('barsToDays 向上取整（与后端 bars_to_days 同口径）；日线恒等', () => {
    expect(barsToDays(10, '30m')).toBe(2) // ceil(10/8)
    expect(barsToDays(8, '30m')).toBe(1)
    expect(barsToDays(17, '30m')).toBe(3)
    expect(barsToDays(5, 'd')).toBe(5)
    expect(barsToDays(0, '30m')).toBe(0)
    expect(barsToDays(10, '3m')).toBeNull()
  })
})

describe('inferIntervalFromDatetimes', () => {
  it('相邻 30 分钟成交 → 反推 30m（午休大缺口被最小间隔天然跳过）', () => {
    expect(
      inferIntervalFromDatetimes([
        '2024-10-09T10:30:00',
        '2024-10-09T11:00:00',
        '2024-10-09T11:30:00',
        '2024-10-09T13:30:00', // 11:30→13:30 午休缺口不应污染判定
      ]),
    ).toBe('30m')
  })

  it('相邻 1 分钟成交 → 反推 1m', () => {
    expect(
      inferIntervalFromDatetimes([
        '2024-10-09T09:31:00',
        '2024-10-09T09:32:00',
        '2024-10-09T09:33:00',
      ]),
    ).toBe('1m')
  })

  it('跨多日：最小间隔取自同日相邻对，不被跨日缺口误导', () => {
    expect(
      inferIntervalFromDatetimes([
        '2024-10-09T10:00:00',
        '2024-10-09T10:15:00',
        '2024-10-10T10:00:00',
        '2024-10-10T10:15:00',
      ]),
    ).toBe('15m')
  })

  it('全部为纯日期/零点 → 判定为日线 d', () => {
    expect(inferIntervalFromDatetimes(['2024-10-09', '2024-10-10', '2024-10-11'])).toBe('d')
    expect(inferIntervalFromDatetimes(['2024-10-09T00:00:00', '2024-10-10T00:00:00'])).toBe('d')
  })

  it('非整周期最小间隔就近吸附到支持周期（32 分钟 → 30m）', () => {
    expect(
      inferIntervalFromDatetimes(['2024-10-09T10:00:00', '2024-10-09T10:32:00']),
    ).toBe('30m')
  })

  it('样本不足/无同日相邻对/全非法 → 返回 null（交由调用方回退）', () => {
    expect(inferIntervalFromDatetimes([])).toBeNull()
    expect(inferIntervalFromDatetimes(['2024-10-09T10:00:00'])).toBeNull()
    expect(inferIntervalFromDatetimes([null, undefined, 'not-a-date'])).toBeNull()
  })
})
