import { describe, it, expect } from 'vitest'

import { barsPerDay, barsToDays, daysToBars } from './barInterval'

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
