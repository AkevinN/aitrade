import React, { useEffect, useState } from 'react'
import { InputNumber, Segmented, Space, Typography } from 'antd'

import { barsPerDay, barsToDays, daysToBars } from '../utils/barInterval'

const { Text } = Typography

/**
 * "按天配置 / 按 bar 手填" 双模数值输入（受控，供 antd Form.Item 包裹）。
 *
 * 作为 Form.Item 的子组件：Form.Item 注入 `value`(bar 根数) 与 `onChange`，本组件
 * 内部用 Segmented 切换两种填法——
 * - **按天**：用户填观测交易日数，按当前周期 `days × 每日 bar 数` 自动换算成 bar 根数
 *   并实时回写（提交侧仍读 bar 字段，无需改）；并显示换算文案。
 * - **按 bar**：直接手填 bar 根数（派生/自定义周期无法换算时强制此模式）。
 *
 * 这样把回看窗口 / 预测跨度 / OCO 最大持有等"以 bar 表达"的配置统一成"可按天配置、
 * 自动换算"，分钟线下用户无需再心算多少根。
 */
interface DayBarFieldProps {
  /** 当前 bar 根数（由 antd Form.Item 注入） */
  value?: number
  /** 值变更回调（由 antd Form.Item 注入）；emit 换算后的 bar 根数 */
  onChange?: (bars: number) => void
  /** 当前 K 线周期，决定每交易日 bar 数 */
  interval: string
  /** bar 根数上限，默认 120 */
  maxBars?: number
  /** bar 根数下限，默认 1 */
  minBars?: number
  /** 首次进入"按天"模式时的默认观测交易日数，默认 5 */
  defaultDays?: number
}

const MODE_OPTIONS = [
  { label: '按天', value: 'window' },
  { label: '按 bar', value: 'manual' },
]

const DayBarField: React.FC<DayBarFieldProps> = ({
  value,
  onChange,
  interval,
  maxBars = 120,
  minBars = 1,
  defaultDays = 5,
}) => {
  const bpd = barsPerDay(interval)
  const canConvert = bpd != null
  // days 上限：保证 days × bpd 不超过 maxBars（按天模式不会换出超限 bar）。
  const maxDays = bpd ? Math.max(1, Math.floor(maxBars / bpd)) : maxBars

  const [mode, setMode] = useState<'window' | 'manual'>(canConvert ? 'window' : 'manual')
  // 观测交易日数：优先从当前 bar 值反推（保留表单初值语义），否则用默认。
  const [days, setDays] = useState<number>(() => {
    if (canConvert && value != null && value > 0) return barsToDays(value, interval) ?? defaultDays
    return defaultDays
  })

  // 周期变成不可换算时强制 manual。
  useEffect(() => {
    if (!canConvert && mode === 'window') setMode('manual')
  }, [canConvert, mode])

  // 按天模式下把 days×bpd 实时回写为 bar 值，使表单值与展示一致（提交读 bar 字段不变）。
  useEffect(() => {
    if (mode === 'window' && canConvert) {
      const bars = daysToBars(days, interval)
      if (bars != null && bars !== value) onChange?.(bars)
    }
    // 仅在 模式/天数/周期 变化时同步；value/onChange 故意不入依赖以避免回写循环。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, days, interval])

  return (
    <Space direction="vertical" size={4} style={{ width: '100%' }}>
      <Segmented
        size="small"
        options={MODE_OPTIONS}
        value={mode}
        disabled={!canConvert}
        onChange={(v) => setMode(v as 'window' | 'manual')}
      />
      {mode === 'window' && canConvert ? (
        <>
          <InputNumber
            min={1}
            max={maxDays}
            value={days}
            onChange={(v) => setDays(Math.max(1, Number(v) || 1))}
            addonAfter="交易日"
            style={{ width: '100%' }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            每交易日 {bpd} 根 × {days} 日 → {daysToBars(days, interval)} 根 bar
          </Text>
        </>
      ) : (
        <>
          <InputNumber
            min={minBars}
            max={maxBars}
            value={value}
            onChange={(v) => onChange?.(Math.max(minBars, Number(v) || minBars))}
            addonAfter="bar"
            style={{ width: '100%' }}
          />
          {!canConvert ? (
            <Text type="secondary" style={{ fontSize: 12 }}>
              周期「{interval}」不在换算表内，请直接填 bar 根数
            </Text>
          ) : null}
        </>
      )}
    </Space>
  )
}

export default DayBarField
