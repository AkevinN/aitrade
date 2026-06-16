import React, { useMemo } from 'react'
import { Button, DatePicker, Space, Typography } from 'antd'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'

const { Text } = Typography
const { RangePicker } = DatePicker

/** 本地数据可用时间范围（ISO 日期字符串）。 */
export interface LocalDateRange {
  /** 可用起始日期（ISO 格式，如 `"2024-01-01"`）。 */
  start: string
  /** 可用结束日期（ISO 格式）。 */
  end: string
}

/**
 * {@link DateRangeSelector} 组件 props。
 */
export interface DateRangeSelectorProps {
  /** 当前选中区间（受控）；`null` 表示未选。 */
  value?: [Dayjs, Dayjs] | null
  /** 区间变化回调；传入 `null` 表示清空选择。 */
  onChange?: (value: [Dayjs, Dayjs] | null) => void
  /**
   * 本地数据可用区间，用于：
   * 1. 约束快捷预设不超出本地范围；
   * 2. 显示「本地全区间」快捷按钮；
   * 3. 展示可用区间提示文字。
   */
  localRange?: LocalDateRange | null
  /** 是否展示快捷预设按钮（近1月/3月/6月等），默认 `true`。 */
  showPresets?: boolean
}

/** 一个快捷区间预设（如「近1月」），描述按钮文案与区间构造方式。 */
type DateRangePreset = {
  /** 预设唯一标识，用作 React key 与内部区分，如 `"1m"`、`"3y"`。 */
  key: string
  /** 按钮上展示的文案，如 `"近1月"`。 */
  label: string
  /**
   * 根据锚点结束日构造区间。
   *
   * @param anchorEnd - 区间右端锚点（通常为本地数据结束日或今天）。
   * @param localRange - 本地可用范围，预留给需感知边界的预设；当前内置预设未使用。
   * @returns `[start, end]` 区间，尚未经 {@link clampRange} 夹紧。
   */
  build: (anchorEnd: Dayjs, localRange?: LocalDateRange | null) => [Dayjs, Dayjs]
}

/** 内置快捷区间预设：近1月/3月/6月/1年/3年，均以锚点结束日向前回溯。 */
const DATE_RANGE_PRESETS: DateRangePreset[] = [
  {
    key: '1m',
    label: '近1月',
    build: (anchorEnd) => [anchorEnd.subtract(1, 'month'), anchorEnd],
  },
  {
    key: '3m',
    label: '近3月',
    build: (anchorEnd) => [anchorEnd.subtract(3, 'month'), anchorEnd],
  },
  {
    key: '6m',
    label: '近6月',
    build: (anchorEnd) => [anchorEnd.subtract(6, 'month'), anchorEnd],
  },
  {
    key: '1y',
    label: '近1年',
    build: (anchorEnd) => [anchorEnd.subtract(1, 'year'), anchorEnd],
  },
  {
    key: '3y',
    label: '近3年',
    build: (anchorEnd) => [anchorEnd.subtract(3, 'year'), anchorEnd],
  },
]

/**
 * 将预设区间收缩到本地可用区间范围内。
 *
 * 若 `localRange` 为空则直接返回原区间；否则将 start 下限夹紧到 localRange.start，
 * end 上限夹紧到 localRange.end；若夹紧后 start > end 则退回整个 localRange。
 *
 * @param range - 原始区间。
 * @param localRange - 本地数据可用范围。
 * @returns 夹紧后的区间。
 */
const clampRange = (
  range: [Dayjs, Dayjs],
  localRange?: LocalDateRange | null,
): [Dayjs, Dayjs] => {
  if (!localRange) {
    return range
  }
  const localStart = dayjs(localRange.start.slice(0, 10))
  const localEnd = dayjs(localRange.end.slice(0, 10))
  let [start, end] = range
  if (start.isBefore(localStart, 'day')) {
    start = localStart
  }
  if (end.isAfter(localEnd, 'day')) {
    end = localEnd
  }
  if (start.isAfter(end, 'day')) {
    return [localStart, localEnd]
  }
  return [start, end]
}

/**
 * 日期区间选择器：Ant Design RangePicker + 快捷预设按钮组。
 *
 * 提供「近1月/3月/6月/1年/3年」快捷预设，以及「使用本地全区间」按钮（当 `localRange` 存在时显示）。
 * 所有预设区间都会被 {@link clampRange} 约束在 `localRange` 范围内，避免请求无数据的区间。
 */
const DateRangeSelector: React.FC<DateRangeSelectorProps> = ({
  value,
  onChange,
  localRange,
  showPresets = true,
}) => {
  const anchorEnd = useMemo(
    () => (localRange ? dayjs(localRange.end.slice(0, 10)) : dayjs()),
    [localRange],
  )

  /**
   * 应用一个快捷预设区间：按预设规则算出起止日期，约束到本地范围后回调出去。
   *
   * 以 `anchorEnd`（本地数据末尾或今天）为锚点调用 `preset.build` 生成区间，
   * 再经 {@link clampRange} 夹到 `localRange` 内，最后通过 `onChange` 通知父组件更新选中值。
   *
   * @param preset - 被点击的预设项，含 `build` 函数与展示标签（如「近1月」）
   */
  const applyPreset = (preset: DateRangePreset) => {
    const nextRange = clampRange(preset.build(anchorEnd, localRange), localRange)
    onChange?.(nextRange)
  }

  /**
   * 一键把选中区间设为本地数据的完整可用区间。
   *
   * 取 `localRange` 的起止日期（截取到日，去掉时间部分）作为新区间并通过 `onChange` 回调；
   * 若未传入 `localRange` 则直接返回、不做任何操作。
   */
  const applyLocalRange = () => {
    if (!localRange) {
      return
    }
    onChange?.([
      dayjs(localRange.start.slice(0, 10)),
      dayjs(localRange.end.slice(0, 10)),
    ])
  }

  return (
    <Space direction="vertical" size={8} style={{ width: '100%' }}>
      <RangePicker
        style={{ width: '100%' }}
        value={value}
        onChange={(nextValue) => {
          if (!nextValue || !nextValue[0] || !nextValue[1]) {
            onChange?.(null)
            return
          }
          onChange?.(clampRange([nextValue[0], nextValue[1]], localRange))
        }}
      />
      {showPresets ? (
        <Space size={[6, 6]} wrap>
          {DATE_RANGE_PRESETS.map((preset) => (
            <Button key={preset.key} size="small" onClick={() => applyPreset(preset)}>
              {preset.label}
            </Button>
          ))}
          {localRange ? (
            <Button size="small" type="primary" ghost onClick={applyLocalRange}>
              使用本地全区间
            </Button>
          ) : null}
        </Space>
      ) : null}
      {localRange ? (
        <Text type="secondary" style={{ fontSize: 12 }}>
          本地数据可用：{localRange.start.slice(0, 10)} ~ {localRange.end.slice(0, 10)}
        </Text>
      ) : null}
    </Space>
  )
}

export default DateRangeSelector
