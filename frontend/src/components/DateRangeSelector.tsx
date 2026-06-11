import React, { useMemo } from 'react'
import { Button, DatePicker, Space, Typography } from 'antd'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'

const { Text } = Typography
const { RangePicker } = DatePicker

export interface LocalDateRange {
  start: string
  end: string
}

export interface DateRangeSelectorProps {
  value?: [Dayjs, Dayjs] | null
  onChange?: (value: [Dayjs, Dayjs] | null) => void
  localRange?: LocalDateRange | null
  showPresets?: boolean
}

type DateRangePreset = {
  key: string
  label: string
  build: (anchorEnd: Dayjs, localRange?: LocalDateRange | null) => [Dayjs, Dayjs]
}

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

  const applyPreset = (preset: DateRangePreset) => {
    const nextRange = clampRange(preset.build(anchorEnd, localRange), localRange)
    onChange?.(nextRange)
  }

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
