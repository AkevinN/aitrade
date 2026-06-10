// 通用「策略 vs 基准」累计收益对比组件，基于 recharts 实现。
// 纯展示组件：仅消费纯数据 props（points），不含任何回测业务语义。
// 业务数据 → 图表数据的转换由 chartAdapters 纯函数完成。
import { Empty } from 'antd'
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { EquityPoint } from './types'

export interface ReturnComparisonChartProps {
  points: EquityPoint[]
  /** 图表高度（px），默认 280 */
  height?: number
}

// 策略（蓝）/ 基准（灰）/ 超额（红绿区域）配色
const STRATEGY_COLOR = '#1668dc'
const BENCHMARK_COLOR = '#8c8c8c'
const EXCESS_COLOR = '#49aa19'

/** 百分比格式化（保留两位小数）。 */
const fmtPct = (v: number | null | undefined) =>
  v === null || v === undefined ? '-' : `${v.toFixed(2)}%`

export default function ReturnComparisonChart({
  points,
  height = 280,
}: ReturnComparisonChartProps) {
  // 无任何基准收益数据 → 占位空状态（与净值曲线一致的空态处理）
  const hasBenchmark = points.some((p) => p.benchmarkReturn !== null && p.benchmarkReturn !== undefined)
  if (points.length === 0 || !hasBenchmark) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Empty description="暂无基准对比数据" />
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={points} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="#f0f0f0" />
        <XAxis dataKey="date" tick={{ fontSize: 10 }} />
        {/* 左轴：策略 / 基准累计收益（%） */}
        <YAxis yAxisId="ret" tick={{ fontSize: 10 }} width={56} unit="%" domain={['auto', 'auto']} />
        {/* 右轴：超额收益（%） */}
        <YAxis
          yAxisId="excess"
          orientation="right"
          tick={{ fontSize: 10 }}
          width={56}
          unit="%"
          domain={['auto', 'auto']}
        />
        <Tooltip formatter={(value) => fmtPct(Number(value))} />
        <Legend />
        <ReferenceLine yAxisId="ret" y={0} stroke="#d9d9d9" />
        {/* 超额收益区域（正绿负红由数值本身体现，统一绿色填充背景） */}
        <Area
          yAxisId="excess"
          type="monotone"
          dataKey="excessReturn"
          name="超额收益(%)"
          stroke={EXCESS_COLOR}
          fill={EXCESS_COLOR}
          fillOpacity={0.12}
          dot={false}
          connectNulls
          isAnimationActive={false}
        />
        {/* 基准（买入持有）累计收益 */}
        <Line
          yAxisId="ret"
          type="monotone"
          dataKey="benchmarkReturn"
          name="基准收益(%)"
          stroke={BENCHMARK_COLOR}
          strokeDasharray="4 2"
          dot={false}
          connectNulls
          isAnimationActive={false}
        />
        {/* 策略累计收益 */}
        <Line
          yAxisId="ret"
          type="monotone"
          dataKey="strategyReturn"
          name="策略收益(%)"
          stroke={STRATEGY_COLOR}
          dot={false}
          connectNulls
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
