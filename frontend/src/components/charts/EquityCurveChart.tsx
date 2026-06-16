// 通用净值/回撤曲线组件，基于 recharts 实现。
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
  ResponsiveContainer,
} from 'recharts'
import type { EquityPoint } from './types'

/**
 * {@link EquityCurveChart} 组件 props。
 */
export interface EquityCurveChartProps {
  /** 净值曲线数据点（按日期升序）；空数组时渲染占位空状态。 */
  points: EquityPoint[]
  /** 图表高度（px），默认 280 */
  height?: number
  /** 是否绘制回撤区域，默认 true */
  showDrawdown?: boolean
}

// 净值线（蓝）/ 回撤区域（红）配色，回撤为负向区域
const BALANCE_COLOR = '#1668dc'
const DRAWDOWN_COLOR = '#dc4446'

/**
 * 净值/回撤曲线图（基于 recharts）。
 *
 * 左轴展示账户净值（蓝色折线），右轴叠加回撤百分比（红色填充区域）。
 * `points` 为空时渲染「暂无净值数据」占位。
 */
export default function EquityCurveChart({
  points,
  height = 280,
  showDrawdown = true,
}: EquityCurveChartProps) {
  // 空数据 → 占位空状态（Req 4.4）
  if (points.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Empty description="暂无净值数据" />
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={points} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="#f0f0f0" />
        {/* 共享 date X 轴 */}
        <XAxis dataKey="date" tick={{ fontSize: 10 }} />
        {/* 左轴：账户净值 balance */}
        <YAxis
          yAxisId="balance"
          tick={{ fontSize: 10 }}
          domain={['auto', 'auto']}
          width={64}
        />
        {/* 右轴：回撤百分比 ddpercent（负向） */}
        {showDrawdown && (
          <YAxis
            yAxisId="ddpercent"
            orientation="right"
            tick={{ fontSize: 10 }}
            domain={['auto', 0]}
            width={48}
            unit="%"
          />
        )}
        <Tooltip />
        <Legend />
        {/* 回撤区域先画，作为净值线的背景 */}
        {showDrawdown && (
          <Area
            yAxisId="ddpercent"
            type="monotone"
            dataKey="ddpercent"
            name="回撤(%)"
            stroke={DRAWDOWN_COLOR}
            fill={DRAWDOWN_COLOR}
            fillOpacity={0.15}
            dot={false}
            isAnimationActive={false}
          />
        )}
        {/* 净值曲线 */}
        <Line
          yAxisId="balance"
          type="monotone"
          dataKey="balance"
          name="净值"
          stroke={BALANCE_COLOR}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
