// 回测结果图表容器（业务接线组件）。
// 承载回测专有逻辑：决定取哪个标的的 K 线、用什么周期/区间，并把通用图表组件
// （EquityCurveChart / KLineChart）与纯函数适配器（chartAdapters）组装起来。
// 通用组件不含任何回测语义，业务接线全部收敛在本文件（见 design.md 组件 #6）。
import { useMemo } from 'react'
import { Card, Alert, Space } from 'antd'
import { useQuery } from '@tanstack/react-query'

import { alphaService } from '../../api/alpha'
import EquityCurveChart from '../../components/charts/EquityCurveChart'
import ReturnComparisonChart from '../../components/charts/ReturnComparisonChart'
import KLineChart from '../../components/charts/KLineChart'
import { toEquityPoints, toOHLCBars, toTradeMarkers } from '../../components/charts/chartAdapters'
import type { ChartTime } from '../../components/charts/types'
import type { BacktestResultPayload } from '../../types/alpha'

/**
 * {@link BacktestCharts} 组件 props。
 */
export interface BacktestChartsProps {
  /** 回测结果载荷（来自 task.result），含成交明细与逐日净值 */
  result?: BacktestResultPayload
  /** 回测周期（CNN: input_interval；Alpha: scheme.interval），用于拉取 OHLC 行情 */
  interval: string
  /** 回测起始日期（YYYY-MM-DD），仅用于 query 缓存键，行情区间由数据本身决定 */
  start: string
  /** 回测结束日期（YYYY-MM-DD），同上 */
  end: string
}

/**
 * 回测结果图表容器（业务接线组件）。
 *
 * 组装净值曲线图（{@link EquityCurveChart}）、策略 vs 基准收益对比图（{@link ReturnComparisonChart}）
 * 和 K 线+买卖点图（{@link KLineChart}），从 `result` 载荷提取目标标的后
 * 自动拉取 OHLC 行情、过滤越界买卖点。
 */
export default function BacktestCharts({ result, interval, start, end }: BacktestChartsProps) {
  // 1. 净值曲线：逐日净值 → EquityPoint[]。空数据由 EquityCurveChart 自身渲染空状态（Req 6.4）。
  const equityPoints = useMemo(
    () => toEquityPoints(result?.equity_curve ?? []),
    [result?.equity_curve],
  )

  // 是否含基准对比数据：任一净值点带有效 benchmarkReturn 即展示「策略 vs 基准」卡片。
  const hasBenchmark = useMemo(
    () => equityPoints.some((p) => p.benchmarkReturn !== null && p.benchmarkReturn !== undefined),
    [equityPoints],
  )

  // 2. 确定 K 线标的：优先 CNN 回测回传的 target_symbol，缺省取首笔成交的 vt_symbol。
  const vtSymbol = result?.target_symbol ?? result?.trades?.[0]?.vt_symbol ?? ''

  // 3. 拉取回测标的在该周期下的 OHLC 行情。仅当标的与周期可用时才发起请求。
  //    加载态/错误态由 query 状态驱动（Req 6.5、6.6）。
  const ohlcQuery = useQuery({
    queryKey: ['backtest-ohlc', interval, vtSymbol, start, end],
    queryFn: () => alphaService.getBarDataDetail(interval, vtSymbol),
    enabled: Boolean(vtSymbol) && Boolean(interval),
  })

  // 行情行 → OHLCBar[]（升序）。脏行/缺列由适配器跳过（Req 5.4）。
  const bars = useMemo(
    () => toOHLCBars(ohlcQuery.data?.preview ?? []),
    [ohlcQuery.data?.preview],
  )

  // K 线时间范围（bars 已按时间升序，取首尾即可），用于过滤越界买卖点（Req 7.4）。
  const barTimeRange = useMemo<{ min: ChartTime; max: ChartTime } | undefined>(() => {
    if (bars.length === 0) return undefined
    return { min: bars[0].time, max: bars[bars.length - 1].time }
  }, [bars])

  // 4. 买卖点：成交 → TradeMarker[]，落在 K 线时间范围内。
  const markers = useMemo(
    () => toTradeMarkers(result?.trades ?? [], barTimeRange),
    [result?.trades, barTimeRange],
  )

  // K 线区内容：标的缺失 / 行情失败 / 正常三种分支。
  // 关键：错误仅在 K 线卡片内呈现，绝不影响上方净值曲线与既有统计卡片（Req 6.6、6.7）。
  const renderKLine = () => {
    if (!vtSymbol) {
      // 无 target_symbol 也无成交 → 本次回测无成交，给空状态而非报错（Req 6.4）。
      return <KLineChart bars={[]} emptyText="本次回测无成交" />
    }
    if (ohlcQuery.isError) {
      return (
        <Alert
          type="error"
          showIcon
          message="K 线行情获取失败"
          description={`无法加载标的 ${vtSymbol} 的 ${interval} 行情数据，净值曲线与统计数字不受影响。`}
        />
      )
    }
    return (
      <KLineChart
        bars={bars}
        markers={markers}
        loading={ohlcQuery.isLoading}
        emptyText="暂无行情数据"
      />
    )
  }

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>
      <Card title="净值 / 回撤曲线" size="small">
        <EquityCurveChart points={equityPoints} />
      </Card>
      {hasBenchmark && (
        <Card title="策略 vs 基准（买入持有）累计收益" size="small">
          <ReturnComparisonChart points={equityPoints} />
        </Card>
      )}
      <Card title="K 线与买卖点" size="small">
        {renderKLine()}
      </Card>
    </Space>
  )
}
