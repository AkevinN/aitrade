import React, { useEffect, useMemo, useState } from 'react'
import { Collapse, Drawer, Empty, Space, Spin, Table, Tag, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'

import type { ScreeningFold, Tier2Verdict } from '../../types/screening'
import type { BacktestResultPayload, BacktestTrade } from '../../types/alpha'
import { screeningService } from '../../api/screening'
import { inferIntervalFromDatetimes } from '../../utils/barInterval'
import BacktestResults from '../Backtest/BacktestResults'
import BacktestCharts from '../Backtest/BacktestCharts'
import GateVerdictHeader from './GateVerdictHeader'
import FoldTable from './FoldTable'

const { Text } = Typography

/** Tier-2 折级详情抽屉的 Props。 */
interface Tier2DetailDrawerProps {
  /** 是否打开 */
  open: boolean
  /** 触发抽屉的榜单行结论；report_id 为空时不应被打开 */
  verdict: Tier2Verdict | null
  /**
   * K 线周期的**最末回落值**——优先用报告自身回测配置（report.request.input_interval），
   * 其次从成交时间戳数据驱动反推，二者都无时才用本 prop，最后回落 'd'。用于拉 K 线图的 OHLC 行情。
   */
  interval?: string
  /** 关闭回调 */
  onClose: () => void
}

/** 成交明细表列：成交流水（无单笔 PnL，仅 价/量/方向/开平）。 */
const tradeColumns = [
  {
    title: '时间',
    dataIndex: 'datetime',
    render: (v: string) => <Text style={{ fontSize: 12 }}>{String(v).replace('T', ' ').slice(0, 19)}</Text>,
  },
  {
    title: '方向',
    key: 'side',
    width: 110,
    render: (_: unknown, t: BacktestTrade) => {
      const isOpen = String(t.offset).toLowerCase().includes('open')
      return <Tag color={isOpen ? 'green' : 'red'}>{isOpen ? '买入 开' : '卖出 平'}</Tag>
    },
  },
  { title: '价格', dataIndex: 'price', width: 100, render: (v: number) => v?.toFixed(3) },
  { title: '数量', dataIndex: 'volume', width: 100, render: (v: number) => v?.toLocaleString() },
]

/**
 * Tier-2 折级详情抽屉：按 `verdict.report_id` 拉取 WF 报告，纵向堆叠展示
 * 门禁头部 → 折级总览表（点行联动）→ 选中折回测指标卡（复用 BacktestResults）→
 * 选中折 OOS 净值/回撤曲线 + K 线买卖点（复用 BacktestCharts）→ 成交明细表。
 *
 * 报告 404/加载失败/无折 时渲染空态占位，不崩溃；门禁头部始终以 `verdict` 渲染
 * （来自榜单行，无需等待报告）。曲线/成交取后端保留的第 0 种子（代表）数据。
 *
 * K 线周期取自报告自身的回测配置（report.request.input_interval），保证「bar 是什么图就用
 * 什么图」——分钟线选股的标的不会再被硬拉日线行情而报错；报告缺该字段时，从折级成交流水的
 * 时间戳**数据驱动**反推周期（成交按 30m 落点即拉 30m），二者皆无才回落 `interval` prop、再到 'd'。
 *
 * @param open - 是否打开
 * @param verdict - 触发抽屉的榜单行结论
 * @param interval - K 线周期的最末回落值（报告含周期或可从成交反推时不生效），缺省 'd'
 * @param onClose - 关闭回调
 */
const Tier2DetailDrawer: React.FC<Tier2DetailDrawerProps> = ({
  open,
  verdict,
  interval = 'd',
  onClose,
}) => {
  const reportId = verdict?.report_id ?? null
  const [selectedFold, setSelectedFold] = useState(0)

  const {
    data: report,
    isFetching,
    isError,
  } = useQuery({
    queryKey: ['screening-report', reportId],
    queryFn: () => screeningService.getScreeningReport(reportId as string),
    enabled: open && !!reportId,
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  // 报告切换时重置选中折为首折，避免沿用上一只标的的选中态。
  useEffect(() => {
    if (report?.folds?.length) {
      setSelectedFold(report.folds[0].fold)
    }
  }, [report?.report_id])

  const selected: ScreeningFold | null =
    report?.folds.find((f) => f.fold === selectedFold) ?? report?.folds[0] ?? null

  // K 线行情应复用该折回测**实际所用**的 bar 周期，而非外部默认的 'd'：折级净值/成交流水
  // 都是按这条 K 线跑出来的，用日线去拉一只只有分钟线的标的（如非日线选股）必然取不到行情而报错。
  //
  // 兜底：报告未回显 input_interval（旧报告/边缘）时，直接从成交流水的时间戳**数据驱动**反推
  // 周期——"bar 是什么图就用什么图"：成交按 30m 落点就反推 30m，绝不在数据本是分钟级时硬回落到 'd'。
  // 聚合所有折的成交以放大样本（周期是报告级、各折一致）。
  const inferredInterval = useMemo(
    () =>
      inferIntervalFromDatetimes(
        (report?.folds ?? []).flatMap((f) => (f.candidate_trades ?? []).map((t) => t.datetime)),
      ),
    [report?.folds],
  )

  // 解析优先级（单一事实源居首，数据驱动兜底其次，外部回落值垫底）：
  //   报告回测配置 report.request.input_interval → 成交时间戳反推 → interval prop（缺省 'd'）
  const reportInterval = report?.request?.input_interval
  const effectiveInterval =
    (typeof reportInterval === 'string' && reportInterval) || inferredInterval || interval

  // 把选中折的净值/成交/标的组装成回测载荷，整块喂 BacktestCharts（复用净值图 + K线买卖点 + OHLC 拉取）。
  const foldResult = useMemo<BacktestResultPayload | undefined>(() => {
    if (!selected || !verdict) return undefined
    return {
      target_symbol: verdict.vt_symbol,
      equity_curve: selected.candidate_equity_curve ?? [],
      trades: selected.candidate_trades ?? [],
    } as BacktestResultPayload
  }, [selected, verdict])

  const renderBody = () => {
    if (!verdict) {
      return <Empty description="无 Tier-2 结论" />
    }
    return (
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <GateVerdictHeader verdict={verdict} />
        {isFetching ? (
          <div style={{ textAlign: 'center', padding: '32px 0' }}>
            <Spin />
          </div>
        ) : isError || !report ? (
          <Empty description="报告不存在或加载失败" />
        ) : report.folds.length === 0 ? (
          <Empty description="该报告无折数据" />
        ) : (
          <>
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                折级总览（点行查看该折指标）
              </Text>
              <FoldTable folds={report.folds} selectedFold={selectedFold} onSelect={setSelectedFold} />
            </div>
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                选中折 #{selectedFold} · 回测指标
              </Text>
              <BacktestResults
                statistics={selected?.candidate_statistics}
                capital={selected?.candidate_statistics?.capital ?? 0}
              />
            </div>
            <BacktestCharts
              result={foldResult}
              interval={effectiveInterval}
              start={selected?.test.start ?? ''}
              end={selected?.test.end ?? ''}
            />
            <Collapse
              size="small"
              items={[
                {
                  key: 'trades',
                  label: `成交明细（${selected?.candidate_trades?.length ?? 0} 笔，代表种子）`,
                  children: (
                    <Table<BacktestTrade>
                      size="small"
                      rowKey={(t, i) => `${t.datetime}-${i}`}
                      dataSource={selected?.candidate_trades ?? []}
                      columns={tradeColumns}
                      pagination={{ pageSize: 10, showSizeChanger: false }}
                      locale={{ emptyText: '该折无成交' }}
                    />
                  ),
                },
              ]}
            />
          </>
        )}
      </Space>
    )
  }

  return (
    <Drawer
      title={verdict ? `${verdict.vt_symbol} · Tier-2 折级详情` : 'Tier-2 折级详情'}
      width={820}
      open={open}
      onClose={onClose}
    >
      {renderBody()}
    </Drawer>
  )
}

export default Tier2DetailDrawer
