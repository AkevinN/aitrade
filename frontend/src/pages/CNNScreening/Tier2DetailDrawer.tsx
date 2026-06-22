import React, { useEffect, useMemo, useState } from 'react'
import { Collapse, Drawer, Empty, Space, Spin, Table, Tag, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'

import type { ScreeningFold, Tier2Verdict } from '../../types/screening'
import type { BacktestResultPayload, BacktestTrade } from '../../types/alpha'
import { screeningService } from '../../api/screening'
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
  /** 本次选股的 K 线周期，用于拉取 K 线图的 OHLC 行情；缺省 'd' */
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
 * @param open - 是否打开
 * @param verdict - 触发抽屉的榜单行结论
 * @param interval - K 线周期（拉 OHLC 用），缺省 'd'
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
              interval={interval}
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
