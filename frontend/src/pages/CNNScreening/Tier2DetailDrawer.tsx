import React, { useEffect, useState } from 'react'
import { Drawer, Empty, Space, Spin, Typography } from 'antd'
import { useQuery } from '@tanstack/react-query'

import type { Tier2Verdict } from '../../types/screening'
import { screeningService } from '../../api/screening'
import BacktestResults from '../Backtest/BacktestResults'
import GateVerdictHeader from './GateVerdictHeader'
import FoldTable from './FoldTable'

const { Text } = Typography

/** Tier-2 折级详情抽屉的 Props。 */
interface Tier2DetailDrawerProps {
  /** 是否打开 */
  open: boolean
  /** 触发抽屉的榜单行结论；report_id 为空时不应被打开 */
  verdict: Tier2Verdict | null
  /** 关闭回调 */
  onClose: () => void
}

/** 第三波（净值/回撤曲线）的占位卡，提示该能力规划中。 */
function CurvePlaceholder() {
  return (
    <div
      style={{
        border: '1px dashed #d9d9d9',
        borderRadius: 6,
        padding: '18px 12px',
        textAlign: 'center',
        color: '#999',
        fontSize: 12,
      }}
    >
      折级 OOS 净值 / 回撤曲线（第三波规划中）
    </div>
  )
}

/**
 * Tier-2 折级详情抽屉：按 `verdict.report_id` 拉取 WF 报告，纵向堆叠展示
 * 门禁头部 → 折级总览表（点行联动）→ 选中折回测指标卡（复用 BacktestResults）→
 * 第三波净值曲线占位。
 *
 * 报告 404/加载失败/无折 时渲染空态占位，不崩溃；门禁头部始终以 `verdict` 渲染
 * （来自榜单行，无需等待报告）。
 *
 * @param open - 是否打开
 * @param verdict - 触发抽屉的榜单行结论
 * @param onClose - 关闭回调
 */
const Tier2DetailDrawer: React.FC<Tier2DetailDrawerProps> = ({ open, verdict, onClose }) => {
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

  const selected =
    report?.folds.find((f) => f.fold === selectedFold) ?? report?.folds[0] ?? null

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
            <CurvePlaceholder />
          </>
        )}
      </Space>
    )
  }

  return (
    <Drawer
      title={verdict ? `${verdict.vt_symbol} · Tier-2 折级详情` : 'Tier-2 折级详情'}
      width={720}
      open={open}
      onClose={onClose}
    >
      {renderBody()}
    </Drawer>
  )
}

export default Tier2DetailDrawer
