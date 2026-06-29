// 逐策略标定抽屉：按策略类型画像并给建议档位，一键回填该策略。
// 条件(跳空)策略→高/低/平开分场景；固定档→全窗一对档；波动/趋势→全窗仅参考。
import React, { useEffect, useMemo, useState } from 'react'
import {
  Drawer, Button, Space, Typography, Table, Alert, Tag, Statistic, Row, Col, message,
} from 'antd'
import { ExperimentOutlined, PlayCircleOutlined } from '@ant-design/icons'
import { useMutation } from '@tanstack/react-query'
import dayjs, { type Dayjs } from 'dayjs'

import { t0Service } from '../../api/t0'
import DateRangeSelector from '../../components/DateRangeSelector'
import { applyFixedSuggestion, applyGapSegments } from './calibrationApply'
import type { TickPolicyCfg, T0BandEdgeRow, T0Profile } from '../../types/t0'

const { Text, Paragraph } = Typography

/** 最少样本天数：低于此值的场景标"样本不足"、不回填。 */
const MIN_DAYS = 5

/** 带符号着色的"分"数值。 */
const fen = (v: number): React.ReactNode => (
  <span style={{ color: v > 0 ? '#3f8600' : v < 0 ? '#cf1322' : undefined }}>
    {v > 0 ? '+' : ''}{v.toFixed(2)}
  </span>
)

/** 偏离-回归边际曲线的紧凑列（偏离 / 逐腿均益+贡献）。 */
const bandCols = [
  { title: '偏离(分)', dataIndex: 'x_fen', key: 'x' },
  { title: '卖均益', dataIndex: 'sell_edge_fen', key: 'se', render: (v: number) => fen(v) },
  { title: '卖贡献', key: 'sc', render: (_: unknown, r: T0BandEdgeRow) => fen(r.sell_fill * r.sell_edge_fen) },
  { title: '买均益', dataIndex: 'buy_edge_fen', key: 'be', render: (v: number) => fen(v) },
  { title: '买贡献', key: 'bc', render: (_: unknown, r: T0BandEdgeRow) => fen(r.buy_fill * r.buy_edge_fen) },
]

/** 单份画像的建议档 + 紧凑曲线表（标定窗内）。 */
const ProfileBlock: React.FC<{ profile: T0Profile }> = ({ profile }) => (
  <>
    <Text>建议 卖 <b>{profile.suggested_sell_tick.toFixed(2)}</b> 元 / 买 <b>{profile.suggested_buy_tick.toFixed(2)}</b> 元</Text>
    <Table<T0BandEdgeRow> size="small" rowKey="x_fen" pagination={false} columns={bandCols}
      dataSource={profile.rows} style={{ marginTop: 6 }}
      rowClassName={(r) => (r.x_fen === Math.round(profile.suggested_buy_tick * 100)
        || r.x_fen === Math.round(profile.suggested_sell_tick * 100)) ? 'ant-table-row-selected' : ''} />
  </>
)

export interface CalibrationDrawerProps {
  /** 是否打开 */
  open: boolean
  /** 被标定的策略；null 时不渲染内容 */
  policy: TickPolicyCfg | null
  /** 当前回测评估窗（用于默认标定窗 + 重叠告警） */
  evalWindow: [Dayjs, Dayjs]
  /** 标的 */
  symbol: string
  /** 单边佣金率 */
  commissionRate: number
  /** 卖出印花税率 */
  stampDuty: number
  /** 画像档位网格上限（分） */
  xMaxFen: number
  /** 应用建议后回传更新后的策略 */
  onApply: (next: TickPolicyCfg) => void
  /** 关闭抽屉 */
  onClose: () => void
}

/**
 * 逐策略标定抽屉。
 *
 * 标定窗默认取评估窗之前的一段（walk-forward）；与评估窗重叠时显著告警。建议为理想撮合上限，
 * 须经回测 FillPolicy 网格验证。条件(跳空)策略按高/低/平开分场景，逐规则给建议并可一键回填。
 */
const CalibrationDrawer: React.FC<CalibrationDrawerProps> = ({
  open, policy, evalWindow, symbol, commissionRate, stampDuty, xMaxFen, onApply, onClose,
}) => {
  const isCond = policy?.kind === 'conditional'
  const isParamOnly = policy?.kind === 'vol_scaled' || policy?.kind === 'trend_tilt'

  // 标定窗默认：评估窗起点前一年 → 评估窗起点前一日（样本外）
  const [calibRange, setCalibRange] = useState<[Dayjs, Dayjs]>(
    [evalWindow[0].subtract(1, 'year'), evalWindow[0].subtract(1, 'day')])

  // 条件策略：分场景阈值取该策略首个跳空规则的阈值绝对值
  const gapThresh = useMemo(() => {
    if (policy?.kind !== 'conditional') return 0.003
    const gr = policy.rules.find((r) => r.lhs === 'gap')
    return gr ? Math.abs(gr.threshold) || 0.003 : 0.003
  }, [policy])

  const profMut = useMutation({
    mutationFn: t0Service.profile,
    onError: (e: unknown) => message.error((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || '画像失败'),
  })
  const segMut = useMutation({
    mutationFn: t0Service.profileSegmented,
    onError: (e: unknown) => message.error((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || '分场景画像失败'),
  })

  // 每次打开 / 切换被标定策略：重置标定窗为默认(评估窗之前) + 清空上次画像结果，
  // 避免抽屉常驻挂载导致串用旧建议（旧窗口/旧策略/旧 gap_thresh 的结果误用到当前策略）。
  useEffect(() => {
    if (!open) return
    setCalibRange([evalWindow[0].subtract(1, 'year'), evalWindow[0].subtract(1, 'day')])
    profMut.reset()
    segMut.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, evalWindow, policy?.kind, policy?.label])

  const onRun = () => {
    const base = {
      symbol, start: calibRange[0].format('YYYY-MM-DD'), end: calibRange[1].format('YYYY-MM-DD'),
      x_max_fen: xMaxFen, commission_rate: commissionRate, stamp_duty: stampDuty,
    }
    if (isCond) segMut.mutate({ ...base, gap_thresh: gapThresh })
    else profMut.mutate(base)
  }

  // walk-forward 违例：标定窗末日不早于评估窗起点
  const overlap = !calibRange[1].isBefore(evalWindow[0])

  const canApply = isCond ? !!segMut.data : (policy?.kind === 'fixed' && !!profMut.data)
  const onApplyClick = () => {
    if (!policy) return
    if (isCond && segMut.data) onApply(applyGapSegments(policy, segMut.data, MIN_DAYS))
    else if (policy.kind === 'fixed' && profMut.data) onApply(applyFixedSuggestion(policy, profMut.data))
    onClose()
  }

  return (
    <Drawer width={720} open={open} onClose={onClose}
      title={<span><ExperimentOutlined /> 标定：{policy?.label ?? ''}（{policy?.kind}）</span>}
      extra={
        <Space>
          <Button onClick={onClose}>关闭</Button>
          <Button type="primary" disabled={!canApply} onClick={onApplyClick}>应用到本策略</Button>
        </Space>
      }>
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <div>
          <Text type="secondary">标定窗（建议早于评估窗，避免样本内调参）</Text>
          <DateRangeSelector value={calibRange}
            onChange={(v) => v && v[0] && v[1] && setCalibRange([v[0], v[1]])} />
        </div>
        {overlap && (
          <Alert type="warning" showIcon message="标定窗与评估窗重叠"
            description="在同一段数据上标定档位又回测=样本内过拟合风险；建议把标定窗整体移到评估窗之前。" />
        )}
        <Alert type="info" showIcon message="画像是理想撮合上限，不是定论"
          description="建议档位假设触价即成交；最终须看回测的成交敏感性区间（FillPolicy 网格）是否仍为正。" />

        <Button type="primary" icon={<PlayCircleOutlined />} loading={profMut.isPending || segMut.isPending}
          onClick={onRun}>统计画像</Button>

        {isParamOnly && (profMut.data) && (
          <Alert type="warning" showIcon message="仅供参考"
            description="波动缩放/趋势倾斜策略的 k/倾斜量 最优画像测不出来，下表只是该窗口的偏离-回归结构参考，请据此手动设定参数。" />
        )}

        {/* 固定 / 波动 / 趋势：全窗画像 */}
        {!isCond && profMut.data && <ProfileBlock profile={profMut.data} />}

        {/* 条件(跳空)：高/低/平开三场景 */}
        {isCond && segMut.data && (
          <Text type="secondary" style={{ fontSize: 12 }}>
            按 gap ±{(gapThresh * 100).toFixed(2)}% 切分高/低/平开（取首个跳空规则的阈值；若高、低开规则阈值不同，两侧统一按此切分，可把两条规则阈值设成相同量级以对齐）。
          </Text>
        )}
        {isCond && segMut.data && segMut.data.segments.map((seg) => (
          <div key={seg.regime}>
            <Space>
              <Text strong>{seg.label}</Text>
              <Tag color={seg.n_days < MIN_DAYS ? 'red' : 'blue'}>样本 {seg.n_days} 天</Tag>
              {seg.n_days < MIN_DAYS && <Text type="danger">样本不足，建议不可靠（应用时跳过此场景）</Text>}
            </Space>
            <ProfileBlock profile={seg.profile} />
          </div>
        ))}

        {isCond && segMut.data && (
          <Paragraph type="secondary" style={{ fontSize: 12, marginBottom: 0 }}>
            「应用到本策略」将把：高开建议→首个 gap&gt; 规则、低开建议→首个 gap&lt; 规则、平开建议→默认档；样本不足的场景跳过。
          </Paragraph>
        )}
      </Space>
    </Drawer>
  )
}

export default CalibrationDrawer
