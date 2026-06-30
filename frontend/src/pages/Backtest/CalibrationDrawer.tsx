// 逐策略标定抽屉：按策略类型画像并给建议档位，一键回填该策略。
// 条件(跳空)策略→高/低/平开分场景；固定档→全窗一对档；波动/趋势→全窗仅参考。
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Drawer, Button, Space, Typography, Table, Alert, Tag, Statistic, Row, Col, message, Tooltip,
} from 'antd'
import { ExperimentOutlined, PlayCircleOutlined, InfoCircleOutlined } from '@ant-design/icons'
import { useMutation } from '@tanstack/react-query'
import dayjs, { type Dayjs } from 'dayjs'

import { t0Service } from '../../api/t0'
import DateRangeSelector, { type LocalDateRange } from '../../components/DateRangeSelector'
import KLineChart from '../../components/charts/KLineChart'
import { applyFixedSuggestion, applyGapSegments } from './calibrationApply'
import { buildLegChart, fixedSuggestionResolver, gapSuggestionResolver, noMarkerResolver } from './t0LegMarkers'
import type { OHLCBar } from '../../components/charts/types'
import type { TickPolicyCfg, T0BandEdgeRow, T0Profile } from '../../types/t0'

const { Text, Paragraph } = Typography

/** 最少样本天数：低于此值的场景标"样本不足"、不回填。 */
const MIN_DAYS = 5

/**
 * 标定窗默认值：评估窗起点前一年 → 前一日（walk-forward），并夹进本地数据可用区间。
 *
 * @param evalStart - 评估窗起点
 * @param localRange - 本地数据可用区间（ISO 字符串）；给定时把默认窗夹进该范围，避免默认就落在无数据区
 * @returns 默认标定窗 `[start, end]`
 */
export function defaultCalibWindow(evalStart: Dayjs, localRange?: LocalDateRange | null): [Dayjs, Dayjs] {
  let s = evalStart.subtract(1, 'year')
  let e = evalStart.subtract(1, 'day')
  if (localRange) {
    const ls = dayjs(localRange.start)
    const le = dayjs(localRange.end)
    if (s.isBefore(ls)) s = ls
    if (e.isAfter(le)) e = le
  }
  return [s, e]
}

/** 带符号着色的"分"数值。 */
const fen = (v: number): React.ReactNode => (
  <span style={{ color: v > 0 ? '#3f8600' : v < 0 ? '#cf1322' : undefined }}>
    {v > 0 ? '+' : ''}{v.toFixed(2)}
  </span>
)

/** 成交率百分比单元。 */
const pctCell = (v: number): string => `${(v * 100).toFixed(0)}%`

/** 带"计算方法"hover 提示的表头：hover 列名上的 ⓘ 看该字段怎么算的。 */
const hdr = (label: string, tip: string): React.ReactNode => (
  <Tooltip title={tip}>
    <span>{label} <InfoCircleOutlined style={{ color: '#999' }} /></span>
  </Tooltip>
)

/**
 * 把抽屉宽度夹到 `[min, innerWidth − margin]`，避免拖太窄/超出视口。
 *
 * @param raw - 拖拽算出的原始宽度（px）
 * @param innerWidth - 视口宽度（px）
 * @param min - 最小宽度，默认 480
 * @param margin - 距视口左缘的最小留白，默认 80
 * @returns 夹取后的宽度（px）
 */
export function clampDrawerWidth(raw: number, innerWidth: number, min = 480, margin = 80): number {
  // 视口边界优先：窗口窄于 min+margin 时，宁可比 min 还窄也不溢出视口
  return Math.min(Math.max(raw, min), Math.max(0, innerWidth - margin))
}

/** 偏离-回归边际曲线全字段表（表头 hover 看计算方法）。 */
const bandCols = [
  { title: hdr('偏离(分)', '挂单价相对开盘价的偏离档位；1 分 = 0.01 元。x 越大挂得越远、越难成交。'),
    dataIndex: 'x_fen', key: 'x' },
  { title: hdr('卖成交率', '卖腿可成交的天数占比 = P(当日最高 ≥ 开盘 + x)。'),
    dataIndex: 'sell_fill', key: 'sf', render: (v: number) => pctCell(v) },
  { title: hdr('卖均益(分)', '卖腿每笔净于成本的边际收益 = mean(开+x − 收 | 高≥开+x) − 往返成本，再 ×100 转分。>0 表示触价后倾向回落、做T有底层 edge。'),
    dataIndex: 'sell_edge_fen', key: 'se', render: (v: number) => fen(v) },
  { title: hdr('卖贡献(分/天)', '日均贡献 = 卖成交率 × 卖均益（没成交的日子记 0）。建议卖档取本列在「成交率 > 10%」的行中的峰值；若无任一档 > 10%，回退为最小档（故高亮行未必是肉眼峰值）。'),
    key: 'sc', render: (_: unknown, r: T0BandEdgeRow) => fen(r.sell_fill * r.sell_edge_fen) },
  { title: hdr('买成交率', '买腿可成交的天数占比 = P(当日最低 ≤ 开盘 − x)。'),
    dataIndex: 'buy_fill', key: 'bf', render: (v: number) => pctCell(v) },
  { title: hdr('买均益(分)', '买腿每笔净于成本的边际收益 = mean(收 − (开−x) | 低≤开−x) − 往返成本，再 ×100 转分。'),
    dataIndex: 'buy_edge_fen', key: 'be', render: (v: number) => fen(v) },
  { title: hdr('买贡献(分/天)', '日均贡献 = 买成交率 × 买均益。建议买档取本列在「成交率 > 10%」的行中的峰值；若无任一档 > 10%，回退为最小档（故高亮行未必是肉眼峰值）。'),
    key: 'bc', render: (_: unknown, r: T0BandEdgeRow) => fen(r.buy_fill * r.buy_edge_fen) },
  { title: hdr('全日期望(分)', '全样本每日期望盈亏（含未触价日）×100 = both·2x + 仅卖·(x−Δ收) + 仅买·(x+Δ收)，Δ收=收−开。'),
    dataIndex: 'day_pnl_fen', key: 'dp', render: (v: number) => fen(v) },
]

/** 单份画像的建议档 + 紧凑曲线表（标定窗内）。memo：拖拽调宽时 profile 不变则跳过重渲，避免重画表格。 */
const ProfileBlock = React.memo(function ProfileBlock({ profile }: { profile: T0Profile }) {
  return (
    <>
      <Text>建议 卖 <b>{profile.suggested_sell_tick.toFixed(2)}</b> 元 / 买 <b>{profile.suggested_buy_tick.toFixed(2)}</b> 元</Text>
      <Table<T0BandEdgeRow> size="small" rowKey="x_fen" pagination={false} columns={bandCols}
        dataSource={profile.rows} style={{ marginTop: 6 }} scroll={{ x: 'max-content' }}
        rowClassName={(r) => (r.x_fen === Math.round(profile.suggested_buy_tick * 100)
          || r.x_fen === Math.round(profile.suggested_sell_tick * 100)) ? 'ant-table-row-selected' : ''} />
    </>
  )
})

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
  /** 本地数据可用区间（约束标定窗预设、夹取默认窗）；与评估区间同源 */
  localRange?: LocalDateRange | null
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
  open, policy, evalWindow, symbol, commissionRate, stampDuty, xMaxFen, localRange, onApply, onClose,
}) => {
  const isCond = policy?.kind === 'conditional'
  const isParamOnly = policy?.kind === 'vol_scaled' || policy?.kind === 'trend_tilt'

  // 抽屉宽度（可拖左缘调整）；常驻挂载 → 跨开关保留；初值按当前视口夹一次
  const [drawerWidth, setDrawerWidth] = useState(
    () => clampDrawerWidth(720, typeof window !== 'undefined' ? window.innerWidth : 1440))
  const handleRef = useRef<HTMLDivElement>(null)

  // 视口变化时重新夹宽，避免缩窗后抽屉溢出视口
  useEffect(() => {
    const onResize = () => setDrawerWidth((w) => clampDrawerWidth(w, window.innerWidth))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // 拖拽期间「绕过 React」直接写 DOM 宽度：零重渲、零图表/表格重建，rAF 每帧一次；松手才提交一次 state。
  const startResize = (e: React.MouseEvent) => {
    e.preventDefault()
    const wrapperEl = document.querySelector(
      '.t0-calib-drawer .ant-drawer-content-wrapper') as HTMLElement | null
    if (wrapperEl) wrapperEl.style.transition = 'none'   // 关掉任何宽度过渡，跟手不滞后
    let raf = 0
    let next = drawerWidth
    function apply() {
      raf = 0
      if (wrapperEl) wrapperEl.style.width = `${next}px`
      if (handleRef.current) handleRef.current.style.right = `${next - 6}px`
    }
    function onMove(ev: MouseEvent) {
      if (ev.buttons === 0) { stop(); return }   // 漏接 mouseup（如窗外松手）→ 自终止
      next = clampDrawerWidth(window.innerWidth - ev.clientX, window.innerWidth)
      if (!raf) raf = requestAnimationFrame(apply)
    }
    function stop() {
      if (raf) cancelAnimationFrame(raf)
      raf = 0
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', stop)
      window.removeEventListener('blur', stop)
      document.body.style.userSelect = ''
      if (wrapperEl) wrapperEl.style.transition = ''   // 还原过渡（开合动画）
      setDrawerWidth(next)   // 仅此一次提交 → 受控 width 与 DOM 同步
    }
    document.body.style.userSelect = 'none'   // 拖拽时禁选中，体验更顺
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', stop)
    window.addEventListener('blur', stop)      // 窗口失焦（含窗外松手）兜底终止
  }

  // 标定窗默认：评估窗起点前一年 → 前一日（样本外），并夹进本地数据可用区间
  const [calibRange, setCalibRange] = useState<[Dayjs, Dayjs]>(
    () => defaultCalibWindow(evalWindow[0], localRange))

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
    setCalibRange(defaultCalibWindow(evalWindow[0], localRange))
    profMut.reset()
    segMut.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, evalWindow, policy?.kind, policy?.label, localRange?.start, localRange?.end])

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

  // K 线买卖腿：用画像「建议档」标在标定窗 K 线上（与上方表格建议一致；hover 看每根明细）。
  // 条件策略按当日 gap 所属高/低/平开场景取该场景建议；波动/趋势档位动态、不标腿。
  const legChart = useMemo(() => {
    if (isParamOnly) {
      return profMut.data?.bars ? buildLegChart(profMut.data.bars, noMarkerResolver) : null
    }
    if (isCond) {
      return segMut.data?.bars ? buildLegChart(segMut.data.bars, gapSuggestionResolver(segMut.data)) : null
    }
    return profMut.data?.bars ? buildLegChart(profMut.data.bars, fixedSuggestionResolver(profMut.data)) : null
  }, [isCond, isParamOnly, profMut.data, segMut.data])

  // 稳定 hover 明细格式化器：仅在 legChart（数据）变化时换引用；拖拽调宽期间不变 →
  // KLineChart 的 props 引用稳定，不会因换函数而重建 lightweight-charts 实例（拖拽更丝滑）。
  const legTooltip = useCallback(
    (b: OHLCBar) => legChart?.details.get(String(b.time)) ?? '',
    [legChart])

  const canApply = isCond ? !!segMut.data : (policy?.kind === 'fixed' && !!profMut.data)
  const onApplyClick = () => {
    if (!policy) return
    if (isCond && segMut.data) onApply(applyGapSegments(policy, segMut.data, MIN_DAYS))
    else if (policy.kind === 'fixed' && profMut.data) onApply(applyFixedSuggestion(policy, profMut.data))
    onClose()
  }

  return (
    <>
      {open && (
        <div ref={handleRef} aria-label="拖拽调整标定抽屉宽度" onMouseDown={startResize}
          style={{ position: 'fixed', top: 0, bottom: 0, right: drawerWidth - 6, width: 6,
            cursor: 'col-resize', zIndex: 1003, background: 'rgba(0,0,0,0.04)' }} />
      )}
      <Drawer rootClassName="t0-calib-drawer" width={drawerWidth} open={open} onClose={onClose}
        title={<span><ExperimentOutlined /> 标定：{policy?.label ?? ''}（{policy?.kind}）</span>}
      extra={
        <Space>
          <Button onClick={onClose}>关闭</Button>
          <Button type="primary" disabled={!canApply} onClick={onApplyClick}>应用到本策略</Button>
        </Space>
      }>
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <div>
          <Text type="secondary">标定窗（建议早于评估窗，避免样本内调参；按本地可用数据快速选择）</Text>
          <DateRangeSelector value={calibRange} localRange={localRange}
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

        {/* K 线 + 买卖腿标记（按当前配置档；hover 看每根明细） */}
        {legChart && (
          <div>
            <Text strong>K 线与买卖腿</Text>
            <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>
              按上方「建议档」标在标定窗 K 线上（▲买腿 / ▼卖腿，hover 看该根明细），与表格一致
              {isCond ? '；条件策略按每天所属高/低/平开场景取该场景建议档' : ''}
            </Text>
            {!legChart.supported && (
              <Text type="warning" style={{ fontSize: 12, display: 'block' }}>
                该策略档位按历史动态计算（波动/趋势），K 线只展示行情、暂不标买卖腿。
              </Text>
            )}
            <KLineChart bars={legChart.bars} markers={legChart.markers} showVolume={false} height={300}
              emptyText="无 K 线数据" tooltipFormatter={legTooltip} />
          </div>
        )}

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
    </>
  )
}

export default CalibrationDrawer
