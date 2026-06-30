// 半仓做 T 回测页面 — 同步运行；含标的画像（偏离-回归边际曲线）+ 全部可配置参数
import React, { useState, useMemo, useCallback } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, InputNumber,
  Select, Alert, Table, Statistic, Tag, Tooltip, message, Divider, Segmented,
} from 'antd'
import {
  PlayCircleOutlined, ThunderboltOutlined, InfoCircleOutlined,
  PlusOutlined, MinusCircleOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery } from '@tanstack/react-query'
import dayjs, { type Dayjs } from 'dayjs'

import { t0Service } from '../../api/t0'
import { alphaService } from '../../api/alpha'
import { useAvailableSymbols } from '../../hooks/useAvailableSymbols'
import DateRangeSelector from '../../components/DateRangeSelector'
import TickPolicyEditor from './TickPolicyEditor'
import CalibrationDrawer from './CalibrationDrawer'
import type {
  T0FillCfg, T0FillSensitivityRow, T0PeriodRow, T0Report, TickPolicyCfg,
} from '../../types/t0'

/** 成交假设去重键。 */
export const fillKey = (f: T0FillCfg): string => `${f.penetration}|${f.ratio}`

/** 校验档位策略：label 非空且唯一、条件策略至少一条规则、signal 规则必选信号名。返回错误信息或 null。 */
export const validatePolicies = (policies: TickPolicyCfg[]): string | null => {
  const labels = policies.map((p) => p.label.trim())
  if (labels.some((l) => !l)) return '每个档位策略都需要名称'
  if (new Set(labels).size !== labels.length) return '档位策略名称必须唯一'
  for (const p of policies) {
    if (p.kind === 'conditional') {
      if (p.rules.length === 0) return `策略「${p.label}」至少需要一条规则`
      if (p.rules.some((r) => r.lhs === 'signal' && !r.signal_name)) return `策略「${p.label}」有信号规则未选信号名`
    }
  }
  return null
}

const { Text, Paragraph } = Typography

/** 单条成交假设的前端编辑态：穿越（分）+ 成交率（%）。 */
interface FillEdit { penFen: number; ratioPct: number }

/** 把成交假设格式化为可读标签（穿越与部分成交可叠加，二者皆无则为理想撮合）。 */
export const fillLabel = (f: T0FillCfg): string => {
  const parts: string[] = []
  if (f.penetration > 0) parts.push(`穿越 ${Math.round(f.penetration * 100)} 分`)
  if (f.ratio < 1) parts.push(`部分成交 ${Math.round(f.ratio * 100)}%`)
  return parts.length ? parts.join(' + ') : '理想撮合（触价即成交）'
}

/** 收益百分比着色文本。 */
const pct = (v: number, digits = 1): React.ReactNode => {
  const color = v > 0 ? '#3f8600' : v < 0 ? '#cf1322' : undefined
  return <span style={{ color }}>{v > 0 ? '+' : ''}{(v * 100).toFixed(digits)}%</span>
}

/** 由逐年行算"累计"行：各列全程复利 ∏(1+r)−1，超额取累计策略−累计基准（year=undefined 标记累计行）。 */
const cumRowFor = (yearly: T0PeriodRow[]): T0PeriodRow => {
  const cmp = (k: 'strat' | 'bh' | 'half_bh') => yearly.reduce((a, r) => a * (1 + r[k]), 1) - 1
  const s = cmp('strat'), b = cmp('bh'), h = cmp('half_bh')
  return { year: undefined, strat: s, bh: b, half_bh: h, excess_vs_bh: s - b, excess_vs_half_bh: s - h }
}

/**
 * 半仓做 T 回测页面。
 *
 * 三块：① 标的画像——按偏离档位逐买卖腿统计每笔边际收益（理想撮合）+ 建议档位；
 * ② 回测配置（全部参数可调：档位/摆动/成本/成交假设网格）；③ 结果——成交敏感性区间、
 * 逐年超额、命中分布。画像负责"算出该用什么档位"，成交网格负责"裁决到底值不值"。
 */
const T0Backtest: React.FC = () => {
  const [symbol, setSymbol] = useState('000415.SZSE')
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs]>([dayjs('2023-01-01'), dayjs('2025-12-31')])
  const [tickPolicies, setTickPolicies] = useState<TickPolicyCfg[]>([
    { kind: 'fixed', label: '固定2分', sell_tick: 0.02, buy_tick: 0.02 },
  ])
  const [swingFrac, setSwingFrac] = useState(1.0)
  const [baseWeight, setBaseWeight] = useState(0.5)
  const [capital, setCapital] = useState(1_000_000)
  const [commissionRate, setCommissionRate] = useState(0.0003)
  const [stampDuty, setStampDuty] = useState(0.0005)
  const [fillGrid, setFillGrid] = useState<FillEdit[]>([
    { penFen: 0, ratioPct: 100 },
    { penFen: 1, ratioPct: 100 },
    { penFen: 0, ratioPct: 50 },
  ])
  const [xMaxFen, setXMaxFen] = useState(15)
  const [selFill, setSelFill] = useState(0)     // 逐年/命中分布查看哪个成交假设
  const [selPolicy, setSelPolicy] = useState(0) // 逐年/命中分布查看哪个档位策略
  const [calibIdx, setCalibIdx] = useState<number | null>(null) // 正在标定的策略下标（开抽屉）

  // 可用信号名（供条件规则的 signal 左值选择）
  const { data: signalNames = [] } = useQuery({
    queryKey: ['t0-signals'],
    queryFn: () => t0Service.listSignals(),
  })

  // 复用 CNN 训练同款"按现有数据快速选择"
  const { data: resources } = useQuery({
    queryKey: ['alpha-data-resources'],
    queryFn: () => alphaService.getDataResources(),
  })
  const availabilityMap = useAvailableSymbols(resources)
  const symbolOptions = useMemo(
    () =>
      [...availabilityMap.entries()]
        .filter(([, meta]) => meta.intervals.has('1m'))
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([sym, meta]) => ({ value: sym, range: meta.intervalRanges['1m'] ?? { start: meta.start, end: meta.end } })),
    [availabilityMap],
  )
  const localRange = useMemo(() => {
    const key = symbol.replace(/\.$/, '').toLowerCase()
    const found = symbolOptions.find((o) => o.value.replace(/\.$/, '').toLowerCase() === key)
    return found ? { start: found.range.start.slice(0, 10), end: found.range.end.slice(0, 10) } : null
  }, [symbolOptions, symbol])
  const onSymbolChange = useCallback((val: string) => {
    setSymbol(val)
    const found = symbolOptions.find((o) => o.value === val)
    if (found) setDateRange([dayjs(found.range.start.slice(0, 10)), dayjs(found.range.end.slice(0, 10))])
  }, [symbolOptions])

  // —— 成交假设网格编辑 ——
  const updateFill = (i: number, key: keyof FillEdit, v: number) =>
    setFillGrid((g) => g.map((f, idx) => (idx === i ? { ...f, [key]: v } : f)))
  const addFill = () => setFillGrid((g) => [...g, { penFen: 0, ratioPct: 100 }])
  const removeFill = (i: number) => setFillGrid((g) => g.filter((_, idx) => idx !== i))

  // —— 回测 ——
  const mut = useMutation({
    mutationFn: t0Service.runBacktest,
    onError: (e: unknown) => message.error(
      (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || '做T回测失败'),
  })
  const report: T0Report | undefined = mut.data
  const onRun = useCallback(() => {
    const err = validatePolicies(tickPolicies)
    if (err) { message.error(err); return }
    const firstFixed = tickPolicies.find((p) => p.kind === 'fixed') as
      | Extract<TickPolicyCfg, { kind: 'fixed' }> | undefined
    // label 统一 trim 后再发（与 validatePolicies/标红口径一致，避免空白差异）
    const policies = tickPolicies.map((p) => ({ ...p, label: p.label.trim() }))
    // 成交假设去重（同 穿越×成交率 只跑一次，避免重复卡片/无谓回测）
    const seenFill = new Set<string>()
    const dedupGrid = fillGrid.filter((f) => {
      const k = `${f.penFen}|${f.ratioPct}`
      if (seenFill.has(k)) return false
      seenFill.add(k)
      return true
    })
    mut.mutate({
      symbol: symbol.trim(),
      start: dateRange[0].format('YYYY-MM-DD'),
      end: dateRange[1].format('YYYY-MM-DD'),
      // sell_tick/buy_tick 仅为满足后端必填字段（有 tick_policies 时后端忽略它们）
      sell_tick: firstFixed?.sell_tick ?? 0.02, buy_tick: firstFixed?.buy_tick ?? 0.02,
      swing_frac: swingFrac, base_weight: baseWeight,
      capital, commission_rate: commissionRate, stamp_duty: stampDuty,
      fill_grid: dedupGrid.map((f) => ({ penetration: f.penFen / 100, ratio: f.ratioPct / 100 })),
      tick_policies: policies,
    })
  }, [mut, symbol, dateRange, tickPolicies, swingFrac, baseWeight, capital, commissionRate, stampDuty, fillGrid])

  // 结果 = 档位策略 × 成交假设；按两维分别切换查看
  const results = report?.results ?? []
  const policyLabels = useMemo(() => [...new Set(results.map((r) => r.tick_label))], [results])
  const fillCfgs = useMemo(() => {
    const seen = new Map<string, T0FillCfg>()
    results.forEach((r) => { if (!seen.has(fillKey(r.fill))) seen.set(fillKey(r.fill), r.fill) })
    return [...seen.values()]
  }, [results])
  const selLabel = policyLabels[Math.min(selPolicy, Math.max(0, policyLabels.length - 1))]
  const selFillCfg = fillCfgs[Math.min(selFill, Math.max(0, fillCfgs.length - 1))]
  const sel = results.find((r) => r.tick_label === selLabel && fillKey(r.fill) === fillKey(selFillCfg ?? r.fill))
  const hit = sel?.hit_dist
  // 收益区间只在「当前所选策略」内跨成交假设取（不同策略不混在一起比）
  const spread = useMemo(() => {
    const rows = report?.fill_sensitivity?.filter((r) => r.tick_label === selLabel) ?? []
    if (!rows.length) return null
    const rets = rows.map((r) => r.total_return)
    return { hi: Math.max(...rets), lo: Math.min(...rets) }
  }, [report, selLabel])

  const yearCols = [
    { title: '年份', dataIndex: 'year', key: 'year', render: (v: number | undefined) => (v == null ? <b>累计</b> : v) },
    { title: '策略(净)', dataIndex: 'strat', key: 'strat', render: (v: number) => pct(v) },
    { title: '满仓持有', dataIndex: 'bh', key: 'bh', render: (v: number) => pct(v) },
    { title: '半仓持有', dataIndex: 'half_bh', key: 'half_bh', render: (v: number) => pct(v) },
    { title: '超额(vs满仓)', dataIndex: 'excess_vs_bh', key: 'e1', render: (v: number) => pct(v) },
    { title: '超额(vs半仓)', dataIndex: 'excess_vs_half_bh', key: 'e2', render: (v: number) => pct(v) },
  ]
  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Alert type="info" showIcon
        message="半仓做 T：每天按开盘价挂 ±档位限价单、日内做 T、收盘维持半仓"
        description="为每个档位策略点「画像并建议」按其类型分场景标定档位（条件策略按高/低/平开分别画像、逐规则建议）；再跑回测看「成交敏感性区间」——盈亏几乎全取决于能否在 ±档位真成交，理想撮合是上限、穿越/部分成交更贴近实盘，区间越宽越不可复制。" />

      {/* —— 回测配置（全部参数可调） —— */}
      <Card size="small" title="回测配置">
        <Row gutter={[16, 12]}>
          <Col span={9}>
            <Text type="secondary">标的（仅列出有 1m 数据的）</Text>
            <Select showSearch style={{ width: '100%' }} value={symbol || undefined} placeholder="选择标的"
              onChange={onSymbolChange} options={symbolOptions.map((o) => ({ value: o.value, label: o.value }))}
              filterOption={(input, opt) => String(opt?.value ?? '').toLowerCase().includes(input.toLowerCase())}
              notFoundContent={symbolOptions.length ? '无匹配' : '加载中…'} />
          </Col>
          <Col span={5}>
            <Text type="secondary">初始资金</Text>
            <InputNumber style={{ width: '100%' }} value={capital} min={100000} step={100000}
              onChange={(v) => setCapital(v ?? 1_000_000)} />
          </Col>
          <Col span={5}>
            <Text type="secondary">佣金率（单边）</Text>
            <InputNumber style={{ width: '100%' }} value={commissionRate} min={0} step={0.0001} precision={4}
              onChange={(v) => setCommissionRate(v ?? 0.0003)} />
          </Col>
          <Col span={5}>
            <Text type="secondary">印花税（卖出）</Text>
            <InputNumber style={{ width: '100%' }} value={stampDuty} min={0} step={0.0001} precision={4}
              onChange={(v) => setStampDuty(v ?? 0.0005)} />
          </Col>
          <Col span={24}>
            <Text type="secondary">评估区间（按该标的本地可用数据快速选择）</Text>
            <DateRangeSelector value={dateRange}
              onChange={(v) => v && v[0] && v[1] && setDateRange([v[0], v[1]])} localRange={localRange} />
          </Col>
        </Row>

        <Row gutter={[16, 12]} style={{ marginTop: 4 }}>
          <Col span={8}>
            <Text type="secondary">摆动幅度<Tooltip title="做T一手占半仓比例，1=全半仓摆动"><InfoCircleOutlined /></Tooltip></Text>
            <InputNumber style={{ width: '100%' }} value={swingFrac} min={0.1} max={1} step={0.1}
              onChange={(v) => setSwingFrac(v ?? 1)} />
          </Col>
          <Col span={8}>
            <Text type="secondary">半仓锚权重</Text>
            <InputNumber style={{ width: '100%' }} value={baseWeight} min={0.1} max={0.9} step={0.1}
              onChange={(v) => setBaseWeight(v ?? 0.5)} />
          </Col>
          <Col span={8}>
            <Text type="secondary">画像档位上限(分)</Text>
            <InputNumber style={{ width: '100%' }} value={xMaxFen} min={2} max={50} step={1}
              onChange={(v) => setXMaxFen(v ?? 15)} />
          </Col>
        </Row>

        <Divider style={{ margin: '12px 0' }} orientation="left" plain>
          <Text type="secondary" style={{ fontSize: 13 }}>
            档位策略 <Tooltip title="定义一个或多个命名档位策略，一次回测出「策略 × 成交假设」对比。固定/波动缩放/趋势倾斜为数值参数；条件规则可按跳空/波动/动量/信号挂不同档位。"><InfoCircleOutlined /></Tooltip>
          </Text>
        </Divider>
        <TickPolicyEditor value={tickPolicies} onChange={setTickPolicies} signalNames={signalNames}
          onCalibrate={setCalibIdx} />

        <Divider style={{ margin: '12px 0' }} orientation="left" plain>
          <Text type="secondary" style={{ fontSize: 13 }}>
            成交假设网格 <Tooltip title="穿越=要价格穿过挂单价 N 分才成交（碰一下不算）；成交率=单根触价仅成交的比例。多组用于框出收益区间。"><InfoCircleOutlined /></Tooltip>
          </Text>
        </Divider>
        <Space direction="vertical" style={{ width: '100%' }}>
          {fillGrid.map((f, i) => (
            <Space key={i}>
              <InputNumber addonBefore="穿越" addonAfter="分" min={0} step={1} value={f.penFen}
                onChange={(v) => updateFill(i, 'penFen', v ?? 0)} style={{ width: 150 }} />
              <InputNumber addonBefore="成交率" addonAfter="%" min={10} max={100} step={10} value={f.ratioPct}
                onChange={(v) => updateFill(i, 'ratioPct', v ?? 100)} style={{ width: 170 }} />
              <Tag color={f.penFen > 0 ? 'volcano' : f.ratioPct < 100 ? 'gold' : 'green'}>
                {fillLabel({ penetration: f.penFen / 100, ratio: f.ratioPct / 100 })}
              </Tag>
              <Button danger type="text" icon={<MinusCircleOutlined />} disabled={fillGrid.length <= 1}
                onClick={() => removeFill(i)} />
            </Space>
          ))}
          <Space>
            <Button size="small" icon={<PlusOutlined />} onClick={addFill}>添加成交假设</Button>
            <Button type="primary" icon={<PlayCircleOutlined />} loading={mut.isPending} onClick={onRun}>运行回测</Button>
          </Space>
        </Space>
      </Card>

      {report && (
        <>
          <Card size="small" title={<span><ThunderboltOutlined /> 成交敏感性区间（{report.symbol}，{report.eval_window[0]} ~ {report.eval_window[1]}）</span>}>
            <Row gutter={[16, 16]}>
              {report.fill_sensitivity.map((r: T0FillSensitivityRow, i) => (
                <Col span={8} key={i}>
                  <Card size="small" bordered>
                    {policyLabels.length > 1 && <div><Text strong style={{ fontSize: 12 }}>{r.tick_label}</Text></div>}
                    <Tag color={r.fill.penetration > 0 ? 'volcano' : r.fill.ratio < 1 ? 'gold' : 'green'}>{fillLabel(r.fill)}</Tag>
                    <Statistic value={r.total_return * 100} precision={1} suffix="%"
                      valueStyle={{ color: r.total_return > 0 ? '#3f8600' : '#cf1322' }} />
                    <Text type="secondary">Sharpe {(r.sharpe ?? 0).toFixed(2)} ｜ 回撤 {(r.max_drawdown * 100).toFixed(0)}%</Text>
                  </Card>
                </Col>
              ))}
            </Row>
            {spread && (
              <Alert style={{ marginTop: 12 }} showIcon
                type={spread.hi - spread.lo > 0.3 ? 'warning' : 'success'}
                message={`收益区间宽度 ${((spread.hi - spread.lo) * 100).toFixed(0)} 个百分点（理想 ${(spread.hi * 100).toFixed(0)}% ~ 最差 ${(spread.lo * 100).toFixed(0)}%）`}
                description={spread.hi - spread.lo > 0.3
                  ? '区间很宽：盈亏高度依赖成交质量，实盘极可能拿不到理想撮合那端，谨慎对待。'
                  : '区间较窄：对成交假设相对不敏感。'} />
            )}
          </Card>

          {sel && (
            <Card size="small" title="逐年收益与超额"
              extra={(policyLabels.length > 1 || fillCfgs.length > 1) ? (
                <Space size="small" wrap>
                  {policyLabels.length > 1 && (
                    <>
                      <Text type="secondary" style={{ fontSize: 12 }}>策略</Text>
                      <Segmented size="small" value={Math.min(selPolicy, policyLabels.length - 1)}
                        onChange={(v) => setSelPolicy(Number(v))}
                        options={policyLabels.map((l, i) => ({ label: l, value: i }))} />
                    </>
                  )}
                  {fillCfgs.length > 1 && (
                    <>
                      <Text type="secondary" style={{ fontSize: 12 }}>成交假设</Text>
                      <Segmented size="small" value={Math.min(selFill, fillCfgs.length - 1)}
                        onChange={(v) => setSelFill(Number(v))}
                        options={fillCfgs.map((f, i) => ({ label: fillLabel(f), value: i }))} />
                    </>
                  )}
                </Space>
              ) : null}>
              <Table<T0PeriodRow> size="small" pagination={false} columns={yearCols}
                rowKey={(r) => (r.year == null ? 'cum' : String(r.year))}
                rowClassName={(r) => (r.year == null ? 'ant-table-row-selected' : '')}
                dataSource={[...sel.yearly, cumRowFor(sel.yearly)]} />
              <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
                当前口径：<b>{sel.tick_label}</b> ｜ {fillLabel(sel.fill)}
                {(policyLabels.length > 1 || fillCfgs.length > 1) ? '（切右上角看其它策略/成交假设）' : ''}。
              </Paragraph>
            </Card>
          )}

          {hit && (
            <Card size="small" title={`命中分布（开盘 ±档位 是否被触及）`}>
              <Space size="large">
                <Statistic title="两边都触（对敲）" value={(hit.both * 100).toFixed(0)} suffix="%" />
                <Statistic title="仅触卖" value={(hit.onlyS * 100).toFixed(0)} suffix="%" />
                <Statistic title="仅触买" value={(hit.onlyB * 100).toFixed(0)} suffix="%" />
                <Statistic title="都不触" value={(hit.none * 100).toFixed(0)} suffix="%" />
              </Space>
              <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
                命中分布只由行情与档位决定（各成交假设相同）；{fillLabel(sel.fill)} 口径年化换手 {sel.turnover_annual}x。「两边都触」是对敲落袋主力，「仅触一边」在趋势日易亏。
              </Paragraph>
            </Card>
          )}
        </>
      )}

      <CalibrationDrawer
        open={calibIdx !== null}
        policy={calibIdx !== null ? (tickPolicies[calibIdx] ?? null) : null}
        evalWindow={dateRange}
        symbol={symbol.trim()}
        commissionRate={commissionRate}
        stampDuty={stampDuty}
        xMaxFen={xMaxFen}
        localRange={localRange}
        onApply={(next) => setTickPolicies((ps) => ps.map((p, i) => (i === calibIdx ? next : p)))}
        onClose={() => setCalibIdx(null)}
      />
    </Space>
  )
}

export default T0Backtest
