// 半仓做 T 回测页面 — 同步运行；含标的画像（偏离-回归边际曲线）+ 全部可配置参数
import React, { useState, useMemo, useCallback } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, InputNumber,
  Select, Alert, Table, Statistic, Tag, Tooltip, message, Divider,
} from 'antd'
import {
  PlayCircleOutlined, ThunderboltOutlined, InfoCircleOutlined,
  ExperimentOutlined, PlusOutlined, MinusCircleOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, Tooltip as RTooltip, Legend, CartesianGrid,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import dayjs, { type Dayjs } from 'dayjs'

import { t0Service } from '../../api/t0'
import { alphaService } from '../../api/alpha'
import { useAvailableSymbols } from '../../hooks/useAvailableSymbols'
import DateRangeSelector from '../../components/DateRangeSelector'
import type {
  T0BandEdgeRow, T0FillCfg, T0FillSensitivityRow, T0PeriodRow, T0Profile, T0Report,
} from '../../types/t0'

const { Text, Paragraph } = Typography

/** 单条成交假设的前端编辑态：穿越（分）+ 成交率（%）。 */
interface FillEdit { penFen: number; ratioPct: number }

/** 把成交假设格式化为可读标签。 */
const fillLabel = (f: T0FillCfg): string => {
  if (f.penetration > 0) return `穿越 ${Math.round(f.penetration * 100)} 分`
  if (f.ratio < 1) return `部分成交 ${Math.round(f.ratio * 100)}%`
  return '理想撮合（触价即成交）'
}

/** 收益百分比着色文本。 */
const pct = (v: number, digits = 1): React.ReactNode => {
  const color = v > 0 ? '#3f8600' : v < 0 ? '#cf1322' : undefined
  return <span style={{ color }}>{v > 0 ? '+' : ''}{(v * 100).toFixed(digits)}%</span>
}

/** 带符号着色的"分"数值。 */
const fen = (v: number): React.ReactNode => {
  const color = v > 0 ? '#3f8600' : v < 0 ? '#cf1322' : undefined
  return <span style={{ color }}>{v > 0 ? '+' : ''}{v.toFixed(2)}</span>
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
  const [sellTick, setSellTick] = useState(0.02)
  const [buyTick, setBuyTick] = useState(0.02)
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
    mut.mutate({
      symbol: symbol.trim(),
      start: dateRange[0].format('YYYY-MM-DD'),
      end: dateRange[1].format('YYYY-MM-DD'),
      sell_tick: sellTick, buy_tick: buyTick, swing_frac: swingFrac, base_weight: baseWeight,
      capital, commission_rate: commissionRate, stamp_duty: stampDuty,
      fill_grid: fillGrid.map((f) => ({ penetration: f.penFen / 100, ratio: f.ratioPct / 100 })),
    })
  }, [mut, symbol, dateRange, sellTick, buyTick, swingFrac, baseWeight, capital, commissionRate, stampDuty, fillGrid])

  // —— 画像 ——
  const profMut = useMutation({
    mutationFn: t0Service.profile,
    onError: (e: unknown) => message.error(
      (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || '做T画像失败'),
  })
  const profile: T0Profile | undefined = profMut.data
  const onProfile = useCallback(() => {
    profMut.mutate({
      symbol: symbol.trim(),
      start: dateRange[0].format('YYYY-MM-DD'),
      end: dateRange[1].format('YYYY-MM-DD'),
      x_max_fen: xMaxFen, commission_rate: commissionRate, stamp_duty: stampDuty,
    })
  }, [profMut, symbol, dateRange, xMaxFen, commissionRate, stampDuty])
  const applySuggested = useCallback(() => {
    if (!profile) return
    setSellTick(profile.suggested_sell_tick)
    setBuyTick(profile.suggested_buy_tick)
    message.success(`已填入建议档位：卖 ${profile.suggested_sell_tick} / 买 ${profile.suggested_buy_tick}`)
  }, [profile])

  const spread = useMemo(() => {
    if (!report?.fill_sensitivity?.length) return null
    const rets = report.fill_sensitivity.map((r) => r.total_return)
    return { hi: Math.max(...rets), lo: Math.min(...rets) }
  }, [report])

  const profCols = [
    { title: '偏离(分)', dataIndex: 'x_fen', key: 'x' },
    { title: '卖腿成交率', dataIndex: 'sell_fill', key: 'sf', render: (v: number) => `${(v * 100).toFixed(0)}%` },
    { title: '卖腿均益(分)', dataIndex: 'sell_edge_fen', key: 'se', render: (v: number) => fen(v) },
    { title: '买腿成交率', dataIndex: 'buy_fill', key: 'bf', render: (v: number) => `${(v * 100).toFixed(0)}%` },
    { title: '买腿均益(分)', dataIndex: 'buy_edge_fen', key: 'be', render: (v: number) => fen(v) },
    { title: '全日期望(分)', dataIndex: 'day_pnl_fen', key: 'dp', render: (v: number) => fen(v) },
  ]
  const yearCols = [
    { title: '年份', dataIndex: 'year', key: 'year' },
    { title: '策略(净)', dataIndex: 'strat', key: 'strat', render: (v: number) => pct(v) },
    { title: '满仓持有', dataIndex: 'bh', key: 'bh', render: (v: number) => pct(v) },
    { title: '半仓持有', dataIndex: 'half_bh', key: 'half_bh', render: (v: number) => pct(v) },
    { title: '超额(vs满仓)', dataIndex: 'excess_vs_bh', key: 'e1', render: (v: number) => pct(v) },
    { title: '超额(vs半仓)', dataIndex: 'excess_vs_half_bh', key: 'e2', render: (v: number) => pct(v) },
  ]
  const ideal = report?.results?.[0]
  const hit = ideal?.hit_dist

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Alert type="info" showIcon
        message="半仓做 T：每天按开盘价挂 ±档位限价单、日内做 T、收盘维持半仓"
        description="先用「标的画像」看这只票各档位的回归边际收益、定档位；再跑回测看「成交敏感性区间」——盈亏几乎全取决于能否在 ±档位真成交，理想撮合是上限、穿越/部分成交更贴近实盘，区间越宽越不可复制。" />

      {/* —— 标的画像 —— */}
      <Card size="small" title={<span><ExperimentOutlined /> 标的画像：按偏离开盘 x 分挂单，单腿做 T 的每笔边际收益（理想撮合）</span>}
        extra={<Button icon={<ExperimentOutlined />} loading={profMut.isPending} onClick={onProfile}>统计当前标的/区间</Button>}>
        {!profile && <Text type="secondary">点右上角统计：按 1~{xMaxFen} 分逐档位、分买/卖腿算成交率与净于成本的回归边际收益，并给出建议档位（天然非对称）。</Text>}
        {profile && (
          <>
            <Alert type="success" showIcon style={{ marginBottom: 12 }}
              message={<>建议档位：卖 <b>{profile.suggested_sell_tick.toFixed(2)}</b> 元 / 买 <b>{profile.suggested_buy_tick.toFixed(2)}</b> 元
                <Button type="link" size="small" onClick={applySuggested}>用建议档位填入回测 →</Button></>}
              description={profile.note} />
            <div style={{ width: '100%', height: 280, marginBottom: 12 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={profile.rows} margin={{ top: 8, right: 24, bottom: 16, left: 0 }}>
                  <CartesianGrid stroke="#f0f0f0" />
                  <XAxis dataKey="x_fen" tick={{ fontSize: 11 }}
                    label={{ value: '偏离开盘（分）', position: 'insideBottom', offset: -8, fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} width={48} unit="分"
                    label={{ value: '每笔均益', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                  <RTooltip formatter={(v) => `${Number(v).toFixed(2)} 分`}
                    labelFormatter={(l) => `偏离 ${l} 分`} />
                  <Legend />
                  <ReferenceLine y={0} stroke="#bbb" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="buy_edge_fen" name="买腿均益" stroke="#1d9e75" strokeWidth={2} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="sell_edge_fen" name="卖腿均益" stroke="#d4537e" strokeWidth={2} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="day_pnl_fen" name="全日期望" stroke="#378add" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <Table<T0BandEdgeRow> size="small" rowKey="x_fen" pagination={false} columns={profCols}
              dataSource={profile.rows}
              rowClassName={(r) => (r.x_fen === Math.round(profile.suggested_buy_tick * 100)
                || r.x_fen === Math.round(profile.suggested_sell_tick * 100)) ? 'ant-table-row-selected' : ''} />
            <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
              均益&gt;0=该档位"触价后倾向回归"（做 T 有底层 edge）；买/卖腿峰值常不在同一档→建议档位天然非对称。注意：这是理想撮合上限，最终须看下方成交敏感性区间。
            </Paragraph>
          </>
        )}
      </Card>

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
          <Col span={5}>
            <Text type="secondary">卖单档位（元）<Tooltip title="开盘价 + 该价差挂卖单"><InfoCircleOutlined /></Tooltip></Text>
            <InputNumber style={{ width: '100%' }} value={sellTick} min={0.01} step={0.01} precision={2}
              onChange={(v) => setSellTick(v ?? 0.02)} />
          </Col>
          <Col span={5}>
            <Text type="secondary">买单档位（元）<Tooltip title="开盘价 − 该价差挂买单"><InfoCircleOutlined /></Tooltip></Text>
            <InputNumber style={{ width: '100%' }} value={buyTick} min={0.01} step={0.01} precision={2}
              onChange={(v) => setBuyTick(v ?? 0.02)} />
          </Col>
          <Col span={5}>
            <Text type="secondary">摆动幅度<Tooltip title="做T一手占半仓比例，1=全半仓摆动"><InfoCircleOutlined /></Tooltip></Text>
            <InputNumber style={{ width: '100%' }} value={swingFrac} min={0.1} max={1} step={0.1}
              onChange={(v) => setSwingFrac(v ?? 1)} />
          </Col>
          <Col span={5}>
            <Text type="secondary">半仓锚权重</Text>
            <InputNumber style={{ width: '100%' }} value={baseWeight} min={0.1} max={0.9} step={0.1}
              onChange={(v) => setBaseWeight(v ?? 0.5)} />
          </Col>
          <Col span={4}>
            <Text type="secondary">画像档位上限(分)</Text>
            <InputNumber style={{ width: '100%' }} value={xMaxFen} min={2} max={50} step={1}
              onChange={(v) => setXMaxFen(v ?? 15)} />
          </Col>
        </Row>

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

          {ideal && (
            <Card size="small" title="逐年收益与超额（第一个成交假设口径）">
              <Table<T0PeriodRow> size="small" rowKey={(r) => String(r.year)} pagination={false}
                columns={yearCols} dataSource={ideal.yearly} />
            </Card>
          )}

          {hit && (
            <Card size="small" title="命中分布（开盘 ±档位 是否被触及）">
              <Space size="large">
                <Statistic title="两边都触（对敲）" value={(hit.both * 100).toFixed(0)} suffix="%" />
                <Statistic title="仅触卖" value={(hit.onlyS * 100).toFixed(0)} suffix="%" />
                <Statistic title="仅触买" value={(hit.onlyB * 100).toFixed(0)} suffix="%" />
                <Statistic title="都不触" value={(hit.none * 100).toFixed(0)} suffix="%" />
              </Space>
              <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
                年化换手 {ideal?.turnover_annual}x。「两边都触」是对敲落袋主力，「仅触一边」在趋势日易亏。
              </Paragraph>
            </Card>
          )}
        </>
      )}
    </Space>
  )
}

export default T0Backtest
