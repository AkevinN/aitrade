// 半仓做 T 回测页面 — 同步运行，核心产物是"成交敏感性区间"
import React, { useState, useMemo, useCallback } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, DatePicker, InputNumber,
  Input, Alert, Table, Statistic, Tag, Tooltip, message,
} from 'antd'
import { PlayCircleOutlined, ThunderboltOutlined, InfoCircleOutlined } from '@ant-design/icons'
import { useMutation } from '@tanstack/react-query'
import dayjs, { type Dayjs } from 'dayjs'

import { t0Service } from '../../api/t0'
import type { T0FillCfg, T0FillSensitivityRow, T0PeriodRow, T0Report } from '../../types/t0'

const { Text, Paragraph } = Typography
const { RangePicker } = DatePicker

/** 把成交假设格式化为可读标签。 */
const fillLabel = (f: T0FillCfg): string => {
  if (f.penetration > 0) return `穿越 ${Math.round(f.penetration * 100)} 分`
  if (f.ratio < 1) return `部分成交 ${Math.round(f.ratio * 100)}%`
  return '理想撮合（触价即成交）'
}

/** 收益百分比着色文本。 */
const pct = (v: number, digits = 1): React.ReactNode => {
  const s = `${(v * 100).toFixed(digits)}%`
  const color = v > 0 ? '#3f8600' : v < 0 ? '#cf1322' : undefined
  return <span style={{ color }}>{v > 0 ? '+' : ''}{s}</span>
}

/**
 * 半仓做 T 回测页面。
 *
 * 用户配置标的/区间/挂单档位/摆动幅度后同步运行；结果以**成交敏感性区间**为核心——
 * 同一策略在"理想撮合 / 穿越成交 / 部分成交"下的收益区间，直观暴露这套做 T 对成交质量的依赖。
 * 另给逐年超额（vs 满仓持有 / 每日再平衡半仓）与命中分布。
 */
const T0Backtest: React.FC = () => {
  const [symbol, setSymbol] = useState('000415.SZSE')
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs]>([
    dayjs('2023-01-01'), dayjs('2025-12-31'),
  ])
  const [sellTick, setSellTick] = useState(0.02)
  const [buyTick, setBuyTick] = useState(0.02)
  const [swingFrac, setSwingFrac] = useState(1.0)
  const [baseWeight, setBaseWeight] = useState(0.5)
  const [capital, setCapital] = useState(1_000_000)

  const mut = useMutation({
    mutationFn: t0Service.runBacktest,
    onError: (e: unknown) => {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      message.error(detail || '做T回测失败')
    },
  })
  const report: T0Report | undefined = mut.data

  const onRun = useCallback(() => {
    mut.mutate({
      symbol: symbol.trim(),
      start: dateRange[0].format('YYYY-MM-DD'),
      end: dateRange[1].format('YYYY-MM-DD'),
      sell_tick: sellTick,
      buy_tick: buyTick,
      swing_frac: swingFrac,
      base_weight: baseWeight,
      capital,
      commission_rate: 0.0003,
      stamp_duty: 0.0005,
      fill_grid: [
        { penetration: 0.0, ratio: 1.0 },
        { penetration: 0.01, ratio: 1.0 },
        { penetration: 0.0, ratio: 0.5 },
      ],
    })
  }, [mut, symbol, dateRange, sellTick, buyTick, swingFrac, baseWeight, capital])

  // 理想 vs 穿越 的收益落差（区间宽度），作为"成交质量依赖度"的提示
  const spread = useMemo(() => {
    if (!report?.fill_sensitivity?.length) return null
    const rets = report.fill_sensitivity.map((r) => r.total_return)
    return { hi: Math.max(...rets), lo: Math.min(...rets) }
  }, [report])

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
      <Alert
        type="info"
        showIcon
        message="半仓做 T：每天按开盘价挂 ±档位限价单、日内做 T、收盘维持半仓"
        description="核心产物是「成交敏感性区间」——这套做 T 的盈亏几乎全取决于能否在 ±档位真成交。理想撮合（触价即成交）是上限；穿越/部分成交更贴近实盘。区间越宽，说明越依赖成交质量、实盘越不可复制。"
      />

      <Card size="small" title="回测配置">
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Text type="secondary">标的</Text>
            <Input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="000415.SZSE" />
          </Col>
          <Col span={10}>
            <Text type="secondary">评估区间</Text>
            <RangePicker
              style={{ width: '100%' }}
              value={dateRange}
              onChange={(v) => v && v[0] && v[1] && setDateRange([v[0], v[1]])}
              allowClear={false}
            />
          </Col>
          <Col span={4}>
            <Text type="secondary">初始资金</Text>
            <InputNumber style={{ width: '100%' }} value={capital} min={100000} step={100000}
              onChange={(v) => setCapital(v ?? 1_000_000)} />
          </Col>
        </Row>
        <Row gutter={[16, 16]} style={{ marginTop: 12 }}>
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
          <Col span={4} style={{ display: 'flex', alignItems: 'flex-end' }}>
            <Button type="primary" icon={<PlayCircleOutlined />} loading={mut.isPending}
              onClick={onRun} block>运行回测</Button>
          </Col>
        </Row>
        <Text type="secondary" style={{ fontSize: 12 }}>
          成交假设网格固定为：理想撮合 / 穿越 1 分 / 部分成交 50%（用于框出收益区间）。
        </Text>
      </Card>

      {report && (
        <>
          <Card size="small" title={<span><ThunderboltOutlined /> 成交敏感性区间（{report.symbol}，{report.eval_window[0]} ~ {report.eval_window[1]}）</span>}>
            <Row gutter={16}>
              {report.fill_sensitivity.map((r: T0FillSensitivityRow, i) => (
                <Col span={8} key={i}>
                  <Card size="small" bordered>
                    <Tag color={r.fill.penetration > 0 ? 'volcano' : r.fill.ratio < 1 ? 'gold' : 'green'}>
                      {fillLabel(r.fill)}
                    </Tag>
                    <Statistic value={r.total_return * 100} precision={1} suffix="%"
                      valueStyle={{ color: r.total_return > 0 ? '#3f8600' : '#cf1322' }} />
                    <Text type="secondary">Sharpe {(r.sharpe ?? 0).toFixed(2)} ｜ 回撤 {(r.max_drawdown * 100).toFixed(0)}%</Text>
                  </Card>
                </Col>
              ))}
            </Row>
            {spread && (
              <Alert
                style={{ marginTop: 12 }}
                type={spread.hi - spread.lo > 0.3 ? 'warning' : 'success'}
                showIcon
                message={`收益区间宽度 ${((spread.hi - spread.lo) * 100).toFixed(0)} 个百分点（理想 ${(spread.hi * 100).toFixed(0)}% ~ 最差 ${(spread.lo * 100).toFixed(0)}%）`}
                description={spread.hi - spread.lo > 0.3
                  ? '区间很宽：盈亏高度依赖成交质量，实盘极可能拿不到理想撮合那端，谨慎对待。'
                  : '区间较窄：对成交假设相对不敏感。'}
              />
            )}
          </Card>

          {ideal && (
            <Card size="small" title="逐年收益与超额（理想撮合口径）">
              <Table<T0PeriodRow>
                size="small"
                rowKey={(r) => String(r.year)}
                pagination={false}
                columns={yearCols}
                dataSource={ideal.yearly}
              />
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
                年化换手 {ideal?.turnover_annual}x。命中分布解释收益来源：「两边都触」是对敲落袋的主力，「仅触一边」在趋势日易亏。
              </Paragraph>
            </Card>
          )}
        </>
      )}
    </Space>
  )
}

export default T0Backtest
