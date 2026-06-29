// 档位策略编辑器：增删多个命名策略 + kind 切换（固定/波动缩放/趋势倾斜/条件规则）。
// label 一次请求内唯一（重复标红）；条件策略内嵌 ConditionalRuleEditor。
import React from 'react'
import { Card, Select, InputNumber, Button, Space, Input, Typography, Row, Col } from 'antd'
import { PlusOutlined, DeleteOutlined, ExperimentOutlined } from '@ant-design/icons'

import type { TickPolicyCfg, RuleCfg } from '../../types/t0'
import ConditionalRuleEditor from './ConditionalRuleEditor'

const { Text } = Typography

const toFen = (yuan: number): number => Math.round(yuan * 100)
const fromFen = (fen: number): number => fen / 100

/** 策略类型选项。 */
const KIND_OPTIONS = [
  { value: 'fixed', label: '固定档' },
  { value: 'vol_scaled', label: '波动缩放' },
  { value: 'trend_tilt', label: '趋势倾斜' },
  { value: 'conditional', label: '条件规则' },
] as const

type Kind = TickPolicyCfg['kind']

/** 造一个指定 kind 的默认策略（保留 label）。 */
const blankOfKind = (kind: Kind, label: string): TickPolicyCfg => {
  switch (kind) {
    case 'fixed':
      return { kind: 'fixed', label, sell_tick: 0.02, buy_tick: 0.02 }
    case 'vol_scaled':
      return { kind: 'vol_scaled', label, k: 0.4, n: 20, fallback: 0.02 }
    case 'trend_tilt':
      return { kind: 'trend_tilt', label, base: 0.02, tilt: 0.01, n: 5 }
    case 'conditional':
      return {
        kind: 'conditional', label,
        rules: [
          { name: '高开', lhs: 'gap', op: 'gt', threshold: 0.003, sell_tick: 0.07, buy_tick: 0.01 },
          { name: '低开', lhs: 'gap', op: 'lt', threshold: -0.003, sell_tick: 0.09, buy_tick: 0.01 },
        ],
        default_sell_tick: 0.03, default_buy_tick: 0.03, pricetick: 0.01,
      }
  }
}

export interface TickPolicyEditorProps {
  /** 当前策略列表（后端单位） */
  value: TickPolicyCfg[]
  /** 列表变更回调 */
  onChange: (v: TickPolicyCfg[]) => void
  /** 可用信号名（透传给条件规则编辑器） */
  signalNames: string[]
  /** 点某策略「画像并建议」时触发（传该策略下标）；不传则不显示该入口 */
  onCalibrate?: (index: number) => void
}

/**
 * 档位策略编辑器：定义多个命名策略，一次回测出"策略 × 成交假设"对比。
 *
 * 默认应至少含一个策略；label 重复会标红（上层据此可禁用运行）。条件规则策略内嵌
 * {@link ConditionalRuleEditor}；其余为数值参数表单（档位以"分"展示、后端口径为元）。
 */
const TickPolicyEditor: React.FC<TickPolicyEditorProps> = ({ value, onChange, signalNames, onCalibrate }) => {
  // 以 trim 后的 label 计重复（与运行前校验/提交口径一致）
  const labelCounts = value.reduce<Record<string, number>>((m, p) => {
    const k = p.label.trim()
    m[k] = (m[k] ?? 0) + 1
    return m
  }, {})
  const isDup = (label: string): boolean => (labelCounts[label.trim()] ?? 0) > 1

  const updatePolicy = (i: number, patch: Partial<TickPolicyCfg>) =>
    onChange(value.map((p, idx) => (idx === i ? ({ ...p, ...patch } as TickPolicyCfg) : p)))
  const replacePolicy = (i: number, next: TickPolicyCfg) =>
    onChange(value.map((p, idx) => (idx === i ? next : p)))

  const addPolicy = () => {
    // 取一个未占用的默认名
    let n = value.length + 1
    while (value.some((p) => p.label === `策略${n}`)) n += 1
    onChange([...value, blankOfKind('fixed', `策略${n}`)])
  }
  const removePolicy = (i: number) => onChange(value.filter((_, idx) => idx !== i))

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={10}>
      {value.map((p, i) => (
        <Card key={i} size="small" bordered
          title={
            <Space>
              <Input size="small" style={{ width: 160 }} value={p.label}
                aria-label={`策略${i}名称`} status={isDup(p.label) ? 'error' : ''}
                onChange={(e) => updatePolicy(i, { label: e.target.value } as Partial<TickPolicyCfg>)} />
              <Select<Kind> size="small" style={{ width: 120 }} value={p.kind}
                aria-label={`策略${i}类型`}
                onChange={(k) => replacePolicy(i, blankOfKind(k, p.label))}
                options={KIND_OPTIONS.map((o) => ({ value: o.value, label: o.label }))} />
              {isDup(p.label) && <Text type="danger" style={{ fontSize: 12 }}>名称重复</Text>}
            </Space>
          }
          extra={
            <Space>
              {onCalibrate && (
                <Button size="small" icon={<ExperimentOutlined />} aria-label={`策略${i}画像`}
                  onClick={() => onCalibrate(i)}>画像并建议</Button>
              )}
              <Button size="small" danger type="text" icon={<DeleteOutlined />} disabled={value.length <= 1}
                aria-label={`策略${i}删除`} onClick={() => removePolicy(i)} />
            </Space>
          }>
          {p.kind === 'fixed' && (
            <Row gutter={12}>
              <Col><Text type="secondary">卖档</Text>
                <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 110 }}
                  aria-label={`策略${i}卖档`} value={toFen(p.sell_tick)}
                  onChange={(v) => updatePolicy(i, { sell_tick: fromFen(v ?? 1) } as Partial<TickPolicyCfg>)} /></Col>
              <Col><Text type="secondary">买档</Text>
                <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 110 }}
                  aria-label={`策略${i}买档`} value={toFen(p.buy_tick)}
                  onChange={(v) => updatePolicy(i, { buy_tick: fromFen(v ?? 1) } as Partial<TickPolicyCfg>)} /></Col>
            </Row>
          )}
          {p.kind === 'vol_scaled' && (
            <Row gutter={12}>
              <Col><Text type="secondary">系数 k</Text>
                <InputNumber size="small" min={0.01} step={0.1} style={{ width: 100 }}
                  aria-label={`策略${i}系数k`} value={p.k}
                  onChange={(v) => updatePolicy(i, { k: v ?? 0.4 } as Partial<TickPolicyCfg>)} /></Col>
              <Col><Text type="secondary">窗口 N(日)</Text>
                <InputNumber size="small" min={1} step={1} style={{ width: 100 }}
                  aria-label={`策略${i}窗口`} value={p.n}
                  onChange={(v) => updatePolicy(i, { n: v ?? 20 } as Partial<TickPolicyCfg>)} /></Col>
              <Col><Text type="secondary">回退档</Text>
                <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 110 }}
                  aria-label={`策略${i}回退档`} value={toFen(p.fallback)}
                  onChange={(v) => updatePolicy(i, { fallback: fromFen(v ?? 2) } as Partial<TickPolicyCfg>)} /></Col>
            </Row>
          )}
          {p.kind === 'trend_tilt' && (
            <Row gutter={12}>
              <Col><Text type="secondary">基准档</Text>
                <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 110 }}
                  aria-label={`策略${i}基准档`} value={toFen(p.base)}
                  onChange={(v) => updatePolicy(i, { base: fromFen(v ?? 2) } as Partial<TickPolicyCfg>)} /></Col>
              <Col><Text type="secondary">倾斜量</Text>
                <InputNumber size="small" addonAfter="分" min={0} step={1} style={{ width: 110 }}
                  aria-label={`策略${i}倾斜量`} value={toFen(p.tilt)}
                  onChange={(v) => updatePolicy(i, { tilt: fromFen(v ?? 1) } as Partial<TickPolicyCfg>)} /></Col>
              <Col><Text type="secondary">窗口 N(日)</Text>
                <InputNumber size="small" min={1} step={1} style={{ width: 100 }}
                  aria-label={`策略${i}窗口`} value={p.n}
                  onChange={(v) => updatePolicy(i, { n: v ?? 5 } as Partial<TickPolicyCfg>)} /></Col>
            </Row>
          )}
          {p.kind === 'conditional' && (
            <ConditionalRuleEditor
              rules={p.rules} defaultSellTick={p.default_sell_tick} defaultBuyTick={p.default_buy_tick}
              signalNames={signalNames}
              onChange={(next: { rules: RuleCfg[]; defaultSellTick: number; defaultBuyTick: number }) =>
                updatePolicy(i, {
                  rules: next.rules, default_sell_tick: next.defaultSellTick, default_buy_tick: next.defaultBuyTick,
                } as Partial<TickPolicyCfg>)} />
          )}
        </Card>
      ))}
      <Button size="small" icon={<PlusOutlined />} onClick={addPolicy}>添加档位策略</Button>
    </Space>
  )
}

export default TickPolicyEditor
