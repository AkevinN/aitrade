// 条件规则编辑器：有序单条件规则（左值 op 阈值 → 卖/买档）+ 默认档位。
// 内部以后端口径(ticks 元、gap 阈值小数)保存，渲染以"分/%"友好展示，提交无需再换算。
import React from 'react'
import { Select, InputNumber, Button, Space, Typography, Tooltip } from 'antd'
import {
  PlusOutlined, MinusCircleOutlined, ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined,
} from '@ant-design/icons'

import type { RuleCfg } from '../../types/t0'

const { Text } = Typography

/** 左值选项（与后端 lhs 白名单一致）。 */
const LHS_OPTIONS = [
  { value: 'gap', label: '跳空 gap（今开/昨收−1）' },
  { value: 'mean_range', label: '近 N 日均振幅' },
  { value: 'momentum', label: '近 N 日动量' },
  { value: 'signal', label: '信号值' },
] as const

/** 比较运算符显示。 */
const OP_OPTIONS = [
  { value: 'gt', label: '>' },
  { value: 'ge', label: '≥' },
  { value: 'lt', label: '<' },
  { value: 'le', label: '≤' },
] as const

/** 元→分（四舍五入到整数分）。 */
const toFen = (yuan: number): number => Math.round(yuan * 100)
/** 分→元。 */
const fromFen = (fen: number): number => fen / 100
/** 小数→百分比显示（避免浮点尾噪）。 */
const toPct = (dec: number): number => Number((dec * 100).toFixed(4))

/** 单条规则在 UI 上的左值是否需要 window / signal_name。 */
const needsWindow = (lhs: RuleCfg['lhs']): boolean => lhs === 'mean_range' || lhs === 'momentum'
const needsSignal = (lhs: RuleCfg['lhs']): boolean => lhs === 'signal'

export interface ConditionalRuleEditorProps {
  /** 有序单条件规则（后端单位：ticks 为元、gap 阈值为小数） */
  rules: RuleCfg[]
  /** 无规则命中时的卖档（元） */
  defaultSellTick: number
  /** 无规则命中时的买档（元） */
  defaultBuyTick: number
  /** 可选信号名（lhs="signal" 的下拉来源） */
  signalNames: string[]
  /** 规则或默认档变更回调（整体回传，后端单位） */
  onChange: (next: { rules: RuleCfg[]; defaultSellTick: number; defaultBuyTick: number }) => void
}

/**
 * 条件规则编辑器：增删/上下移有序单条件规则，并配默认档位。
 *
 * 每条规则 = 左值(gap/振幅/动量/信号) `op` 阈值 → (卖档, 买档)；按序首个命中即生效，
 * 无命中走默认档。校验交由上层（如 signal 缺 signal_name 时上层禁用运行）。
 */
const ConditionalRuleEditor: React.FC<ConditionalRuleEditorProps> = ({
  rules, defaultSellTick, defaultBuyTick, signalNames, onChange,
}) => {
  const emit = (
    nextRules: RuleCfg[],
    nextDS: number = defaultSellTick,
    nextDB: number = defaultBuyTick,
  ) => onChange({ rules: nextRules, defaultSellTick: nextDS, defaultBuyTick: nextDB })

  const updateRule = (i: number, patch: Partial<RuleCfg>) =>
    emit(rules.map((r, idx) => (idx === i ? { ...r, ...patch } : r)))

  const onLhsChange = (i: number, lhs: RuleCfg['lhs']) => {
    // 切左值时清理不适用字段，置默认，避免脏字段提交
    const patch: Partial<RuleCfg> = { lhs }
    patch.window = needsWindow(lhs) ? (rules[i].window ?? (lhs === 'mean_range' ? 20 : 5)) : undefined
    patch.signal_name = needsSignal(lhs) ? (rules[i].signal_name ?? signalNames[0]) : undefined
    if (lhs === 'gap') patch.threshold = 0.003
    updateRule(i, patch)
  }

  const addRule = () => emit([
    ...rules,
    { name: '', lhs: 'gap', op: 'gt', threshold: 0.003, sell_tick: 0.07, buy_tick: 0.01 },
  ])
  const removeRule = (i: number) => emit(rules.filter((_, idx) => idx !== i))
  const moveRule = (i: number, dir: -1 | 1) => {
    const j = i + dir
    if (j < 0 || j >= rules.length) return
    const next = [...rules]
    ;[next[i], next[j]] = [next[j], next[i]]
    emit(next)
  }

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={6}>
      {rules.map((r, i) => (
        <Space key={i} wrap>
          <Text type="secondary" style={{ fontSize: 12 }}>若</Text>
          <Select<RuleCfg['lhs']> size="small" style={{ width: 170 }} value={r.lhs}
            aria-label={`规则${i}左值`}
            onChange={(v) => onLhsChange(i, v)}
            options={LHS_OPTIONS.map((o) => ({ value: o.value, label: o.label }))} />
          {needsWindow(r.lhs) && (
            <InputNumber size="small" addonBefore="N" addonAfter="日" min={1} max={250} style={{ width: 130 }}
              aria-label={`规则${i}窗口`} value={r.window ?? 20}
              onChange={(v) => updateRule(i, { window: v ?? 20 })} />
          )}
          {needsSignal(r.lhs) && (
            <Select size="small" style={{ width: 180 }} placeholder="选择信号" value={r.signal_name}
              aria-label={`规则${i}信号`} notFoundContent={signalNames.length ? '无匹配' : '无可用信号'}
              onChange={(v) => updateRule(i, { signal_name: v })}
              options={signalNames.map((n) => ({ value: n, label: n }))} />
          )}
          <Select<RuleCfg['op']> size="small" style={{ width: 70 }} value={r.op}
            aria-label={`规则${i}运算`}
            onChange={(v) => updateRule(i, { op: v })}
            options={OP_OPTIONS.map((o) => ({ value: o.value, label: o.label }))} />
          {r.lhs === 'gap' ? (
            <InputNumber size="small" addonAfter="%" step={0.1} style={{ width: 110 }}
              aria-label={`规则${i}阈值`} value={toPct(r.threshold)}
              onChange={(v) => updateRule(i, { threshold: (v ?? 0) / 100 })} />
          ) : (
            <InputNumber size="small" step={0.1} style={{ width: 110 }}
              aria-label={`规则${i}阈值`} value={r.threshold}
              onChange={(v) => updateRule(i, { threshold: v ?? 0 })} />
          )}
          <Text type="secondary" style={{ fontSize: 12 }}>则 卖</Text>
          <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 100 }}
            aria-label={`规则${i}卖档`} value={toFen(r.sell_tick)}
            onChange={(v) => updateRule(i, { sell_tick: fromFen(v ?? 1) })} />
          <Text type="secondary" style={{ fontSize: 12 }}>买</Text>
          <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 100 }}
            aria-label={`规则${i}买档`} value={toFen(r.buy_tick)}
            onChange={(v) => updateRule(i, { buy_tick: fromFen(v ?? 1) })} />
          <Button size="small" type="text" icon={<ArrowUpOutlined />} disabled={i === 0}
            aria-label={`规则${i}上移`} onClick={() => moveRule(i, -1)} />
          <Button size="small" type="text" icon={<ArrowDownOutlined />} disabled={i === rules.length - 1}
            aria-label={`规则${i}下移`} onClick={() => moveRule(i, 1)} />
          <Button size="small" danger type="text" icon={<MinusCircleOutlined />}
            aria-label={`规则${i}删除`} disabled={rules.length <= 1} onClick={() => removeRule(i)} />
        </Space>
      ))}
      <Space>
        <Button size="small" icon={<PlusOutlined />} onClick={addRule}>添加规则</Button>
        <Text type="secondary" style={{ fontSize: 12 }}>
          无规则命中 → 默认 <Tooltip title="所有规则都不满足时用此档位"><InfoCircleOutlined /></Tooltip> 卖
        </Text>
        <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 100 }}
          aria-label="默认卖档" value={toFen(defaultSellTick)}
          onChange={(v) => emit(rules, fromFen(v ?? 1), defaultBuyTick)} />
        <Text type="secondary" style={{ fontSize: 12 }}>买</Text>
        <InputNumber size="small" addonAfter="分" min={1} step={1} style={{ width: 100 }}
          aria-label="默认买档" value={toFen(defaultBuyTick)}
          onChange={(v) => emit(rules, defaultSellTick, fromFen(v ?? 1))} />
      </Space>
    </Space>
  )
}

export default ConditionalRuleEditor
