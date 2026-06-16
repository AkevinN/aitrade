import React, { useState } from 'react'
import {
  Alert,
  Collapse,
  Descriptions,
  Empty,
  Spin,
  Tag,
  Typography,
} from 'antd'
import { useQuery } from '@tanstack/react-query'

import { liveService } from '../../api/liveApi'
import RiskDetailPanel from './RiskDetailPanel'
import type { Task } from '../../types/alpha'
import type {
  DecisionTrace,
  LiveDecisionResult,
  TraceDecisionLogicSection,
  TraceInferenceSection,
  TracePricingSection,
  TraceResultSection,
  TraceRiskSection,
  TraceRunHeaderSection,
} from '../../types/live'

const { Text } = Typography

/**
 * {@link DecisionTracePanel} 组件 props。
 */
interface DecisionTracePanelProps {
  /** 关联决策的 signal_id；为空时不渲染（无可观测对象）。 */
  signalId?: string | null
}

/** 六段顺序与中文标题（运行头/推理段/取价段/决策逻辑段/风控段/结果段）。 */
const SECTION_TITLES: { key: string; title: string }[] = [
  { key: 'run_header', title: '运行头' },
  { key: 'inference', title: '推理段' },
  { key: 'pricing', title: '取价段' },
  { key: 'decision_logic', title: '决策逻辑段' },
  { key: 'risk', title: '风控段' },
  { key: 'result', title: '结果段' },
]

/**
 * 数字字段安全格式化：把可能缺失的数值转成可直接渲染的字符串。
 *
 * @param value - 待格式化的数值；null/undefined/NaN 视为缺失
 * @returns 缺失时返回占位符 "—"，否则返回数值的字符串形式
 */
function fmtNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return String(value)
}

/**
 * 把布尔值渲染为「是/否」彩色标签，缺失值降级为占位符。
 *
 * @param value - 布尔字段；null/undefined 视为缺失
 * @returns 缺失时返回灰色占位符 "—"，true 渲染绿色「是」，false 渲染默认色「否」
 */
function boolTag(value: boolean | null | undefined): React.ReactNode {
  if (value === null || value === undefined) return <Text type="secondary">—</Text>
  return value ? <Tag color="green">是</Tag> : <Tag color="default">否</Tag>
}

/** ① 运行头段内容。 */
const RunHeaderContent: React.FC<{ data?: TraceRunHeaderSection }> = ({ data }) => {
  if (!data) return <Empty description="无运行头信息" />
  return (
    <Descriptions size="small" column={1} bordered>
      <Descriptions.Item label="run_id">
        <Text code>{data.run_id}</Text>
      </Descriptions.Item>
      <Descriptions.Item label="模型">
        {data.model_name}
        {data.model_version ? ` @ ${data.model_version}` : ''}
      </Descriptions.Item>
      <Descriptions.Item label="目标标的">{data.vt_symbol}</Descriptions.Item>
      <Descriptions.Item label="方案">{data.scheme}</Descriptions.Item>
      <Descriptions.Item label="决策时刻">{data.as_of}</Descriptions.Item>
      <Descriptions.Item label="bar 频率">{data.bar_freq}</Descriptions.Item>
      <Descriptions.Item label="数据源类型">{data.data_source_type}</Descriptions.Item>
      <Descriptions.Item label="买入阈值">{fmtNumber(data.buy_threshold)}</Descriptions.Item>
      <Descriptions.Item label="组合总市值">
        {fmtNumber(data.portfolio?.portfolio_value)}
      </Descriptions.Item>
      <Descriptions.Item label="风控摘要">
        总仓上限 {fmtNumber(data.risk_config_summary?.max_total_position_ratio)} / 单票上限{' '}
        {fmtNumber(data.risk_config_summary?.max_single_position_ratio)} / 黑名单{' '}
        {fmtNumber(data.risk_config_summary?.blacklist_size)} 项
      </Descriptions.Item>
    </Descriptions>
  )
}

/** ② 推理段内容。 */
const InferenceContent: React.FC<{ data?: TraceInferenceSection }> = ({ data }) => {
  if (!data) return <Empty description="无推理信息" />
  return (
    <Descriptions size="small" column={1} bordered>
      <Descriptions.Item label="目标标的">{data.target_symbol}</Descriptions.Item>
      <Descriptions.Item label="回看窗口">{fmtNumber(data.lookback)}</Descriptions.Item>
      <Descriptions.Item label="输入周期">{data.input_interval}</Descriptions.Item>
      <Descriptions.Item label="训练目标">{data.objective}</Descriptions.Item>
      <Descriptions.Item label="观测标的">
        {data.observation_symbols?.join('、') || '—'}（{fmtNumber(data.observation_group_count)} 组）
      </Descriptions.Item>
      <Descriptions.Item label="预热起点">{data.warmup_start}</Descriptions.Item>
      <Descriptions.Item label="对齐步数 / 有效点">
        {fmtNumber(data.total_steps)} / {fmtNumber(data.valid_points)}
      </Descriptions.Item>
      <Descriptions.Item label="信号序列统计">
        count={fmtNumber(data.signal_seq_stats?.count)}，mean=
        {fmtNumber(data.signal_seq_stats?.mean)}，min={fmtNumber(data.signal_seq_stats?.min)}，max=
        {fmtNumber(data.signal_seq_stats?.max)}
      </Descriptions.Item>
      <Descriptions.Item label="决策 bar 信号">
        {fmtNumber(data.decision_signal)}
      </Descriptions.Item>
    </Descriptions>
  )
}

/** ③ 取价段内容。 */
const PricingContent: React.FC<{ data?: TracePricingSection }> = ({ data }) => {
  if (!data) return <Empty description="无取价信息" />
  return (
    <Descriptions size="small" column={1} bordered>
      <Descriptions.Item label="取价周期">{data.interval_used}</Descriptions.Item>
      <Descriptions.Item label="决策日收盘价">{fmtNumber(data.close_price)}</Descriptions.Item>
    </Descriptions>
  )
}

/** ④ 决策逻辑段内容。 */
const DecisionLogicContent: React.FC<{ data?: TraceDecisionLogicSection }> = ({ data }) => {
  if (!data) return <Empty description="无决策逻辑信息" />
  return (
    <Descriptions size="small" column={1} bordered>
      <Descriptions.Item label="决策日信号">{fmtNumber(data.signal)}</Descriptions.Item>
      <Descriptions.Item label="买入阈值">{fmtNumber(data.buy_threshold)}</Descriptions.Item>
      <Descriptions.Item label="信号是否达标">{boolTag(data.signal_passed)}</Descriptions.Item>
      <Descriptions.Item label="目标仓位市值">{fmtNumber(data.target_value)}</Descriptions.Item>
      <Descriptions.Item label="计划成交手数">{fmtNumber(data.volume)}</Descriptions.Item>
      <Descriptions.Item label="计划成交市值">{fmtNumber(data.intended_value)}</Descriptions.Item>
      <Descriptions.Item label="是否触发出场">{boolTag(data.should_exit)}</Descriptions.Item>
      <Descriptions.Item label="是否停牌/封死">{boolTag(data.halted)}</Descriptions.Item>
    </Descriptions>
  )
}

/**
 * ⑤ 风控段内容。
 * 复用既有 RiskDetailPanel：其按 task.result.risk_detail 渲染逐项 check/passed/detail，
 * 因此用 trace 的 records 构造一个完成态的合成 Task 交给它，避免重复实现风控明细表。
 */
const RiskContent: React.FC<{ data?: TraceRiskSection }> = ({ data }) => {
  if (!data) return <Empty description="无风控信息" />
  // 构造合成 Task：仅 RiskDetailPanel 关心的字段（status / result.risk_detail）。
  const syntheticResult: LiveDecisionResult = {
    // RiskDetailPanel 不读取 decision，给一个占位即可。
    decision: undefined as never,
    risk_detail: data.records ?? [],
    idempotent_hit: false,
  }
  const syntheticTask = {
    status: 'completed',
    result: syntheticResult as unknown as Record<string, unknown>,
  } as unknown as Task
  return (
    <>
      <RiskDetailPanel task={syntheticTask} />
      <div style={{ marginTop: 8 }}>
        <Text type="secondary">
          权威结论（RiskManager.check_buy）：{data.authoritative_ok ? '放行' : '拦截'}
        </Text>
      </div>
    </>
  )
}

/** ⑥ 结果段内容：含 idempotent_hit / trace_persisted / abort_reason 可观测标记。 */
const ResultContent: React.FC<{ data?: TraceResultSection }> = ({ data }) => {
  if (!data) return <Empty description="无结果信息" />
  return (
    <>
      <Descriptions size="small" column={1} bordered>
        <Descriptions.Item label="action">{data.action ?? '—'}</Descriptions.Item>
        <Descriptions.Item label="手数">{fmtNumber(data.volume)}</Descriptions.Item>
        <Descriptions.Item label="价位">{fmtNumber(data.price)}</Descriptions.Item>
        <Descriptions.Item label="reason">{data.reason ?? '—'}</Descriptions.Item>
        <Descriptions.Item label="signal_id">
          <Text code style={{ fontSize: 12 }}>
            {data.signal_id}
          </Text>
        </Descriptions.Item>
      </Descriptions>
      <div style={{ marginTop: 8 }}>
        <Tag color={data.idempotent_hit ? 'gold' : 'default'}>
          idempotent_hit：{data.idempotent_hit ? '是' : '否'}
        </Tag>
        <Tag color={data.trace_persisted ? 'green' : 'red'}>
          trace_persisted：{data.trace_persisted ? '是' : '否'}
        </Tag>
        {data.abort_reason ? (
          <Tag color="red">abort_reason：{data.abort_reason}</Tag>
        ) : (
          <Tag color="default">abort_reason：无</Tag>
        )}
      </div>
      {data.trace_persist_error ? (
        <Alert
          style={{ marginTop: 8 }}
          type="warning"
          showIcon
          message="过程档案持久化失败"
          description={data.trace_persist_error}
        />
      ) : null}
    </>
  )
}

/**
 * 按段名分发到对应的段内容组件，并从 trace 中取出该段数据交给它。
 *
 * @param key - 六段之一的标识（run_header/inference/pricing/decision_logic/risk/result）
 * @param trace - 完整过程档案；为 undefined 时各段组件自行渲染空态
 * @returns 对应段的内容节点；key 不在六段内时返回 null
 */
function renderSection(key: string, trace?: DecisionTrace): React.ReactNode {
  const sections = trace?.sections
  switch (key) {
    case 'run_header':
      return <RunHeaderContent data={sections?.run_header} />
    case 'inference':
      return <InferenceContent data={sections?.inference} />
    case 'pricing':
      return <PricingContent data={sections?.pricing} />
    case 'decision_logic':
      return <DecisionLogicContent data={sections?.decision_logic} />
    case 'risk':
      return <RiskContent data={sections?.risk} />
    case 'result':
      return <ResultContent data={sections?.result} />
    default:
      return null
  }
}

/**
 * 决策过程面板（任务 20.1）。
 *
 * 将一份 Decision_Trace 渲染为六段可折叠分组（运行头/推理段/取价段/决策逻辑段/风控段/结果段），
 * 默认全部折叠。懒加载：仅当展开任意分组时才调 liveService.getDecisionTrace(signalId)。
 * 后端 404（无过程档案）时显示「暂无过程档案」。（Req 8.6）
 */
const DecisionTracePanel: React.FC<DecisionTracePanelProps> = ({ signalId }) => {
  // Collapse 当前展开的段；默认空数组 = 全部折叠。
  const [activeKeys, setActiveKeys] = useState<string[]>([])

  // 懒加载：仅当有 signalId 且至少展开一个分组时才发起 /trace 请求。
  const enabled = !!signalId && activeKeys.length > 0

  const {
    data: trace,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['decision-trace', signalId],
    queryFn: () => liveService.getDecisionTrace(signalId as string),
    enabled,
    retry: false,
  })

  if (!signalId) {
    return <Empty description="完成决策后将在此展示决策过程档案" />
  }

  // 404（无过程档案）→ 专用文案；其余错误 → 通用错误文案。
  const status = (error as { response?: { status?: number } } | undefined)?.response?.status
  const isNotFound = isError && status === 404

  /**
   * 各分组内容的统一状态包装：把请求状态映射为对应展示。
   *
   * 按优先级判定：加载中显示 Spin；404（无过程档案）显示「暂无过程档案」；
   * 其它错误显示通用错误提示；正常则委托 {@link renderSection} 渲染该段内容。
   *
   * @param key - 当前分组对应的段标识，用于命中正常态时选择段内容
   * @returns 与当前请求状态匹配的内容节点
   */
  const wrapContent = (key: string): React.ReactNode => {
    if (isLoading) {
      return (
        <div style={{ textAlign: 'center', padding: 16 }}>
          <Spin />
        </div>
      )
    }
    if (isNotFound) {
      return <Empty description="暂无过程档案" />
    }
    if (isError) {
      return (
        <Alert
          type="error"
          showIcon
          message="决策过程档案加载失败"
          description="无法获取该决策的过程档案，请稍后重试。"
        />
      )
    }
    return renderSection(key, trace)
  }

  const items = SECTION_TITLES.map(({ key, title }) => ({
    key,
    label: title,
    children: wrapContent(key),
  }))

  return (
    <Collapse
      items={items}
      activeKey={activeKeys}
      onChange={(keys) => setActiveKeys(keys as string[])}
    />
  )
}

export default DecisionTracePanel
