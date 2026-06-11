// 本地聚合工作区：数据驱动的级联配置组件。
//
// 合约通过多选可搜索下拉从本地已有数据资源中选取（不再手填 TextArea）；
// 来源类型/来源周期/目标周期/时间范围/时段规则的可选项与默认值都依据所选合约
// 实际拥有的本地数据自动计算并联动刷新（见 design.md Components and Interfaces）。
//
// 所有联动计算均委托给 `../../utils/aggregation` 中的纯函数，组件本身只负责
// 受控状态、归一化副作用与渲染。

import React, { useEffect, useMemo, useState } from 'react'
import { Alert, App, Button, Card, Empty, Select, Space, Spin, Typography } from 'antd'
import { DatabaseOutlined, PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import type { AxiosError } from 'axios'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'

import { alphaService } from '../../api/alpha'
import DateRangeSelector from '../../components/DateRangeSelector'
import { useAvailableSymbols } from '../../hooks/useAvailableSymbols'
import type {
  AggregationConfig,
  InvalidDimension,
  SourceKind,
} from '../../utils/aggregation'
import {
  buildAggregateRequest,
  computeAvailableSourceKinds,
  computeCommonRange,
  computeSourceIntervalOptions,
  computeTargetIntervalOptions,
  defaultSourceKind,
  reconcileSelected,
  selectWorkspaceState,
  validateAggregation,
} from '../../utils/aggregation'
import type { DataResourceList } from '../../types/alpha'

const { Text } = Typography

export interface AggregationWorkspaceProps {
  /** 本地数据资源（来自页面级 useQuery，复用，不新增请求）。 */
  resources: DataResourceList | undefined
  /** 数据资源是否正在加载。 */
  isLoading: boolean
  /** 数据资源加载错误（任意类型，转为 hasError 处理）。 */
  error: unknown
  /** 聚合任务启动后回调，用于让页面设置 taskId（驱动 TaskStatusPanel）。 */
  onTaskStarted: (taskId: string) => void
  /** 错误态重试入口（通常为 queryClient.invalidateQueries / refetch）。 */
  onRetry?: () => void
  /** 嵌入其他面板时不再额外渲染外层 Card。 */
  embedded?: boolean
}

/** 无效维度 -> 定向中文提示（见 design.md Error Handling 表）。 */
const INVALID_MESSAGES: Record<InvalidDimension, string> = {
  'no-symbol': '请至少选择一个合约',
  'no-common-source': '所选合约无公共可用来源数据',
  'no-source-interval': '所选合约无可用的分钟级来源周期',
  'no-target': '该来源周期无可聚合的目标周期',
  'no-range-overlap': '所选时间范围与公共可用区间无重叠',
}

const SOURCE_KIND_LABELS: Record<SourceKind, string> = {
  bar: '原始K线',
  tick: '历史Tick',
}

const INTERVAL_LABELS: Record<string, string> = {
  '1m': '1分钟',
  '5m': '5分钟',
  '10m': '10分钟',
  '15m': '15分钟',
  '30m': '30分钟',
  '60m': '60分钟',
}

const formatIntervalLabel = (value: string) => INTERVAL_LABELS[value] || value

/** FastAPI 校验错误条目（422 时 detail 为其数组）。 */
interface ValidationErrorItem {
  loc?: (string | number)[]
  msg?: string
}

/**
 * 从任意错误中提取可安全渲染的字符串提示。
 *
 * 注意：FastAPI 422 的 `detail` 是 `{type, loc, msg, input}` 对象数组，
 * 直接交给 `message.error` 会被当作 React 子节点渲染而抛错（白屏）。
 * 这里将数组归并为可读字符串，对象/字符串分别处理。
 */
const getErrorMessage = (error: unknown, fallback: string): string => {
  const detail = (error as AxiosError<{ detail?: unknown }>)?.response?.data?.detail
  if (typeof detail === 'string' && detail) {
    return detail
  }
  if (Array.isArray(detail)) {
    const text = (detail as ValidationErrorItem[])
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.join('.') : ''
        return field ? `${field}: ${item.msg ?? ''}` : item.msg ?? ''
      })
      .filter(Boolean)
      .join('；')
    if (text) {
      return text
    }
  }
  return (error as AxiosError)?.message || fallback
}

const INITIAL_CONFIG: AggregationConfig = {
  selectedSymbols: [],
  sourceKind: null,
  sourceInterval: null,
  targetInterval: null,
  range: null,
  sessionProfile: 'cn_equity',
}

const AggregationWorkspace: React.FC<AggregationWorkspaceProps> = ({
  resources,
  isLoading,
  error,
  onTaskStarted,
  onRetry,
  embedded = false,
}) => {
  const { message } = App.useApp()
  const [config, setConfig] = useState<AggregationConfig>(INITIAL_CONFIG)
  const [submitting, setSubmitting] = useState(false)

  const availability = useAvailableSymbols(resources)

  const hasError = Boolean(error)
  const symbolCount = availability.size
  const workspaceState = selectWorkspaceState(isLoading, hasError, symbolCount)
  const renderShell = (children: React.ReactNode) => (
    embedded ? <>{children}</> : <Card title="本地聚合工作区">{children}</Card>
  )

  // 合约选择器选项（按代码升序）。
  const symbolOptions = useMemo(
    () =>
      [...availability.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([sym, meta]) => ({
          value: sym,
          label: sym,
          intervals: [...meta.intervals],
          dateRange: `${meta.start.slice(0, 10)} ~ ${meta.end.slice(0, 10)}`,
        })),
    [availability],
  )

  // 派生选项（全部委托纯函数）。
  const availableSourceKinds = useMemo(
    () => computeAvailableSourceKinds(config.selectedSymbols, availability),
    [config.selectedSymbols, availability],
  )

  const sourceIntervalOptions = useMemo(
    () => computeSourceIntervalOptions(config.selectedSymbols, availability),
    [config.selectedSymbols, availability],
  )

  const targetIntervalOptions = useMemo(
    () =>
      config.sourceKind === null
        ? []
        : computeTargetIntervalOptions(config.sourceKind, config.sourceInterval),
    [config.sourceKind, config.sourceInterval],
  )

  const commonRange = useMemo(
    () =>
      config.sourceKind === null
        ? null
        : computeCommonRange(
            config.selectedSymbols,
            availability,
            config.sourceKind,
            config.sourceInterval,
          ),
    [config.selectedSymbols, availability, config.sourceKind, config.sourceInterval],
  )

  // 归一化：默认来源类型 / 越界重置（Requirement 2.5, 2.6）。
  useEffect(() => {
    setConfig((c) => {
      if (c.sourceKind !== null && availableSourceKinds.has(c.sourceKind)) {
        return c
      }
      const next = defaultSourceKind(availableSourceKinds)
      return next === c.sourceKind ? c : { ...c, sourceKind: next }
    })
  }, [availableSourceKinds])

  // 归一化：来源周期（bar 取最细/越界重置，tick 清空）（Requirement 3.5, 3.6, 3.7）。
  useEffect(() => {
    setConfig((c) => {
      if (c.sourceKind === 'tick') {
        return c.sourceInterval === null ? c : { ...c, sourceInterval: null }
      }
      if (c.sourceKind === 'bar') {
        const next = reconcileSelected(c.sourceInterval, sourceIntervalOptions)
        return next === c.sourceInterval ? c : { ...c, sourceInterval: next }
      }
      return c
    })
  }, [sourceIntervalOptions, config.sourceKind])

  // 归一化：目标周期（最细/越界重置）（Requirement 4.5）。
  useEffect(() => {
    setConfig((c) => {
      const next = reconcileSelected(c.targetInterval, targetIntervalOptions)
      return next === c.targetInterval ? c : { ...c, targetInterval: next }
    })
  }, [targetIntervalOptions])

  // 归一化：时间范围默认填充为公共可用区间（Requirement 5.3）。
  useEffect(() => {
    setConfig((c) => {
      if (!commonRange) {
        return c.range === null ? c : { ...c, range: null }
      }
      const next: [string, string] = [commonRange.start, commonRange.end]
      if (c.range && c.range[0] === next[0] && c.range[1] === next[1]) {
        return c
      }
      return { ...c, range: next }
    })
  }, [commonRange])

  const handleSubmit = async () => {
    const validation = validateAggregation(config, availability)
    if (!validation.valid) {
      // 无效组合：不调用 aggregateData，保留配置，给出定向提示（Requirement 7.2）。
      message.warning(INVALID_MESSAGES[validation.reason as InvalidDimension])
      return
    }
    if (!commonRange) {
      // 防御性兜底：校验通过即应存在公共区间。
      message.warning(INVALID_MESSAGES['no-range-overlap'])
      return
    }

    const request = buildAggregateRequest(config, commonRange)
    setSubmitting(true)
    try {
      const result = await alphaService.aggregateData(request)
      onTaskStarted(result.task_id)
      message.success('派生周期聚合任务已启动')
    } catch (err) {
      message.error(getErrorMessage(err, '派生周期聚合任务启动失败'))
    } finally {
      setSubmitting(false)
    }
  }

  const rangeValue = useMemo<[Dayjs, Dayjs] | null>(
    () => (config.range ? [dayjs(config.range[0]), dayjs(config.range[1])] : null),
    [config.range],
  )

  // ---------------------------------------------------------------------------
  // 降级四态：互斥渲染（Requirement 8.1, 8.3, 8.4, 8.5、Property 11）。
  // ---------------------------------------------------------------------------

  if (workspaceState === 'loading') {
    return renderShell(
        <Space direction="vertical" align="center" style={{ width: '100%', padding: '24px 0' }}>
          <Spin />
          <Text type="secondary">正在读取数据资源...</Text>
        </Space>,
    )
  }

  if (workspaceState === 'error') {
    return renderShell(
        <Alert
          type="error"
          showIcon
          message="数据资源加载失败"
          description="无法获取本地数据资源，请重试。"
          action={
            onRetry ? (
              <Button size="small" icon={<ReloadOutlined />} onClick={onRetry}>
                重试
              </Button>
            ) : null
          }
        />,
    )
  }

  if (workspaceState === 'empty') {
    return renderShell(
      <Empty description="暂无本地数据，请先在上方准备本地数据（下载/导入）后再聚合" />,
    )
  }

  // ready 态：完整的数据驱动配置表单。
  const sourceKindOptions = (['bar', 'tick'] as SourceKind[]).map((kind) => ({
    value: kind,
    label: SOURCE_KIND_LABELS[kind],
    disabled: !availableSourceKinds.has(kind),
  }))

  const someSourceKindDisabled = sourceKindOptions.some((option) => option.disabled)
  const noCommonSource = availableSourceKinds.size === 0
  const noSourceInterval = config.sourceKind === 'bar' && sourceIntervalOptions.length === 0
  const noTarget = targetIntervalOptions.length === 0
  const noRangeOverlap = config.selectedSymbols.length > 0 && config.sourceKind !== null && commonRange === null

  const submitDisabled = noTarget

  return renderShell(
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Space direction="vertical" size={4} style={{ width: '100%' }}>
          <Text strong>合约</Text>
          <Select
            mode="multiple"
            showSearch
            allowClear
            optionFilterProp="label"
            placeholder="选择要聚合的合约（支持搜索）"
            notFoundContent="暂无本地数据"
            style={{ width: '100%' }}
            value={config.selectedSymbols}
            onChange={(values: string[]) =>
              setConfig((c) => ({ ...c, selectedSymbols: values }))
            }
            options={symbolOptions.map((sym) => ({ value: sym.value, label: sym.label }))}
            optionRender={(option) => {
              const meta = symbolOptions.find((s) => s.value === option.value)
              return (
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Space size={4}>
                    <DatabaseOutlined style={{ color: '#52c41a' }} />
                    <span>{option.label}</span>
                  </Space>
                  {meta ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {meta.intervals.join('/')} · {meta.dateRange}
                    </Text>
                  ) : null}
                </Space>
              )
            }}
          />
        </Space>

        <Space size={16} wrap align="start">
          <Space direction="vertical" size={4}>
            <Text strong>来源类型</Text>
            <Select<SourceKind>
              style={{ width: 140 }}
              placeholder="来源类型"
              value={config.sourceKind ?? undefined}
              onChange={(value) => setConfig((c) => ({ ...c, sourceKind: value }))}
              options={sourceKindOptions}
            />
          </Space>

          {config.sourceKind === 'bar' ? (
            <Space direction="vertical" size={4}>
              <Text strong>来源周期</Text>
              <Select
                style={{ width: 120 }}
                placeholder="来源周期"
                value={config.sourceInterval ?? undefined}
                onChange={(value) => setConfig((c) => ({ ...c, sourceInterval: value }))}
                options={sourceIntervalOptions.map((interval) => ({
                  value: interval,
                  label: formatIntervalLabel(interval),
                }))}
              />
            </Space>
          ) : null}

          <Space direction="vertical" size={4}>
            <Text strong>目标周期</Text>
            <Select
              style={{ width: 120 }}
              placeholder="目标周期"
              value={config.targetInterval ?? undefined}
              onChange={(value) => setConfig((c) => ({ ...c, targetInterval: value }))}
              options={targetIntervalOptions.map((interval) => ({
                value: interval,
                label: formatIntervalLabel(interval),
              }))}
            />
          </Space>

          <Space direction="vertical" size={4}>
            <Text strong>时段规则</Text>
            <Select
              style={{ width: 200 }}
              value={config.sessionProfile}
              onChange={(value) => setConfig((c) => ({ ...c, sessionProfile: value }))}
              options={[{ label: 'A股日内时段', value: 'cn_equity' }]}
            />
          </Space>
        </Space>

        <Space direction="vertical" size={4} style={{ width: '100%', maxWidth: 420 }}>
          <Text strong>时间范围</Text>
          <DateRangeSelector
            value={rangeValue}
            localRange={commonRange ? { start: commonRange.start, end: commonRange.end } : null}
            onChange={(value) => {
              if (!value) {
                setConfig((c) => ({ ...c, range: null }))
                return
              }
              setConfig((c) => ({
                ...c,
                range: [value[0].format('YYYY-MM-DD'), value[1].format('YYYY-MM-DD')],
              }))
            }}
          />
        </Space>

        {someSourceKindDisabled && !noCommonSource ? (
          <Alert
            type="info"
            showIcon
            message="部分来源类型在所选合约下无公共数据，已置为不可选。"
          />
        ) : null}
        {noCommonSource && config.selectedSymbols.length > 0 ? (
          <Alert type="warning" showIcon message={INVALID_MESSAGES['no-common-source']} />
        ) : null}
        {noSourceInterval ? (
          <Alert type="warning" showIcon message={INVALID_MESSAGES['no-source-interval']} />
        ) : null}
        {noTarget && config.sourceKind !== null && !noCommonSource && !noSourceInterval ? (
          <Alert type="warning" showIcon message={INVALID_MESSAGES['no-target']} />
        ) : null}
        {noRangeOverlap ? (
          <Alert type="warning" showIcon message={INVALID_MESSAGES['no-range-overlap']} />
        ) : null}

        <div>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            loading={submitting}
            disabled={submitDisabled}
            onClick={() => void handleSubmit()}
          >
            生成派生周期
          </Button>
        </div>
      </Space>,
  )
}

export default AggregationWorkspace
