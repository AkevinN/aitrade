import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Collapse,
  DatePicker,
  Descriptions,
  Drawer,
  Empty,
  InputNumber,
  List,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  Tooltip,
} from 'antd'
import { ExperimentOutlined, HistoryOutlined, InfoCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import { useMutation, useQuery } from '@tanstack/react-query'
import dayjs, { type Dayjs } from 'dayjs'

import { alphaService } from '../../api/alpha'
import type {
  ConfidenceLevel,
  GroupProfile,
  MetricBlock,
  ObservationGroup,
  SchemeSuggestion,
  SymbolProfileResponse,
} from '../../types/alpha'
import {
  buildProfilingRequest,
  confidenceStyle,
  formatMetricValue,
  isLowConfidenceItem,
  mapSuggestionToFormValues,
  metricHelp,
  shouldDropObservation,
} from '../../utils/profiling'

const { Text } = Typography

const BLOCK_LABEL: Record<MetricBlock['block'], string> = {
  data_quality: '数据质量',
  liquidity: '流动性',
  volatility: '波动性',
  predictability: '可预测性',
}

/**
 * 将置信度枚举值渲染为带 Tooltip 的 Ant Design Tag。
 *
 * @param confidence - 置信度枚举值。
 * @returns 带颜色和 Tooltip 说明的 Tag 节点。
 */
function confidenceTag(confidence: ConfidenceLevel) {
  const style = confidenceStyle(confidence)
  return (
    <Tooltip title={style.description}>
      <Tag color={style.color} style={{ cursor: 'help' }}>
        {style.text}
      </Tag>
    </Tooltip>
  )
}

/**
 * 从任意异常对象中提取可读的错误文本。
 *
 * @param error - 任意异常对象（通常为 AxiosError）。
 * @returns 后端 `detail` 字段、`message` 字段或兜底文案。
 */
function errorText(error: unknown): string {
  const maybe = error as { response?: { data?: { detail?: string } }; message?: string }
  return maybe.response?.data?.detail || maybe.message || '画像请求失败'
}

/**
 * 根据错误类型给出下一步操作提示文案。
 *
 * @param isArtifactError - 是否为 Artifact 查询错误（404）。
 * @returns 针对性的操作提示字符串。
 */
function errorNextStep(isArtifactError: boolean): string {
  return isArtifactError
    ? '请确认 artifact_id 是否完整，或重新运行画像并持久化后再查看。'
    : '请检查目标证券、as_of、回看天数和本地行情区间后重试。'
}

const summaryItemStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '6px 10px',
  border: '1px solid rgba(145, 202, 255, 0.16)',
  borderRadius: 6,
  background: '#111b26',
}

const summaryLabelStyle: React.CSSProperties = {
  color: '#8c8c8c',
}

const summaryValueStyle: React.CSSProperties = {
  color: '#d6e4ff',
  fontWeight: 600,
}

const profileSectionStyle: React.CSSProperties = {
  border: '1px solid rgba(145, 202, 255, 0.12)',
  borderRadius: 6,
  padding: 12,
  background: 'rgba(17, 27, 38, 0.68)',
}

/**
 * 单个画像指标块（数据质量/流动性/波动性/可预测性）的展示组件。
 *
 * @param block - 待渲染的 {@link MetricBlock} 数据。
 */
export const MetricBlockView: React.FC<{ block: MetricBlock }> = ({ block }) => {
  return (
    <section style={profileSectionStyle}>
      <Space direction="vertical" size={8} style={{ width: '100%' }}>
        <Space wrap>
          <Text strong>{BLOCK_LABEL[block.block]}</Text>
          {block.level ? <Tag>{block.level}</Tag> : null}
        </Space>
        <List
          size="small"
          dataSource={block.metrics}
          renderItem={(metric) => {
            const style = confidenceStyle(metric.confidence)
            const help = metricHelp(metric.key)
            return (
              <List.Item>
                <Space direction="vertical" size={2} style={{ width: '100%' }}>
                  <Space wrap style={{ justifyContent: 'space-between', width: '100%' }}>
                    <Space size={6} wrap>
                      <Text strong type={style.weak ? 'secondary' : undefined}>
                        {help.label}
                      </Text>
                      <Text code type="secondary">
                        {metric.key}
                      </Text>
                      <Tooltip title={help.description}>
                        <InfoCircleOutlined style={{ color: '#8c8c8c', cursor: 'help' }} />
                      </Tooltip>
                    </Space>
                    <Space size={4}>
                      {confidenceTag(metric.confidence)}
                      <Text type={style.weak ? 'secondary' : undefined}>
                        {formatMetricValue(metric.value, metric.key)}
                      </Text>
                    </Space>
                  </Space>
                  {metric.note ? (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {metric.note}
                    </Text>
                  ) : null}
                </Space>
              </List.Item>
            )
          }}
        />
      </Space>
    </section>
  )
}

/**
 * 画像参数建议展示组件：渲染「方案建议」列表并提供「一键应用」按钮。
 *
 * @param suggestion - 画像返回的参数建议；`null`/`undefined` 时不渲染。
 * @param onApply - 点击「应用建议」时的回调，传入可映射字段值和未映射条目数。
 */
export const SuggestionView: React.FC<{
  suggestion: SchemeSuggestion | null | undefined
  onApply: (values: Record<string, unknown>, unmappedCount: number) => void
}> = ({ suggestion, onApply }) => {
  if (!suggestion) {
    return null
  }

  const mapped = mapSuggestionToFormValues(suggestion)
  const mappedCount = Object.keys(mapped.values).length

  /**
   * 点击「填充到训练表单」按钮时的处理器。
   *
   * 把已映射的建议字段值和未映射条目数透传给 `onApply` 回调，由父组件决定如何回填表单。
   */
  const handleApply = () => {
    onApply(mapped.values, mapped.unmapped.length)
  }

  return (
    <section style={profileSectionStyle}>
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Space wrap style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space wrap>
            <Text strong>方案建议</Text>
            <Tag color="gold">草稿 / 待确认</Tag>
            {suggestion.degraded ? <Tag color="orange">已降级</Tag> : null}
          </Space>
          <Button size="small" onClick={handleApply}>
            {mappedCount > 0 ? `填充 ${mappedCount} 项到训练表单` : '填充到训练表单'}
          </Button>
        </Space>
        <Text type="secondary" style={{ fontSize: 12 }}>
          可直接回填 {mappedCount} 项；{mapped.unmapped.length} 项需人工确认或不适用于当前训练表单。
        </Text>

        {suggestion.degraded ? (
          <Alert
            type="warning"
            showIcon
            message="样本或置信度不足，仅展示前置建议"
            description={suggestion.note || '低置信度下不会自动填充强超参数。'}
          />
        ) : suggestion.note ? (
          <Alert type="info" showIcon message={suggestion.note} />
        ) : null}

        <List
          size="small"
          dataSource={suggestion.items}
          renderItem={(item) => (
            <List.Item>
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                <Space wrap>
                  <Text code>{item.field}</Text>
                  <Text>{String(item.value)}</Text>
                  {confidenceTag(item.based_on_confidence)}
                  {isLowConfidenceItem(item) ? <Tag color="orange">低置信</Tag> : null}
                </Space>
                <Text type="secondary">{item.reason}</Text>
              </Space>
            </List.Item>
          )}
        />
      </Space>
    </section>
  )
}

/**
 * 观测组关联性表格组件：展示每个观测标的与目标标的的相关系数，并给出保留/剔除建议。
 *
 * @param groupProfile - 观测组关联性数据；`null`/`undefined` 或成员列表为空时不渲染。
 */
export const GroupProfileView: React.FC<{ groupProfile: GroupProfile | null | undefined }> = ({
  groupProfile,
}) => {
  if (!groupProfile || groupProfile.members.length === 0) {
    return null
  }

  const data = groupProfile.members.map((member) => {
    const corr = groupProfile.correlation_summary[member]
    return {
      member,
      correlation: corr,
      drop: shouldDropObservation(groupProfile.alignment_coverage, corr),
    }
  })

  return (
    <section style={profileSectionStyle}>
      <Space direction="vertical" size={8} style={{ width: '100%' }}>
        <Space wrap style={{ justifyContent: 'space-between', width: '100%' }}>
          <Text strong>观测组关联性</Text>
          <Tag color={groupProfile.alignment_coverage >= 0.8 ? 'green' : 'orange'}>
            整体对齐覆盖率 {formatMetricValue(groupProfile.alignment_coverage, 'alignment_coverage')}
          </Tag>
        </Space>
        <Table
          size="small"
          rowKey="member"
          pagination={false}
          dataSource={data}
          columns={[
            { title: '观测标的', dataIndex: 'member' },
            {
              title: '相关性',
              dataIndex: 'correlation',
              render: (value?: number) => (value === undefined ? '-' : value.toFixed(3)),
            },
            {
              title: '反馈',
              dataIndex: 'drop',
              render: (drop: boolean) =>
                drop ? <Tag color="orange">建议剔除 / 可能为噪声</Tag> : <Tag color="green">保留观察</Tag>,
            },
          ]}
        />
      </Space>
    </section>
  )
}

/**
 * {@link ProfilingPanel} 组件 props。
 */
export interface ProfilingPanelProps {
  /** 是否展开 Drawer。 */
  open: boolean
  /** 关闭 Drawer 的回调。 */
  onClose: () => void
  /** 待画像的目标合约代码。 */
  targetSymbol: string
  /** K 线周期（如 `d`、`30m`）。 */
  interval: string
  /** 默认画像基准时刻（dayjs 实例）。 */
  defaultAsOf: Dayjs
  /** 观测分组列表，用于收集 observation_symbols。 */
  observationGroups: ObservationGroup[]
  /**
   * 应用参数建议的回调；调用方据此批量更新 Form 字段。
   *
   * @param values - 已映射的字段值。
   * @param unmappedCount - 未能自动映射的建议条目数。
   */
  onApplySuggestion: (values: Record<string, unknown>, unmappedCount: number) => void
  /**
   * 画像结果变化回调（运行新画像或切换历史时均触发）。
   *
   * @param profile - 最新的画像响应。
   * @param historical - 是否为历史 Artifact 查询（非当次运行）。
   */
  onResultChange?: (profile: SymbolProfileResponse, historical: boolean) => void
}

/**
 * 品种画像侧边 Drawer：提供运行画像、查看历史 Artifact、展示指标块与参数建议的一体化面板。
 *
 * 常在 CNNTrain 页面中使用，帮助用户在配置训练方案前评估目标标的的画像质量。
 */
const ProfilingPanel: React.FC<ProfilingPanelProps> = ({
  open,
  onClose,
  targetSymbol,
  interval,
  defaultAsOf,
  observationGroups,
  onApplySuggestion,
  onResultChange,
}) => {
  const [asOf, setAsOf] = useState<Dayjs>(defaultAsOf)
  const [lookbackDays, setLookbackDays] = useState(250)
  const [withSuggestion, setWithSuggestion] = useState(true)
  const [artifactId, setArtifactId] = useState('')
  const [result, setResult] = useState<SymbolProfileResponse | null>(null)
  const [historical, setHistorical] = useState(false)

  const artifactListQuery = useQuery({
    queryKey: ['alpha-profiling-artifacts'],
    queryFn: () => alphaService.listProfilingArtifacts(),
    enabled: open,
  })

  useEffect(() => {
    if (open) {
      setAsOf(defaultAsOf)
      setHistorical(false)
    }
  }, [defaultAsOf, open])

  const runMutation = useMutation({
    mutationFn: () =>
      alphaService.runProfiling(
        buildProfilingRequest({
          targetSymbol,
          interval,
          asOf,
          lookbackDays,
          observationGroups,
          withSuggestion,
        }),
      ),
    onSuccess: (profile) => {
      setResult(profile)
      setHistorical(false)
      onResultChange?.(profile, false)
      if (profile.artifact_id) {
        setArtifactId(profile.artifact_id)
        void artifactListQuery.refetch()
      }
    },
  })

  const artifactMutation = useMutation({
    mutationFn: (id: string) => alphaService.getProfilingArtifact(id),
    onSuccess: (profile) => {
      setResult(profile)
      setHistorical(true)
      onResultChange?.(profile, true)
    },
  })

  const loading = runMutation.isPending || artifactMutation.isPending
  const error = runMutation.error || artifactMutation.error

  const observationCount = useMemo(
    () => new Set(observationGroups.flatMap((group) => group.symbols)).size,
    [observationGroups],
  )
  const artifactOptions = useMemo(() => {
    const ids = new Set(artifactListQuery.data || [])
    if (result?.artifact_id) {
      ids.add(result.artifact_id)
    }
    return [...ids].sort().reverse().map((id) => ({ label: id, value: id }))
  }, [artifactListQuery.data, result?.artifact_id])

  useEffect(() => {
    if (open && !artifactId && artifactOptions.length > 0) {
      setArtifactId(artifactOptions[0].value)
    }
  }, [artifactId, artifactOptions, open])

  /**
   * 触发「按标的现算」的画像评估。
   *
   * 缺少 `targetSymbol` 时直接清空结果并退出；否则先重置两条 mutation 与已有结果，
   * 再发起 `runMutation` 重新计算当前标的的画像。
   */
  const triggerRun = () => {
    if (!targetSymbol) {
      setResult(null)
      return
    }
    runMutation.reset()
    artifactMutation.reset()
    setResult(null)
    runMutation.mutate()
  }

  /**
   * 触发「从已有产物回放」的画像评估。
   *
   * 读取并去空白后的 `artifactId`，为空则不做任何动作；否则先重置两条 mutation 与已有结果，
   * 再以该产物 id 发起 `artifactMutation`，复现历史画像而不重新现算。
   */
  const triggerArtifact = () => {
    const id = artifactId.trim()
    if (!id) {
      return
    }
    runMutation.reset()
    artifactMutation.reset()
    setResult(null)
    artifactMutation.mutate(id)
  }

  return (
    <Drawer
      title="标的画像评估"
      open={open}
      onClose={onClose}
      width={780}
      destroyOnHidden
    >
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <Alert
          type="info"
          showIcon
          message="时间隔离：画像仅基于 as_of 之前的本地行情计算"
          description="建议保持草稿状态，填充训练表单后仍需人工确认。"
        />
        <Alert
          type="success"
          showIcon
          message="置信度说明"
          description="高/中：可作为主要或辅助参考；低：谨慎参考；样本不足：不展示数值，也不应用于强参数建议。"
        />

        <section>
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Space wrap>
              <Text strong>目标标的</Text>
              {targetSymbol ? <Tag color="blue">{targetSymbol}</Tag> : <Tag color="red">未选择</Tag>}
              <Tag>{interval}</Tag>
              <Tag color="purple">{observationCount} 个观测标的</Tag>
            </Space>
            <Space wrap align="end">
              <Space direction="vertical" size={4}>
                <Text type="secondary">as_of</Text>
                <DatePicker
                  showTime
                  value={asOf}
                  onChange={(value) => setAsOf(value || dayjs())}
                  style={{ width: 210 }}
                />
              </Space>
              <Space direction="vertical" size={4}>
                <Text type="secondary">回看天数</Text>
                <InputNumber
                  min={1}
                  max={3000}
                  value={lookbackDays}
                  onChange={(value) => setLookbackDays(Number(value || 1))}
                  style={{ width: 120 }}
                />
              </Space>
              <Space direction="vertical" size={4}>
                <Text type="secondary">生成建议</Text>
                <Switch checked={withSuggestion} onChange={setWithSuggestion} />
              </Space>
              <Button
                type="primary"
                icon={<ExperimentOutlined />}
                loading={runMutation.isPending}
                disabled={loading || !targetSymbol}
                onClick={triggerRun}
              >
                开始评估
              </Button>
            </Space>
          </Space>
        </section>

        <Collapse
          size="small"
          items={[
            {
              key: 'history',
              label: '查看历史画像',
              children: (
                <Space direction="vertical" size={8} style={{ width: '100%' }}>
                  <Space.Compact style={{ width: '100%' }}>
                    <Select
                      showSearch
                      allowClear
                      placeholder="选择已保存的画像"
                      value={artifactId || undefined}
                      options={artifactOptions}
                      loading={artifactListQuery.isFetching}
                      notFoundContent={artifactListQuery.isFetching ? '正在加载历史画像' : '暂无已保存画像'}
                      onChange={(value) => setArtifactId(value || '')}
                      onSelect={(value) => setArtifactId(value)}
                      style={{ width: '100%' }}
                    />
                    <Button
                      icon={<ReloadOutlined />}
                      loading={artifactListQuery.isFetching}
                      onClick={() => void artifactListQuery.refetch()}
                    >
                      刷新
                    </Button>
                    <Button
                      icon={<HistoryOutlined />}
                      loading={artifactMutation.isPending}
                      disabled={loading || !artifactId.trim()}
                      onClick={triggerArtifact}
                    >
                      查看历史
                    </Button>
                  </Space.Compact>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    评估成功并保存后会自动出现在这里，也可从“最近画像”的 artifact_id 复制确认。
                  </Text>
                </Space>
              ),
            },
          ]}
        />

        {error ? (
          <Alert
            type="error"
            showIcon
            message={artifactMutation.error ? '历史画像读取失败' : '画像评估失败'}
            description={
              <Space direction="vertical" size={2}>
                <Text>{errorText(error)}</Text>
                <Text type="secondary">{errorNextStep(Boolean(artifactMutation.error))}</Text>
              </Space>
            }
            action={
              <Button size="small" icon={<ReloadOutlined />} onClick={artifactMutation.error ? triggerArtifact : triggerRun}>
                重试
              </Button>
            }
          />
        ) : null}

        {loading ? <Alert type="info" showIcon message="正在计算画像..." /> : null}

        {!loading && !error && !result ? (
          <Empty description="设置 as_of 与回看窗口后开始评估" />
        ) : null}

        {result ? (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            {historical ? (
              <Alert type="info" showIcon message={`历史画像 · 创建于 ${result.created_at}`} />
            ) : null}
            <Space wrap>
              <span style={summaryItemStyle}>
                <span style={summaryLabelStyle}>综合置信度</span>
                {confidenceTag(result.overall_confidence)}
              </span>
              <span style={summaryItemStyle}>
                <span style={summaryLabelStyle}>有效 bar</span>
                <span style={summaryValueStyle}>{result.input.effective_bar_count}</span>
              </span>
              <span style={summaryItemStyle}>
                <span style={summaryLabelStyle}>实际右边界</span>
                <span style={summaryValueStyle}>{result.input.effective_right_bound || '-'}</span>
              </span>
              <span style={summaryItemStyle}>
                <span style={summaryLabelStyle}>规则</span>
                <span style={summaryValueStyle}>{result.input.rules_id}</span>
              </span>
            </Space>
            {result.artifact_id ? (
              <Alert
                type="info"
                showIcon
                message="画像已保存，可用于历史查询"
                description={
                  <Space wrap>
                    <Text code>{result.artifact_id}</Text>
                    <Button size="small" onClick={() => setArtifactId(result.artifact_id || '')}>
                      填入历史查询
                    </Button>
                  </Space>
                }
              />
            ) : null}
            <Descriptions size="small" bordered column={2}>
              <Descriptions.Item label="综合置信度">
                {confidenceTag(result.overall_confidence)}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {result.available ? <Tag color="green">可用</Tag> : <Tag color="orange">数据不可用</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="as_of">{result.input.as_of}</Descriptions.Item>
              <Descriptions.Item label="实际右边界">
                {result.input.effective_right_bound || '-'}
              </Descriptions.Item>
              <Descriptions.Item label="有效 bar 数">{result.input.effective_bar_count}</Descriptions.Item>
              <Descriptions.Item label="规则">{result.input.rules_id}</Descriptions.Item>
            </Descriptions>

            {!result.available ? (
              <Alert
                type="warning"
                showIcon
                message="画像数据不可用"
                description={result.unavailable_reason || '窗口内没有可用本地行情。'}
              />
            ) : (
              <>
                {result.blocks.map((block) => (
                  <MetricBlockView key={block.block} block={block} />
                ))}
                <SuggestionView suggestion={result.suggestion} onApply={onApplySuggestion} />
                <GroupProfileView groupProfile={result.group_profile} />
              </>
            )}
          </Space>
        ) : null}
      </Space>
    </Drawer>
  )
}

export default ProfilingPanel
