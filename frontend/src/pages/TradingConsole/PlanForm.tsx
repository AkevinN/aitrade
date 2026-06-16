import React from 'react'
import {
  Button,
  Col,
  Divider,
  Form,
  Input,
  InputNumber,
  Radio,
  Row,
  Select,
  Space,
  Switch,
  TimePicker,
  Typography,
} from 'antd'
import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs, { type Dayjs } from 'dayjs'

import { cnnService } from '../../api/cnn'
import { strategyService } from '../../api/strategy'
import type { CNNModelInfo } from '../../types/cnn'
import type { SignalSourceInfo } from '../../types/strategy'
import type {
  NotifyChannel,
  TradingPlan,
  TradingPlanRequest,
} from '../../types/live'
import { barFreqLabel, barFreqOfInterval, isIntradayBarFreq } from '../../utils/barFreq'

const { Text } = Typography

const DATA_SOURCE_OPTIONS = [
  { label: '接口拉取（Tushare / 增量）', value: 'pull' },
  { label: '上传文件（CSV / parquet）', value: 'upload' },
]

const NOTIFY_CHANNEL_OPTIONS: { label: string; value: NotifyChannel }[] = [
  { label: '钉钉机器人', value: 'dingtalk' },
  { label: '企业微信', value: 'wecom' },
  { label: 'Server酱', value: 'serverchan' },
  { label: '通用 Webhook', value: 'webhook' },
]

const TRIGGER_SCHEDULE_OPTIONS = [
  { label: '每交易日', value: 'daily' },
  { label: '每周首个交易日', value: 'weekly_first' },
  { label: '每月首个交易日', value: 'monthly_first' },
]

/**
 * 把 "HH:mm" 字符串解析为 Dayjs 对象（不依赖 customParseFormat 插件）。
 *
 * 直接在今天的日期上设置时分秒，供 TimePicker 回显使用。
 *
 * @param value - "HH:mm" 格式的时间字符串，如 "15:05"
 * @returns 带有指定时分的 Dayjs 对象（秒/毫秒清零）
 */
const parseHHmm = (value: string): Dayjs => {
  const [hh, mm] = value.split(':').map((x) => Number(x) || 0)
  return dayjs().hour(hh).minute(mm).second(0).millisecond(0)
}

/**
 * 交易计划表单的内部字段模型。
 *
 * 同时承载 cnn / rule 两种策略的输入：cnn 模式只用上半部分字段，rule 模式额外用末尾的
 * signal_* / trigger_schedule / portfolio_id 字段；提交时由 handleSubmit 按 strategy_type
 * 分支裁剪并组装为 TradingPlanRequest。注意这是表单态结构（trigger_times 为 Dayjs），
 * 与落库的 TradingPlanRequest（trigger_times 为 "HH:mm" 字符串）不同。
 */
interface PlanFormValues {
  /** 计划名称；提交时去除首尾空白。 */
  name: string
  /** 策略类型，决定显示哪组字段：cnn 为 CNN 决策，rule 为规则调仓。 */
  strategy_type: 'cnn' | 'rule'
  /** CNN 模型名（cnn 模式）；rule 模式提交时置空串。 */
  model: string
  /** 目标标的代码，如 "000001.SZSE"（cnn 模式）；rule 模式提交时置空串。 */
  vt_symbol: string
  /** 方案名，如 "eod_buy_v1"（cnn 模式）；rule 模式提交时置空串。 */
  scheme: string
  /** 每日唤醒时刻列表，表单态为 Dayjs；提交时去重排序后格式化为 "HH:mm"。 */
  trigger_times: Dayjs[]
  /** 推送通道名列表；仅通道名，凭证由后端环境变量管理（不在前端录入）。 */
  notify_channels: NotifyChannel[]
  /** 数据来源：pull 为接口拉取，upload 为上传文件。 */
  data_source: 'upload' | 'pull'
  /** 是否启用自动调度；启用后调度器在决策时点触发，仅提醒不下单。 */
  enabled: boolean
  /** 买入概率阈值，取值 [0, 1]（cnn 模式）；rule 模式提交时为 0。 */
  buy_threshold: number
  /** 目标仓位比例，取值 [0, 1]（cnn 模式）；rule 模式提交时为 0。 */
  position_ratio: number
  /** 最小成交手数，下界 1（cnn 模式）；rule 模式提交时为 0。 */
  min_volume: number
  /** 模型版本号；为空串表示用模型默认版本。隐藏字段。 */
  model_version: string
  /** 组合总资金，单位元（cnn 模式）。 */
  portfolio_value: number
  /** 当前总持仓市值，单位元（cnn 模式）。 */
  total_position_value: number
  /** 目标标的当前持仓数量，单位股（cnn 模式）。 */
  current_position: number
  /** 目标标的当前持仓市值，单位元（cnn 模式）。 */
  current_symbol_value: number
  // rule 模式字段
  /** 信号源名（rule 模式）；对应后端注册的 SignalSource。 */
  signal_source: string
  /** 触发周期（rule 模式）：每交易日 / 每周首个交易日 / 每月首个交易日。 */
  trigger_schedule: 'daily' | 'weekly_first' | 'monthly_first'
  /** 持仓组合标识符（rule 模式）；与后端账本关联，提交时去除首尾空白。 */
  portfolio_id: string
  /** 股票池文本（rule 模式）；按换行/逗号/空白分割成代码列表后写入 signal_params.universe。 */
  signal_universe: string
  /** 最大持仓数量（rule 模式）；写入 plan.portfolio.top_k，后端 _trigger_plan 从此读取。 */
  signal_top_k: number
}

/** 交易计划表单的 Props。 */
interface PlanFormProps {
  /** 编辑模式下的初始计划；为空表示新建。 */
  initialPlan?: TradingPlan | null
  /** 提交回调，组装好的请求体上抛给父级（由父级 mutation 落库）。 */
  onSubmit: (req: TradingPlanRequest) => void
  /** 取消编辑。 */
  onCancel?: () => void
  /** 提交中状态（由父级 mutation.isPending 传入，用于禁用提交按钮）。 */
  submitting?: boolean
}

/**
 * 交易计划表单（v2）。支持两种策略类型：
 *
 * - **cnn 模式**：复用决策配置字段（model/vt_symbol/buy_threshold/portfolio 快照等），
 *   与 v1 行为完全一致，既有测试零回归。
 * - **rule 模式**：隐藏 CNN 专属字段；显示信号源（signal_source）、触发周期
 *   （trigger_schedule）、组合 ID（portfolio_id）、股票池（signal_params.universe）
 *   和 top_k（写入 portfolio.top_k——后端 _trigger_plan 从此取，见注释）；bar_freq 固定 "1d"。
 *
 * 安全约束（Req 9.4 / 9.5）：通知通道仅选择「通道名」，凭证由后端环境变量管理，
 * 前端永不录入/展示任何 webhook/secret/token。
 */
const PlanForm: React.FC<PlanFormProps> = ({ initialPlan, onSubmit, onCancel, submitting }) => {
  const [form] = Form.useForm<PlanFormValues>()

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })

  const { data: sources, isLoading: sourcesLoading } = useQuery({
    queryKey: ['strategy-sources'],
    queryFn: () => strategyService.listSources(),
  })

  // 监听 strategy_type 切换
  const strategyType = Form.useWatch('strategy_type', form) ?? 'cnn'
  const isRule = strategyType === 'rule'

  // bar_freq 派生（cnn 模式）：所选模型的训练间隔 → bar_freq；rule 模式固定 "1d"
  const selectedModel = Form.useWatch('model', form)
  const pickedMeta = (models || []).find((m: CNNModelInfo) => m.name === selectedModel)
  const barFreq = isRule
    ? '1d'
    : (pickedMeta
        ? barFreqOfInterval(pickedMeta.input_interval)
        : initialPlan?.bar_freq || '1d')
  const intraday = !isRule && isIntradayBarFreq(barFreq)

  React.useEffect(() => {
    if (!initialPlan) {
      form.resetFields()
      return
    }
    const st = initialPlan.strategy_type ?? 'cnn'
    const universeRaw = initialPlan.signal_params?.universe
    const universeStr = Array.isArray(universeRaw)
      ? (universeRaw as string[]).join('\n')
      : typeof universeRaw === 'string'
        ? universeRaw
        : ''
    form.setFieldsValue({
      name: initialPlan.name,
      strategy_type: st,
      model: initialPlan.model,
      vt_symbol: initialPlan.vt_symbol,
      scheme: initialPlan.scheme,
      trigger_times: (initialPlan.trigger_times?.length
        ? initialPlan.trigger_times
        : ['15:05']
      ).map(parseHHmm),
      notify_channels: initialPlan.notify_channels,
      data_source: initialPlan.data_source,
      enabled: initialPlan.enabled,
      buy_threshold: initialPlan.buy_threshold,
      position_ratio: initialPlan.position_ratio,
      min_volume: initialPlan.min_volume,
      model_version: initialPlan.model_version,
      portfolio_value: initialPlan.portfolio.portfolio_value,
      total_position_value: initialPlan.portfolio.total_position_value ?? 0,
      current_position: initialPlan.portfolio.current_position ?? 0,
      current_symbol_value: initialPlan.portfolio.current_symbol_value ?? 0,
      // rule 模式字段
      signal_source: initialPlan.signal_source ?? '',
      trigger_schedule: initialPlan.trigger_schedule ?? 'daily',
      portfolio_id: initialPlan.portfolio_id ?? '',
      signal_universe: universeStr,
      signal_top_k: typeof initialPlan.portfolio === 'object'
        ? ((initialPlan.portfolio as unknown as Record<string, unknown>).top_k as number | undefined) ?? 10
        : 10,
    })
  }, [form, initialPlan])

  /**
   * 表单提交处理：校验字段后按 strategy_type 分支组装请求体并上抛给父级 onSubmit。
   *
   * - **rule 模式**：裁剪 CNN 专属字段；top_k 写入 portfolio.top_k（后端隐晦约定）；
   *   universe 文本按空白/逗号分割为列表。
   * - **cnn 模式**：原有逻辑不变，日内模式（isIntradayBarFreq）不传 trigger_times。
   */
  const handleSubmit = async () => {
    let values: PlanFormValues
    try {
      values = await form.validateFields()
    } catch {
      return
    }

    if (isRule) {
      // rule 模式：裁剪 CNN 专属字段，添加 rule 字段
      // 注意：top_k 写入 portfolio.top_k——后端 _trigger_plan 从 plan.portfolio.top_k 取
      const universeList = (values.signal_universe || '')
        .split(/[\n,，\s]+/)
        .map((s) => s.trim())
        .filter(Boolean)
      const req: TradingPlanRequest = {
        name: values.name.trim(),
        model: '',
        vt_symbol: '',
        scheme: '',
        bar_freq: '1d',
        trigger_times: Array.from(new Set((values.trigger_times || []).map((d) => d.format('HH:mm')))).sort(),
        notify_channels: values.notify_channels || [],
        data_source: values.data_source,
        enabled: values.enabled,
        buy_threshold: 0,
        position_ratio: 0,
        min_volume: 0,
        model_version: '',
        portfolio: {
          portfolio_value: 0,
          // 后端 _trigger_plan 从 plan.portfolio.top_k 取 top_k 参数（隐晦约定）
          ...(values.signal_top_k ? { top_k: values.signal_top_k } as unknown as Record<string, number> : {}),
        } as typeof req.portfolio,
        strategy_type: 'rule',
        signal_source: values.signal_source,
        signal_params: { universe: universeList },
        trigger_schedule: values.trigger_schedule,
        portfolio_id: values.portfolio_id?.trim(),
      }
      onSubmit(req)
    } else {
      // cnn 模式：原有逻辑不变
      const req: TradingPlanRequest = {
        name: values.name.trim(),
        model: values.model,
        vt_symbol: values.vt_symbol.trim(),
        scheme: values.scheme.trim(),
        trigger_times: intraday
          ? []
          : Array.from(new Set((values.trigger_times || []).map((d) => d.format('HH:mm')))).sort(),
        bar_freq: barFreq,
        notify_channels: values.notify_channels || [],
        data_source: values.data_source,
        enabled: values.enabled,
        buy_threshold: values.buy_threshold,
        position_ratio: values.position_ratio,
        min_volume: values.min_volume,
        model_version: values.model_version || '',
        portfolio: {
          portfolio_value: values.portfolio_value,
          total_position_value: values.total_position_value,
          current_position: values.current_position,
          current_symbol_value: values.current_symbol_value,
        },
        strategy_type: 'cnn',
      }
      onSubmit(req)
    }
  }

  return (
    <Form<PlanFormValues>
      form={form}
      layout="vertical"
      initialValues={{
        name: '',
        strategy_type: 'cnn',
        vt_symbol: '',
        scheme: 'eod_buy_v1',
        trigger_times: [parseHHmm('15:05')],
        notify_channels: [],
        data_source: 'pull',
        enabled: false,
        buy_threshold: 0.6,
        position_ratio: 0.95,
        min_volume: 100,
        model_version: '',
        portfolio_value: 1000000,
        total_position_value: 0,
        current_position: 0,
        current_symbol_value: 0,
        trigger_schedule: 'daily',
        portfolio_id: '',
        signal_universe: '',
        signal_top_k: 10,
      }}
    >
      <Form.Item name="model_version" hidden>
        <Input />
      </Form.Item>

      <Form.Item label="计划名称" name="name" rules={[{ required: true, message: '请输入计划名称' }]}>
        <Input placeholder="如：平安银行尾盘买入计划" allowClear />
      </Form.Item>

      <Form.Item label="策略类型" name="strategy_type">
        <Radio.Group
          options={[
            { label: 'CNN 决策', value: 'cnn' },
            { label: '规则调仓', value: 'rule' },
          ]}
          optionType="button"
          buttonStyle="solid"
        />
      </Form.Item>

      {/* ===================== CNN 模式字段 ===================== */}
      {!isRule && (
        <>
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item
                label="CNN 模型"
                name="model"
                rules={[{ required: true, message: '请选择模型' }]}
                extra="决策 bar 频率由模型训练间隔自动派生，不可手动更改。"
              >
                <Select
                  showSearch
                  loading={modelsLoading}
                  placeholder="选择模型"
                  optionFilterProp="label"
                  options={(models || []).map((m: CNNModelInfo) => ({
                    value: m.name,
                    label: `${m.name}（${m.input_interval || 'd'}）`,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="目标标的" name="vt_symbol" rules={[{ required: true, message: '请输入标的' }]}>
                <Input placeholder="000001.SZSE" allowClear />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col span={12}>
              <Form.Item label="方案" name="scheme" rules={[{ required: true, message: '请输入方案名' }]}>
                <Input placeholder="eod_buy_v1" allowClear />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="数据源" name="data_source" rules={[{ required: true }]}>
                <Select options={DATA_SOURCE_OPTIONS} />
              </Form.Item>
            </Col>
          </Row>
        </>
      )}

      {/* ===================== 规则调仓模式字段 ===================== */}
      {isRule && (
        <>
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item
                label="信号源"
                name="signal_source"
                rules={[{ required: true, message: '请选择信号源' }]}
              >
                <Select
                  showSearch
                  loading={sourcesLoading}
                  placeholder="选择信号源"
                  optionFilterProp="label"
                  options={(sources || []).map((s: SignalSourceInfo) => ({
                    value: s.name,
                    label: s.description ? `${s.name}（${s.description}）` : s.name,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="触发周期"
                name="trigger_schedule"
                rules={[{ required: true, message: '请选择触发周期' }]}
              >
                <Select options={TRIGGER_SCHEDULE_OPTIONS} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            label="组合 ID"
            name="portfolio_id"
            rules={[{ required: true, message: '请输入组合 ID' }]}
            extra="与后端账本关联的持仓组合标识符。"
          >
            <Input placeholder="portfolio-001" allowClear />
          </Form.Item>

          <Form.Item
            label="股票池（universe）"
            name="signal_universe"
            extra="每行（或逗号分隔）一个标的代码，写入 signal_params.universe。"
          >
            <Input.TextArea
              rows={4}
              placeholder={"000001.SZSE\n000002.SZSE\n600000.SSE"}
            />
          </Form.Item>

          <Form.Item
            label="持仓数量上限（top_k）"
            name="signal_top_k"
            extra="最多持有几只；写入 plan.portfolio.top_k——后端 _trigger_plan 从此读取（隐晦约定）。"
          >
            <InputNumber min={1} step={1} style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item
            label="唤醒时刻（可配置多个）"
            required
            extra="在触发日的几点产出调仓建议；调度器以 trigger_times 确定当日触发时刻（与 CNN 日频计划相同路径）。"
          >
            <Form.List
              name="trigger_times"
              rules={[
                {
                  validator: async (_, value: Dayjs[] | undefined) => {
                    if (!value || value.length < 1) {
                      return Promise.reject(new Error('至少配置一个唤醒时刻'))
                    }
                  },
                },
              ]}
            >
              {(fields, { add, remove }, { errors }) => (
                <>
                  {fields.map((field) => (
                    <Space key={field.key} align="baseline" style={{ display: 'flex', marginBottom: 8 }}>
                      <Form.Item
                        {...field}
                        rules={[{ required: true, message: '请选择唤醒时刻' }]}
                        noStyle
                      >
                        <TimePicker format="HH:mm" allowClear={false} minuteStep={5} />
                      </Form.Item>
                      {fields.length > 1 ? (
                        <MinusCircleOutlined
                          aria-label="删除时刻"
                          onClick={() => remove(field.name)}
                        />
                      ) : null}
                    </Space>
                  ))}
                  <Button
                    type="dashed"
                    onClick={() => add(parseHHmm('15:05'))}
                    icon={<PlusOutlined />}
                    style={{ width: '100%' }}
                  >
                    添加时刻
                  </Button>
                  <Form.ErrorList errors={errors} />
                </>
              )}
            </Form.List>
          </Form.Item>
        </>
      )}

      <Divider orientation="left" style={{ margin: '4px 0 16px' }}>
        调度与触发
      </Divider>

      {!isRule && (
        <Row gutter={12}>
          {intraday ? null : (
            <Col span={12}>
              <Form.Item
                label="唤醒时刻（可配置多个）"
                required
                extra="每个时刻每交易日各触发一次；决策 bar 由 as_of 截断决定（无前视，无需收盘时点校验）。"
              >
                <Form.List
                  name="trigger_times"
                  rules={[
                    {
                      validator: async (_, value: Dayjs[] | undefined) => {
                        if (!value || value.length < 1) {
                          return Promise.reject(new Error('至少配置一个唤醒时刻'))
                        }
                      },
                    },
                  ]}
                >
                  {(fields, { add, remove }, { errors }) => (
                    <>
                      {fields.map((field) => (
                        <Space key={field.key} align="baseline" style={{ display: 'flex', marginBottom: 8 }}>
                          <Form.Item
                            {...field}
                            rules={[{ required: true, message: '请选择唤醒时刻' }]}
                            noStyle
                          >
                            <TimePicker format="HH:mm" allowClear={false} minuteStep={5} />
                          </Form.Item>
                          {fields.length > 1 ? (
                            <MinusCircleOutlined
                              aria-label="删除时刻"
                              onClick={() => remove(field.name)}
                            />
                          ) : null}
                        </Space>
                      ))}
                      <Button
                        type="dashed"
                        onClick={() => add(parseHHmm('15:05'))}
                        icon={<PlusOutlined />}
                        style={{ width: '100%' }}
                      >
                        添加时刻
                      </Button>
                      <Form.ErrorList errors={errors} />
                    </>
                  )}
                </Form.List>
              </Form.Item>
            </Col>
          )}
          <Col span={12}>
            <Form.Item label="决策 bar 频率" extra="由所选模型的训练间隔派生，不可更改。">
              <Input value={barFreqLabel(barFreq)} readOnly />
            </Form.Item>
          </Col>
        </Row>
      )}

      {!isRule && (
        <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 16 }}>
          {intraday
            ? `盘中监控模式：交易时段内每根 ${barFreq} bar 收盘后自动决策一次（按 bar 网格调度，幂等去重）；无需配置唤醒时刻。`
            : '多个时点为同一交易日的多次触发尝试；受决策层幂等约束，当日至多产出一次决策与一次提醒（避免前视）。'}
        </Text>
      )}

      <Form.Item
        label="通知通道"
        name="notify_channels"
        extra="仅选择通道名；webhook/secret/token 等凭证由后端环境变量管理，前端不录入。"
      >
        <Select mode="multiple" placeholder="选择推送通道（可多选）" options={NOTIFY_CHANNEL_OPTIONS} allowClear />
      </Form.Item>

      <Form.Item label="启用自动调度" name="enabled" valuePropName="checked" extra="启用后由调度器在决策时点自动触发；仅提醒，不下单。">
        <Switch checkedChildren="启用" unCheckedChildren="停用" />
      </Form.Item>

      {/* CNN 模式：组合快照与阈值 */}
      {!isRule && (
        <>
          <Divider orientation="left" style={{ margin: '4px 0 16px' }}>
            组合快照与阈值
          </Divider>

          <Row gutter={12}>
            <Col span={12}>
              <Form.Item label="总资金" name="portfolio_value" rules={[{ required: true, message: '请输入总市值' }]}>
                <InputNumber min={0} step={10000} style={{ width: '100%' }} addonAfter="元" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="当前总持仓市值" name="total_position_value" rules={[{ required: true }]}>
                <InputNumber min={0} step={10000} style={{ width: '100%' }} addonAfter="元" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="当前持仓（股）" name="current_position" rules={[{ required: true }]}>
                <InputNumber min={0} step={100} style={{ width: '100%' }} addonAfter="股" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="持仓市值" name="current_symbol_value" rules={[{ required: true }]}>
                <InputNumber min={0} step={10000} style={{ width: '100%' }} addonAfter="元" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col span={8}>
              <Form.Item label="买入阈值" name="buy_threshold">
                <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="目标仓位比例" name="position_ratio">
                <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="最小手数" name="min_volume">
                <InputNumber min={1} step={100} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        </>
      )}

      <Space>
        <Button type="primary" loading={submitting} onClick={() => void handleSubmit()}>
          {initialPlan ? '保存计划' : '创建计划'}
        </Button>
        {onCancel ? <Button onClick={onCancel}>取消</Button> : null}
      </Space>
      <div style={{ marginTop: 8 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          提示：计划仅产出决策与提醒，不会向任何券商提交真实订单。
        </Text>
      </div>
    </Form>
  )
}

export default PlanForm
