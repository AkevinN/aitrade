import React from 'react'
import {
  Button,
  Col,
  Divider,
  Form,
  Input,
  InputNumber,
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
import type { CNNModelInfo } from '../../types/cnn'
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

/** 把 "HH:mm" 解析为 Dayjs（不依赖 customParseFormat 插件，直接设置时分）。 */
const parseHHmm = (value: string): Dayjs => {
  const [hh, mm] = value.split(':').map((x) => Number(x) || 0)
  return dayjs().hour(hh).minute(mm).second(0).millisecond(0)
}

interface PlanFormValues {
  name: string
  model: string
  vt_symbol: string
  scheme: string
  trigger_times: Dayjs[]
  notify_channels: NotifyChannel[]
  data_source: 'upload' | 'pull'
  enabled: boolean
  buy_threshold: number
  position_ratio: number
  min_volume: number
  model_version: string
  portfolio_value: number
  total_position_value: number
  current_position: number
  current_symbol_value: number
}

interface PlanFormProps {
  /** 编辑模式下的初始计划；为空表示新建。 */
  initialPlan?: TradingPlan | null
  /** 提交回调，组装好的请求体上抛给父级（由父级 mutation 落库）。 */
  onSubmit: (req: TradingPlanRequest) => void
  /** 取消编辑。 */
  onCancel?: () => void
  /** 提交中状态。 */
  submitting?: boolean
}

/**
 * 交易计划表单。复用决策配置字段，并增加调度相关字段：
 * 唤醒时刻（trigger_times，仅日频）、通知通道多选（notify_channels）、启用开关（enabled）。
 *
 * 间隔锁定：决策 bar 频率（bar_freq）由所选模型的训练间隔自动派生（只读展示），
 * 不提供自由选择——模型在固定间隔上训练，喂别的周期是分布外输入。
 * 日内模型 → 盘中监控模式：交易时段内每根 bar 收盘后自动决策一次，无需配置唤醒时刻。
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

  // bar_freq 派生：所选模型的训练间隔 → bar_freq；模型未选/列表未加载时回退编辑中计划的值。
  const selectedModel = Form.useWatch('model', form)
  const pickedMeta = (models || []).find((m: CNNModelInfo) => m.name === selectedModel)
  const barFreq = pickedMeta
    ? barFreqOfInterval(pickedMeta.input_interval)
    : initialPlan?.bar_freq || '1d'
  const intraday = isIntradayBarFreq(barFreq)

  React.useEffect(() => {
    if (!initialPlan) {
      form.resetFields()
      return
    }
    form.setFieldsValue({
      name: initialPlan.name,
      model: initialPlan.model,
      vt_symbol: initialPlan.vt_symbol,
      scheme: initialPlan.scheme,
      // 回填生效唤醒时刻（多时刻；日内计划为空时给默认值占位，提交时按频率丢弃）。
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
    })
  }, [form, initialPlan])

  const handleSubmit = async () => {
    let values: PlanFormValues
    try {
      values = await form.validateFields()
    } catch {
      return
    }
    const req: TradingPlanRequest = {
      name: values.name.trim(),
      model: values.model,
      vt_symbol: values.vt_symbol.trim(),
      scheme: values.scheme.trim(),
      // bar_freq 由模型训练间隔派生（间隔锁定）；日内计划按 Bar_Grid 自动调度，不传唤醒时刻。
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
    }
    onSubmit(req)
  }

  return (
    <Form<PlanFormValues>
      form={form}
      layout="vertical"
      initialValues={{
        name: '',
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
      }}
    >
      <Form.Item name="model_version" hidden>
        <Input />
      </Form.Item>

      <Form.Item label="计划名称" name="name" rules={[{ required: true, message: '请输入计划名称' }]}>
        <Input placeholder="如：平安银行尾盘买入计划" allowClear />
      </Form.Item>

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

      <Divider orientation="left" style={{ margin: '4px 0 16px' }}>
        调度与触发
      </Divider>

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

      <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 16 }}>
        {intraday
          ? `盘中监控模式：交易时段内每根 ${barFreq} bar 收盘后自动决策一次（按 bar 网格调度，幂等去重）；无需配置唤醒时刻。`
          : '多个时点为同一交易日的多次触发尝试；受决策层幂等约束，当日至多产出一次决策与一次提醒（避免前视）。'}
      </Text>

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
