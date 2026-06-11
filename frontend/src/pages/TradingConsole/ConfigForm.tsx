import React from 'react'
import {
  App,
  Button,
  Col,
  DatePicker,
  Divider,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Tag,
  Typography,
} from 'antd'
import { DatabaseOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { useMutation, useQuery } from '@tanstack/react-query'
import dayjs, { type Dayjs } from 'dayjs'

import { cnnService } from '../../api/cnn'
import { governanceService } from '../../api/governance'
import { liveService } from '../../api/liveApi'
import type { CNNModelInfo } from '../../types/cnn'
import type { LiveDecisionRequest } from '../../types/live'
import { barFreqOfInterval, isIntradayBarFreq } from '../../utils/barFreq'

const { Text } = Typography

const DATA_SOURCE_OPTIONS = [
  { label: '接口拉取（Tushare / 增量）', value: 'pull' },
  { label: '上传文件（CSV / parquet）', value: 'upload' },
]

interface ConfigFormProps {
  /** 拿到 task_id 后回传给页面，供进度/结果/风控区消费（任务 9.3~9.5）。 */
  onStarted: (taskId: string) => void
  /** 是否已触发过至少一次决策（用于「触发 / 重新触发」按钮文案）。 */
  hasTriggered: boolean
  /** 当前是否有决策任务正在执行（按钮 loading + 禁用）。 */
  running?: boolean
}

/**
 * 表单内部字段类型。
 * as_of_date：日频模型 = 决策日（提交时转为该日收盘后 15:05）；
 * 日内模型 = 精确到分钟的决策时刻（直接提交）。
 */
interface ConfigFormValues {
  model: string
  vt_symbol: string
  scheme: string
  as_of_date: Dayjs
  data_source: 'upload' | 'pull'
  portfolio_value: number
  total_position_value: number
  current_position: number
  current_symbol_value: number
  buy_threshold: number
  model_version: string
}

/**
 * 配置表单（任务 9.2）。
 *
 * 覆盖输入：模型（拉 /api/cnn/models）、目标标的、方案、决策日期（默认当天）、
 * 数据源（upload/pull）、组合快照（总资金 / 当前持仓 / 持仓市值）。
 *
 * 提交经 useMutation 调 liveService.startDecision 拿 task_id，并通过 onStarted 上抛给页面。
 * 始终提供「触发 / 重新触发」按钮（Req 6.6）。
 */
const ConfigForm: React.FC<ConfigFormProps> = ({ onStarted, hasTriggered, running }) => {
  const { message } = App.useApp()
  const [form] = Form.useForm<ConfigFormValues>()

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })
  const { data: production } = useQuery({
    queryKey: ['cnn-governance-production'],
    queryFn: () => governanceService.getProduction(),
  })

  React.useEffect(() => {
    if (!production?.model_name) return
    form.setFieldsValue({
      model: production.model_name,
      vt_symbol: production.target_symbol || form.getFieldValue('vt_symbol'),
      model_version: production.model_version || '',
    })
  }, [form, production?.model_name, production?.model_version, production?.target_symbol])

  // bar_freq 由所选模型的训练间隔派生（间隔锁定，与后端校验一致）。
  const selectedModel = Form.useWatch('model', form)
  const pickedMeta = (models || []).find((m: CNNModelInfo) => m.name === selectedModel)
  const barFreq = barFreqOfInterval(pickedMeta?.input_interval)
  const intraday = isIntradayBarFreq(barFreq)

  const mutation = useMutation({
    mutationFn: (req: LiveDecisionRequest) => liveService.startDecision(req),
    onSuccess: (res) => {
      message.success(res.message || '今日决策任务已启动')
      onStarted(res.task_id)
    },
    onError: (error: unknown) => {
      message.error(error instanceof Error ? error.message : '触发决策失败')
    },
  })

  // 选择模型时，若目标标的为空则用模型的 target_symbol 预填，省去重复输入。
  const handleModelChange = (value: string) => {
    const picked = (models || []).find((m) => m.name === value)
    if (picked?.target_symbol && !form.getFieldValue('vt_symbol')) {
      form.setFieldValue('vt_symbol', picked.target_symbol)
    }
    if (value !== production?.model_name) {
      form.setFieldValue('model_version', '')
    } else {
      form.setFieldValue('model_version', production?.model_version || '')
    }
  }

  const handleSubmit = async () => {
    let values: ConfigFormValues
    try {
      values = await form.validateFields()
    } catch {
      return // 校验失败由表单自身提示
    }

    const req: LiveDecisionRequest = {
      model: values.model,
      vt_symbol: values.vt_symbol.trim(),
      scheme: values.scheme.trim(),
      // 日频：决策时刻 = 所选决策日的收盘后（15:05，本地朴素时间，避免时区偏移）；
      // 日内：决策时刻 = 用户选择的精确分钟（决策 bar 为该时刻前最后一根已收盘 bar，无前视）。
      as_of: (intraday
        ? values.as_of_date.second(0).millisecond(0)
        : values.as_of_date.hour(15).minute(5).second(0).millisecond(0)
      ).format('YYYY-MM-DDTHH:mm:ss'),
      bar_freq: barFreq,
      data_source: values.data_source,
      portfolio: {
        portfolio_value: values.portfolio_value,
        total_position_value: values.total_position_value,
        current_position: values.current_position,
        current_symbol_value: values.current_symbol_value,
      },
      buy_threshold: values.buy_threshold,
      model_version: values.model_version || '',
    }
    mutation.mutate(req)
  }

  const submitting = mutation.isPending
  const buttonLabel = hasTriggered ? '重新触发决策' : '触发决策'

  return (
    <Form<ConfigFormValues>
      form={form}
      layout="vertical"
      initialValues={{
        model: undefined,
        vt_symbol: '',
        scheme: 'eod_buy_v1',
        as_of_date: dayjs(), // 决策日默认当天（as_of=当日收盘后）
        data_source: 'pull',
        portfolio_value: 1000000,
        total_position_value: 0,
        current_position: 0,
        current_symbol_value: 0,
        buy_threshold: 0.6,
        model_version: '',
      }}
    >
      <Form.Item name="model_version" hidden>
        <Input />
      </Form.Item>
      <Form.Item
        label="CNN 模型"
        name="model"
        rules={[{ required: true, message: '请选择 CNN 模型' }]}
        extra={
          models && models.length === 0
            ? '暂无可用模型，请先在「CNN 训练」中训练模型'
            : production?.model_name
              ? `默认生产模型: ${production.model_name} @ ${production.model_version || '-'}`
              : '来自 /api/cnn/models'
        }
      >
        <Select
          showSearch
          loading={modelsLoading}
          placeholder="选择用于今日推理的模型"
          optionFilterProp="label"
          onChange={handleModelChange}
          notFoundContent={modelsLoading ? '加载中...' : '暂无模型'}
          options={(models || []).map((m: CNNModelInfo) => ({
            value: m.name,
            label: m.name === production?.model_name ? `${m.name} (生产)` : m.name,
          }))}
          optionRender={(option) => {
            const meta = (models || []).find((m) => m.name === option.value)
            return (
              <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                <Space size={4}>
                  <DatabaseOutlined style={{ color: '#52c41a' }} />
                  <span>{option.label}</span>
                </Space>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {[meta?.input_interval || 'd', meta?.target_symbol]
                    .filter(Boolean)
                    .join(' · ')}
                  {meta?.name === production?.model_name ? ' · 生产' : ''}
                </Text>
              </Space>
            )
          }}
        />
      </Form.Item>

      <Form.Item
        label="目标标的"
        name="vt_symbol"
        rules={[{ required: true, message: '请输入目标标的' }]}
        extra="如 000001.SZSE。选择模型后会自动填入其训练目标证券。"
      >
        <Input placeholder="000001.SZSE" allowClear />
      </Form.Item>

      <Form.Item
        label="方案"
        name="scheme"
        rules={[{ required: true, message: '请输入方案名' }]}
        extra="参与 signal_id 生成（日期 + 方案 + 模型版本）。"
      >
        <Input placeholder="eod_buy_v1" allowClear />
      </Form.Item>

      <Row gutter={12}>
        <Col span={12}>
          <Form.Item
            label={intraday ? '决策时刻' : '决策日期'}
            name="as_of_date"
            rules={[{ required: true, message: intraday ? '请选择决策时刻' : '请选择决策日期' }]}
            extra={
              intraday
                ? `日内模型（${barFreq}）：取该时刻前最后一根已收盘 bar 决策（无前视）`
                : '默认当天；按 1d 收盘后 as_of 决策（无前视）'
            }
          >
            {intraday ? (
              <DatePicker
                style={{ width: '100%' }}
                format="YYYY-MM-DD HH:mm"
                showTime={{ format: 'HH:mm' }}
                allowClear={false}
              />
            ) : (
              <DatePicker style={{ width: '100%' }} format="YYYY-MM-DD" allowClear={false} />
            )}
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item label="数据源" name="data_source" rules={[{ required: true }]}>
            <Select options={DATA_SOURCE_OPTIONS} />
          </Form.Item>
        </Col>
      </Row>

      <Divider orientation="left" style={{ margin: '4px 0 16px' }}>
        <Space size={4}>
          组合快照
          <Tag color="blue">用于风控限额</Tag>
        </Space>
      </Divider>

      <Row gutter={12}>
        <Col span={12}>
          <Form.Item
            label="总资金"
            name="portfolio_value"
            rules={[{ required: true, message: '请输入组合总市值' }]}
            tooltip="组合总市值（现金 + 持仓），用于计算仓位限额。"
          >
            <InputNumber<number>
              min={0}
              step={10000}
              style={{ width: '100%' }}
              addonAfter="元"
              formatter={(v) => `${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
              parser={(v) => Number((v || '').replace(/,/g, ''))}
            />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item
            label="当前总持仓市值"
            name="total_position_value"
            rules={[{ required: true, message: '请输入当前总持仓市值' }]}
            tooltip="组合内全部持仓的市值合计，用于总仓位上限检查。"
          >
            <InputNumber min={0} step={10000} style={{ width: '100%' }} addonAfter="元" />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item
            label="当前持仓（股）"
            name="current_position"
            rules={[{ required: true, message: '请输入目标标的当前持仓股数' }]}
            tooltip="目标标的当前持有的股数；> 0 时可触发出场逻辑。"
          >
            <InputNumber min={0} step={100} style={{ width: '100%' }} addonAfter="股" />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item
            label="持仓市值"
            name="current_symbol_value"
            rules={[{ required: true, message: '请输入目标标的当前持仓市值' }]}
            tooltip="目标标的当前持仓市值，用于单票上限检查。"
          >
            <InputNumber min={0} step={10000} style={{ width: '100%' }} addonAfter="元" />
          </Form.Item>
        </Col>
      </Row>

      <Form.Item
        label="买入阈值"
        name="buy_threshold"
        tooltip="信号概率达到该阈值才视为买入候选。"
      >
        <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
      </Form.Item>

      <Button
        type="primary"
        icon={<ThunderboltOutlined />}
        block
        loading={submitting || running}
        onClick={() => void handleSubmit()}
      >
        {buttonLabel}
      </Button>
    </Form>
  )
}

export default ConfigForm
