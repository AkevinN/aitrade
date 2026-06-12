import React, { useMemo, useState } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, Tag, DatePicker,
  Input, Select, Table, Progress, message, Statistic,
} from 'antd'
import {
  ThunderboltOutlined, RobotOutlined, ArrowUpOutlined, ArrowDownOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs from 'dayjs'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell,
} from 'recharts'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'

const { Text } = Typography
const { RangePicker } = DatePicker

interface SignalRow {
  datetime: string
  vt_symbol: string
  signal: number
}

/**
 * 信号生成页面：选择 Alpha 模型与目标标的，启动信号生成任务，并在完成后展示信号列表与分布图。
 */
const Signal: React.FC = () => {
  const [signalName, setSignalName] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [symbolsText, setSymbolsText] = useState('')
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(6, 'month'),
    dayjs(),
  ])
  const [taskId, setTaskId] = useState<string | null>(null)
  const [signals, setSignals] = useState<SignalRow[]>([])

  const task = useTask(taskId)

  const { data: models } = useQuery({
    queryKey: ['alpha-models'],
    queryFn: () => alphaService.listModels(),
  })

  const { data: existingSignals } = useQuery({
    queryKey: ['alpha-signals'],
    queryFn: () => alphaService.listSignals(),
  })

  const symbols = useMemo(() => (
    symbolsText
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
  ), [symbolsText])

  const handleGenerate = async () => {
    if (!selectedModel || symbols.length === 0) {
      message.warning('请选择模型并输入证券代码')
      return
    }

    const name = signalName || `signal_${selectedModel}_${Date.now()}`
    try {
      const result = await alphaService.generateSignal({
        name,
        model: selectedModel,
        vt_symbols: symbols,
        start: dateRange[0].format('YYYY-MM-DD'),
        end: dateRange[1].format('YYYY-MM-DD'),
      })
      setTaskId(result.task_id)
      message.success('信号生成任务已启动')
    } catch (error) {
      console.error(error)
      message.error('信号生成失败')
    }
  }

  const handleLoadSignal = async (name: string) => {
    try {
      const info = await alphaService.getSignal(name)
      setSignals(
        info.preview.map((row) => ({
          datetime: String(row.datetime ?? ''),
          vt_symbol: String(row.vt_symbol ?? ''),
          signal: Number(row.signal ?? 0),
        }))
      )
    } catch (error) {
      console.error(error)
      message.error('加载信号失败')
    }
  }

  const signalStats = useMemo(() => {
    if (signals.length === 0) return null
    const bullish = signals.filter((item) => item.signal > 0).length
    const bearish = signals.filter((item) => item.signal < 0).length
    const neutral = signals.filter((item) => item.signal === 0).length
    return { total: signals.length, bullish, bearish, neutral }
  }, [signals])

  const distributionData = useMemo(() => {
    if (signals.length === 0) return []
    const bins = [
      { range: '< -0.5', count: 0 },
      { range: '-0.5 ~ -0.2', count: 0 },
      { range: '-0.2 ~ 0', count: 0 },
      { range: '0 ~ 0.2', count: 0 },
      { range: '0.2 ~ 0.5', count: 0 },
      { range: '> 0.5', count: 0 },
    ]

    signals.forEach((item) => {
      const value = item.signal
      if (value < -0.5) bins[0].count += 1
      else if (value < -0.2) bins[1].count += 1
      else if (value < 0) bins[2].count += 1
      else if (value < 0.2) bins[3].count += 1
      else if (value < 0.5) bins[4].count += 1
      else bins[5].count += 1
    })
    return bins
  }, [signals])

  const lineData = useMemo(() => (
    signals.map((item) => ({
      time: item.datetime.slice(0, 10),
      value: item.signal,
    }))
  ), [signals])

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        信号分析
      </Typography.Title>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}>
          <Card title={<><RobotOutlined /> 生成信号</>} size="small">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text type="secondary">信号名称：</Text>
                <Input
                  style={{ marginTop: 8 }}
                  value={signalName}
                  onChange={(e) => setSignalName(e.target.value)}
                  placeholder="留空则自动生成"
                />
              </div>
              <div>
                <Text type="secondary">模型：</Text>
                <Select
                  style={{ width: '100%', marginTop: 8 }}
                  value={selectedModel || undefined}
                  onChange={setSelectedModel}
                  placeholder="选择模型"
                  options={(models || []).map((item) => ({ label: item, value: item }))}
                />
              </div>
              <div>
                <Text type="secondary">证券列表（每行一个）：</Text>
                <Input.TextArea
                  style={{ marginTop: 8 }}
                  rows={4}
                  value={symbolsText}
                  onChange={(e) => setSymbolsText(e.target.value)}
                  placeholder="000001.SZSE&#10;600000.SSE"
                />
              </div>
              <div>
                <Text type="secondary">日期范围：</Text>
                <RangePicker
                  style={{ width: '100%', marginTop: 8 }}
                  value={dateRange}
                  onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
                />
              </div>
              <Button
                type="primary"
                icon={<ThunderboltOutlined />}
                onClick={handleGenerate}
                loading={task.data?.status === 'running'}
                block
              >
                生成信号
              </Button>
              {task.data && (
                <Card size="small">
                  <Progress
                    percent={Math.round(task.data.progress)}
                    status={task.data.status === 'failed' ? 'exception' : 'active'}
                  />
                  <Text type="secondary">{task.data.message || task.data.status}</Text>
                </Card>
              )}
            </Space>
          </Card>

          <Card title="已有信号" size="small" style={{ marginTop: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {(existingSignals || []).map((item) => (
                <Button key={item} onClick={() => handleLoadSignal(item)} block>
                  {item}
                </Button>
              ))}
              {(existingSignals || []).length === 0 && (
                <Text type="secondary">还没有信号</Text>
              )}
            </Space>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          {signalStats && (
            <Row gutter={[8, 8]} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Card size="small">
                  <Statistic title="总数" value={signalStats.total} />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic title="看多" value={signalStats.bullish} prefix={<ArrowUpOutlined />} valueStyle={{ color: '#49aa19' }} />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic title="看空" value={signalStats.bearish} prefix={<ArrowDownOutlined />} valueStyle={{ color: '#dc4446' }} />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic title="中性" value={signalStats.neutral} />
                </Card>
              </Col>
            </Row>
          )}

          <Card title="信号时间线" size="small" style={{ marginBottom: 16 }}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lineData}>
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#1668dc" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="信号分布" size="small" style={{ marginBottom: 16 }}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={distributionData}>
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count">
                  {distributionData.map((item) => (
                    <Cell
                      key={item.range}
                      fill={item.range.includes('-') ? '#dc4446' : item.range.includes('0 ~') ? '#9ca3af' : '#49aa19'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="信号预览" size="small">
            <Table<SignalRow>
              size="small"
              rowKey={(row) => `${row.datetime}_${row.vt_symbol}`}
              dataSource={signals}
              pagination={{ pageSize: 20 }}
              columns={[
                { title: '时间', dataIndex: 'datetime', key: 'datetime' },
                { title: '证券', dataIndex: 'vt_symbol', key: 'vt_symbol' },
                {
                  title: '信号',
                  dataIndex: 'signal',
                  key: 'signal',
                  render: (value: number) => (
                    <Space>
                      <Tag color={value > 0 ? 'green' : value < 0 ? 'red' : 'default'}>
                        {value > 0 ? 'LONG' : value < 0 ? 'SHORT' : 'NEUTRAL'}
                      </Tag>
                      <Text>{value.toFixed(4)}</Text>
                    </Space>
                  ),
                },
              ]}
              locale={{ emptyText: '加载信号后查看数据' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Signal
