import React, { useState, useCallback, useMemo } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, Tag, DatePicker,
  Input, Select, Table, Progress, message, Statistic, Empty,
} from 'antd'
import {
  ThunderboltOutlined, RobotOutlined, ArrowUpOutlined, ArrowDownOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import dayjs from 'dayjs'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell,
} from 'recharts'
import { createChart, ColorType } from 'lightweight-charts'
import { useEffect, useRef } from 'react'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import { taskStore } from '../../stores/taskStore'

const { Text } = Typography
const { RangePicker } = DatePicker

interface SignalRow {
  datetime: string
  vt_symbol: string
  signal: number
}

const Signal: React.FC = () => {
  const queryClient = useQueryClient()

  const [signalName, setSignalName] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [symbolsText, setSymbolsText] = useState('')
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(6, 'month'),
    dayjs(),
  ])
  const [taskId, setTaskId] = useState<string | null>(null)
  const [signals, setSignals] = useState<SignalRow[]>([])

  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null)

  const task = useTask(taskId)

  const { data: models } = useQuery({
    queryKey: ['alpha-models'],
    queryFn: () => alphaService.listModels(),
  })

  const { data: existingSignals } = useQuery({
    queryKey: ['alpha-signals'],
    queryFn: () => alphaService.listSignals(),
  })

  const symbols = symbolsText
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)

  const handleGenerate = useCallback(async () => {
    if (!selectedModel) {
      message.warning('Please select a model')
      return
    }
    const name = signalName || `signal_${selectedModel}_${Date.now()}`
    try {
      const result = await alphaService.generateSignal({
        name,
        model_name: selectedModel,
        dataset_name: selectedModel.split('_')[0] || 'default',
        start_date: dateRange[0].format('YYYY-MM-DD'),
        end_date: dateRange[1].format('YYYY-MM-DD'),
      })
      setTaskId(result.task_id)
      taskStore.getState().addTask({
        id: result.task_id,
        name: `Generate Signal: ${name}`,
        status: 'running',
        progress: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      message.success('Signal generation started')
    } catch {
      message.error('Failed to generate signal')
    }
  }, [signalName, selectedModel, dateRange])

  const handleLoadSignal = useCallback(async (signalName: string) => {
    try {
      const info = await alphaService.getSignal(signalName)
      if (info.preview && info.preview.length > 0) {
        setSignals(
          info.preview.map((p) => ({
            datetime: p.date,
            vt_symbol: p.symbol,
            signal: p.signal,
          }))
        )
      }
    } catch {
      message.error('Failed to load signal')
    }
  }, [])

  const signalStats = useMemo(() => {
    if (signals.length === 0) return null
    const bullish = signals.filter((s) => s.signal > 0).length
    const bearish = signals.filter((s) => s.signal < 0).length
    const neutral = signals.filter((s) => s.signal === 0).length
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
    signals.forEach((s) => {
      const v = s.signal
      if (v < -0.5) bins[0].count++
      else if (v < -0.2) bins[1].count++
      else if (v < 0) bins[2].count++
      else if (v < 0.2) bins[3].count++
      else if (v < 0.5) bins[4].count++
      else bins[5].count++
    })
    return bins
  }, [signals])

  const chartData = useMemo(() => {
    if (signals.length === 0) return []
    return signals.map((s) => ({
      time: s.datetime.slice(0, 10),
      value: s.signal,
    }))
  }, [signals])

  useEffect(() => {
    if (!chartContainerRef.current || chartData.length === 0) return

    if (chartRef.current) {
      chartRef.current.remove()
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 200,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: 'rgba(197,203,206,0.06)' },
        horzLines: { color: 'rgba(197,203,206,0.06)' },
      },
    })

    const lineSeries = chart.addLineSeries({
      color: '#1668dc',
      lineWidth: 2,
    })
    lineSeries.setData(chartData as { time: string; value: number }[])

    chart.timeScale().fitContent()
    chartRef.current = chart

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [chartData])

  const signalColumns = [
    {
      title: 'DateTime',
      dataIndex: 'datetime',
      key: 'datetime',
      width: 150,
    },
    {
      title: 'Symbol',
      dataIndex: 'vt_symbol',
      key: 'vt_symbol',
      width: 150,
    },
    {
      title: 'Signal Value',
      dataIndex: 'signal',
      key: 'signal',
      width: 120,
      render: (v: number) => {
        const color = v > 0 ? '#49aa19' : v < 0 ? '#dc4446' : '#9ca3af'
        const icon = v > 0 ? <ArrowUpOutlined /> : v < 0 ? <ArrowDownOutlined /> : null
        return (
          <Space>
            <Tag color={v > 0 ? 'green' : v < 0 ? 'red' : 'default'}>
              {v > 0 ? 'LONG' : v < 0 ? 'SHORT' : 'NEUTRAL'}
            </Tag>
            <Text style={{ color }}>{v.toFixed(4)}</Text>
          </Space>
        )
      },
    },
  ]

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        Signal Analysis
      </Typography.Title>

      <Row gutter={[16, 16]}>
        {/* Left: Form */}
        <Col xs={24} lg={8}>
          <Card title={<><RobotOutlined /> Generate Signal</>} size="small">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text type="secondary">Signal Name:</Text>
                <Input
                  style={{ marginTop: 8 }}
                  value={signalName}
                  onChange={(e) => setSignalName(e.target.value)}
                  placeholder="Auto-generated if empty"
                />
              </div>
              <div>
                <Text type="secondary">Model:</Text>
                <Select
                  style={{ width: '100%', marginTop: 8 }}
                  value={selectedModel || undefined}
                  onChange={setSelectedModel}
                  placeholder="Select model"
                  options={(models || []).map((m) => ({ label: m, value: m }))}
                />
              </div>
              <div>
                <Text type="secondary">Symbols (one per line):</Text>
                <Input.TextArea
                  style={{ marginTop: 8 }}
                  rows={3}
                  value={symbolsText}
                  onChange={(e) => setSymbolsText(e.target.value)}
                  placeholder="000001.SZSE&#10;600000.SSE"
                />
              </div>
              <div>
                <Text type="secondary">Date Range:</Text>
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
                loading={task?.status === 'running'}
                block
              >
                Generate Signal
              </Button>
              {task && (
                <Card size="small">
                  <Progress
                    percent={Math.round(task.progress)}
                    status={task.status === 'failed' ? 'exception' : 'active'}
                  />
                  <Text type="secondary">{task.status}</Text>
                </Card>
              )}
            </Space>
          </Card>

          <Card title="Existing Signals" size="small" style={{ marginTop: 16 }}>
            <Table
              size="small"
              dataSource={(existingSignals || []).map((name) => ({ name, key: name }))}
              columns={[
                { title: 'Name', dataIndex: 'name', key: 'name' },
                {
                  title: 'Action',
                  key: 'action',
                  width: 80,
                  render: (_: unknown, record: { name: string }) => (
                    <Button size="small" onClick={() => handleLoadSignal(record.name)}>
                      Load
                    </Button>
                  ),
                },
              ]}
              pagination={false}
              locale={{ emptyText: 'No signals generated yet' }}
            />
          </Card>
        </Col>

        {/* Right: Signal Table & Charts */}
        <Col xs={24} lg={16}>
          {signalStats && (
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Card size="small">
                  <Statistic
                    title="Total"
                    value={signalStats.total}
                    valueStyle={{ fontSize: 18 }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic
                    title="Long"
                    value={signalStats.bullish}
                    valueStyle={{ fontSize: 18, color: '#49aa19' }}
                    prefix={<ArrowUpOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic
                    title="Short"
                    value={signalStats.bearish}
                    valueStyle={{ fontSize: 18, color: '#dc4446' }}
                    prefix={<ArrowDownOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic
                    title="Neutral"
                    value={signalStats.neutral}
                    valueStyle={{ fontSize: 18, color: '#9ca3af' }}
                  />
                </Card>
              </Col>
            </Row>
          )}

          {chartData.length > 0 && (
            <Card title="Signal Distribution" size="small" style={{ marginBottom: 16 }}>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={distributionData}>
                  <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {distributionData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={index < 3 ? '#dc4446' : index > 3 ? '#49aa19' : '#1668dc'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {chartData.length > 0 && (
            <Card title="Signal Timeline" size="small" style={{ marginBottom: 16 }}>
              <div ref={chartContainerRef} style={{ width: '100%' }} />
            </Card>
          )}

          <Card title="Signal Details" size="small">
            <Table
              size="small"
              columns={signalColumns}
              dataSource={signals.map((s, i) => ({ ...s, key: i }))}
              pagination={{ pageSize: 10, size: 'small' }}
              locale={{ emptyText: 'No signals loaded' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Signal
