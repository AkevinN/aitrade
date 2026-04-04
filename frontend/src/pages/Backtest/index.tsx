import React, { useState, useCallback, useMemo } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, DatePicker,
  InputNumber, Select, Progress, message, Statistic,
} from 'antd'
import {
  PlayCircleOutlined, LineChartOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs from 'dayjs'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import { taskStore } from '../../stores/taskStore'

const { Text } = Typography
const { RangePicker } = DatePicker

interface BacktestResult {
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  profit_factor: number
}

interface DailyPnL {
  date: string
  pnl: number
  balance: number
}

const Backtest: React.FC = () => {
  const [backtestName, setBacktestName] = useState('')
  const [selectedSignal, setSelectedSignal] = useState('')
  const [capital, setCapital] = useState(1000000)
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(1, 'year'),
    dayjs(),
  ])
  const [benchmark, setBenchmark] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)

  const task = useTask(taskId)

  const { data: signals } = useQuery({
    queryKey: ['alpha-signals'],
    queryFn: () => alphaService.listSignals(),
  })

  const handleRun = useCallback(async () => {
    if (!selectedSignal) {
      message.warning('Please select a signal')
      return
    }
    const name = backtestName || `backtest_${selectedSignal}_${Date.now()}`
    try {
      const res = await alphaService.runBacktest({
        name,
        signal: selectedSignal,
        capital,
        start: dateRange[0].format('YYYY-MM-DD'),
        end: dateRange[1].format('YYYY-MM-DD'),
      })
      setTaskId(res.id)
      taskStore.getState().addTask({
        id: res.id,
        type: 'backtest',
        status: 'running',
        progress: 0,
        message: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      message.success('Backtest started')
    } catch {
      message.error('Failed to run backtest')
    }
  }, [backtestName, selectedSignal, capital, dateRange])

  const simulatedResult = useMemo((): BacktestResult => ({
    total_return: (Math.random() * 40 - 10) / 100,
    sharpe_ratio: (Math.random() * 2 + 0.5),
    max_drawdown: (Math.random() * 20 + 5) / 100,
    win_rate: (Math.random() * 30 + 45) / 100,
    total_trades: Math.floor(Math.random() * 200 + 50),
    profit_factor: (Math.random() * 1.5 + 0.8),
  }), [])

  const simulatedDailyPnL = useMemo((): DailyPnL[] => {
    const days = dateRange[1].diff(dateRange[0], 'day')
    const data: DailyPnL[] = []
    let balance = capital
    let cumulative = 0

    for (let i = 0; i < Math.min(days, 365); i++) {
      const date = dateRange[0].add(i, 'day').format('YYYY-MM-DD')
      const pnl = (Math.random() - 0.48) * capital * 0.02
      cumulative += pnl
      balance += pnl
      data.push({
        date,
        pnl,
        balance,
      })
    }
    return data
  }, [dateRange, capital])

  const pnlChartData = useMemo(() => {
    return simulatedDailyPnL.map((d) => ({
      date: d.date.slice(5, 10),
      pnl: d.pnl,
    }))
  }, [simulatedDailyPnL])

  const balanceChartData = useMemo(() => {
    return simulatedDailyPnL.map((d) => ({
      date: d.date.slice(5, 10),
      balance: d.balance,
    }))
  }, [simulatedDailyPnL])

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        Strategy Backtest
      </Typography.Title>

      <Row gutter={[16, 16]}>
        {/* Left: Config */}
        <Col xs={24} lg={8}>
          <Card title={<><LineChartOutlined /> Backtest Config</>} size="small">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text type="secondary">Backtest Name:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 8 }}
                  value={backtestName}
                  onChange={(v) => setBacktestName(v || '')}
                  placeholder="Auto-generated if empty"
                />
              </div>
              <div>
                <Text type="secondary">Signal:</Text>
                <Select
                  style={{ width: '100%', marginTop: 8 }}
                  value={selectedSignal || undefined}
                  onChange={setSelectedSignal}
                  placeholder="Select signal"
                  options={(signals || []).map((s) => ({ label: s, value: s }))}
                />
              </div>
              <div>
                <Text type="secondary">Initial Capital:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 8 }}
                  value={capital}
                  onChange={(v) => setCapital(v || 1000000)}
                  min={10000}
                  step={100000}
                  formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={(value) => value?.replace(/\$\s?|(,*)/g, '') as unknown as number}
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
              <div>
                <Text type="secondary">Benchmark (optional):</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 8 }}
                  value={benchmark}
                  onChange={(v) => setBenchmark(v || '')}
                  placeholder="e.g., 000300.SSE"
                />
              </div>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleRun}
                loading={task?.data?.status === 'running'}
                block
              >
                Run Backtest
              </Button>
              {task?.data && (
                <Card size="small">
                  <Progress
                    percent={Math.round(task.data.progress)}
                    status={task.data.status === 'failed' ? 'exception' : 'active'}
                  />
                  <Text type="secondary">{task.data.status}</Text>
                </Card>
              )}
            </Space>
          </Card>
        </Col>

        {/* Right: Results */}
        <Col xs={24} lg={16}>
          {/* Summary Stats */}
          <Row gutter={[8, 8]} style={{ marginBottom: 16 }}>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="Total Return"
                  value={simulatedResult.total_return * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ fontSize: 18, color: simulatedResult.total_return >= 0 ? '#49aa19' : '#dc4446' }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="Sharpe Ratio"
                  value={simulatedResult.sharpe_ratio}
                  precision={2}
                  valueStyle={{ fontSize: 18 }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="Max Drawdown"
                  value={simulatedResult.max_drawdown * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ fontSize: 18, color: '#dc4446' }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="Win Rate"
                  value={simulatedResult.win_rate * 100}
                  precision={1}
                  suffix="%"
                  valueStyle={{ fontSize: 18, color: '#1668dc' }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="Total Trades"
                  value={simulatedResult.total_trades}
                  valueStyle={{ fontSize: 18 }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="Profit Factor"
                  value={simulatedResult.profit_factor}
                  precision={2}
                  valueStyle={{ fontSize: 18 }}
                />
              </Card>
            </Col>
          </Row>

          {/* Balance Curve */}
          <Card title="Balance Curve" size="small" style={{ marginBottom: 16 }}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={balanceChartData}>
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                <Tooltip formatter={(value) => typeof value === 'number' ? value.toFixed(2) : value} />
                <Line
                  type="monotone"
                  dataKey="balance"
                  stroke="#1668dc"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Daily PnL */}
          <Card title="Daily PnL" size="small">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={pnlChartData}>
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip formatter={(value) => typeof value === 'number' ? value.toFixed(2) : value} />
                <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                  {pnlChartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.pnl >= 0 ? '#49aa19' : '#dc4446'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Backtest
