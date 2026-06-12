import React, { useState, useCallback, useMemo } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, DatePicker,
  InputNumber, Select, Progress, message, Input, Alert,
} from 'antd'
import {
  PlayCircleOutlined, LineChartOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import BacktestResults from './BacktestResults'
import BacktestCharts from './BacktestCharts'
import type { BacktestResultPayload, BacktestStatistics } from '../../types/alpha'

const { Text } = Typography
const { RangePicker } = DatePicker

/**
 * Alpha 模型回测页面：选择信号、资金、区间与基准，启动回测任务并展示统计指标与图表。
 */
const AlphaBacktest: React.FC = () => {
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
  const statistics = useMemo(
    () => (task.data?.result as { statistics?: BacktestStatistics } | undefined)?.statistics,
    [task.data?.result]
  )
  // 完整回测结果载荷（含成交明细与逐日净值），供图表容器消费
  const result = useMemo(
    () => task.data?.result as BacktestResultPayload | undefined,
    [task.data?.result]
  )

  const { data: signals } = useQuery({
    queryKey: ['alpha-signals'],
    queryFn: () => alphaService.listSignals(),
  })

  const handleRun = useCallback(async () => {
    if (!selectedSignal) {
      message.warning('请选择一个信号')
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
        benchmark: benchmark || undefined,
      })
      setTaskId(res.task_id)
      message.success('回测任务已启动')
    } catch {
      message.error('启动回测失败')
    }
  }, [backtestName, benchmark, capital, dateRange, selectedSignal])

  return (
    <Row gutter={[16, 16]}>
      {/* Left: Config */}
      <Col xs={24} lg={8}>
        <Card title={<><LineChartOutlined /> 回测配置</>} size="small">
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <div>
              <Text type="secondary">回测名称：</Text>
              <Input
                style={{ width: '100%', marginTop: 8 }}
                value={backtestName}
                onChange={(e) => setBacktestName(e.target.value)}
                placeholder="留空则自动生成"
              />
            </div>
            <div>
              <Text type="secondary">信号：</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                value={selectedSignal || undefined}
                onChange={setSelectedSignal}
                placeholder="选择信号"
                options={(signals || []).map((s) => ({ label: s, value: s }))}
              />
            </div>
            <div>
              <Text type="secondary">初始资金：</Text>
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
              <Text type="secondary">日期范围：</Text>
              <RangePicker
                style={{ width: '100%', marginTop: 8 }}
                value={dateRange}
                onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
              />
            </div>
            <div>
              <Text type="secondary">基准（可选）：</Text>
              <Input
                style={{ width: '100%', marginTop: 8 }}
                value={benchmark}
                onChange={(e) => setBenchmark(e.target.value)}
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
              启动回测
            </Button>
            {task?.data && (
              <Card size="small">
                <Progress
                  percent={Math.round(task.data.progress)}
                  status={task.data.status === 'failed' ? 'exception' : task.data.status === 'completed' ? 'success' : 'active'}
                />
                <Text type="secondary">{task.data.message || task.data.status}</Text>
              </Card>
            )}
          </Space>
        </Card>
      </Col>

      <Col xs={24} lg={16}>
        {task.data?.status === 'failed' && (
          <Alert
            type="error"
            message="回测失败"
            description={task.data.message}
            style={{ marginBottom: 16 }}
          />
        )}
        <BacktestResults statistics={statistics} capital={capital} />
        {/* 回测完成后在统计卡片下方展示净值曲线与买卖点 K 线；不改 BacktestResults 既有展示。
            Alpha 信号回测引擎固定按日线（interval="d"）撮合，故 K 线取日线周期。 */}
        {task.data?.status === 'completed' && result && (
          <div style={{ marginTop: 16 }}>
            <BacktestCharts
              result={result}
              interval="d"
              start={dateRange[0].format('YYYY-MM-DD')}
              end={dateRange[1].format('YYYY-MM-DD')}
            />
          </div>
        )}
      </Col>
    </Row>
  )
}

export default AlphaBacktest
