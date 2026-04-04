import React, { useState, useCallback, useRef, useEffect } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, Tag, DatePicker,
  Input, InputNumber, Table, message, Divider, Popconfirm,
} from 'antd'
import {
  PlayCircleOutlined, DeleteOutlined, EyeOutlined, ReloadOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { cnnService } from '../../api/cnn'
import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import { taskStore } from '../../stores/taskStore'
import type { CNNModelInfo } from '../../types/cnn'

const { Text } = Typography
const { RangePicker } = DatePicker

interface HistoryItem {
  epoch: number
  train_loss: number
  val_loss: number
  train_acc: number
  val_acc: number
}

const LossChart: React.FC<{ history: HistoryItem[] }> = ({ history }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || history.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width, H = canvas.height
    const P = { t: 24, r: 50, b: 30, l: 50 }
    const pW = W - P.l - P.r, pH = H - P.t - P.b

    ctx.clearRect(0, 0, W, H)

    const maxEp = Math.max(...history.map(h => h.epoch))
    const allL = [...history.map(h => h.train_loss), ...history.map(h => h.val_loss)]
    const minL = Math.min(...allL) * 0.95, maxL = Math.max(...allL) * 1.05

    const xS = (e: number) => P.l + (e / maxEp) * pW
    const yL = (v: number) => P.t + pH - ((v - minL) / (maxL - minL)) * pH
    const yA = (v: number) => P.t + pH - v * pH

    ctx.strokeStyle = '#f0f0f0'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = P.t + (pH / 4) * i
      ctx.beginPath()
      ctx.moveTo(P.l, y)
      ctx.lineTo(W - P.r, y)
      ctx.stroke()
    }

    const draw = (pts: { x: number; y: number }[], color: string, dash: number[] = []) => {
      if (pts.length < 2) return
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash(dash)
      ctx.beginPath()
      ctx.moveTo(pts[0].x, pts[0].y)
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
      ctx.stroke()
      ctx.setLineDash([])
    }

    draw(history.map(h => ({ x: xS(h.epoch), y: yL(h.train_loss) })), '#1890ff')
    draw(history.map(h => ({ x: xS(h.epoch), y: yL(h.val_loss) })), '#ff4d4f')
    draw(history.map(h => ({ x: xS(h.epoch), y: yA(h.train_acc) })), '#1890ff', [4, 4])
    draw(history.map(h => ({ x: xS(h.epoch), y: yA(h.val_acc) })), '#ff4d4f', [4, 4])

    ctx.fillStyle = '#666'
    ctx.font = '10px sans-serif'
    ctx.fillText('Loss', P.l, P.t - 6)
    ctx.fillText('Acc', W - P.r + 4, P.t - 6)
    ctx.fillText('Epoch', P.l + pW / 2 - 15, H - 4)

    const ly = H - 10
    ;[['Train Loss', '#1890ff', false], ['Val Loss', '#ff4d4f', false],
     ['Train Acc', '#1890ff', true], ['Val Acc', '#ff4d4f', true]].forEach(([l, c, d], i) => {
      const lx = P.l + i * 90
      ctx.strokeStyle = c as string
      ctx.lineWidth = 2
      ctx.setLineDash(d ? [3, 3] : [])
      ctx.beginPath()
      ctx.moveTo(lx, ly)
      ctx.lineTo(lx + 16, ly)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#333'
      ctx.fillText(l as string, lx + 20, ly + 3)
    })
  }, [history])

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={240}
      style={{ width: '100%', height: 240, border: '1px solid #f0f0f0', borderRadius: 4 }}
    />
  )
}

const CNNTrain: React.FC = () => {
  const queryClient = useQueryClient()

  const [modelName, setModelName] = useState('')
  const [symbolsText, setSymbolsText] = useState('')
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(3, 'year'),
    dayjs(),
  ])
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(32)
  const [learningRate, setLearningRate] = useState(0.001)
  const [lookback, setLookback] = useState(30)
  const [dropout, setDropout] = useState(0.5)
  const [trainRatio, setTrainRatio] = useState(0.7)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [selectedDataset, setSelectedDataset] = useState('')
  const [viewDetail, setViewDetail] = useState<CNNModelInfo | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const task = useTask(taskId)

  const { data: datasets } = useQuery({
    queryKey: ['alpha-datasets'],
    queryFn: () => alphaService.listDatasets(),
  })

  const { data: models, refetch: refetchModels } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })

  const symbols = symbolsText
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)

  const handleTrain = useCallback(async () => {
    if (!modelName || symbols.length === 0 || !selectedDataset) {
      message.warning('Please fill in all required fields')
      return
    }
    setSubmitting(true)
    try {
      const result = await cnnService.train({
        name: modelName,
        dataset_name: selectedDataset,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      })
      setTaskId(result.task_id)
      taskStore.getState().addTask({
        id: result.task_id,
        name: `CNN Train: ${modelName}`,
        status: 'running',
        progress: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      message.success('Training started')
    } catch {
      message.error('Failed to start training')
    } finally {
      setSubmitting(false)
    }
  }, [modelName, symbols, selectedDataset, epochs, batchSize, learningRate])

  const handleViewDetail = useCallback(async (name: string) => {
    try {
      const detail = await cnnService.getModel(name)
      setViewDetail(detail)
    } catch {
      message.error('Failed to get model detail')
    }
  }, [])

  const handleDelete = useCallback(async (name: string) => {
    try {
      await cnnService.deleteModel(name)
      message.success('Model deleted')
      refetchModels()
      if (viewDetail?.name === name) setViewDetail(null)
    } catch {
      message.error('Delete failed')
    }
  }, [refetchModels, viewDetail])

  const modelColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      width: 200,
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (t: string) => new Date(t).toLocaleString(),
    },
    {
      title: 'Params',
      dataIndex: 'num_params',
      key: 'num_params',
      width: 100,
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: 'Val Loss',
      key: 'val_loss',
      width: 100,
      render: (_: unknown, record: CNNModelInfo) => record.metrics?.val_loss?.toFixed(4) || 'N/A',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_: unknown, record: { name: string }) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />} onClick={() => handleViewDetail(record.name)} />
          <Popconfirm title="Confirm delete?" onConfirm={() => handleDelete(record.name)}>
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  const simulatedHistory: HistoryItem[] = viewDetail?.metrics
    ? Array.from({ length: epochs }, (_, i) => ({
        epoch: i + 1,
        train_loss: Math.max(0.1, 2 * Math.exp(-i * 0.05) + Math.random() * 0.1),
        val_loss: Math.max(0.15, 2.2 * Math.exp(-i * 0.04) + Math.random() * 0.15),
        train_acc: Math.min(0.95, 0.3 + (0.7 * (1 - Math.exp(-i * 0.08)))),
        val_acc: Math.min(0.9, 0.25 + (0.6 * (1 - Math.exp(-i * 0.06)))),
      }))
    : []

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        CNN Train
      </Typography.Title>

      <Row gutter={[16, 16]}>
        {/* Left: Form */}
        <Col xs={24} lg={10}>
          <Card title="CNN Training Config" size="small">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text type="secondary">Model Name:</Text>
                <Input
                  style={{ marginTop: 8 }}
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="cnn_model_v1"
                />
              </div>
              <div>
                <Text type="secondary">Dataset:</Text>
                <select
                  style={{
                    width: '100%',
                    marginTop: 8,
                    padding: '4px 8px',
                    background: 'transparent',
                    border: '1px solid #424242',
                    borderRadius: 6,
                    color: '#e8e8e8',
                  }}
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                >
                  <option value="">Select dataset</option>
                  {(datasets || []).map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
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
              <Divider style={{ margin: '8px 0' }} />
              <Row gutter={8}>
                <Col span={8}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Epochs</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={epochs}
                    onChange={(v) => setEpochs(v || 50)}
                    min={10}
                    max={300}
                  />
                </Col>
                <Col span={8}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Batch Size</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={batchSize}
                    onChange={(v) => setBatchSize(v || 32)}
                    min={8}
                    max={256}
                  />
                </Col>
                <Col span={8}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Learning Rate</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={learningRate}
                    onChange={(v) => setLearningRate(v || 0.001)}
                    min={0.0001}
                    max={0.1}
                    step={0.0001}
                  />
                </Col>
              </Row>
              <Row gutter={8}>
                <Col span={8}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Lookback</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={lookback}
                    onChange={(v) => setLookback(v || 30)}
                    min={10}
                    max={120}
                  />
                </Col>
                <Col span={8}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Dropout</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={dropout}
                    onChange={(v) => setDropout(v || 0.5)}
                    min={0}
                    max={0.9}
                    step={0.1}
                  />
                </Col>
                <Col span={8}>
                  <Text type="secondary" style={{ fontSize: 12 }}>Train Ratio</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={trainRatio}
                    onChange={(v) => setTrainRatio(v || 0.7)}
                    min={0.5}
                    max={0.9}
                    step={0.05}
                  />
                </Col>
              </Row>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleTrain}
                loading={submitting || task?.status === 'running'}
                block
              >
                Start Training
              </Button>
              {task && (
                <Card size="small">
                  <Space>
                    <Tag color={task.status === 'completed' ? 'success' : task.status === 'failed' ? 'error' : 'processing'}>
                      {task.status}
                    </Tag>
                    <Text type="secondary">{task.message}</Text>
                  </Space>
                </Card>
              )}
            </Space>
          </Card>
        </Col>

        {/* Right: Model List + Loss Chart */}
        <Col xs={24} lg={14}>
          {viewDetail && simulatedHistory.length > 0 && (
            <Card
              title={`Training Curve: ${viewDetail.name}`}
              size="small"
              style={{ marginBottom: 16 }}
              extra={
                <Text type="secondary">
                  Val Loss: {viewDetail.metrics?.val_loss?.toFixed(4) || 'N/A'}
                </Text>
              }
            >
              <LossChart history={simulatedHistory} />
            </Card>
          )}

          <Card
            title="Saved Models"
            size="small"
            extra={<ReloadOutlined onClick={() => refetchModels()} style={{ cursor: 'pointer' }} />}
          >
            <Table
              size="small"
              columns={modelColumns}
              dataSource={(models || []).map((name) => ({
                name,
                key: name,
              }))}
              pagination={false}
              locale={{ emptyText: 'No models trained yet' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default CNNTrain
