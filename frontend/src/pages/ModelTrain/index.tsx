import React, { useState, useCallback } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, Tag, Select, DatePicker,
  Input, InputNumber, Progress, Table, message, Checkbox,
} from 'antd'
import {
  DatabaseOutlined, RobotOutlined, PlayCircleOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import { taskStore } from '../../stores/taskStore'

const { Text } = Typography
const { RangePicker } = DatePicker
const CheckableTag = Tag.CheckableTag

type TabKey = 'dataset' | 'train'

const ModelTrain: React.FC = () => {
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<TabKey>('dataset')

  // Dataset form state
  const [dsName, setDsName] = useState('')
  const [dsSymbolsText, setDsSymbolsText] = useState('')
  const [dsDateRange, setDsDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(1, 'year'),
    dayjs(),
  ])
  const [dsTrainEnd, setDsTrainEnd] = useState<dayjs.Dayjs>(dayjs().subtract(3, 'month'))
  const [dsFeatures, setDsFeatures] = useState<string[]>(['alpha158'])
  const [dsLabelPeriod, setDsLabelPeriod] = useState(5)
  const [datasetTaskId, setDatasetTaskId] = useState<string | null>(null)

  // Train form state
  const [modelName, setModelName] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [modelType, setModelType] = useState<'lgb' | 'mlp' | 'lasso'>('lgb')
  const [trainTaskId, setTrainTaskId] = useState<string | null>(null)

  // Params based on model type
  const [lgbParams, setLgbParams] = useState({ num_leaves: 31, learning_rate: 0.05, n_estimators: 100 })
  const [mlpParams, setMlpParams] = useState({ hidden_layer_sizes: 100, max_iter: 200 })
  const [lassoParams, setLassoParams] = useState({ alpha: 1.0 })

  const datasetTask = useTask(datasetTaskId)
  const trainTask = useTask(trainTaskId)

  const { data: datasets } = useQuery({
    queryKey: ['alpha-datasets'],
    queryFn: () => alphaService.listDatasets(),
  })

  const { data: models, refetch: refetchModels } = useQuery({
    queryKey: ['alpha-models'],
    queryFn: () => alphaService.listModels(),
  })

  const handleCreateDataset = useCallback(async () => {
    if (!dsName || !dsSymbolsText.trim()) {
      message.warning('Please enter dataset name and symbols')
      return
    }
    const symbols = dsSymbolsText.split(/[\n,]+/).map((s) => s.trim()).filter((s) => s.length > 0)
    if (symbols.length === 0) {
      message.warning('Please enter at least one symbol')
      return
    }
    try {
      const result = await alphaService.createDataset({
        name: dsName,
        vt_symbols: symbols,
        start: dsDateRange[0].format('YYYY-MM-DD'),
        end: dsDateRange[1].format('YYYY-MM-DD'),
        train_end: dsTrainEnd.format('YYYY-MM-DD'),
        features: dsFeatures,
        label_period: dsLabelPeriod,
      })
      setDatasetTaskId(result.id)
      taskStore.getState().addTask({
        id: result.id,
        name: `Create Dataset: ${dsName}`,
        status: 'running',
        progress: 0,
        message: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      message.success('Dataset creation started')
    } catch {
      message.error('Failed to create dataset')
    }
  }, [dsName, dsSymbolsText, dsDateRange, dsFeatures, dsLabelPeriod])

  const handleTrainModel = useCallback(async () => {
    if (!modelName || !selectedDataset) {
      message.warning('Please enter model name and select dataset')
      return
    }
    let params: Record<string, unknown> = {}
    switch (modelType) {
      case 'lgb':
        params = lgbParams
        break
      case 'mlp':
        params = mlpParams
        break
      case 'lasso':
        params = lassoParams
        break
    }
    try {
      const result = await alphaService.trainModel({
        name: modelName,
        dataset: selectedDataset,
        model_type: modelType,
        params,
      })
      setTrainTaskId(result.id)
      taskStore.getState().addTask({
        id: result.id,
        name: `Train Model: ${modelName}`,
        status: 'running',
        progress: 0,
        message: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      message.success('Model training started')
    } catch {
      message.error('Failed to start training')
    }
  }, [modelName, selectedDataset, modelType, lgbParams, mlpParams, lassoParams])

  const modelColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Type',
      dataIndex: 'model_type',
      key: 'model_type',
      render: (t: string) => <Tag>{t}</Tag>,
    },
    {
      title: 'Dataset',
      dataIndex: 'dataset_name',
      key: 'dataset_name',
    },
    {
      title: 'Features',
      dataIndex: 'num_features',
      key: 'num_features',
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (t: string) => new Date(t).toLocaleString(),
    },
  ]

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        Model Train
      </Typography.Title>

      <Row gutter={[16, 16]}>
        {/* Left: Forms */}
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {/* Create Dataset */}
            <Card
              title={<><DatabaseOutlined /> Create Dataset</>}
              extra={
                <Button size="small" onClick={() => setActiveTab('dataset')}>
                  {activeTab === 'dataset' ? 'Hide' : 'Show'}
                </Button>
              }
            >
              {activeTab === 'dataset' && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text type="secondary">Name:</Text>
                    <Input
                      style={{ marginTop: 8 }}
                      value={dsName}
                      onChange={(e) => setDsName(e.target.value)}
                      placeholder="my_dataset"
                    />
                  </div>
                  <div>
                    <Text type="secondary">Symbols (one per line):</Text>
                    <Input.TextArea
                      style={{ marginTop: 8 }}
                      rows={3}
                      value={dsSymbolsText}
                      onChange={(e) => setDsSymbolsText(e.target.value)}
                      placeholder="000001.SZSE&#10;600000.SSE"
                    />
                  </div>
                  <div>
                    <Text type="secondary">Date Range:</Text>
                    <RangePicker
                      style={{ width: '100%', marginTop: 8 }}
                      value={dsDateRange}
                      onChange={(dates) => dates && setDsDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
                    />
                  </div>
                  <div>
                    <Text type="secondary">Train End Date:</Text>
                    <DatePicker
                      style={{ width: '100%', marginTop: 8 }}
                      value={dsTrainEnd}
                      onChange={(d) => d && setDsTrainEnd(d)}
                    />
                  </div>
                  <div>
                    <Text type="secondary">Features:</Text>
                    <Checkbox.Group
                      style={{ marginTop: 8 }}
                      value={dsFeatures}
                      onChange={(vals) => setDsFeatures(vals as string[])}
                      options={[
                        { label: 'Alpha158', value: 'alpha158' },
                        { label: 'Alpha101', value: 'alpha101' },
                      ]}
                    />
                  </div>
                  <div>
                    <Text type="secondary">Label Period:</Text>
                    <InputNumber
                      style={{ marginTop: 8, width: '100%' }}
                      value={dsLabelPeriod}
                      onChange={(v) => setDsLabelPeriod(v || 5)}
                      min={1}
                      max={30}
                    />
                  </div>
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    onClick={handleCreateDataset}
                    loading={datasetTask?.data?.status === 'running'}
                    block
                  >
                    Create Dataset
                  </Button>
                  {datasetTask?.data && (
                    <Card size="small">
                      <Progress
                        percent={Math.round(datasetTask.data.progress)}
                        status={datasetTask.data.status === 'failed' ? 'exception' : 'active'}
                      />
                      <Text type="secondary">{datasetTask.data.status}</Text>
                    </Card>
                  )}
                </Space>
              )}
            </Card>

            {/* Train Model */}
            <Card
              title={<><RobotOutlined /> Train Model</>}
              extra={
                <Button size="small" onClick={() => setActiveTab('train')}>
                  {activeTab === 'train' ? 'Hide' : 'Show'}
                </Button>
              }
            >
              {activeTab === 'train' && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text type="secondary">Model Name:</Text>
                    <Input
                      style={{ marginTop: 8 }}
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      placeholder="my_model_v1"
                    />
                  </div>
                  <div>
                    <Text type="secondary">Dataset:</Text>
                    <Select
                      style={{ width: '100%', marginTop: 8 }}
                      value={selectedDataset || undefined}
                      onChange={setSelectedDataset}
                      placeholder="Select dataset"
                      options={(datasets || []).map((d) => ({ label: d, value: d }))}
                    />
                  </div>
                  <div>
                    <Text type="secondary">Model Type:</Text>
                    <Space style={{ marginTop: 8 }}>
                      <CheckableTag
                        checked={modelType === 'lgb'}
                        onChange={() => setModelType('lgb')}
                      >
                        LightGBM
                      </CheckableTag>
                      <CheckableTag
                        checked={modelType === 'mlp'}
                        onChange={() => setModelType('mlp')}
                      >
                        MLP
                      </CheckableTag>
                      <CheckableTag
                        checked={modelType === 'lasso'}
                        onChange={() => setModelType('lasso')}
                      >
                        Lasso
                      </CheckableTag>
                    </Space>
                  </div>

                  {modelType === 'lgb' && (
                    <Card size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text type="secondary">LightGBM Parameters:</Text>
                        <Row gutter={8}>
                          <Col span={8}>
                            <Text type="secondary" style={{ fontSize: 12 }}>num_leaves</Text>
                            <InputNumber
                              style={{ width: '100%' }}
                              value={lgbParams.num_leaves}
                              onChange={(v) => setLgbParams({ ...lgbParams, num_leaves: v || 31 })}
                              min={1}
                            />
                          </Col>
                          <Col span={8}>
                            <Text type="secondary" style={{ fontSize: 12 }}>learning_rate</Text>
                            <InputNumber
                              style={{ width: '100%' }}
                              value={lgbParams.learning_rate}
                              onChange={(v) => setLgbParams({ ...lgbParams, learning_rate: v || 0.05 })}
                              min={0.001}
                              max={1}
                              step={0.01}
                            />
                          </Col>
                          <Col span={8}>
                            <Text type="secondary" style={{ fontSize: 12 }}>n_estimators</Text>
                            <InputNumber
                              style={{ width: '100%' }}
                              value={lgbParams.n_estimators}
                              onChange={(v) => setLgbParams({ ...lgbParams, n_estimators: v || 100 })}
                              min={10}
                            />
                          </Col>
                        </Row>
                      </Space>
                    </Card>
                  )}

                  {modelType === 'mlp' && (
                    <Card size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text type="secondary">MLP Parameters:</Text>
                        <Row gutter={8}>
                          <Col span={12}>
                            <Text type="secondary" style={{ fontSize: 12 }}>hidden_layer_sizes</Text>
                            <InputNumber
                              style={{ width: '100%' }}
                              value={mlpParams.hidden_layer_sizes}
                              onChange={(v) => setMlpParams({ ...mlpParams, hidden_layer_sizes: v || 100 })}
                              min={10}
                            />
                          </Col>
                          <Col span={12}>
                            <Text type="secondary" style={{ fontSize: 12 }}>max_iter</Text>
                            <InputNumber
                              style={{ width: '100%' }}
                              value={mlpParams.max_iter}
                              onChange={(v) => setMlpParams({ ...mlpParams, max_iter: v || 200 })}
                              min={50}
                            />
                          </Col>
                        </Row>
                      </Space>
                    </Card>
                  )}

                  {modelType === 'lasso' && (
                    <Card size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text type="secondary">Lasso Parameters:</Text>
                        <InputNumber
                          style={{ width: '100%' }}
                          value={lassoParams.alpha}
                          onChange={(v) => setLassoParams({ alpha: v || 1.0 })}
                          min={0.001}
                          max={10}
                          step={0.1}
                        />
                      </Space>
                    </Card>
                  )}

                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    onClick={handleTrainModel}
                    loading={trainTask?.data?.status === 'running'}
                    block
                  >
                    Train Model
                  </Button>
                  {trainTask?.data && (
                    <Card size="small">
                      <Progress
                        percent={Math.round(trainTask.data.progress)}
                        status={trainTask.data.status === 'failed' ? 'exception' : 'active'}
                      />
                      <Text type="secondary">{trainTask.data.status}</Text>
                    </Card>
                  )}
                </Space>
              )}
            </Card>
          </Space>
        </Col>

        {/* Right: Model List */}
        <Col xs={24} lg={12}>
          <Card title="Trained Models" size="small">
            <Table
              size="small"
              columns={modelColumns}
              dataSource={(models || []).map((m) => ({ name: m, key: m }))}
              pagination={false}
              locale={{ emptyText: 'No models trained yet' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default ModelTrain
