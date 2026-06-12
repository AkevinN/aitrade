import React, { useMemo, useState, useEffect } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, Tag, Select, DatePicker,
  Input, InputNumber, Table, message,
} from 'antd'
import {
  DatabaseOutlined, RobotOutlined, PlayCircleOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import TaskStatusPanel from '../../components/TaskStatusPanel'

const { Text } = Typography
const { RangePicker } = DatePicker
const CheckableTag = Tag.CheckableTag

/**
 * Alpha 模型训练页面：创建数据集、配置并启动 LGB/MLP/Lasso 模型训练，展示任务进度。
 */
const ModelTrain: React.FC = () => {
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

  const [modelName, setModelName] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [modelType, setModelType] = useState<'lgb' | 'mlp' | 'lasso'>('lgb')
  const [trainTaskId, setTrainTaskId] = useState<string | null>(null)

  const [lgbParams, setLgbParams] = useState({
    num_leaves: 31,
    learning_rate: 0.05,
    num_boost_round: 200,
  })
  const [mlpHiddenSizes, setMlpHiddenSizes] = useState('256,128')
  const [mlpParams, setMlpParams] = useState({
    lr: 0.001,
    n_epochs: 200,
    batch_size: 1024,
    early_stop_rounds: 30,
    eval_steps: 10,
    optimizer: 'adam' as 'adam' | 'sgd',
    weight_decay: 0,
    device: 'auto' as 'auto' | 'cpu' | 'cuda',
  })
  const [lassoParams, setLassoParams] = useState({ alpha: 0.0005 })

  const datasetTask = useTask(datasetTaskId)
  const trainTask = useTask(trainTaskId)

  const { data: datasets, refetch: refetchDatasets } = useQuery({
    queryKey: ['alpha-datasets'],
    queryFn: () => alphaService.listDatasets(),
  })

  const { data: models, refetch: refetchModels } = useQuery({
    queryKey: ['alpha-models'],
    queryFn: () => alphaService.listModels(),
  })

  useEffect(() => {
    if (datasetTask.data?.status === 'completed') {
      refetchDatasets()
    }
  }, [datasetTask.data?.status, refetchDatasets])

  useEffect(() => {
    if (trainTask.data?.status === 'completed') {
      refetchModels()
    }
  }, [trainTask.data?.status, refetchModels])

  const datasetSymbols = useMemo(() => (
    dsSymbolsText
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
  ), [dsSymbolsText])

  const handleCreateDataset = async () => {
    if (!dsName || datasetSymbols.length === 0) {
      message.warning('请输入数据集名称和证券代码')
      return
    }

    try {
      const result = await alphaService.createDataset({
        name: dsName,
        vt_symbols: datasetSymbols,
        start: dsDateRange[0].format('YYYY-MM-DD'),
        end: dsDateRange[1].format('YYYY-MM-DD'),
        train_end: dsTrainEnd.format('YYYY-MM-DD'),
        features: dsFeatures,
        label_period: dsLabelPeriod,
      })
      setDatasetTaskId(result.task_id)
      message.success('数据集创建任务已启动')
    } catch (error) {
      console.error(error)
      message.error('创建数据集失败')
    }
  }

  const handleTrainModel = async () => {
    if (!modelName || !selectedDataset) {
      message.warning('请输入模型名称并选择数据集')
      return
    }

    let params: Record<string, unknown>
    if (modelType === 'lgb') {
      params = lgbParams
    } else if (modelType === 'mlp') {
      const hiddenSizes = mlpHiddenSizes
        .split(',')
        .map((item) => Number(item.trim()))
        .filter((item) => Number.isFinite(item) && item > 0)

      params = {
        ...mlpParams,
        hidden_sizes: hiddenSizes,
      }
      if (mlpParams.device === 'auto') {
        delete params.device
      }
    } else {
      params = lassoParams
    }

    try {
      const result = await alphaService.trainModel({
        name: modelName,
        dataset: selectedDataset,
        model_type: modelType,
        params,
      })
      setTrainTaskId(result.task_id)
      message.success('模型训练任务已启动')
    } catch (error) {
      console.error(error)
      message.error('启动训练失败')
    }
  }

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        Alpha 模型训练
      </Typography.Title>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Card title={<><DatabaseOutlined /> 创建数据集</>} size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text type="secondary">数据集名称：</Text>
                  <Input
                    style={{ marginTop: 8 }}
                    value={dsName}
                    onChange={(e) => setDsName(e.target.value)}
                    placeholder="my_dataset"
                  />
                </div>
                <div>
                  <Text type="secondary">证券列表（每行一个）：</Text>
                  <Input.TextArea
                    style={{ marginTop: 8 }}
                    rows={3}
                    value={dsSymbolsText}
                    onChange={(e) => setDsSymbolsText(e.target.value)}
                    placeholder={'000001.SZSE\n600000.SSE'}
                  />
                </div>
                <div>
                  <Text type="secondary">日期范围：</Text>
                  <RangePicker
                    style={{ width: '100%', marginTop: 8 }}
                    value={dsDateRange}
                    onChange={(dates) => dates && setDsDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
                  />
                </div>
                <div>
                  <Text type="secondary">训练集截止日期：</Text>
                  <DatePicker
                    style={{ width: '100%', marginTop: 8 }}
                    value={dsTrainEnd}
                    onChange={(value) => value && setDsTrainEnd(value)}
                  />
                </div>
                <div>
                  <Text type="secondary">特征集：</Text>
                  <Space style={{ marginTop: 8 }}>
                    <CheckableTag
                      checked={dsFeatures.includes('alpha158')}
                      onChange={(checked) => {
                        setDsFeatures((current) => (
                          checked
                            ? [...new Set([...current, 'alpha158'])]
                            : current.filter((item) => item !== 'alpha158')
                        ))
                      }}
                    >
                      Alpha158
                    </CheckableTag>
                    <CheckableTag
                      checked={dsFeatures.includes('alpha101')}
                      onChange={(checked) => {
                        setDsFeatures((current) => (
                          checked
                            ? [...new Set([...current, 'alpha101'])]
                            : current.filter((item) => item !== 'alpha101')
                        ))
                      }}
                    >
                      Alpha101
                    </CheckableTag>
                  </Space>
                </div>
                <div>
                  <Text type="secondary">标签周期：</Text>
                  <InputNumber
                    style={{ width: '100%', marginTop: 8 }}
                    min={1}
                    max={30}
                    value={dsLabelPeriod}
                    onChange={(value) => setDsLabelPeriod(value || 5)}
                  />
                </div>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleCreateDataset}
                  loading={datasetTask.data?.status === 'running'}
                  block
                >
                  创建数据集
                </Button>
                <TaskStatusPanel task={datasetTask.data || null} title="数据集创建任务" />
              </Space>
            </Card>

            <Card title={<><RobotOutlined /> 训练模型</>} size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text type="secondary">模型名称：</Text>
                  <Input
                    style={{ marginTop: 8 }}
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    placeholder="my_model_v1"
                  />
                </div>
                <div>
                  <Text type="secondary">数据集：</Text>
                  <Select
                    style={{ width: '100%', marginTop: 8 }}
                    value={selectedDataset || undefined}
                    onChange={setSelectedDataset}
                    placeholder="选择数据集"
                    options={(datasets || []).map((name) => ({ label: name, value: name }))}
                  />
                </div>
                <div>
                  <Text type="secondary">模型类型：</Text>
                  <Space style={{ marginTop: 8 }}>
                    <CheckableTag checked={modelType === 'lgb'} onChange={() => setModelType('lgb')}>
                      LightGBM
                    </CheckableTag>
                    <CheckableTag checked={modelType === 'mlp'} onChange={() => setModelType('mlp')}>
                      MLP
                    </CheckableTag>
                    <CheckableTag checked={modelType === 'lasso'} onChange={() => setModelType('lasso')}>
                      Lasso
                    </CheckableTag>
                  </Space>
                </div>

                {modelType === 'lgb' && (
                  <Card size="small">
                    <Row gutter={8}>
                      <Col span={8}>
                        <Text type="secondary" style={{ fontSize: 12 }}>num_leaves</Text>
                        <InputNumber
                          style={{ width: '100%' }}
                          min={2}
                          value={lgbParams.num_leaves}
                          onChange={(value) => setLgbParams((current) => ({ ...current, num_leaves: value || 31 }))}
                        />
                      </Col>
                      <Col span={8}>
                        <Text type="secondary" style={{ fontSize: 12 }}>learning_rate</Text>
                        <InputNumber
                          style={{ width: '100%' }}
                          min={0.001}
                          step={0.01}
                          value={lgbParams.learning_rate}
                          onChange={(value) => setLgbParams((current) => ({ ...current, learning_rate: value || 0.05 }))}
                        />
                      </Col>
                      <Col span={8}>
                        <Text type="secondary" style={{ fontSize: 12 }}>num_boost_round</Text>
                        <InputNumber
                          style={{ width: '100%' }}
                          min={10}
                          value={lgbParams.num_boost_round}
                          onChange={(value) => setLgbParams((current) => ({ ...current, num_boost_round: value || 200 }))}
                        />
                      </Col>
                    </Row>
                  </Card>
                )}

                {modelType === 'mlp' && (
                  <Card size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>hidden_sizes（逗号分隔）</Text>
                        <Input
                          style={{ marginTop: 4 }}
                          value={mlpHiddenSizes}
                          onChange={(e) => setMlpHiddenSizes(e.target.value)}
                          placeholder="256,128"
                        />
                      </div>
                      <Row gutter={8}>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>学习率</Text>
                          <InputNumber
                            style={{ width: '100%' }}
                            min={0.0001}
                            step={0.0001}
                            value={mlpParams.lr}
                            onChange={(value) => setMlpParams((current) => ({ ...current, lr: value || 0.001 }))}
                          />
                        </Col>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>训练轮数</Text>
                          <InputNumber
                            style={{ width: '100%' }}
                            min={10}
                            value={mlpParams.n_epochs}
                            onChange={(value) => setMlpParams((current) => ({ ...current, n_epochs: value || 200 }))}
                          />
                        </Col>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>批大小</Text>
                          <InputNumber
                            style={{ width: '100%' }}
                            min={32}
                            value={mlpParams.batch_size}
                            onChange={(value) => setMlpParams((current) => ({ ...current, batch_size: value || 1024 }))}
                          />
                        </Col>
                      </Row>
                      <Row gutter={8}>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>早停轮数</Text>
                          <InputNumber
                            style={{ width: '100%' }}
                            min={5}
                            value={mlpParams.early_stop_rounds}
                            onChange={(value) => setMlpParams((current) => ({ ...current, early_stop_rounds: value || 30 }))}
                          />
                        </Col>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>评估步长</Text>
                          <InputNumber
                            style={{ width: '100%' }}
                            min={1}
                            value={mlpParams.eval_steps}
                            onChange={(value) => setMlpParams((current) => ({ ...current, eval_steps: value || 10 }))}
                          />
                        </Col>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>权重衰减</Text>
                          <InputNumber
                            style={{ width: '100%' }}
                            min={0}
                            step={0.0001}
                            value={mlpParams.weight_decay}
                            onChange={(value) => setMlpParams((current) => ({ ...current, weight_decay: value || 0 }))}
                          />
                        </Col>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>优化器</Text>
                          <Select
                            style={{ width: '100%' }}
                            value={mlpParams.optimizer}
                            onChange={(value) => setMlpParams((current) => ({ ...current, optimizer: value }))}
                            options={[
                              { label: 'Adam', value: 'adam' },
                              { label: 'SGD', value: 'sgd' },
                            ]}
                          />
                        </Col>
                      </Row>
                      <Row gutter={8}>
                        <Col span={8}>
                          <Text type="secondary" style={{ fontSize: 12 }}>设备</Text>
                          <Select
                            style={{ width: '100%' }}
                            value={mlpParams.device}
                            onChange={(value) => setMlpParams((current) => ({ ...current, device: value }))}
                            options={[
                              { label: '自动', value: 'auto' },
                              { label: 'CPU', value: 'cpu' },
                              { label: 'CUDA', value: 'cuda' },
                            ]}
                          />
                        </Col>
                      </Row>
                    </Space>
                  </Card>
                )}

                {modelType === 'lasso' && (
                  <Card size="small">
                    <Text type="secondary" style={{ fontSize: 12 }}>正则化系数 alpha</Text>
                    <InputNumber
                      style={{ width: '100%', marginTop: 4 }}
                      min={0.0001}
                      step={0.0001}
                      value={lassoParams.alpha}
                      onChange={(value) => setLassoParams({ alpha: value || 0.0005 })}
                    />
                  </Card>
                )}

                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleTrainModel}
                  loading={trainTask.data?.status === 'running'}
                  block
                >
                  启动训练
                </Button>
                <TaskStatusPanel task={trainTask.data || null} title="模型训练任务" />
              </Space>
            </Card>
          </Space>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="已训练模型" size="small">
            <Table
              size="small"
              dataSource={(models || []).map((name) => ({ key: name, name }))}
              columns={[
                { title: '模型名称', dataIndex: 'name', key: 'name' },
              ]}
              pagination={false}
              locale={{ emptyText: '还没有模型' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default ModelTrain
