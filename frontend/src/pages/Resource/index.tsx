import React, { useMemo, useState } from 'react'
import {
  Button,
  Card,
  Col,
  Empty,
  List,
  Popconfirm,
  Row,
  Space,
  Statistic,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  DeleteOutlined,
  ExperimentOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation, useNavigate } from 'react-router-dom'

import { alphaService } from '../../api/alpha'
import { cnnService } from '../../api/cnn'
import type { DataResourceSummary } from '../../types/alpha'
import type { CNNModelInfo } from '../../types/cnn'

const { Title, Text } = Typography

/**
 * 资源管理页面：统一管理本地数据资源、Alpha 数据集、Alpha 模型、交易信号与 CNN 模型。
 *
 * 按标签页分组展示各类资源列表，支持查看详情和删除操作，
 * 并通过 `location.state.tab` 支持从其他页面跳转时预激活指定标签。
 */
const Resource: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const initialTab = (location.state as { tab?: string } | null)?.tab || 'data'
  const [activeTab, setActiveTab] = useState(initialTab)

  const { data: resources } = useQuery({
    queryKey: ['alpha-data-resources'],
    queryFn: () => alphaService.getDataResources(),
  })

  const { data: datasets } = useQuery({
    queryKey: ['alpha-datasets'],
    queryFn: () => alphaService.listDatasets(),
  })

  const { data: alphaModels } = useQuery({
    queryKey: ['alpha-models'],
    queryFn: () => alphaService.listModels(),
  })

  const { data: signals } = useQuery({
    queryKey: ['alpha-signals'],
    queryFn: () => alphaService.listSignals(),
  })

  const { data: cnnModels } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })

  const allDataResources = useMemo(
    () => [
      ...(resources?.raw_bars || []),
      ...(resources?.raw_ticks || []),
      ...(resources?.derived_bars || []),
    ],
    [resources],
  )

  const deleteResource = async (resource: DataResourceSummary) => {
    try {
      await alphaService.deleteDataResource(resource.kind, resource.key)
      message.success('资源已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
    } catch {
      message.error('删除失败')
    }
  }

  const deleteDataset = async (name: string) => {
    try {
      await alphaService.deleteDataset(name)
      message.success('数据集已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-datasets'] })
    } catch {
      message.error('删除失败')
    }
  }

  const deleteAlphaModel = async (name: string) => {
    try {
      await alphaService.deleteModel(name)
      message.success('模型已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-models'] })
    } catch {
      message.error('删除失败')
    }
  }

  const deleteSignal = async (name: string) => {
    try {
      await alphaService.deleteSignal(name)
      message.success('信号已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-signals'] })
    } catch {
      message.error('删除失败')
    }
  }

  const deleteCnnModel = async (name: string) => {
    try {
      await cnnService.deleteModel(name)
      message.success('CNN 模型已删除')
      queryClient.invalidateQueries({ queryKey: ['cnn-models'] })
    } catch {
      message.error('删除失败')
    }
  }

  const renderResourceList = () => (
    allDataResources.length > 0 ? (
      <List
        itemLayout="vertical"
        dataSource={allDataResources}
        renderItem={(resource) => (
          <List.Item
            key={resource.key}
            actions={[
              resource.kind !== 'raw_tick'
                ? (
                  <Button
                    key="train"
                    type="link"
                    onClick={() => navigate('/cnn-train', {
                      state: {
                        preset: {
                          target_symbol: resource.vt_symbol,
                          input_data_kind: resource.kind === 'raw_tick' ? 'tick' : 'bar',
                          input_interval: resource.target_interval || resource.interval,
                          symbols: [resource.vt_symbol],
                        },
                      },
                    })}
                  >
                    用于 CNN 训练
                  </Button>
                )
                : null,
              <Popconfirm key="delete" title="确认删除这个资源？" onConfirm={() => deleteResource(resource)}>
                <Button type="link" danger icon={<DeleteOutlined />}>删除</Button>
              </Popconfirm>,
            ]}
          >
            <List.Item.Meta
              title={
                <Space wrap>
                  <Text strong>{resource.vt_symbol}</Text>
                  <Tag color={resource.kind === 'raw_tick' ? 'gold' : resource.kind === 'derived_bar' ? 'purple' : 'blue'}>
                    {resource.kind === 'raw_tick' ? 'Tick' : resource.kind === 'derived_bar' ? '派生K线' : '原始K线'}
                  </Tag>
                  <Tag>{resource.interval}</Tag>
                </Space>
              }
              description={
                <Space direction="vertical" size={2}>
                  <Text type="secondary">
                    {resource.start} ~ {resource.end}
                  </Text>
                  <Text type="secondary">
                    {resource.row_count.toLocaleString()} 行 · {resource.file_size_kb.toFixed(1)} KB
                    {resource.kind === 'derived_bar'
                      ? ` · ${resource.source_kind}:${resource.source_interval} → ${resource.target_interval}`
                      : ''}
                  </Text>
                </Space>
              }
            />
          </List.Item>
        )}
      />
    ) : (
      <Empty description="还没有数据资源" />
    )
  )

  const renderNameList = (
    items: string[] | undefined,
    emptyText: string,
    onDelete: (name: string) => void,
    actionLabel: string,
    onJump: (name: string) => void,
  ) => (
    items && items.length > 0 ? (
      <List
        dataSource={items}
        renderItem={(name) => (
          <List.Item
            actions={[
              <Button key="jump" type="link" onClick={() => onJump(name)}>{actionLabel}</Button>,
              <Popconfirm key="delete" title="确认删除？" onConfirm={() => onDelete(name)}>
                <Button type="link" danger>删除</Button>
              </Popconfirm>,
            ]}
          >
            <Text strong>{name}</Text>
          </List.Item>
        )}
      />
    ) : (
      <Empty description={emptyText} />
    )
  )

  const renderCnnModels = (
    cnnModels && cnnModels.length > 0 ? (
      <List<CNNModelInfo>
        dataSource={cnnModels}
        renderItem={(model) => (
          <List.Item
            actions={[
              <Button key="view" type="link" onClick={() => navigate('/cnn-train', { state: { modelName: model.name } })}>
                查看训练详情
              </Button>,
              <Popconfirm key="delete" title="确认删除这个 CNN 模型？" onConfirm={() => deleteCnnModel(model.name)}>
                <Button type="link" danger>删除</Button>
              </Popconfirm>,
            ]}
          >
            <List.Item.Meta
              avatar={<ExperimentOutlined style={{ fontSize: 20, color: '#faad14' }} />}
              title={
                <Space wrap>
                  <Text strong>{model.name}</Text>
                  <Tag>{model.input_interval || 'd'}</Tag>
                  <Tag color="purple">{model.group_count || 1} 组</Tag>
                </Space>
              }
              description={
                <Space direction="vertical" size={2}>
                  <Text type="secondary">
                    目标证券：{model.target_symbol || '-'} · 输入：{model.input_data_kind || 'bar'}
                  </Text>
                  <Text type="secondary">
                    最佳验证损失：{model.best_val_loss?.toFixed(4) || '-'} · 创建时间：{new Date(model.created_at).toLocaleString()}
                  </Text>
                </Space>
              }
            />
          </List.Item>
        )}
      />
    ) : (
      <Empty description="还没有 CNN 模型" />
    )
  )

  return (
    <div className="page-enter">
      <Space direction="vertical" size={20} style={{ width: '100%' }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>资源库</Title>
          <Text type="secondary">
            统一查看原始数据、派生周期、模型和信号，并从资源直接跳回下一步工作流。
          </Text>
        </div>

        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="数据资源" value={allDataResources.length} prefix={<FolderOpenOutlined />} /></Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="Alpha 数据集" value={datasets?.length || 0} /></Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="Alpha / CNN 模型" value={(alphaModels?.length || 0) + (cnnModels?.length || 0)} /></Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="交易信号" value={signals?.length || 0} /></Card>
          </Col>
        </Row>

        <Card
          extra={
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
                queryClient.invalidateQueries({ queryKey: ['alpha-datasets'] })
                queryClient.invalidateQueries({ queryKey: ['alpha-models'] })
                queryClient.invalidateQueries({ queryKey: ['alpha-signals'] })
                queryClient.invalidateQueries({ queryKey: ['cnn-models'] })
              }}
            >
              刷新资源
            </Button>
          }
        >
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              {
                key: 'data',
                label: `数据资源 (${allDataResources.length})`,
                children: renderResourceList(),
              },
              {
                key: 'datasets',
                label: `Alpha 数据集 (${datasets?.length || 0})`,
                children: renderNameList(
                  datasets,
                  '还没有 Alpha 数据集',
                  deleteDataset,
                  '去模型训练',
                  (name) => navigate('/model-train', { state: { datasetName: name } }),
                ),
              },
              {
                key: 'alpha-models',
                label: `Alpha 模型 (${alphaModels?.length || 0})`,
                children: renderNameList(
                  alphaModels,
                  '还没有 Alpha 模型',
                  deleteAlphaModel,
                  '去信号生成',
                  (name) => navigate('/signal', { state: { modelName: name } }),
                ),
              },
              {
                key: 'cnn-models',
                label: `CNN 模型 (${cnnModels?.length || 0})`,
                children: renderCnnModels,
              },
              {
                key: 'signals',
                label: `交易信号 (${signals?.length || 0})`,
                children: renderNameList(
                  signals,
                  '还没有交易信号',
                  deleteSignal,
                  '去回测',
                  (name) => navigate('/backtest', { state: { signalName: name } }),
                ),
              },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default Resource
