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

  // 把原始K线、原始Tick、派生K线三类资源合并成统一的扁平列表，供数据资源标签页与顶部统计卡共用；任一桶缺失时按空数组处理。
  const allDataResources = useMemo(
    () => [
      ...(resources?.raw_bars || []),
      ...(resources?.raw_ticks || []),
      ...(resources?.derived_bars || []),
    ],
    [resources],
  )

  /**
   * 删除一条本地数据资源，成功后刷新数据资源列表。
   *
   * 失败时只弹出错误提示、不抛出（异常被吞掉），调用方无需 try/catch。
   *
   * @param resource - 待删除资源；用其 kind 与 key 定位后端记录
   */
  const deleteResource = async (resource: DataResourceSummary) => {
    try {
      await alphaService.deleteDataResource(resource.kind, resource.key)
      message.success('资源已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
    } catch {
      message.error('删除失败')
    }
  }

  /**
   * 删除指定 Alpha 数据集，成功后刷新数据集列表。
   *
   * 失败时只弹出错误提示、不抛出（异常被吞掉）。
   *
   * @param name - 数据集名称，作为后端删除接口的唯一标识
   */
  const deleteDataset = async (name: string) => {
    try {
      await alphaService.deleteDataset(name)
      message.success('数据集已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-datasets'] })
    } catch {
      message.error('删除失败')
    }
  }

  /**
   * 删除指定 Alpha 模型，成功后刷新 Alpha 模型列表。
   *
   * 失败时只弹出错误提示、不抛出（异常被吞掉）。
   *
   * @param name - 模型名称，作为后端删除接口的唯一标识
   */
  const deleteAlphaModel = async (name: string) => {
    try {
      await alphaService.deleteModel(name)
      message.success('模型已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-models'] })
    } catch {
      message.error('删除失败')
    }
  }

  /**
   * 删除指定交易信号，成功后刷新信号列表。
   *
   * 失败时只弹出错误提示、不抛出（异常被吞掉）。
   *
   * @param name - 信号名称，作为后端删除接口的唯一标识
   */
  const deleteSignal = async (name: string) => {
    try {
      await alphaService.deleteSignal(name)
      message.success('信号已删除')
      queryClient.invalidateQueries({ queryKey: ['alpha-signals'] })
    } catch {
      message.error('删除失败')
    }
  }

  /**
   * 删除指定 CNN 模型，成功后刷新 CNN 模型列表。
   *
   * 失败时只弹出错误提示、不抛出（异常被吞掉）。
   *
   * @param name - 模型名称，作为后端删除接口的唯一标识
   */
  const deleteCnnModel = async (name: string) => {
    try {
      await cnnService.deleteModel(name)
      message.success('CNN 模型已删除')
      queryClient.invalidateQueries({ queryKey: ['cnn-models'] })
    } catch {
      message.error('删除失败')
    }
  }

  /**
   * 渲染「数据资源」标签页：列出全部 K 线/Tick 资源，附带标的、类型、周期与起止区间等元信息。
   *
   * 非 Tick 资源额外提供「用于 CNN 训练」入口；列表为空时渲染占位提示。
   *
   * @returns 资源列表或空状态的 React 节点
   */
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

  /**
   * 渲染一个「仅按名称列举」的资源列表，供数据集/Alpha 模型/信号等同构标签页复用。
   *
   * 每行提供一个跳转到下一步工作流的按钮和一个带二次确认的删除按钮。
   *
   * @param items - 资源名称列表；为 undefined 或空数组时渲染空状态
   * @param emptyText - 列表为空时的占位文案
   * @param onDelete - 点击删除并确认后调用，入参为该行资源名
   * @param actionLabel - 跳转按钮的文案，如「去模型训练」
   * @param onJump - 点击跳转按钮时调用，入参为该行资源名
   * @returns 名称列表或空状态的 React 节点
   */
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

  // 「CNN 模型」标签页内容：逐个列出模型的名称、输入周期、分组数、目标证券、最佳验证损失与创建时间，并提供查看训练详情/删除入口；为空时渲染占位提示。
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
