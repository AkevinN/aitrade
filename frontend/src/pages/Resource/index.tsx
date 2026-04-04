import React, { useState, useCallback } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, Tabs, Table, Tag, Popconfirm, message,
} from 'antd'
import {
  DatabaseOutlined, RobotOutlined, ThunderboltOutlined, DeleteOutlined, ReloadOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'

import { alphaService } from '../../api/alpha'
import { cnnService } from '../../api/cnn'

const { Text } = Typography

interface DatasetItem {
  name: string
  created_at: string
  size: number
  num_samples: number
  symbols: string[]
  fields: string[]
}

interface ModelItem {
  name: string
  model_type: string
  dataset_name: string
  created_at: string
  num_features: number
  metrics?: Record<string, number>
}

interface SignalItem {
  name: string
  model_name: string
  created_at: string
  num_signals: number
}

const Resource: React.FC = () => {
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState('datasets')

  const { data: datasets, refetch: refetchDatasets } = useQuery({
    queryKey: ['alpha-datasets'],
    queryFn: () => alphaService.listDatasets(),
  })

  const { data: models, refetch: refetchModels } = useQuery({
    queryKey: ['alpha-models'],
    queryFn: () => alphaService.listModels(),
  })

  const { data: signals, refetch: refetchSignals } = useQuery({
    queryKey: ['alpha-signals'],
    queryFn: () => alphaService.listSignals(),
  })

  const { data: cnnModels, refetch: refetchCnnModels } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })

  const handleDeleteDataset = useCallback(async (name: string) => {
    try {
      await alphaService.deleteDataset(name)
      message.success('Dataset deleted')
      refetchDatasets()
    } catch {
      message.error('Delete failed')
    }
  }, [refetchDatasets])

  const handleDeleteModel = useCallback(async (name: string) => {
    try {
      await alphaService.deleteModel(name)
      message.success('Model deleted')
      refetchModels()
    } catch {
      message.error('Delete failed')
    }
  }, [refetchModels])

  const handleDeleteSignal = useCallback(async (name: string) => {
    try {
      await alphaService.deleteSignal(name)
      message.success('Signal deleted')
      refetchSignals()
    } catch {
      message.error('Delete failed')
    }
  }, [refetchSignals])

  const handleDeleteCnnModel = useCallback(async (name: string) => {
    try {
      await cnnService.deleteModel(name)
      message.success('CNN model deleted')
      refetchCnnModels()
    } catch {
      message.error('Delete failed')
    }
  }, [refetchCnnModels])

  const datasetColumns = [
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
      title: 'Size',
      dataIndex: 'size',
      key: 'size',
      width: 100,
      render: (v: number) => {
        if (v > 1024 * 1024) return `${(v / (1024 * 1024)).toFixed(2)} MB`
        if (v > 1024) return `${(v / 1024).toFixed(2)} KB`
        return `${v} B`
      },
    },
    {
      title: 'Samples',
      dataIndex: 'num_samples',
      key: 'num_samples',
      width: 100,
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: 'Symbols',
      dataIndex: 'symbols',
      key: 'symbols',
      render: (arr: string[]) => arr.slice(0, 3).join(', ') + (arr.length > 3 ? '...' : ''),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_: unknown, record: { name: string }) => (
        <Popconfirm title="Confirm delete?" onConfirm={() => handleDeleteDataset(record.name)}>
          <Button size="small" danger icon={<DeleteOutlined />}>
            Delete
          </Button>
        </Popconfirm>
      ),
    },
  ]

  const modelColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      width: 180,
    },
    {
      title: 'Type',
      dataIndex: 'model_type',
      key: 'model_type',
      width: 80,
      render: (t: string) => <Tag>{t}</Tag>,
    },
    {
      title: 'Dataset',
      dataIndex: 'dataset_name',
      key: 'dataset_name',
      width: 150,
    },
    {
      title: 'Features',
      dataIndex: 'num_features',
      key: 'num_features',
      width: 80,
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (t: string) => new Date(t).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_: unknown, record: { name: string }) => (
        <Popconfirm title="Confirm delete?" onConfirm={() => handleDeleteModel(record.name)}>
          <Button size="small" danger icon={<DeleteOutlined />}>
            Delete
          </Button>
        </Popconfirm>
      ),
    },
  ]

  const signalColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      width: 200,
    },
    {
      title: 'Model',
      dataIndex: 'model_name',
      key: 'model_name',
      width: 150,
    },
    {
      title: 'Signals',
      dataIndex: 'num_signals',
      key: 'num_signals',
      width: 100,
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (t: string) => new Date(t).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_: unknown, record: { name: string }) => (
        <Popconfirm title="Confirm delete?" onConfirm={() => handleDeleteSignal(record.name)}>
          <Button size="small" danger icon={<DeleteOutlined />}>
            Delete
          </Button>
        </Popconfirm>
      ),
    },
  ]

  const cnnModelColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      width: 200,
    },
    {
      title: 'Type',
      key: 'type',
      width: 80,
      render: () => <Tag color="orange">CNN</Tag>,
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (t: string) => new Date(t).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_: unknown, record: { name: string }) => (
        <Popconfirm title="Confirm delete?" onConfirm={() => handleDeleteCnnModel(record.name)}>
          <Button size="small" danger icon={<DeleteOutlined />}>
            Delete
          </Button>
        </Popconfirm>
      ),
    },
  ]

  const tabItems = [
    {
      key: 'datasets',
      label: (
        <Space>
          <DatabaseOutlined />
          Datasets ({(datasets?.length || 0)})
        </Space>
      ),
      children: (
        <Table
          size="small"
          columns={datasetColumns}
          dataSource={(datasets || []).map((name) => ({
            name,
            key: name,
            created_at: new Date().toISOString(),
            size: 0,
            num_samples: 0,
            symbols: [],
            fields: [],
          }))}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: 'No datasets' }}
        />
      ),
    },
    {
      key: 'models',
      label: (
        <Space>
          <RobotOutlined />
          Models ({(models?.length || 0) + (cnnModels?.length || 0)})
        </Space>
      ),
      children: (
        <>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>ML Models</Text>
          <Table
            size="small"
            columns={modelColumns}
            dataSource={(models || []).map((name) => ({
              name,
              key: name,
              model_type: 'ml',
              dataset_name: '-',
              num_features: 0,
              created_at: new Date().toISOString(),
            }))}
            pagination={false}
            locale={{ emptyText: 'No ML models' }}
            style={{ marginBottom: 16 }}
          />
          <Text strong style={{ display: 'block', marginBottom: 8 }}>CNN Models</Text>
          <Table
            size="small"
            columns={cnnModelColumns}
            dataSource={(cnnModels || []).map((name) => ({
              name,
              key: name,
              created_at: new Date().toISOString(),
            }))}
            pagination={false}
            locale={{ emptyText: 'No CNN models' }}
          />
        </>
      ),
    },
    {
      key: 'signals',
      label: (
        <Space>
          <ThunderboltOutlined />
          Signals ({(signals?.length || 0)})
        </Space>
      ),
      children: (
        <Table
          size="small"
          columns={signalColumns}
          dataSource={(signals || []).map((name) => ({
            name,
            key: name,
            model_name: '-',
            num_signals: 0,
            created_at: new Date().toISOString(),
          }))}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: 'No signals' }}
        />
      ),
    },
  ]

  return (
    <div className="page-enter">
      <Typography.Title level={4} style={{ marginBottom: 20 }}>
        Resource Management
      </Typography.Title>

      <Card
        size="small"
        extra={
          <Space>
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => {
                queryClient.invalidateQueries({ queryKey: ['alpha-datasets'] })
                queryClient.invalidateQueries({ queryKey: ['alpha-models'] })
                queryClient.invalidateQueries({ queryKey: ['alpha-signals'] })
                queryClient.invalidateQueries({ queryKey: ['cnn-models'] })
              }}
            >
              Refresh All
            </Button>
          </Space>
        }
      >
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabItems}
        />
      </Card>
    </div>
  )
}

export default Resource
