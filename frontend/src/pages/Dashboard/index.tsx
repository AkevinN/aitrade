import React, { useEffect, useMemo } from 'react'
import { Row, Col, Card, Statistic, Typography, Space, Table, Tag, Divider } from 'antd'
import {
  DatabaseOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'

import { alphaService } from '../../api/alpha'
import { cnnService } from '../../api/cnn'
import { useTaskList } from '../../hooks/useTask'
import { taskStore } from '../../stores/taskStore'

const { Title, Text } = Typography

const Dashboard: React.FC = () => {
  const navigate = useNavigate()
  const { data: alphaStatus } = useQuery({
    queryKey: ['alpha-status'],
    queryFn: () => alphaService.getStatus(),
  })

  const { data: cnnModels } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })

  const { data: alphaModels } = useQuery({
    queryKey: ['alpha-models'],
    queryFn: () => alphaService.listModels(),
  })

  const { data: signals } = useQuery({
    queryKey: ['alpha-signals'],
    queryFn: () => alphaService.listSignals(),
  })

  const { data: barData } = useQuery({
    queryKey: ['alpha-bar-data'],
    queryFn: () => alphaService.getBarData(),
  })

  const { data: tasks } = useTaskList()

  const recentTasks = useMemo(() => {
    if (!tasks) return []
    return [...tasks]
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      .slice(0, 10)
  }, [tasks])

  const totalDatasets = useMemo(() => {
    if (!barData) return 0
    return (barData.daily?.length || 0) + (barData.minute?.length || 0)
  }, [barData])

  const totalModels = useMemo(() => {
    return (alphaModels?.length || 0) + (cnnModels?.length || 0)
  }, [alphaModels, cnnModels])

  const totalSignals = useMemo(() => {
    return signals?.length || 0
  }, [signals])

  const taskColumns = [
    {
      title: '任务ID',
      dataIndex: 'task_id',
      key: 'task_id',
      width: 100,
      ellipsis: true,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 120,
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          pending: 'default',
          running: 'processing',
          completed: 'success',
          failed: 'error',
        }
        return <Tag color={colorMap[status] || 'default'}>{status}</Tag>
      },
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 80,
      render: (progress: number) => `${Math.round(progress)}%`,
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      width: 150,
      render: (time: string) => new Date(time).toLocaleString(),
    },
  ]

  return (
    <div className="page-enter">
      <Title level={4} style={{ marginBottom: 20 }}>
        AI Trade Dashboard
      </Title>

      {/* Stat Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 10 }}>
            <Statistic
              title="Total Datasets"
              value={totalDatasets}
              prefix={<DatabaseOutlined style={{ color: '#1668dc' }} />}
              valueStyle={{ fontSize: 24, fontWeight: 600 }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              Daily: {(barData?.daily?.length || 0)} | Minute: {(barData?.minute?.length || 0)}
            </Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 10 }}>
            <Statistic
              title="Total Models"
              value={totalModels}
              prefix={<RobotOutlined style={{ color: '#a78bfa' }} />}
              valueStyle={{ fontSize: 24, fontWeight: 600 }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              ML: {(alphaModels?.length || 0)} | CNN: {(cnnModels?.length || 0)}
            </Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 10 }}>
            <Statistic
              title="Total Signals"
              value={totalSignals}
              prefix={<ThunderboltOutlined style={{ color: '#f59e0b' }} />}
              valueStyle={{ fontSize: 24, fontWeight: 600 }}
            />
            <Text type="secondary" style={{ fontSize: 12 }}>
              Generated from ML models
            </Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card variant="borderless" style={{ borderRadius: 10 }}>
            <Space direction="vertical" size={4}>
              <Space>
                {alphaStatus?.installed ? (
                  <CheckCircleOutlined style={{ color: '#49aa19' }} />
                ) : (
                  <WarningOutlined style={{ color: '#dc4446' }} />
                )}
                <Text>Alpha Module</Text>
                <Tag color={alphaStatus?.installed ? 'success' : 'error'}>
                  {alphaStatus?.installed ? 'Installed' : 'Not Installed'}
                </Tag>
              </Space>
              <Space>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Version: {alphaStatus?.version || 'N/A'}
                </Text>
              </Space>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Recent Tasks */}
      <Card
        title="Recent Tasks"
        size="small"
        style={{ marginBottom: 16 }}
        extra={<Text type="secondary">{recentTasks.length} tasks shown</Text>}
      >
        <Table
          size="small"
          columns={taskColumns}
          dataSource={recentTasks}
          pagination={false}
          rowKey="task_id"
          locale={{ emptyText: 'No tasks found' }}
        />
      </Card>

      {/* Quick Links */}
      <Card title="Quick Links" size="small">
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Space wrap>
              <Tag
                color="blue"
                style={{ cursor: 'pointer', padding: '8px 16px', fontSize: 14 }}
                onClick={() => navigate('/data-prepare')}
              >
                <DatabaseOutlined /> Data Prepare
              </Tag>
              <Tag
                color="purple"
                style={{ cursor: 'pointer', padding: '8px 16px', fontSize: 14 }}
                onClick={() => navigate('/model-train')}
              >
                <RobotOutlined /> Model Train
              </Tag>
              <Tag
                color="orange"
                style={{ cursor: 'pointer', padding: '8px 16px', fontSize: 14 }}
                onClick={() => navigate('/cnn-train')}
              >
                <RobotOutlined /> CNN Train
              </Tag>
              <Tag
                color="gold"
                style={{ cursor: 'pointer', padding: '8px 16px', fontSize: 14 }}
                onClick={() => navigate('/signal')}
              >
                <ThunderboltOutlined /> Signal
              </Tag>
              <Tag
                color="green"
                style={{ cursor: 'pointer', padding: '8px 16px', fontSize: 14 }}
                onClick={() => navigate('/backtest')}
              >
                <ThunderboltOutlined /> Backtest
              </Tag>
              <Tag
                color="cyan"
                style={{ cursor: 'pointer', padding: '8px 16px', fontSize: 14 }}
                onClick={() => navigate('/resource')}
              >
                <DatabaseOutlined /> Resources
              </Tag>
            </Space>
          </Col>
        </Row>
      </Card>
    </div>
  )
}

export default Dashboard
