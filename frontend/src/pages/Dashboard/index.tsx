import React, { useMemo } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  List,
  Row,
  Space,
  Statistic,
  Tag,
  Typography,
} from 'antd'
import {
  ArrowRightOutlined,
  BarChartOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FundOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'

import { alphaService } from '../../api/alpha'
import { cnnService } from '../../api/cnn'
import { useTaskList } from '../../hooks/useTask'

const { Title, Text } = Typography

const Dashboard: React.FC = () => {
  const navigate = useNavigate()

  const { data: status } = useQuery({
    queryKey: ['alpha-status'],
    queryFn: () => alphaService.getStatus(),
  })

  const { data: resources } = useQuery({
    queryKey: ['alpha-data-resources'],
    queryFn: () => alphaService.getDataResources(),
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

  const { data: tasks } = useTaskList()

  const recentTasks = useMemo(() => {
    if (!tasks) return []
    return [...tasks]
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      .slice(0, 6)
  }, [tasks])

  const failedTasks = recentTasks.filter((task) => task.status === 'failed')
  const rawBarCount = resources?.raw_bars.length || 0
  const rawTickCount = resources?.raw_ticks.length || 0
  const derivedCount = resources?.derived_bars.length || 0
  const cnnCount = cnnModels?.length || 0

  const readinessCards = [
    {
      title: '原始K线',
      value: rawBarCount,
      icon: <DatabaseOutlined style={{ color: '#1677ff' }} />,
      hint: rawBarCount > 0 ? `已覆盖 ${resources?.raw_bar_intervals.join(' / ') || '多个周期'}` : '先下载日线或分钟线',
      action: () => navigate('/data-prepare'),
      actionLabel: rawBarCount > 0 ? '继续补数据' : '去下载K线',
    },
    {
      title: '历史Tick',
      value: rawTickCount,
      icon: <FundOutlined style={{ color: '#00b96b' }} />,
      hint: rawTickCount > 0 ? '已可生成自定义分钟周期' : '导入 Tick 后才能做更细粒度聚合',
      action: () => navigate('/data-prepare', { state: { focus: 'tick-import' } }),
      actionLabel: rawTickCount > 0 ? '继续导入 Tick' : '导入 Tick',
    },
    {
      title: '派生周期',
      value: derivedCount,
      icon: <BarChartOutlined style={{ color: '#fa8c16' }} />,
      hint: derivedCount > 0 ? `已有 ${resources?.derived_intervals.join(' / ')}` : '从 1m 或 Tick 聚合 5m / 10m / 30m',
      action: () => navigate('/data-prepare', { state: { focus: 'aggregate' } }),
      actionLabel: derivedCount > 0 ? '继续做聚合' : '创建派生周期',
    },
    {
      title: 'CNN模型',
      value: cnnCount,
      icon: <ExperimentOutlined style={{ color: '#b37feb' }} />,
      hint: cnnCount > 0 ? '可直接查看训练细节或继续迭代' : '配置目标股与观测组后启动训练',
      action: () => navigate('/cnn-train'),
      actionLabel: cnnCount > 0 ? '继续训练 CNN' : '新建 CNN 训练',
    },
  ]

  return (
    <div className="page-enter">
      <Space direction="vertical" size={20} style={{ width: '100%' }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>数据就绪面板</Title>
          <Text type="secondary">
            先准备原始数据，再生成派生周期，最后把目标股和市场语义组装进 CNN 训练。
          </Text>
        </div>

        {!status?.installed ? (
          <Alert type="error" showIcon message="Alpha 模块未安装，当前无法使用数据与训练功能" />
        ) : null}

        {failedTasks.length > 0 ? (
          <Alert
            type="warning"
            showIcon
            message={`最近有 ${failedTasks.length} 个任务失败`}
            description={failedTasks.map((task) => `${task.title || task.type}: ${task.message}`).join('；')}
          />
        ) : null}

        <Row gutter={[16, 16]}>
          {readinessCards.map((card) => (
            <Col key={card.title} xs={24} sm={12} xl={6}>
              <Card
                variant="borderless"
                style={{ borderRadius: 20, minHeight: 210 }}
                styles={{ body: { display: 'flex', flexDirection: 'column', gap: 16 } }}
              >
                <Space size="middle" align="start">
                  <div style={{ fontSize: 24 }}>{card.icon}</div>
                  <div>
                    <Text type="secondary">{card.title}</Text>
                    <Statistic value={card.value} valueStyle={{ fontSize: 28, marginTop: 4 }} />
                  </div>
                </Space>
                <Text type="secondary" style={{ minHeight: 44 }}>{card.hint}</Text>
                <Button type="primary" icon={<ArrowRightOutlined />} onClick={card.action}>
                  {card.actionLabel}
                </Button>
              </Card>
            </Col>
          ))}
        </Row>

        <Row gutter={[16, 16]}>
          <Col xs={24} xl={14}>
            <Card
              title="推荐下一步"
              extra={<Button type="link" onClick={() => navigate('/resource')}>查看全部资源</Button>}
              styles={{ body: { display: 'flex', flexDirection: 'column', gap: 12 } }}
            >
              {rawBarCount === 0 ? (
                <Alert
                  type="info"
                  showIcon
                  message="还没有原始K线"
                  description="先到数据准备页下载日线或分钟线，后续训练和聚合都依赖这里。"
                  action={<Button size="small" onClick={() => navigate('/data-prepare')}>去下载</Button>}
                />
              ) : null}
              {rawBarCount > 0 && rawTickCount === 0 ? (
                <Alert
                  type="info"
                  showIcon
                  message="可以补充历史 Tick"
                  description="Tick 可以聚合出更灵活的 5m / 10m / 30m 输入，适合做更细粒度的 CNN。"
                  action={<Button size="small" onClick={() => navigate('/data-prepare', { state: { focus: 'tick-import' } })}>导入 Tick</Button>}
                />
              ) : null}
              {rawBarCount > 0 && derivedCount === 0 ? (
                <Alert
                  type="success"
                  showIcon
                  message="原始数据已就绪"
                  description="下一步建议从 1m 或 Tick 聚合出一个训练用周期，比如 5m 或 10m。"
                  action={<Button size="small" onClick={() => navigate('/data-prepare', { state: { focus: 'aggregate' } })}>做聚合</Button>}
                />
              ) : null}
              {derivedCount > 0 && cnnCount === 0 ? (
                <Alert
                  type="success"
                  showIcon
                  message="派生周期可用"
                  description="现在可以在 CNN 页面配置目标股、大盘、板块、龙头等观测组，并选择标签定义。"
                  action={<Button size="small" type="primary" onClick={() => navigate('/cnn-train')}>开始训练</Button>}
                />
              ) : null}
              {cnnCount > 0 ? (
                <Alert
                  type="success"
                  showIcon
                  message="CNN 模型已经可用"
                  description="你可以继续迭代观测组配置，或者在资源页统一管理已有训练结果。"
                  action={<Button size="small" onClick={() => navigate('/resource', { state: { tab: 'cnn-models' } })}>查看模型</Button>}
                />
              ) : null}
            </Card>
          </Col>

          <Col xs={24} xl={10}>
            <Card title="其它资源概览">
              <Row gutter={[12, 12]}>
                <Col span={12}>
                  <Card size="small" styles={{ body: { padding: 16 } }}>
                    <Statistic title="Alpha模型" value={alphaModels?.length || 0} />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small" styles={{ body: { padding: 16 } }}>
                    <Statistic title="交易信号" value={signals?.length || 0} prefix={<ThunderboltOutlined />} />
                  </Card>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>

        <Card title="最近任务" extra={<Button type="link" onClick={() => navigate('/resource')}>去资源页</Button>}>
          {recentTasks.length > 0 ? (
            <List
              dataSource={recentTasks}
              renderItem={(task) => (
                <List.Item
                  actions={[
                    <Tag key="status" color={task.status === 'failed' ? 'error' : task.status === 'completed' ? 'success' : 'processing'}>
                      {task.status}
                    </Tag>,
                  ]}
                >
                  <List.Item.Meta
                    title={task.title || task.type}
                    description={
                      <Space direction="vertical" size={2}>
                        <Text type="secondary">{task.message || '等待任务状态更新'}</Text>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {new Date(task.updated_at).toLocaleString()}
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          ) : (
            <Empty description="还没有任务记录" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default Dashboard
