import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu, Space, Typography } from 'antd'
import {
  DashboardOutlined,
  DatabaseOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  FolderOutlined,
  FundOutlined,
  ControlOutlined,
  PieChartOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import DataPrepare from './pages/DataPrepare'
import ModelTrain from './pages/ModelTrain'
import CNNTrain from './pages/CNNTrain'
import CNNGovernance from './pages/CNNGovernance'
import Signal from './pages/Signal'
import Backtest from './pages/Backtest'
import Resource from './pages/Resource'
import TradingConsole from './pages/TradingConsole'
import Portfolio from './pages/Portfolio'
import { useTaskList } from './hooks/useTask'

const { Header, Sider, Content } = Layout
const { Text } = Typography

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '工作台' },
  { key: '/data-prepare', icon: <DatabaseOutlined />, label: '数据准备' },
  { key: '/model-train', icon: <RobotOutlined />, label: '模型训练' },
  { key: '/cnn-train', icon: <ExperimentOutlined />, label: 'CNN训练' },
  { key: '/cnn-governance', icon: <ControlOutlined />, label: 'CNN治理' },
  { key: '/signal', icon: <ThunderboltOutlined />, label: '信号分析' },
  { key: '/backtest', icon: <LineChartOutlined />, label: '回测' },
  { key: '/trading-console', icon: <FundOutlined />, label: '交易操作台' },
  { key: '/portfolio', icon: <PieChartOutlined />, label: '策略组合' },
  { key: '/resource', icon: <FolderOutlined />, label: '资源管理' },
]

const pageMeta: Record<string, { title: string; description: string }> = {
  '/': {
    title: '工作台',
    description: '查看当前准备度、任务状态和关键入口。',
  },
  '/data-prepare': {
    title: '数据准备',
    description: '获取原始行情、导入 Tick，并生成可训练的派生周期。',
  },
  '/model-train': {
    title: '模型训练',
    description: '创建 Alpha 数据集并训练传统因子模型。',
  },
  '/cnn-train': {
    title: 'CNN 训练',
    description: '配置观测组、标签规则和输入周期，训练卷积模型。',
  },
  '/cnn-governance': {
    title: 'CNN 治理',
    description: '滚动评估、候选模型、半自动晋级、回滚与治理回放回测。',
  },
  '/signal': {
    title: '信号分析',
    description: '生成并检查预测信号，准备下游评估。',
  },
  '/backtest': {
    title: '回测',
    description: '加载信号并验证策略表现。',
  },
  '/trading-console': {
    title: '交易操作台',
    description: '临近收盘基于 CNN 预测产出今日决策与风控明细，仅提醒不下单。',
  },
  '/portfolio': {
    title: '策略组合',
    description: '查看组合持仓账本、熔断状态与历史调仓记录。',
  },
  '/resource': {
    title: '资源管理',
    description: '集中查看数据、模型和信号资源。',
  },
}

/**
 * 应用外层布局（侧边导航 + 顶部标题栏 + 内容区）。
 *
 * - 侧边导航宽度自适应：≥768px 展开 200px；<768px 折叠至 72px 图标模式。
 * - 顶部标题栏展示当前页面标题、描述与运行中任务计数（来自 useTaskList）。
 * - children 渲染路由出口（由 App 组件传入）。
 *
 * @param children - 页面路由出口内容
 */
const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const { data: tasks } = useTaskList()
  const [isNarrow, setIsNarrow] = React.useState(
    () => typeof window !== 'undefined' && window.innerWidth < 768,
  )
  const runningCount = tasks?.filter((task) => task.status === 'running').length || 0
  const meta = pageMeta[location.pathname] || pageMeta['/']

  React.useEffect(() => {
    const handleResize = () => setIsNarrow(window.innerWidth < 768)
    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <Layout style={{ height: '100vh', overflow: 'hidden' }}>
      <Sider
        width={isNarrow ? 72 : 200}
        collapsed={isNarrow}
        collapsedWidth={72}
        style={{
          height: '100vh',
          overflow: 'auto',
          background: '#1a1a1a',
          borderRight: '1px solid #303030',
        }}
      >
        <div
          style={{
            minHeight: 72,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #303030',
            padding: '12px 16px',
          }}
        >
          <Space direction="vertical" size={0} style={{ width: '100%' }}>
	            <Text
	              strong
	              style={{
	                color: '#1668dc',
	                fontSize: 18,
	                letterSpacing: 0.6,
	              }}
	            >
	              {isNarrow ? 'AIT' : 'AITrade'}
	            </Text>
            {!isNarrow ? <Text style={{ color: '#8c8c8c', fontSize: 12 }}>
              量化研究工作台
            </Text> : null}
          </Space>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{
            background: 'transparent',
            border: 'none',
            marginTop: 8,
          }}
        />
      </Sider>

      <Layout style={{ minWidth: 0, height: '100vh', overflow: 'hidden' }}>
        <Header
          style={{
            height: 'auto',
            lineHeight: 'normal',
            background: 'rgba(20, 20, 20, 0.92)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid #303030',
            padding: isNarrow ? '12px 14px' : '14px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 16,
          }}
        >
          <Space direction="vertical" size={0}>
            <Text style={{ color: '#8c8c8c', fontSize: 12 }}>
              AITrade / {meta.title}
            </Text>
            <Text style={{ color: '#f0f0f0', fontSize: 18, fontWeight: 600 }}>
              {meta.title}
            </Text>
            <Text style={{ color: '#8c8c8c', fontSize: 12 }}>
              {meta.description}
            </Text>
          </Space>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid #303030',
              borderRadius: 999,
              padding: '8px 14px',
              background: '#111b26',
              color: runningCount > 0 ? '#91caff' : '#8c8c8c',
              fontSize: 12,
              whiteSpace: 'nowrap',
            }}
          >
            {runningCount > 0 ? `运行中任务 ${runningCount}` : '当前没有运行中的任务'}
          </div>
        </Header>
        <Content
	          style={{
	            flex: 1,
	            background: '#0a0a0a',
	            minWidth: 0,
            minHeight: 0,
            overflow: 'auto',
            padding: isNarrow ? '14px 14px 24px' : '20px 24px 32px',
          }}
        >
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

/**
 * 应用根组件，挂载路由树。
 *
 * 新增路由（本分支）：`/portfolio` → `Portfolio`（策略组合页面）。
 * 其余路由与既有页面保持不变。
 */
const App: React.FC = () => {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/data-prepare" element={<DataPrepare />} />
        <Route path="/model-train" element={<ModelTrain />} />
        <Route path="/cnn-train" element={<CNNTrain />} />
        <Route path="/cnn-governance" element={<CNNGovernance />} />
        <Route path="/signal" element={<Signal />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/trading-console" element={<TradingConsole />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/resource" element={<Resource />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppLayout>
  )
}

export default App
