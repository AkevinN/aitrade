import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu, Typography } from 'antd'
import {
  DashboardOutlined,
  DatabaseOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  FolderOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import DataPrepare from './pages/DataPrepare'
import ModelTrain from './pages/ModelTrain'
import CNNTrain from './pages/CNNTrain'
import Signal from './pages/Signal'
import Backtest from './pages/Backtest'
import Resource from './pages/Resource'

const { Header, Sider, Content } = Layout
const { Text } = Typography

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: 'Dashboard' },
  { key: '/data-prepare', icon: <DatabaseOutlined />, label: '数据准备' },
  { key: '/model-train', icon: <RobotOutlined />, label: '模型训练' },
  { key: '/cnn-train', icon: <ExperimentOutlined />, label: 'CNN训练' },
  { key: '/signal', icon: <ThunderboltOutlined />, label: '信号分析' },
  { key: '/backtest', icon: <LineChartOutlined />, label: '回测' },
  { key: '/resource', icon: <FolderOutlined />, label: '资源管理' },
]

const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        width={200}
        style={{
          background: '#1a1a1a',
          borderRight: '1px solid #303030',
        }}
      >
        <div
          style={{
            height: 48,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #303030',
          }}
        >
          <Text
            strong
            style={{
              color: '#1668dc',
              fontSize: 16,
              letterSpacing: 1,
            }}
          >
            AITrade
          </Text>
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

      <Layout>
        <Header
          style={{
            background: '#141414',
            borderBottom: '1px solid #303030',
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Text style={{ color: '#e8e8e8', fontSize: 14 }}>
            AI 量化投研平台
          </Text>
        </Header>
        <Content
          style={{
            padding: 24,
            background: '#0a0a0a',
            minHeight: 'calc(100vh - 48px)',
          }}
        >
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

const App: React.FC = () => {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/data-prepare" element={<DataPrepare />} />
        <Route path="/model-train" element={<ModelTrain />} />
        <Route path="/cnn-train" element={<CNNTrain />} />
        <Route path="/signal" element={<Signal />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/resource" element={<Resource />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppLayout>
  )
}

export default App
