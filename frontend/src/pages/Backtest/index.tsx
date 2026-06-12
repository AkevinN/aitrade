import React from 'react'
import { Alert, Button, Space, Tabs } from 'antd'
import {
  LineChartOutlined, ExperimentOutlined, ControlOutlined,
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import AlphaBacktest from './AlphaBacktest'
import CNNBacktest from './CNNBacktest'
import RuleBacktest from './RuleBacktest'

/**
 * 回测页面（Tab 容器）。
 *
 * 包含四个标签：
 * - Alpha 因子回测
 * - CNN 模型回测
 * - 规则策略回测（本分支新增）
 * - 治理回放回测（跳转至 CNN 治理页）
 */
const Backtest: React.FC = () => {
  const navigate = useNavigate()
  return (
    <div className="page-enter">
      <Tabs
        size="large"
        items={[
          {
            key: 'alpha',
            label: (
              <span>
                <LineChartOutlined />
                Alpha 因子回测
              </span>
            ),
            children: <AlphaBacktest />,
          },
          {
            key: 'cnn',
            label: (
              <span>
                <ExperimentOutlined />
                CNN 模型回测
              </span>
            ),
            children: <CNNBacktest />,
          },
          {
            key: 'rule',
            label: (
              <span>
                <LineChartOutlined />
                规则策略回测
              </span>
            ),
            children: <RuleBacktest />,
          },
          {
            key: 'cnn-governance-replay',
            label: (
              <span>
                <ControlOutlined />
                治理回放回测
              </span>
            ),
            children: (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Alert
                  type="info"
                  showIcon
                  message="验证模型更新机制，而不只是验证单个模型"
                  description="治理回放会对比固定初始模型、无脑定期重训、治理筛选晋级和买入持有，判断半自动晋级机制在历史上是否真正改善收益、回撤和稳定性。"
                />
                <Button type="primary" onClick={() => navigate('/cnn-governance')}>
                  前往 CNN 治理页运行回放
                </Button>
              </Space>
            ),
          },
        ]}
      />
    </div>
  )
}

export default Backtest
