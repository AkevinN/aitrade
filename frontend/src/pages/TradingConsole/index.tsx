import React, { useState } from 'react'
import { Alert, Card, Col, Row, Space, Tabs, Typography } from 'antd'

import ConfigForm from './ConfigForm'
import ProgressCard from './ProgressCard'
import DecisionResultCard from './DecisionResultCard'
import RiskDetailPanel from './RiskDetailPanel'
import HistoryTable from './HistoryTable'
import PlanManager from './PlanManager'
import { useTask } from '../../hooks/useTask'

const { Title, Text } = Typography

/**
 * 交易操作台（Trading Console）页面骨架。
 *
 * 本文件仅搭建页面容器与各区块占位，具体子组件由后续任务填充：
 *  - 9.2 ConfigForm 配置表单（模型/标的/方案/决策日/数据源/组合快照）
 *  - 9.3 进度联动（复用既有 WS + taskStore）
 *  - 9.4 DecisionResultCard 决策结果卡片
 *  - 9.5 RiskDetailPanel 风控明细面板
 *  - 9.6 HistoryTable 历史决策表
 *
 * 安全约束（Req 7.4）：决策结果区始终提示「仅提醒，不自动下单」。
 */
const TradingConsole: React.FC = () => {
  // task_id 提升到页面层，供进度联动（9.3）、决策结果（9.4）、风控明细（9.5）消费。
  const [taskId, setTaskId] = useState<string | null>(null)

  // 复用既有 task 机制（useTask 轮询 + WS task_update 写入同一 Task 结构）订阅该 task_id 的进度。
  // task 对象提升到页面层，9.4/9.5 将读取 task.result.decision / task.result.risk_detail。
  const taskQuery = useTask(taskId)
  const task = taskQuery.data ?? null
  const isRunning = task?.status === 'running' || task?.status === 'pending'

  return (
    <div className="page-enter">
      <Space direction="vertical" size={20} style={{ width: '100%' }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>
            交易操作台
          </Title>
          <Text type="secondary">
            临近收盘时，基于 CNN 预测产出今日买卖决策与风控明细。仅提醒，不自动下单。
          </Text>
        </div>

        <Tabs
          defaultActiveKey="manual"
          items={[
            {
              key: 'manual',
              label: '手动决策',
              children: (
                <Space direction="vertical" size={16} style={{ width: '100%' }}>
                  <Alert
                    type="warning"
                    showIcon
                    message="仅提醒，不自动下单"
                    description="本页面只产出决策建议与提醒，不会向任何券商网关提交真实订单。当前后端处于无鉴权环境，真实下单能力依赖鉴权与 kill-switch UI 前置条件。"
                  />

                  <Row gutter={[16, 16]}>
                    {/* 左列：配置表单（任务 9.2）+ 进度（任务 9.3） */}
                    <Col xs={24} xl={10}>
                      <Space direction="vertical" size={16} style={{ width: '100%' }}>
                        <Card title="决策配置">
                          <ConfigForm
                            onStarted={setTaskId}
                            hasTriggered={taskId !== null}
                            running={isRunning}
                          />
                        </Card>

                        <Card title="任务进度">
                          <ProgressCard task={task} hasTaskId={taskId !== null} />
                        </Card>
                      </Space>
                    </Col>

                    {/* 右列：决策结果（9.4）+ 风控明细（9.5）+ 历史（9.6） */}
                    <Col xs={24} xl={14}>
                      <Space direction="vertical" size={16} style={{ width: '100%' }}>
                        <Card title="决策结果">
                          <DecisionResultCard task={task} />
                        </Card>

                        <Card title="风控明细">
                          <RiskDetailPanel task={task} />
                        </Card>

                        <Card title="历史决策">
                          <HistoryTable task={task} />
                        </Card>
                      </Space>
                    </Col>
                  </Row>
                </Space>
              ),
            },
            {
              key: 'plans',
              label: '交易计划',
              children: <PlanManager />,
            },
          ]}
        />
      </Space>
    </div>
  )
}

export default TradingConsole
