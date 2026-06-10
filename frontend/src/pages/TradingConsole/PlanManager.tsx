import React, { useState } from 'react'
import { Alert, App, Button, Card, Col, Modal, Row, Space, Typography } from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import PlanList from './PlanList'
import PlanForm from './PlanForm'
import SchedulerStatusCard from './SchedulerStatusCard'
import ProgressCard from './ProgressCard'
import DecisionResultCard from './DecisionResultCard'
import RiskDetailPanel from './RiskDetailPanel'
import { liveService } from '../../api/liveApi'
import { useTask } from '../../hooks/useTask'
import type { TradingPlan, TradingPlanRequest } from '../../types/live'

const { Text } = Typography

/**
 * 交易计划管理（任务 11）。
 *
 * 组合 PlanList（列表/启停/触发/编辑/删除）+ PlanForm（新建/编辑弹窗）+
 * SchedulerStatusCard（调度状态）+ 立即触发联动（useTask + 决策结果/风控/进度）。
 *
 * 安全约束（Req 9.5）：始终提示「仅提醒，不自动下单」；通知通道仅选名，凭证由后端管理。
 */
const PlanManager: React.FC = () => {
  const { message } = App.useApp()
  const queryClient = useQueryClient()

  const [formOpen, setFormOpen] = useState(false)
  const [editingPlan, setEditingPlan] = useState<TradingPlan | null>(null)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [runningId, setRunningId] = useState<string | null>(null)
  const [togglingId, setTogglingId] = useState<string | null>(null)

  const plansQuery = useQuery({
    queryKey: ['live-plans'],
    queryFn: () => liveService.listPlans(),
  })
  const schedulerQuery = useQuery({
    queryKey: ['scheduler-status'],
    queryFn: () => liveService.getSchedulerStatus(),
    refetchInterval: 5000,
  })

  const taskQuery = useTask(taskId)
  const task = taskQuery.data ?? null
  const isRunning = task?.status === 'running' || task?.status === 'pending'

  const invalidatePlans = () => {
    void queryClient.invalidateQueries({ queryKey: ['live-plans'] })
    void queryClient.invalidateQueries({ queryKey: ['scheduler-status'] })
  }

  const saveMutation = useMutation({
    mutationFn: (req: TradingPlanRequest) =>
      editingPlan ? liveService.updatePlan(editingPlan.plan_id, req) : liveService.createPlan(req),
    onSuccess: () => {
      message.success(editingPlan ? '计划已保存' : '计划已创建')
      setFormOpen(false)
      setEditingPlan(null)
      invalidatePlans()
    },
    onError: (e: unknown) => message.error(e instanceof Error ? e.message : '保存失败'),
  })

  const deleteMutation = useMutation({
    mutationFn: (planId: string) => liveService.deletePlan(planId),
    onSuccess: () => {
      message.success('计划已删除')
      invalidatePlans()
    },
    onError: (e: unknown) => message.error(e instanceof Error ? e.message : '删除失败'),
  })

  const handleToggle = async (planId: string, enabled: boolean) => {
    setTogglingId(planId)
    try {
      await liveService.togglePlan(planId, enabled)
      invalidatePlans()
    } catch (e) {
      message.error(e instanceof Error ? e.message : '切换失败')
    } finally {
      setTogglingId(null)
    }
  }

  const handleRun = async (planId: string) => {
    setRunningId(planId)
    try {
      const res = await liveService.runPlan(planId)
      setTaskId(res.task_id)
      message.success(res.message || '已按计划触发')
    } catch (e) {
      message.error(e instanceof Error ? e.message : '触发失败')
    } finally {
      setRunningId(null)
    }
  }

  const handleEdit = async (planId: string) => {
    try {
      const plan = await liveService.getPlan(planId)
      setEditingPlan(plan)
      setFormOpen(true)
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载计划失败')
    }
  }

  const handleNew = () => {
    setEditingPlan(null)
    setFormOpen(true)
  }

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Alert
        type="warning"
        showIcon
        message="仅提醒，不自动下单"
        description="交易计划在决策时点产出今日买卖建议并推送提醒，不会向任何券商网关提交真实订单。通知通道仅需选择通道名，webhook/secret/token 等凭证由后端环境变量管理。"
      />

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <Card
            title="交易计划"
            extra={
              <Button type="primary" icon={<PlusOutlined />} onClick={handleNew}>
                新建计划
              </Button>
            }
          >
            <PlanList
              plans={plansQuery.data || []}
              loading={plansQuery.isLoading}
              togglingId={togglingId}
              runningId={runningId}
              onToggle={(id, enabled) => void handleToggle(id, enabled)}
              onRun={(id) => void handleRun(id)}
              onEdit={(id) => void handleEdit(id)}
              onDelete={(id) => deleteMutation.mutate(id)}
            />
          </Card>
        </Col>
        <Col xs={24} xl={8}>
          <Card title="调度器状态">
            <SchedulerStatusCard status={schedulerQuery.data} loading={schedulerQuery.isLoading} />
          </Card>
        </Col>
      </Row>

      {taskId ? (
        <Card title="立即触发结果">
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <ProgressCard task={task} hasTaskId={taskId !== null} />
            <DecisionResultCard task={task} />
            <RiskDetailPanel task={task} />
            {isRunning ? <Text type="secondary">决策任务执行中…</Text> : null}
          </Space>
        </Card>
      ) : null}

      <Modal
        open={formOpen}
        title={editingPlan ? '编辑交易计划' : '新建交易计划'}
        footer={null}
        width={720}
        destroyOnClose
        onCancel={() => {
          setFormOpen(false)
          setEditingPlan(null)
        }}
      >
        <PlanForm
          initialPlan={editingPlan}
          submitting={saveMutation.isPending}
          onSubmit={(req) => saveMutation.mutate(req)}
          onCancel={() => {
            setFormOpen(false)
            setEditingPlan(null)
          }}
        />
      </Modal>
    </Space>
  )
}

export default PlanManager
