import React, { useState } from 'react'
import { Alert, App, Button, Card, Col, Modal, Row, Space, Typography } from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import PlanList from './PlanList'
import PlanForm from './PlanForm'
import SchedulerStatusCard from './SchedulerStatusCard'
import SchedulerRunsCard from './SchedulerRunsCard'
import ProgressCard from './ProgressCard'
import DecisionResultCard from './DecisionResultCard'
import RebalancePlanCard from './RebalancePlanCard'
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
  /** 最近被触发的计划（用于判断 strategy_type 以切换结果渲染组件）。 */
  const [triggeredPlan, setTriggeredPlan] = useState<TradingPlan | null>(null)

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

  /**
   * 使计划列表与调度状态查询失效，触发自动重拉。
   *
   * 在计划创建/更新/删除/启停操作后调用。
   */
  const invalidatePlans = () => {
    void queryClient.invalidateQueries({ queryKey: ['live-plans'] })
    void queryClient.invalidateQueries({ queryKey: ['scheduler-status'] })
  }

  /**
   * 保存交易计划的 mutation：依据 editingPlan 是否存在自动选择更新或新建。
   *
   * editingPlan 非空时调用 updatePlan 走更新分支，为空时调用 createPlan 走新建分支。
   * 成功后弹提示、关闭表单弹窗、清空编辑态并使列表/调度状态失效重拉；失败时弹错误提示。
   */
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

  /**
   * 删除交易计划的 mutation。
   *
   * 入参为待删除计划的 plan_id；成功后弹提示并使列表/调度状态失效重拉，失败时弹错误提示。
   */
  const deleteMutation = useMutation({
    mutationFn: (planId: string) => liveService.deletePlan(planId),
    onSuccess: () => {
      message.success('计划已删除')
      invalidatePlans()
    },
    onError: (e: unknown) => message.error(e instanceof Error ? e.message : '删除失败'),
  })

  /**
   * 切换计划的自动调度开关。
   *
   * @param planId - 要切换的计划 ID
   * @param enabled - 目标状态：true=启用；false=停用
   */
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

  /**
   * 立即触发指定计划（忽略调度时间，直接产出今日决策）。
   *
   * 先拉取计划详情（getPlan）记录 strategy_type，再调用 runPlan 获取 task_id 开始轮询。
   * getPlan 失败不阻断触发，结果区默认渲染 cnn 卡片。
   * strategy_type=rule 时结果区切换为 RebalancePlanCard。
   *
   * @param planId - 要触发的计划 ID
   */
  const handleRun = async (planId: string) => {
    setRunningId(planId)
    setTriggeredPlan(null)
    try {
      // 先拉取计划详情，记录 strategy_type 以便结果区切换渲染组件
      let plan: TradingPlan | null = null
      try {
        plan = await liveService.getPlan(planId)
        setTriggeredPlan(plan)
      } catch {
        // 拉取失败不阻断触发，结果区默认渲染 cnn 卡片
      }
      const res = await liveService.runPlan(planId)
      setTaskId(res.task_id)
      message.success(res.message || '已按计划触发')
    } catch (e) {
      message.error(e instanceof Error ? e.message : '触发失败')
    } finally {
      setRunningId(null)
    }
  }

  /**
   * 加载计划详情并打开编辑弹窗。
   *
   * @param planId - 要编辑的计划 ID
   */
  const handleEdit = async (planId: string) => {
    try {
      const plan = await liveService.getPlan(planId)
      setEditingPlan(plan)
      setFormOpen(true)
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载计划失败')
    }
  }

  /** 打开新建计划弹窗（清除编辑状态）。 */
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

      <SchedulerRunsCard plans={plansQuery.data ?? []} />

      {taskId ? (
        <Card title="立即触发结果">
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <ProgressCard task={task} hasTaskId={taskId !== null} />
            {/* 根据被触发计划的 strategy_type 切换渲染：rule → RebalancePlanCard，cnn → 原有三卡 */}
            {triggeredPlan?.strategy_type === 'rule' ? (
              <RebalancePlanCard task={task} />
            ) : (
              <>
                <DecisionResultCard task={task} />
                <RiskDetailPanel task={task} />
              </>
            )}
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
