// 策略组合页——持仓账本、熔断状态、调仓历史三合一视图（Task 5.4）。
import React, { useState } from 'react'
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Modal,
  Popconfirm,
  Row,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
} from 'antd'
import {
  ReloadOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { ColumnsType } from 'antd/es/table'

import { liveService } from '../../api/liveApi'
import type { TradingPlanSummary, RebalanceDecision, RebalanceItem } from '../../types/live'

const { Text, Title } = Typography

// ============================================================
// 调仓明细表格列定义（参考 RebalancePlanCard 思路，独立定义不复用组件）
// ============================================================

/**
 * 调仓详情 Modal 内单条调仓指令表格的列定义。
 *
 * 渲染规则：方向 buy 标红「买入」、其余标绿「卖出」；数量按千分位展示；
 * 参考价 price 与信号值 signal 为 null/undefined 时显示占位符「—」，
 * 分别保留 2、4 位小数；原因列超长时省略。常量级定义，组件外共享一份引用。
 */
const DETAIL_COLUMNS: ColumnsType<RebalanceItem> = [
  {
    title: '标的',
    dataIndex: 'vt_symbol',
    key: 'vt_symbol',
    render: (v: string) => <Text code>{v}</Text>,
  },
  {
    title: '方向',
    dataIndex: 'action',
    key: 'action',
    render: (action: string) =>
      action === 'buy' ? (
        <Tag color="red">买入</Tag>
      ) : (
        <Tag color="green">卖出</Tag>
      ),
  },
  {
    title: '数量（股）',
    dataIndex: 'volume',
    key: 'volume',
    align: 'right' as const,
    render: (v: number) => v.toLocaleString(),
  },
  {
    title: '参考价',
    dataIndex: 'price',
    key: 'price',
    align: 'right' as const,
    render: (v: number) => (v !== null && v !== undefined ? v.toFixed(2) : '—'),
  },
  {
    title: '信号值',
    dataIndex: 'signal',
    key: 'signal',
    align: 'right' as const,
    render: (v: number | null | undefined) =>
      v !== null && v !== undefined ? v.toFixed(4) : '—',
  },
  {
    title: '原因',
    dataIndex: 'reason',
    key: 'reason',
    ellipsis: true,
  },
]

// ============================================================
// 调仓历史表格行类型
// ============================================================

/** 调仓历史列表的单行摘要，对应后端 listRebalances 返回项中页面用到的字段。 */
interface RebalanceSummary {
  /** 调仓信号唯一标识，点击行时用于拉取调仓详情 */
  signal_id: string
  /** 调仓状态，"confirmed" 渲染为「已确认」，其余视为「待确认」 */
  status: string
  /** 所属组合 ID，页面侧按当前选中组合过滤历史列表 */
  portfolio_id: string
  /** 创建时刻，ISO 字符串，展示时转本地时区 */
  created_at: string
}

// ============================================================
// 主页面组件
// ============================================================

/** 策略组合页面的 Props。 */
interface PortfolioProps {
  /** 仅测试用：预设选中的组合 ID，跳过 Select 交互 */
  _testPortfolioId?: string
}

/**
 * 策略组合页面（Task 5.4）。
 *
 * 三合一视图，选定组合后同时展示：
 * - **持仓账本**：{vt_symbol: 股数} 的当前持仓 + 现金余额
 * - **熔断状态**：peak_value / broken / broken_date；支持人工复位
 * - **调仓历史**：点击行展开调仓详情 Modal（含调仓清单与风控摘要）
 *
 * 组合选项由交易操作台的 rule 类型计划提供，无 rule 计划时显示引导。
 */
const Portfolio: React.FC<PortfolioProps> = ({ _testPortfolioId }) => {
  const { message } = App.useApp()
  const queryClient = useQueryClient()

  const [portfolioId, setPortfolioId] = useState<string | undefined>(_testPortfolioId)
  const [detailModal, setDetailModal] = useState<{
    open: boolean
    signalId?: string
  }>({ open: false })
  const [resetting, setResetting] = useState(false)

  // ── 计划列表（提取 rule 计划的 portfolio_id 去重）
  const { data: plans } = useQuery({
    queryKey: ['live-plans'],
    queryFn: () => liveService.listPlans(),
  })

  // TradingPlanSummary 已含 strategy_type / portfolio_id / signal_source，
  // 直接过滤 rule 计划的 portfolio_id 构建组合选择器选项。
  const rulePlans = (plans ?? []) as TradingPlanSummary[]
  const portfolioOptions = Array.from(
    new Set(
      rulePlans
        .filter((p) => p.strategy_type === 'rule' && p.portfolio_id)
        .map((p) => p.portfolio_id),
    ),
  ).map((pid) => ({ label: pid, value: pid }))

  /**
   * 一键刷新所有查询（计划列表、持仓账本、熔断状态、调仓历史）。
   *
   * 通过 queryClient.invalidateQueries 驱动 React Query 自动重拉。
   */
  const refreshAll = () => {
    void queryClient.invalidateQueries({ queryKey: ['live-plans'] })
    if (portfolioId) {
      void queryClient.invalidateQueries({ queryKey: ['portfolio', portfolioId] })
      void queryClient.invalidateQueries({ queryKey: ['portfolio-risk', portfolioId] })
      void queryClient.invalidateQueries({ queryKey: ['rebalances'] })
    }
  }

  // ── 持仓账本
  const { data: portfolio, isLoading: loadingPortfolio } = useQuery({
    queryKey: ['portfolio', portfolioId],
    queryFn: () => liveService.getPortfolio(portfolioId!),
    enabled: !!portfolioId,
  })

  // ── 熔断状态
  const { data: risk, isLoading: loadingRisk } = useQuery({
    queryKey: ['portfolio-risk', portfolioId],
    queryFn: () => liveService.getPortfolioRisk(portfolioId!),
    enabled: !!portfolioId,
  })

  // ── 调仓历史（全量取，页面侧过滤当前 portfolio_id）
  const { data: rebalances, isLoading: loadingRebalances } = useQuery({
    queryKey: ['rebalances'],
    queryFn: () => liveService.listRebalances(),
    enabled: !!portfolioId,
  })
  const filteredRebalances: RebalanceSummary[] = (rebalances ?? []).filter(
    (r) => r.portfolio_id === portfolioId,
  )

  // ── 调仓详情（按需拉取）
  const { data: rebalanceDetail, isLoading: loadingDetail } = useQuery({
    queryKey: ['rebalance-detail', detailModal.signalId],
    queryFn: () => liveService.getRebalance(detailModal.signalId!),
    enabled: !!detailModal.open && !!detailModal.signalId,
  })

  /**
   * 复位组合熔断状态。
   *
   * 调用 `liveService.resetPortfolioRisk`，成功后刷新风险状态查询。
   * 返回 409 等错误时展示后端明细；通用错误展示 message 字段。
   */
  const handleReset = async () => {
    if (!portfolioId) return
    setResetting(true)
    try {
      await liveService.resetPortfolioRisk(portfolioId)
      void message.success('熔断已复位')
      void queryClient.invalidateQueries({ queryKey: ['portfolio-risk', portfolioId] })
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string }
      void message.error(err?.response?.data?.detail ?? err?.message ?? '复位失败')
    } finally {
      setResetting(false)
    }
  }

  // ── 持仓账本表格数据
  const positionRows = portfolio
    ? Object.entries(portfolio.positions).map(([vt_symbol, shares]) => ({
        key: vt_symbol,
        vt_symbol,
        shares,
      }))
    : []
  const positionColumns: ColumnsType<{ key: string; vt_symbol: string; shares: number }> = [
    {
      title: '标的',
      dataIndex: 'vt_symbol',
      key: 'vt_symbol',
      render: (v: string) => <Text code>{v}</Text>,
    },
    {
      title: '持仓股数',
      dataIndex: 'shares',
      key: 'shares',
      align: 'right',
      render: (v: number) => v.toLocaleString(),
    },
  ]

  // ── 调仓历史表格列
  const rebalanceColumns: ColumnsType<RebalanceSummary> = [
    {
      title: '信号 ID',
      dataIndex: 'signal_id',
      key: 'signal_id',
      render: (v: string) => (
        <Text code style={{ fontSize: 12 }}>
          {v}
        </Text>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) =>
        status === 'confirmed' ? (
          <Tag color="green">已确认</Tag>
        ) : (
          <Tag color="orange">待确认</Tag>
        ),
    },
    {
      title: '创建时刻',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v: string) => new Date(v).toLocaleString('zh-CN'),
    },
  ]

  // ── 无 rule 计划时的引导内容
  const noRulePlans = portfolioOptions.length === 0

  return (
    <Space direction="vertical" size={20} style={{ width: '100%' }}>
      {/* 顶栏：标题 + 刷新 */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Title level={4} style={{ margin: 0 }}>
          策略组合
        </Title>
        <Button icon={<ReloadOutlined />} onClick={refreshAll}>
          刷新
        </Button>
      </div>

      {/* 组合选择器 */}
      <Card size="small" title="选择组合">
        {noRulePlans ? (
          <Empty description="先在交易操作台创建规则调仓计划" />
        ) : (
          <Select
            showSearch
            allowClear
            placeholder="选择或输入组合 ID"
            style={{ width: 360 }}
            value={portfolioId}
            options={portfolioOptions}
            onChange={(val) => setPortfolioId(val as string | undefined)}
            filterOption={(input, option) =>
              String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())
            }
            // 允许自定义输入（搜索后未匹配时直接用输入值）
            onSearch={(val) => {
              if (val && !portfolioOptions.find((o) => o.value === val)) {
                // 不强制选已有选项，用户可直接按回车确认
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                const input = (e.target as HTMLInputElement).value
                if (input) setPortfolioId(input)
              }
            }}
          />
        )}
      </Card>

      {/* 组合详情区：选了 portfolioId 才展示 */}
      {portfolioId && (
        <Row gutter={[16, 16]}>
          {/* 持仓账本卡 */}
          <Col xs={24} xl={12}>
            <Card
              title="持仓账本"
              extra={
                loadingPortfolio ? <Spin size="small" /> : null
              }
            >
              {portfolio ? (
                <Space direction="vertical" size={12} style={{ width: '100%' }}>
                  <Descriptions size="small" column={2} bordered>
                    <Descriptions.Item label="组合 ID">
                      <Text code>{portfolio.portfolio_id}</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="现金">
                      {portfolio.cash !== null && portfolio.cash !== undefined
                        ? portfolio.cash.toLocaleString('zh-CN', { style: 'currency', currency: 'CNY' })
                        : '未跟踪'}
                    </Descriptions.Item>
                    {portfolio.last_signal_id && (
                      <Descriptions.Item label="最近 signal_id" span={2}>
                        <Text code style={{ fontSize: 12 }}>
                          {portfolio.last_signal_id}
                        </Text>
                      </Descriptions.Item>
                    )}
                    {portfolio.updated_at && (
                      <Descriptions.Item label="更新时刻" span={2}>
                        {new Date(portfolio.updated_at).toLocaleString('zh-CN')}
                      </Descriptions.Item>
                    )}
                  </Descriptions>
                  <Table
                    size="small"
                    rowKey="vt_symbol"
                    columns={positionColumns}
                    dataSource={positionRows}
                    pagination={false}
                    locale={{ emptyText: '暂无持仓记录' }}
                  />
                </Space>
              ) : loadingPortfolio ? (
                <Spin />
              ) : (
                <Empty description="暂无持仓记录" />
              )}
            </Card>
          </Col>

          {/* 熔断状态卡 */}
          <Col xs={24} xl={12}>
            <Card
              title="熔断状态"
              extra={loadingRisk ? <Spin size="small" /> : null}
            >
              {risk ? (
                <Space direction="vertical" size={12} style={{ width: '100%' }}>
                  {risk.broken ? (
                    <>
                      <Alert
                        type="error"
                        showIcon
                        icon={<WarningOutlined />}
                        message="熔断已触发"
                        description={
                          <Space direction="vertical" size={4}>
                            {risk.broken_date && (
                              <Text>触发日期：{risk.broken_date}</Text>
                            )}
                            {risk.reason && <Text>原因：{risk.reason}</Text>}
                          </Space>
                        }
                      />
                      <Popconfirm
                        title="确认复位熔断？"
                        description="确认人工处置完毕、复位熔断？"
                        onConfirm={() => void handleReset()}
                        okText="确认复位"
                        cancelText="取消"
                      >
                        <Button danger loading={resetting}>
                          复位熔断
                        </Button>
                      </Popconfirm>
                    </>
                  ) : (
                    <Space direction="vertical" size={8}>
                      <Tag color="success" style={{ fontSize: 14, padding: '4px 12px' }}>
                        正常
                      </Tag>
                      {risk.peak_value !== null && risk.peak_value !== undefined && (
                        <Descriptions size="small" column={1}>
                          <Descriptions.Item label="历史峰值净值">
                            {risk.peak_value.toLocaleString('zh-CN', {
                              maximumFractionDigits: 4,
                            })}
                          </Descriptions.Item>
                        </Descriptions>
                      )}
                    </Space>
                  )}
                </Space>
              ) : loadingRisk ? (
                <Spin />
              ) : (
                <Empty description="暂无风险状态" />
              )}
            </Card>
          </Col>

          {/* 调仓历史卡 */}
          <Col xs={24}>
            <Card
              title="调仓历史"
              extra={loadingRebalances ? <Spin size="small" /> : null}
            >
              <Table<RebalanceSummary>
                size="small"
                rowKey="signal_id"
                columns={rebalanceColumns}
                dataSource={filteredRebalances}
                pagination={{ pageSize: 10, hideOnSinglePage: true }}
                locale={{ emptyText: '暂无调仓记录' }}
                onRow={(record) => ({
                  onClick: () =>
                    setDetailModal({ open: true, signalId: record.signal_id }),
                  style: { cursor: 'pointer' },
                })}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 调仓详情 Modal */}
      <Modal
        title={`调仓详情：${detailModal.signalId ?? ''}`}
        open={detailModal.open}
        onCancel={() => setDetailModal({ open: false })}
        footer={null}
        width={800}
      >
        {loadingDetail ? (
          <Spin />
        ) : rebalanceDetail ? (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Descriptions size="small" column={2} bordered>
              <Descriptions.Item label="signal_id" span={2}>
                <Text code style={{ fontSize: 12 }}>
                  {rebalanceDetail.signal_id}
                </Text>
              </Descriptions.Item>
              <Descriptions.Item label="组合 ID">
                <Text code>{rebalanceDetail.portfolio_id}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {rebalanceDetail.status === 'confirmed' ? (
                  <Tag color="green">已确认</Tag>
                ) : (
                  <Tag color="orange">待确认</Tag>
                )}
              </Descriptions.Item>
              <Descriptions.Item label="决策时刻">
                {rebalanceDetail.as_of}
              </Descriptions.Item>
              <Descriptions.Item label="创建时刻">
                {new Date(rebalanceDetail.created_at).toLocaleString('zh-CN')}
              </Descriptions.Item>
            </Descriptions>
            <Table<RebalanceItem>
              size="small"
              rowKey="vt_symbol"
              columns={DETAIL_COLUMNS}
              dataSource={rebalanceDetail.items as RebalanceDecision['items']}
              pagination={false}
              locale={{ emptyText: '暂无调仓指令' }}
            />
          </Space>
        ) : (
          <Empty description="暂无详情" />
        )}
      </Modal>
    </Space>
  )
}

export default Portfolio
