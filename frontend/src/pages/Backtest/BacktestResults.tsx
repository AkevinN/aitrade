import React from 'react'
import { Card, Row, Col, Statistic, Descriptions, Empty } from 'antd'

import type { BacktestStatistics } from '../../types/alpha'

/**
 * {@link BacktestResults} 组件 props。
 */
interface BacktestResultsProps {
  /** 回测统计指标；未开始回测时为 `undefined`（渲染占位空状态）。 */
  statistics?: BacktestStatistics
  /** 初始资金（元），用于在统计数据缺失时展示默认余额。 */
  capital: number
}

/**
 * 回测统计指标面板：展示总收益、年化、Sharpe、最大回撤等关键指标卡片，
 * 以及成本假设、基准对比、持仓配置等详细 Descriptions。
 *
 * `statistics` 为 `undefined` 时展示「启动回测查看统计结果」占位。
 */
const BacktestResults: React.FC<BacktestResultsProps> = ({ statistics, capital }) => {
  if (!statistics) {
    return (
      <Card size="small">
        <Empty description="启动回测查看统计结果" />
      </Card>
    )
  }

  const totalReturn = statistics.total_return || 0
  const annualReturn = statistics.annual_return || 0
  const sharpeRatio = statistics.sharpe_ratio || 0
  const maxDdpercent = statistics.max_ddpercent ?? statistics.max_drawdown ?? 0
  const totalTradeCount = statistics.total_trade_count || 0
  const totalNetPnl = statistics.total_net_pnl || 0
  const endBalance = statistics.end_balance || capital
  const startDate = statistics.start_date || '-'
  const endDate = statistics.end_date || '-'
  const totalDays = statistics.total_days || 0
  const profitDays = statistics.profit_days || 0
  const lossDays = statistics.loss_days || 0
  const totalCommission = statistics.total_commission || 0
  const capitalValue = statistics.capital || capital
  const error = statistics.error

  const hasBenchmark =
    statistics.benchmark_return !== undefined || statistics.excess_return !== undefined
  const benchmarkReturn = statistics.benchmark_return ?? 0
  const excessReturn = statistics.excess_return ?? 0
  const benchmarkSymbol = statistics.benchmark_symbol

  const hasCostAssumption =
    statistics.commission_rate !== undefined ||
    statistics.stamp_duty !== undefined ||
    statistics.slippage !== undefined ||
    statistics.price_add !== undefined
  const bps = (value?: number) => (value === undefined ? '-' : `${(value * 10000).toFixed(1)} bp`)

  return (
    <>
      <Row gutter={[8, 8]} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="总收益"
              value={totalReturn}
              precision={2}
              suffix="%"
              valueStyle={{ fontSize: 18, color: totalReturn >= 0 ? '#49aa19' : '#dc4446' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="夏普比率"
              value={sharpeRatio}
              precision={2}
              valueStyle={{ fontSize: 18 }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="最大回撤"
              value={maxDdpercent}
              precision={2}
              suffix="%"
              valueStyle={{ fontSize: 18, color: '#dc4446' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="总交易"
              value={totalTradeCount}
              valueStyle={{ fontSize: 18 }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="净盈亏"
              value={totalNetPnl}
              precision={2}
              valueStyle={{ fontSize: 18, color: totalNetPnl >= 0 ? '#49aa19' : '#dc4446' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card size="small">
            <Statistic
              title="期末余额"
              value={endBalance}
              precision={2}
              valueStyle={{ fontSize: 18 }}
            />
          </Card>
        </Col>
      </Row>

      {hasBenchmark && (
        <Row gutter={[8, 8]} style={{ marginBottom: 16 }}>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title={`超额收益${benchmarkSymbol ? `（基准 ${benchmarkSymbol}）` : ''}`}
                value={excessReturn}
                precision={2}
                prefix={excessReturn >= 0 ? '+' : ''}
                suffix="%"
                valueStyle={{ fontSize: 18, color: excessReturn >= 0 ? '#49aa19' : '#dc4446' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="基准收益（买入持有）"
                value={benchmarkReturn}
                precision={2}
                suffix="%"
                valueStyle={{ fontSize: 18, color: benchmarkReturn >= 0 ? '#49aa19' : '#dc4446' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="策略收益"
                value={totalReturn}
                precision={2}
                suffix="%"
                valueStyle={{ fontSize: 18, color: totalReturn >= 0 ? '#49aa19' : '#dc4446' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Card title="回测摘要" size="small">
        <Descriptions bordered column={2} size="small">
          <Descriptions.Item label="开始日期">{startDate}</Descriptions.Item>
          <Descriptions.Item label="结束日期">{endDate}</Descriptions.Item>
          <Descriptions.Item label="总天数">{totalDays}</Descriptions.Item>
          <Descriptions.Item label="盈利天数">{profitDays}</Descriptions.Item>
          <Descriptions.Item label="亏损天数">{lossDays}</Descriptions.Item>
          <Descriptions.Item label="年化收益">
            {annualReturn.toFixed(2)}%
          </Descriptions.Item>
          <Descriptions.Item label="手续费及税费">
            {totalCommission.toFixed(2)}
          </Descriptions.Item>
          <Descriptions.Item label="初始资金">
            {capitalValue.toFixed(2)}
          </Descriptions.Item>
          {error && (
            <Descriptions.Item label="备注" span={2}>
              {error}
            </Descriptions.Item>
          )}
        </Descriptions>
      </Card>

      {hasCostAssumption && (
        <Card title="成本假设" size="small" style={{ marginTop: 16 }}>
          <Descriptions bordered column={2} size="small">
            <Descriptions.Item label="佣金率">{bps(statistics.commission_rate)}</Descriptions.Item>
            <Descriptions.Item label="卖出印花税">{bps(statistics.stamp_duty)}</Descriptions.Item>
            <Descriptions.Item label="成交滑点">{bps(statistics.slippage)}</Descriptions.Item>
            <Descriptions.Item label="限价缓冲">{bps(statistics.price_add)}</Descriptions.Item>
            {statistics.veto_count !== undefined && statistics.veto_count > 0 && (
              <Descriptions.Item label="否决买入次数" span={2}>
                {statistics.veto_count}（prob_sl ≥ veto_threshold 而放弃的买入信号数）
              </Descriptions.Item>
            )}
          </Descriptions>
        </Card>
      )}
    </>
  )
}

export default BacktestResults
