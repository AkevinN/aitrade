import React from 'react'
import { Space, Table, Tooltip, Typography } from 'antd'
import { InfoCircleOutlined } from '@ant-design/icons'

import type { ScreeningFold } from '../../types/screening'
import { metricMeta } from '../../utils/screening'

const { Text } = Typography

/** 折级表的 Props。 */
interface FoldTableProps {
  /** 逐折明细 */
  folds: ScreeningFold[]
  /** 当前选中折的 fold 序号 */
  selectedFold: number
  /** 点击某折时回调（传入该折 fold 序号） */
  onSelect: (fold: number) => void
}

/** 列头标签：复用 metricMeta 的中英标签 + Tooltip，与榜单列头风格一致。 */
function colTitle(key: string) {
  const meta = metricMeta(key)
  return (
    <Space size={4}>
      <span>{meta.label}</span>
      <Tooltip title={meta.tooltip}>
        <InfoCircleOutlined style={{ color: '#8c8c8c', cursor: 'help' }} />
      </Tooltip>
    </Space>
  )
}

/** 把 YYYY-MM-DD... 的 ISO 串截到日期。 */
function asDate(s: string): string {
  return s ? String(s).slice(0, 10) : '-'
}

/**
 * Tier-2 折级总览表：每折一行，点行高亮并联动外部选中态。
 *
 * 列为 折 / 测试窗口 / 核心分 / 跨种子σ；当存在生产模型对照（任一折
 * `production_score` 非空）时追加「生产分 / 分差」两列，否则隐藏（选股场景无生产模型）。
 *
 * @param folds - 逐折明细
 * @param selectedFold - 当前选中折序号（高亮）
 * @param onSelect - 点行回调
 */
const FoldTable: React.FC<FoldTableProps> = ({ folds, selectedFold, onSelect }) => {
  const showProduction = folds.some((f) => f.production_score != null)

  const columns = [
    {
      title: colTitle('fold'),
      dataIndex: 'fold',
      width: 70,
      render: (v: number) => <Text strong>#{v}</Text>,
    },
    {
      title: colTitle('fold_test_days'),
      key: 'test',
      width: 200,
      render: (_: unknown, row: ScreeningFold) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {asDate(row.test.start)} ~ {asDate(row.test.end)}
        </Text>
      ),
    },
    {
      title: colTitle('candidate_score'),
      dataIndex: 'candidate_score',
      width: 120,
      render: (v: number) => (
        <Text type={v > 0 ? 'success' : 'danger'}>{v != null ? v.toFixed(4) : '-'}</Text>
      ),
    },
    {
      title: colTitle('avg_cross_seed_std'),
      key: 'cross_seed_std',
      width: 120,
      render: (_: unknown, row: ScreeningFold) =>
        row.cross_seed?.std != null ? row.cross_seed.std.toFixed(4) : '-',
    },
    ...(showProduction
      ? [
          {
            title: colTitle('production_score'),
            dataIndex: 'production_score',
            width: 120,
            render: (v: number | null) => (v != null ? v.toFixed(4) : '-'),
          },
          {
            title: colTitle('score_delta'),
            dataIndex: 'score_delta',
            width: 110,
            render: (v: number | null) => (v != null ? v.toFixed(4) : '-'),
          },
        ]
      : []),
  ]

  return (
    <Table<ScreeningFold>
      size="small"
      rowKey="fold"
      dataSource={folds}
      columns={columns}
      pagination={false}
      onRow={(row) => ({
        onClick: () => onSelect(row.fold),
        style: { cursor: 'pointer' },
      })}
      rowClassName={(row) => (row.fold === selectedFold ? 'ant-table-row-selected' : '')}
    />
  )
}

export default FoldTable
