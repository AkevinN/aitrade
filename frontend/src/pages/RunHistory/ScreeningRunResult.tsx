import React from 'react'
import { Empty, Space, Table, Tag, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'

import type { LeaderboardRow, ScreeningResult } from '../../types/screening'

const { Text } = Typography

/** 置信度等级 → Tag 颜色。 */
const confidenceColor: Record<string, string> = {
  high: 'green',
  medium: 'blue',
  low: 'orange',
  insufficient: 'default',
}

/** 置信度等级 → 中文。 */
function confidenceLabel(level: string): string {
  const map: Record<string, string> = {
    high: '高',
    medium: '中',
    low: '低',
    insufficient: '不足',
  }
  return map[level] ?? level
}

/**
 * 历史选股运行的只读榜单视图。
 *
 * 专供「运行历史」详情抽屉回看一次 CNN 选股运行的 `ScreeningResult`。刻意自包含、
 * 只读（不带表单/操作），以**不触碰**正在迭代中的 `CNNScreening/index.tsx`——待其
 * Tier-2 详情 WIP 落地后可再抽共享 `ScreeningResultView` 统一两处。
 *
 * 数据为空（无榜单）时渲染空态而非崩溃。
 */
interface ScreeningRunResultProps {
  /** 一次选股运行的产物（= CNN_SCREENING 任务的 result） */
  result?: ScreeningResult | null
}

const ScreeningRunResult: React.FC<ScreeningRunResultProps> = ({ result }) => {
  const rows = result?.leaderboard ?? []
  if (rows.length === 0) {
    return <Empty description="该次选股无榜单数据" />
  }

  const columns: ColumnsType<LeaderboardRow> = [
    {
      title: '排名',
      dataIndex: 'rank',
      width: 64,
      sorter: (a, b) => a.rank - b.rank,
      defaultSortOrder: 'ascend',
    },
    {
      title: '标的',
      key: 'vt_symbol',
      width: 130,
      render: (_, row) => <Text strong>{row.tier1.vt_symbol}</Text>,
    },
    {
      title: 'CNN 适配度',
      key: 'fitness',
      width: 110,
      sorter: (a, b) => (a.tier1.fitness_score ?? -1) - (b.tier1.fitness_score ?? -1),
      render: (_, row) =>
        row.tier1.fitness_score == null ? (
          <Text type="secondary">—</Text>
        ) : (
          row.tier1.fitness_score.toFixed(3)
        ),
    },
    {
      title: '置信度',
      key: 'confidence',
      width: 90,
      render: (_, row) => {
        const lvl = row.tier1.overall_confidence
        return <Tag color={confidenceColor[lvl] ?? 'default'}>{confidenceLabel(lvl)}</Tag>
      },
    },
    {
      title: 'Tier-2 实证',
      key: 'tier2',
      render: (_, row) => {
        if (!row.promoted_to_tier2) return <Text type="secondary">未入围</Text>
        const t2 = row.tier2
        if (!t2 || !t2.evaluable) {
          return (
            <Tag color="orange" title={t2?.note ?? undefined}>
              不可评估
            </Tag>
          )
        }
        return (
          <Space size={6}>
            <Tag color={t2.edge_ok ? 'green' : 'red'}>{t2.edge_ok ? 'edge 通过' : 'edge 未过'}</Tag>
            {t2.avg_score != null ? (
              <Text type="secondary">分 {t2.avg_score.toFixed(3)}</Text>
            ) : null}
          </Space>
        )
      },
    },
  ]

  return (
    <Table<LeaderboardRow>
      size="small"
      rowKey={(row) => row.tier1.vt_symbol}
      columns={columns}
      dataSource={rows}
      pagination={rows.length > 20 ? { pageSize: 20 } : false}
    />
  )
}

export default ScreeningRunResult
