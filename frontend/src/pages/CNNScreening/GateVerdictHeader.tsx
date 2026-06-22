import React from 'react'
import { Space, Tag, Tooltip, Typography } from 'antd'
import { CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons'

import type { Tier2Verdict } from '../../types/screening'

const { Text } = Typography

/** 门禁结论头部的 Props。 */
interface GateVerdictHeaderProps {
  /** 来自榜单行的权威 Tier-2 结论；edge_ok 以服务端为准 */
  verdict: Tier2Verdict
  /**
   * 正分折占比阈值，默认 0.5（ScreeningRules.min_positive_fold_ratio 默认值）。
   * 仅用于"正分折≥阈值"这一条件的可视化；整体结论始终以 verdict.edge_ok 为准。
   */
  threshold?: number
}

/** 单个判据 chip：通过→绿色对勾，未通过→红色叉。null 值渲染为中性占位。 */
function ConditionChip({ ok, text }: { ok: boolean | null; text: string }) {
  if (ok === null) return <Tag>{text}：—</Tag>
  return (
    <Tag
      color={ok ? 'success' : 'error'}
      icon={ok ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
    >
      {text}
    </Tag>
  )
}

/**
 * Tier-2 绝对 edge 门禁的自解释头部。
 *
 * 把权威结论 `edge_ok` 拆成两条判据可视化——「跨折平均分 > 0」与
 * 「正分折占比 ≥ 阈值」——让用户一眼看出门禁因哪条通过/未通过。
 * `evaluable=false` 时渲染"不可评估"并附 note 原因。
 *
 * @param verdict - 权威 Tier-2 结论
 * @param threshold - 正分折占比阈值，默认 0.5
 */
const GateVerdictHeader: React.FC<GateVerdictHeaderProps> = ({ verdict, threshold = 0.5 }) => {
  if (!verdict.evaluable) {
    return (
      <Space wrap>
        <Tag color="orange">不可评估</Tag>
        {verdict.note ? <Text type="secondary">{verdict.note}</Text> : null}
      </Space>
    )
  }

  const avg = verdict.avg_score
  const pos = verdict.pos_fold_ratio
  const avgText = avg != null ? `平均分 ${avg.toFixed(4)} > 0` : '平均分 > 0'
  const posText =
    pos != null
      ? `正分折 ${(pos * 100).toFixed(1)}% ≥ ${(threshold * 100).toFixed(0)}%`
      : `正分折 ≥ ${(threshold * 100).toFixed(0)}%`

  return (
    <Space wrap size={8} align="center">
      <Tag color={verdict.edge_ok ? 'green' : 'red'} style={{ fontWeight: 600 }}>
        {verdict.edge_ok ? '✓ edge_ok 通过' : '✗ edge_ok 未通过'}
      </Tag>
      <Tooltip title="跨折平均核心分是否为正（总账是否赚）">
        <span>
          <ConditionChip ok={avg != null ? avg > 0 : null} text={avgText} />
        </span>
      </Tooltip>
      <Tooltip title="candidate_score > 0 的折数占比是否达阈值（赢得稳不稳）">
        <span>
          <ConditionChip ok={pos != null ? pos >= threshold : null} text={posText} />
        </span>
      </Tooltip>
      {verdict.avg_cross_seed_std != null ? (
        <Text type="secondary">跨种子σ {verdict.avg_cross_seed_std.toFixed(4)}</Text>
      ) : null}
    </Space>
  )
}

export default GateVerdictHeader
