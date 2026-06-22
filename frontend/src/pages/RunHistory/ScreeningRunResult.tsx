import React, { useState } from 'react'
import { Empty } from 'antd'
import { useNavigate } from 'react-router-dom'

import type { ScreeningResult, Tier2Verdict } from '../../types/screening'
import { ScreeningResultPanel, buildLeaderboardColumns } from '../CNNScreening'
import Tier2DetailDrawer from '../CNNScreening/Tier2DetailDrawer'

/**
 * 历史选股运行的结果视图——与 live 选股页"刚选完一样"丰富。
 *
 * 直接复用 live 页的 `ScreeningResultPanel` + `buildLeaderboardColumns` + `Tier2DetailDrawer`，
 * 因此历史详情与刚跑完的展示**逐项一致**（同一套榜单列、可排序、贡献明细展开、每行"查看
 * 详情"钻取 Tier-2 折级实证报告、"带入训练"跳转 CNN 训练页）。无重复实现、无展示漂移。
 */
interface ScreeningRunResultProps {
  /** 一次选股运行的产物（= CNN_SCREENING 任务的 result） */
  result?: ScreeningResult | null
}

const ScreeningRunResult: React.FC<ScreeningRunResultProps> = ({ result }) => {
  const navigate = useNavigate()
  const [detailVerdict, setDetailVerdict] = useState<Tier2Verdict | null>(null)

  if (!result) {
    return <Empty description="该次选股无结果数据" />
  }

  const interval =
    result.input?.interval != null ? String(result.input.interval) : 'd'

  const columns = buildLeaderboardColumns({
    interval,
    onDetailOpen: setDetailVerdict,
    onTrainClick: (vtSymbol, itv) =>
      navigate('/cnn-train', {
        state: {
          preset: { target_symbol: vtSymbol, input_interval: itv, input_data_kind: 'bar' },
        },
      }),
  })

  return (
    <>
      <ScreeningResultPanel result={result} leaderboardColumns={columns} />
      <Tier2DetailDrawer
        open={detailVerdict != null}
        verdict={detailVerdict}
        onClose={() => setDetailVerdict(null)}
      />
    </>
  )
}

export default ScreeningRunResult
