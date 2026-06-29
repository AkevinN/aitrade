// 半仓做 T 回测 API 服务 — 对应后端 /api/t0 路由组（同步返回，非异步任务）
import api from './client'
import type {
  T0BacktestRequest, T0Report, T0Profile, T0ProfileRequest,
  T0ProfileSegmentedRequest, T0SegmentedProfile,
} from '../types/t0'

/**
 * 半仓做 T 回测 API 服务。
 *
 * 与其它回测不同，做 T 单次扫描数秒内完成，故为**同步**返回（无 task 轮询）：
 * 调用方用 useMutation 管理 loading/error 即可。
 */
export const t0Service = {
  /**
   * 运行一次半仓做 T 回测，返回成交敏感性区间报告。
   *
   * @param req - 回测请求（标的/区间/档位/摆动/成交网格）
   * @returns T0Report（含 fill_sensitivity 区间、逐年/逐月超额、命中分布）
   */
  runBacktest: (req: T0BacktestRequest): Promise<T0Report> =>
    api.post<T0Report>('/api/t0/backtest', req).then((r) => r.data),

  /**
   * 统计某标的"按偏离开盘 x 分挂单、单腿做 T 的每笔边际收益"曲线（理想撮合），并给建议档位。
   *
   * @param req - 画像请求（标的/标定窗/档位上限/成本）
   * @returns T0Profile（逐档位逐腿边际曲线 + 建议卖/买档位）
   */
  profile: (req: T0ProfileRequest): Promise<T0Profile> =>
    api.post<T0Profile>('/api/t0/profile', req).then((r) => r.data),

  /**
   * 按高/低/平开分场景统计做 T 画像，供条件(跳空)策略逐规则标定档位。
   *
   * @param req - 分场景画像请求（标的/标定窗/档位上限/成本 + gap_thresh）
   * @returns 三段画像（高开/低开/平开），各带建议档位与样本天数
   */
  profileSegmented: (req: T0ProfileSegmentedRequest): Promise<T0SegmentedProfile> =>
    api.post<T0SegmentedProfile>('/api/t0/profile_segmented', req).then((r) => r.data),

  /**
   * 列出可用于条件规则（`lhs="signal"`）的持久化模型信号名。
   *
   * @returns 信号名升序数组；Alpha 未安装时为空数组
   */
  listSignals: (): Promise<string[]> =>
    api.get<{ names: string[] }>('/api/t0/signals').then((r) => r.data.names),
}
