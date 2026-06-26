// 半仓做 T 回测 API 服务 — 对应后端 /api/t0 路由组（同步返回，非异步任务）
import api from './client'
import type { T0BacktestRequest, T0Report, T0Profile, T0ProfileRequest } from '../types/t0'

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
}
