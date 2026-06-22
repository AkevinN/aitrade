/**
 * CNN 选股（Tier-2）相关的 HTTP 服务封装。
 *
 * 与 `governanceService` 同范式：经共享 axios 实例 `api` 调后端，方法返回
 * 解包后的 `response.data`。当前仅含按 `report_id` 回读 WF 报告的只读接口。
 */
import api from './client'
import type { ScreeningWfReport } from '../types/screening'

export const screeningService = {
  /**
   * 按 `Tier2Verdict.report_id` 回读 Tier-2 walk-forward/OOS 报告详情。
   *
   * 命中后端隔离 store（`SCREENING_GOVERNANCE_PATH`）下的完整报告，供折级详情抽屉
   * 渲染门禁头部、折级表与选中折回测指标卡。报告不存在时后端返回 404，调用方应捕获。
   *
   * @param reportId - WF 报告 ID（形如 `wf_YYYYMMDDHHMMSS_xxxxxx`）
   * @returns 完整的 {@link ScreeningWfReport}
   * @throws 报告不存在时后端 404，axios 抛出错误
   *
   * @example
   * ```ts
   * const report = await screeningService.getScreeningReport('wf_20250622153012_a1b2c3')
   * ```
   */
  getScreeningReport: (reportId: string) =>
    api
      .get<ScreeningWfReport>(`/api/cnn/screening/reports/${reportId}`)
      .then((r) => r.data),
}
