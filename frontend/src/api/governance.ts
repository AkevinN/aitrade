/**
 * CNN 治理模块 API 服务对象——封装所有与后端 `/api/cnn/governance/` 端点的 HTTP 交互。
 *
 * 提供治理配置读写、候选模型管理（训练/评估/晋升/拒绝/回滚）、
 * 历史事件查询、治理报告获取及历史重演（Replay）能力。
 * 所有方法均返回 `Promise<T>`，错误由全局 Axios 拦截器透传。
 */
import api from './client'
import type {
  CNNCandidate,
  CNNCandidateTrainRequest,
  CNNGovernanceConfig,
  CNNGovernanceReplayReport,
  CNNGovernanceReplayRequest,
  CNNGovernanceReport,
  CNNProductionModel,
  CNNWalkForwardRequest,
  GovernanceHistoryEvent,
  TaskStartResponse,
} from '../types/governance'

export const governanceService = {
  /**
   * 拉取当前治理配置（晋级阈值、重训周期、自动化开关等）。
   *
   * @returns 后端持久化的治理配置对象。
   */
  getConfig: () =>
    api.get<CNNGovernanceConfig>('/api/cnn/governance/config').then((r) => r.data),

  /**
   * 整体覆盖写入治理配置（PUT 全量替换，非字段级合并）。
   *
   * @param req - 完整的治理配置；缺省字段将按后端规则被重置，需先 getConfig 取全量再改。
   * @returns 写入后由后端回显的最新配置。
   */
  updateConfig: (req: CNNGovernanceConfig) =>
    api.put<CNNGovernanceConfig>('/api/cnn/governance/config', req).then((r) => r.data),

  /**
   * 获取当前线上生产模型（含版本、晋级来源、关键指标）。
   *
   * @returns 当前在产的生产模型信息。
   */
  getProduction: () =>
    api.get<CNNProductionModel>('/api/cnn/governance/production').then((r) => r.data),

  /**
   * 列出全部候选模型（涵盖待评估/已评估/已晋级/已拒绝各状态）。
   *
   * @returns 候选模型数组，顺序由后端决定。
   */
  listCandidates: () =>
    api.get<CNNCandidate[]>('/api/cnn/governance/candidates').then((r) => r.data),

  /**
   * 按 ID 获取单个候选模型的详情。
   *
   * @param candidateId - 候选模型 ID；不存在时后端返回 404，错误经全局拦截器透传。
   * @returns 对应候选模型对象。
   */
  getCandidate: (candidateId: string) =>
    api.get<CNNCandidate>(`/api/cnn/governance/candidates/${candidateId}`).then((r) => r.data),

  /**
   * 发起走步前进（walk-forward）评估，异步执行。
   *
   * @param req - 评估参数（标的、窗口、阈值等）。
   * @returns 任务启动响应，含 task_id，需后续轮询/订阅任务进度。
   */
  evaluate: (req: CNNWalkForwardRequest) =>
    api.post<TaskStartResponse>('/api/cnn/governance/evaluate', req).then((r) => r.data),

  /**
   * 触发训练一个新的候选模型，异步执行。
   *
   * @param req - 候选模型训练参数（标的、周期、超参等）。
   * @returns 任务启动响应，含 task_id，需后续轮询/订阅任务进度。
   */
  trainCandidate: (req: CNNCandidateTrainRequest) =>
    api.post<TaskStartResponse>('/api/cnn/governance/candidates/train', req).then((r) => r.data),

  /**
   * 将指定候选模型晋级为生产模型（人工触发，promoted_by 固定为 'manual'）。
   *
   * @param candidateId - 待晋级的候选模型 ID。
   * @param note - 晋级备注，写入治理历史；默认空串。
   * @returns 晋级后的新生产模型信息。
   */
  promoteCandidate: (candidateId: string, note = '') =>
    api
      .post<CNNProductionModel>(`/api/cnn/governance/candidates/${candidateId}/promote`, {
        promoted_by: 'manual',
        note,
      })
      .then((r) => r.data),

  /**
   * 拒绝指定候选模型（人工触发，promoted_by 固定为 'manual'）。
   *
   * @param candidateId - 待拒绝的候选模型 ID。
   * @param note - 拒绝原因备注，写入治理历史；默认空串。
   * @returns 状态更新为已拒绝后的候选模型对象。
   */
  rejectCandidate: (candidateId: string, note = '') =>
    api
      .post<CNNCandidate>(`/api/cnn/governance/candidates/${candidateId}/reject`, {
        promoted_by: 'manual',
        note,
      })
      .then((r) => r.data),

  /**
   * 回滚生产模型至上一个版本（人工触发，requested_by 固定为 'manual'）。
   *
   * @param note - 回滚原因备注，写入治理历史；默认空串。
   * @returns 回滚后的生产模型信息。
   */
  rollback: (note = '') =>
    api
      .post<CNNProductionModel>('/api/cnn/governance/rollback', {
        requested_by: 'manual',
        note,
      })
      .then((r) => r.data),

  /**
   * 列出治理历史事件（训练/评估/晋级/拒绝/回滚等的时间线）。
   *
   * @returns 治理历史事件数组。
   */
  listHistory: () =>
    api.get<GovernanceHistoryEvent[]>('/api/cnn/governance/history').then((r) => r.data),

  /**
   * 按报告 ID 获取一份治理评估报告（候选与生产模型的对比指标）。
   *
   * @param reportId - 治理报告 ID。
   * @returns 对应的治理报告对象。
   */
  getReport: (reportId: string) =>
    api.get<CNNGovernanceReport>(`/api/cnn/governance/reports/${reportId}`).then((r) => r.data),

  /**
   * 发起一次治理回放（历史重演），异步执行。
   *
   * 对比固定初始模型、定期重训、治理筛选晋级与买入持有等策略，
   * 评估半自动晋级机制在历史区间上的实际收益与稳定性。
   *
   * @param req - 回放参数（标的、回放区间、对比策略集等）。
   * @returns 任务启动响应，含 task_id，需后续轮询/订阅任务进度。
   */
  runReplay: (req: CNNGovernanceReplayRequest) =>
    api.post<TaskStartResponse>('/api/cnn/governance/replay/run', req).then((r) => r.data),

  /**
   * 列出全部已生成的治理回放报告。
   *
   * @returns 治理回放报告数组。
   */
  listReplays: () =>
    api.get<CNNGovernanceReplayReport[]>('/api/cnn/governance/replay').then((r) => r.data),

  /**
   * 按回放 ID 获取单份治理回放报告详情。
   *
   * @param replayId - 治理回放报告 ID。
   * @returns 对应的治理回放报告对象。
   */
  getReplay: (replayId: string) =>
    api.get<CNNGovernanceReplayReport>(`/api/cnn/governance/replay/${replayId}`).then((r) => r.data),
}
