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
  getConfig: () =>
    api.get<CNNGovernanceConfig>('/api/cnn/governance/config').then((r) => r.data),

  updateConfig: (req: CNNGovernanceConfig) =>
    api.put<CNNGovernanceConfig>('/api/cnn/governance/config', req).then((r) => r.data),

  getProduction: () =>
    api.get<CNNProductionModel>('/api/cnn/governance/production').then((r) => r.data),

  listCandidates: () =>
    api.get<CNNCandidate[]>('/api/cnn/governance/candidates').then((r) => r.data),

  getCandidate: (candidateId: string) =>
    api.get<CNNCandidate>(`/api/cnn/governance/candidates/${candidateId}`).then((r) => r.data),

  evaluate: (req: CNNWalkForwardRequest) =>
    api.post<TaskStartResponse>('/api/cnn/governance/evaluate', req).then((r) => r.data),

  trainCandidate: (req: CNNCandidateTrainRequest) =>
    api.post<TaskStartResponse>('/api/cnn/governance/candidates/train', req).then((r) => r.data),

  promoteCandidate: (candidateId: string, note = '') =>
    api
      .post<CNNProductionModel>(`/api/cnn/governance/candidates/${candidateId}/promote`, {
        promoted_by: 'manual',
        note,
      })
      .then((r) => r.data),

  rejectCandidate: (candidateId: string, note = '') =>
    api
      .post<CNNCandidate>(`/api/cnn/governance/candidates/${candidateId}/reject`, {
        promoted_by: 'manual',
        note,
      })
      .then((r) => r.data),

  rollback: (note = '') =>
    api
      .post<CNNProductionModel>('/api/cnn/governance/rollback', {
        requested_by: 'manual',
        note,
      })
      .then((r) => r.data),

  listHistory: () =>
    api.get<GovernanceHistoryEvent[]>('/api/cnn/governance/history').then((r) => r.data),

  getReport: (reportId: string) =>
    api.get<CNNGovernanceReport>(`/api/cnn/governance/reports/${reportId}`).then((r) => r.data),

  runReplay: (req: CNNGovernanceReplayRequest) =>
    api.post<TaskStartResponse>('/api/cnn/governance/replay/run', req).then((r) => r.data),

  listReplays: () =>
    api.get<CNNGovernanceReplayReport[]>('/api/cnn/governance/replay').then((r) => r.data),

  getReplay: (replayId: string) =>
    api.get<CNNGovernanceReplayReport>(`/api/cnn/governance/replay/${replayId}`).then((r) => r.data),
}
