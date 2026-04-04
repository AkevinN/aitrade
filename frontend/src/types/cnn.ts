import type { TaskStartResponse } from './alpha'

export interface CNNModelInfo {
  name: string
  created_at: string
  num_params?: number
  history?: {
    train_loss: number[]
    val_loss: number[]
    accuracy: number[]
  }
}

export interface CNNStatus {
  torch_installed: boolean
  device: string
}

export type { TaskStartResponse, CNNTrainRequest } from './alpha'
