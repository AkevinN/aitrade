import { useEffect, useRef, useCallback } from 'react'
import { taskStore } from '../stores/taskStore'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || API_BASE.replace(/^http/, 'ws') + '/ws'
const RECONNECT_DELAY = 3000
const HEARTBEAT_INTERVAL = 30000

/**
 * 维护全局 WebSocket 连接并将 `task_update` 消息写入 `taskStore`。
 *
 * 连接成功后自动订阅指定 topics，每 30 秒发送心跳 ping，断线后延迟 3 秒重连。
 * 组件卸载时清理所有定时器并关闭连接，防止内存泄漏。
 *
 * @param topics - 需订阅的 WS topic 列表；空数组表示不发订阅消息，默认 `[]`。
 * @returns 当前 WebSocket 实例的 ref（通常无需直接使用）。
 *
 * @remarks
 * 应在应用根组件调用一次，使整个会话共享同一条 WS 连接。
 * `topics` 数组引用变化会触发重连，建议用 `useMemo` / 常量稳定引用。
 *
 * @example
 * ```tsx
 * // App.tsx 顶层
 * useWebSocket(['task_updates'])
 * ```
 */
export function useWebSocket(topics: string[] = []) {
  const wsRef = useRef<WebSocket | null>(null)
  const heartbeatRef = useRef<number | undefined>(undefined)
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[WS] Connected')
      if (topics.length > 0) {
        ws.send(JSON.stringify({ action: 'subscribe', topics }))
      }
      heartbeatRef.current = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'ping' }))
        }
      }, HEARTBEAT_INTERVAL)
    }

    ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data)
        const { type, data } = msg

        if (type === 'task_update' && data) {
          taskStore.getState().addTask(data)
        }
      } catch {
        // ignore parse errors
      }
    }

    ws.onclose = () => {
      console.log('[WS] Disconnected, reconnecting...')
      if (heartbeatRef.current !== undefined) window.clearInterval(heartbeatRef.current)
      if (mountedRef.current) {
        reconnectRef.current = window.setTimeout(connect, RECONNECT_DELAY)
      }
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [topics])

  useEffect(() => {
    mountedRef.current = true
    connect()

    return () => {
      mountedRef.current = false
      if (heartbeatRef.current) clearInterval(heartbeatRef.current)
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return wsRef
}
