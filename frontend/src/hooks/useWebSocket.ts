import { useEffect, useRef, useCallback } from 'react'
import { taskStore } from '../stores/taskStore'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || API_BASE.replace(/^http/, 'ws') + '/ws'
const RECONNECT_DELAY = 3000
const HEARTBEAT_INTERVAL = 30000

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
