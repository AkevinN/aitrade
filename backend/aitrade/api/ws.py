"""
WebSocket endpoint — /ws.

Manages client connections, topic subscriptions, and broadcasts messages
from async tasks / data sources to all subscribed clients.
"""

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """
    WebSocket connection pool — manages clients, subscriptions, and broadcasts.

    All methods are async and thread-safe (all callers are async, no locks needed).
    """

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._subscriptions: dict[WebSocket, set[str]] = {}
        self.ALL_TOPICS: set[str] = {
            "tick",
            "order",
            "trade",
            "position",
            "account",
            "contract",
            "log",
            "task",
        }

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket client."""
        await websocket.accept()
        self._connections.add(websocket)
        self._subscriptions[websocket] = self.ALL_TOPICS.copy()

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected client."""
        self._connections.discard(websocket)
        self._subscriptions.pop(websocket, None)

    async def handle_client_message(self, websocket: WebSocket, text: str) -> None:
        """
        Process client control messages.

        Supported actions:
          - subscribe / unsubscribe: update topic set
          - ping: respond with pong
        """
        try:
            msg: dict[str, Any] = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            await websocket.send_json({"error": "Invalid JSON"})
            return

        action: str = msg.get("action", "")

        if action == "subscribe":
            topics: list[str] = msg.get("topics", [])
            valid_topics = set(topics) & self.ALL_TOPICS
            if websocket in self._subscriptions:
                self._subscriptions[websocket] |= valid_topics
            await websocket.send_json({
                "action": "subscribed",
                "topics": sorted(self._subscriptions.get(websocket, set())),
            })

        elif action == "unsubscribe":
            topics = msg.get("topics", [])
            if websocket in self._subscriptions:
                self._subscriptions[websocket] -= set(topics)
            await websocket.send_json({
                "action": "unsubscribed",
                "topics": sorted(self._subscriptions.get(websocket, set())),
            })

        elif action == "ping":
            await websocket.send_json({"action": "pong"})

        else:
            await websocket.send_json({"error": f"Unknown action: {action}"})

    async def broadcast(self, message: dict[str, Any], topic: str) -> None:
        """
        Broadcast a message to all clients subscribed to *topic*.

        Silently skips disconnected clients.
        """
        disconnected: list[WebSocket] = []
        message_text: str = json.dumps(message, ensure_ascii=False, default=str)

        for ws in self._connections:
            if topic not in self._subscriptions.get(ws, set()):
                continue
            try:
                await ws.send_text(message_text)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    @property
    def active_count(self) -> int:
        """Number of active connections."""
        return len(self._connections)


# Global singleton shared across the app
ws_manager = ConnectionManager()
