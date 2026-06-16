"""WebSocket 推送端点 `/ws` 的连接管理。

维护客户端连接、按主题（topic）的订阅关系，并把来自异步任务 / 数据源的消息
广播给所有订阅了对应主题的客户端。模块导出全局单例 ``ws_manager`` 供路由层复用。
"""

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """WebSocket 连接池：管理客户端连接、主题订阅与消息广播。

    所有方法均为协程，依赖单事件循环串行执行保证一致性（全部调用方都在 async
    上下文内，故不加锁）。新连接默认订阅全部主题，可由客户端通过 subscribe /
    unsubscribe 控制消息收窄。

    Attributes:
        _connections: 当前所有活跃的 WebSocket 连接集合。
        _subscriptions: 每个连接到其已订阅主题集合的映射；断开时随之清理。
        ALL_TOPICS: 系统支持的全部主题白名单（tick/order/trade/position/
            account/contract/log/task），订阅时会与之取交集做合法性过滤。
    """

    def __init__(self) -> None:
        """初始化空连接池并声明支持的主题白名单。"""
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
        """接受一个新的 WebSocket 客户端并登记到连接池。

        会先完成握手（``websocket.accept()``），随后默认订阅全部主题
        （``ALL_TOPICS`` 的副本），客户端可再发 unsubscribe 收窄。

        Args:
            websocket: 待接入的 FastAPI WebSocket 连接对象。

        Returns:
            None。副作用：连接被加入 ``_connections`` 并初始化订阅集合。
        """
        await websocket.accept()
        self._connections.add(websocket)
        self._subscriptions[websocket] = self.ALL_TOPICS.copy()

    def disconnect(self, websocket: WebSocket) -> None:
        """把一个已断开的客户端从连接池与订阅表中移除。

        对未登记的连接调用是安全的（幂等，使用 discard / pop 默认值）。

        Args:
            websocket: 已断开或需清理的 WebSocket 连接对象。

        Returns:
            None。副作用：从 ``_connections`` 和 ``_subscriptions`` 中删除该连接。
        """
        self._connections.discard(websocket)
        self._subscriptions.pop(websocket, None)

    async def handle_client_message(self, websocket: WebSocket, text: str) -> None:
        """解析并处理客户端发来的一条控制消息（JSON 文本）。

        支持的 ``action``：
          - ``subscribe``：把 ``topics`` 与 ``ALL_TOPICS`` 的交集并入该连接订阅集，
            回送 ``{"action": "subscribed", "topics": [...]}``（已排序的当前订阅）。
          - ``unsubscribe``：从订阅集移除 ``topics``，回送 ``unsubscribed`` 确认。
          - ``ping``：回送 ``{"action": "pong"}``。
        非法 JSON 回送 ``{"error": "Invalid JSON"}``；未知 action 回送
        ``{"error": "Unknown action: ..."}``。

        Args:
            websocket: 发来消息的连接，也是回执的发送目标。
            text: 客户端发来的原始文本，预期为 JSON 字符串。

        Returns:
            None。所有结果均通过 ``websocket.send_json`` 异步回送，不抛出解析异常
            （JSON 解析失败被捕获并以 error 消息回执）。
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
        """把一条消息广播给所有订阅了指定主题的客户端。

        消息先用 ``json.dumps``（``ensure_ascii=False``、非序列化对象回退 ``str``）
        序列化一次，再逐连接发送；发送时抛异常的连接视为已断开，本轮结束后统一
        从连接池清理，故对失效连接是静默跳过、不影响其余客户端。

        Args:
            message: 待广播的消息体，必须可被 ``json.dumps`` 序列化（不可序列化的
                字段会用 ``str()`` 兜底）。
            topic: 目标主题；只有订阅集中包含该主题的连接才会收到。

        Returns:
            None。副作用：向匹配连接发送文本，并把发送失败的连接移出连接池。
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
        """当前活跃连接数。

        Returns:
            连接池中 WebSocket 连接的数量；无连接时为 0。
        """
        return len(self._connections)


# Global singleton shared across the app
ws_manager = ConnectionManager()
