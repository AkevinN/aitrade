"""
交易提醒（迭代 6）：把信号/告警推送到人。

- Notifier 协议：send(title, message) -> bool。
- LogNotifier：写日志（默认兜底，永远可用）。
- MultiNotifier：多通道扇出，单通道失败不影响其它，任一成功即视为送达。
- RetryNotifier：对单个通道做有限重试。

真实通道（钉钉/企微/server酱/邮件/飞书）实现 Notifier 即可接入；
网络调用应放在各自实现内并设超时，本模块只负责编排与失败隔离。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Notifier(Protocol):
    def send(self, title: str, message: str) -> bool:
        ...


class LogNotifier:
    """写日志的兜底通道，永远可用。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, title: str, message: str) -> bool:
        logger.info("[通知] %s | %s", title, message)
        self.sent.append((title, message))
        return True


class MultiNotifier:
    """多通道扇出：逐个发送，隔离单通道异常，任一成功即返回 True。"""

    def __init__(self, channels: list[Notifier]) -> None:
        self.channels = channels

    def send(self, title: str, message: str) -> bool:
        ok_any = False
        for ch in self.channels:
            try:
                if ch.send(title, message):
                    ok_any = True
            except Exception as exc:  # noqa: BLE001  单通道失败不应影响其它通道
                logger.warning("通知通道 %r 发送失败：%s", ch, exc)
        return ok_any


class RetryNotifier:
    """对单个通道做有限重试（同步、退避由调用方控制，这里简单重试）。"""

    def __init__(self, channel: Notifier, retries: int = 2) -> None:
        self.channel = channel
        self.retries = max(0, retries)

    def send(self, title: str, message: str) -> bool:
        for attempt in range(self.retries + 1):
            try:
                if self.channel.send(title, message):
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("通知重试 %d 失败：%s", attempt + 1, exc)
        return False
