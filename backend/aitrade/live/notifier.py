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
    """通知器协议：实现 send(title, message) -> bool 即可接入。

    返回 True 表示至少一条通知成功送达；False 表示全部失败。
    实现类应在内部处理超时与异常，不应向调用方传播网络错误。
    """

    def send(self, title: str, message: str) -> bool:
        ...


class LogNotifier:
    """写日志的兜底通道，永远可用。

    用于测试或作为 MultiNotifier 无真实通道时的降级兜底。
    所有发送历史记录在 self.sent 属性中，便于测试断言。

    Example:
        >>> n = LogNotifier()
        >>> n.send("买入信号", "000001.SZSE 买入 100 股")
        True
    """

    def __init__(self) -> None:
        """初始化 LogNotifier，sent 列表用于记录历史通知。"""
        self.sent: list[tuple[str, str]] = []

    def send(self, title: str, message: str) -> bool:
        """写 INFO 日志并记录到 sent，永远返回 True。

        Args:
            title:   通知标题。
            message: 通知正文。

        Returns:
            始终 True（日志写入不会失败）。
        """
        logger.info("[通知] %s | %s", title, message)
        self.sent.append((title, message))
        return True


class MultiNotifier:
    """多通道扇出：逐个发送，隔离单通道异常，任一成功即返回 True。

    单通道抛出异常时记 WARNING 日志并继续，不影响其它通道。
    用于组合多个真实通道（钉钉 + 企微等），避免单点故障阻断提醒。

    Example:
        >>> n = MultiNotifier([DingTalkNotifier(url), WeComNotifier(url)])
        >>> n.send("信号", "...")
    """

    def __init__(self, channels: list[Notifier]) -> None:
        """初始化多通道扇出器。

        Args:
            channels: Notifier 列表，至少一个；空列表时 send 始终返回 False。
        """
        self.channels = channels

    def send(self, title: str, message: str) -> bool:
        """逐通道扇出发送，任一成功返回 True；全部失败返回 False。

        Args:
            title:   通知标题。
            message: 通知正文。

        Returns:
            任意通道发送成功则 True，全部失败则 False。
        """
        ok_any = False
        for ch in self.channels:
            try:
                if ch.send(title, message):
                    ok_any = True
            except Exception as exc:  # noqa: BLE001  单通道失败不应影响其它通道
                logger.warning("通知通道 %r 发送失败：%s", ch, exc)
        return ok_any


class RetryNotifier:
    """对单个通道做有限次同步重试。

    适用于网络偶发抖动场景；退避策略由调用方控制，本类仅做线性重试。
    全部重试失败时返回 False（不抛出异常）。

    Example:
        >>> n = RetryNotifier(DingTalkNotifier(url), retries=3)
        >>> n.send("信号", "...")
    """

    def __init__(self, channel: Notifier, retries: int = 2) -> None:
        """初始化重试通知器。

        Args:
            channel: 被包裹的通知通道。
            retries: 额外重试次数（0 表示仅尝试一次），负值自动归零。
        """
        self.channel = channel
        self.retries = max(0, retries)

    def send(self, title: str, message: str) -> bool:
        """发送通知，失败时重试最多 retries 次。

        Args:
            title:   通知标题。
            message: 通知正文。

        Returns:
            任意一次发送返回 True 则立即返回 True；全部失败则返回 False。
        """
        for attempt in range(self.retries + 1):
            try:
                if self.channel.send(title, message):
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("通知重试 %d 失败：%s", attempt + 1, exc)
        return False
