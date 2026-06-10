"""
真实通知通道（交易计划自动化，新增粘合 1）：把买卖信号经 HTTP webhook 推送到人。

实现既有 `Notifier` 协议（`send(title, message) -> bool`），与既有
`MultiNotifier`/`RetryNotifier`/`LogNotifier` 完全兼容。

脱敏红线（需求 9.4）：通道的 webhook 地址 / secret / token 只从**环境变量**读取，
**绝不**回传调用方、写入交易计划、写入 API 响应或日志。`build_notifier` 接收的是
**通道名列表**（来自 TradingPlan.notify_channels），凭证由约定环境变量在运行时解析。
"""

from __future__ import annotations

import logging
import os

import httpx

from .notifier import LogNotifier, MultiNotifier, Notifier, RetryNotifier

logger = logging.getLogger(__name__)

# 网络调用统一超时（秒），避免推送阻塞决策线程。
_TIMEOUT = 5.0


class WebhookNotifier:
    """通用 Webhook 通道基类：POST JSON 到 webhook_url。子类定制 payload 形态。

    凭证（webhook_url）由 build_notifier 从环境变量注入，本对象不对外暴露该地址
    （`__repr__` 仅显示通道名，避免凭证经 repr/序列化泄露）。
    """

    def __init__(self, webhook_url: str, *, name: str = "webhook") -> None:
        self._url = webhook_url
        self._name = name

    def _payload(self, title: str, message: str) -> dict:
        return {"title": title, "text": message}

    def send(self, title: str, message: str) -> bool:
        try:
            resp = httpx.post(self._url, json=self._payload(title, message), timeout=_TIMEOUT)
            return resp.status_code < 400
        except Exception as exc:  # noqa: BLE001  失败由 MultiNotifier 隔离
            logger.warning("通知通道 %s 发送失败：%s", self._name, exc)
            return False

    def __repr__(self) -> str:  # 脱敏：不暴露 webhook url
        return f"<{type(self).__name__} name={self._name!r}>"


class DingTalkNotifier(WebhookNotifier):
    """钉钉自定义机器人。payload: {msgtype:'text', text:{content}}。"""

    def __init__(self, webhook_url: str) -> None:
        super().__init__(webhook_url, name="dingtalk")

    def _payload(self, title: str, message: str) -> dict:
        return {"msgtype": "text", "text": {"content": f"{title}\n{message}"}}


class WeComNotifier(WebhookNotifier):
    """企业微信群机器人。payload: {msgtype:'text', text:{content}}。"""

    def __init__(self, webhook_url: str) -> None:
        super().__init__(webhook_url, name="wecom")

    def _payload(self, title: str, message: str) -> dict:
        return {"msgtype": "text", "text": {"content": f"{title}\n{message}"}}


class ServerChanNotifier(WebhookNotifier):
    """Server酱：POST {title, desp}。"""

    def __init__(self, webhook_url: str) -> None:
        super().__init__(webhook_url, name="serverchan")

    def _payload(self, title: str, message: str) -> dict:
        return {"title": title, "desp": message}


def _build_webhook(url: str) -> WebhookNotifier:
    return WebhookNotifier(url, name="webhook")


# 通道名 → (环境变量名, 构造器)。凭证只在 build_notifier 内经环境变量解析。
_CHANNEL_REGISTRY: dict[str, tuple[str, "type[WebhookNotifier] | object"]] = {
    "dingtalk": ("AITRADE_NOTIFY_DINGTALK_WEBHOOK", DingTalkNotifier),
    "wecom": ("AITRADE_NOTIFY_WECOM_WEBHOOK", WeComNotifier),
    "serverchan": ("AITRADE_NOTIFY_SERVERCHAN_URL", ServerChanNotifier),
    "webhook": ("AITRADE_NOTIFY_WEBHOOK_URL", _build_webhook),
}

# 合法通道名集合（供模型层校验复用）。
SUPPORTED_CHANNELS: tuple[str, ...] = tuple(_CHANNEL_REGISTRY.keys())


def build_notifier(channels: list[str] | None, *, retries: int = 2) -> Notifier:
    """按通道名列表装配 Notifier。

    - 仅装配「在 channels 列表中声明」且「在环境变量中配置了凭证」的通道。
    - 每个通道用 RetryNotifier 包装（有限重试），整体用 MultiNotifier 扇出
      （任一成功即送达、单通道失败隔离）。
    - 无可用真实通道时退回 LogNotifier（写日志兜底，永远可用，需求 1.5）。

    凭证只在此处从环境读取，绝不回传调用方/计划/日志（需求 1.6 / 9.4）。
    """
    built: list[Notifier] = []
    for name in channels or []:
        spec = _CHANNEL_REGISTRY.get(name)
        if spec is None:
            continue
        env_key, ctor = spec
        url = os.getenv(env_key, "").strip()
        if not url:
            logger.warning("通知通道 %s 已声明但未配置环境变量 %s，跳过", name, env_key)
            continue
        built.append(RetryNotifier(ctor(url), retries=retries))  # type: ignore[operator]
    if not built:
        return LogNotifier()
    return MultiNotifier(built)
