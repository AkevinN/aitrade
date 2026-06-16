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
        """初始化通用 Webhook 通道。

        Args:
            webhook_url: 完整的 webhook 推送地址（含 token/secret 等凭证部分）；
                由 build_notifier 从环境变量注入，调用方不应硬编码或外传。
            name: 通道名，仅用于日志与 __repr__ 标识，不参与请求；默认 "webhook"。
        """
        self._url = webhook_url
        self._name = name

    def _payload(self, title: str, message: str) -> dict:
        """构造 HTTP 请求 JSON payload，子类可覆盖以定制格式。

        Args:
            title:   通知标题。
            message: 通知正文。

        Returns:
            默认格式 {"title": ..., "text": ...}。
        """
        return {"title": title, "text": message}

    def send(self, title: str, message: str) -> bool:
        """POST 到 webhook_url，HTTP 状态码 < 400 视为成功。

        网络异常时记 WARNING 日志并返回 False（由 MultiNotifier 隔离）。

        Args:
            title:   通知标题。
            message: 通知正文。

        Returns:
            HTTP 响应状态码 < 400 返回 True，网络异常或 4xx/5xx 返回 False。
        """
        try:
            resp = httpx.post(self._url, json=self._payload(title, message), timeout=_TIMEOUT)
            return resp.status_code < 400
        except Exception as exc:  # noqa: BLE001  失败由 MultiNotifier 隔离
            logger.warning("通知通道 %s 发送失败：%s", self._name, exc)
            return False

    def __repr__(self) -> str:  # 脱敏：不暴露 webhook url
        """脱敏的调试表示：只露通道类名与通道名，不暴露 webhook url。"""
        return f"<{type(self).__name__} name={self._name!r}>"


class DingTalkNotifier(WebhookNotifier):
    """钉钉自定义机器人。payload: {msgtype:'text', text:{content}}。"""

    def __init__(self, webhook_url: str) -> None:
        """初始化钉钉通道，固定通道名为 "dingtalk"。

        Args:
            webhook_url: 钉钉自定义机器人的完整 webhook 地址（含 access_token）。
        """
        super().__init__(webhook_url, name="dingtalk")

    def _payload(self, title: str, message: str) -> dict:
        """构造钉钉机器人 text 消息体。

        Args:
            title: 通知标题，作为正文首行。
            message: 通知正文，紧随标题换行拼接。

        Returns:
            形如 {"msgtype": "text", "text": {"content": "<title>\\n<message>"}} 的 dict。
        """
        return {"msgtype": "text", "text": {"content": f"{title}\n{message}"}}


class WeComNotifier(WebhookNotifier):
    """企业微信群机器人。payload: {msgtype:'text', text:{content}}。"""

    def __init__(self, webhook_url: str) -> None:
        """初始化企业微信通道，固定通道名为 "wecom"。

        Args:
            webhook_url: 企业微信群机器人的完整 webhook 地址（含 key 参数）。
        """
        super().__init__(webhook_url, name="wecom")

    def _payload(self, title: str, message: str) -> dict:
        """构造企业微信群机器人 text 消息体。

        Args:
            title: 通知标题，作为正文首行。
            message: 通知正文，紧随标题换行拼接。

        Returns:
            形如 {"msgtype": "text", "text": {"content": "<title>\\n<message>"}} 的 dict。
        """
        return {"msgtype": "text", "text": {"content": f"{title}\n{message}"}}


class ServerChanNotifier(WebhookNotifier):
    """Server酱：POST {title, desp}。"""

    def __init__(self, webhook_url: str) -> None:
        """初始化 Server酱 通道，固定通道名为 "serverchan"。

        Args:
            webhook_url: Server酱的完整推送地址（含 SendKey 的 URL）。
        """
        super().__init__(webhook_url, name="serverchan")

    def _payload(self, title: str, message: str) -> dict:
        """构造 Server酱 消息体（title 作标题、desp 作正文/支持 Markdown）。

        Args:
            title: 通知标题，对应 Server酱 的 title 字段。
            message: 通知正文，对应 Server酱 的 desp 字段。

        Returns:
            形如 {"title": <title>, "desp": <message>} 的 dict。
        """
        return {"title": title, "desp": message}


def _build_webhook(url: str) -> WebhookNotifier:
    """构造通用 Webhook 通道的工厂函数（注册表中 "webhook" 通道的构造器）。

    用于让通用 WebhookNotifier 与各定制子类在 _CHANNEL_REGISTRY 中保持一致的
    "单参数构造器" 调用形态。

    Args:
        url: 完整的 webhook 推送地址，来自环境变量 AITRADE_NOTIFY_WEBHOOK_URL。

    Returns:
        以默认 {"title", "text"} payload 形态发送的 WebhookNotifier 实例。
    """
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

    Args:
        channels: 通道名列表（来自 TradingPlan.notify_channels），合法值见
            SUPPORTED_CHANNELS；None 或空列表表示不声明任何真实通道。列表中
            未注册的通道名、或虽注册但未配置对应环境变量的通道，均被静默跳过
            （后者会记 WARNING 日志）。
        retries: 单通道发送失败时的重试次数，透传给 RetryNotifier；默认 2。

    Returns:
        装配好的 Notifier 实现：当至少有一个通道可用时返回扇出多通道的
        MultiNotifier；否则返回 LogNotifier 作为永远可用的日志兜底。
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
