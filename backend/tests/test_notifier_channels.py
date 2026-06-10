"""通知通道与装配工厂单元测试（交易计划自动化 任务 2）。"""

from __future__ import annotations

import json

import pytest

from aitrade.live import notifier_channels as nc
from aitrade.live.notifier import LogNotifier, MultiNotifier


def test_dingtalk_payload_shape() -> None:
    n = nc.DingTalkNotifier("https://example.com/hook")
    assert n._payload("标题", "正文") == {"msgtype": "text", "text": {"content": "标题\n正文"}}


def test_wecom_payload_shape() -> None:
    n = nc.WeComNotifier("https://example.com/hook")
    assert n._payload("标题", "正文") == {"msgtype": "text", "text": {"content": "标题\n正文"}}


def test_serverchan_payload_shape() -> None:
    n = nc.ServerChanNotifier("https://sctapi.ftqq.com/x.send")
    assert n._payload("标题", "正文") == {"title": "标题", "desp": "正文"}


def test_send_network_error_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("network down")

    monkeypatch.setattr(nc.httpx, "post", _boom)
    n = nc.WebhookNotifier("https://example.com/hook", name="webhook")
    assert n.send("t", "m") is False  # 不抛，返回 False


def test_send_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200

    monkeypatch.setattr(nc.httpx, "post", lambda *a, **k: _Resp())
    n = nc.DingTalkNotifier("https://example.com/hook")
    assert n.send("t", "m") is True


def test_build_notifier_no_credentials_falls_back_to_log(monkeypatch: pytest.MonkeyPatch) -> None:
    for _, (env_key, _) in nc._CHANNEL_REGISTRY.items():
        monkeypatch.delenv(env_key, raising=False)
    result = nc.build_notifier(["dingtalk", "wecom"])
    assert isinstance(result, LogNotifier)  # 无凭证退回 LogNotifier（Req 1.5）


def test_build_notifier_with_credentials_returns_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITRADE_NOTIFY_DINGTALK_WEBHOOK", "https://dt/hook")
    monkeypatch.setenv("AITRADE_NOTIFY_WECOM_WEBHOOK", "https://wc/hook")
    result = nc.build_notifier(["dingtalk", "wecom"])
    assert isinstance(result, MultiNotifier)
    assert len(result.channels) == 2


def test_build_notifier_skips_unconfigured_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITRADE_NOTIFY_DINGTALK_WEBHOOK", "https://dt/hook")
    monkeypatch.delenv("AITRADE_NOTIFY_WECOM_WEBHOOK", raising=False)
    result = nc.build_notifier(["dingtalk", "wecom"])
    assert isinstance(result, MultiNotifier)
    assert len(result.channels) == 1  # 仅装配配置了凭证的 dingtalk


def test_build_notifier_repr_and_serialization_exclude_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "https://secret-webhook.example.com/SUPERSECRETTOKEN"
    monkeypatch.setenv("AITRADE_NOTIFY_DINGTALK_WEBHOOK", sentinel)
    result = nc.build_notifier(["dingtalk"])
    # repr 不含凭证
    assert sentinel not in repr(result)
    for ch in result.channels:  # type: ignore[attr-defined]
        assert sentinel not in repr(ch)
    # 通道名仍可见
    assert "dingtalk" in repr(result.channels[0].channel)  # type: ignore[attr-defined]
