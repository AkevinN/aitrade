"""QMT 桥配置项存在性测试。"""

import importlib


def test_qmt_bridge_config_defaults(monkeypatch):
    monkeypatch.delenv("QMT_BRIDGE_URL", raising=False)
    monkeypatch.delenv("QMT_BRIDGE_TOKEN", raising=False)
    import aitrade.config as cfg
    importlib.reload(cfg)
    assert cfg.QMT_BRIDGE_URL == ""
    assert cfg.QMT_BRIDGE_TOKEN == ""
