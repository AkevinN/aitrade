"""QMT 桥注册到 manager 与选源链接入测试。"""

from aitrade.datasource.manager import DataSourceManager
from aitrade.datasource.qmt_bridge_provider import QmtBridgeProvider
from aitrade.datasource.types import DataCategory


def test_registered_first_for_bar_history():
    mgr = DataSourceManager()
    mgr.register(QmtBridgeProvider(url="http://win", token="t"), priority=-10)

    class _Other(QmtBridgeProvider):
        name = "other"

    mgr.register(_Other(url="http://win", token="t"), priority=0)

    providers = mgr._resolve_providers(DataCategory.BAR_HISTORY)
    assert providers[0].name == "qmt"


def test_pick_bar_provider_includes_qmt():
    from aitrade.api import alpha_service
    import inspect
    src = inspect.getsource(alpha_service._pick_bar_provider)
    assert "qmt" in src
