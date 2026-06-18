# qmt-bridge

跑在 **Windows** 上、连 QMT/miniQMT、把行情/基本面数据经 REST 暴露给 Mac 上的 aitrade。

## 部署（Windows）
1. 安装券商 QMT/miniQMT 客户端并登录（默认连接模式①），或配置独立 xtdatacenter（模式②，需迅投 VIP token）。
2. 把本目录拷到 Windows，确保 `xtquant` 在 PYTHONPATH（随 QMT 安装目录提供，或 `pip install xtquant`）。
3. `pip install -e .`，设置环境变量（见 qmt_bridge/config.py），`uvicorn qmt_bridge.app:app --host 0.0.0.0 --port 58610`。
4. **只在内网 / Tailscale 暴露，绝不开公网。**

## 开发（Mac）
不连真 QMT，用假 xtdata 单测：`backend/.venv/bin/python -m pytest qmt-bridge/tests -v`
