"""qmt-bridge 配置（环境变量）。"""

from __future__ import annotations

import os

# REST 鉴权 token（Mac↔Win）
BRIDGE_TOKEN = os.getenv("QMT_BRIDGE_TOKEN", "")
# 监听
HOST = os.getenv("QMT_BRIDGE_HOST", "0.0.0.0")
PORT = int(os.getenv("QMT_BRIDGE_PORT", "58610"))
# 连接模式：'client'=连券商 QMT 客户端（默认）；'xtdc'=独立 xtdatacenter
CONNECT_MODE = os.getenv("QMT_BRIDGE_CONNECT_MODE", "client")
# 模式②用：迅投 VIP token
XTDC_TOKEN = os.getenv("QMT_XTDC_TOKEN", "")
# 复权口径：等比 or 普通
RATIO_ADJUST = os.getenv("QMT_BRIDGE_RATIO_ADJUST", "false").lower() in ("1", "true")
