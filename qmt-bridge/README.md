# qmt-bridge

跑在 **Windows** 上、连 QMT/miniQMT、把行情/基本面数据经 REST 暴露给 Mac 上的 aitrade。
Mac 端的 `QmtBridgeProvider` 永远经 REST 调它，Mac 永不 import xtquant。

---

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `QMT_BRIDGE_TOKEN` | `""` | **REST 鉴权 token（必设）**。Mac 端 `QMT_BRIDGE_TOKEN` 必须与此一致。空值时服务拒绝所有数据请求。 |
| `QMT_BRIDGE_HOST` | `0.0.0.0` | 监听网卡。`0.0.0.0` = 所有网卡。 |
| `QMT_BRIDGE_PORT` | `58610` | 监听端口。 |
| `QMT_BRIDGE_CONNECT_MODE` | `client` | `client`=连券商 QMT/miniQMT 客户端（免费行情，客户端需登录）；`xtdc`=独立 xtdatacenter（无 GUI 常驻，需迅投 VIP token）。 |
| `QMT_XTDC_TOKEN` | `""` | 仅 `xtdc` 模式用：迅投 VIP token。 |
| `QMT_BRIDGE_RATIO_ADJUST` | `false` | 复权口径：`false`=普通(等额)前/后复权，`true`=等比(`front_ratio`/`back_ratio`)。 |

---

## 部署（Windows）

1. 安装券商 QMT/miniQMT 客户端并登录（默认连接模式 `client`），或配置独立 xtdatacenter（`xtdc` 模式，需迅投 VIP token）。
2. 把本目录拷到 Windows，确保 `xtquant` 在 PYTHONPATH（随 QMT 安装目录提供，或 `pip install xtquant`）。
3. 安装并启动：
   ```bat
   pip install -e .
   set QMT_BRIDGE_TOKEN=<一段强随机串>
   uvicorn qmt_bridge.app:app --host 0.0.0.0 --port 58610
   ```
4. **放行防火墙入站端口**（否则 Mac 连不上）：
   ```bat
   netsh advfirewall firewall add rule name="qmt-bridge" dir=in action=allow protocol=TCP localport=58610
   ```
5. **只在内网 / Tailscale 暴露，绝不开公网。**

---

## 组网：Mac 怎么访问 Windows

Mac 端只需把 `QMT_BRIDGE_URL` 指到"Windows 上能被 Mac 到达的 地址:端口"。地址怎么定取决于两机的网络关系。

### 推荐：Tailscale（同网/异网都用同一个地址，免切换）

适用于"有时同一局域网、有时家↔公司不同网络"。Tailscale 给每台机一个**跟着机器走、不随物理网络变**的私有地址，同网自动走局域网直连（低延迟），异网自动加密穿透；`QMT_BRIDGE_URL` 始终不变。

1. 两台机都装 [Tailscale](https://tailscale.com/download) 并**登录同一账号**：
   - Windows：装客户端 → `tailscale up` → 登录。
   - Mac：`brew install --cask tailscale`（或官网客户端）→ 登录同一账号。
2. 在 Tailscale 后台开启 **MagicDNS**（可用机器名代替 IP）。
3. Windows 服务照常绑 `0.0.0.0:58610` + 防火墙放行（见上）。
4. Mac 端用 Tailscale 机器名（或 `100.x.y.z` 私有 IP）：
   ```bash
   export QMT_BRIDGE_URL=http://<windows-机器名>:58610
   ```

> 异网场景下，家里那台 Windows 需开机 + Tailscale 在跑 + 服务在跑。好在本系统是"先同步到本地 Parquet、研究离线读缓存"，平时研究/回测不需要 Windows 在线，**只有同步新数据那一刻**才需要它可达。

### 简单：同一局域网

两机长期在同一 LAN 时够用。
1. Windows `ipconfig` 看 IPv4（形如 `192.168.1.23`）。
2. Mac：`export QMT_BRIDGE_URL=http://192.168.1.23:58610`

> ⚠️ LAN IP 是 DHCP 分配会变 → 在路由器做 IP 绑定/保留，或用静态 IP，否则地址变了就连不上。出了这个网即不可达。

### Windows 跑在 Mac 的虚拟机里（Parallels/UTM）

用虚拟机的宿主可达 IP（共享网络下 Parallels 常见 `10.211.55.x`），`export QMT_BRIDGE_URL=http://<VM-IP>:58610`。

---

## Mac 端接入与自测

1. 先在 Mac 上手测连通性（`/health` 不需要 token）：
   ```bash
   curl http://<地址>:58610/health
   # 期望返回 JSON，如 {"connected": true, "version": "0.1.0"}

   # 带 token 验鉴权（数据路由需要）
   curl -H "Authorization: Bearer <你的token>" \
        "http://<地址>:58610/contracts?include_bse=false" | head
   ```
   `/health` 通了说明网络 + 防火墙 OK——这正是 `QmtBridgeProvider.init()` 内部做的探活。
2. 设两个环境变量再启动 aitrade backend：
   ```bash
   export QMT_BRIDGE_URL=http://<地址>:58610
   export QMT_BRIDGE_TOKEN=<和 Windows 端 QMT_BRIDGE_TOKEN 一致的串>
   ```
   后端启动时会自动注册 `QmtBridgeProvider`（优先级最前）并 `init → /health`，连上即 `AVAILABLE`；QMT 给不了的（估值/行业等）自动落回 tushare。
3. 前端"数据下载"页用 `provider='qmt'` 即走 QMT 桥取数。

---

## 安全红线

- **始终带 bearer token**：Windows 端 `QMT_BRIDGE_TOKEN` 设强随机串，Mac 端一致。
- **优先 Tailscale**，把端口留在私有网内；可让服务只绑 Tailscale 网卡 + 防火墙挡掉其他来源。
- **绝不给 58610 做公网端口映射。**
- 区分两个 token：迅投 VIP token（`QMT_XTDC_TOKEN`，仅 `xtdc` 模式）vs REST 鉴权 token（`QMT_BRIDGE_TOKEN`，Mac↔Win），别混。

---

## 开发（Mac）

不连真 QMT，用假 xtdata 单测：

```bash
backend/.venv/bin/python -m pytest qmt-bridge/tests -v
```
