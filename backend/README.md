# aitrade 后端服务

基于 FastAPI 的量化交易数据服务后端，提供股票数据查询、因子计算、模型训练等功能。

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

在运行前设置必要的环境变量：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `AITRADE_HOST` | `0.0.0.0` | API 服务监听地址 |
| `AITRADE_PORT` | `8000` | API 服务监听端口 |
| `TUSHARE_TOKEN` | - | Tushare API Token（必需） |
| `TUSHARE_BACKEND` | `tushare` | 数据后端类型 |
| `AITRADE_MAX_WORKERS` | `4` | 最大并发工作线程数 |

也可以创建 `.env` 文件：

```bash
TUSHARE_TOKEN=your_token_here
AITRADE_PORT=8000
```

### 3. 启动服务

方式一（推荐）：

```bash
python run.py
```

方式二（使用 CLI）：

```bash
python -m aitrade
```

方式三（直接使用 uvicorn）：

```bash
uvicorn aitrade.main:app --host 0.0.0.0 --port 8000
```

服务启动后访问 `http://localhost:8000/docs` 查看 API 文档。

## 目录结构

```
backend/
├── aitrade/
│   ├── __init__.py
│   ├── __main__.py      # CLI 入口
│   ├── main.py           # FastAPI 应用主入口
│   ├── config.py         # 配置管理
│   ├── api/              # API 路由
│   │   ├── __init__.py
│   │   ├── data.py       # 数据查询 API
│   │   ├── factors.py    # 因子计算 API
│   │   └── models.py     # 模型管理 API
│   ├── core/             # 核心模块
│   │   ├── __init__.py
│   │   ├── data_manager.py    # 数据管理器
│   │   ├── factor_engine.py   # 因子引擎
│   │   └── tushare_client.py  # Tushare 客户端
│   ├── models/           # 数据模型
│   │   ├── __init__.py
│   │   ├── stock.py      # 股票数据模型
│   │   └── factor.py     # 因子数据模型
│   └── utils/            # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── run.py               # 启动脚本
├── requirements.txt
└── README.md
```

## API 端点概览

### 数据接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/stock/list` | 获取股票列表 |
| `GET` | `/api/v1/stock/daily` | 获取日线数据 |
| `GET` | `/api/v1/stock/minute` | 获取分钟线数据 |
| `GET` | `/api/v1/stock/info` | 获取股票基本信息 |

### 因子接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/factors/alpha` | 计算 Alpha 因子 |
| `POST` | `/api/v1/factors/batch` | 批量计算因子 |
| `GET` | `/api/v1/factors/list` | 获取可用因子列表 |

### 模型接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/models/train` | 训练模型 |
| `GET` | `/api/v1/models/predict` | 模型预测 |
| `GET` | `/api/v1/models/list` | 获取模型列表 |
| `DELETE` | `/api/v1/models/{id}` | 删除模型 |

### 系统接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/status` | 服务状态 |

## 数据存储布局

数据存储在项目根目录下的 `./.aitrade/`（可通过环境变量 `AITRADE_HOME` 覆盖）：

```
${AITRADE_HOME}/
├── alpha_lab/            # K线/数据集/模型存储（Parquet + pickle）
└── cnn_models/          # CNN 模型（.pt + history）
```

## 故障排除

### Tushare Token 未配置

错误信息：`Tushare token not configured`

解决方式：设置 `TUSHARE_TOKEN` 环境变量或在 `.env` 文件中配置。

### 端口被占用

错误信息：`Address already in use`

解决方式：
1. 检查是否有其他进程占用端口 `8000`
2. 修改 `AITRADE_PORT` 环境变量使用其他端口

### 数据拉取失败

错误信息：`Failed to fetch data from Tushare`

解决方式：
1. 检查网络连接
2. 确认 Tushare Token 有效
3. 检查 Tushare 接口配额是否用尽

### 依赖安装失败

某些依赖（如 `ta-lib`）可能需要系统依赖：

```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# CentOS/RHEL
sudo yum install ta-lib
```

## 许可证

MIT License
