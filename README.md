# aitrade

智能量化交易研究平台 - 基于 AI 的量化因子挖掘与交易策略分析系统。

## 项目概述

aitrade 是一个端到端的量化交易研究平台，融合机器学习与金融数据分析。通过 AlphaNet 等深度学习模型自动挖掘有效因子，结合可视化分析工具，帮助量化研究者快速验证交易想法。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         aitrade                                 │
│                    智能量化交易研究平台                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────┐    ┌─────────────────────────┐   │
│  │        Frontend         │    │        Backend          │   │
│  │        (React)          │    │       (FastAPI)         │   │
│  │                         │    │                         │   │
│  │  ┌─────────────────┐   │    │  ┌─────────────────┐   │   │
│  │  │   Dashboard    │   │    │  │    API Routes   │   │   │
│  │  │   (数据看板)    │   │◄──►│  │   /api/v1/*     │   │   │
│  │  └─────────────────┘   │    │  └─────────────────┘   │   │
│  │  ┌─────────────────┐   │    │  ┌─────────────────┐   │   │
│  │  │   Factors UI    │   │    │  │  Factor Engine  │   │   │
│  │  │   (因子分析)    │   │    │  │  (因子计算引擎)   │   │   │
│  │  └─────────────────┘   │    │  └─────────────────┘   │   │
│  │  ┌─────────────────┐   │    │  ┌─────────────────┐   │   │
│  │  │   Models UI     │   │    │  │   CNN Trainer   │   │   │
│  │  │   (模型管理)    │   │    │  │  (模型训练器)    │   │   │
│  │  └─────────────────┘   │    │  └─────────────────┘   │   │
│  └─────────────────────────┘    └─────────────────────────┘   │
│                                      │                         │
│                                      ▼                         │
│                        ┌─────────────────────────┐            │
│                        │     Data Layer          │            │
│                        │     (Tushare API)        │            │
│                        └─────────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 前置要求

- Python 3.10+
- Node.js 18+
- Tushare Token（[申请地址](https://tushare.pro/)）

### 1. 启动后端服务

```bash
cd aitrade/backend

# 安装 Python 依赖
pip install -r requirements.txt

# 配置 Tushare Token
export TUSHARE_TOKEN=your_token_here

# 启动服务
python run.py
```

后端服务将在 `http://localhost:8000` 启动，API 文档访问 `http://localhost:8000/docs`。

### 2. 启动前端开发服务器

```bash
cd aitrade/frontend

# 安装 Node 依赖
npm install

# 启动开发服务器
npm run dev
```

前端应用将在 `http://localhost:5173` 启动。

### 3. 访问系统

打开浏览器访问 `http://localhost:5173` 即可使用系统。

## 功能特性

### 数据管理
- 股票列表查询与筛选
- 日线、分钟线历史数据获取
- 上市公司基本信息查询
- 数据本地缓存加速访问

### 因子研究
- 内置经典 Alpha 因子库
- 自定义因子计算
- 因子 IC/IR 分析
- 因子相关性热力图
- Alphalens 回测框架集成

### 模型训练
- AlphaNet 深度学习模型
- CNN 图像识别模式挖掘
- 模型训练进度可视化
- 模型预测与评估
- 模型版本管理

### 可视化分析
- K 线图表与成交量
- 因子时间序列图
- 收益分布直方图
- 多因子对比分析

## 技术栈

### 后端
| 技术 | 用途 |
|------|------|
| FastAPI | Web 框架 |
| Uvicorn | ASGI 服务器 |
| Pydantic | 数据验证 |
| Loguru | 日志记录 |
| Polars | 数据分析 |
| NumPy/SciPy | 科学计算 |
| scikit-learn | 传统 ML |
| LightGBM | 梯度提升 |
| PyTorch | 深度学习 |
| Tushare | 数据源 |
| TA-Lib | 技术指标 |

### 前端
| 技术 | 用途 |
|------|------|
| React 18 | UI 框架 |
| TypeScript | 类型安全 |
| Vite | 构建工具 |
| React Router | 路由管理 |
| TanStack Query | 数据请求 |
| Zustand | 状态管理 |
| Recharts | 图表可视化 |
| Ant Design | UI 组件库 |

## 目录结构

```
aitrade/
├── backend/                    # 后端服务
│   ├── aitrade/                # 主包
│   │   ├── __init__.py
│   │   ├── __main__.py         # CLI 入口
│   │   ├── main.py             # FastAPI 应用
│   │   ├── config.py           # 配置管理
│   │   ├── api/                # API 路由
│   │   │   ├── __init__.py
│   │   │   ├── data.py
│   │   │   ├── factors.py
│   │   │   └── models.py
│   │   ├── core/               # 核心模块
│   │   │   ├── __init__.py
│   │   │   ├── data_manager.py
│   │   │   ├── factor_engine.py
│   │   │   └── tushare_client.py
│   │   ├── models/             # 数据模型
│   │   │   ├── __init__.py
│   │   │   ├── stock.py
│   │   │   └── factor.py
│   │   └── utils/              # 工具函数
│   │       └── helpers.py
│   ├── run.py                  # 启动脚本
│   └── requirements.txt
│
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── api/               # API 请求
│   │   ├── components/         # 组件
│   │   ├── pages/              # 页面
│   │   ├── stores/            # 状态
│   │   ├── hooks/             # Hooks
│   │   ├── types/             # 类型定义
│   │   └── utils/             # 工具函数
│   ├── package.json
│   └── vite.config.ts
│
├── .gitignore
└── README.md
```

## 数据存储

运行时数据存储在 `~/.aitrade/` 目录：

```
~/.aitrade/
├── data/           # 股票数据缓存
│   ├── daily/      # 日线数据
│   └── minute/    # 分钟线数据
├── factors/        # 因子计算结果
├── models/         # 训练好的模型
├── logs/           # 日志文件
└── config.json     # 配置文件
```

## 环境变量

### 后端环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AITRADE_HOST` | `0.0.0.0` | 服务监听地址 |
| `AITRADE_PORT` | `8000` | 服务监听端口 |
| `TUSHARE_TOKEN` | - | Tushare Token（必需） |
| `TUSHARE_BACKEND` | `tushare` | 数据后端 |
| `AITRADE_MAX_WORKERS` | `4` | 最大工作线程 |

### 前端环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VITE_API_BASE_URL` | `http://localhost:8000` | API 地址 |
| `VITE_WS_URL` | `ws://localhost:8000` | WebSocket 地址 |

## 相关项目

- [vnpy](https://github.com/vnpy/vnpy) - VN.PY 量化交易框架
- [Alphalens](https://github.com/alphalens/alphalens) - 因子分析工具
- [Tushare](https://tushare.pro/) - 金融数据接口

## 许可证

MIT License
