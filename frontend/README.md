# aitrade 前端界面

基于 React + TypeScript 的量化交易分析前端，提供可视化数据展示、因子配置、模型管理等功能。

## 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 配置环境变量

创建 `.env.local` 文件（可选）：

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### 3. 启动开发服务器

```bash
npm run dev
```

访问 `http://localhost:5173` 查看应用。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VITE_API_BASE_URL` | `http://localhost:8000` | API 服务地址 |
| `VITE_WS_URL` | `ws://localhost:8000` | WebSocket 服务地址 |

## 可用脚本

### 开发模式

```bash
npm run dev
```

启动开发服务器，支持热重载。

### 构建生产版本

```bash
npm run build
```

构建优化后的生产版本，输出到 `dist/` 目录。

### 预览生产版本

```bash
npm run preview
```

在本地预览构建后的生产版本。

### 代码检查

```bash
npm run lint
```

运行 ESLint 检查代码规范。

## 依赖概览

### 主要依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `react` | ^18.x | UI 框架 |
| `react-dom` | ^18.x | React DOM 渲染 |
| `react-router-dom` | ^6.x | 路由管理 |
| `@tanstack/react-query` | ^5.x | 数据请求与缓存 |
| `zustand` | ^4.x | 状态管理 |
| `recharts` | ^2.x | 图表可视化 |
| `antd` | ^5.x | UI 组件库 |
| `dayjs` | ^1.x | 日期时间处理 |

### 开发依赖

| 依赖 | 用途 |
|------|------|
| `vite` | 构建工具 |
| `typescript` | 类型检查 |
| `eslint` | 代码规范检查 |
| `@vitejs/plugin-react` | React 插件 |

## 目录结构

```
frontend/
├── src/
│   ├── api/              # API 请求封装
│   │   ├── client.ts     # API 客户端
│   │   ├── data.ts       # 数据接口
│   │   ├── factors.ts    # 因子接口
│   │   └── models.ts     # 模型接口
│   ├── components/       # 公共组件
│   │   ├── Layout/
│   │   ├── Charts/
│   │   └── Common/
│   ├── pages/            # 页面组件
│   │   ├── Dashboard/    # 仪表盘
│   │   ├── Data/         # 数据查询
│   │   ├── Factors/      # 因子管理
│   │   └── Models/        # 模型管理
│   ├── stores/           # 状态管理
│   │   ├── appStore.ts
│   │   └── dataStore.ts
│   ├── hooks/            # 自定义 Hooks
│   ├── utils/            # 工具函数
│   ├── types/            # TypeScript 类型定义
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── public/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── .env.example
└── README.md
```

## 主要功能

- **数据看板**：股票行情、涨跌幅、成交量等实时数据展示
- **因子分析**：Alpha 因子计算与可视化
- **模型管理**：CNN 模型训练与预测
- **回测分析**：基于 Alphalens 的因子回测
- **数据导出**：支持 CSV/Excel 格式导出

## 许可证

MIT License
