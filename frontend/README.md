# Paper Agent Frontend

基于 **Next.js 14 App Router** + **TailwindCSS** + **shadcn/ui** 构建的 LLM Agent 科研平台前端。

## 技术栈

| 技术 | 用途 |
|------|------|
| Next.js 14 (App Router) | 路由 & SSR |
| TailwindCSS | 样式 |
| shadcn/ui (Radix UI) | 组件库 |
| Zustand | 状态管理 |
| React Flow | 知识图谱可视化 |
| react-markdown + rehype-highlight | Markdown 渲染 & 代码高亮 |
| Axios | HTTP 请求 |

## 快速启动

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问 [http://localhost:3000](http://localhost:3000)

> 确保后端 FastAPI 服务已在 `http://127.0.0.1:8000` 启动，Next.js 会自动代理 `/api` 和 `/knowledge` 请求。

## 页面结构

```
/chat      → 聊天页面（多会话 + 流式回复 + Markdown）
/rag       → 知识库管理（创建/编辑/删除 + 文档上传）
/graphrag  → 知识图谱可视化（实体节点 + 关系边 + 交互）
```

## 目录结构

```
src/
├── app/                  # Next.js App Router
│   ├── chat/page.tsx
│   ├── rag/page.tsx
│   ├── graphrag/page.tsx
│   ├── layout.tsx        # 全局布局（TopNav）
│   └── globals.css
├── components/
│   ├── layout/           # TopNav, ThemeProvider
│   ├── chat/             # ChatSidebar, ChatWindow, MessageList, ChatInput, MessageItem
│   ├── rag/              # RagPage, KnowledgeBaseCard, DocumentTable, CreateModal
│   ├── graph/            # GraphViewer, GraphNode, NodeInfoPanel, GraphToolbar, GraphRagPage
│   └── ui/               # Button, Input, Card, Dialog, Badge, ...
├── store/
│   ├── chatStore.ts      # 会话 & 消息状态
│   ├── ragStore.ts       # 知识库状态
│   └── graphStore.ts     # 图谱状态
├── lib/
│   ├── api.ts            # API 封装
│   └── utils.ts          # 工具函数
└── types/
    └── index.ts          # TypeScript 类型
```
