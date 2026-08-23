# 从 NexusMind 0.1 迁移到 0.2

NexusMind 0.2 将产品范围收敛为 KnowledgeBase。这是一次有意的 breaking change。

## 已删除的入口

CLI 不再提供：

- `nexusmind chat`
- `nexusmind tools`
- `nexusmind runs`
- `nexusmind mcp`
- `nexusmind skill`

Python 包不再包含 Agent Runtime、Harness、Tool Registry、Workspace、命令 Profile、MCP、Agent Skill、Run History、Checkpoint 和 Run Lease API。

这些能力没有兼容 shim。依赖它们的调用方应继续固定使用 NexusMind 0.1，或将 Agent 功能迁移到独立项目。

## KnowledgeBase CLI

0.2 的 CLI 只包含：

```text
nexusmind create
nexusmind source add|list|remove
nexusmind sync
nexusmind search
nexusmind query
nexusmind inspect
nexusmind diagnose
nexusmind-kb
```

`query` 现在通过 Knowledge-native `OpenAICompatibleAnswerProvider` 调用模型，不再依赖 Agent message、runtime event 或 tool-call contract。模型环境变量保持不变。

## Python API

`KnowledgeBase`、ingestion、chunking、retrieval、context assembly、citation validation、inspection、diagnostics 和 evaluation API 保留。新的 provider 可直接导入：

```python
from nexusmind import OpenAICompatibleAnswerProvider
```

## Persistence 与 Skill

删除的是 Agent Run History persistence，不是否定 Knowledge persistence。未来的 Knowledge Trace、Query History 和 Evaluation Record 可以作为独立、显式启用的 Knowledge Layer 能力加入。

删除的也是依赖 Agent Tool/MCP 的 Skill 实现。未来的 `citation_export`、`document_summary` 或 `knowledge_transform` 等 Knowledge-native Skill 应直接组合公开 Knowledge API。
