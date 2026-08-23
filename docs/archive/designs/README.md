# 历史设计归档

本目录保存 NexusMind Knowledge Runtime 在各功能首次实现时的设计记录。

这些文档具有架构决策和实现演进的参考价值，但其中的 scope、non-goals、roadmap、测试数量和产品状态均以当时为准，**不代表当前产品规范**。当前行为应以以下内容为准：

- [README](../../../README.md)：安装、CLI 和 Python API 使用方式
- [当前技术架构](../../architecture.md)：现行组件边界、数据流和安全模型
- 公共 Python contract 与测试：具体行为和兼容边界

## 归档内容

- Document chunking
- BM25 与 Unicode/CJK lexical analysis
- Semantic 与 Hybrid-RRF retrieval
- Bounded reranking
- Resolved provenance-aware search
- Retrieval evaluation 与 backend comparison
- KnowledgeBase source registry 和持久化布局
- Knowledge inspection 与 retrieval diagnostics

已经执行完成的逐步实施计划未保留；Git 历史仍可用于追溯这些计划。
