# NexusMind

NexusMind 是一个面向本地知识的 Knowledge Runtime / KnowledgeBase 工具。它可以注册本地文本来源、显式同步内容、执行本地检索，并通过模型生成带可验证引用的回答。

你可以通过本地桌面界面管理知识库，通过 CLI 搜索和提问，也可以将 `KnowledgeBase` 作为 Python 库集成到自己的应用中。

> 当前支持 Windows 与 Python 3.11、3.12、3.13。Linux 和 macOS 暂不属于正式支持平台。

## 当前状态

当前 `main` 已经具备完整的第一版 KnowledgeBase 工作流，但仓库目前**没有公开的 GitHub Release**。如需开发使用，可以直接从源码安装；Windows portable 产物可以通过仓库构建脚本或 CI 构建。

NexusMind 当前重点是验证真实知识库使用体验和检索质量，而不是继续扩展更多数据源或 Agent 能力。

## 五步快速开始

先准备一个严格 UTF-8 编码的文本来源：

```powershell
New-Item -ItemType Directory -Force .\security-notes | Out-Null
Set-Content -Encoding utf8 .\security-notes\keys.md "密钥轮换需要记录负责人和生效时间。"
```

然后创建 KnowledgeBase、注册来源、显式同步、搜索并检查 canonical 状态：

```powershell
nexusmind create .\security-kb
nexusmind source add .\security-notes --knowledge-base .\security-kb
nexusmind sync --knowledge-base .\security-kb
nexusmind search "密钥轮换" --knowledge-base .\security-kb --limit 5
nexusmind inspect --knowledge-base .\security-kb
```

当前版本支持本地文件和目录中的严格 UTF-8 `.txt`、`.md`、`.markdown`，以及 `.doc`、`.docx`、`.pdf`、`.ppt`、`.pptx`、`.rtf`、`.epub`、`.odt` 结构化文档。注册来源后必须运行 `sync`，内容才会进入 KnowledgeBase。

默认 `search` 使用完全离线、支持 Unicode/CJK 的 BM25。Semantic、Hybrid-RRF、reranking 和模型问答属于可选能力。

当前不支持 Git/GitHub 来源、扫描版 PDF OCR、电子表格 ingestion、后台同步或文件监控、云端 KnowledgeBase，也不会持久化派生的 Chunk、embedding 或检索索引；重新打开时会从 canonical documents 重建这些派生状态。

## 核心能力

- 创建和重新打开本地持久化 KnowledgeBase
- 注册本地文件或目录，并显式控制同步时机
- 保存 canonical document 和不可变的内部文档版本历史
- 默认使用完全离线、支持 Unicode/CJK 的 BM25 检索
- 可选使用 Semantic、Hybrid-RRF 和 reranker 检索后端
- 对用户可见搜索结果执行 document-aware diversification，同时保留 backend 原始分数
- 为模型问答执行有界的 LLM query expansion 和 query-level RRF
- Query Expansion 失败时自动回退到原始问题检索
- 根据检索内容生成回答，并验证回答中的引用
- 检查来源、文档、分块和各检索阶段的诊断信息
- 提供本地桌面界面、CLI 和 Python API
- 提供 Windows portable 构建和真实 smoke test

底层数据流、存储格式、检索实现和安全边界请参阅 [技术架构](docs/architecture.md)。

## 开发环境安装与桌面界面

### 1. 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

也可以通过 `requirements.txt` 安装运行依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. 打开 KnowledgeBase 界面

```powershell
nexusmind-kb
```

在界面中可以：

1. 创建新的 KnowledgeBase，或打开已有知识库；
2. 注册本地 UTF-8 文件或目录；
3. 显式同步全部来源或单个来源；
4. 搜索内容并查看来源、相关度、Chunk ID 和原文片段；
5. 删除不再需要的来源。

界面基于 Python 标准库 `tkinter`，不需要本地服务器、外部服务或额外账号。

## 通过 CLI 使用

CLI 覆盖 KnowledgeBase 的创建、来源管理、同步、搜索、问答、检查与诊断：

```powershell
nexusmind create ./security-kb
nexusmind source add ./security-notes --knowledge-base ./security-kb
nexusmind source list --knowledge-base ./security-kb
nexusmind sync --knowledge-base ./security-kb
nexusmind search "密钥轮换" --knowledge-base ./security-kb --limit 5
nexusmind inspect --knowledge-base ./security-kb
nexusmind diagnose "密钥轮换" --knowledge-base ./security-kb --limit 5
```

这组命令对应一条完整的“创建 → 注册来源 → 同步 → 检索 → 排查”工作流：

| 命令 | 用途 | 是否修改 KnowledgeBase |
| --- | --- | --- |
| `create` | 创建 KnowledgeBase。内部 ID 由程序自动生成，目标目录必须不存在或为空。 | 是 |
| `source add` | 注册文件或目录来源。注册只保存配置，不读取或索引内容。 | 是，仅注册来源 |
| `source list` | 列出已注册来源类型和路径。 | 否 |
| `sync` | 读取来源，把当前文件内容提交为 canonical documents，并重建派生检索状态。 | 是 |
| `search` | 调用配置的 retrieval backend，并在有界候选上执行 document-aware 最终选择。 | 否 |
| `inspect` | 查看 KnowledgeBase、来源、canonical documents 和文档 Chunk。 | 否 |
| `diagnose` | 输出 raw retrieval diagnostics，用于分析各阶段排名、分数和命中情况。 | 否 |
| `query` | 进行 query expansion、多查询融合、上下文组装和带引用回答。 | 否 |

需要特别区分以下概念：

- **注册不等于同步**：`source add` 只告诉 KnowledgeBase“去哪里找资料”；`sync` 才真正读取文件。
- **来源不等于文档**：一个 directory 来源可以产生多篇 canonical documents；`source list` 看注册配置，`inspect` 看同步后的知识状态。
- **搜索不等于诊断**：`search` 返回经过最终 document-aware 选择的用户结果；`diagnose` 保留 raw backend ranking，更适合调试相关度。
- **搜索不等于问答**：`search` 不调用 LLM Query Expander 或 Answer Generator；`query` 才会进入模型问答链路。

### Search 的检索语义

`search` 的核心链路是：

```text
configured retrieval backend
(BM25 / Semantic / Hybrid / reranker)
        ↓
bounded candidate retrieval
        ↓
document-aware diversification
        ↓
final Top-K
```

`--limit` 始终表示最终结果数上限。Document-aware selection 不会改写 backend 原始 `score`。`diagnose` 不执行这一最终选择，保持 raw backend ranking。

本地来源直接读取严格 UTF-8 编码的 `.txt`、`.md`、`.markdown`；`.doc`、`.docx`、`.pdf`、`.ppt`、`.pptx`、`.rtf`、`.epub`、`.odt` 由 AnyDoc 在本地转换为 canonical Markdown。相对路径均以运行命令时的当前目录为基准。

来源内部 ID 由来源类型和规范化路径稳定派生，用户通过路径注册和删除来源。同一类型和路径在删除后重新注册会得到同一个内部 ID，因此可以继续已有文档版本链。

同一个规范化来源路径在一个 KnowledgeBase 中只能注册一次。CLI、桌面界面和 Python API 都会执行这项检查。

删除来源及其当前 canonical documents：

```powershell
nexusmind source remove ./security-notes --knowledge-base ./security-kb
```

除 `create` 外，如果省略 `--knowledge-base`，CLI 会把当前目录当作 KnowledgeBase。读取类命令支持 `--json` 输出以便脚本处理。

## 生成带引用的回答

`nexusmind query` 用于向已有 KnowledgeBase 提问。模型配置可以写入 `.env`，也可以在 PowerShell 中设置：

```powershell
$env:NEXUSMIND_MODEL_BASE_URL = "https://api.openai.com/v1"
$env:NEXUSMIND_MODEL_API_KEY = "your-api-key"
$env:NEXUSMIND_MODEL_NAME = "gpt-4.1-mini"
$env:NEXUSMIND_MODEL_TIMEOUT = "60"
```

发起查询：

```powershell
nexusmind query "Binder caller UID 是如何获取的？" --knowledge-base ./security-kb
```

CLI `query` 当前执行以下链路：

```text
Original Question
        ↓
LLM Query Expansion
        ↓
Q0 original + 最多 3 条 expanded queries
        ↓
每条 query 独立走 configured retrieval backend
        ↓
query-level RRF
        ↓
document-aware diversification
        ↓
context assembly
        ↓
Answer LLM
        ↓
validated citations
```

Query Expansion 有以下约束：

- 原始问题始终作为 `Q0` 保留，不会被 rewritten query 替换；
- 最多生成 3 条 expanded queries；
- API、函数名、进程名、文件名、缩写、数字/十六进制错误码以及常见 CamelCase/PascalCase 技术标识符必须原样保留；
- expansion 超时、网络错误、无效 JSON、标识符校验失败等情况会 fail open，自动退回原始问题检索；
- query-level RRF 只用于多查询融合，不改写 retrieval backend 自己的 `SearchHit.score`。

查看检索和上下文调试信息：

```powershell
nexusmind query "Binder caller UID 是如何获取的？" `
  --knowledge-base ./security-kb `
  --debug
```

`--debug` 会额外显示 retrieval backend、实际 retrieval queries、Query Expansion fallback 状态、context 大小和 trace ID。JSON debug 输出还包含 fused result 的 query-index/rank provenance。

获取适合程序处理的 JSON：

```powershell
nexusmind query "Binder caller UID 是如何获取的？" `
  --knowledge-base ./security-kb `
  --json
```

## Python 接入

NexusMind 提供公开的 `KnowledgeBase` Python API。它适合将本地知识检索嵌入其他服务、脚本或桌面应用。

### 创建与同步

下例中的 `security-notes` 目录必须存在，并至少包含一个 UTF-8 编码的 `.txt`、`.md` 或 `.markdown` 文件：

```python
from nexusmind import KnowledgeBase, LocalDirectorySourceConfig

kb = KnowledgeBase.create("./security-kb")

registered = kb.add_source(
    LocalDirectorySourceConfig(path="./security-notes")
)
kb.sync_source(registered.source_id)
kb.close()
```

`add_source()` 只保存来源注册，不会立即读取文件，并返回包含最终规范化路径与内部 ID 的注册配置。只有调用 `sync()` 或 `sync_source()` 时，KnowledgeBase 才会读取并提交最新内容。

### 搜索

```python
from nexusmind import KnowledgeBase

kb = KnowledgeBase.open("./security-kb")

results = kb.search("密钥轮换", limit=5)
for result in results:
    print(result.document.logical_path)
    print(result.hit.score)
    print(result.hit.chunk.content)

kb.close()
```

默认检索不需要 API Key，也不会访问网络。它使用确定性的 Unicode/CJK BM25，并保留每条结果的 canonical source 和 document provenance。

### 生成带引用的回答

Python API 中，`answer_generator` 和 `query_expander` 是两个独立的可选依赖。CLI `query` 默认会同时创建 OpenAI-compatible Answer Provider 和 Query Expander；Python API 如果希望获得相同的 query-expansion 行为，需要显式注入两者：

```python
from nexusmind import (
    KnowledgeBase,
    OpenAICompatibleAnswerProvider,
    OpenAICompatibleQueryExpander,
)
from nexusmind.config import load_model_config_from_env

config = load_model_config_from_env()
answer_provider = OpenAICompatibleAnswerProvider(config)
query_expander = OpenAICompatibleQueryExpander(config)

kb = KnowledgeBase.open(
    "./security-kb",
    answer_generator=answer_provider,
    query_expander=query_expander,
)

result = kb.query("Binder caller UID 是如何获取的？")

print(result.answer.text)
for citation in result.citations:
    print(citation)

kb.close()
```

如果只注入 `answer_generator` 而不注入 `query_expander`，`query()` 会直接使用原始问题完成一次检索，然后进入现有 context / answer / citation 流程。

生成器返回的 citation handle 不会被直接信任。NexusMind 只允许引用本次实际提供给模型的 passages；未知、重复、格式错误或未提供给模型的 handle 会被拒绝。

完整流程见 [技术架构：Knowledge 查询流程](docs/architecture.md#knowledge-查询流程)。

### 检查知识库

```python
kb = KnowledgeBase.open("./security-kb")

inspection = kb.inspect()
print(inspection.status)
print(inspection.sources)
print(inspection.documents)

document = kb.inspect_document(
    inspection.documents[0].document_id,
    preview_chars=120,
)
print(document.chunks)

kb.close()
```

- `status()`：查看知识库 ID 和数量统计
- `list_sources()`：列出已注册来源
- `list_documents()`：列出当前 canonical documents
- `inspect()`：获取来源和文档的 coherent 只读视图
- `inspect_document()`：检查文档及其派生 Chunk

这些接口不会扫描来源目录的 dirty state，也不会自动同步内容。

### 诊断检索

```python
kb = KnowledgeBase.open("./security-kb")
diagnostics = kb.diagnose_search("密钥轮换", limit=5)

for candidate in diagnostics.candidates:
    row = candidate.diagnostic
    print(row.stage, row.rank, row.score, row.selected)

kb.close()
```

诊断结果可以包含 lexical、semantic、fusion 和 reranker 阶段的排名、分数、命中词与 RRF contribution。该能力要求当前 retrieval backend 实现 `DiagnosticChunkIndex`。

Query Expansion 不会改变 `search()` 和 `diagnose_search()` 的语义；它只属于 `KnowledgeBase.query()` 的可选 query-time pipeline。

## KnowledgeBase 如何保存数据

每个知识库目录包含：

```text
security-kb/
├── manifest.json          # KnowledgeBase 配置和来源注册
├── knowledge.db           # canonical 来源、当前文档与内部版本历史
└── .knowledge-base.lock   # 跨 handle/process 协调
```

同步会比较文档内容哈希：首次出现、内容变化，或删除后重新出现时，KnowledgeBase 会追加一条不可变的内部版本记录；内容未变化时不会创建新版本。

文档或来源从当前快照移除后，已有版本仍保留在 SQLite 中，用于后续维护和审计基础。

搜索、问答、检查和索引始终只使用当前有效文档。历史版本不会被分块或进入检索结果，本版本也不提供列出或读取历史的公开 API。

Chunk、embedding 和检索索引不会持久化，重新打开时会根据当前 canonical documents 重建。不要手动删除、替换或链接 `.knowledge-base.lock`。

当前使用单一、严格的持久化契约：`manifest.json` 与完整 SQLite schema 均为 version 1。未知版本、不完整 schema 或不一致历史都会 fail closed，不执行自动迁移或修复。

更完整的同步原子性、并发协调与恢复策略见 [技术架构：KnowledgeBase 存储](docs/architecture.md#knowledgebase-存储)。

## 当前限制

- 本地来源支持严格 UTF-8 文本，以及 `.doc`、`.docx`、text-layer `.pdf`、`.ppt`、`.pptx`、`.rtf`、`.epub`、`.odt`
- 来源不会被后台监控或自动同步
- 默认 `search` 是词法 BM25，不自动改写查询
- CLI `query` 在配置模型后会执行有界的 LLM Query Expansion，再进行多查询检索与 RRF 融合
- Semantic embedding、Query Expansion 和模型回答可能产生网络延迟与服务费用
- KnowledgeBase 当前没有持久化 embedding、Chunk 或检索索引
- 扫描版 PDF OCR、XLS/XLSX/CSV、网页和 GitHub ingestion 暂未提供
- Linux 和 macOS 当前没有纳入正式支持矩阵

## 安全、隐私与模型调用

- 默认 BM25 `search` 完全在本地执行
- 使用 Semantic provider 时，文档内容或查询可能发送给 embedding 服务
- CLI `query` 默认会先把**原始问题**发送给 Query Expansion 模型；Query Expansion 阶段不会发送知识库文档内容
- Answer 阶段会把原始问题和本次选中的 evidence passages 发送给模型服务商
- 因此一次 CLI `query` 通常至少包含一次 Query Expansion 请求和一次 Answer 请求，可能产生额外网络延迟和模型费用
- Query Expansion 失败会回退到原始问题检索，不会阻止后续问答
- 来源读取有文件大小、数量、总字节和目录深度限制
- 本地 adapter 拒绝或跳过符号链接、junction 和 Windows reparse point
- 引用只能由本次模型上下文中的 provenance allowlist 构造
- Manifest、SQLite 和锁文件身份异常时，KnowledgeBase 会 fail closed

## Windows 可移植 CLI

Windows portable 使用 PyInstaller `onedir` 模式，将 `nexusmind.exe`、Python runtime 和运行依赖放在同一个应用目录中。

维护者可以在 Windows PowerShell 中构建 portable ZIP：

```powershell
python -m pip install -e ".[dev,packaging]"
.\scripts\build-portable.ps1
```

构建后可以运行：

```powershell
.\dist\nexusmind\nexusmind.exe --help
```

构建脚本会从应用目录外执行 smoke test，并创建：

```text
dist\nexusmind-windows-portable.zip
```

CI 也会在 `windows-latest` 上执行真实 portable 构建，并把归档重新解压到仓库外；其中的 `nexusmind.exe` 会同步一个包含 Markdown、DOCX 和 text-layer PDF 的混合目录，验证 marker 检索与原始文件 provenance，无需另装 Python 或 AnyDoc runtime。

Portable runtime 使用独立的 `.nexusmind` 可写运行目录，用于保存配置、数据、日志和模型缓存。可以通过 `NEXUSMIND_RUNTIME_DIR` 指定其他绝对路径。

`logs\nexusmind.log` 使用有界轮转的单行 JSON 日志，记录启动、退出、同步、搜索、问答和失败诊断。日志不会记录 API Key、完整问题、文档正文或完整回答。

当前仓库没有公开 GitHub Release，因此 README 不提供不存在的版本化 wheel / sdist / portable 下载地址。正式重新发布时，再由 Release workflow 生成并附加对应产物。

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m pip check
python -m compileall -q src
python -m pytest -q
```

CI 在 `windows-latest` 上运行离线测试，不依赖真实 API Key、模型服务或开发者本机目录。

## 项目文档与评测

- [KnowledgeBase 技术架构](docs/architecture.md#knowledge-runtime)
- [检索 Benchmark 说明与结果](evals/knowledge/benchmark.md)
- [Document-aware diversification 评测](evals/knowledge/diversification.md)
- [Query Expansion deterministic evaluation](evals/knowledge/query_expansion/)

NexusMind 当前定位是本地 Knowledge Runtime。早期 Agent Runtime、Tool、Workspace、MCP 和 Agent Skill 实现已移除；当前阶段优先通过真实知识库 dogfooding 发现检索、问答、诊断和可用性问题。

## License

[MIT](LICENSE)
