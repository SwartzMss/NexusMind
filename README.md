# NexusMind

NexusMind 是一个面向本地知识的 KnowledgeBase 工具。它可以注册本地文本来源、显式同步内容、执行离线检索，并通过模型生成带可验证引用的回答。

你可以通过本地桌面界面管理知识库，通过 CLI 发起知识问答，也可以将 KnowledgeBase 作为 Python 库集成到自己的应用中。

> 当前支持 Windows 与 Python 3.11、3.12、3.13。Linux 和 macOS 暂不属于支持平台。

## 核心能力

- 创建和重新打开本地持久化 KnowledgeBase
- 注册本地文件或目录，并显式控制同步时机
- 默认使用完全离线、支持 Unicode/CJK 的 BM25 检索
- 可选使用 semantic、Hybrid-RRF 和 reranker 检索后端
- 根据检索内容生成回答，并验证回答中的引用
- 检查来源、文档、分块和各检索阶段的诊断信息
- 提供本地桌面界面、查询 CLI 和 Python API

底层数据流、存储格式、检索实现和安全边界请参阅[技术架构](docs/architecture.md)。

## 快速开始

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
nexusmind create ./security-kb --name "Security Notes"
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
| `create` | 在 `./security-kb` 创建一个 KnowledgeBase，内部 ID 由程序自动生成。`--name "Security Notes"` 是展示名称。目标目录必须不存在或为空。 | 是 |
| `source add` | 注册 `./security-notes` 来源。程序会根据路径自动判断它是文件还是目录，并生成内部来源 ID。注册只保存配置，不会读取或索引文件。 | 是，仅注册来源 |
| `source list` | 列出已经注册的来源类型和路径。它适合确认 `source add` 是否成功，但不能说明内容是否已经同步。 | 否 |
| `sync` | 读取所有已注册来源，把当前文件内容提交为 canonical documents，并重建用于检索的分块和索引。以后新增、修改或删除来源文件后，需要再次运行此命令。使用 `--source ./security-notes` 可以只同步一个来源。 | 是 |
| `search` | 在已同步内容中搜索“密钥轮换”，返回相关度最高的最多 5 个分块，显示文档路径、分数和原文。它只做检索，不调用回答模型。 | 否 |
| `inspect` | 查看 KnowledgeBase 的整体状态，包括来源数和 canonical document 数。使用 `--document <document-id>` 可进一步查看某篇文档的分块、字符范围和预览。 | 否 |
| `diagnose` | 对同一个搜索词输出检索管线的候选项，包含阶段、排名、分数和文档路径，用来分析为什么某篇文档命中、排序不理想或没有进入最终结果。根据所配置的后端，阶段可能包括 `lexical`、`semantic`、`fusion` 和 `reranker`。 | 否 |

需要特别区分以下三组概念：

- **注册不等于同步**：`source add` 只告诉 KnowledgeBase“去哪里找资料”；`sync` 才真正读取文件。首次注册后不执行 `sync`，`search` 不会搜到这些文件。
- **来源不等于文档**：一个 directory 来源可以产生多篇 canonical documents；`source list` 看注册配置，`inspect` 看同步后的知识状态。
- **搜索不等于诊断**：`search` 给日常使用的最终结果；`diagnose` 暴露检索各阶段的候选项，更适合调试相关度和后端配置。

本地来源当前只读取严格 UTF-8 编码的 `.txt`、`.md` 和 `.markdown` 文件。上述相对路径均以运行命令时的当前目录为基准。KnowledgeBase 的内部 ID 由程序自动生成；新来源的 ID 根据来源类型和规范化路径稳定派生。删除来源时，KnowledgeBase 会在 manifest 中保留其内部身份映射；以后重新注册相同类型和路径时会恢复原身份并延续文档版本历史。这也适用于由旧版 CLI 显式指定 ID 的 legacy 来源。用户通过路径操作它们。除 `create` 外，如果省略 `--knowledge-base`，CLI 会把当前目录当作 KnowledgeBase；读取类命令可添加 `--json` 以便脚本处理。

同一个规范化来源路径在一个 KnowledgeBase 中只能注册一次。CLI、桌面界面和 Python API 都会执行这项检查，避免同一份内容以不同内部 ID 被重复同步。

如果旧版 KnowledgeBase 已经包含相同路径的多个来源，按路径操作会报告歧义。此时可通过 `source list --json` 查看旧的内部 ID，并使用恢复选项删除其中一个重复注册：

```powershell
nexusmind source remove --id docs-a --knowledge-base ./security-kb
```

`--id` 仅用于修复旧数据；新来源仍应始终通过路径操作。

删除来源及其 canonical documents：

```powershell
nexusmind source remove ./security-notes --knowledge-base ./security-kb
```

所有读取类命令均支持适合脚本处理的 `--json` 输出。

### 生成带引用的回答

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

查看检索和上下文调试信息：

```powershell
nexusmind query "Binder caller UID 是如何获取的？" `
  --knowledge-base ./security-kb `
  --debug
```

获取适合程序处理的 JSON：

```powershell
nexusmind query "Binder caller UID 是如何获取的？" `
  --knowledge-base ./security-kb `
  --json
```

如果省略 `--knowledge-base`，CLI 默认将当前目录作为 KnowledgeBase。

## Python 接入

NexusMind 提供公开的 KnowledgeBase Python API。它适合将本地知识检索嵌入其他服务、脚本或桌面应用。

### 创建与同步

下例中的 `security-notes` 目录必须存在，并至少包含一个 UTF-8 编码的 `.txt`、`.md` 或 `.markdown` 文件：

```python
from nexusmind import KnowledgeBase, LocalDirectorySourceConfig

kb = KnowledgeBase.create(
    "./security-kb",
    display_name="Security Notes",
)

registered = kb.add_source(
    LocalDirectorySourceConfig(
        path="./security-notes",
    )
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

创建或打开 KnowledgeBase 时注入 `answer_generator`：

```python
from nexusmind import KnowledgeBase

kb = KnowledgeBase.open(
    "./security-kb",
    answer_generator=generator,
)

result = kb.query("Binder caller UID 是如何获取的？")

print(result.answer.text)
for citation in result.citations:
    print(citation)

kb.close()
```

生成器返回的 citation handle 不会被直接信任。NexusMind 只允许引用本次实际提供给模型的 passages；未知、重复、格式错误或未提供给模型的 handle 会被拒绝。

完整流程见[技术架构：Knowledge 查询流程](docs/architecture.md#knowledge-查询流程)。

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

- `status()`：查看知识库 ID、名称和数量统计
- `list_sources()`：列出已注册来源
- `list_documents()`：列出 canonical documents
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

诊断结果可以包含 lexical、semantic、fusion 和 reranker 阶段的排名、分数、命中词与 RRF contribution。该能力要求当前检索 backend 实现 `DiagnosticChunkIndex`。

## KnowledgeBase 如何保存数据

每个知识库目录包含：

```text
security-kb/
├── manifest.json          # KnowledgeBase 配置和来源注册
├── knowledge.db           # canonical 来源、当前文档与内部版本历史
└── .knowledge-base.lock   # 跨 handle/process 协调
```

同步会比较文档内容哈希：首次出现、内容变化，或删除后重新出现时，
KnowledgeBase 会追加一条不可变的内部版本记录；内容未变化时不会创建新版本。
文档或来源从当前快照移除后，已有版本仍保留在 SQLite 中，用于后续维护和审计基础。

搜索、问答、检查和索引始终只使用当前有效文档。历史版本不会被分块或进入检索结果，
本版本也不提供列出或读取历史的公开 API。Git 历史分析、后台自动同步和 UI 时间线不在此功能范围内。

Chunk、embedding 和检索索引不会持久化，重新打开时会根据当前 canonical documents 重建。不要手动删除、替换或链接 `.knowledge-base.lock`。

更完整的同步原子性、并发协调与恢复策略见[技术架构：KnowledgeBase 存储](docs/architecture.md#knowledgebase-存储)。

## 当前限制

- 本地来源仅支持严格 UTF-8 的 `.txt`、`.md` 和 `.markdown`
- 来源不会被后台监控或自动同步
- 默认检索是词法 BM25，不包含同义词、自动改写或语义理解
- semantic embedding 和模型问答可能产生网络延迟与服务费用
- KnowledgeBase 当前没有持久化 embedding 或索引
- PDF、Office、网页和 GitHub ingestion 暂未提供

## 安全与隐私

- 默认 BM25 搜索完全在本地执行
- 使用 semantic provider 时，文档或查询可能发送给 embedding 服务
- 使用 `query()` 或 CLI 问答时，问题和选中的知识片段会发送给模型服务商
- 来源读取有文件大小、数量、总字节和目录深度限制
- 本地 adapter 拒绝或跳过符号链接、junction 和 Windows reparse point
- 引用只能由本次模型上下文中的 provenance allowlist 构造
- Manifest、SQLite 和锁文件身份异常时，KnowledgeBase 会 fail closed

## Windows 可移植 CLI

Windows portable artifact 使用 PyInstaller `onedir` 模式，将 `nexusmind.exe`、
Python runtime 和运行依赖放在同一个应用目录中。解压
`nexusmind-windows-portable.zip` 后可直接运行：

```powershell
.\nexusmind\nexusmind.exe --help
```

无需另行安装 Python。ZIP 只包含程序文件，不预置用户数据、配置、日志或模型；
GUI、安装器、自动更新和模型分发不属于该 artifact。

首次启动会在 `nexusmind.exe` 同级创建 `.nexusmind` 可写运行目录：

```text
nexusmind\
├── nexusmind.exe
├── _internal\
└── .nexusmind\
    ├── config\
    ├── data\
    ├── logs\
    │   └── nexusmind.log
    └── models\
```

该位置由 exe 的绝对路径确定，不受启动时的当前工作目录影响。exe 所在目录必须可写；
如果目录只读，可将整个 portable 目录移动到可写位置，或通过
`NEXUSMIND_RUNTIME_DIR` 指定其他绝对路径。相对覆盖路径仍会被拒绝。现有
KnowledgeBase 路径参数由用户显式指定；建议将需要与 portable runtime 一起管理的
本地数据库放在 `.nexusmind\data\` 下。

旧版本默认使用的用户目录数据不会自动迁移到 exe 同级。需要保留旧数据时，请手动
复制原 `.nexusmind` 内容，或将 `NEXUSMIND_RUNTIME_DIR` 暂时指向旧目录。升级时应
保留或备份 exe 同级的 `.nexusmind`；删除整个解压目录也会删除其中的日志、配置、
模型及其他本地数据。

`logs\nexusmind.log` 使用有界轮转的单行 JSON 日志，记录启动、退出、同步、搜索、
问答和失败诊断。日志只保留操作名、计数、耗时、错误类型等诊断字段，不写入 API
Key、完整问题、文档内容、回答或工具输出。未预期异常会在终端显示简短提示和日志
位置，并以非零状态退出。

维护者可在 Windows PowerShell 中构建同一 portable ZIP：

```powershell
python -m pip install -e ".[dev,packaging]"
.\scripts\build-portable.ps1
```

脚本会构建 `dist\nexusmind\nexusmind.exe`、从应用目录外执行 `--help` smoke test，
确认 exe 同级的 `.nexusmind\logs\nexusmind.log` 已生成，然后创建
`dist\nexusmind-windows-portable.zip`。CI 也会在 `windows-latest` 上执行该真实
打包和 smoke test；Actions Artifact 直接包含 portable 目录内容，下载后只需解压
一次。

## 开发验证

```powershell
python -m pip install -e ".[dev]"
python -m pip check
python -m compileall -q src
python -m pytest -q
```

CI 在 `windows-latest` 上运行离线测试，不依赖真实 API Key、模型服务或开发者本机目录。

## 项目文档

- [KnowledgeBase 技术架构](docs/architecture.md#knowledge-runtime)
- [检索 Benchmark 说明与结果](evals/knowledge/benchmark.md)

当前版本已移除早期 Agent Runtime、Tool、Workspace、MCP 和 Agent Skill 实现。未来可以在不引入 Agent Tool Loop 的前提下增加 Knowledge-native Skill 和 Knowledge persistence。

## License

[MIT](LICENSE)
