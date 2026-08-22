# NexusMind

NexusMind 是一个与模型服务商解耦、带明确边界和执行上限的 Agent Runtime / Harness。它把流式模型、工具调用、Workspace、MCP、Skills 以及可选的持久化能力组合成可审计、可停止的单 Agent 执行。

当前版本支持 Windows，以及 Python 3.11、3.12 和 3.13。Linux 和 macOS 不属于支持的平台。

## 当前能力与架构

NexusMind 的主要能力层如下：

- Provider-neutral 消息和运行时事件契约，以及异步流式 `ChatModel` 抽象。
- OpenAI-compatible 流式模型适配器和 Tool Call 组装。
- 有界的单 Agent Harness / Tool Loop，包含模型轮次、工具调用、参数和结果大小限制，以及明确的 stop semantics。
- Provider-neutral Tool Registry / Executor：JSON Schema 校验、超时、结构化错误、风险级别、策略判断和 CLI 审批检查点。
- Workspace 读工具（`list_files`、`read_file`、`search_text`）和受 SHA-256 乐观并发保护的写工具（`write_file`、`replace_text`）。
- Windows-only 的 Host-approved `run_command` Profile，使用固定 argv/cwd/timeout 和 Windows Job Object 回收进程树。
- MCP stdio 工具发现、调用及其在 Agent Loop 中的集成。
- 声明式 Skills、工具白名单、收紧后的执行限制和多 MCP Server 解析。
- 可选的 SQLite Run History、Harness Checkpoints、受支持状态的 Resume，以及 Run Lease ownership controls。

### KnowledgeBase 产品 API

`KnowledgeBase` 是面向库调用者的持久化知识库 API，不是 Agent Runtime 的 `Workspace`。它将本地来源注册、显式同步、canonical 状态、搜索和重启恢复收敛到一个对象中。运行下例前，`./security-notes` 必须已存在，并至少包含一个 UTF-8 编码的 `.txt`、`.md` 或 `.markdown` 文件：

```python
from nexusmind import KnowledgeBase, LocalDirectorySourceConfig

kb = KnowledgeBase.create(
    "./security-kb",
    knowledge_base_id="security",
    display_name="Security Notes",
)
kb.add_source(
    LocalDirectorySourceConfig(source_id="docs", path="./security-notes")
)
sync_results = kb.sync()
results = kb.search("密钥轮换", limit=5)
kb.close()

kb = KnowledgeBase.open("./security-kb")
results_after_restart = kb.search("密钥轮换", limit=5)
# Keep this reopened handle for the inspection and diagnostics examples below.
```

`LocalDirectorySourceConfig` 是已注册的产品配置：它说明下次去哪里加载。Canonical `KnowledgeSource` 则是上次成功同步后已提交内容的 provenance，两者不是同一份状态。`add_source()` 只原子持久化注册；它不构造 adapter、不读取文件，也不调用 embedding 或 reranking provider。`sync()` 和 `sync_source()` 都是调用者显式发起的同步操作；`sync()` 按 `source_id` 排序处理全部注册，整批成功才提交，任一来源失败都不会留下部分新状态。

`unregister_source()` 只删除从未同步、因而没有 canonical `KnowledgeSource` 的注册。`remove_source()` 同时删除注册和该来源的 canonical documents：它先提交 SQLite，再原子替换 manifest；后一步失败时会补偿恢复旧 snapshot，补偿也失败则将当前对象 poison/close，后续调用 fail closed。

每个知识库目录有两个产品状态文件和一个内部协调文件：

```text
security-kb/
├── manifest.json          # 仅产品配置与已注册来源
├── knowledge.db           # 仅 canonical KnowledgeSource / Document
└── .knowledge-base.lock   # 仅跨 handle/process 协调
```

Chunk、embedding、lexical/semantic/hybrid index 和 reranker 状态都不持久化，重启时从 canonical documents 重建。默认检索是完全离线、确定性的 Unicode/CJK BM25；高级调用者可在 `create()` 和每次 `open()` 时注入未持久化的 `index_factory`。Benchmark fixture reranker 不是默认值，默认路径不需要 API key。

`create()` 只接受不存在或已存在但为空的真实目录，并以 no-clobber 方式创建完整布局；manifest 更新使用同目录临时文件和原子替换。`open()` 严格要求三个 artifact 存在、类型和身份有效，不会将缺失状态补成空库。同一实例的 mutation 由进程内锁串行化；不同 handle/process 通过 `.knowledge-base.lock` 的 no-wait OS advisory lock 协调。进程崩溃会由 OS 释放锁，文件中的旧内容无害。协作进程不得删除、替换或链接该文件；缺失、symlink/reparse point 或 identity 替换都会 fail closed，运行时也不会 unlink 它。

`status()` 只返回 ID、display name、注册数、canonical source 数和 document 数，不扫描文件系统 dirty state，也不虚构 last-sync 状态。`list_sources()` 按 `source_id` 返回 frozen config；`list_documents()` 返回与内部状态脱离的 canonical documents；`search()` 保留 retrieval backend 顺序和 canonical provenance。当前没有 KnowledgeBase CLI、watcher、后台同步或自动持久化 index。

### 一次性知识问答与引用

`KnowledgeBase.answer()` 提供一次性 question → retrieval → `ContextPackage` → `AnswerGenerator` → `KnowledgeAnswer` 组合。`AnswerGenerator` 是知识域 capability seam：它不执行检索，也不进入 Agent Tool Loop；可以在 `KnowledgeBase.create()` / `open()` 时作为纯运行时配置注入，或在单次 `answer(..., generator=...)` 调用中覆盖。生成器只返回未信任的答案文本和 `K1`、`K2` 等 citation handles，不能直接创建可信 provenance。

NexusMind 按 `ContextPackage.passages` 的稳定顺序分配 handles，确定性渲染 model-visible knowledge context，并在 `ModelContextRecord` 中记录 question、passage identities、canonical document content hashes、offsets、原文、limits 和 generator config identity。最终 `KnowledgeCitation` 只能由这份 allowlist 构造；未知、重复、格式错误或未提供给模型的 handle 都会 fail closed。`KnowledgeAnswer` 同时返回答案、验证后的 citations 和可检查/重放的 model-context record。

`ContextPassage` 是紧凑的 immutable provenance reference：只保留 source/document/chunk IDs、logical path、canonical document hash、原 chunk 与选中区间、选中文本、score 和 matched terms。Assembly 在构造它之前仍以 canonical `Document` 验证 chunk 内容和 offsets，但不会把完整 `KnowledgeSource`、`Document` 或 `Chunk` 对象复制进每个 passage；因此多个 passage 来自同一大文档时，内存只随实际选中的 bounded content 增长。

`AnswerGenerationLimits` 显式限制 question、rendered context、passages、answer 和 citations 的字符/token/数量。retrieval、context assembly、provider 或 citation validation 任一失败都不会返回部分可信答案；provider 异常会映射为不含密钥、私有文档或原始请求/响应的受控错误。生成答案和 replay record 只是 derived runtime output，不写入 `KnowledgeSnapshot`、manifest 或 SQLite，也不产生聊天历史、工具循环、自动规划或隐藏搜索。现有 provider-neutral `ChatModel` 可由具体 `AnswerGenerator` adapter 复用，但知识问答 API 不依赖 coding-agent runtime。

### KnowledgeBase 检查与检索诊断

`inspect()`、`inspect_document()` 和 `diagnose_search()` 是面向 Python 调用者的只读结构化 API；它们不提供 CLI 或 UI 的展示格式。`inspect()` 返回一个 coherent 的当前 manifest/canonical 状态视图。每个已注册来源的 `sync_status` 为 `"synced"`，仅表示 canonical state 中存在该来源（即使该次成功同步得到零份 document）；否则为 `"registered"`。这不是文件系统 dirty scan，也不说明来源路径此刻是否发生变化。

```python
inspection = kb.inspect()
document_id = inspection.documents[0].document_id
inspection_view = {
    "status": {
        "knowledge_base_id": inspection.status.knowledge_base_id,
        "registered_source_count": inspection.status.registered_source_count,
        "canonical_source_count": inspection.status.canonical_source_count,
        "document_count": inspection.status.document_count,
    },
    "sources": [
        {
            "source_id": item.config.source_id,
            "sync_status": item.sync_status.value,
            "document_count": item.document_count,
            "chunk_count": item.chunk_count,
        }
        for item in inspection.sources
    ],
    "documents": [
        {
            "document_id": item.document_id,
            "source_id": item.source_id,
            "logical_path": item.logical_path,
            "content_hash": item.content_hash,
            "character_count": item.character_count,
            "chunk_count": item.chunk_count,
        }
        for item in inspection.documents
    ],
}
```

`inspect_document(document_id, *, preview_chars=160)` 返回 detached canonical `source`、`document` 和 derived chunk summaries。Chunk 不写入新的 cache：每次检查都用当前 collection 的 chunker 从 canonical Document 重新派生并校验。`preview` 最多为 `preview_chars` 个字符；`start_offset` / `end_offset` 是 canonical document 中的半开字符区间。

```python
document_inspection = kb.inspect_document(document_id, preview_chars=120)
document_view = {
    "source": {
        "source_id": document_inspection.source.source_id,
        "source_type": document_inspection.source.source_type,
        "logical_location": document_inspection.source.logical_location,
    },
    "document": {
        "document_id": document_inspection.document.document_id,
        "logical_path": document_inspection.document.logical_path,
        "content_hash": document_inspection.document.content_hash,
        "content_type": document_inspection.document.content_type,
    },
    "chunks": [
        {
            "ordinal": chunk.ordinal,
            "chunk_id": chunk.chunk_id,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "character_count": chunk.character_count,
            "preview": chunk.preview,
        }
        for chunk in document_inspection.chunks
    ],
}
```

`diagnose_search(query, *, limit=10)` 返回 `KnowledgeRetrievalDiagnostics`。最终 `results` 和所有中间 `candidates` 都附带经过校验的 canonical `source` / `document` provenance；`candidate.diagnostic` 是 backend trace row。其 `stage` 是 `lexical`、`semantic`、`fusion` 或 `reranker`，`rank` 是该阶段的一基排名，`score` 是该 backend 的原始 score，`matched_terms` 为 backend 提供的命中词。Hybrid 的 lexical / semantic rows 还以 `rrf_contribution` 精确给出 `1 / (rrf_k + rank)`；fusion score 是这些贡献的和。Reranker rows 的 `score` 是 reranker provider score，`selected` 表示该 candidate 是否留在最终结果。

```python
diagnostics = kb.diagnose_search("密钥轮换", limit=5)
diagnostics_view = {
    "query": diagnostics.query,
    "results": [
        {
            "source_id": result.source.source_id,
            "document_id": result.document.document_id,
            "logical_path": result.document.logical_path,
            "chunk_id": result.hit.chunk.chunk_id,
            "score": result.hit.score,
            "matched_terms": result.hit.matched_terms,
        }
        for result in diagnostics.results
    ],
    "candidates": [
        {
            "source_id": candidate.source.source_id,
            "document_id": candidate.document.document_id,
            "logical_path": candidate.document.logical_path,
            "stage": candidate.diagnostic.stage.value,
            "rank": candidate.diagnostic.rank,
            "chunk_id": candidate.diagnostic.chunk.chunk_id,
            "score": candidate.diagnostic.score,
            "matched_terms": candidate.diagnostic.matched_terms,
            "rrf_contribution": candidate.diagnostic.rrf_contribution,
            "selected": candidate.diagnostic.selected,
        }
        for candidate in diagnostics.candidates
    ],
}
kb.close()
```

`diagnose_search()` 每次请求只执行一次已配置的诊断 backend/provider 路径（例如 semantic query embedding 或 reranker 各一次），不会先运行普通 `search()` 再重新计算 trace。只有实现可选 `DiagnosticChunkIndex.diagnose()` 协议的 backend 支持该 API；实现 `add()`、`replace_document()`、`remove_document()`、`search()` 和 `clone()` 的自定义 `CloneableChunkIndex` 即使不提供 `diagnose()`，仍可用于搜索、同步、恢复和持久化，但调用诊断会以受控的 `KnowledgeBaseSourceError` 失败。

这三个 API 不读取来源路径、不保存 manifest/SQLite、不改变 index 或 canonical state，也不保留 query history。Inspection、chunk、diagnostic rows、scores 和 query 都不持久化，因此不引入 manifest、snapshot 或 SQLite schema 变更。现有本地 KnowledgeBase UI 在本 issue 中没有加入 diagnostics 展示或新的检查界面。

### KnowledgeBase 本地界面

安装项目后运行以下命令可打开首个本地 KnowledgeBase 桌面界面：

```powershell
nexusmind-kb
```

界面支持创建或打开一个明确的 KnowledgeBase、通过系统选择器注册本地文件或目录、查看来源和状态、显式同步全部或单个来源、删除来源，以及按有限结果数搜索并查看 `source_id`、逻辑路径、score、chunk ID 和原文片段。注册来源不会自动同步；所有 KnowledgeBase 操作统一在后台 worker 运行，其间相关控件会禁用。若此时请求关闭窗口，界面会等待当前操作完成，再关闭 handle 和窗口。

该界面使用 Python 标准库 `tkinter`，不需要本地服务器、外部服务或账号。窗口层只处理控件、选择器和后台调度；可 headless 测试的 controller 仅调用公开 `KnowledgeBase` API。它不读取 manifest/SQLite，不构造 ingestion adapter，也不依赖 `KnowledgeCollection` 或索引实现。错误只映射为有限的产品级提示，不回显底层异常文本、路径、查询或文档内容。

NexusMind 开始引入独立的 Knowledge Runtime / Knowledge Layer，与现有 Agent Runtime 解耦。`KnowledgeSource` 表示知识来源，`source_id` 由 Host 或 Source adapter 提供；`Document` 表示来源下由逻辑标识定位的一份文本内容。Document 的 `document_id` 由 `source_id + logical_path` 派生，并使用 UTF-8 SHA-256 `content_hash` 检测同一逻辑文档的内容变化。`Chunk` 是由 Document 派生的 source-neutral 原文切片，使用 Python 字符偏移记录半开区间 `[start_offset, end_offset)`，并保证 `chunk.content == document.content[start_offset:end_offset]`。本地文件/目录只是可以映射到 Knowledge Runtime 的一个来源示例，后续可由 Agent、Skill 或其他消费者使用。

本地知识 ingestion 位于独立的 Knowledge Ingestion 层。`LocalFileAdapter` 和 `LocalDirectoryAdapter` 负责受限的文件发现、严格 UTF-8 读取以及来源相对路径映射；文件扩展名、扫描上限和文件系统安全策略都属于 adapter 实现，不会进入 provider-neutral Knowledge Core：

```text
Local File / Directory
    -> KnowledgeSourceAdapter
    -> KnowledgeSource
    -> Document[]
```

第一版只支持 UTF-8 的 `.txt`、`.md` 和 `.markdown` 文件，并限制单文件大小、文档数量、总读取字节数、扫描条目数和目录深度。目录 adapter 收集后按完整的来源相对路径排序，跳过符号链接、junction 和其他 Windows reparse point，以及不支持的扩展名；单文件来源或根路径遇到这些类型会拒绝。Discovery 会记录每个文件的 identity；读取时从已打开的文件句柄读取，并依次复核 discovered identity、opened identity、当前路径 identity 与 containment，避免 scan 后路径被替换而读取其他目标。PDF、Office、Web、GitHub 和 MCP 属于后续 ingestion 能力；索引和检索位于独立的 Knowledge Retrieval 层。

Knowledge Chunking 位于 ingestion 之后的独立层。`TextChunker` 使用确定性的字符分块策略，默认 `chunk_size=1000`、`overlap=100`、`max_chunks=10000`；配置会被严格校验，空 Document 返回空 tuple，超过最大块数会在生成任何部分结果前失败。Chunk ID 由 Document ID、内容 hash、字符区间和影响边界的分块配置确定性派生，因此同一输入与配置保持稳定，文档内容变化时不会让同一个 ID 指向不同切片。

Knowledge Retrieval 在 chunking 之后提供 source-neutral 的 `ChunkIndex` / `SearchHit` 契约。首个 `InMemoryChunkIndex` 进行进程内 BM25 词法检索，Chunk 和 query 都通过同一个显式 `LexicalAnalyzer` 边界。默认的 `UnicodeCJKLexicalAnalyzer` 先做 NFKC 检索归一化（不修改 canonical Document 原文），再将标点、符号、空白、独立组合标记、控制字符和未分配字符作为边界；非 Han 的 Unicode 字母/数字连续段使用 `str.casefold()` 成为词项，Han 连续段产生重叠字符 bigram，只有单个 Han 字符的段则产生 singleton。Han 与非 Han 段之间也是边界。

为使 Python 3.11–3.13 上的可检索字符集一致，默认 analyzer 显式采用 Unicode 14.0 common repertoire policy：Unicode 15.0/15.1 才分配的字符在 NFKC 前就作为边界，Han 识别也固定为 Unicode 14.0 已分配的显式区间。对需要完全保留历史“Unicode 空白切分后 casefold”行为的调用方，可显式选择兼容 analyzer：

```python
from nexusmind import InMemoryChunkIndex, WhitespaceLexicalAnalyzer

index = InMemoryChunkIndex()
legacy_index = InMemoryChunkIndex(analyzer=WhitespaceLexicalAnalyzer())
```

`WhitespaceLexicalAnalyzer` 会保留标点附着于空白 token 的旧行为。两种 analyzer 都不做词典分词、stemming、stop-word 过滤、同义词或语义理解；Han bigram 可能过度匹配常见相邻字，singleton query 的特异性较弱，兼容归一化也会有意合并某些视觉形式。

Public custom `LexicalAnalyzer` 必须 deterministic、effectively immutable，且 `analyze()` 的结果不能依赖可变调用历史；它还必须能够安全地由多个 index clone 共享。`InMemoryChunkIndex.clone()` 会共享同一个 analyzer instance，只复制 chunks 与 BM25 统计等可变 index state，不会 `deepcopy()` analyzer 或调用 analyzer factory。内置 analyzer 均为 frozen、stateless 实现。

BM25 使用 `k1 = 1.2`、`b = 0.75` 和始终为正的 IDF：

```text
idf(term) = log(1 + (N - df(term) + 0.5) / (df(term) + 0.5))

score(q, D) = sum over distinct matched query terms:
    idf(term) * tf(term, D) * (k1 + 1)
    ---------------------------------------------------
    tf(term, D) + k1 * (1 - b + b * |D| / avgdl)
```

`SearchHit.score` 是有限非负 float，结果按 score 降序、`chunk_id` 升序稳定排序。Query 的 `max_query_chars` 先于 analysis 检查；`max_query_terms` 和 `max_query_analyzed_chars` 则应用于 analyzer 返回的原始词项流，在按首次出现顺序去重之前分别限制词项数和词项字符总数，因此重复输入和 analyzer 放大都不能绕过上限。候选 corpus 还受 `max_total_analyzed_tokens` 和 `max_total_analyzed_token_chars` 的全局上限约束；默认值分别是 10,000,000 和 200,000,000，后者为默认内容字符上限的 20 倍，为 NFKC/casefold 扩展保留有界余量；查询 analyzed 字符默认上限同样是输入字符上限的 20 倍，即 20,480。Custom analyzer 在返回前的内部分配不受索引控制；返回值经严格类型验证后，会在保留 corpus 统计或进入 query 评分前执行这些上限。Analyzer 选择是未持久化的 runtime configuration；analyzed tokens、TF、chunk frequency、token length 和平均 chunk length 是未持久化的 derived runtime state。Add/replace/remove 会在候选 corpus 上重建统计并原子交换，restore 则从 canonical Documents 重新分块，并使用当前 collection/index 配置的 analyzer 重建 derived state。Snapshot 和 SQLite 都不保存 analyzer、tokens 或 postings。

同一 `ChunkIndex` 契约现在也可由 `InMemorySemanticChunkIndex` 实现。语义路径通过同步、provider-neutral 的 `EmbeddingProvider` 明确区分批量 `embed_documents()` 与单条 `embed_query()`；`EmbeddingVector` 是经过校验的不可变 finite、non-zero float tuple。内置 `OpenAICompatibleEmbeddingProvider` 调用同步 `/v1/embeddings` HTTP 接口，支持注入 transport 以便离线测试。同步调用意味着 `sync()` / `restore()` 会承担文档 embedding 的网络延迟和费用，`search()` 会承担 query embedding 的延迟和费用。

```python
from nexusmind import (
    InMemorySemanticChunkIndex,
    KnowledgeCollection,
    OpenAICompatibleEmbeddingProvider,
)

provider = OpenAICompatibleEmbeddingProvider(
    base_url="https://api.example.com/v1",
    api_key="...",
    model="text-embedding-model",
)
collection = KnowledgeCollection(
    index_factory=lambda: InMemorySemanticChunkIndex(
        embedding_provider=provider,
    )
)
```

语义 index 在进程内对全部 chunks 做有界 brute-force cosine 排序，score 范围为 `[-1.0, 1.0]`，按 score 降序、`chunk_id` 升序稳定返回；负分和零分候选不会被静默丢弃，`matched_terms` 固定为空。Cosine 与 BM25 是 backend-specific 分数，不能直接跨 backend 比较。首个非空 commit 固定 vector dimension；显式移除到空状态后才允许建立新维度。Chunk 数、内容字符、batch、dimension、保留的 vector values、query 字符和结果数均有显式上限。Add/replace 在 provider 调用和完整候选校验成功后才交换状态，remove 不调用 provider。

Custom `EmbeddingProvider` 必须可安全地在 index clones 之间共享；`InMemorySemanticChunkIndex.clone()` 共享同一个 provider instance 和不可变 limits，只复制 chunks、vectors、ownership 与 accounting 等可变状态。Provider 配置属于未持久化的 runtime configuration，vectors 属于未持久化的 derived state。Snapshot / SQLite 不保存 provider 或 embedding；restore/restart 会使用当前 provider 重新 embedding，因此可能产生远程费用、延迟以及模型版本漂移。持久化 embedding/cache、ANN 和 RAG 不在当前范围内。

`RerankedChunkIndex` 是任意 cloneable `ChunkIndex` 外的有界第二阶段装饰器。它只向 base index 执行一次固定 `candidate_depth` 搜索，reranker 只能对该精确候选 tuple 重新评分和排序，不能获取额外候选、修改 canonical `Chunk` / `matched_terms` 或拥有持久化状态。`RerankerLimits` 在 provider 工作前限制 query 字符、候选数、候选总字符和结果数；所有值与 `candidate_depth` 都必须是正的 plain integer。

reranker 输出必须是精确 tuple，并逐项对应输入候选；ghost、duplicate、冲突 Chunk、非法 score 或越界结果都会受控 fail closed。最终 score 来自 reranker，同分按 first-stage rank、再按 `chunk_id` 确定性排序。合法 underrun 会原样返回较短结果，不从原排名补齐，也不会在 provider 失败时 fallback。wrapper 的 mutation 与 clone 使用独立 base clone 和 commit-after-success；reranker 作为确定性、无状态 runtime configuration 在 clones 间共享。Snapshot / SQLite 仍只保存 canonical Source 与 Document。

离线语义 fixture、固定配置、复现命令和描述性 Hit@3 / Recall@3 / MRR 位于 [`evals/knowledge/semantic/`](evals/knowledge/semantic/)。它通过真实 ingestion -> collection -> semantic index -> provenance -> evaluator 路径运行，使用固定概念向量验证接线，而不是评估外部 embedding 模型质量，也不是 release gate。

`retrieval_evaluation` 提供确定性的离线 Hit@K、Recall@K 和 MRR 评估，ground truth 使用 canonical Document `(source_id, logical_path)`，同时保留实际 chunk 排名和重复 document hits 作为诊断信息。首个原创 corpus、15 个显式 labels、固定配置、previous whitespace 与 current Unicode/CJK 指标及复现命令见 [`evals/knowledge/baseline.md`](evals/knowledge/baseline.md)；当前 baseline 显式配置 `UnicodeCJKLexicalAnalyzer`，不依赖未来可能变化的 index 默认值。该 baseline 通过真实 LocalDirectoryAdapter -> KnowledgeCollection -> BM25 -> provenance 路径运行，不是 release gate 或公共 benchmark。

CJK analyzer 对比评估的语料、labels、固定配置、复现命令和当前 Hit@3 / Recall@3 / MRR 位于 [`evals/knowledge/cjk/`](evals/knowledge/cjk/)。它使用 7 份原创文档（包含刻意加入的近邻文档与排名余量）和 10 个文档相关性 cases，通过真实 ingestion -> collection -> BM25 -> provenance -> evaluator 路径比较两种 analyzer。记录值只是 descriptive、non-gate 指标，测试检查确定性、范围、fixture coverage、定性改进和非饱和 MRR，不将精确指标锁定为 CI 质量门槛。

`KnowledgeCollection` 组合现有 adapter、chunker 和 index，提供显式的 `sync()` / `search()` 工作流。每次同步加载完整 source snapshot，以 `document_id` 和 `content_hash` 确定新增、更新、未变化及删除的 Documents；只有新增/变化的文档会重新分块，删除文档的旧 chunks 会从检索中移除。第一版 collection 依赖独立的 `CloneableChunkIndex` staging capability：所有变更先在克隆的候选 index 上按稳定顺序完成，成功后才交换 collection snapshot 和 index，因此失败不会留下部分状态。可选的 `index_factory=` 必须在每次调用时创建一个全新、空且由 collection 独占的 index，避免 searchable state 脱离 authoritative source/document snapshot；调用方应通过 collection 搜索已提交状态。基础 `ChunkIndex` 仍只定义 add/replace/remove/search，未来事务型持久化 backend 不必支持 clone。同步由调用方显式触发，不包含后台监听或定时刷新。

`ChunkIndex.search()` 返回 retrieval-layer `SearchHit`，只描述匹配 Chunk、分数和命中词。`KnowledgeCollection.search()` 会按 backend 原始顺序把每个 hit 的 `document_id` 解析到当前已提交的 canonical `Document` 和所属 `KnowledgeSource`，并验证 Chunk offsets 合法且内容等于 canonical Document 的对应字符切片，然后返回 `KnowledgeSearchResult(source, document, hit)`；Source 和 Document 是深拷贝，调用方修改嵌套 metadata 不会影响 collection 状态。无法解析的 ghost/malformed/stale hit 会以受控的 `KnowledgeSearchResolutionError` fail closed，不会伪造 provenance 或返回部分结果。

collection 和 index 仍只存在于当前进程；canonical source/document snapshot 可由显式 store 保存并在重启后加载，但 derived Chunk/Index（包括语义 vectors、hybrid fusion 与 reranking composition）会重新构建。当前没有向量数据库或持久化 retrieval index；一次性答案生成同样是非持久化 derived runtime state。

`KnowledgeCollection.snapshot()` 可按稳定 identity 顺序导出 frozen container 形式的 `KnowledgeSnapshot`，其中包含与 collection 内部状态脱离的 canonical `KnowledgeSource` / `Document` 副本；嵌套 metadata 仍是普通可变 mapping，因此它不是递归 deep-immutable 对象。`restore()` 把 snapshot 视为完整 authoritative replacement，先验证 source/document 图和 collection limits，再使用当前 chunker 与 `index_factory` 创建的全新空 index 重建所有 derived `Chunk` / retrieval state，全部成功后才原子交换 collection 状态。Snapshot 不包含 Chunk、Index 或 SearchHit；使用不同 chunker 或 retrieval backend 恢复同一 canonical snapshot 时，可以得到不同的 derived state。

```text
External Source -> sync -> KnowledgeCollection -> snapshot() -> KnowledgeSnapshot

KnowledgeSnapshot
    -> restore()
    -> Document
    -> Chunking
    -> Index
    -> SearchHit[]
    -> KnowledgeCollection provenance resolution
    -> KnowledgeSearchResult[]
```

`KnowledgeSnapshot` 本身是进程内导出/恢复契约，不是 JSON、文件或数据库 schema；只有通过 snapshot store 显式保存后才能跨进程保留。Embedding 配置和 vectors 不属于 snapshot；持久化 semantic index 和 RAG 仍属于未来工作。

`KnowledgeSnapshotStore` 是 snapshot 与持久化 backend 之间的 source-neutral 边界。首个 `SQLiteKnowledgeSnapshotStore` 使用显式 `save()` / `load()` 保存一个完整 authoritative snapshot；save 在 SQLite transaction 中全量替换旧 Source/Document rows，失败会回滚到先前 snapshot。V1 schema 只保存 canonical `KnowledgeSource` / `Document` 字段、严格 JSON-compatible metadata 和最小 schema version，不保存 Chunk、Index、SearchHit、FTS postings 或 embedding。加载后仍必须调用 `KnowledgeCollection.restore()`，使用当前 chunker 和 fresh index 重建 derived retrieval state。

```text
External Source
    -> KnowledgeCollection.sync()
    -> KnowledgeCollection.snapshot()
    -> KnowledgeSnapshotStore.save()
    -> SQLite

restart

SQLite
    -> KnowledgeSnapshotStore.load()
    -> KnowledgeCollection.restore()
    -> rebuild Chunk / Index
    -> search
```

保存由调用方显式触发；当前没有 autosave、watcher、增量持久化或多进程 writer orchestration。SQLite 是第一个可替换 backend，不属于核心 Knowledge model；FTS、持久化 embedding、vector database 和 RAG 仍是未来工作。

```text
External Source
    -> KnowledgeSourceAdapter
    -> KnowledgeCollection.sync()
       -> Document snapshot
       -> Chunking
       -> Chunk Index
    -> KnowledgeCollection.search()
    -> KnowledgeSearchResult[]
```

```text
Knowledge Ingestion -> KnowledgeSource -> Document
Knowledge Chunking                           -> Chunk
Knowledge Index / Retrieval                  -> Lexical / Semantic / Hybrid -> Rerank -> SearchHit[]
Knowledge Collection                                         -> KnowledgeSearchResult[]
future                                       -> Persistent Chunk / Index
                                             -> Persistent / ANN Retrieval
                                             -> RAG
```

第二阶段 reranking 仍停留在 retrieval backend 内，不改变 collection provenance 层。一次性知识问答在 context assembly 之后分配 citation handles 并验证 provenance；当前仍不包含 token-based / semantic chunking、持久化 Index、query rewriting、多轮对话或 Agent 式 RAG 编排。

执行关系可以概括为：

```text
Model Provider
    -> ChatRuntime                 # CLI-facing chat adapter
    -> HarnessRunner               # provider-neutral bounded boundary
    -> Tool Policy / Approval
    -> ToolExecutor
       -> Workspace tools
       -> run_command
       -> MCP tools

Optional execution services
    -> Run History
    -> Harness Checkpoints
    -> Harness Resume
    -> Run Lease
```

`ChatRuntime` 负责 CLI chat 的事件、生命周期和可选服务编排；`HarnessRunner` 是可被库调用的 provider-neutral、有界执行边界。两者都不是多 Agent 调度器，也不提供任意 Shell、交互式终端、后台任务或 Git 操作。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

如果你的本地工具链更习惯使用 requirements 文件，也可以安装同一组运行依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 配置

复制 `.env.example` 为 `.env`，或直接在 shell 中导出环境变量：

```powershell
$env:NEXUSMIND_MODEL_BASE_URL = "https://api.openai.com/v1"
$env:NEXUSMIND_MODEL_API_KEY = "your-api-key"
$env:NEXUSMIND_MODEL_NAME = "gpt-4.1-mini"
$env:NEXUSMIND_MODEL_TIMEOUT = "60"
```

CLI 在错误信息中会刻意避免打印 API Key。

## 运行

```powershell
nexusmind chat "介绍一下你自己"
```

也可以启动一次交互式单轮提示：

```powershell
nexusmind chat
```

`chat` 的能力通过显式 flags 开启：`--workspace-write` 和 `--workspace-exec` 都要求同时提供 `--workspace`；`--workspace-exec` 还要求 `--command-config`。`--mcp-server` 只用于直接的 `chat` MCP 集成，并要求 `--mcp-config`；Skill 会根据自身的 `allowed_tools` 从 `--mcp-config` 解析需要的 Server。`--checkpoint-run-id` 要求 `--checkpoint-db`，`--lease-run-id` 要求 `--lease-db`，不匹配当前 execution `run_id` 时会 fail fast。

## 工具

工具系统不依赖模型配置或 API Key。

列出已注册的内置工具：

```powershell
nexusmind tools list
```

调用内置 echo 工具：

```powershell
nexusmind tools call echo '{"text":"hello"}'
```

`tools call` 是用户主动直调工具，不会经过 Agent Tool Loop，也不会触发审批。

内置 `approval_demo` 工具用于通过 `chat` 演示审批流程。它标记为 `LOCAL_WRITE` 以触发 Allow once / Deny，但不会修改本地状态：

```powershell
nexusmind chat "请调用 approval_demo 工具，message 设置为 hello"
```

### Workspace 只读文件工具

默认情况下，普通 `chat` 不会暴露文件系统工具。用户必须为单次运行显式提供一个 Workspace Root：

```powershell
nexusmind chat --workspace ./project "分析这个项目的入口和主要模块"
```

提供 `--workspace` 后，NexusMind 会注册三个只读工具：`list_files`、`read_file` 和 `search_text`。这些工具只能访问 Workspace Root 内部，只返回相对路径，第一版不跟随任何符号链接，仅支持严格 UTF-8 文本，并对目录遍历、文件大小、扫描字节、匹配数和输出大小设置上限。

Workspace 内容可能通过工具结果发送给模型 Provider。文件工具不支持写文件、Patch、Shell、glob 或正则搜索。

Skill 可以在 `allowed_tools` 中引用文件工具，但不能声明或改变 Workspace Root；Host 仍必须在运行时传入 `--workspace`：

```powershell
nexusmind skill run workspace-code-review `
  --skills-dir ./examples/skills `
  --workspace ./project `
  "检查项目中的并发与资源释放问题"
```

默认 `--workspace` 是只读模式。只有显式增加 `--workspace-write` 时，NexusMind 才会向模型暴露 `write_file` 和 `replace_text`：

```powershell
nexusmind chat `
  --workspace ./project `
  --workspace-write `
  "修改代码"
```

写工具标记为本地写入风险，每次调用默认仍需要用户批准。覆盖现有文件必须使用 `read_file` 返回的 `sha256` 作为 `expected_sha256`；如果文件在读取后被用户或其他进程修改，写入会失败而不是静默覆盖。

第一版写入能力只支持 UTF-8 普通文件的创建、完整替换和精确文本替换；不支持删除、重命名、创建目录、Shell 或 Git。修改后的内容可能在后续模型轮次中通过读取工具发送给模型 Provider。

覆盖写入使用同目录临时文件和原子替换，并保留目标文件的基础权限位；不承诺复制 owner/group、ACL、扩展属性或平台安全标签。

### Workspace 命令 Profile

`--workspace-exec` 是高风险能力。它会向模型暴露 `run_command`，但模型只能选择 Host 在 `--command-config` 中预先声明的固定 Profile，不能提供 shell 字符串、额外 argv、cwd、env 或 timeout：

```powershell
nexusmind chat `
  --workspace ./project `
  --workspace-exec `
  --command-config ./examples/commands/python-ci.json `
  "运行测试并解释失败"
```

命令 Profile 配置等价于授权 NexusMind 以当前 OS 用户权限执行固定本地程序。该能力不是容器、虚拟机、OS 沙箱、cgroup scope 或 Windows 安全边界；cwd 被限制在 Workspace 内也不代表进程无法访问 Workspace 外文件，也不提供网络隔离。

每次 `run_command` 默认仍需要用户批准。审批摘要会显示 Profile、Workspace 相对 cwd、固定 argv 摘要和超时。命令输出会作为 ToolResult 发送给模型 Provider，程序主动打印的本地路径或敏感内容不会被 NexusMind 自动清除。

当前版本仅支持 Windows。`run_command` 使用 Windows Job Object 在 timeout、取消和正常结束时回收进程树；Linux 和 macOS 不属于支持的平台，命令 Profile 会显式失败。第一版不支持任意 Shell、交互式终端、stdin、后台任务、模型自定义环境变量或密钥注入。

## MCP Stdio 工具

MCP server 配置从 JSON 文件读取。如果配置文件包含环境变量或密钥，请把它当作敏感文件处理。

示例 `mcp.json`：

```json
{
  "servers": {
    "demo": {
      "transport": "stdio",
      "command": "python",
      "args": ["tests/fixtures/mcp_echo_server.py"],
      "cwd": null,
      "env": {}
    }
  }
}
```

列出该 server 暴露的工具：

```powershell
nexusmind mcp tools --config mcp.json --server demo
```

通过 NexusMind 注册表和执行器调用发现到的 MCP 工具：

```powershell
nexusmind mcp call --config mcp.json --server demo --tool demo__echo_290c9db7d5 --arguments '{"text":"hello"}'
```

把单个 MCP Stdio Server 接入 `chat` 的 Agent Tool Loop：

```powershell
nexusmind chat --mcp-config mcp.json --mcp-server demo "请调用 MCP 工具完成任务"
```

`chat` 会在首次模型调用前发现并注册该 Server 的工具。MCP 工具默认风险级别为 `UNSPECIFIED`，因此模型请求调用时会经过 CLI Allow once / Deny 审批。

## Skill

Skill 是本地声明式任务定义，由 instructions、工具白名单和收紧后的运行上限组成。示例：

```powershell
nexusmind skill list --skills-dir ./examples/skills
nexusmind skill show mcp-echo --skills-dir ./examples/skills
```

运行不依赖 MCP 的 Skill：

```powershell
nexusmind skill run code-review --skills-dir ./skills "检查当前项目"
```

运行引用多个 MCP Server 的 Skill 时，只需要提供 Host 管理的 MCP 配置文件；NexusMind 会从 `allowed_tools` 中自动推导并连接实际需要的 Server，不会启动未引用的 Server：

```powershell
nexusmind skill run multi-mcp-review `
  --skills-dir ./examples/skills `
  --mcp-config mcp.json `
  "检查这个 PR"
```

普通 `chat` 仍只支持显式指定单个 `--mcp-server`。

## 模型工具调用

NexusMind 可以把已注册的 `ToolDefinition` 传给 OpenAI-compatible chat model，把流式 `tool_calls` 解析成与服务商无关的事件，通过 `ToolExecutor` 执行模型请求的工具，并把结构化 `role=tool` 结果回填到下一轮模型调用。运行时使用有界的单 Agent 循环限制，避免模型轮次或工具结果无限增长。

### Run History（可选）

可以显式启用本地 SQLite Run Store：

```bash
nexusmind chat --state-db ./.nexusmind/state.db "分析这个项目"
nexusmind runs list --state-db ./.nexusmind/state.db
nexusmind runs show <run_id> --state-db ./.nexusmind/state.db --json
nexusmind runs prune --state-db ./.nexusmind/state.db --older-than-days 30
nexusmind runs recover --state-db ./.nexusmind/state.db
```

未提供 `--state-db` 时不会创建数据库。默认只记录执行元数据；`--record-content` 才会保存有界的输入预览。数据库可能包含任务、模型和工具执行元数据，应按敏感数据保护。它只记录历史，不支持恢复、重放或自动重新执行工具。确认旧进程已停止后，可显式使用 `runs recover` 将遗留的 `running` Run 标记为 `abandoned`。

### Harness Checkpoint 与 Resume

Harness Checkpoint 是执行状态快照，不是 Run History 的别名。它在安全边界保存消息 transcript、模型/工具计数、Tool Call 身份和 Harness phase；活跃 Tool 尚未完成时不会创建可恢复 checkpoint。`CheckpointBoundary` 支持 `before_model`、`after_model`、`before_tool`、`after_tool` 和 `run_terminal`。启用 checkpoint persistence 后，`CheckpointCoordinator` 会在 `AFTER_MODEL`、`AFTER_TOOL` 和 `RUN_TERMINAL` 边界自动持久化；终态 checkpoint 可以用 `--no-terminal-checkpoint` 关闭。

启用 SQLite checkpoint：

```powershell
nexusmind chat `
  --checkpoint-db ./.nexusmind/checkpoints.db `
  --checkpoint-run-id run-123 `
  "执行一个可恢复的任务"
```

`--checkpoint-run-id` 必须同时提供 `--checkpoint-db`，并且如果本次运行同时启用了 Run History，必须与该 execution 的 `run_id` 一致。未提供 `--checkpoint-run-id` 时，CLI 会复用已有 Run History 的 `run_id`，否则生成新的 ID。checkpoint store 使用版本化 SQLite schema、单调递增的 per-run sequence 和事务写入；创建、保存或提交失败会以 checkpoint failure 终止或标记对应边界，避免把未持久化的状态当成已安全保存。checkpoint 数据包含消息和 Tool 结果，应按本地敏感状态保护。

Resume 目前是库/runtime 能力，不是独立的 `nexusmind resume <run-id>` CLI 命令。Host 可以从 `SQLiteCheckpointStore` 读取 checkpoint，并通过 `HarnessRunner.resume_execution(HarnessResumeRequest(...))` 恢复受支持的 `BEFORE_MODEL`、`AFTER_MODEL`、`BEFORE_TOOL` 或 `AFTER_TOOL` 状态；恢复前会校验 checkpoint、transcript、Tool 定义、已消耗的 limits 和未完成 Tool Call，已完成的 Model/Tool 工作不会被重放。终态 checkpoint 用于审计，不表示存在一个可继续执行的终态。

这几个概念解决的是不同问题：

| 能力 | 作用 | 当前入口 |
| --- | --- | --- |
| Run History | 记录一次运行发生了什么，供 list/show/prune/recover 审计；不会自动 replay 或 resume | CLI `--state-db`、`nexusmind runs ...` |
| Harness Checkpoint | 保存可验证的 Harness 状态边界 | CLI `--checkpoint-db`；`CheckpointStore` / `SQLiteCheckpointStore` |
| Harness Resume | 从受支持的 checkpoint 状态继续执行，避免重复已完成工作 | Library `HarnessRunner.resume_execution(...)` |
| Run Lease | 为 live `run_id` 提供单一 owner，阻止并发执行推进同一运行 | CLI `--lease-db` / `--lease-run-id`；`RunLeaseCoordinator` |

### Run 执行租约

需要为同一 `run_id` 保证单一执行 owner 时，使用独立的租约数据库：

```powershell
nexusmind chat `
  --lease-db ./.nexusmind/leases.db `
  --lease-run-id run-123 `
  "分析这个项目"
```

`--lease-db` 会在模型或工具执行前原子获取 SQLite 租约，并在运行期间 heartbeat；活动租约冲突、续租失败或 owner 不匹配都会 fail closed。正常完成、失败和取消会尝试按 owner 释放租约；崩溃后的租约只有到期后才能被新 owner 接管。`--lease-run-id` 可省略，此时复用 Run History/checkpoint 的 `run_id`，若两者都未启用则生成新 ID。租约数据独立于 Harness checkpoint，建议使用单独的数据库文件。

模型 Provider、工具执行器和异步事件迭代器必须遵守 cooperative cancellation contract：收到 `asyncio.CancelledError` 后应在有限时间内结束，不得无限吞掉取消。运行时会对当前迭代任务停止 heartbeat、保留租约到 TTL 并阻止同一 `ChatRuntime` 复用，避免未知的后台执行与新 owner 重叠；但 Python 的 `asyncio.run()` 无法强制终止任意不合作的 coroutine，因此 Host 若需要进程级硬 shutdown，应将 Provider 隔离到可终止的 worker process。

SQLite 租约的 `clock` 是测试注入点。生产环境应让所有访问同一个 lease database 的 `SQLiteRunLeaseStore` 使用默认 UTC 时钟；如果注入自定义时钟，所有 contender 必须共享同一个 authoritative clock source。SQLite 文件不会协调不同 store 实例之间的时钟偏差，因此不同时间源可能让本地 ownership proof 滞后于数据库 takeover。对暴露 `clock` 的 store，`RunLeaseCoordinator` 会将 guard 绑定到 store 时钟；传入 coordinator 的 `clock` 只适用于未暴露 authoritative clock 的自定义 store。

工具定义的默认风险级别是 `UNSPECIFIED`，默认策略会要求审批；确认只读工具时应显式设置 `ToolRiskLevel.READ_ONLY`。

## 分类检索评估与后端比较

`evals/knowledge/benchmark/` 是离线、UTF-8、人工标注的检索基准，比较
BM25-only、Semantic-only、Hybrid-RRF 与 Hybrid-RRF + Rerank。第四个 backend
使用完全离线、由 query/candidate 内容驱动的确定性 fixture，只重排固定 Hybrid 候选，
不读取 case ID 或 relevance answer。每个 case 必须显式使用一个严格类别：
`exact_term`、`identifier`、`cjk`、`paraphrase`、`cross_language`、
`multi_document`、`distractor_heavy` 或 `mixed_signal`；未知类别、重复 case ID
与重复相关文档都会 fail closed，类别不会从 ID 推断。

评估同时生成 K=1、3、5、10 的 Hit@K、Recall@K 与 MRR。每个 backend/query
只执行一次 `max(K)` 搜索，较小 K 严格来自同一排名的前缀，不会重新搜索、去重
或改变 relevance label。Hit 表示前 K 中是否至少存在一个相关文档，Recall 按不同
相关文档的覆盖率计算，MRR 使用首个相关 chunk 的真实排名。较大 K 的 Recall 仍低
通常表示候选召回问题；较大 K Recall 高而较小 K Hit/MRR 低则表示排序问题。

比较开始前，所有 backend 的完整 canonical `KnowledgeSnapshot`（包括来源、文档、
内容、metadata 与顺序）必须完全相等。确定性 semantic fixture 只验证架构和诊断，
不代表真实 embedding model 的质量；人工 relevance label 独立于后端输出，不为使
Hybrid 获胜而调整。生成的 [比较报告](evals/knowledge/benchmark.md) 是
descriptive/non-gate，不设置任意质量阈值。

重新生成报告：

```bash
PYTHONPATH=src python -m nexusmind.retrieval_benchmark --write evals/knowledge/benchmark.md
```

renderer 是结构化 report 与固定显示配置之间的纯函数，不读取 collection、文件系统、
环境变量、时钟或随机数。测试要求重新生成的 Markdown 与 checked-in baseline
逐字节相同。结果可用于决定后续研究召回、reranking、embedding persistence 或索引
策略，但本评估不会自动实现或选择这些功能。

### Retrieval Runtime v1 roadmap

有界 lexical、semantic、Hybrid-RRF、second-stage reranking、canonical provenance、
snapshot/SQLite restore 与分类离线评估共同完成 Retrieval Runtime v1 的底层边界。
下一阶段是面向使用者的 Knowledge Base / Workspace APIs；query rewriting、远程 reranker、
持久化 embedding/index、RAG 生成和产品 UI 仍不属于本阶段。

## 开发验证

```bash
python -m pip install -e ".[dev]"
python -m pip check
python -m compileall -q src
python -m pytest -q
```

项目支持 Python 3.11、3.12 和 3.13，运行平台为 Windows。CI 在 `windows-latest` 上运行完整离线测试套件；Linux 和 macOS 不属于当前支持范围。

测试不得依赖真实 API Key、真实模型服务、真实外部 MCP Server 或开发者本机目录。新增功能应包含对应测试；MCP 测试应继续使用仓库内离线 fixture。
