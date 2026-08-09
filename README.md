# NexusMind

NexusMind 从一个轻量、与模型服务商解耦的模型运行时开始。当前基线支持：

- NexusMind 自有的消息与运行时事件契约。
- 支持异步流式输出的 `ChatModel` 抽象。
- OpenAI-compatible HTTP 适配器。
- 与服务商无关的工具注册表和工具执行器。
- 用于流式聊天输出的 CLI 入口。
- 用于离线测试的 fake model。

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

第一版仅支持 Windows 和 Linux。NexusMind 会尽力在 timeout、取消和正常结束时回收进程组或 Job Object，但在 Linux 上如果被执行代码杀死 supervisor 或主动脱离执行单元，后台后代清理是 best-effort，不是内核级 containment 保证。第一版不支持任意 Shell、交互式终端、stdin、后台任务、模型自定义环境变量或密钥注入。

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

### Run 执行租约

需要为同一 `run_id` 保证单一执行 owner 时，使用独立的租约数据库：

```powershell
nexusmind chat `
  --lease-db ./.nexusmind/leases.db `
  --lease-run-id run-123 `
  "分析这个项目"
```

`--lease-db` 会在模型或工具执行前原子获取 SQLite 租约，并在运行期间 heartbeat；活动租约冲突、续租失败或 owner 不匹配都会 fail closed。正常完成、失败和取消会尝试按 owner 释放租约；崩溃后的租约只有到期后才能被新 owner 接管。`--lease-run-id` 可省略，此时复用 Run History/checkpoint 的 `run_id`，若两者都未启用则生成新 ID。租约数据独立于 Harness checkpoint，建议使用单独的数据库文件。

工具定义的默认风险级别是 `UNSPECIFIED`，默认策略会要求审批；确认只读工具时应显式设置 `ToolRiskLevel.READ_ONLY`。

## 开发验证

```bash
python -m pip install -e ".[dev]"
python -m pip check
python -m compileall -q src
python -m pytest -q
```

项目支持 Python 3.11、3.12 和 3.13。CI 在 Ubuntu 和 Windows 上运行完整离线测试套件。

测试不得依赖真实 API Key、真实模型服务、真实外部 MCP Server 或开发者本机目录。新增功能应包含对应测试；MCP 测试应继续使用仓库内离线 fixture。
