# fNIRS 深度学习多智能体自助平台源程序说明文档

## 1. 文档目的

本文档从源代码维护角度说明项目中每个主要程序文件的职责、核心类与函数、输入输出、调用关系、异常路径和维护注意事项。

本文档适合以下场景：

- 新开发者接手项目。
- 测试人员根据源码补充用例。
- 后续扩展真实深度学习训练。
- 排查 API、RAG、数据解析、实验运行、解释或报告问题。
- 对照设计文档和架构文档理解实现落点。

## 2. 项目源程序总览

项目主要分为四个代码区域：

```text
backend/       FastAPI、服务编排、数据库、多智能体、论文工作流
fnirs_core/    fNIRS 领域核心能力
frontend/      Vue 3 工作台
tests/         后端与核心能力测试
```

运行时数据主要分布在：

```text
storage/       SQLite 数据库
artifacts/     上传文件、实验结果、向量库、论文缓存
knowledge/     内置知识和抽取后的托管知识
logs/          运行日志
```

## 3. 后端程序说明

## 3.1 backend/main.py

### 文件职责

`backend/main.py` 是 FastAPI 后端入口。

它负责：

- 创建 FastAPI app。
- 配置 CORS。
- 在生命周期启动阶段初始化运行环境。
- 定义所有 HTTP API。
- 定义 SSE 格式。
- 将 service 层异常转换为 HTTP 错误。
- 包装流式聊天输出。

它不负责：

- 数据库表定义。
- 领域算法。
- 知识库索引细节。
- 实验执行细节。
- 前端状态管理。

### 生命周期

```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_runtime()
    yield
```

`ensure_runtime()` 来自 `backend.services`，会初始化数据库、目录和知识库。

### App 配置

```python
app = FastAPI(
    title="fNIRS Multi-Agent Platform API",
    version="1.0.0",
    lifespan=lifespan,
)
```

CORS 当前配置为全开放，便于本地开发：

```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

### SSE 工具函数

```python
def _sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
```

输出格式示例：

```text
event: content_chunk
data: {"content":"你好", "assembled":"你好"}
```

### 流式聊天函数

```python
def _chat_stream(request: ChatRequest) -> Iterator[str]:
```

核心逻辑：

1. 获取 orchestrator。
2. 初始化 assembled、sources、agent_steps。
3. 先发送 `status`。
4. 遍历 orchestrator.stream。
5. 根据 event type 更新本地聚合状态。
6. 将每个 event 作为 SSE 发送。
7. 完成后保存聊天会话。
8. 发送 `session` 和 `done`。
9. 异常时发送 `error` 和 `done`。

需要注意：

- 流式事件中的 `type` 字段会被转成 SSE event 名。
- `content_chunk` 会更新 assembled。
- `final` 会覆盖 assembled、sources 和 agent_steps。
- 会话保存发生在模型流结束后。

### API 路由清单

#### 健康与设置

| 方法 | 路径 | 函数 | 返回 |
| --- | --- | --- | --- |
| GET | `/api/health` | `health()` | 运行状态 |
| PUT | `/api/settings` | `settings_update()` | 更新后的运行状态 |

`settings_update()` 会捕获 `ValueError` 并返回 400。

#### Dashboard

| 方法 | 路径 | 函数 |
| --- | --- | --- |
| GET | `/api/dashboard` | `dashboard_snapshot()` |

#### Knowledge

| 方法 | 路径 | 函数 |
| --- | --- | --- |
| GET | `/api/knowledge` | `knowledge_status()` |
| POST | `/api/knowledge/refresh` | `knowledge_refresh()` |
| GET | `/api/knowledge/documents` | `knowledge_documents()` |
| POST | `/api/knowledge/documents` | `knowledge_create()` |
| GET | `/api/knowledge/documents/{document_id}` | `knowledge_detail()` |
| PUT | `/api/knowledge/documents/{document_id}` | `knowledge_update()` |
| PATCH | `/api/knowledge/documents/{document_id}/chunks/{order}` | `knowledge_chunk_update()` |
| DELETE | `/api/knowledge/documents/{document_id}` | `knowledge_delete()` |
| POST | `/api/knowledge/upload` | `knowledge_upload()` |

知识库相关错误主要捕获 `KnowledgeBaseError`。

#### Chat

| 方法 | 路径 | 函数 |
| --- | --- | --- |
| POST | `/api/chat/stream` | `chat_stream()` |
| POST | `/api/chat` | `chat()` |
| GET | `/api/chat/sessions` | `chat_sessions()` |
| GET | `/api/chat/sessions/{session_id}` | `chat_session_detail()` |
| DELETE | `/api/chat/sessions/{session_id}` | `chat_session_delete()` |

非流式 `chat()` 与流式 `_chat_stream()` 使用同一 orchestrator，只是前者在服务端聚合所有事件后一次性返回。

#### Datasets

| 方法 | 路径 | 函数 |
| --- | --- | --- |
| GET | `/api/datasets` | `datasets()` |
| POST | `/api/datasets/upload` | `dataset_upload()` |
| GET | `/api/datasets/{dataset_id}/summary` | `dataset_summary()` |
| DELETE | `/api/datasets/{dataset_id}` | `dataset_delete()` |

上传文件名为空时返回 400。

#### Experiments

| 方法 | 路径 | 函数 |
| --- | --- | --- |
| GET | `/api/experiments` | `experiments()` |
| POST | `/api/experiments` | `experiment_create()` |
| GET | `/api/experiments/{experiment_id}` | `experiment_detail()` |
| DELETE | `/api/experiments/{experiment_id}` | `experiment_delete()` |
| POST | `/api/experiments/{experiment_id}/run` | `experiment_run()` |
| GET | `/api/experiments/{experiment_id}/results` | `experiment_results()` |
| DELETE | `/api/experiments/{experiment_id}/results` | `experiment_results_delete()` |
| POST | `/api/experiments/{experiment_id}/explain` | `experiment_explain()` |

#### Jobs

| 方法 | 路径 | 函数 |
| --- | --- | --- |
| GET | `/api/jobs` | `jobs()` |
| GET | `/api/jobs/{job_id}` | `job_detail()` |

#### Reports

| 方法 | 路径 | 函数 |
| --- | --- | --- |
| GET | `/api/reports/{experiment_id}/download` | `report_download()` |

`report_download()` 会先调用 `generate_report()`，再检查文件是否存在，最后返回 `FileResponse`。

### 维护注意事项

- 新增 API 时优先在 `backend.schemas` 定义输入输出模型。
- route 内不建议写复杂业务逻辑，应该委托给 `backend.services`。
- SSE event payload 必须可 JSON 序列化。
- 文件上传必须检查文件名和后缀。
- 删除类接口应由 service 层做路径边界校验。

## 3.2 backend/services.py

### 文件职责

`backend/services.py` 是后端业务编排中心。

它连接：

- FastAPI route。
- SQLite。
- 本地文件系统。
- `fnirs_core`。
- 多智能体 orchestrator。
- Ollama 配置。

### 关键常量

| 常量 | 默认路径或值 |
| --- | --- |
| `ROOT_DIR` | 项目根目录 |
| `ARTIFACTS_DIR` | `artifacts/` |
| `DATASET_DIR` | `artifacts/datasets/` |
| `RAW_KNOWLEDGE_DIR` | `artifacts/knowledge_uploads/raw/` |
| `EXTRACTED_KNOWLEDGE_DIR` | `knowledge/uploads/extracted/` |
| `REPORT_DIR` | `artifacts/reports/` |
| `EXPERIMENT_DIR` | `artifacts/experiments/` |
| `BASE_KNOWLEDGE_DIR` | `knowledge/base/` |
| `DEFAULT_OLLAMA_BASE_URL` | `http://localhost:11434` |
| `DEFAULT_CHAT_MODEL` | `qwen3:8b` |
| `DEFAULT_EMBEDDING_MODEL` | `qwen3-embedding:8b` |

### RuntimeConfig

```python
@dataclass(slots=True)
class RuntimeConfig:
    ollama_base_url: str
    chat_model: str
    embedding_model: str
```

用于统一传递 Ollama base URL、chat 模型和 embedding 模型。

### 运行配置函数

#### get_runtime_config()

读取配置顺序：

1. 环境变量。
2. SQLite `runtime_settings` 表。
3. 默认值。

会对 Ollama base URL 执行 `rstrip("/")`。

#### update_runtime_config()

支持更新：

- `chat_model`
- `embedding_model`

会调用 `_clean_model_name()`：

- 不允许空字符串。
- 不允许超过 200 字符。

更新后返回 `probe_health()`。

### ensure_runtime()

职责：

1. 调用 `db.init_db()`。
2. 创建运行目录。
3. 调用 `_ensure_seed_knowledge()` 创建内置知识。
4. 尝试构建知识库。
5. 如果构建失败，强制刷新知识库。

内置知识文件：

```text
knowledge/base/fnirs_platform_guide.md
```

### 知识库服务函数

#### build_default_knowledge_base(refresh=False)

将运行时 embedding 配置传给 `fnirs_core.knowledge.build_default_knowledge_base()`。

#### build_knowledge_status()

返回 `KnowledgeStatusResponse`。

#### refresh_knowledge_base()

强制刷新索引并返回状态。

#### list_knowledge_documents()

返回：

- documents。
- knowledge status。

#### get_knowledge_document(document_id)

返回：

- 文档摘要字段。
- 文档全文。
- 文档 chunks。

#### create_knowledge_document(title, content)

逻辑：

1. 检查内容非空。
2. 在托管知识目录中创建唯一 Markdown 路径。
3. 写入 `# title` 和正文。
4. 刷新知识库。
5. 返回新文档详情。

#### update_knowledge_document(document_id, title, content)

逻辑：

1. 查找文档。
2. 不存在则抛 `KnowledgeBaseError`。
3. 非托管文档拒绝编辑。
4. 写回 Markdown。
5. 刷新知识库。
6. 返回文档详情。

#### delete_knowledge_document(document_id)

逻辑：

1. 查找文档。
2. 非托管文档拒绝删除。
3. 校验路径在 `EXTRACTED_KNOWLEDGE_DIR` 下。
4. 删除文件。
5. 刷新知识库。

#### set_knowledge_chunk_enabled(document_id, order, enabled)

调用 `KnowledgeBase.set_chunk_enabled()` 并返回文档详情。

#### ingest_knowledge_file(filename, content)

逻辑：

1. 检查后缀。
2. 保存原始文件。
3. 调用 `extract_text_from_document()`。
4. 如果抽取为空，删除原始文件并报错。
5. 写入托管 Markdown。
6. 刷新知识库。
7. 返回上传结果。

支持后缀：

```text
.pdf, .docx, .doc, .md, .markdown, .txt, .text
```

### 数据集服务函数

#### upload_dataset(filename, content)

逻辑：

1. 检查后缀在 `SUPPORTED_DATA_SUFFIXES`。
2. 创建数据集 ID。
3. 保存文件到 `artifacts/datasets/{dataset_id}/`。
4. 调用 `summarize_file()`。
5. 写入 SQLite。
6. 返回 `DatasetResponse`。

#### delete_dataset(dataset_id)

逻辑：

1. 读取数据集。
2. 删除 datasets 记录。
3. 查找关联实验。
4. 将关联实验 `dataset_id` 置空。
5. 更新关联实验 config 中的 `dataset_id` 和 `dataset_path`。
6. 删除数据集目录。

### 实验服务函数

#### create_experiment(payload)

构建 config：

```python
config = {
    "name": ...,
    "dataset_id": ...,
    "dataset_path": ...,
    "preprocessing": ...,
    "model": ...,
    "validation_strategy": ...,
    "num_folds": ...,
    "seed": ...,
    "output_dir": str(EXPERIMENT_DIR),
}
```

如果 `dataset_id` 存在，会通过 `get_dataset()` 获取 `dataset_path`。

#### run_experiment_job(experiment_id)

逻辑：

1. 获取实验。
2. 创建 `experiment_run` job。
3. 启动 daemon thread。
4. thread 内更新实验状态为 running。
5. 调用 `fnirs_core.experiments.run_experiment()`。
6. 成功后写 `result_json` 并更新 job succeeded。
7. 失败后写实验 failed 和 job failed。

progress callback：

```python
def progress(value: float, message: str) -> None:
    update_job(job_id, status="running", progress=value, message=message, log=message)
```

#### explain_experiment_job(experiment_id)

逻辑：

1. 获取实验。
2. 创建 `experiment_explain` job。
3. 启动 daemon thread。
4. 调用 `explain_experiment()`。
5. 写入 `explanation_json`。
6. 更新 job 状态。

#### generate_report(experiment_id)

逻辑：

1. 获取实验。
2. 确定 output_dir。
3. 组装 experiment payload。
4. 调用 `generate_experiment_report()`。
5. 写入 `report_path`。
6. 返回报告路径。

#### delete_experiment(experiment_id)

逻辑：

1. 获取实验。
2. 收集实验输出目录。
3. 收集 report_path。
4. 删除 experiments 记录。
5. 删除 payload 中关联该 experiment 的 jobs。
6. 删除输出目录。
7. 删除报告文件。

#### delete_experiment_results(experiment_id)

逻辑：

1. 获取实验。
2. 收集输出目录和报告路径。
3. 将 status 改回 `created`。
4. 清空 result_json、explanation_json、report_path。
5. 删除输出目录。
6. 删除报告文件。

### Job 服务函数

#### create_job(kind, payload)

创建 job，初始值：

```text
status = queued
progress = 0.0
message = 任务已创建
logs_json = []
```

#### update_job(...)

读取当前 job，追加 log，最多保留最近 200 条。

注意：

- `result_json` 使用 `COALESCE(?, result_json)`，传入 None 时保留旧值。
- `error` 同样使用 COALESCE。

#### list_jobs(limit=20)

按 `created_at DESC` 返回最近任务。

### 聊天会话函数

#### save_chat_session(...)

如果传入的 `session_id` 存在：

- 读取旧 messages。
- 追加 user 和 assistant 消息。
- 更新 sources 和 agent_steps。

如果不存在：

- 创建新会话。
- title 使用 user message 前 60 字。

#### _latest_agent_steps(steps)

只返回最新一次 `"Supervisor Agent" + "route"` 之后的步骤。

设计目的：

- 多轮会话消息完整保留。
- 右侧智能体步骤只显示最新一轮，避免混淆。

### 健康和 dashboard 函数

#### probe_health()

调用 Ollama `/api/tags`。

异常处理：

- `urllib_error.URLError` -> `ollama_status = "down"`
- 其他异常 -> `ollama_status = "error"`

#### dashboard()

聚合：

- health。
- knowledge。
- datasets 前 5 个。
- experiments 前 5 个。
- jobs 前 5 个。

### 路径工具函数

#### _safe_filename(value)

只保留：

- 字母数字。
- `-`
- `_`
- `.`

其他字符替换为 `_`，最多 120 字符。

#### _unique_path(path)

如果路径存在，追加 `_1` 到 `_999`。

#### _ensure_under(path, root)

通过 `path.relative_to(root)` 校验路径边界。

#### _remove_tree_if_under(path, root)

先校验，再 `shutil.rmtree()`。

#### _remove_report_file(path)

只允许删除 `REPORT_DIR` 或 `EXPERIMENT_DIR` 下的报告。

### 维护注意事项

- 涉及删除文件时必须继续使用 `_ensure_under` 风格校验。
- 新增 artifact 根目录时应加入 `ensure_runtime()`。
- 修改实验 config schema 时要同步前端表单和 Pydantic schema。
- 后台线程中捕获异常后必须更新 job 状态，避免任务永久 running。

## 3.3 backend/db.py

### 文件职责

`backend/db.py` 封装 SQLite 连接、初始化、JSON 序列化和 ID/时间工具。

### 关键路径

```python
STORAGE_DIR = ROOT_DIR / "storage"
DB_PATH = STORAGE_DIR / "app.db"
```

### 工具函数

#### now_iso()

返回 UTC ISO 时间。

#### new_id(prefix)

返回：

```text
{prefix}_{uuid4 hex 前 12 位}
```

#### connect()

上下文管理器：

- 创建 storage 目录。
- 打开 SQLite。
- 设置 `row_factory = sqlite3.Row`。
- yield connection。
- 正常退出 commit。
- finally close。

### init_db()

创建表：

- projects。
- chat_sessions。
- datasets。
- experiments。
- jobs。
- runtime_settings。

然后调用 `_ensure_default_project()`。

### loads(value, default)

安全 JSON 解析。解析失败返回 default。

### dumps(value)

使用：

```python
json.dumps(value, ensure_ascii=False)
```

保证中文不转义。

### 维护注意事项

- 新增表时写在 `init_db()` 中，保持幂等。
- 当前没有 migration 框架，结构变更要考虑旧库兼容。
- JSON 字段读取必须使用 `db.loads()`，避免坏数据导致接口崩溃。

## 3.4 backend/schemas.py

### 文件职责

定义 FastAPI 请求和响应的 Pydantic 模型。

### 主要模型分组

#### 状态模型

- `BackendStatusResponse`
- `RuntimeSettingsUpdateRequest`
- `DashboardResponse`

#### 知识库模型

- `KnowledgeStatusResponse`
- `KnowledgeDocumentResponse`
- `KnowledgeChunkResponse`
- `KnowledgeDocumentDetailResponse`
- `KnowledgeDocumentsResponse`
- `KnowledgeDocumentCreateRequest`
- `KnowledgeDocumentUpdateRequest`
- `KnowledgeChunkUpdateRequest`
- `KnowledgeUploadResponse`

#### 对话模型

- `ChatRequest`
- `AgentStepResponse`
- `ChatResponse`
- `ChatSessionResponse`
- `ChatSessionListResponse`

#### 数据集模型

- `DatasetResponse`
- `DatasetListResponse`
- `DatasetUploadResponse`

#### 实验和任务模型

- `ExperimentCreateRequest`
- `ExperimentResponse`
- `ExperimentListResponse`
- `JobResponse`

### 维护注意事项

- 前端依赖这些字段名称，改名会影响 UI。
- 新增响应字段通常向后兼容；删除字段会破坏前端。
- 请求字段可以用 `Field` 约束长度和默认值。

## 3.5 backend/agents.py

### 文件职责

实现轻量多智能体编排和 Ollama chat 调用。

### 领域关键词

文件顶部定义多组关键词：

- `DOMAIN_KEYWORDS`
- `IDENTITY_CHAT_KEYWORDS`
- `CASUAL_CHAT_MESSAGES`
- `EXPERIMENT_KEYWORDS`
- `DATA_INTENT_KEYWORDS`
- `DATA_FILE_SUFFIXES`
- `PAPER_INTENT_KEYWORDS`

这些关键词用于路由判断和边界提示。

### AgentStep

```python
@dataclass(slots=True)
class AgentStep:
    agent: str
    action: str
    detail: str
    status: str = "completed"
```

表示一次可展示步骤。

### AgentContext

字段：

- `query`
- `route`
- `sources`
- `retrieval_results`
- `steps`

方法：

- `add_step(agent, action, detail, status="completed")`

### OllamaChatClient

职责：

- 调用 Ollama `/api/chat`。
- 按 stream 返回文本片段。

构造参数：

- `model`
- `base_url`
- `timeout`

stream 请求：

```json
{
  "model": "...",
  "messages": [],
  "stream": true,
  "options": {"temperature": 0.2}
}
```

异常：

- `urllib_error.URLError` -> `RuntimeError("无法连接 Ollama...")`
- 其他异常 -> `RuntimeError("Ollama 响应异常...")`

### MultiAgentOrchestrator

核心入口：

```python
def stream(self, query: str) -> Iterator[dict[str, Any]]:
```

流程：

1. 创建 context。
2. 调用 `_route(query)`。
3. 发送 Supervisor route step。
4. 调用 `_review_guardrail(query)`。
5. 发送 Reviewer step。
6. 根据 route 分发：
   - data -> `_stream_agent_model_answer(context, "Data Agent")`
   - experiment -> `_stream_agent_model_answer(context, "Experiment Agent")`
   - explain -> `_stream_agent_model_answer(context, "Explain Agent")`
   - report -> `_stream_agent_model_answer(context, "Report Agent")`
   - paper -> `_stream_paper_workflow(context)`
   - chat -> `_stream_chat_answer(context)`
   - 默认 -> `_stream_rag_answer(context)`

### 路由规则

```python
def _route(self, query: str) -> str:
```

顺序：

1. basic chat。
2. paper。
3. data。
4. experiment。
5. explain。
6. report。
7. rag。

顺序很重要。例如“你是什么模型”虽然包含“模型”，但会先被 basic chat 捕获，不会误路由到 Experiment Agent。

### RAG 回答

`_stream_rag_answer()`：

1. 发送 retrieve step。
2. 调用 `knowledge_base.search(query, top_k=FNIRS_RAG_TOP_K or 4)`。
3. 发送 retrieval。
4. 发送 generate step。
5. 构造带知识库上下文的 prompt。
6. 调用 llm.stream。
7. 输出 content_chunk。
8. 输出 final。

### Paper 工作流

`_stream_paper_workflow()`：

1. Paper Agent retrieve step。
2. 调用 `collect_paper_material_with_model()`。
3. 发送 retrieval。
4. Paper Agent generate step。
5. 调用模型生成阅读报告文本。
6. Paper Agent archive step。
7. 调用 `finalize_paper_workflow()`。
8. Paper Agent generate step。
9. 调用模型生成完成摘要。
10. 输出 final。

异常时：

- 增加 failed step。
- 尝试用模型生成失败说明。
- 如果模型也失败，抛出 `_model_error()`。

### build_agent_prompt()

为 Data/Experiment/Explain/Report Agent 构造系统提示词。

所有 agent 都强调：

- 中文回答。
- 直接可执行。
- 由当前模型生成，不能使用后端固定模板。

### format_context()

将 RAG 检索结果拼接为 prompt 上下文。

### 维护注意事项

- 新增 route 时要同步前端 `formatters.js` 的 agent/action 文案。
- `_route()` 的判断顺序会影响用户体验。
- Agent 输出事件必须保持前端可解析字段。
- 当前 chat 失败时不提供固定模板回答，这是为了保证内容由模型生成。

## 3.6 backend/papers.py

### 文件职责

实现论文发现、下载、阅读报告生成、归档、数据集发现和 RAG 入库辅助。

### 常量

| 常量 | 含义 |
| --- | --- |
| `PAPER_CACHE_DIR` | `artifacts/papers` |
| `SUPPORTED_PAPER_SUFFIXES` | 支持论文/文档后缀 |
| `DATA_URL_SUFFIXES` | 可识别数据 URL 后缀 |
| `USER_AGENT` | HTTP User-Agent |
| `MAX_DOWNLOAD_BYTES` | 最大下载 120 MB |
| `MODEL_HINT_LIMIT` | 模型候选上限 |

### 数据类

#### PaperCandidate

字段：

- title。
- authors。
- year。
- abstract。
- source。
- url。
- pdf_url。
- doi。

方法：

- `citation()`

#### DownloadedFile

字段：

- name。
- path。
- url。
- suffix。

#### PaperMaterial

字段：

- query。
- search_query。
- candidate。
- workspace。
- paper_path。
- paper_text。
- data_urls。
- data_files。
- search_results。

方法：

- `model_context(max_text_chars=18000)`

#### PaperWorkflowResult

字段：

- title。
- paper_file。
- report_file。
- metadata_file。
- rag_refreshed。
- datasets。
- dataset_errors。

方法：

- `model_context()`

### collect_paper_material_with_model()

流程：

1. 从用户请求提取检索式。
2. 如果有 LLM，调用 `_model_paper_hints()` 生成搜索提示。
3. 调用 `find_paper_candidates()`。
4. 排序和过滤候选。
5. 如果没有候选，尝试使用模型给出的 candidates。
6. 遍历候选，创建 workspace。
7. 调用 `_download_candidate_assets()`。
8. 抽取正文。
9. 发现数据 URL。
10. 下载数据 URL。
11. 返回 `PaperMaterial`。

失败时聚合前几个错误并抛 `PaperWorkflowError`。

### find_paper_candidates()

候选来源：

- `_direct_url_candidates()`
- `_model_candidates()`
- `_semantic_scholar_search()`
- `_arxiv_search()`
- `_crossref_search()`

最终去重并最多返回 8 个。

### finalize_paper_workflow()

流程：

1. 导入 `backend.services`。
2. 确定 RAG 归档目录。
3. 复制论文原文。
4. 调用 `_write_docx_report()` 写 Word 阅读报告。
5. 尝试将 data_files 入库。
6. 调用 `services.refresh_knowledge_base()`。
7. 写 `paper_workflow.json`。
8. 返回 `PaperWorkflowResult`。

### discover_data_urls()

从文本中匹配 HTTP/HTTPS URL，按后缀筛选数据文件。

### build_paper_report_messages()

构造阅读报告 prompt。要求模型覆盖：

- 题名与引用。
- 研究问题。
- 数据与被试。
- 方法流程。
- 模型/统计方法。
- 主要结果。
- 创新点。
- 局限。
- 可复现建议。
- 与平台关系。
- 是否发现可入库数据。

### _safe_extract_zip()

用于论文下载到 zip 时解压，包含路径穿越检查。

### _ingest_dataset_file()

把论文工作流中发现的数据文件复制到 `artifacts/datasets/`，解析摘要并写入 datasets 表。

### 维护注意事项

- 论文检索依赖网络，不应在无网络测试中直接调用外部 API。
- 下载大小限制是安全边界，不建议随意增大。
- Word 报告依赖 `python-docx`。
- 新增论文来源时应在 `find_paper_candidates()` 中合并，并保持去重。

## 4. 核心领域程序说明

## 4.1 fnirs_core/data.py

### 文件职责

提供 fNIRS 数据加载、标准数据结构、摘要和 demo 数据生成。

### 支持后缀

```python
SUPPORTED_DATA_SUFFIXES = {".snirf", ".nirs", ".csv", ".mat", ".zip", ".json"}
```

### NIRSDataError

数据解析异常基类。

### DatasetSample

当前定义但未作为主流程使用，预留给后续样本级数据结构。

字段：

- signal。
- label。
- subject_id。
- metadata。

### NIRSData

标准 fNIRS 数据对象。

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `raw_data` | `np.ndarray` | 形状 `(n_channels, n_samples)` |
| `sampling_rate` | `float` | 采样率 |
| `channel_names` | `list[str]` | 通道名 |
| `events` | `np.ndarray` | 形状 `(n_events, 3)` |
| `hbo` | `np.ndarray | None` | HbO 信号 |
| `hbr` | `np.ndarray | None` | HbR 信号 |
| `metadata` | `dict` | 元数据 |
| `event_id_map` | `dict[int, str]` | 标签映射 |
| `subject_ids` | `np.ndarray | None` | 事件级 subject |

#### __post_init__()

校验：

- `raw_data` 必须为二维。
- `sampling_rate` 必须大于 0。
- `channel_names` 长度必须等于通道数。
- `events` 必须规范化。
- `hbo/hbr` 如果存在，形状必须和 raw_data 一致。

#### summary()

返回数据摘要：

- 通道数。
- 采样点数。
- 时长。
- 采样率。
- 通道名。
- 是否含 HbO/HbR。
- 事件数。
- 标签分布。
- subject 数。
- source_format。

#### get_epochs()

参数：

- `tmin`
- `tmax`
- `baseline`
- `include_hbo_hbr`

输出字典：

- `data`
- `labels`
- `events`
- `groups`
- `times`
- `sampling_rate`
- `channel_names`
- 可选 `hbo`
- 可选 `hbr`

逻辑：

1. 检查事件存在。
2. 计算 epoch 起止采样点。
3. 跳过越界事件。
4. 做 baseline 校正。
5. 收集 labels 和 groups。
6. 堆叠输出。

### load_fNIRS_data()

根据后缀分发：

- `.csv` -> `_load_csv()`
- `.json` -> `_load_json()`
- `.zip` -> `_load_zip()`
- `.mat` / `.nirs` -> `_load_mat()`
- `.snirf` -> `_load_snirf()`

### summarize_file()

加载数据并调用 `summary()`。

### make_demo_nirs_data()

生成合成 demo 数据：

- 默认 6 个 subject。
- 每个 subject 6 个 trial。
- 16 通道。
- 每 trial 160 时间点。
- 10 Hz。
- 标签 alternating control/task。
- 生成 HbO/HbR。

### CSV 解析

`_load_csv()`：

1. UTF-8-SIG 读取。
2. sniff delimiter。
3. fallback 到逗号。
4. 判断 header。
5. 识别 label、subject、time 列。
6. 其他数值列作为通道。
7. 推断采样率。
8. 根据 label 列构造 events。

事件构造方式：

- 每一行对应一个事件。
- onset 为行 index。
- event code 为标签排序后的整数编码。

### JSON 解析

要求 JSON 至少包含：

```json
{
  "raw_data": [[...]],
  "sampling_rate": 10.0,
  "channel_names": [],
  "events": []
}
```

如果 `raw_data` 行数大于列数，会自动转置。

### Zip 解析

`_load_zip()`：

1. 创建临时目录。
2. 安全解压。
3. 搜索支持的数据文件。
4. 优先尝试 `.snirf/.nirs/.mat`。
5. 如果某候选失败，继续尝试下一个。
6. 全部失败时抛出聚合错误。

### MAT/NIRS 解析

`_load_mat()` 依赖 SciPy。

逻辑：

1. `loadmat(..., simplify_cells=True)`。
2. 查找 signal。
3. 规范化 signal 矩阵为 `(channels, samples)`。
4. 查找采样率。
5. 查找事件。
6. 生成默认通道名。

signal 查找：

- 优先候选字段。
- 支持嵌套 dict。
- 支持 scipy struct。
- 支持 object array。
- 用 `_signal_score()` 给候选排序。

### SNIRF 解析

`_load_snirf()` 依赖 MNE。

流程：

1. `mne.io.read_raw_snirf()`。
2. `raw.get_data()`。
3. `mne.events_from_annotations()`。
4. 转为 `NIRSData`。

### 维护注意事项

- 新增格式时必须输出标准 `NIRSData`。
- 数据矩阵方向统一为 `(channels, samples)`。
- 事件矩阵统一为 `(n_events, 3)`。
- zip 解压安全检查不能删除。
- 无法确定 subject 时，后续实验可能无法真正 subject-wise。

## 4.2 fnirs_core/preprocessing.py

### 文件职责

提供 NumPy 预处理 pipeline。

### PreprocessConfig

字段：

| 字段 | 默认值 |
| --- | --- |
| `apply_optical_density` | `True` |
| `apply_beer_lambert` | `True` |
| `apply_tddr` | `True` |
| `bandpass_low` | `0.01` |
| `bandpass_high` | `0.2` |
| `baseline_start` | `-2.0` |
| `baseline_end` | `0.0` |
| `epoch_start` | `-2.0` |
| `epoch_end` | `10.0` |
| `include_hbo_hbr` | `True` |

### PreprocessingResult

字段：

- `epochs`
- `labels`
- `groups`
- `times`
- `channel_names`
- `band_names`
- `summary`
- `config`

### recommend_preprocessing()

根据数据摘要推荐配置：

- 如果已有 HbO，则不默认做光密度和 Beer-Lambert。
- 根据采样率设置高频 cutoff。
- 默认启用 TDDR。

### PreprocessingPipeline.run()

流程：

1. 复制 raw signal。
2. 可选 `optical_density()`。
3. 可选 `beer_lambert()`。
4. 可选 `tddr()`。
5. 可选 `bandpass_filter()`。
6. 创建 working `NIRSData`。
7. 如果无事件，调用 `_fallback_whole_recording()`。
8. 否则调用 `working.get_epochs()`。
9. `_stack_bands()`。
10. 返回 `PreprocessingResult`。

### optical_density()

计算：

```python
-np.log(clipped / baseline)
```

其中 baseline 为每通道均值，最小值裁剪到 `1e-6`。

### beer_lambert()

当前实现为按通道标准差缩放。

注意：这是轻量近似，不是完整物理建模。

### tddr()

逻辑：

- 计算时间导数。
- 用 MAD 估计导数异常。
- 超阈值点替换为通道导数中位数。
- 重新累积得到信号。

### bandpass_filter()

优先 SciPy：

- `butter(3, cutoff, btype=...)`
- `filtfilt()`

失败时 fallback 到 `_fft_bandpass()`。

### _fallback_whole_recording()

无事件时使用。

逻辑：

- 窗口长度约为 `min(max(sampling_rate * 10, 20), n_samples)`。
- 步长为窗口一半。
- 不足窗口则补零。
- labels 交替 0/1。
- groups 使用 `metadata.subject_id` 或 `subject_0`。
- 如果只有一个 epoch，则复制一个并赋 label 1。
- summary 写 warning。

### 维护注意事项

- 输出 shape 必须保持 `(epochs, bands, channels, times)`。
- 无事件 fallback 必须保留 warning。
- 新增预处理方法要写入 config 和 summary，保证可追踪。

## 4.3 fnirs_core/models.py

### 文件职责

定义模型族、模型配置和当前轻量分类器。

### MODEL_FAMILIES

```python
{
  "fnirs-eegnet": "fNIRS-EEGNet",
  "cnn-lstm": "CNN-LSTM",
  "tcn": "Temporal Convolution Network",
  "graph-tcn": "Graph-TCN",
  "hybrid-3d-cnn": "Hybrid 3D CNN",
}
```

### ModelConfig

字段：

- `model_family`
- `learning_rate`
- `weight_decay`
- `batch_size`
- `max_epochs`
- `seed`
- `extra_params`

#### normalized_family()

把 `_` 替换为 `-`，小写后检查是否支持。

不支持时抛 `ModelError`。

### PrototypeClassifier

当前实验模型。

#### fit(x, y)

流程：

1. 提取特征。
2. 计算 feature mean/std。
3. 标准化。
4. 获取类别。
5. 每类计算质心。

#### predict_proba(x)

流程：

1. 检查已 fit。
2. 提取并标准化特征。
3. 计算到每个类别质心的 L2 距离。
4. 使用负距离作为 logits。
5. softmax 得到概率。

#### predict(x)

返回最大概率对应类别。

#### explain_features(x, y=None)

基于绝对值激活计算：

- channel_importance。
- time_importance。
- band_importance。

#### _features(x)

要求 x 形状：

```text
(epochs, bands, channels, times)
```

基础特征：

- mean。
- std。
- peak。
- trough。
- temporal_slope。

模型族附加：

- `tcn` 和 `graph-tcn`：diff mean。
- `hybrid-3d-cnn`：energy。

### create_model(config)

根据 `ModelConfig` 创建 `PrototypeClassifier`。

### model_registry()

返回模型族列表，用于后续 API 或前端扩展。

### 维护注意事项

- 替换真实模型时尽量保持 fit/predict/predict_proba/explain_features 接口。
- 真实模型 checkpoint 需要更新 experiments.py 的保存逻辑。
- 不要让模型层依赖 FastAPI。

## 4.4 fnirs_core/experiments.py

### 文件职责

执行完整实验流程。

### ExperimentConfig

字段：

- `name`
- `dataset_id`
- `dataset_path`
- `preprocessing`
- `model`
- `validation_strategy`
- `num_folds`
- `seed`
- `output_dir`

### FoldResult

字段：

- `fold_name`
- `train_size`
- `val_size`
- `accuracy`
- `labels`
- `predictions`
- `subject_ids`

### ExperimentResult

字段：

- `experiment_id`
- `name`
- `status`
- `metrics`
- `folds`
- `output_dir`
- `checkpoint_path`
- `config`
- `preprocessing_summary`

### run_experiment()

输入：

- `experiment_id`
- `ExperimentConfig` 或 dict。
- 可选 progress callback。

流程：

1. 规范 config。
2. 创建 RNG。
3. progress 0.05：加载数据。
4. 有 `dataset_path` 则加载真实数据，否则生成 demo。
5. progress 0.15：预处理。
6. 得到 x、y、groups。
7. 检查至少两个 label。
8. 构建 folds。
9. 构建 model config。
10. 每个 fold 创建新模型。
11. fit 训练集。
12. predict 验证集。
13. 计算 fold accuracy。
14. 汇总 labels/predictions。
15. progress 0.9：保存结果。
16. 写 `prototype_model.json`。
17. 计算 metrics。
18. 写 `result.json`。
19. progress 1.0。
20. 返回 `ExperimentResult`。

### build_subject_folds()

输入：

- groups。
- strategy。
- num_folds。
- seed。

逻辑：

- 如果 unique groups 少于 2：
  - 样本少于 2 则返回空。
  - 否则使用 holdout。
- 如果 strategy 是 `loso`：
  - 每个 subject 作为验证集。
- 否则：
  - shuffle subjects。
  - 按 subject 分成 K 份。
  - 生成 GroupKFold。

### build_metrics()

计算：

- accuracy。
- classes。
- confusion_matrix。
- per_class precision/recall/f1。
- n_samples。

### 维护注意事项

- 实验输出路径必须在受管 output_dir 下。
- 新增指标时要同步报告和前端结果展示。
- 真实模型训练可能耗时很长，应配合任务队列改造。

## 4.5 fnirs_core/explain.py

### 文件职责

生成实验解释结果。

### ExplanationResult

字段：

- `experiment_id`
- `method`
- `channel_importance`
- `time_importance`
- `band_importance`
- `top_channels`
- `output_path`

### explain_experiment()

输入：

- experiment_id。
- experiment_config。
- output_dir。

流程：

1. 根据 dataset_path 加载真实数据或 demo。
2. 根据 preprocessing config 运行 pipeline。
3. 创建模型。
4. 用全部数据 fit。
5. 调用 `model.explain_features()`。
6. channel importance 归一化。
7. 取前 8 个通道。
8. 写 `explanation.json`。
9. 返回 `ExplanationResult`。

### 维护注意事项

- 当前解释会重新训练原型模型，不读取 checkpoint 中的质心。
- 后续真实模型应读取训练后的 checkpoint。
- 解释结果不能直接解释为因果。

## 4.6 fnirs_core/reports.py

### 文件职责

生成 Markdown 实验报告。

### generate_experiment_report()

输入：

- `experiment`
- `result`
- `explanation`
- `output_dir`

输出：

```text
{output_dir}/report.md
```

报告结构：

- 标题。
- 实验概览。
- 关键指标。
- Fold 结果。
- 可解释性摘要。
- 配置快照。
- 边界说明。

### 维护注意事项

- 报告文件名当前固定为 `report.md`。
- 新增报告字段时同步 Results 页面。
- 如果要支持 PDF/DOCX，建议新增函数，不要破坏现有 Markdown 下载。

## 4.7 fnirs_core/knowledge.py

### 文件职责

实现本地 RAG 知识库。

### 支持文档

```python
SUPPORTED_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".pdf", ".docx", ".doc"}
```

### 文本读取

`_read_text(path)`：

- Markdown/Text：直接 UTF-8 读取。
- PDF：`pypdf.PdfReader`。
- DOCX：`python-docx.Document`。
- DOC：尝试多编码 decode。

### extract_text_from_document(path)

支持 zip：

1. 安全解压。
2. 找第一个支持文档。
3. 递归抽取文本。

### document_id_from_source(source)

返回 source 的 SHA1 前 16 位。

### _chunk_text()

按段落和滑窗切分文本。

参数：

- chunk_size。
- chunk_overlap。

### _hash_embedding()

本地 embedding fallback。

逻辑：

1. 正规化文本。
2. 提取 token。
3. 长 token 切 3-gram。
4. 用 blake2b hash 映射到向量维度。
5. L2 归一化。

### 数据类

#### DocumentChunk

- chunk_id。
- source。
- title。
- content。
- order。
- enabled。

#### KnowledgeDocument

- id。
- source。
- title。
- path。
- suffix。
- size_chars。
- chunk_count。
- updated_at。
- managed。

#### SearchResult

- source。
- title。
- content。
- score。
- order。

`to_dict()` 会生成 snippet，最多 320 字。

#### KnowledgeStats

- total_documents。
- total_chunks。
- source_files。
- source_roots。
- vector_store_path。
- embedding_model。
- embedding_dim。
- index_updated_at。

### OllamaEmbeddingClient

调用：

```text
POST {base_url}/api/embed
```

支持 batch。

返回 NumPy float32 matrix。

### KnowledgeBase

#### 构造参数

- sources。
- chunk_size。
- chunk_overlap。
- vector_store_dir。
- embedding_model。
- embedding_base_url。
- embedding_dim。
- managed_roots。
- allow_embedding_fallback。

#### refresh()

流程：

1. 发现源文件。
2. 加载旧 metadata，保留 enabled 状态。
3. 抽取文本。
4. chunk 切分。
5. 构造 DocumentChunk。
6. embedding。
7. 更新索引时间。
8. 持久化向量库。

#### stats()

确保加载后返回统计。

#### list_documents()

按 source 聚合 chunks，返回文档列表。

#### find_document()

按 document id 查找。

#### get_document()

返回文档对象和全文。

#### get_document_chunks()

返回文档对应 chunks。

#### set_chunk_enabled()

修改 chunk enabled 并持久化 metadata。

#### search()

流程：

1. query 为空则返回空。
2. ensure loaded。
3. embed query。
4. 如果维度或数量不匹配则 refresh。
5. 点积打分。
6. 降序排序。
7. 跳过 disabled chunk。
8. 返回 top_k。

#### _metadata_matches_sources()

判断现有索引是否可复用。

检查：

- configured embedding model。
- embedding base URL。
- active embedding model。
- source roots。
- source files。
- updated_at。
- size_chars。

### build_default_knowledge_base()

默认 sources：

- `knowledge/base`
- `knowledge/uploads/extracted`

默认 vector store：

- `artifacts/vector_store`

managed roots：

- `knowledge/uploads/extracted`

### 维护注意事项

- 新增 source root 时会导致旧索引失效。
- 修改 chunk 参数可能导致 chunk_id/order 变化。
- embedding fallback 模型名会写入 metadata。
- chunk enabled 状态依赖 chunk_id，文档内容大幅变化可能无法完全保留旧状态。

## 5. 前端程序说明

## 5.1 frontend/src/main.js

### 文件职责

Vue 应用入口。

典型职责：

- 创建 Vue app。
- 注册 Pinia。
- 注册 router。
- 引入全局样式。
- 挂载到 `#app`。

## 5.2 frontend/src/App.vue

### 文件职责

应用 shell。

包含：

- 左侧 sidebar。
- 品牌区。
- 导航菜单。
- 本地模型状态。
- 主内容 `<RouterView />`。
- `FeedbackHost`。

### 导航项

| 路由 | 文案 |
| --- | --- |
| `/` | 总览 |
| `/chat` | 对话 |
| `/knowledge` | 知识库 |
| `/data` | 数据 |
| `/experiments` | 实验 |
| `/results` | 结果 |
| `/settings` | 设置 |

### 生命周期

`onMounted()` 调用：

```js
store.refreshAll();
```

## 5.3 frontend/src/router/index.js

### 文件职责

定义 Vue Router 路由。

使用：

```js
createWebHistory()
```

每个 route 直接绑定一个 view 组件。

## 5.4 frontend/src/api.js

### 文件职责

统一前端 API 工具。

### Axios 实例

```js
export const api = axios.create({
  baseURL: "/api",
  timeout: 600000,
});
```

### getApiError()

优先返回：

1. `error.response.data.detail`
2. `error.message`
3. fallback

### parseSseBlock()

解析 SSE block：

- `event:` 行决定事件名。
- `data:` 行合并后 JSON.parse。

返回：

```js
{ event, ...payload }
```

### streamChat()

输入：

- message。
- sessionId。
- onEvent callback。

流程：

1. fetch `/api/chat/stream`。
2. 检查 response.ok。
3. 读取 response.body reader。
4. 使用 UTF-8 decoder。
5. 按 `\n\n` 拆 SSE block。
6. 每个 block 调用 onEvent。
7. 处理 trailing buffer。

### 维护注意事项

- SSE payload 必须是 JSON。
- 如果后端 event 名变化，ChatView 需要同步修改。

## 5.5 frontend/src/stores/platform.js

### 文件职责

主业务状态 store。

### State 说明

| 字段 | 说明 |
| --- | --- |
| `health` | 后端健康状态 |
| `knowledge` | 知识库统计 |
| `documents` | 知识文档列表 |
| `activeDocument` | 当前知识文档详情 |
| `datasets` | 数据集列表 |
| `experiments` | 实验列表 |
| `jobs` | 任务列表 |
| `activeExperimentId` | 当前实验 ID |
| `sessionId` | 当前对话会话 ID |
| `chatSessions` | 历史会话 |
| `messages` | 当前对话消息 |
| `sources` | 当前检索来源 |
| `agentSteps` | 当前智能体步骤 |
| `timeline` | 简单状态时间线 |
| `currentStatus` | 当前状态文字 |
| loading flags | 各类加载状态 |

### Getters

#### busy

任意 loading flag 为 true 即 busy。

#### activeExperiment

优先返回 `activeExperimentId` 对应实验，否则返回第一个实验。

### Actions 分组

#### 全局刷新

- `refreshAll()`

并发刷新：

- health。
- knowledge。
- documents。
- datasets。
- experiments。
- jobs。
- chatSessions。

使用 `Promise.allSettled()`，单个请求失败不会阻塞其他刷新。

#### 设置

- `refreshHealth()`
- `updateRuntimeConfig(payload)`

#### 知识库

- `refreshKnowledge()`
- `refreshDocuments()`
- `rebuildKnowledge()`
- `uploadKnowledge(file, options)`
- `createKnowledge(payload)`
- `updateKnowledge(documentId, payload)`
- `loadKnowledgeDocument(documentId)`
- `toggleChunk(documentId, order, enabled)`
- `deleteKnowledge(documentId)`

#### 对话会话

- `refreshChatSessions()`
- `loadChatSession(sessionId)`
- `deleteChatSession(sessionId)`

删除当前会话时会重置本地 messages、sources、agentSteps。

#### 数据集

- `refreshDatasets()`
- `uploadDataset(file, options)`
- `deleteDataset(datasetId)`

删除数据集后同时刷新 datasets 和 experiments。

#### 实验

- `refreshExperiments()`
- `createExperiment(payload)`
- `runExperiment(experimentId)`
- `explainExperiment(experimentId)`
- `deleteExperiment(experimentId)`
- `deleteExperimentResults(experimentId)`
- `refreshExperimentResults(experimentId)`

`refreshExperiments()` 会维护 activeExperimentId：

- 如果当前 active 不存在，置空。
- 如果为空且有实验，选第一个实验。

#### Jobs

- `refreshJobs()`

### 维护注意事项

- 新增 API 时优先加到 store action，页面不要重复写请求逻辑。
- 删除操作后要刷新相关列表。
- loading flag 要在 finally 中恢复。

## 5.6 frontend/src/stores/feedback.js

### 文件职责

管理用户反馈。

### State

- `toasts`
- `uploads`
- `confirmationDialog`

### Actions

#### Toast

- `notify()`
- `success()`
- `error()`
- `info()`
- `dismissToast()`

#### Confirmation

- `requestConfirmation()`
- `resolveConfirmation()`

内部使用 `confirmationResolver` 保存 Promise resolver。

#### Upload

- `startUpload()`
- `updateUpload()`
- `finishUpload()`
- `failUpload()`
- `removeUpload()`

上传百分比通过 `clampPercent()` 限制在 0 到 100。

## 5.7 frontend/src/formatters.js

### 文件职责

展示层格式化和中文映射。

### 映射表

- `STATUS_TEXT`
- `JOB_KIND_TEXT`
- `AGENT_TEXT`
- `ACTION_TEXT`
- `MODEL_TEXT`
- `VALIDATION_TEXT`
- `SUMMARY_KEY_TEXT`

### 主要函数

- `statusText()`
- `jobKindText()`
- `agentText()`
- `actionText()`
- `modelText()`
- `localModelSummary()`
- `configuredText()`
- `validationText()`
- `foldNameText()`
- `displayText()`
- `yesNo()`
- `formatDate()`
- `formatJsonChinese()`

### displayText()

特殊处理 `"No events found"`：

会显示中文警告，强调 placeholder labels 仅用于验证本地流程，不能作为科研结论。

### 维护注意事项

- 后端新增状态码、agent、action 后要补映射。
- 不建议后端直接返回中文状态码，否则会影响程序稳定性。

## 5.8 frontend/src/views/DashboardView.vue

### 文件职责

总览页。

展示：

- 本地模型状态。
- 知识文档数。
- 知识片段数。
- 数据集数量。
- 实验数。
- 任务数。
- 最近实验。
- JobMonitor。

交互：

- 刷新按钮调用 `store.refreshAll()`。

## 5.9 frontend/src/views/ChatView.vue

### 文件职责

智能体对话页。

包含：

- 消息列表。
- 输入框。
- 新对话按钮。
- `RightRail`。

### send()

流程：

1. 校验输入非空且不在 loading。
2. 清空输入。
3. 设置 loading。
4. 清空 sources 和 agentSteps。
5. 设置 currentStatus。
6. 追加 user 消息。
7. 追加空 assistant 消息。
8. 调用 `streamChat()`。
9. 根据 event 更新状态：
   - status -> currentStatus/timeline。
   - agent_step -> agentSteps。
   - retrieval -> sources。
   - content_chunk -> 追加 assistant 内容。
   - final -> 覆盖最终 assistant 内容、sources、agentSteps。
   - session -> 更新 sessionId 并刷新会话。
   - error -> 抛错。
10. catch 中写失败消息。
11. finally 恢复 loading 和状态。

### newChat()

清空 sessionId、sources、agentSteps，并重置 messages。

### 维护注意事项

- assistant 消息通过 index 原地更新。
- 如果后端 SSE event 增加，需要在 send() 中处理。

## 5.10 frontend/src/components/RightRail.vue

### 文件职责

Chat 页右侧信息栏。

展示：

- 历史会话。
- 智能体步骤。
- 检索来源。

交互：

- 刷新会话。
- 加载会话。
- 删除会话。

删除前使用 `feedback.requestConfirmation()`。

## 5.11 frontend/src/views/KnowledgeView.vue

### 文件职责

知识库管理页。

展示：

- 文档数。
- chunk 数。
- embedding 模型。
- 文档列表。
- 文档编辑区。
- chunk 列表。

交互：

- 上传文档。
- 重建索引。
- 刷新文档。
- 点击文档加载详情。
- 新建文档。
- 编辑托管文档。
- 删除托管文档。
- 启停 chunk。

### isReadOnly

```js
Boolean(store.activeDocument && !store.activeDocument.managed)
```

内置文档只读。

### upload()

使用反馈 store 创建上传进度，调用 `store.uploadKnowledge()`。

### submitDocument()

如果 activeDocument 是托管文档：

- 调用 update。

否则：

- 调用 create。

### watch activeDocument

切换文档时同步表单 title/content。

content 会去掉首个 Markdown H1：

```js
replace(/^# .+?\n\n/, "")
```

## 5.12 frontend/src/views/DataView.vue

### 文件职责

数据集管理页。

展示：

- 数据集列表。
- 最近数据集摘要。
- 通道数、采样点、事件数、被试数、采样率、时长。
- 中文化 JSON 摘要。

交互：

- 上传数据集。
- 删除数据集。
- 刷新数据集。

删除提示说明：

- 关联实验会保留。
- 关联实验会解除数据集引用。

## 5.13 frontend/src/views/ExperimentsView.vue

### 文件职责

实验编排页。

展示：

- 实验配置表单。
- 实验列表。
- 当前任务队列。

表单字段：

- 实验名称。
- 数据集。
- 模型族。
- 验证策略。
- 折数。
- 随机种子。

交互：

- 创建实验。
- 运行实验。
- 生成解释。
- 删除实验。
- 刷新实验。

删除实验提示：

- 实验结果、解释、报告和关联任务也会删除。

## 5.14 frontend/src/views/ResultsView.vue

### 文件职责

结果分析页。

展示：

- 实验选择列表。
- 核心指标。
- metrics JSON。
- fold 结果。
- top channels。
- 报告下载链接。

交互：

- 刷新结果。
- 生成解释。
- 删除当前实验结果。
- 删除指定实验结果。
- 下载报告。

报告下载链接：

```html
href="/api/reports/{active.id}/download"
```

## 5.15 frontend/src/views/SettingsView.vue

### 文件职责

运行设置页。

展示：

- Ollama 地址。
- 对话模型。
- 向量模型。
- 数据库路径。
- 向量库路径。
- API 状态。
- 模型服务状态。
- 对话模型状态。
- 向量模型状态。

交互：

- 刷新健康状态。
- 保存 chat_model 和 embedding_model。

### canSave

只有当：

- 不在 saving。
- 两个模型字段都非空。
- 任一字段与当前 health 不同。

才允许保存。

## 5.16 frontend/src/components/JobMonitor.vue

### 文件职责

展示任务队列。

字段：

- job kind。
- message。
- progress bar。
- status badge。

交互：

- 刷新 jobs。

## 5.17 frontend/src/components/MetricTile.vue

### 文件职责

复用指标展示组件。

用于 Dashboard、Knowledge、Data、Results、Settings。

## 5.18 frontend/src/components/FeedbackHost.vue

### 文件职责

全局反馈容器。

通常包含：

- toast 列表。
- 上传进度列表。
- 确认弹窗。

由 `feedback` store 驱动。

## 6. 测试程序说明

## 6.1 tests/test_core.py

### 测试范围

覆盖 `fnirs_core` 和论文归档关键路径。

### 主要测试

#### test_knowledge_base_fallback_search

验证：

- KnowledgeBase 可构建。
- embedding fallback 可检索。
- 查询 LOSO 能找到相关 chunk。

#### test_knowledge_chunks_preserve_overlap

验证 chunk overlap 保留。

#### test_knowledge_index_invalidates_when_embedding_model_changes

验证 embedding 模型改变时 metadata 不匹配。

#### test_knowledge_zip_rejects_unsafe_paths

验证知识 zip 路径穿越会被拒绝。

#### test_preprocessing_demo_epochs

验证 demo 数据可预处理并得到 4D epochs。

#### test_no_event_recording_fallback_produces_two_placeholder_labels

验证无事件 fallback 会生成 0/1 placeholder labels，并带 warning。

#### test_quick_experiment_runs

验证 demo quick experiment 可跑通。

#### test_no_event_csv_experiment_runs_with_fallback

验证无事件 CSV 也能通过 fallback 跑通实验。

#### test_ragged_csv_dataset_summary_does_not_crash

验证不规整 CSV 不崩溃。

#### test_mat_struct_signal_dataset_summary_does_not_crash

验证嵌套 MAT 结构体信号可解析。

#### test_zip_skips_unparseable_candidate_and_loads_next_file

验证 zip 中坏文件可跳过，继续加载好文件。

#### test_same_paper_reuses_rag_directory

验证相同论文复用同一 RAG 目录。

#### test_data_zip_rejects_unsafe_paths

验证数据 zip 路径穿越会被拒绝。

## 6.2 tests/test_api.py

### 测试范围

覆盖 API、会话、删除、运行设置、多智能体路由。

### 主要测试

#### test_chat_session_history_is_readable

验证：

- POST /api/chat 保存会话。
- GET /api/chat/sessions 可列出。
- GET /api/chat/sessions/{id} 可读取。
- DELETE 可删除。

#### test_chat_session_agent_steps_keep_latest_turn_only

验证多轮会话只展示最新轮 agent steps。

#### test_dataset_experiment_and_result_deletes

验证：

- 数据集上传。
- 实验创建。
- 删除结果清空 result/explanation/report。
- 删除数据集解除实验关联。
- 删除实验清理记录。

#### test_runtime_model_settings_are_persisted

验证 `/api/settings` 持久化模型名。

#### test_identity_question_routes_to_basic_chat

验证身份类问题 route 到 chat。

#### test_training_model_question_still_routes_to_experiment

验证训练模型问题 route 到 experiment。

#### test_fnirs_definition_routes_to_rag_not_data_agent

验证 fNIRS 定义问题 route 到 RAG。

#### test_paper_request_routes_to_paper_workflow

验证论文请求 route 到 paper，并调用报告和完成摘要模型 prompt。

### 测试隔离模式

测试大量使用 monkeypatch 改写目录，避免真实 workspace 被污染。

维护新增测试时应保持这种隔离方式。

## 7. 程序调用链速查

### 7.1 上传知识文档

```text
KnowledgeView.upload
  -> platform.uploadKnowledge
  -> POST /api/knowledge/upload
  -> main.knowledge_upload
  -> services.ingest_knowledge_file
  -> knowledge.extract_text_from_document
  -> services.refresh_knowledge_base
  -> KnowledgeBase.refresh
```

### 7.2 RAG 对话

```text
ChatView.send
  -> streamChat
  -> POST /api/chat/stream
  -> main._chat_stream
  -> services.get_orchestrator
  -> MultiAgentOrchestrator.stream
  -> KnowledgeBase.search
  -> OllamaChatClient.stream
  -> services.save_chat_session
```

### 7.3 上传数据集

```text
DataView.upload
  -> platform.uploadDataset
  -> POST /api/datasets/upload
  -> main.dataset_upload
  -> services.upload_dataset
  -> data.summarize_file
  -> data.load_fNIRS_data
  -> db INSERT datasets
```

### 7.4 创建实验

```text
ExperimentsView.create
  -> platform.createExperiment
  -> POST /api/experiments
  -> main.experiment_create
  -> services.create_experiment
  -> db INSERT experiments
```

### 7.5 运行实验

```text
ExperimentsView.runActive
  -> platform.runExperiment
  -> POST /api/experiments/{id}/run
  -> services.run_experiment_job
  -> services.create_job
  -> worker thread
  -> experiments.run_experiment
  -> data.load_fNIRS_data or make_demo_nirs_data
  -> PreprocessingPipeline.run
  -> build_subject_folds
  -> create_model
  -> PrototypeClassifier.fit/predict
  -> build_metrics
  -> write result.json
  -> db UPDATE experiments/jobs
```

### 7.6 生成解释

```text
ResultsView.explain or ExperimentsView.explainActive
  -> platform.explainExperiment
  -> POST /api/experiments/{id}/explain
  -> services.explain_experiment_job
  -> worker thread
  -> explain.explain_experiment
  -> write explanation.json
  -> db UPDATE experiments/jobs
```

### 7.7 下载报告

```text
ResultsView download link
  -> GET /api/reports/{id}/download
  -> main.report_download
  -> services.generate_report
  -> reports.generate_experiment_report
  -> FileResponse(report.md)
```

### 7.8 论文工作流

```text
ChatView.send
  -> streamChat
  -> MultiAgentOrchestrator._stream_paper_workflow
  -> collect_paper_material_with_model
  -> find_paper_candidates
  -> download / extract
  -> llm build_paper_report_messages
  -> finalize_paper_workflow
  -> write reading_report.docx
  -> refresh_knowledge_base
  -> llm build_paper_final_messages
```

## 8. 维护和扩展建议

### 8.1 新增 API

推荐步骤：

1. 在 `backend/schemas.py` 定义请求/响应模型。
2. 在 `backend/services.py` 写业务函数。
3. 在 `backend/main.py` 添加 route。
4. 在 `frontend/src/stores/platform.js` 添加 action。
5. 在页面组件中调用 store action。
6. 添加 API 测试。

### 8.2 新增数据格式

推荐步骤：

1. 在 `SUPPORTED_DATA_SUFFIXES` 添加后缀。
2. 在 `load_fNIRS_data()` 添加分支。
3. 实现 `_load_xxx()`。
4. 输出标准 `NIRSData`。
5. 添加解析成功和异常测试。

### 8.3 替换真实模型训练

推荐步骤：

1. 保留 `ModelConfig`。
2. 保留 `create_model()` 或新增 registry。
3. 实现真实模型类的 fit/predict/predict_proba。
4. 更新 checkpoint 保存。
5. 更新解释读取 checkpoint 的逻辑。
6. 扩展 job progress。
7. 添加耗时任务测试或集成测试。

### 8.4 增加前端页面

推荐步骤：

1. 新增 view。
2. 在 router 中注册。
3. 在 App.vue navItems 添加入口。
4. 优先复用 `platform` 和 `feedback` store。
5. 状态码和文案放入 `formatters.js`。

### 8.5 增加后台任务类型

推荐步骤：

1. 设计 job kind。
2. 在 `services.create_job()` 调用处创建 payload。
3. 写 worker thread 或未来任务队列。
4. 在 `formatters.js` 添加 `JOB_KIND_TEXT`。
5. 在测试中验证状态变化。

## 9. 常见排错

### 9.1 前端显示 Ollama 未连接

检查：

```powershell
ollama serve
ollama list
```

检查 Settings 页面中的模型名是否与本地一致。

### 9.2 知识库没有检索结果

检查：

- 是否有 `knowledge/base/` 或 `knowledge/uploads/extracted/` 文档。
- 是否点击重建索引。
- chunk 是否被停用。
- `artifacts/vector_store/metadata.json` 是否存在。

### 9.3 数据上传失败

检查：

- 文件后缀是否在支持列表。
- zip 是否包含路径穿越。
- CSV 是否有数值通道列。
- MAT 是否包含可识别的数值信号矩阵。
- SNIRF 是否安装 MNE。

### 9.4 实验失败

检查：

- 数据是否至少两个标签。
- 是否有足够样本构建 fold。
- 预处理 epoch 窗口是否越界。
- 数据集中 subject 信息是否足够。
- job error 字段。

### 9.5 报告下载失败

检查：

- 实验是否存在。
- 实验是否有 result。
- output_dir 是否存在。
- report_path 是否位于受管目录下。

## 10. 与其他文档的关系

- [设计文档](design_document.md)：说明为什么这样做，关注产品目标、用户场景和设计边界。
- [架构文档](architecture_document.md)：说明系统如何分层、模块如何连接、数据如何流动。
- 本文档：说明每个源程序文件具体做什么、主要类和函数如何工作、如何维护和扩展。
