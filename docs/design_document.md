# fNIRS 深度学习多智能体自助平台设计文档

## 1. 文档目的

本文档用于说明 fNIRS 深度学习多智能体自助平台的产品目标、用户场景、功能边界、交互设计、数据流设计、领域算法设计、安全与降级策略，以及当前版本的验收标准。

本文档面向以下人员：

- 项目负责人：用于确认平台建设目标、阶段范围与验收口径。
- 研发人员：用于理解每个功能为什么存在、应该服务哪个流程、边界在哪里。
- 测试人员：用于设计端到端验收用例与异常路径用例。
- fNIRS 研究使用者：用于理解平台适合解决什么问题、哪些结果需要人工复核。

本文档描述的是当前源码中的真实实现与当前设计意图。对未来扩展能力，例如完整 PyTorch 训练、分布式任务队列、PDF 报告导出、多用户权限等，会在“演进方向”中说明，不把它们视为当前已完成能力。

## 2. 项目定位

本项目是一个本地单用户 fNIRS 深度学习研究工作台。平台把以下能力整合在一个可运行的本地应用中：

- fNIRS 数据上传、解析与摘要。
- fNIRS 预处理与 epoch 提取。
- subject-wise 实验验证，包括 LOSO 与 Group K-Fold。
- 轻量原型分类器实验，用于快速验证完整流程。
- 可解释性结果生成，包括通道重要性、时间重要性和波段重要性。
- Markdown 实验报告生成。
- RAG 知识库管理，包括文档上传、文本抽取、chunk 切分、向量索引、Top-K 检索和 chunk 启停。
- 多智能体式对话路由，包括 Supervisor、Reviewer、RAG、Data、Experiment、Explain、Report、Paper 等角色。
- 论文检索、原文归档、Word 阅读报告生成和 RAG 入库。
- 本地运行状态管理，包括 SQLite、Ollama 模型与向量库状态。

平台是 local-first 工具，优先服务单机、本地、可追踪、可回放的研究流程。默认不依赖云端数据库、云端对象存储、远程队列或多用户鉴权系统。

## 3. 设计背景

fNIRS 深度学习实验通常同时涉及信号处理、数据质量、被试划分、模型训练、解释分析和论文依据。单独使用脚本时，研究者容易遇到以下问题：

- 数据格式多样：`.snirf`、`.nirs`、`.mat`、`.csv`、`.json` 与压缩包结构不统一。
- 事件、标签、被试编号缺失或命名不一致，导致实验流程不容易自动化。
- trial-level 随机划分容易把同一被试的数据泄漏到训练集和验证集。
- 论文、实验配置、结果、解释和报告分散存放，复盘成本高。
- RAG 知识库若索引不刷新，容易引用过期片段。
- 本地大模型或 embedding 服务不可用时，平台如果完全不可用会影响研究流程。

因此当前设计优先解决“可运行、可追踪、可降级、可复盘”的问题，而不是一开始追求重型训练框架或复杂部署。

## 4. 设计目标

### 4.1 总体目标

平台应让单机用户能够完成从知识准备、数据上传、实验创建、结果解释到报告生成的闭环流程。

核心目标如下：

- 用户可以上传或创建知识文档，并把它们纳入 RAG 检索。
- 用户可以上传 fNIRS 数据文件，并得到通道数、采样率、事件数、被试数等摘要。
- 用户可以创建本地实验，选择模型族和验证策略，并提交后台任务运行。
- 实验流程应默认强调 subject-wise 验证，避免 trial-level 泄漏。
- 用户可以查看 fold 指标、整体指标、解释结果和报告。
- 用户可以通过对话入口提出知识库问题、数据问题、实验问题、解释问题、报告问题或论文检索任务。
- 平台应在 Ollama 不可用时尽量保持可启动，并在 embedding 不可用时使用本地 hashing fallback。
- 所有关键状态应可查看，关键资源应可删除或刷新。

### 4.2 非目标

当前版本明确不覆盖以下目标：

- 多租户、多用户登录与权限管理。
- 远程生产部署与公网访问安全方案。
- 分布式训练或 GPU 集群调度。
- 完整 PyTorch 深度学习训练框架。
- 数据标注平台。
- 医疗诊断结论生成。
- 自动替代科研人工审查。

## 5. 用户角色

### 5.1 fNIRS 研究者

需求：

- 上传 fNIRS 数据。
- 快速查看数据是否具备实验条件。
- 使用 subject-wise 验证跑通实验。
- 查看解释结果和报告。
- 基于知识库询问信号处理、模型和验证策略。

关注点：

- 数据是否可解析。
- 是否有事件和标签。
- 是否有被试信息。
- 验证策略是否避免泄漏。
- 结果是否可追溯。

### 5.2 深度学习实验开发者

需求：

- 查看当前模型族和配置。
- 调整实验参数。
- 验证数据管线和评估指标是否工作。
- 后续扩展真实模型训练。

关注点：

- 输入张量形状是否稳定。
- fold 划分是否按 subject 分组。
- 结果文件是否可复用。
- 后端服务层是否容易扩展。

### 5.3 课程或课题组成员

需求：

- 在本机搭建一个可演示的研究工作台。
- 管理论文、阅读报告和知识库。
- 通过 UI 理解实验流程。

关注点：

- 启动简单。
- 页面功能清楚。
- 报告可下载。
- 错误信息可理解。

## 6. 典型使用场景

### 6.1 知识库问答

用户上传 fNIRS 论文、笔记或平台指南，然后在 Chat 页面提问，例如：

- “fNIRS 预处理一般包括哪些步骤？”
- “LOSO 和 Group K-Fold 有什么区别？”
- “为什么 trial-level 随机划分会造成数据泄漏？”

平台行为：

- Supervisor Agent 判断请求类型。
- Reviewer Agent 做边界检查。
- RAG Agent 检索本地知识库。
- Ollama 生成回答。
- UI 展示回答、检索来源和智能体步骤。
- 对话保存到 SQLite，用户可在历史会话中回看。

### 6.2 数据上传与摘要

用户在 Data 页面上传 `.snirf`、`.nirs`、`.mat`、`.csv`、`.zip` 或 `.json` 文件。

平台行为：

- 后端保存文件到 `artifacts/datasets/{dataset_id}/`。
- `fnirs_core.data` 根据后缀选择解析器。
- 解析得到 `NIRSData` 对象。
- 返回摘要：通道数、采样点数、采样率、时长、事件数、标签分布、被试数、来源格式等。
- 前端展示最近上传数据集摘要。

### 6.3 创建并运行实验

用户在 Experiments 页面选择数据集、模型族、验证策略、折数和随机种子。

平台行为：

- 后端创建实验记录，保存配置到 SQLite。
- 用户点击运行后，后端创建 `experiment_run` job。
- 后端启动线程执行实验。
- 实验流程加载数据、预处理、构建 subject-wise folds、训练原型分类器、计算指标、写出结果。
- job 状态从 queued 到 running，再到 succeeded 或 failed。
- 前端通过刷新任务队列查看进度。

### 6.4 生成解释与报告

用户在 Results 页面选择实验并生成解释。

平台行为：

- 后端创建 `experiment_explain` job。
- 解释模块重新加载实验数据与配置，训练同类原型模型。
- 输出通道重要性、时间重要性、波段重要性和 top channels。
- 用户可下载 Markdown 报告。

### 6.5 论文工作流

用户在 Chat 页面提出论文类请求，例如：

- “帮我找一篇 fNIRS deep learning 论文并生成阅读报告，存入本地 RAG。”

平台行为：

- Supervisor Agent 将请求路由到 Paper Agent。
- 论文模块尝试通过模型提示、Semantic Scholar、arXiv、Crossref 或直接 URL 获取候选。
- 下载可访问原文或摘要。
- 用模型生成 Word 阅读报告内容。
- 保存论文原文、阅读报告和元数据。
- 抽取可入库文本并刷新 RAG。
- 若发现可下载数据文件，尝试入库为数据集。

## 7. 信息架构

前端不是营销页，而是研究工作台。第一屏直接进入可操作系统。

当前页面结构如下：

| 页面 | 路由 | 核心用途 |
| --- | --- | --- |
| 总览 | `/` | 查看本地模型、知识库、数据集、实验、任务概况 |
| 对话 | `/chat` | 多智能体对话、RAG 问答、历史会话、来源和步骤 |
| 知识库 | `/knowledge` | 上传文档、创建文档、编辑托管文档、启停 chunk、重建索引 |
| 数据 | `/data` | 上传 fNIRS 数据并查看摘要 |
| 实验 | `/experiments` | 创建实验、运行实验、生成解释、查看任务 |
| 结果 | `/results` | 查看指标、fold、解释和下载报告 |
| 设置 | `/settings` | 查看健康状态、修改 chat/embedding 模型名 |

## 8. 核心功能设计

### 8.1 总览设计

总览页服务于“系统现在能不能用”的快速判断。

展示内容：

- Ollama 服务状态。
- 对话模型状态。
- 知识文档数。
- 知识 chunk 数。
- 数据集数量。
- 实验数量。
- 任务数量。
- 最近实验。
- 任务队列。

设计原则：

- 信息密度高于装饰性。
- 用户进入后能立刻判断本地服务是否健康。
- 如果模型服务异常，不隐藏平台其他能力。

### 8.2 对话设计

对话是平台的统一研究助手入口。它承担三类职责：

- 回答知识库问题。
- 引导用户理解数据、实验、解释、报告流程。
- 执行论文检索与阅读报告工作流。

对话 UI 包含：

- 消息列表。
- 输入框。
- 历史会话。
- 智能体步骤。
- 检索来源。

SSE 事件设计：

| 事件 | 含义 | 前端处理 |
| --- | --- | --- |
| `status` | 当前处理状态 | 更新状态文字和 timeline |
| `agent_step` | 智能体步骤 | 追加到步骤列表 |
| `retrieval` | 检索来源 | 更新来源列表 |
| `content_chunk` | 流式回答片段 | 追加到 assistant 消息 |
| `final` | 完整回答 | 覆盖最终内容、来源和步骤 |
| `session` | 保存后的会话 ID | 绑定当前会话并刷新历史 |
| `error` | 错误信息 | 显示失败消息 |
| `done` | 流结束 | 结束加载状态 |

路由策略：

- 寒暄、身份问题走 Chat Agent。
- 论文、文献、DOI、arXiv、PubMed 等请求走 Paper Agent。
- 上传、数据集、采样率、事件、标签、subject 等请求走 Data Agent。
- 训练、实验、模型、LOSO、Group K、accuracy、fold 等请求走 Experiment Agent。
- 解释、重要性、SHAP、Grad-CAM、通道重要等请求走 Explain Agent。
- 报告、下载、总结等请求走 Report Agent。
- 其他问题默认走 RAG Agent。

### 8.3 知识库设计

知识库支持两类来源：

- 内置知识：`knowledge/base/`。
- 用户托管知识：`knowledge/uploads/extracted/`。

知识文档处理流程：

1. 用户上传或创建文档。
2. 上传文件原件保存到 `artifacts/knowledge_uploads/raw/`。
3. 平台抽取文本。
4. 抽取后的 Markdown 保存到 `knowledge/uploads/extracted/`。
5. 知识库刷新索引。
6. 文档被切分为 chunk。
7. chunk 生成 embedding。
8. 向量与元数据写入 `artifacts/vector_store/`。

支持的知识文档类型：

- `.pdf`
- `.docx`
- `.doc`
- `.md`
- `.markdown`
- `.txt`
- `.text`

托管文档能力：

- 新建。
- 编辑。
- 删除。
- 查看 chunk。
- 启用或停用单个 chunk。

内置文档能力：

- 查看。
- 参与索引。
- 不允许通过页面编辑或删除。

chunk 启停设计：

- 启停状态保存在向量库元数据中。
- 检索时跳过停用 chunk。
- 重建索引时尽量保留已有 chunk 的启停状态。

索引一致性设计：

索引加载时会检查：

- 配置的 embedding 模型是否改变。
- embedding base URL 是否改变。
- source roots 是否改变。
- 当前文件集合是否和元数据一致。
- 文件更新时间是否一致。
- 文本长度是否一致。
- 向量数量是否和 chunk 数量一致。

发现不一致时自动刷新索引，避免复用过期向量。

### 8.4 数据管理设计

数据管理页服务于三个问题：

- 这是什么格式？
- 是否包含通道、事件、标签和被试信息？
- 能否进入实验流程？

支持格式：

- `.snirf`
- `.nirs`
- `.mat`
- `.csv`
- `.zip`
- `.json`

摘要字段：

| 字段 | 含义 |
| --- | --- |
| `n_channels` | 通道数 |
| `n_samples` | 采样点数 |
| `duration_seconds` | 记录时长 |
| `sampling_rate` | 采样率 |
| `channel_names` | 通道名称 |
| `has_hbo` | 是否包含 HbO |
| `has_hbr` | 是否包含 HbR |
| `n_events` | 事件数 |
| `event_label_distribution` | 事件标签分布 |
| `subject_count` | 被试数 |
| `source_format` | 来源格式 |

CSV 设计约定：

- 可识别事件列：`event`、`label`、`stim`、`trigger`、`condition`、`class`。
- 可识别被试列：`subject`、`subject_id`、`participant`、`participant_id`、`sub`。
- 可识别时间列：`time`、`timestamp`、`t`、`seconds`。
- 除事件、被试和时间列以外的数值列被视为通道。
- 若存在时间列，根据相邻时间差推断采样率。

MAT/NIRS 设计约定：

- 优先查找 `raw_data`、`dataTimeSeries`、`data_time_series`、`d`、`Y`、`signal`、`signals`、`x`、`dc`、`dod`、`hbo`、`hbr`、`data` 等候选字段。
- 优先查找 `sampling_rate`、`sample_rate`、`fs`、`srate`、`sfreq` 等采样率字段。
- 优先查找 `events`、`event`、`stim`、`trigger`、`triggers`、`s`、`labels`、`label`、`condition`、`class` 等事件字段。
- 对结构体、嵌套字典和 object array 做递归候选搜索。

Zip 设计约定：

- 解压前校验成员路径。
- 拒绝包含 `../` 等路径穿越的压缩包。
- 在压缩包中寻找第一个可解析 fNIRS 文件。
- 如果前面的候选不可解析，会继续尝试后续候选。

### 8.5 实验编排设计

实验编排页服务于完整实验闭环。

实验配置字段：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `name` | `Quick fNIRS Experiment` | 实验名称 |
| `dataset_id` | 空 | 数据集 ID；为空时使用 demo 数据 |
| `preprocessing` | `{}` | 预处理配置 |
| `model.model_family` | `cnn-lstm` | 模型族 |
| `validation_strategy` | `loso` | 验证策略 |
| `num_folds` | `5` | Group K-Fold 折数 |
| `seed` | `42` | 随机种子 |

支持模型族：

- `fnirs-eegnet`
- `cnn-lstm`
- `tcn`
- `graph-tcn`
- `hybrid-3d-cnn`

当前模型实现说明：

当前版本的模型是 `PrototypeClassifier`，不是完整深度学习训练网络。它使用 NumPy 提取均值、标准差、峰值、谷值、时间斜率等特征，并按类别质心进行分类。不同模型族会影响特征组合，用于模拟不同模型族的实验入口。

这样设计的原因：

- 保证本地无 GPU、无 PyTorch 环境时仍可运行。
- 先打通端到端流程、结果追踪和页面交互。
- 为未来替换真实模型训练保留统一接口。

### 8.6 预处理设计

默认预处理配置：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `apply_optical_density` | `true` | 是否执行光密度转换 |
| `apply_beer_lambert` | `true` | 是否执行 Beer-Lambert 风格标准化 |
| `apply_tddr` | `true` | 是否执行 TDDR 风格伪影修正 |
| `bandpass_low` | `0.01` | 带通低频 |
| `bandpass_high` | `0.2` | 带通高频 |
| `baseline_start` | `-2.0` | baseline 起点 |
| `baseline_end` | `0.0` | baseline 终点 |
| `epoch_start` | `-2.0` | epoch 起点 |
| `epoch_end` | `10.0` | epoch 终点 |
| `include_hbo_hbr` | `true` | 有 HbO/HbR 时是否纳入多 band |

输出张量形状：

```text
(epochs, bands, channels, times)
```

预处理结果包含：

- epochs。
- labels。
- groups。
- times。
- channel_names。
- band_names。
- summary。
- config。

无事件 fallback：

如果数据没有事件标记，平台不会直接中断，而是进入滑窗 fallback：

- 按约 10 秒窗口切分连续记录。
- 窗口步长为窗口长度的一半。
- 生成交替 placeholder labels。
- 若只有一个窗口，会复制一个窗口以保证至少两个标签。
- summary 中写入 warning。

该 fallback 仅用于验证本地流程是否跑通，不能作为正式科研结论。

### 8.7 验证策略设计

平台优先使用 subject-wise 验证。

支持策略：

- `loso`：Leave-One-Subject-Out，每个被试轮流作为验证集。
- `group-kfold`：按被试分组后做 K 折。
- `holdout` fallback：当只有一个 subject 但样本数足够时，退化为简单留出验证。

设计原则：

- 默认不采用 trial-level 随机划分。
- fold 中训练集和验证集应尽量没有相同 subject。
- 当数据没有 subject 字段时，系统会把事件归入默认 subject，可能导致无法构建真正 subject-wise fold。用户需要补充 subject 信息。

### 8.8 结果设计

实验运行完成后保存：

- `result.json`
- `prototype_model.json`
- 可能的 `explanation.json`
- `report.md`

指标包括：

- accuracy。
- classes。
- confusion_matrix。
- per_class precision。
- per_class recall。
- per_class f1。
- n_samples。

fold 结果包括：

- fold_name。
- train_size。
- val_size。
- accuracy。
- labels。
- predictions。
- subject_ids。

### 8.9 可解释性设计

当前解释方法是 `prototype-activation`。

它基于输入张量绝对激活强度计算：

- channel_importance。
- time_importance。
- band_importance。

并输出 top channels。

解释边界：

- 当前解释结果是轻量方法，不等价于因果脑区定位。
- 通道重要性只能说明模型在该输入和该原型方法下的关注强度。
- 正式科研结论需要结合实验范式、通道布置、统计检验和人工复核。

### 8.10 报告设计

报告格式为 Markdown。

报告内容包括：

- 实验概览。
- 实验 ID。
- 数据集。
- 验证策略。
- 模型。
- accuracy。
- 样本数。
- fold 结果。
- top channels。
- 配置快照。
- 边界说明。

下载入口：

```text
GET /api/reports/{experiment_id}/download
```

### 8.11 设置设计

设置页用于查看和调整本地模型配置。

可编辑字段：

- 对话模型：默认 `qwen3:8b`。
- 向量模型：默认 `qwen3-embedding:8b`。

只读字段：

- Ollama 地址。
- SQLite 数据库路径。
- 本地向量库路径。

配置来源优先级：

1. SQLite `runtime_settings` 表。
2. 环境变量。
3. 默认值。

## 9. 多智能体设计

### 9.1 智能体角色

| 智能体 | 职责 |
| --- | --- |
| Supervisor Agent | 判断用户请求类型并决定路由 |
| Reviewer Agent | 做领域边界与安全提示 |
| Chat Agent | 回答基础寒暄、身份和普通对话 |
| RAG Agent | 检索知识库并生成基于证据的回答 |
| Data Agent | 回答数据上传、格式、采样率、事件、标签和 subject 问题 |
| Experiment Agent | 回答实验创建、模型、验证、fold、accuracy 等问题 |
| Explain Agent | 回答解释、通道重要性、时间重要性和方法边界 |
| Report Agent | 回答报告、下载、总结和科研记录问题 |
| Paper Agent | 执行论文检索、阅读报告、归档和 RAG 入库工作流 |

### 9.2 编排方式

当前版本采用轻量 graph orchestrator，不依赖外部多智能体框架。

编排流程：

1. 创建 `AgentContext`。
2. Supervisor Agent 计算 route。
3. Reviewer Agent 输出 guardrail。
4. 根据 route 分发到对应 agent。
5. agent 调用本地 Ollama 生成内容。
6. 通过 SSE 输出事件。
7. 后端保存完整会话。

### 9.3 事件与可追踪性

每次对话保存：

- 用户消息。
- assistant 消息。
- 检索来源。
- 智能体步骤。
- session ID。
- created_at。
- updated_at。

多轮会话中，消息会继续追加。页面展示智能体步骤时只展示最新一轮 route 之后的步骤，避免旧轮步骤干扰用户判断。

## 10. 数据存储设计

### 10.1 SQLite 元数据

SQLite 数据库位于：

```text
storage/app.db
```

表：

- `projects`
- `chat_sessions`
- `datasets`
- `experiments`
- `jobs`
- `runtime_settings`

SQLite 负责保存元数据，不保存大型二进制内容。

### 10.2 文件存储

| 路径 | 内容 |
| --- | --- |
| `artifacts/datasets/` | 上传后的 fNIRS 数据文件 |
| `artifacts/knowledge_uploads/raw/` | 知识文档上传原件 |
| `knowledge/base/` | 内置知识文档 |
| `knowledge/uploads/extracted/` | 抽取后的托管知识 Markdown |
| `artifacts/vector_store/` | 向量库文件与元数据 |
| `artifacts/experiments/` | 实验结果、checkpoint、解释和报告 |
| `artifacts/reports/` | 预留报告目录 |
| `artifacts/papers/` | 论文工作流缓存 |

## 11. 安全设计

### 11.1 本地单用户边界

当前版本面向本机单用户，不提供登录和权限体系。因此它不应直接作为公网服务暴露。

### 11.2 文件类型限制

知识库上传限制在文档格式。

数据上传限制在 fNIRS 相关格式。

未知后缀会被拒绝。

### 11.3 Zip 路径穿越防护

知识文档和数据文件的 zip 解压都检查成员路径：

- 将目标路径 resolve。
- 要求目标路径仍在临时解压根目录下。
- 如果出现路径穿越，抛出错误并拒绝处理。

### 11.4 删除边界

删除数据集、实验结果、实验输出和报告文件时，后端会校验目标路径位于受管目录下。

受管目录包括：

- `artifacts/datasets/`
- `artifacts/experiments/`
- `artifacts/reports/`
- `knowledge/uploads/extracted/`

平台不会删除这些目录之外的文件。

### 11.5 研究结论边界

平台自动生成的实验结果、解释和报告都需要人工复核。

尤其是以下情况不能作为正式科研结论：

- 使用 demo 数据。
- 使用无事件 fallback 生成 placeholder labels。
- 只有单个 subject。
- 使用当前原型分类器而非真实深度学习模型。
- 数据质量、通道布局和范式设计未经人工确认。

## 12. 降级设计

### 12.1 Ollama chat 不可用

当前对话生成必须调用模型。如果 chat 模型调用失败，对应 agent 会返回明确错误，提示检查 Ollama 服务、模型名和地址。

### 12.2 Ollama embedding 不可用

知识库 embedding 优先调用 Ollama `/api/embed`。

如果失败且允许 fallback，则使用 `local-hashing-char-ngram`：

- 按 token 和字符 n-gram 做 hashing。
- 生成固定维度向量。
- 归一化后用于相似度检索。

这样即使本地 embedding 模型不可用，知识库仍能构建和检索。

### 12.3 无真实数据

创建实验时如果没有选择数据集，使用 demo fNIRS 数据。

demo 数据包含：

- 多 subject。
- 多 trial。
- 合成通道信号。
- HbO/HbR。
- control/task 标签。

用于验证流程，不代表真实科研数据。

### 12.4 无事件数据

如果上传数据没有事件，预处理进入滑窗 fallback，并在 summary 中写 warning。

UI 会把英文 warning 转为中文提示，强调不能作为科研结论。

## 13. 用户界面设计原则

### 13.1 工作台优先

本项目不使用 landing page。用户打开页面后直接进入工作台。

### 13.2 中文界面

用户可见页面以中文为主。

保留英文的情况：

- 模型 ID。
- API 路径。
- 文件路径。
- validation strategy 缩写。
- fNIRS 术语。
- 上传内容原文。

### 13.3 信息密度

这是研究工具，不是展示型官网。页面采用：

- 侧边栏导航。
- 指标块。
- 列表。
- 面板。
- 表单。
- 任务队列。
- 右侧详情栏。

### 13.4 可回看

所有关键流程都应能回看：

- 对话历史。
- 检索来源。
- 智能体步骤。
- 数据摘要。
- 实验配置。
- fold 结果。
- 解释结果。
- 报告。

## 14. API 设计原则

### 14.1 薄入口

`backend/main.py` 只做：

- 请求接收。
- schema 绑定。
- 调用 service。
- HTTP 错误转换。
- SSE 包装。

业务逻辑不写在 route 中。

### 14.2 服务层集中编排

`backend/services.py` 负责：

- 初始化运行目录。
- 管理运行配置。
- 知识库服务。
- 数据集服务。
- 实验服务。
- job 服务。
- 会话服务。
- 健康检查。

### 14.3 核心领域能力独立

`fnirs_core/` 不依赖 FastAPI 和前端，它提供可测试的领域能力：

- 数据解析。
- 预处理。
- 模型。
- 实验。
- 解释。
- 报告。
- 知识库。

这种分层便于后续把核心算法替换为更重的实现。

## 15. 验收标准

### 15.1 启动验收

- 后端可以通过 `uvicorn backend.main:app` 启动。
- 前端可以通过 `npm run dev` 启动。
- `GET /api/health` 返回 API、数据库、Ollama、模型和向量库状态。

### 15.2 知识库验收

- 可以上传支持的知识文档。
- 可以抽取文本并生成托管 Markdown。
- 可以刷新向量索引。
- 可以列出文档和 chunk。
- 可以启停 chunk。
- 可以编辑和删除托管文档。
- 不能编辑和删除内置文档。
- embedding 模型改变后索引应判定失效。
- zip 路径穿越应被拒绝。

### 15.3 对话验收

- 可以发送普通对话。
- 身份类问题走基础聊天路由。
- fNIRS 知识问题走 RAG。
- 训练和验证问题走 Experiment Agent。
- 数据上传和格式问题走 Data Agent。
- 论文请求走 Paper Agent。
- 会话可保存、读取和删除。
- 多轮会话保留消息历史。
- 智能体步骤展示最新一轮。

### 15.4 数据验收

- 可以上传 `.csv` 数据并得到摘要。
- ragged CSV 不应导致崩溃。
- MAT/NIRS 中嵌套 signal 应尽量解析。
- zip 中不可解析候选应跳过并尝试下一个文件。
- zip 路径穿越应被拒绝。
- 删除数据集后关联实验应解除 dataset 引用。

### 15.5 实验验收

- 无上传数据时可以运行 demo 实验。
- 有事件数据时可以提取 epoch。
- 无事件数据时可以使用带 warning 的滑窗 fallback。
- 可以构建 LOSO folds。
- 可以构建 Group K-Fold folds。
- 可以输出 metrics、folds、result.json 和 checkpoint。

### 15.6 结果验收

- 可以生成解释结果。
- 可以展示 top channels。
- 可以生成并下载 Markdown 报告。
- 可以删除实验结果、解释和报告，同时保留实验配置。
- 删除实验时应删除关联输出目录和任务。

## 16. 当前限制

- 对话生成依赖 Ollama chat，模型不可用时无法自动生成自然语言回答。
- 当前模型是原型分类器，不是完整深度神经网络。
- 当前解释是轻量激活统计，不是完整 Captum、SHAP 或 Grad-CAM。
- job runner 是线程模型，没有持久化队列和重试机制。
- 没有登录鉴权。
- 没有并发用户隔离。
- 没有实验版本管理。
- 报告是 Markdown，不是 PDF。
- SNIRF 解析依赖 `mne`，MAT/NIRS 解析依赖 `scipy`。
- DOCX 报告生成依赖 `python-docx`。

## 17. 演进方向

后续可按以下路线扩展：

- 引入真实 PyTorch 训练管线。
- 为每个模型族实现独立模型结构。
- 加入 GPU/CPU 资源检测。
- 加入更完整的实验版本管理。
- 将线程 job runner 替换为队列式异步任务系统。
- 支持 PDF 报告导出。
- 增加 Captum、SHAP、Grad-CAM 等解释方法。
- 增加数据质量控制报告。
- 增加通道布局与脑区映射。
- 增加用户权限与项目隔离。
- 增加审计日志。
- 增加前端自动轮询 job 状态。

## 18. 术语表

| 术语 | 说明 |
| --- | --- |
| fNIRS | Functional Near-Infrared Spectroscopy，功能性近红外光谱 |
| HbO | Oxygenated hemoglobin，含氧血红蛋白 |
| HbR | Deoxygenated hemoglobin，脱氧血红蛋白 |
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| chunk | 知识文档切分后的检索片段 |
| embedding | 文本向量表示 |
| LOSO | Leave-One-Subject-Out，留一被试验证 |
| Group K-Fold | 按 group 分组的 K 折交叉验证 |
| TDDR | Temporal Derivative Distribution Repair，常用于 fNIRS 运动伪影处理思路 |
| epoch | 围绕事件截取的时间窗样本 |
| fallback | 主路径不可用时的降级路径 |
| placeholder labels | 为跑通流程临时生成的占位标签，不可用于科研结论 |
