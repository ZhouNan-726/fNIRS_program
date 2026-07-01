const STATUS_TEXT = {
  healthy: "正常",
  ready: "就绪",
  missing: "缺失",
  down: "未连接",
  error: "异常",
  checking: "检查中",
  created: "已创建",
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  completed: "已完成",
};

const JOB_KIND_TEXT = {
  experiment_run: "实验运行",
  experiment_explain: "解释生成",
};

const AGENT_TEXT = {
  "Supervisor Agent": "调度智能体",
  "Reviewer Agent": "审查智能体",
  "RAG Agent": "检索智能体",
  "Data Agent": "数据智能体",
  "Preprocess Agent": "预处理智能体",
  "Modeling Agent": "建模智能体",
  "Experiment Agent": "实验智能体",
  "Explain Agent": "解释智能体",
  "Report Agent": "报告智能体",
  "Paper Agent": "论文智能体",
};

const ACTION_TEXT = {
  route: "路由判断",
  guardrail: "边界检查",
  retrieve: "知识检索",
  generate: "生成回答",
  archive: "归档入库",
  advise: "流程建议",
};

const MODEL_TEXT = {
  "fnirs-eegnet": "fNIRS-EEGNet（近红外轻量网络）",
  "cnn-lstm": "CNN-LSTM（卷积循环网络）",
  tcn: "TCN（时序卷积网络）",
  "graph-tcn": "Graph-TCN（图时序卷积网络）",
  "hybrid-3d-cnn": "Hybrid 3D CNN（混合三维卷积网络）",
};

const VALIDATION_TEXT = {
  loso: "LOSO（留一被试验证）",
  "group-kfold": "Group K-Fold（按被试分组交叉验证）",
};

const SUMMARY_KEY_TEXT = {
  n_channels: "通道数",
  n_samples: "采样点数",
  duration_seconds: "时长（秒）",
  sampling_rate: "采样率",
  channel_names: "通道名称",
  has_hbo: "含氧血红蛋白",
  has_hbr: "脱氧血红蛋白",
  n_events: "事件数",
  event_label_distribution: "事件标签分布",
  subject_count: "被试数",
  source_format: "来源格式",
  warning: "警告",
  accuracy: "准确率",
  classes: "类别",
  confusion_matrix: "混淆矩阵",
  per_class: "各类别指标",
  n_samples_metric: "样本数",
  precision: "精确率",
  recall: "召回率",
  f1: "调和均值",
  api_status: "接口状态",
  database_path: "数据库路径",
  ollama_base_url: "本地模型地址",
  chat_model: "对话模型",
  embedding_model: "向量模型",
  ollama_status: "模型服务状态",
  chat_model_status: "对话模型状态",
  embedding_model_status: "向量模型状态",
  vector_store_path: "向量库路径",
  message: "提示",
};

export function statusText(value) {
  return STATUS_TEXT[value] || value || "-";
}

export function jobKindText(value) {
  return JOB_KIND_TEXT[value] || value || "-";
}

export function agentText(value) {
  return AGENT_TEXT[value] || value || "-";
}

export function actionText(value) {
  return ACTION_TEXT[value] || value || "-";
}

export function modelText(value) {
  return MODEL_TEXT[value] || value || "-";
}

export function localModelSummary(status) {
  if (!status) return "未检测";
  return statusText(status);
}

export function configuredText(value, configured = "已配置") {
  return value ? configured : "未配置";
}

export function validationText(value) {
  return VALIDATION_TEXT[value] || value || "-";
}

export function foldNameText(value) {
  if (!value) return "-";
  return String(value)
    .replace(/^LOSO\s+(.+)$/i, "LOSO（留一被试）：$1")
    .replace(/^GroupKFold\s+(.+)$/i, "GroupKFold（分组折次）：$1")
    .replace(/^holdout$/i, "Holdout（留出验证）");
}

export function displayText(value) {
  if (!value) return "";
  const text = String(value);
  if (text.includes("No events found")) {
    return "未发现事件标记；系统已生成带 placeholder labels 的滑窗样本。该结果仅用于验证本地流程，不能作为科研结论。";
  }
  return text
    .replaceAll("ready", "就绪")
    .replaceAll("healthy", "正常")
    .replaceAll("missing", "缺失")
    .replace(/\bCh(\d+)\b/g, "Ch$1");
}

export function yesNo(value) {
  return value ? "是" : "否";
}

export function formatDate(value) {
  if (!value) return "";
  return new Date(value).toLocaleString("zh-CN");
}

export function formatJsonChinese(value) {
  return JSON.stringify(localizeKeys(value), null, 2);
}

function localizeKeys(value) {
  if (Array.isArray(value)) {
    return value.map((item) => localizeKeys(item));
  }
  if (typeof value === "boolean") {
    return yesNo(value);
  }
  if (!value || typeof value !== "object") {
    return typeof value === "string" ? displayText(statusText(value)) : value;
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [
      SUMMARY_KEY_TEXT[key] || key,
      localizeKeys(item),
    ]),
  );
}
