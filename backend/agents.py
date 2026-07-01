"""Lightweight multi-agent graph orchestrator for chat and task routing."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib import error as urllib_error
from urllib import request as urllib_request

from backend.papers import (
    build_paper_error_messages,
    build_paper_final_messages,
    build_paper_report_messages,
    collect_paper_material_with_model,
    finalize_paper_workflow,
)
from fnirs_core.knowledge import KnowledgeBase, SearchResult


DOMAIN_KEYWORDS = {
    "fnirs",
    "nirs",
    "hbo",
    "hbr",
    "近红外",
    "功能性近红外",
    "脑功能",
    "血氧",
    "血红蛋白",
    "预处理",
    "tddr",
    "beer",
    "lambert",
    "cnn",
    "lstm",
    "tcn",
    "eegnet",
    "graph",
    "loso",
    "group k-fold",
    "grad-cam",
    "integrated gradients",
    "shap",
    "知识库",
    "rag",
    "数据",
    "训练",
    "实验",
    "解释",
    "报告",
}

IDENTITY_CHAT_KEYWORDS = (
    "你是谁",
    "你是什么",
    "你是什么模型",
    "你用的什么模型",
    "你用的是什么模型",
    "你是哪个模型",
    "你基于什么模型",
    "介绍一下你自己",
    "who are you",
    "what model are you",
)

CASUAL_CHAT_MESSAGES = {
    "你好",
    "您好",
    "hello",
    "hi",
    "hey",
    "在吗",
    "谢谢",
    "感谢",
}

EXPERIMENT_KEYWORDS = (
    "训练",
    "实验",
    "建模",
    "模型训练",
    "模型选择",
    "模型架构",
    "分类模型",
    "深度学习模型",
    "loso",
    "group k",
    "验证",
    "accuracy",
    "fold",
    "epoch",
    "cnn-lstm",
    "tcn",
    "graph-tcn",
)

DATA_INTENT_KEYWORDS = (
    "上传数据",
    "上传",
    "数据集",
    "数据文件",
    "数据格式",
    "质控",
    "采样率",
    "事件",
    "标签",
    "subject",
    "subject_id",
    "csv",
    "snirf",
    "dataset",
    "upload data",
)

DATA_FILE_SUFFIXES = (".snirf", ".nirs", ".mat", ".csv", ".zip", ".json")

PAPER_INTENT_KEYWORDS = (
    "论文",
    "文献",
    "paper",
    "article",
    "doi",
    "arxiv",
    "pubmed",
    "阅读报告",
    "原文",
    "找一篇",
    "找论文",
)


@dataclass(slots=True)
class AgentStep:
    agent: str
    action: str
    detail: str
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "detail": self.detail,
            "status": self.status,
        }


@dataclass(slots=True)
class AgentContext:
    query: str
    route: str = "rag"
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval_results: list[SearchResult] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)

    def add_step(self, agent: str, action: str, detail: str, status: str = "completed") -> AgentStep:
        step = AgentStep(agent=agent, action=action, detail=detail, status=status)
        self.steps.append(step)
        return step


class OllamaChatClient:
    def __init__(self, *, model: str | None = None, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.model = model or os.getenv("FNIRS_CHAT_MODEL", "qwen3:8b")
        self.base_url = (base_url or os.getenv("FNIRS_OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.2},
        }
        request = urllib_request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    content = payload.get("message", {}).get("content", "")
                    if content:
                        yield str(content)
                    if payload.get("done"):
                        return
        except urllib_error.URLError as exc:
            raise RuntimeError(f"无法连接 Ollama：{exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama 响应异常：{exc}") from exc


class MultiAgentOrchestrator:
    def __init__(self, *, knowledge_base: KnowledgeBase, llm: OllamaChatClient | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.llm = llm or OllamaChatClient()

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        context = AgentContext(query=query)
        supervisor_step = context.add_step("Supervisor Agent", "route", self._route(query))
        context.route = supervisor_step.detail
        yield {"type": "agent_step", **supervisor_step.to_dict()}

        review_step = context.add_step("Reviewer Agent", "guardrail", self._review_guardrail(query))
        yield {"type": "agent_step", **review_step.to_dict()}

        if context.route == "data":
            yield from self._stream_agent_model_answer(context, "Data Agent")
            return
        if context.route == "experiment":
            yield from self._stream_agent_model_answer(context, "Experiment Agent")
            return
        if context.route == "explain":
            yield from self._stream_agent_model_answer(context, "Explain Agent")
            return
        if context.route == "report":
            yield from self._stream_agent_model_answer(context, "Report Agent")
            return
        if context.route == "paper":
            yield from self._stream_paper_workflow(context)
            return
        if context.route == "chat":
            yield from self._stream_chat_answer(context)
            return

        yield from self._stream_rag_answer(context)

    def _stream_paper_workflow(self, context: AgentContext) -> Iterator[dict[str, Any]]:
        step = context.add_step("Paper Agent", "retrieve", "检索论文候选并获取可读原文")
        yield {"type": "agent_step", **step.to_dict()}
        try:
            material = collect_paper_material_with_model(context.query, self.llm)
            context.sources = [
                {
                    "title": material.candidate.title,
                    "source": material.candidate.source,
                    "score": 1.0,
                    "order": 0,
                    "snippet": material.candidate.abstract[:320] or material.candidate.url or material.candidate.pdf_url or "",
                }
            ]
            yield {"type": "retrieval", "sources": context.sources}

            report_step = context.add_step("Paper Agent", "generate", "调用模型生成 Word 阅读报告内容")
            yield {"type": "agent_step", **report_step.to_dict()}
            report_text = ""
            for piece in self.llm.stream(build_paper_report_messages(material)):
                report_text += piece

            archive_step = context.add_step("Paper Agent", "archive", "保存论文原文和阅读报告，并刷新本地 RAG")
            yield {"type": "agent_step", **archive_step.to_dict()}
            result = finalize_paper_workflow(material, report_text)

            final_step = context.add_step("Paper Agent", "generate", "调用模型生成工作流完成摘要")
            yield {"type": "agent_step", **final_step.to_dict()}
            assembled = ""
            for piece in self.llm.stream(build_paper_final_messages(material, result)):
                assembled += piece
                yield {"type": "content_chunk", "content": piece, "assembled": assembled}
        except Exception as exc:
            error_step = context.add_step("Paper Agent", "generate", "调用模型生成论文工作流失败说明", status="failed")
            yield {"type": "agent_step", **error_step.to_dict()}
            error_message = str(exc)
            assembled = ""
            try:
                for piece in self.llm.stream(build_paper_error_messages(context.query, error_message)):
                    assembled += piece
                    yield {"type": "content_chunk", "content": piece, "assembled": assembled}
            except Exception as model_exc:
                raise self._model_error("Paper Agent", model_exc) from model_exc
            yield {
                "type": "final",
                "content": assembled,
                "sources": context.sources,
                "agent_steps": [step.to_dict() for step in context.steps],
            }
            return

        yield {
            "type": "final",
            "content": assembled,
            "sources": context.sources,
            "agent_steps": [step.to_dict() for step in context.steps],
        }

    def _stream_chat_answer(self, context: AgentContext) -> Iterator[dict[str, Any]]:
        step = context.add_step("Chat Agent", "respond", "处理基础对话")
        yield {"type": "agent_step", **step.to_dict()}

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "你是 fNIRS 深度学习多智能体平台里的中文助手。"
                    "用户问普通寒暄、身份、能力或非专业问题时，直接自然回答；"
                    "只有当用户明确询问 fNIRS 数据、训练、实验、解释、报告或知识库时，才引导到平台工作流。"
                    f"当前后端配置的聊天模型是 `{self.llm.model}`，服务地址是 `{self.llm.base_url}`。"
                ),
            },
            {"role": "user", "content": context.query},
        ]
        assembled = ""
        try:
            for piece in self.llm.stream(prompt_messages):
                assembled += piece
                yield {"type": "content_chunk", "content": piece, "assembled": assembled}
        except Exception as exc:
            raise self._model_error("Chat Agent", exc) from exc
        yield {
            "type": "final",
            "content": assembled,
            "sources": [],
            "agent_steps": [step.to_dict() for step in context.steps],
        }

    def _stream_agent_model_answer(self, context: AgentContext, agent_name: str) -> Iterator[dict[str, Any]]:
        step = context.add_step(agent_name, "generate", "调用模型生成回答")
        yield {"type": "agent_step", **step.to_dict()}
        prompt_messages = build_agent_prompt(
            query=context.query,
            route=context.route,
            agent_name=agent_name,
            model=self.llm.model,
            base_url=self.llm.base_url,
        )
        assembled = ""
        try:
            for piece in self.llm.stream(prompt_messages):
                assembled += piece
                yield {"type": "content_chunk", "content": piece, "assembled": assembled}
        except Exception as exc:
            raise self._model_error(agent_name, exc) from exc
        yield {
            "type": "final",
            "content": assembled,
            "sources": [],
            "agent_steps": [step.to_dict() for step in context.steps],
        }

    def _stream_rag_answer(self, context: AgentContext) -> Iterator[dict[str, Any]]:
        step = context.add_step("RAG Agent", "retrieve", "检索本地 fNIRS + 深度学习知识库")
        yield {"type": "agent_step", **step.to_dict()}
        results = self.knowledge_base.search(context.query, top_k=int(os.getenv("FNIRS_RAG_TOP_K", "4")))
        context.retrieval_results = results
        context.sources = [result.to_dict() for result in results]
        yield {"type": "retrieval", "sources": context.sources}

        generation_step = context.add_step("RAG Agent", "generate", "基于检索片段生成带边界说明的回答")
        yield {"type": "agent_step", **generation_step.to_dict()}
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "你是 fNIRS 深度学习多智能体平台的中文研究助手。"
                    "优先基于给定知识库上下文回答；如果证据不足，明确说明边界。"
                    "涉及验证必须强调 subject-wise split、LOSO 或 Group K-Fold，避免 trial-level 数据泄漏。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{context.query}\n\n知识库上下文：\n{format_context(results)}",
            },
        ]
        assembled = ""
        try:
            for piece in self.llm.stream(prompt_messages):
                assembled += piece
                yield {"type": "content_chunk", "content": piece, "assembled": assembled}
        except Exception as exc:
            raise self._model_error("RAG Agent", exc) from exc
        yield {
            "type": "final",
            "content": assembled,
            "sources": context.sources,
            "agent_steps": [step.to_dict() for step in context.steps],
        }

    def _model_error(self, agent_name: str, exc: Exception) -> RuntimeError:
        return RuntimeError(
            f"{agent_name} 无法完成回答，因为必须调用模型生成内容，但当前模型调用失败：{exc}\n\n"
            f"请检查 Ollama 服务、模型 `{self.llm.model}` 和地址 `{self.llm.base_url}` 是否可用。"
        )

    def _route(self, query: str) -> str:
        normalized = query.lower()
        if is_basic_chat(query):
            return "chat"
        if is_paper_query(query):
            return "paper"
        if is_data_query(query):
            return "data"
        if any(word in normalized for word in EXPERIMENT_KEYWORDS):
            return "experiment"
        if any(word in normalized for word in ("解释", "重要性", "grad", "shap", "通道重要")):
            return "explain"
        if any(word in normalized for word in ("报告", "下载", "总结")):
            return "report"
        return "rag"

    def _review_guardrail(self, query: str) -> str:
        if is_basic_chat(query):
            return "普通对话请求，无需知识库检索或实验流程路由"
        normalized = query.lower()
        if any(keyword in normalized for keyword in DOMAIN_KEYWORDS):
            return "问题位于 fNIRS / 深度学习 / 平台自助范围内"
        return "未命中强领域词，将优先尝试知识库检索并显式说明证据边界"


def is_basic_chat(query: str) -> bool:
    normalized = query.strip().lower()
    compact = "".join(normalized.split())
    if not compact:
        return True
    if compact in CASUAL_CHAT_MESSAGES:
        return True
    return any("".join(keyword.split()) in compact for keyword in IDENTITY_CHAT_KEYWORDS)


def is_data_query(query: str) -> bool:
    normalized = query.lower()
    if any(keyword in normalized for keyword in DATA_INTENT_KEYWORDS):
        return True
    return any(re.search(rf"(^|[^\w]){re.escape(suffix)}([^\w]|$)", normalized) for suffix in DATA_FILE_SUFFIXES)


def is_paper_query(query: str) -> bool:
    normalized = query.lower()
    if any(keyword in normalized for keyword in PAPER_INTENT_KEYWORDS):
        return True
    return any(re.search(rf"(^|[^\w]){suffix}([^\w]|$)", normalized) for suffix in (r"\.pdf", r"\.docx"))


def build_agent_prompt(*, query: str, route: str, agent_name: str, model: str, base_url: str) -> list[dict[str, str]]:
    route_guidance = {
        "data": (
            "你负责回答数据上传、数据格式、采样率、通道、事件、标签和 subject 字段相关问题。"
            "可以提到平台 Data 页面，但必须先回答用户问题本身。"
        ),
        "experiment": (
            "你负责回答实验创建、训练、模型选择、LOSO、Group K-Fold、accuracy 和评估相关问题。"
            "强调 subject-wise split，避免 trial-level 数据泄漏。"
        ),
        "explain": (
            "你负责回答模型解释、通道重要性、时间重要性、SHAP、Grad-CAM、Integrated Gradients 等问题。"
            "说明解释结果的边界，不要把相关性直接说成因果结论。"
        ),
        "report": (
            "你负责回答结果汇总、报告生成、下载和科研记录相关问题。"
            "说明报告应包含配置、验证策略、指标、解释摘要和边界说明。"
        ),
    }.get(route, "你负责回答用户问题。")
    return [
        {
            "role": "system",
            "content": (
                f"你是 fNIRS 深度学习多智能体平台中的 {agent_name}。"
                "所有给用户看的问答内容都必须由当前模型生成，不能使用后端固定模板。"
                f"当前模型配置是 `{model}`，服务地址是 `{base_url}`。"
                f"{route_guidance}"
                "请使用中文，回答简洁、直接、可执行。"
            ),
        },
        {"role": "user", "content": query},
    ]


def format_context(results: list[SearchResult]) -> str:
    if not results:
        return "未检索到高相关片段。"
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            f"[{index}] 标题：{result.title}\n来源：{result.source}\n相关度：{result.score:.4f}\n内容：\n{result.content}"
        )
    return "\n\n---\n\n".join(blocks)


def stream_text(text: str, chunk_size: int = 28) -> Iterator[str]:
    buffer = ""
    for char in text:
        buffer += char
        if len(buffer) >= chunk_size or char in "。！？\n":
            yield buffer
            buffer = ""
    if buffer:
        yield buffer
