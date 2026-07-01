"""FastAPI entrypoint for the fNIRS self-service platform."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.schemas import (
    BackendStatusResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
    DashboardResponse,
    DatasetListResponse,
    DatasetUploadResponse,
    ExperimentCreateRequest,
    ExperimentListResponse,
    ExperimentResponse,
    JobResponse,
    KnowledgeChunkUpdateRequest,
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentDetailResponse,
    KnowledgeDocumentsResponse,
    KnowledgeDocumentUpdateRequest,
    KnowledgeStatusResponse,
    KnowledgeUploadResponse,
    RuntimeSettingsUpdateRequest,
)
from backend.services import (
    build_knowledge_status,
    create_experiment,
    create_knowledge_document,
    dashboard,
    delete_chat_session,
    delete_dataset,
    delete_experiment,
    delete_experiment_results,
    delete_knowledge_document,
    explain_experiment_job,
    generate_report,
    get_dataset,
    get_experiment,
    get_job,
    get_chat_session,
    get_knowledge_document,
    get_orchestrator,
    ingest_knowledge_file,
    list_datasets,
    list_experiments,
    list_jobs,
    list_chat_sessions,
    list_knowledge_documents,
    probe_health,
    refresh_knowledge_base,
    run_experiment_job,
    save_chat_session,
    set_knowledge_chunk_enabled,
    update_runtime_config,
    update_knowledge_document,
    upload_dataset,
    ensure_runtime,
)
from fnirs_core.knowledge import KnowledgeBaseError


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_runtime()
    yield


app = FastAPI(title="fNIRS Multi-Agent Platform API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chat_stream(request: ChatRequest) -> Iterator[str]:
    orchestrator = get_orchestrator()
    assembled = ""
    sources: list[dict[str, Any]] = []
    agent_steps: list[dict[str, Any]] = []
    try:
        yield _sse("status", {"message": "Supervisor Agent 正在分析请求"})
        for event in orchestrator.stream(request.message):
            event_type = str(event.pop("type", "message"))
            if event_type == "content_chunk":
                assembled = str(event.get("assembled", assembled))
            elif event_type == "retrieval":
                sources = list(event.get("sources", sources))
            elif event_type == "agent_step":
                agent_steps.append(dict(event))
            elif event_type == "final":
                assembled = str(event.get("content", assembled))
                sources = list(event.get("sources", sources))
                agent_steps = list(event.get("agent_steps", agent_steps))
            yield _sse(event_type, event)

        session_id = save_chat_session(
            session_id=request.session_id,
            user_message=request.message,
            assistant_message=assembled,
            sources=sources,
            agent_steps=agent_steps,
        )
        yield _sse("session", {"session_id": session_id})
        yield _sse("done", {"ok": True})
    except Exception as exc:
        yield _sse("error", {"message": str(exc)})
        yield _sse("done", {"ok": False})


@app.get("/api/health")
def health():
    return probe_health()


@app.put("/api/settings", response_model=BackendStatusResponse)
def settings_update(request: RuntimeSettingsUpdateRequest) -> BackendStatusResponse:
    try:
        return update_runtime_config(
            chat_model=request.chat_model,
            embedding_model=request.embedding_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/dashboard", response_model=DashboardResponse)
def dashboard_snapshot() -> DashboardResponse:
    return DashboardResponse(**dashboard())


@app.get("/api/knowledge", response_model=KnowledgeStatusResponse)
def knowledge_status() -> KnowledgeStatusResponse:
    return build_knowledge_status()


@app.post("/api/knowledge/refresh", response_model=KnowledgeStatusResponse)
def knowledge_refresh() -> KnowledgeStatusResponse:
    return refresh_knowledge_base()


@app.get("/api/knowledge/documents", response_model=KnowledgeDocumentsResponse)
def knowledge_documents() -> KnowledgeDocumentsResponse:
    return list_knowledge_documents()


@app.post("/api/knowledge/documents", response_model=KnowledgeDocumentDetailResponse)
def knowledge_create(request: KnowledgeDocumentCreateRequest) -> KnowledgeDocumentDetailResponse:
    try:
        return create_knowledge_document(request.title, request.content)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/knowledge/documents/{document_id}", response_model=KnowledgeDocumentDetailResponse)
def knowledge_detail(document_id: str) -> KnowledgeDocumentDetailResponse:
    try:
        return get_knowledge_document(document_id)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/knowledge/documents/{document_id}", response_model=KnowledgeDocumentDetailResponse)
def knowledge_update(document_id: str, request: KnowledgeDocumentUpdateRequest) -> KnowledgeDocumentDetailResponse:
    try:
        return update_knowledge_document(document_id, request.title, request.content)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/knowledge/documents/{document_id}/chunks/{order}", response_model=KnowledgeDocumentDetailResponse)
def knowledge_chunk_update(
    document_id: str,
    order: int,
    request: KnowledgeChunkUpdateRequest,
) -> KnowledgeDocumentDetailResponse:
    try:
        return set_knowledge_chunk_enabled(document_id, order, request.enabled)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/knowledge/documents/{document_id}", response_model=KnowledgeStatusResponse)
def knowledge_delete(document_id: str) -> KnowledgeStatusResponse:
    try:
        return delete_knowledge_document(document_id)
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/knowledge/upload", response_model=KnowledgeUploadResponse)
async def knowledge_upload(file: UploadFile = File(...)) -> KnowledgeUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少上传文件名。")
    try:
        return ingest_knowledge_file(file.filename, await file.read())
    except KnowledgeBaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _chat_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    assembled = ""
    sources: list[dict[str, Any]] = []
    agent_steps: list[dict[str, Any]] = []
    for event in get_orchestrator().stream(request.message):
        event_type = event.get("type")
        if event_type == "content_chunk":
            assembled = str(event.get("assembled", assembled))
        elif event_type == "retrieval":
            sources = list(event.get("sources", sources))
        elif event_type == "agent_step":
            agent_steps.append({key: value for key, value in event.items() if key != "type"})
        elif event_type == "final":
            assembled = str(event.get("content", assembled))
            sources = list(event.get("sources", sources))
            agent_steps = list(event.get("agent_steps", agent_steps))
    session_id = save_chat_session(
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=assembled,
        sources=sources,
        agent_steps=agent_steps,
    )
    return ChatResponse(output=assembled, sources=sources, agent_steps=agent_steps, session_id=session_id)


@app.get("/api/chat/sessions", response_model=ChatSessionListResponse)
def chat_sessions() -> ChatSessionListResponse:
    return ChatSessionListResponse(sessions=list_chat_sessions())


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSessionResponse)
def chat_session_detail(session_id: str) -> ChatSessionResponse:
    try:
        return get_chat_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/chat/sessions/{session_id}")
def chat_session_delete(session_id: str) -> dict[str, bool]:
    try:
        delete_chat_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/datasets", response_model=DatasetListResponse)
def datasets() -> DatasetListResponse:
    return DatasetListResponse(datasets=list_datasets())


@app.post("/api/datasets/upload", response_model=DatasetUploadResponse)
async def dataset_upload(file: UploadFile = File(...)) -> DatasetUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少上传文件名。")
    try:
        return DatasetUploadResponse(dataset=upload_dataset(file.filename, await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/summary")
def dataset_summary(dataset_id: str):
    try:
        return get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/datasets/{dataset_id}")
def dataset_delete(dataset_id: str) -> dict[str, bool]:
    try:
        delete_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/experiments", response_model=ExperimentListResponse)
def experiments() -> ExperimentListResponse:
    return ExperimentListResponse(experiments=list_experiments())


@app.post("/api/experiments", response_model=ExperimentResponse)
def experiment_create(request: ExperimentCreateRequest) -> ExperimentResponse:
    try:
        return create_experiment(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
def experiment_detail(experiment_id: str) -> ExperimentResponse:
    try:
        return get_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/experiments/{experiment_id}")
def experiment_delete(experiment_id: str) -> dict[str, bool]:
    try:
        delete_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/experiments/{experiment_id}/run", response_model=JobResponse)
def experiment_run(experiment_id: str) -> JobResponse:
    try:
        return run_experiment_job(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/experiments/{experiment_id}/results")
def experiment_results(experiment_id: str):
    try:
        experiment = get_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "experiment": experiment,
        "result": experiment.result,
        "explanation": experiment.explanation,
        "report_path": experiment.report_path,
    }


@app.delete("/api/experiments/{experiment_id}/results", response_model=ExperimentResponse)
def experiment_results_delete(experiment_id: str) -> ExperimentResponse:
    try:
        return delete_experiment_results(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/experiments/{experiment_id}/explain", response_model=JobResponse)
def experiment_explain(experiment_id: str) -> JobResponse:
    try:
        return explain_experiment_job(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs", response_model=list[JobResponse])
def jobs() -> list[JobResponse]:
    return list_jobs()


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def job_detail(job_id: str) -> JobResponse:
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/reports/{experiment_id}/download")
def report_download(experiment_id: str) -> FileResponse:
    try:
        report_path = generate_report(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = Path(report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="报告不存在。")
    return FileResponse(path, filename=path.name, media_type="text/markdown")
