"""Core domain utilities for the fNIRS self-service platform."""

from .knowledge import (
    DocumentChunk,
    KnowledgeBase,
    KnowledgeBaseError,
    KnowledgeDocument,
    KnowledgeStats,
    SearchResult,
    build_default_knowledge_base,
    document_id_from_source,
    extract_text_from_document,
)

