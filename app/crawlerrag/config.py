"""Runtime configuration, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # The vector database this project owns (Postgres 16 + pgvector): documents, chunks, vectors,
    # conversations, and the ingestion watermarks.
    database_url: str = "postgresql://rag:rag@localhost:5434/rag"
    # The crawler's database, read-only (role analyst_ro): the change log crawl.record_change and the
    # normalised tables the documents are built from. Never written to.
    source_database_url: str = "postgresql://analyst_ro:analyst_ro@localhost:5433/crawler"
    log_level: str = "INFO"
    log_format: str = "json"

    # ---- models: configuration, not code ("openai" covers every OpenAI-compatible endpoint) ----
    rag_embed_provider: str = Field("gemini", pattern="^(gemini|openai|ollama|vertex)$")
    rag_embed_model: str = "gemini-embedding-001"
    rag_embed_base_url: str | None = None
    rag_embed_api_key: str | None = None
    # gemini-embedding-001 returns 3072 dims but pgvector's HNSW index stops at 2000, so the vector is
    # truncated (cosine ranking ignores the missing re-normalisation).
    rag_embed_dim: int | None = Field(None, ge=1, le=16000)
    # Texts per embedding request. Vertex AI's reference says gemini-embedding-001 takes one text per
    # request, but on 2026-10-04 it answered 64 texts with 64 distinct vectors (docs/DESIGN.md 5.3).
    rag_embed_batch: int = Field(64, ge=1, le=256)
    rag_chat_provider: str = Field("gemini", pattern="^(gemini|openai|anthropic|ollama|vertex)$")
    rag_chat_model: str = "gemini-flash-latest"
    rag_chat_base_url: str | None = None
    rag_chat_api_key: str | None = None
    rag_chat_max_tokens: int = Field(4096, ge=64, le=32000)   # ceiling; Gemini 2.5 thinking counts against it
    rag_chat_temperature: float = Field(0.1, ge=0, le=2)
    rag_chat_thinking_budget: int | None = Field(None, ge=0, le=24576)
    rag_vertex_project: str | None = None
    rag_vertex_location: str = "us-central1"
    rag_api_timeout_s: float = Field(120.0, gt=0)
    rag_max_attempts: int = Field(5, ge=1, le=20)

    # ---- documents and retrieval ----
    rag_chunk_chars: int = Field(1600, ge=200, le=8000)
    rag_chunk_overlap: int = Field(200, ge=0, le=2000)
    rag_top_k: int = Field(8, ge=1, le=50)                    # excerpts handed to the model
    rag_candidates: int = Field(40, ge=1, le=500)              # per retriever, before fusion
    rag_doc_types: str = "drug_recall,cpsc_recall"            # what `ingest` builds by default
    rag_web_daily_limit: int = Field(300, ge=0, le=1_000_000)  # web chat questions per UTC day, 0 = no cap

    # ---- business rules (YAML) ----
    # The folder that describes the document types, the quality rules and the chat input gate. Mounted
    # into the container, so it can be edited without rebuilding the image.
    rules_dir: str = "rules"
    # Ask again, with examples from the index, when a question names a subject but asks nothing. The
    # turn pauses (a LangGraph interrupt) and resumes with the corrected question, so it needs the
    # checkpointer tables in the `graph` schema. Off = such a question gets its message and ends there.
    rag_chat_clarify: bool = True
    # Pause the ingestion run before the only step that costs money, when it would embed more than this
    # many distinct chunk texts. 0 = never pause. The pause is a LangGraph interrupt, so it needs the
    # checkpointer (INGEST_GRAPH_CHECKPOINT): without somewhere to store the state there would be
    # nothing to resume. `crawlerrag embed` is the deliberate way past the gate.
    ingest_embed_approval_chunks: int = Field(0, ge=0)

    # ---- ingestion ----
    ingest_page_docs: int = Field(2000, ge=1, le=100_000)     # documents written per target transaction
    ingest_interval_s: int = Field(900, ge=60)                 # `ingest --loop`: pause between incremental runs
    # Store the LangGraph state after each node in the `graph` schema, so a run that died inside a node
    # can be resumed there. What gets *read* from the source is still decided by ingest.watermark.
    ingest_graph_checkpoint: bool = True

    @field_validator("rag_embed_base_url", "rag_embed_api_key", "rag_embed_dim", "rag_chat_base_url",
                     "rag_chat_api_key", "rag_vertex_project", "rag_chat_thinking_budget", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
