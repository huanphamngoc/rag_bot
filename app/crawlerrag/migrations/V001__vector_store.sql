-- =====================================================================
-- V001: the vector store - documents, chunks (text + full-text vector + embedding), index state,
-- conversations and the question log. Same layout as the crawler's rag schema (its V009-V013),
-- except that rag.document.source_id is plain text: the sources live in another database.
--
-- The embedding column has no dimension yet: `crawlerrag init` probes the configured model,
-- pins vector(n) and builds the HNSW index.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS rag;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag.document (
    doc_id        text        PRIMARY KEY,                 -- "<doc_type>:<record_key of the source record>"
    doc_type      text        NOT NULL,
    source_id     text        NOT NULL,                    -- crawler source (crawl.source.source_id there)
    title         text        NOT NULL,
    body          text        NOT NULL,
    url           text,
    metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb, -- structured filters (state, year, classification, ...)
    content_hash  char(64)    NOT NULL,                    -- sha256(title || body): unchanged text is never re-chunked
    is_active     boolean     NOT NULL DEFAULT true,        -- follows the soft-delete flag of the source row
    indexed_at    timestamptz,                             -- set when every chunk has an embedding
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX document_doc_type_idx  ON rag.document (doc_type) WHERE is_active;
CREATE INDEX document_pending_idx   ON rag.document (doc_type) WHERE indexed_at IS NULL;
CREATE INDEX document_metadata_idx  ON rag.document USING gin (metadata jsonb_path_ops);

CREATE TABLE rag.chunk (
    chunk_id     bigserial   PRIMARY KEY,
    doc_id       text        NOT NULL REFERENCES rag.document(doc_id) ON DELETE CASCADE,
    ord          smallint    NOT NULL,
    text         text        NOT NULL,
    text_hash    char(64)    NOT NULL,                     -- identical chunk text re-uses an existing embedding
    tsv          tsvector    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding    vector,
    embed_model  text,
    embedded_at  timestamptz,
    UNIQUE (doc_id, ord)
);

CREATE INDEX chunk_tsv_idx       ON rag.chunk USING gin (tsv);
CREATE INDEX chunk_pending_idx   ON rag.chunk (chunk_id) WHERE embedding IS NULL;
CREATE INDEX chunk_text_hash_idx ON rag.chunk (text_hash) WHERE embedding IS NOT NULL;
CREATE INDEX chunk_doc_idx       ON rag.chunk (doc_id);

-- Which provider/model the stored vectors come from; `init` refuses to mix two vector spaces.
CREATE TABLE rag.index_state (
    id             smallint    PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    embed_provider text        NOT NULL,
    embed_model    text        NOT NULL,
    embed_dim      integer     NOT NULL CHECK (embed_dim BETWEEN 1 AND 16000),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- Chunks per lexeme: the keyword half of the hybrid search skips words found in most chunks.
CREATE TABLE rag.lexeme_stat (
    lexeme  text    PRIMARY KEY,
    ndoc    integer NOT NULL
);

CREATE TABLE rag.conversation (
    conversation_id  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    title            text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE rag.query_log (
    query_id            bigserial   PRIMARY KEY,
    asked_at            timestamptz NOT NULL DEFAULT now(),
    question            text        NOT NULL,
    doc_types           text[],
    top_k               smallint,
    retrieved           jsonb       NOT NULL DEFAULT '[]'::jsonb,   -- [{rank, chunk_id, doc_id, score, matched_by}]
    answer              text,
    embed_provider      text,
    embed_model         text,
    chat_provider       text,
    chat_model          text,
    prompt_tokens       integer,
    output_tokens       integer,
    duration_ms         integer,
    error               text,
    trace_id            text,                                       -- MLflow trace id (tr-...) when tracing is on
    conversation_id     uuid REFERENCES rag.conversation (conversation_id) ON DELETE CASCADE,
    turn                smallint,
    standalone_question text
);

CREATE INDEX query_log_asked_at_idx ON rag.query_log (asked_at DESC);
CREATE INDEX query_log_trace_id_idx ON rag.query_log (trace_id) WHERE trace_id IS NOT NULL;
CREATE UNIQUE INDEX query_log_conversation_turn_idx
    ON rag.query_log (conversation_id, turn) WHERE conversation_id IS NOT NULL;
