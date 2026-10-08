-- =====================================================================
-- V002: state and audit of the ingestion pipeline (crawlerrag/ingest).
--
-- The watermark is the crawler's own change log position: crawl.record_change.change_id of the
-- source, read in the same snapshot as the rows the documents were built from. It is written in
-- the same transaction as the last page of documents of a batch, so documents and watermark never
-- disagree after a crash (docs/DESIGN.md section 4).
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS ingest;

CREATE TABLE ingest.watermark (
    doc_type    text        PRIMARY KEY,
    source_id   text        NOT NULL,
    change_id   bigint      NOT NULL CHECK (change_id >= 0),   -- last source change reflected in rag.document
    signature   text        NOT NULL,                         -- builder version + chunk settings used
    batch_id    bigint      NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per run of the document stage for one doc type.
CREATE TABLE ingest.batch (
    batch_id        bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_type        text        NOT NULL,
    source_id       text        NOT NULL,
    mode            text        NOT NULL CHECK (mode IN ('full', 'incremental')),
    reason          text        NOT NULL,                  -- why this mode: first load, signature change, requested, ...
    from_change_id  bigint      NOT NULL,                  -- exclusive
    to_change_id    bigint,                                -- inclusive; upper bound read in the source snapshot
    status          text        NOT NULL DEFAULT 'running'
                                CHECK (status IN ('running', 'succeeded', 'failed', 'nothing')),
    keys            integer,                               -- source keys in the window (full: every row)
    inserted        integer,
    updated         integer,                               -- text changed: re-chunked
    refreshed       integer,                               -- only url / metadata changed: no new chunks
    unchanged       integer,
    deactivated     integer,
    reactivated     integer,
    chunks_added    integer,
    chunks_removed  integer,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    error           text
);

CREATE INDEX batch_doc_type_idx ON ingest.batch (doc_type, batch_id DESC);

-- One row per run of the embedding stage.
CREATE TABLE ingest.embed_run (
    run_id         bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    embed_model    text        NOT NULL,
    reused         integer     NOT NULL DEFAULT 0,          -- vectors copied from identical chunk text
    embedded       integer     NOT NULL DEFAULT 0,          -- chunks sent to the model
    requests       integer     NOT NULL DEFAULT 0,
    input_chars    bigint      NOT NULL DEFAULT 0,
    pending_after  integer,
    status         text        NOT NULL DEFAULT 'running'
                               CHECK (status IN ('running', 'succeeded', 'failed', 'stopped')),
    started_at     timestamptz NOT NULL DEFAULT now(),
    finished_at    timestamptz,
    error          text
);
