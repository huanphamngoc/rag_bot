-- =====================================================================
-- V005: rag.document becomes a Type 2 dimension.
--
-- Until now a document was overwritten in place, so "what did this recall notice say in March?" had
-- no answer: the FDA changes a classification or a reason, and the old wording was gone. From here on
-- every change to the tracked text closes the current row (valid_to, is_current = false) and opens a
-- new one, numbered from 1 per document.
--
-- The business key stays rag.document.doc_id ("<doc_type>:<record key>"); the primary key becomes the
-- surrogate key doc_sk, and rag.chunk hangs off doc_sk instead of doc_id. Chunks exist only for the
-- current version: when a new version opens they are either re-chunked (text changed, vectors of
-- unchanged chunk text carried over) or simply moved (only the soft-delete flag changed - identical
-- text, so nothing is embedded again).
--
-- This runs in one transaction on an index that already holds 27,989 documents and 35,796 embedded
-- chunks (measured 2026-10-05), and it must not lose a single vector: the chunk rows are never
-- rewritten, only pointed at a different column.
-- =====================================================================

-- ---- 1. the surrogate key -------------------------------------------------
ALTER TABLE rag.document ADD COLUMN doc_sk bigint;
CREATE SEQUENCE rag.document_doc_sk_seq OWNED BY rag.document.doc_sk;
ALTER TABLE rag.document ALTER COLUMN doc_sk SET DEFAULT nextval('rag.document_doc_sk_seq');
UPDATE rag.document SET doc_sk = nextval('rag.document_doc_sk_seq');
ALTER TABLE rag.document ALTER COLUMN doc_sk SET NOT NULL;

-- ---- 2. the Type 2 columns ------------------------------------------------
ALTER TABLE rag.document
    ADD COLUMN version          integer     NOT NULL DEFAULT 1 CHECK (version >= 1),
    ADD COLUMN is_current       boolean     NOT NULL DEFAULT true,
    ADD COLUMN valid_from       timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN valid_to         timestamptz,
    ADD COLUMN change_reason    text        NOT NULL DEFAULT 'first version',
    ADD COLUMN source_change_id bigint,                 -- crawl.record_change.change_id this reflects
    ADD COLUMN batch_id         bigint,                 -- ingest.batch that wrote it
    ADD CONSTRAINT document_closed_after_opening CHECK (valid_to IS NULL OR valid_to >= valid_from),
    ADD CONSTRAINT document_current_is_open      CHECK (is_current = (valid_to IS NULL));

-- The rows that existed before this migration are version 1 and have been valid since they were built.
UPDATE rag.document SET valid_from = created_at;

-- ---- 3. the key swap ------------------------------------------------------
-- Dropping rag.chunk.doc_id takes its foreign key, its UNIQUE (doc_id, ord) and chunk_doc_idx with it,
-- so the new column is filled first.
ALTER TABLE rag.chunk ADD COLUMN doc_sk bigint;
UPDATE rag.chunk c SET doc_sk = d.doc_sk FROM rag.document d WHERE d.doc_id = c.doc_id;
ALTER TABLE rag.chunk ALTER COLUMN doc_sk SET NOT NULL;
ALTER TABLE rag.chunk DROP COLUMN doc_id;

ALTER TABLE rag.document DROP CONSTRAINT document_pkey;
ALTER TABLE rag.document ADD CONSTRAINT document_pkey PRIMARY KEY (doc_sk);

ALTER TABLE rag.chunk
    ADD CONSTRAINT chunk_doc_sk_fkey FOREIGN KEY (doc_sk) REFERENCES rag.document (doc_sk) ON DELETE CASCADE,
    ADD CONSTRAINT chunk_doc_sk_ord_key UNIQUE (doc_sk, ord);
CREATE INDEX chunk_doc_idx ON rag.chunk (doc_sk);

-- ---- 4. the indexes the new shape needs -----------------------------------
CREATE UNIQUE INDEX document_version_idx ON rag.document (doc_id, version);
-- The invariant everything else leans on: one open version per document.
CREATE UNIQUE INDEX document_current_idx ON rag.document (doc_id) WHERE is_current;

DROP INDEX rag.document_doc_type_idx;
DROP INDEX rag.document_pending_idx;
CREATE INDEX document_doc_type_idx ON rag.document (doc_type) WHERE is_current AND is_active;
CREATE INDEX document_pending_idx  ON rag.document (doc_type) WHERE is_current AND indexed_at IS NULL;
CREATE INDEX document_history_idx  ON rag.document (doc_id, version DESC) WHERE NOT is_current;

-- ---- 5. what retrieval and the CLI read ----------------------------------
CREATE VIEW rag.v_document_current AS
SELECT doc_sk, doc_id, doc_type, source_id, title, body, url, metadata, content_hash, is_active,
       version, valid_from, change_reason, indexed_at, created_at, updated_at
  FROM rag.document
 WHERE is_current;

COMMENT ON COLUMN rag.document.doc_sk IS 'surrogate key: one row per version of a document';
COMMENT ON COLUMN rag.document.doc_id IS 'business key: <doc_type>:<record key of the source record>';
COMMENT ON COLUMN rag.document.content_hash IS 'sha256 of the attributes rules/*.yaml marks as tracked';
