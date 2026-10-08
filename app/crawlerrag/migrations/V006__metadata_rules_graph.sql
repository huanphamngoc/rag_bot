-- =====================================================================
-- V006: three things the YAML rules and the LangGraph flows need.
--
-- 1. meta.*     the metadata extracted from the crawler's Postgres - tables, columns, keys and
--               relationships - so a rule file that names a column can be checked against the database
--               before a single row is read. Relationships are stored with how they were found:
--               'declared' (a foreign key) or 'inferred' (the child's key starts with the parent's).
-- 2. ingest.quality_finding   what the business rules in rules/*.yaml found on the extracted rows.
-- 3. graph      the schema the LangGraph Postgres checkpointer creates its own tables in, and two
--               columns on rag.query_log recording what the qualify node decided about a question.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS graph;

-- One row per extraction; the digest covers the structure only, so an unchanged schema keeps it.
CREATE TABLE meta.catalog_run (
    run_id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    captured_at        timestamptz NOT NULL,
    source             text,                              -- how it was captured (cli, ingest, test)
    digest             char(64)    NOT NULL,
    table_count        integer     NOT NULL,
    column_count       integer     NOT NULL,
    relationship_count integer     NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE meta.table_info (
    run_id       bigint   NOT NULL REFERENCES meta.catalog_run (run_id) ON DELETE CASCADE,
    schema_name  text     NOT NULL,
    table_name   text     NOT NULL,
    kind         text     NOT NULL,                       -- table / view / materialized view
    primary_key  text[]   NOT NULL DEFAULT '{}',
    -- Other unique keys, as a list of column lists; jsonb because they differ in width.
    unique_keys  jsonb    NOT NULL DEFAULT '[]'::jsonb,
    est_rows     bigint,                                  -- planner estimate, not a count
    comment      text,
    PRIMARY KEY (run_id, schema_name, table_name)
);

CREATE TABLE meta.column_info (
    run_id       bigint   NOT NULL REFERENCES meta.catalog_run (run_id) ON DELETE CASCADE,
    schema_name  text     NOT NULL,
    table_name   text     NOT NULL,
    column_name  text     NOT NULL,
    ordinal      integer  NOT NULL,
    data_type    text     NOT NULL,
    is_nullable  boolean  NOT NULL,
    has_default  boolean  NOT NULL DEFAULT false,
    max_length   integer,
    precision    integer,
    scale        integer,
    comment      text,
    PRIMARY KEY (run_id, schema_name, table_name, column_name)
);

CREATE TABLE meta.relationship (
    run_id       bigint   NOT NULL REFERENCES meta.catalog_run (run_id) ON DELETE CASCADE,
    from_table   text     NOT NULL,                       -- schema-qualified
    from_columns text[]   NOT NULL,
    to_table     text     NOT NULL,
    to_columns   text[]   NOT NULL,
    kind         text     NOT NULL CHECK (kind IN ('declared', 'inferred')),
    constraint_name text,
    PRIMARY KEY (run_id, from_table, from_columns, to_table)
);

-- What the business rules found. An 'error' failed its batch; a 'warning' only landed here.
CREATE TABLE ingest.quality_finding (
    finding_id   bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id     bigint      NOT NULL REFERENCES ingest.batch (batch_id) ON DELETE CASCADE,
    doc_type     text        NOT NULL,
    level        text        NOT NULL CHECK (level IN ('error', 'warning')),
    rule         text        NOT NULL,
    column_name  text,
    failed_rows  integer,
    checked_rows integer,
    message      text        NOT NULL,
    found_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX quality_finding_batch_idx ON ingest.quality_finding (batch_id);
CREATE INDEX quality_finding_open_idx  ON ingest.quality_finding (doc_type, found_at DESC) WHERE level = 'error';

-- What the gate in front of the model decided, so the cost of a question can be explained afterwards.
ALTER TABLE rag.query_log
    ADD COLUMN qualify_decision text,
    ADD COLUMN qualify_rule     text;

CREATE INDEX query_log_qualify_idx ON rag.query_log (qualify_decision, asked_at DESC)
    WHERE qualify_decision IS DISTINCT FROM 'pass';

COMMENT ON SCHEMA graph IS 'LangGraph checkpointer tables (created by the checkpointer itself)';
COMMENT ON COLUMN rag.query_log.qualify_decision IS 'pass / needs_sql / clarify / reject';
