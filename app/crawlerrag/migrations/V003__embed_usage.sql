-- =====================================================================
-- V003: what the embedding provider itself reports per run. Vertex AI returns, for every text,
-- statistics.token_count (billed tokens) and statistics.truncated (text cut at the model's input
-- limit); NULL for providers that report neither.
-- =====================================================================

ALTER TABLE ingest.embed_run
    ADD COLUMN input_tokens bigint,
    ADD COLUMN truncated    integer;
