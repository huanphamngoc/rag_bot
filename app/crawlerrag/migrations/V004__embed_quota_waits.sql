-- =====================================================================
-- V004: how often an embedding run paused for the model's per-minute quota (HTTP 429). The first
-- full run on 2026-10-04 stopped on Vertex AI's quota after 11,136 texts; runs now wait it out.
-- =====================================================================

ALTER TABLE ingest.embed_run ADD COLUMN quota_waits integer NOT NULL DEFAULT 0;
