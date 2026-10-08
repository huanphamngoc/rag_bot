-- The crawler tables the ingestion reads, with the columns, types and keys of the crawler's database
-- (read from information_schema of E:/Job/crawler on 2026-10-04). Recreated by every integration run,
-- plus the read-only role analyst_ro with the grants the crawler's V005/V008 give it.
DROP SCHEMA IF EXISTS crawl, drug, retail CASCADE;
CREATE SCHEMA crawl;
CREATE SCHEMA drug;
CREATE SCHEMA retail;

CREATE TABLE crawl.crawl_run (
    run_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id  text NOT NULL,
    status     text NOT NULL DEFAULT 'running'
               CHECK (status IN ('running', 'succeeded', 'partial', 'failed', 'aborted'))
);
CREATE UNIQUE INDEX crawl_run_one_running_per_source ON crawl.crawl_run (source_id) WHERE status = 'running';

CREATE TABLE crawl.record_change (
    change_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id       text        NOT NULL,
    record_key      text        NOT NULL,
    run_id          bigint      NOT NULL REFERENCES crawl.crawl_run (run_id),
    change_type     text        NOT NULL CHECK (change_type IN ('insert', 'update', 'deactivate', 'reactivate')),
    old_hash        char(64),
    new_hash        char(64),
    changed_fields  text[],
    changed_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX record_change_key_idx ON crawl.record_change (source_id, record_key);

CREATE TABLE drug.recall (
    recall_number text PRIMARY KEY, event_id text, status text, classification text, product_type text,
    recalling_firm text, address_1 text, address_2 text, city text, state text, postal_code text, country text,
    voluntary_mandated text, initial_firm_notification text, distribution_pattern text, product_description text,
    product_quantity text, reason_for_recall text, code_info text, more_code_info text, recall_initiation_date date,
    center_classification_date date, report_date date, termination_date date,
    is_active boolean NOT NULL DEFAULT true
);
CREATE TABLE drug.recall_product_ndc (
    recall_number text NOT NULL, product_ndc text NOT NULL, PRIMARY KEY (recall_number, product_ndc)
);

CREATE TABLE retail.cpsc_recall (
    recall_id text PRIMARY KEY, recall_number text, recall_date date, last_publish_date date, title text,
    description text, url text, consumer_contact text, injuries text[] NOT NULL DEFAULT '{}',
    remedies text[] NOT NULL DEFAULT '{}', related_urls text[] NOT NULL DEFAULT '{}', units_total_approx bigint,
    is_active boolean NOT NULL DEFAULT true
);
CREATE TABLE retail.cpsc_recall_hazard (
    recall_id text NOT NULL, seq smallint NOT NULL, description text NOT NULL, hazard_type text, hazard_type_id text,
    PRIMARY KEY (recall_id, seq)
);
CREATE TABLE retail.cpsc_recall_product (
    recall_id text NOT NULL, seq smallint NOT NULL, name text, description text, model text, product_type text,
    category_id text, number_of_units text, units_approx bigint, PRIMARY KEY (recall_id, seq)
);
CREATE TABLE retail.cpsc_recall_company (
    recall_id text NOT NULL, role text NOT NULL, seq smallint NOT NULL, name text NOT NULL, company_id text,
    price_min numeric, price_max numeric, PRIMARY KEY (recall_id, role, seq)
);
CREATE TABLE retail.cpsc_recall_major_retailer (
    recall_id text NOT NULL, retailer text NOT NULL, PRIMARY KEY (recall_id, retailer)
);
CREATE TABLE retail.cpsc_recall_remedy_option (
    recall_id text NOT NULL, seq smallint NOT NULL, option text NOT NULL, PRIMARY KEY (recall_id, seq)
);
CREATE TABLE retail.cpsc_recall_country (
    recall_id text NOT NULL, seq smallint NOT NULL, country text NOT NULL, PRIMARY KEY (recall_id, seq)
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analyst_ro') THEN
        CREATE ROLE analyst_ro LOGIN PASSWORD 'analyst_ro';
    END IF;
END $$;
GRANT USAGE ON SCHEMA crawl, drug, retail TO analyst_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA crawl, drug, retail TO analyst_ro;
