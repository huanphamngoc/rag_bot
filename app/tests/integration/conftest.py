"""Integration fixtures: a throwaway crawler database (source) and a throwaway pgvector database (target).

Run with:  docker compose --profile test run --rm test
Outside that compose service the IT_* variables are missing and these tests are skipped.

FakeCrawler writes like the real crawler (E:/Job/crawler/app/jobcrawler/store.py): the normalised rows
and their crawl.record_change rows in ONE transaction, one running crawl per source.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from crawlerrag.config import Settings
from crawlerrag.migrate import migrate
from crawlerrag.rag import index
from crawlerrag.rules import clear_cache, load_rules

DSN_ADMIN = os.environ.get("IT_SOURCE_ADMIN_DSN")
DSN_RO = os.environ.get("IT_SOURCE_RO_DSN")
DSN_VECTOR = os.environ.get("IT_VECTOR_DSN")
SCHEMA = Path(__file__).with_name("source_schema.sql")
RULES_DIR = Path(__file__).parents[2] / "rules"
DIM = 8


def pytest_collection_modifyitems(config, items):
    if DSN_ADMIN and DSN_RO and DSN_VECTOR:
        return
    skip = pytest.mark.skip(reason="integration databases not configured (docker compose --profile test run --rm test)")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


def _connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


@pytest.fixture(autouse=True)
def _rules_are_read_fresh():
    """A test that edits a rule must not see the copy another test cached."""
    clear_cache()
    yield
    clear_cache()


class FakeEmbedder:
    """Deterministic vectors from the text's hash; records what it was asked to embed."""
    provider, model, batch_size = "fake", "fake-embed", 64

    def __init__(self, *, fail: bool = False, quota_refusals: int = 0):
        self.fail = fail
        self.quota_refusals = quota_refusals          # this many calls end in HTTP 429 first
        self.texts: list[str] = []
        self.requests = 0

    def embed(self, texts, *, query=False):
        from crawlerrag.rag.providers import ProviderError
        if self.fail:
            raise ProviderError("fake outage")
        if self.quota_refusals:
            self.quota_refusals -= 1
            raise ProviderError("HTTP 429: Quota exceeded for ...embed_content_requests_per_minute...", status=429)
        self.requests += 1
        if not query:
            self.texts.extend(texts)
        self.last_usage = {"tokens": sum(len(t.split()) for t in texts), "truncated": 0}
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            out.append([b / 255 + 0.01 for b in digest[:DIM]])
        return out

    def probe_dim(self):
        return DIM

    def close(self):
        pass


class FakeCrawler:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    @contextlib.contextmanager
    def run(self, source_id: str, status: str = "succeeded"):
        run_id = self.conn.execute("INSERT INTO crawl.crawl_run (source_id) VALUES (%s) RETURNING run_id",
                                   (source_id,)).fetchone()["run_id"]
        yield run_id
        self.conn.execute("UPDATE crawl.crawl_run SET status = %s WHERE run_id = %s", (status, run_id))

    def _log(self, run_id, source_id, key, change_type):
        self.conn.execute("INSERT INTO crawl.record_change (source_id, record_key, run_id, change_type) "
                          "VALUES (%s, %s, %s, %s)", (source_id, key, run_id, change_type))

    def drug(self, run_id, recall_number, *, change="insert", firm="Acme Pharma", reason="Lack of sterility assurance",
             description="Hydromorphone 2 mg/mL injection", classification="Class II", ndcs=(), log=True,
             year=2017):
        with self.conn.transaction():
            self.conn.execute("""
                INSERT INTO drug.recall (recall_number, event_id, status, classification, product_type, recalling_firm,
                                         city, state, country, product_description, reason_for_recall,
                                         recall_initiation_date)
                VALUES (%s, 'E1', 'Ongoing', %s, 'Drugs', %s, 'Little Rock', 'AR', 'United States', %s, %s,
                        make_date(%s, 3, 1))
                ON CONFLICT (recall_number) DO UPDATE SET classification = EXCLUDED.classification,
                    recalling_firm = EXCLUDED.recalling_firm, product_description = EXCLUDED.product_description,
                    reason_for_recall = EXCLUDED.reason_for_recall,
                    recall_initiation_date = EXCLUDED.recall_initiation_date
            """, (recall_number, classification, firm, description, reason, year))
            self.conn.execute("DELETE FROM drug.recall_product_ndc WHERE recall_number = %s", (recall_number,))
            for ndc in ndcs:
                self.conn.execute("INSERT INTO drug.recall_product_ndc VALUES (%s, %s)", (recall_number, ndc))
            if log:
                self._log(run_id, "openfda_enforcement", recall_number, change)

    def cpsc(self, run_id, recall_id, *, change="insert", title="Power banks recalled", description="Short text.",
             url="https://www.cpsc.gov/Recalls/1", hazards=("The battery can overheat, posing fire and burn hazards.",),
             remedies=("Refund",), log=True):
        with self.conn.transaction():
            self.conn.execute("""
                INSERT INTO retail.cpsc_recall (recall_id, recall_number, recall_date, title, description, url,
                                                injuries, remedies, units_total_approx)
                VALUES (%s, %s, '2024-05-02', %s, %s, %s, %s, %s, 12000)
                ON CONFLICT (recall_id) DO UPDATE SET title = EXCLUDED.title, description = EXCLUDED.description,
                    url = EXCLUDED.url, remedies = EXCLUDED.remedies
            """, (recall_id, f"24{recall_id}", title, description, url, ["None reported"], list(remedies)))
            self.conn.execute("DELETE FROM retail.cpsc_recall_hazard WHERE recall_id = %s", (recall_id,))
            for seq, hazard in enumerate(hazards, start=1):
                self.conn.execute("INSERT INTO retail.cpsc_recall_hazard (recall_id, seq, description, hazard_type) "
                                  "VALUES (%s, %s, %s, 'Fire')", (recall_id, seq, hazard))
            if log:
                self._log(run_id, "cpsc_recall", recall_id, change)

    def set_active(self, run_id, source_id, key, active: bool):
        table, column = (("drug.recall", "recall_number") if source_id == "openfda_enforcement"
                         else ("retail.cpsc_recall", "recall_id"))
        with self.conn.transaction():
            self.conn.execute(f"UPDATE {table} SET is_active = %s WHERE {column} = %s", (active, key))
            self._log(run_id, source_id, key, "reactivate" if active else "deactivate")


@pytest.fixture
def env():
    admin = _connect(DSN_ADMIN)
    admin.execute(SCHEMA.read_text(encoding="utf-8"))
    vector = _connect(DSN_VECTOR)
    # Every schema the migrations own, including the LangGraph checkpointer's: schema_migration goes with
    # them, so a leftover meta.catalog_run would make V006 fail on the next run.
    vector.execute("DROP SCHEMA IF EXISTS rag, ingest, meta, graph CASCADE")
    vector.execute("DROP TABLE IF EXISTS public.schema_migration")
    migrate(vector)
    index.init_index(vector, FakeEmbedder(), dim=DIM)
    source = _connect(DSN_RO)
    settings = Settings(_env_file=None, rag_chunk_chars=400, rag_chunk_overlap=60, ingest_page_docs=2000,
                        rag_doc_types="drug_recall,cpsc_recall", rag_top_k=5, rag_candidates=20,
                        rules_dir=str(RULES_DIR), ingest_graph_checkpoint=False,
                        # The checkpointer opens its own connection from the settings, not from the
                        # fixture's, so these have to point at the throwaway databases too.
                        database_url=DSN_VECTOR, source_database_url=DSN_RO)
    try:
        yield {"admin": admin, "crawler": FakeCrawler(admin), "vector": vector, "source": source,
               "settings": settings, "ruleset": load_rules(RULES_DIR)}
    finally:
        for conn in (admin, vector, source):
            conn.close()
