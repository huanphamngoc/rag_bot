# crawler-rag

Grounded question answering over the FDA drug recalls and CPSC consumer product recalls collected by the
crawler project (`E:\Job\crawler`). It owns its own vector database (Postgres 16 + pgvector) and loads
**incrementally** from the crawler's change log: each run reads only the records that changed, and embeds
only the text that changed.

```
app/rules/*.yaml ──► doc types · quality rules · the question gate
                     │
crawler Postgres ──(analyst_ro, one snapshot)──► ingest graph ──► pgvector ──► chat graph ──► web / CLI
  crawl.record_change = watermark          plan→extract→quality→   rag.document   qualify→condense→
  38 tables → meta.*                       stage(SCD2)→embed       one row/version  retrieve→answer
```

- **The business rules are YAML** (`app/rules/`): one document type is one file, and no SQL is left in Python.
- **The metadata is read from Postgres** (`meta.*`): the YAML is checked against the real tables, columns, keys and relationships before any data is read.
- **`rag.document` is SCD Type 2**: every version is kept, and retrieval reads only the current one.
- **LangGraph** drives both flows; the question flow has a **qualify** step that stops questions that should not reach the model.
- **Self-hosted MLflow** (`:5001`) records every answer: spans, tokens, cost — a blocked question produces a trace with no model span in it at all.

The design, and the evidence behind it: [docs/DESIGN.md](docs/DESIGN.md).

## What you need

- Docker Desktop.
- The crawler stack running (`E:\Job\crawler`: `docker compose up -d postgres`), Postgres on port 5433, role `analyst_ro`.
- A model: Vertex AI by default, using this machine's `gcloud auth application-default login` (`GCLOUD_CONFIG` in `.env`).

## Getting started

```bash
cp .env.example .env            # fill in VECTORDB_PASSWORD, SOURCE_PG_PASSWORD (the crawler's ANALYST_RO_PASSWORD), GCLOUD_CONFIG
docker compose up -d vectordb
docker compose run --rm app migrate
docker compose run --rm app init           # probe the model once, fix vector(1536), build the HNSW index
docker compose run --rm app catalog        # read the metadata out of the crawler's database (read only)
docker compose run --rm app rules-check    # check rules/*.yaml against that metadata
docker compose run --rm app plan           # read only: the mode, the change_id window, the chunks and characters to embed
docker compose run --rm app ingest         # load (the first run is full; every run after it is incremental)
docker compose up -d web                   # http://localhost:8089
```

## Day to day

| What | Command |
|---|---|
| Load what is new after a crawler run | `docker compose run --rm app ingest` |
| Load every 15 minutes by itself | `docker compose --profile scheduler up -d` |
| The watermark against the source, and recent batches | `docker compose run --rm app status` |
| Reconcile everything again (after a `crawler rebuild`) | `docker compose run --rm app ingest --full` |
| Build documents now, embed later / cap the cost | `ingest --no-embed`, `embed --max-chunks 5000` |
| Ask | `docker compose run --rm app ask "..."`, `app chat`, `app search "..."` |
| Look at the traces of the answers | `docker compose --profile mlflow up -d` → http://localhost:5001 |
| Score one answer onto its trace | `docker compose run --rm app feedback 3 good` |
| Tests | `docker compose --profile test run --rm test` |

After editing `app/rules/`:

| What | Command |
|---|---|
| Check the rules against the real schema (tables, columns, keys, relationships) | `docker compose run --rm app rules-check` |
| Read the metadata again, look at it or export it | `app catalog`, `app catalog --export catalog.yaml` |
| Every version of one document | `app history "drug_recall:D-0853-2026"` |
| Try the question gate, with no model call | `app qualify "how many recalls in 2026?"` |

`app/rules/` is mounted into the container, so **the image does not need rebuilding** when a rule changes.
Edit a file that affects document text and the next `ingest` switches to `full` by itself (the watermark
signature contains a digest of the file) — but it still re-chunks only the documents that really changed.

## Tracing with MLflow

```bash
# MLFLOW_DB_PASSWORD in .env (any string), then:
docker compose --profile mlflow up -d --build     # UI: http://localhost:5001
# MLFLOW_TRACKING_URI=http://mlflow:5000 in .env turns tracing on, then restart web
docker compose up -d web
```

MLflow 3.16.1, self-hosted, with its traces in a separate `mlflow` database inside the `vectordb` container.
Port 5001 because the crawler project's own MLflow already holds 5000. MLflow's telemetry is off.

Every answer is one trace: `rag_ask` → `qualify_question` (GUARDRAIL) → `hybrid_retrieve` (plus embed/vector/text)
→ `generate_answer`, with tokens and the **cost** the server works out itself. A question the gate stops
produces a trace with **2 spans and no model span** — which is how "did this question cost anything?" gets
answered from the trace itself. The `qualify_decision` and `qualify_rule` tags are searchable in the trace list.

## Approving the embedding bill

Embedding is the only step that costs money. `INGEST_EMBED_APPROVAL_CHUNKS=50` in `.env` means: when a run
would have to embed more than 50 new chunks, it **stops and waits for you** instead of paying by itself.

```
⏸  Paused before embedding: 132 new chunks (144,251 characters) is over the threshold of 50.
   Documents and the watermark are written; only the embedding step is waiting for you.
   Approve: crawlerrag approve 4d07233cf13e4e459d7f10dc443bd615
   Refuse:  crawlerrag approve 4d07233cf13e4e459d7f10dc443bd615 --no
```

When it pauses, the documents and the watermark are **already written** — waiting costs none of the work
done so far, and the next run does not read that window again. The number printed is what will be charged
(identical text is bought once). Refusing is not an error: the chunks stay pending and `crawlerrag embed`
takes them whenever you want.

Set it to `0` to switch the gate off. It needs `INGEST_GRAPH_CHECKPOINT=true` (the default), because a pause
that cannot be stored cannot be answered.

## When the question is not clear, it asks back

Type `insulin` — a subject, not yet a question — and the turn **pauses** instead of being refused:

```
That is a subject, not a question yet. Here is what the records actually say about it.

  • Why did Eli Lilly & Company recall a drug?      FDA drug recall D-0445-2024 - Eli Lilly & Company
  • Why did Novo Nordisk Inc recall a drug?         FDA drug recall D-0615-2021 - Novo Nordisk Inc
```

Click one (or type your own) and **the waiting turn** carries on — not a new one. The examples are built
from records that are **really in the index**, from the word you just typed, so they can always be answered
and never drift as the data changes. Building them uses the lexical side only, with no embedding call, so
asking back is free.

Only a **vague** question is asked about. Out of scope, prompt injection, or a request for help doing harm
is refused and closed — inviting someone to reword a question about building a weapon is inviting them to
try again.

Turn it off with `RAG_CHAT_CLARIFY=false` in `.env` (needs `docker compose up -d web` to take effect).

## The sources shown are the ones the answer used

Retrieval gives the model eight excerpts, and the answer usually leans on one. Measured over 14 real
answers, eight were retrieved every time and the median answer cited one — so the list used to show
seven records the answer never touched.

Now it shows what was cited, with the rest one click away under *"N more retrieved, not cited in the
answer"*. Nothing is dropped: what was retrieved is how a wrong answer gets explained. In the CLI,
`--show-sources` still prints every excerpt in full.

## Streaming on the web page

The page calls `POST /api/ask/stream` and shows the answer **while the model writes it**, then rebuilds the
finished bubble (citations, sources, tokens, the trace link) when it is done. The conversation is still
kept, so the next question continues in the same session.

A question the gate stops streams nothing at all — because no model is called. A provider that cannot
stream (ollama, OpenAI-shaped endpoints) still works, just without the typing effect.

## Upgrading an index that already exists

V005 turns `rag.document` into SCD Type 2 and moves `rag.chunk` from `doc_id` to `doc_sk`. It rewrites **no**
chunk rows, so the vectors (and the HNSW entries) are kept.

**Migrate before rebuilding the image.** A new image against an old database returns HTTP 500
(`column "is_current" does not exist`).

```bash
# a way back: an un-migrated copy, a few seconds, and it does not touch the original
docker exec crawler-rag-vectordb psql -U rag -d postgres -c "CREATE DATABASE rag_pre_scd2 TEMPLATE rag"
docker compose run --rm app migrate
docker compose run --rm app plan         # expect: mode full, 0 chunks to embed
docker compose run --rm app ingest
```

Run for real on the 793 MB index (see the results below): 35,796/35,796 vectors kept, and the first `ingest`
afterwards made **0 embedding requests** in 9.1 s.

## Results (2026-10-04, the crawler's real data, on this machine)

**The first load (full)**

| Step | Result |
|---|---|
| `plan` (read only) | drug_recall 17,937 keys → 18,771 chunks to embed, 17.5M characters; cpsc_recall 10,002 keys → 16,931 chunks, 18.5M characters |
| `ingest --no-embed` | 27,939 documents, 35,746 chunks, 199,310 lexemes in 29 s (batch 1: 18.0 s, batch 2: 11.1 s) |
| Against the crawler's own older index | 27,939/27,939 documents and 35,746/35,746 chunks matched by hash: the text is identical byte for byte |
| Embedding, run 1 | stopped after 11,136 texts / 174 requests / 19 minutes on Vertex AI's per-minute quota (HTTP 429); what was embedded was kept |
| Embedding, run 2 | 24,616 texts / 385 requests / 41 minutes, `succeeded`. 184 transient 429s, all handled by the provider's retry; the new quota-wait mechanism (`quota_waits`) was never needed |
| The embedding total | 35,752 distinct texts, 36.0M characters, **11,092,546 tokens** (as Vertex reported), 0 texts truncated; about **$1.66** at $0.00015 per 1,000 tokens |
| Size | vector database 793 MB, of which HNSW is 279 MB |

**An incremental load (real)**

1. The crawler ran `crawl openfda_enforcement --mode incremental` (run #20): 2 HTTP requests, 123 records → 50 new,
   7 changed, 66 unchanged; 57 rows added to the change log.
2. `plan`: change_id window (155,804, 1,528,261], 57 keys → 50 new documents, 6 changed, **1 unchanged** (the source
   record changed but the document text did not), 56 chunks to embed.
3. `ingest`: exactly as planned, and the batch took **0.077 s** (the full batch took 18 s); cpsc_recall was `nothing`.
4. `ingest` again: both `nothing`, 0 model calls, 2.2 s including container start-up.

**Answering** (web on `:8089`, Vertex `gemini-2.5-flash`):

- "Why did Pfizer recall a drug?" with the filter `year=2026` → source [1] is `D-0853-2026`, the record that arrived
  in the incremental batch, found by both retrievers; the answer was "Lack of Assurance of Sterility [1]"
  (3.9 s; 4,073 / 147 tokens).
- The same question written "in September 2026", with **no** filter → that record is not found: older Pfizer recalls
  rank above it, and the date in the text is written `2026-09-04`. The model correctly said the retrieved excerpts
  do not contain the information rather than inventing it. Questions about time currently need the `year` filter
  (docs/DESIGN.md §11).

## Results (2026-10-05, after the move to YAML + SCD2 + LangGraph)

A **read-only** run against the real crawler database and the running index — nothing written, nothing migrated:

| Measured | Result |
|---|---|
| Catalog of `crawl`, `drug`, `retail` | 38 tables, 383 columns, 25 relationships (every one a real declared foreign key) |
| `rules-check` against that catalog | **0 errors, 0 warnings** |
| Rebuilding every document from the YAML and comparing `content_hash` with the running index | **27,989/27,989 identical**, 0 different → moving to YAML cost nothing in embedding |
| The index as it stands | 27,989 documents, 35,796 chunks, 35,796 vectors, 793 MB, at V004 |
| The qualify gate over 10 real questions | all 10 classified correctly (the full table is in docs/DESIGN.md §9) |

Something the real questions turned up: Vietnamese typed **without diacritics** ("Tong so vu thu hoi ... la bao
nhieu?") was classified as out of scope at first. Rule matching now folds the diacritics away on both sides, while
what the user typed is never changed.

### The migration and tracing, run for real on the 793 MB index

| Step | Result |
|---|---|
| `migrate` (after taking the `rag_pre_scd2` copy as a way back) | all 27,989 documents `is_current` at version 1; **35,796/35,796 vectors kept**; 793 MB → 905 MB (the new indexes) |
| `plan` | mode `full`, **0 new, 0 changed, 27,989 unchanged, 0 chunks to embed** (7.9 s) |
| The first `ingest` | exactly as `plan` said, **0 embedding requests** (9.1 s) |
| Every `ingest` after it | `incremental` / `nothing`, 0 model calls |
| The quality gate | found 1 row in 17,987 with `classification = "Not Yet Classified"` — a value the FDA really uses; the rule was corrected to accept it |
| Fixing that quality rule and running again | back to `incremental` (the digest did not change) and in effect **with no image rebuild** |
| The web page | `/api/ask` answered with `D-0853-2026` and its `trace_url` |

**Traces, measured** (real Vertex AI, cost worked out by MLflow):

| Question | Decision | Spans | Time | Cost |
|---|---|---|---|---|
| Why did Pfizer recall a drug in 2026? | `pass` | 7 | 9.4 s | $0.00289 |
| What hazard did CPSC report for the stroller? | `pass` | 7 | 7.3 s | $0.00293 |
| How many drug recalls were there in 2026? | `needs_sql` | **2** | 0.49 s | **0** |
| Ignore all previous instructions… | `reject` | **2** | 0.51 s | **0** |
| What is the capital of France? | `reject` | **2** | 0.70 s | **0** |

**Tests:** 686 pass (`docker compose --profile test run --rm test`), up from 144.
