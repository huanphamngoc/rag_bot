"""CLI handlers for ``python -m crawlerrag ...`` (messages in Vietnamese, like the crawler's CLI)."""
from __future__ import annotations

import logging
import textwrap
import time

import psycopg

from crawlerrag import db
from crawlerrag.ingest import embed as embed_mod
from crawlerrag.ingest import pipeline
from crawlerrag.meta import catalog as meta_catalog
from crawlerrag.meta import introspect
from crawlerrag.rag import answer as answer_mod
from crawlerrag.rag import graph as chat_graph
from crawlerrag.rag import conversation, index, retrieve, tracing
from crawlerrag.rag.providers import ProviderError, chat_provider, embedding_provider

log = logging.getLogger(__name__)

MATCH_ICON = {"both": "◆", "vector": "◇", "text": "▪"}


def _doc_type_names(value: str | None, settings) -> list[str] | None:
    """None means "every type", which is what a search wants by default."""
    if not value:
        return None
    return [d.doc_type for d in pipeline.doc_types_for(settings, value.split(","))]


def _source(settings) -> psycopg.Connection:
    return db.connect(settings.source_database_url, application_name="crawler-rag-ingest")


# ---------------------------------------------------------------- init
def init(settings, conn, args) -> int:
    embedder = embedding_provider(settings)
    try:
        result = index.init_index(conn, embedder, force=args.force, dim=args.dim)
    finally:
        embedder.close()
    print(f"Index ready: {result['provider']}/{result['model']}, {result['dim']} dimensions.")
    if result["vectors_discarded"]:
        print(f"Discarded {result['vectors_discarded']:,} vectors from the previous model - run 'embed' "
              "to rebuild them.")
    return 0


# ---------------------------------------------------------------- ingest / plan / embed
def _print_batches(batches) -> None:
    print(f"{'doc_type':<14}{'batch':>6}  {'mode':<12}{'change_id window':>22}{'key':>8}{'added':>8}{'changed':>9}"
          f"{'meta':>6}{'same':>8}{'off':>6}{'on':>5}{'version':>9}{'chunk+':>8}{'chunk-':>8}{'moved':>8}"
          f"{'kept vecs':>12}")
    for b in batches:
        s = b.stats
        window = f"({b.from_change_id:,}, {b.to_change_id:,}]"
        print(f"{b.doc_type:<14}{b.batch_id:>6}  {b.mode:<12}{window:>22}{b.keys:>8,}{s.inserted:>8,}{s.updated:>9,}"
              f"{s.refreshed:>6,}{s.unchanged:>8,}{s.deactivated:>6,}{s.reactivated:>5,}"
              f"{s.versions_created:>9,}{s.chunks_added:>8,}{s.chunks_removed:>8,}{s.chunks_moved:>8,}"
              f"{s.vectors_carried:>12,}")
        if b.status == "nothing":
            print(f"{'':<20}no new changes ({b.reason})")
        elif b.mode == "full":
            print(f"{'':<20}full reconcile: {b.reason}")


def _print_embed(stats: embed_mod.EmbedStats | None) -> None:
    if stats is None:
        return
    print(f"\nEmbedding (run #{stats.run_id}): {stats.embedded:,} chunks over {stats.requests:,} requests "
          f"({stats.input_chars:,} characters, {stats.seconds}s), reused {stats.reused:,} chunks whose text "
          f"was already embedded, {stats.pending_after:,} still pending.")
    if stats.quota_waits:
        print(f"Waited for the model's per-minute quota: {stats.quota_waits:,} times.")
    if stats.input_tokens is not None:
        print(f"Tokens reported by the provider: {stats.input_tokens:,}; truncated for length: "
              f"{stats.truncated:,} chunks.")
    for err in stats.errors:
        print(f"  ! {err[:300]}")


def ingest(settings, conn, args) -> int:
    doc_types = pipeline.doc_types_for(settings, args.doc_types)
    while True:
        started = time.monotonic()
        try:
            with _source(settings) as src:
                result = pipeline.run(conn, src, settings, doc_types, full=args.full, embed=not args.no_embed,
                                      max_chunks=args.max_chunks)
        except pipeline.AlreadyRunning as exc:
            print(f"! {exc}")
            if not args.loop:
                return 2
        except (pipeline.WatermarkAhead, index.IndexStateError, ProviderError, psycopg.OperationalError) as exc:
            # in --loop mode a source database that is down (crawler stack stopped) only skips this run
            print(f"! {str(exc).strip()}")
            if not args.loop:
                return 1
        else:
            _print_batches(result.batches)
            if result.lexemes is not None:
                print(f"Lexeme statistics: {result.lexemes:,} words.")
            if result.paused:
                _print_pause(result)
                if not args.loop:
                    return 3
            _print_embed(result.embed)
            if args.no_embed:
                print(f"\nEmbedding skipped (--no-embed). Pending: {embed_mod.pending_count(conn):,} chunks.")
            if not args.loop:
                failed = result.embed is not None and result.embed.status == "failed"
                return 1 if failed else 0
        if args.loop:
            wait = max(0.0, settings.ingest_interval_s - (time.monotonic() - started))
            log.info("next incremental run", extra={"in_s": round(wait)})
            time.sleep(wait)


def _print_pause(result) -> None:
    """The run stopped before the only paid step. Documents and watermark are already written."""
    w = result.waiting or {}
    print(f"\n⏸  Paused before embedding: {w.get('chunks', 0):,} new chunks "
          f"({w.get('chars', 0):,} characters) is over the threshold of {w.get('limit', 0):,} "
          f"(INGEST_EMBED_APPROVAL_CHUNKS).")
    print("   Documents and the watermark are written; only the embedding step is waiting for you.")
    print(f"   Approve: crawlerrag approve {result.thread_id}")
    print(f"   Refuse:  crawlerrag approve {result.thread_id} --no")
    print("   (or embed straight away, bypassing this gate: crawlerrag embed)")


def approve(settings, conn, args) -> int:
    """Answer a run parked at the approval gate."""
    from crawlerrag.ingest.graph import NoSuchRun
    doc_types = pipeline.doc_types_for(settings, None)
    try:
        with _source(settings) as src:
            result = pipeline.answer_approval(conn, src, settings, doc_types,
                                              thread_id=args.thread_id, approved=not args.no)
    except NoSuchRun as exc:
        print(f"! {exc}")
        return 2
    if result.paused:
        _print_pause(result)
        return 3
    if args.no:
        print(f"Embedding refused. Pending: {embed_mod.pending_count(conn):,} chunks "
              f"(embed them later with: crawlerrag embed).")
        return 0
    _print_embed(result.embed)
    return 1 if (result.embed is not None and result.embed.status == "failed") else 0


def plan(settings, conn, args) -> int:
    doc_types = pipeline.doc_types_for(settings, args.doc_types)
    with _source(settings) as src:
        rows = pipeline.plan(conn, src, settings, doc_types, full=args.full)
    print("Read only - nothing is written. The embedding cost is estimated from the character count.\n")
    print(f"{'doc_type':<14}{'mode':<13}{'change_id window':>22}{'key':>8}{'new':>8}{'changed':>9}{'same':>8}"
          f"{'new versions':>13}{'chunks to embed':>17}{'characters':>14}")
    for p in rows:
        window = f"({p.from_change_id:,}, {p.to_change_id:,}]"
        print(f"{p.doc_type:<14}{p.mode:<13}{window:>22}{p.keys:>8,}{p.new_docs:>8,}{p.changed_docs:>9,}"
              f"{p.unchanged_docs:>8,}{p.new_versions:>13,}{p.chunks_to_embed:>17,}{p.chars_to_embed:>14,}")
        print(f"{'':<14}{p.reason}")
    return 0


def embed(settings, conn, args) -> int:
    embedder = embedding_provider(settings)
    try:
        index.require_state(conn, embedder)
        with pipeline.exclusive(conn):
            stats = embed_mod.embed_pending(conn, embedder, max_chunks=args.max_chunks)
    finally:
        embedder.close()
    _print_embed(stats)
    return 1 if stats.status == "failed" else 0


# ---------------------------------------------------------------- status
def status(settings, conn, args) -> int:
    state = index.read_state(conn)
    if state is None:
        print("The index is not initialised yet. Run: crawlerrag init")
    else:
        print(f"Embedding: {state['embed_provider']}/{state['embed_model']}, {state['embed_dim']} dimensions")
    rows = index.index_stats(conn)
    if rows:
        print(f"\n{'doc_type':<14}{'documents':>11}{'active':>11}{'versions':>11}{'indexed':>11}"
              f"{'chunks':>11}{'embedded':>11}")
        for r in rows:
            print(f"{r['doc_type']:<14}{r['documents']:>11,}{r['active']:>11,}{r['versions']:>11,}"
                  f"{r['indexed']:>11,}{r['chunks']:>11,}{r['embedded']:>11,}")
    watermarks = {r["doc_type"]: r for r in conn.execute("SELECT * FROM ingest.watermark").fetchall()}
    print(f"\n{'doc_type':<14}{'watermark':>11}{'source head':>16}{'changes behind':>16}  updated")
    try:
        with _source(settings) as src:
            for dt in pipeline.doc_types_for(settings, ['all']):
                wm = watermarks.get(dt.doc_type)
                mark = int(wm["change_id"]) if wm else 0
                head = pipeline.source_head(src, dt.source_id)
                behind = db.scalar(src, "SELECT count(*) FROM crawl.record_change "
                                        "WHERE source_id = %s AND change_id > %s", (dt.source_id, mark))
                when = f"{wm['updated_at']:%Y-%m-%d %H:%M UTC}" if wm else "never loaded"
                print(f"{dt.doc_type:<14}{mark:>11,}{head:>16,}{behind:>16,}  {when}")
    except psycopg.OperationalError as exc:
        print(f"(could not read the source database: {str(exc).strip()[:200]})")
    batches = conn.execute("SELECT * FROM ingest.batch ORDER BY batch_id DESC LIMIT %s", (args.limit,)).fetchall()
    if batches:
        print(f"\n{'batch':>6}  {'doc_type':<14}{'mode':<12}{'status':<11}{'key':>8}{'added':>7}{'changed':>8}"
              f"{'off':>5}  started")
        for b in batches:
            print(f"{b['batch_id']:>6}  {b['doc_type']:<14}{b['mode']:<12}{b['status']:<11}{b['keys'] or 0:>8,}"
                  f"{b['inserted'] or 0:>7,}{b['updated'] or 0:>8,}{b['deactivated'] or 0:>5,}"
                  f"  {b['started_at']:%Y-%m-%d %H:%M:%S}")
    print(f"\nChunks pending embedding: {embed_mod.pending_count(conn):,}")
    return 0


# ---------------------------------------------------------------- search / ask
def query(settings, conn, args, *, generate: bool) -> int:
    question = " ".join(args.question).strip()
    doc_types = _doc_type_names(args.doc_types, settings)
    filters = retrieve.parse_filters(args.filter)
    embedder = embedding_provider(settings)
    chat = chat_provider(settings) if generate else None
    try:
        if not generate:
            result = retrieve.search(conn, settings, embedder, question, doc_types=doc_types,
                                     top_k=args.top_k, candidates=args.candidates, filters=filters)
            _print_hits(result.hits, full=args.full)
            print(f"\n{len(result.hits)} excerpts (vector {result.vector_candidates}, "
                  f"full-text {result.text_candidates} candidates).")
            return 0
        conversation_id = _resolve_conversation(conn, args.conversation)
        reply = answer_mod.ask(conn, settings, embedder, chat, question, doc_types=doc_types,
                               top_k=args.top_k, candidates=args.candidates, filters=filters,
                               conversation_id=conversation_id)
    finally:
        embedder.close()
        if chat is not None:
            chat.close()

    _print_answer(reply, show_sources=args.show_sources)
    if reply.pending_question:
        # `ask` is one shot: there is no second input to resume with, so say how to continue.
        print("(ask again with a clearer question, or use `crawlerrag chat` to answer it in place)")
        return 0
    if reply.conversation_id:
        print(f'Ask again: crawlerrag ask --conversation {reply.conversation_id} "..."   '
              f"(or crawlerrag chat --conversation {reply.conversation_id})")
    return 1 if reply.error else 0


def _resolve_conversation(conn, value: str | None) -> str | None:
    """``--conversation new`` starts one; an id must exist, a typo should not start a fresh chat."""
    if not value:
        return None
    if value == "new":
        return conversation.create(conn)
    if not conversation.exists(conn, value):
        raise ValueError(f"no conversation {value} (see: crawlerrag conversations)")
    return value


def _print_answer(reply, *, show_sources: bool = False) -> None:
    print()
    if reply.pending_question:
        # The turn is paused at the clarify step. No answer, no model call - a question back, with
        # examples built from records that are really in the index.
        print(textwrap.fill(reply.pending_question, width=100))
        for item in reply.samples:
            print(f"\n  • {item['question']}")
            if item.get("title"):
                print(f"    {item['title'][:96]}")
        print(f"\nType one of those, or your own question. (thread {reply.thread_id})")
        return
    if reply.standalone_question:
        print(f"(searched as: {reply.standalone_question})\n")
    print(textwrap.fill(reply.text, width=100, replace_whitespace=False) if reply.text
          else "(the model returned no text)")
    if reply.error:
        print(f"\n! {reply.error[:500]}")
    if reply.truncated:
        print("\n! The answer was cut off at the output token limit: raise RAG_CHAT_MAX_TOKENS, or cap "
              "Gemini 2.5's thinking with RAG_CHAT_THINKING_BUDGET.")
    print("\nSources:")
    for n, hit in enumerate(reply.hits, start=1):
        print(f"  [{n}] {hit.doc_id}  {hit.title[:80]}")
        if hit.url:
            print(f"      {hit.url}")
    tokens = f"{reply.prompt_tokens or '?'} in / {reply.output_tokens or '?'} out"
    turn = f" · turn {reply.turn}" if reply.turn else ""
    print(f"\n{reply.duration_ms} ms · token {tokens} · query #{reply.query_id}{turn}")
    if reply.trace_id:
        print(f"MLflow trace: {tracing.trace_url(reply.trace_id) or reply.trace_id}")
        print(f"  (score it: crawlerrag feedback {reply.query_id} good|bad)")
    if show_sources:
        print()
        _print_hits(reply.hits, full=True)


# ---------------------------------------------------------------- chat
CHAT_HELP = ("Type a question and press Enter. Commands: /new a new conversation · /history the turns "
             "so far · /sources the full sources of the last answer · /exit leave")


def _chat(settings, conn, args) -> int:
    doc_types = _doc_type_names(args.doc_types, settings)
    filters = retrieve.parse_filters(args.filter)
    if not args.verbose:
        # one JSON log line per retrieval would bury the conversation
        logging.getLogger("crawlerrag").setLevel(logging.WARNING)
    conversation_id = _resolve_conversation(conn, args.conversation or "new")
    embedder, chat = embedding_provider(settings), chat_provider(settings)
    last = None
    pending_thread: str | None = None      # set while a turn is waiting for a clearer question
    print(f"Conversation {conversation_id}\n{CHAT_HELP}")
    try:
        while True:
            try:
                line = input("\nYou › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/exit", "/quit", "/q"):
                break
            if line == "/help":
                print(CHAT_HELP)
                continue
            if line == "/new":
                conversation_id, last, pending_thread = conversation.create(conn), None, None
                print(f"New conversation {conversation_id}")
                continue
            if line == "/history":
                turns = conversation.history(conn, conversation_id, limit=50)
                for t in turns:
                    print(f"\n[{t.turn}] You: {t.question}\n    Bot: {textwrap.shorten(t.answer, 300)}")
                if not turns:
                    print("(no turns yet)")
                continue
            if line == "/sources":
                if last is None:
                    print("(no answer yet)")
                else:
                    _print_hits(last.hits, full=True)
                continue
            try:
                if pending_thread:
                    # The previous turn asked for a better question; this line is the answer to it.
                    last = answer_mod.resume(conn, settings, embedder, chat,
                                             thread_id=pending_thread, question=line)
                else:
                    last = answer_mod.ask(conn, settings, embedder, chat, line, doc_types=doc_types,
                                          top_k=args.top_k, candidates=args.candidates, filters=filters,
                                          conversation_id=conversation_id)
            except chat_graph.NoSuchTurn as exc:
                print(f"\n! {exc}")
                pending_thread = None
                continue
            except (ProviderError, index.IndexStateError, ValueError) as exc:
                print(f"\n! {exc}")         # a failed turn does not end the conversation
                continue
            pending_thread = last.thread_id
            _print_answer(last)
    finally:
        embedder.close()
        chat.close()
    print(f"Continue later: crawlerrag chat --conversation {conversation_id}")
    if (url := tracing.session_url(conversation_id)) is not None:
        print(f"The whole conversation in MLflow: {url}")
    return 0


def chat(settings, conn, args) -> int:
    return _chat(settings, conn, args)


def conversations(settings, conn, args) -> int:
    rows = conn.execute(
        """
        SELECT c.conversation_id, c.title, c.updated_at, count(q.query_id) AS turns
          FROM rag.conversation c LEFT JOIN rag.query_log q USING (conversation_id)
         GROUP BY c.conversation_id ORDER BY c.updated_at DESC LIMIT %s
        """,
        (args.limit,),
    ).fetchall()
    if not rows:
        print("No conversations yet (start one with: crawlerrag chat).")
        return 0
    for r in rows:
        print(f"{r['conversation_id']}  {r['updated_at']:%Y-%m-%d %H:%M}  {r['turns']:>3} turns  "
              f"{(r['title'] or '(empty)')[:70]}")
    return 0


def _feedback(conn, args) -> int:
    row = conn.execute("SELECT trace_id FROM rag.query_log WHERE query_id = %s", (args.query_id,)).fetchone()
    if row is None:
        print(f"No query #{args.query_id} in rag.query_log.")
        return 1
    trace_id = row["trace_id"]
    if trace_id is None:
        print(f"Query #{args.query_id} has no trace (tracing was off when it was asked).")
        return 1
    if not str(trace_id).startswith("tr-"):
        print(f"Query #{args.query_id} has trace_id '{trace_id}', which is not an MLflow one (tr-...).")
        return 1
    if not tracing.enabled():
        print("Tracing is off: set MLFLOW_TRACKING_URI to send feedback.")
        return 1
    tracing.send_feedback(trace_id, good=args.verdict == "good", comment=args.comment)
    print(f"Recorded feedback '{args.verdict}' for query #{args.query_id} (trace {trace_id}).")
    if (url := tracing.trace_url(trace_id)) is not None:
        print(url)
    return 0


def feedback(settings, conn, args) -> int:
    return _feedback(conn, args)


def _print_hits(hits, *, full: bool) -> None:
    print()
    for n, hit in enumerate(hits, start=1):
        dist = f"d={hit.distance:.4f}" if hit.distance is not None else "d=-"
        print(f"[{n}] {MATCH_ICON[hit.matched_by]} {hit.score:.4f}  {dist}  {hit.doc_type}  {hit.doc_id}")
        print(f"    {hit.title[:110]}")
        body = hit.text if full else hit.text[:280].replace("\n", " · ")
        print(textwrap.indent(body if full else body + ("..." if len(hit.text) > 280 else ""), "    "))
        if full:
            print()


# ---------------------------------------------------------------- metadata, rules, history
def catalog(settings, conn, args) -> int:
    """Read the metadata out of the crawler's Postgres and store it in the vector database's meta.*."""
    from crawlerrag.rules import load_rules

    ruleset = load_rules(settings.rules_dir)
    with _source(settings) as src:
        cat = introspect.read_catalog(src, ruleset.catalog.schemas, ruleset.catalog.exclude)
    run_id = meta_catalog.save_catalog(conn, cat, source="cli")
    print(f"Catalog #{run_id}: {cat.table_count} tables, {cat.column_count} columns, "
          f"{cat.relationship_count} relationships (digest {cat.digest[:12]}).")
    print(f"\n{'table':<34}{'kind':<12}{'cols':>5}{'rows (estimate)':>18}  primary key")
    for table in cat.tables:
        rows = f"{table.est_rows:,}" if table.est_rows is not None else "-"
        print(f"{table.qualified_name:<34}{table.kind:<12}{len(table.columns):>5}{rows:>18}  "
              f"{', '.join(table.primary_key) or '-'}")
    declared = [r for r in cat.relationships if r.kind == "declared"]
    inferred = [r for r in cat.relationships if r.kind == "inferred"]
    print(f"\nRelationships: {len(declared)} declared (foreign keys), {len(inferred)} inferred from "
          "primary keys.")
    for rel in cat.relationships:
        print(f"  {rel.from_table}({', '.join(rel.from_columns)}) -> {rel.to_table}  [{rel.kind}]")
    if args.export:
        from pathlib import Path
        Path(args.export).write_text(cat.to_yaml(), encoding="utf-8")
        print(f"\nWrote {args.export}.")
    return 0


def rules_check(settings, conn, args) -> int:
    """Check rules/*.yaml against the metadata that was read. Any error means exit 1."""
    from crawlerrag.rules import clear_cache, load_rules
    from crawlerrag.rules.validate import errors, validate_ruleset

    clear_cache()
    ruleset = load_rules(settings.rules_dir)
    print(f"Loaded {len(ruleset.doc_types)} doc types from {ruleset.path}:")
    for name, rule in ruleset.doc_types.items():
        print(f"  {name:<14} {rule.source.qualified_name:<24} key={rule.source.key:<14}"
              f" {len(rule.body)} body lines, {len(rule.children)} child tables, "
              f"{len(rule.quality)} quality rules, digest {rule.text_digest[:12]}")
    cat = meta_catalog.load_catalog(conn)
    if cat is None or args.refresh:
        with _source(settings) as src:
            cat = introspect.read_catalog(src, ruleset.catalog.schemas, ruleset.catalog.exclude)
            meta_catalog.save_catalog(conn, cat, source="rules-check")
        print(f"\nRead the metadata again: {cat.table_count} tables (digest {cat.digest[:12]}).")
    else:
        print(f"\nUsing the stored catalog: {cat.table_count} tables, captured at "
              f"{cat.captured_at:%Y-%m-%d %H:%M UTC}.")
    findings = validate_ruleset(ruleset, cat)
    if not findings:
        print("Every table, column and key named in the rules matches the database.")
        return 0
    for finding in findings:
        print(f"  {finding.render()}")
    bad = errors(findings)
    print(f"\n{len(bad)} errors, {len(findings) - len(bad)} warnings.")
    return 1 if bad else 0


def history(settings, conn, args) -> int:
    """Every version of one document (SCD Type 2)."""
    rows = pipeline.document_history(conn, args.doc_id)
    if not rows:
        print(f"No document with doc_id {args.doc_id!r}.")
        return 1
    print(f"{args.doc_id}: {len(rows)} version\n")
    print(f"{'v':>3}{'current':>10}{'active':>8}{'from':>20}{'to':>20}  reason")
    for r in rows:
        since = f"{r['valid_from']:%Y-%m-%d %H:%M}"
        until = f"{r['valid_to']:%Y-%m-%d %H:%M}" if r["valid_to"] else "-"
        print(f"{r['version']:>3}{('yes' if r['is_current'] else ''):>10}"
              f"{('yes' if r['is_active'] else 'off'):>8}{since:>20}{until:>20}  {r['change_reason']}")
    return 0


def qualify(settings, conn, args) -> int:
    """Show what the gate decides about a question, calling no model."""
    from crawlerrag.rag import qualify as qualify_mod
    from crawlerrag.rules import load_rules_cached

    question = " ".join(args.question)
    result = qualify_mod.qualify(load_rules_cached(settings.rules_dir).qualify, question,
                                filters=retrieve.parse_filters(args.filter),
                                is_follow_up=args.follow_up)
    print(f"Decision: {result.decision}" + (f" (rule {result.rule_id})" if result.rule_id else ""))
    print(f"The question as normalised: {result.question!r}")
    if result.blocked:
        print(f"\n{qualify_mod.blocked_answer(result)}")
        print("\nNo model was called.")
    return 0
