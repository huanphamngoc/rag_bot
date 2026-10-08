"""Command line: python -m crawlerrag <command> (in Docker: docker compose run --rm app <command>)."""
from __future__ import annotations

import argparse
import sys

from crawlerrag import commands
from crawlerrag.config import get_settings
from crawlerrag.db import wait_for_db
from crawlerrag.ingest.pipeline import AlreadyRunning, WatermarkAhead
from crawlerrag.rules import RuleError
from crawlerrag.logging_setup import setup_logging
from crawlerrag.migrate import migrate
from crawlerrag.rag.index import IndexStateError
from crawlerrag.rag.providers import ProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crawlerrag",
                                     description="RAG over the crawler's data: incremental loading into pgvector, and grounded answers")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="create or update the rag and ingest schemas in the vector database")

    p = sub.add_parser("init", help="fix the vector dimension (by probing the model) and build the HNSW index")
    p.add_argument("--force", action="store_true", help="drop the existing vectors when the provider or model changes")
    p.add_argument("--dim", type=int, help="set the dimension instead of probing the model")

    names = "see rules/doc_types/*.yaml"
    for name, help_text in (("ingest", "read changes from the crawler's change log, build documents and chunks, then embed"),
                            ("plan", "preview what ingest would do and how many characters it would embed (read only)")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("doc_types", nargs="*", help=f"defaults to RAG_DOC_TYPES; 'all' means every type. Available: {names}")
        p.add_argument("--full", action="store_true",
                       help="reconcile the whole source instead of the changes only (still writes and embeds "
                            "nothing but the differences)")
        if name == "ingest":
            p.add_argument("--no-embed", action="store_true", help="build documents and chunks only, without calling the embedding API")
            p.add_argument("--max-chunks", type=int, help="cap the chunks embedded in this run, to control what it costs")
            p.add_argument("--loop", action="store_true",
                           help="run every INGEST_INTERVAL_S seconds (the scheduler service)")

    p = sub.add_parser("approve", help="answer a run that is paused for approval before embedding")
    p.add_argument("thread_id", help="printed at the end of the run that paused")
    p.add_argument("--no", action="store_true", help="refuse: embed nothing, leave the chunks pending")

    p = sub.add_parser("embed", help="embed the pending chunks and nothing else")
    p.add_argument("--max-chunks", type=int)

    p = sub.add_parser("status", help="the index, the watermark against the source, and recent batches")
    p.add_argument("--limit", type=int, default=10, help="how many recent batches to show")

    p = sub.add_parser("catalog", help="read the metadata (tables, columns, relationships) out of the "
                                      "crawler's Postgres")
    p.add_argument("--export", metavar="FILE", help="also write it to a YAML file to read through")

    p = sub.add_parser("rules-check", help="check rules/*.yaml against the metadata that was read")
    p.add_argument("--refresh", action="store_true", help="read the metadata again instead of using the stored copy")

    p = sub.add_parser("history", help="every version of one document (SCD Type 2)")
    p.add_argument("doc_id", help='for example "drug_recall:D-0853-2026"')

    p = sub.add_parser("qualify", help="show what the gate decides about a question, calling no model")
    p.add_argument("question", nargs="+")
    p.add_argument("--filter", action="append", metavar="KEY=VALUE")
    p.add_argument("--follow-up", action="store_true", help="judge it as a follow-up inside a conversation")

    for name, help_text in (("search", "retrieve only, with no LLM call"),
                            ("ask", "retrieve, then have the LLM answer with its sources")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question", nargs="+")
        p.add_argument("--doc-types", help="limit the document types, e.g. 'cpsc_recall,drug_recall'")
        p.add_argument("--top-k", type=int)
        p.add_argument("--candidates", type=int)
        p.add_argument("--filter", action="append", metavar="KEY=VALUE", help="filter on metadata, e.g. --filter year=2024")
        p.add_argument("--full", action="store_true", help="print each excerpt in full")
        p.add_argument("--show-sources", action="store_true")
        if name == "ask":
            p.add_argument("--conversation", metavar="ID|new")

    p = sub.add_parser("chat", help="a multi-turn conversation (type /exit to leave)")
    p.add_argument("--conversation", metavar="ID")
    p.add_argument("--doc-types")
    p.add_argument("--top-k", type=int)
    p.add_argument("--candidates", type=int)
    p.add_argument("--filter", action="append", metavar="KEY=VALUE")
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("conversations", help="recent conversations")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("feedback", help="score one answer onto its MLflow trace")
    p.add_argument("query_id", type=int)
    p.add_argument("verdict", choices=("good", "bad"))
    p.add_argument("--comment")
    return parser


HANDLERS = {
    "init": commands.init, "ingest": commands.ingest, "plan": commands.plan, "embed": commands.embed,
    "status": commands.status, "chat": commands.chat, "conversations": commands.conversations,
    "feedback": commands.feedback, "catalog": commands.catalog, "rules-check": commands.rules_check,
    "history": commands.history, "qualify": commands.qualify,
    "approve": commands.approve,
    "search": lambda s, c, a: commands.query(s, c, a, generate=False),
    "ask": lambda s, c, a: commands.query(s, c, a, generate=True),
}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    conn = wait_for_db(settings.database_url)
    try:
        migrate(conn)
        if args.command == "migrate":
            print("Migrations up to date.")
            return 0
        try:
            return HANDLERS[args.command](settings, conn, args)
        except (AlreadyRunning, WatermarkAhead, IndexStateError, ProviderError, RuleError,
                ValueError) as exc:
            print(f"\n{exc}")
            return 1
    finally:
        conn.close()
