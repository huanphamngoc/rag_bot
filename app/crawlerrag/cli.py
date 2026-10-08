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
                                     description="RAG trên dữ liệu của crawler: nạp tăng dần vào pgvector + hỏi đáp")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="tạo/cập nhật schema rag + ingest trong vector DB")

    p = sub.add_parser("init", help="cố định số chiều vector (thăm dò model) + tạo index HNSW")
    p.add_argument("--force", action="store_true", help="xoá vector cũ khi đổi provider/model")
    p.add_argument("--dim", type=int, help="ấn định số chiều thay vì thăm dò model")

    names = "xem rules/doc_types/*.yaml"
    for name, help_text in (("ingest", "đọc thay đổi từ change log của crawler, dựng tài liệu/chunk rồi nhúng"),
                            ("plan", "xem trước ingest sẽ làm gì và phải nhúng bao nhiêu ký tự (chỉ đọc)")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("doc_types", nargs="*", help=f"mặc định RAG_DOC_TYPES; 'all' = tất cả. Có: {names}")
        p.add_argument("--full", action="store_true",
                       help="đối chiếu toàn bộ nguồn thay vì chỉ phần thay đổi (chỉ ghi/nhúng phần khác)")
        if name == "ingest":
            p.add_argument("--no-embed", action="store_true", help="chỉ dựng tài liệu/chunk, chưa gọi API nhúng")
            p.add_argument("--max-chunks", type=int, help="giới hạn số chunk nhúng lần này (kiểm soát chi phí)")
            p.add_argument("--loop", action="store_true",
                           help="chạy lặp mỗi INGEST_INTERVAL_S giây (dịch vụ scheduler)")

    p = sub.add_parser("approve", help="trả lời một lần chạy đang chờ duyệt trước bước nhúng")
    p.add_argument("thread_id", help="in ra ở cuối lần chạy bị dừng")
    p.add_argument("--no", action="store_true", help="từ chối: không nhúng, chunk vẫn nằm chờ")

    p = sub.add_parser("embed", help="chỉ nhúng các chunk đang chờ")
    p.add_argument("--max-chunks", type=int)

    p = sub.add_parser("status", help="index, watermark so với nguồn, các batch gần đây")
    p.add_argument("--limit", type=int, default=10, help="số batch gần nhất")

    p = sub.add_parser("catalog", help="rút metadata (bảng, cột, quan hệ) từ Postgres của crawler")
    p.add_argument("--export", metavar="FILE", help="ghi thêm ra file YAML để xem lại")

    p = sub.add_parser("rules-check", help="đối chiếu rules/*.yaml với metadata đã rút")
    p.add_argument("--refresh", action="store_true", help="rút metadata mới thay vì dùng bản đã lưu")

    p = sub.add_parser("history", help="các version của một tài liệu (SCD Type 2)")
    p.add_argument("doc_id", help='ví dụ "drug_recall:D-0853-2026"')

    p = sub.add_parser("qualify", help="xem cổng chặn quyết định gì với một câu hỏi (không gọi model)")
    p.add_argument("question", nargs="+")
    p.add_argument("--filter", action="append", metavar="KEY=VALUE")
    p.add_argument("--follow-up", action="store_true", help="coi như câu hỏi tiếp nối trong hội thoại")

    for name, help_text in (("search", "chỉ truy hồi, không gọi LLM"),
                            ("ask", "truy hồi rồi để LLM trả lời có dẫn nguồn")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question", nargs="+")
        p.add_argument("--doc-types", help="giới hạn loại tài liệu, vd 'cpsc_recall,drug_recall'")
        p.add_argument("--top-k", type=int)
        p.add_argument("--candidates", type=int)
        p.add_argument("--filter", action="append", metavar="KEY=VALUE", help="lọc metadata, vd --filter year=2024")
        p.add_argument("--full", action="store_true", help="in trọn nội dung đoạn")
        p.add_argument("--show-sources", action="store_true")
        if name == "ask":
            p.add_argument("--conversation", metavar="ID|new")

    p = sub.add_parser("chat", help="hỏi đáp nhiều lượt (gõ /exit để thoát)")
    p.add_argument("--conversation", metavar="ID")
    p.add_argument("--doc-types")
    p.add_argument("--top-k", type=int)
    p.add_argument("--candidates", type=int)
    p.add_argument("--filter", action="append", metavar="KEY=VALUE")
    p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("conversations", help="các hội thoại gần đây")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("feedback", help="chấm điểm một câu trả lời lên trace MLflow của nó")
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
