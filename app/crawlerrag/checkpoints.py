"""The Postgres store behind LangGraph's checkpointer, shared by the two graphs.

Both flows pause: the ingestion run waits for the embedding bill to be approved, and a chat turn waits
for a vague question to be corrected. A pause is only answerable if the state was stored, so both need a
checkpointer - but they are switched on by their own settings and must not be coupled through one.

``search_path`` is how the checkpointer's tables stay out of ``rag`` and ``ingest``: the saver creates
and queries them unqualified, so they land in ``graph``.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Iterator

log = logging.getLogger(__name__)


@contextlib.contextmanager
def postgres_saver(database_url: str) -> Iterator[object]:
    """A ``PostgresSaver`` on its own connection, for as long as the caller needs it.

    ``setup()`` runs every time. Caching it per process looked like a saving and was a bug: the
    integration suite drops the ``graph`` schema between tests, so a later turn skipped setup and found
    no tables. It is idempotent and costs a few milliseconds against seconds of model time.
    """
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    with psycopg.connect(database_url, autocommit=True, row_factory=dict_row,
                         options="-c search_path=graph,public") as conn:
        saver = PostgresSaver(conn)
        saver.setup()
        yield saver
