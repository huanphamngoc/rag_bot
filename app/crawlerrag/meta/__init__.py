"""Metadata extracted from the crawler's Postgres: tables, columns, keys, relationships.

:mod:`introspect` reads it (read-only), :mod:`catalog` stores it in ``meta.*`` of the vector database,
and ``crawlerrag.rules.validate`` checks the YAML rules against it.
"""
