# rules/ — the declarations

This directory is the **only** place a document type and the rules over it are described. No SQL and
no formatting is left in Python: `crawlerrag/rules/` reads these files, generates the SELECT, builds
the documents and runs the quality checks.

```
rules/
  catalog.yaml        which schemas in the crawler's Postgres the metadata is read from
  qualify.yaml        the gate in front of the model, applied to the user's question
  doc_types/
    drug_recall.yaml  one file per document type (the file name is the doc_type)
    cpsc_recall.yaml
```

Edit a file here and run `rules-check` to compare it against the metadata read from Postgres: a wrong
table or column name is an error, a missing parent-child relationship is only a warning. No Python to
change, and no image to rebuild - the directory is mounted into the container.

## What a doc type holds

| Block | What it does |
|---|---|
| `source` | the parent table, the key (which must equal `crawl.record_change.record_key`), the soft-delete column, the crawler's `source_id` |
| `children` | each entry is one correlated `array_agg` over a child table |
| `title`, `body`, `url`, `metadata` | the document's text and its filters |
| `scd2` | which attributes open a new version when they change, and which are overwritten in place |
| `quality` | checks on the extracted rows, run **before** anything is written |

## The value language

A *value spec* is one of:

| Key | Meaning |
|---|---|
| `field: col` | a parent column, or an alias from `children` |
| `join: [a, b]` | the non-empty values joined by `separator` (default `", "`) |
| `template: "...{col}..."` | columns filled into a pattern; **empty if any column is empty** |
| `coalesce: [spec, spec]` | the first spec that renders a non-empty string |
| `const: "text"` | a constant |

Add `format:` as `iso` (a date), `year` (the year as a number), or `thousands` (`1,234,567`; 0 and
NULL render as empty).

The trick worth knowing: `template` renders empty when a column is missing, so a `coalesce` of
`template`s is how you say "and if that is not there, use this instead".

## Two things to keep in mind

1. **The text must not change by accident.** `tests/test_rules_build.py` compares it byte for byte
   against the Python builders it replaced. Changing one label in `body` changes the `content_hash`
   of every document, so the next `ingest` re-chunks all of them. A chunk whose text is identical
   keeps its vector, but new text has to be embedded again and **that costs money**.
2. **`version:`** only needs to go up when you *want* a full reconcile. Change detection is already
   automatic: the watermark signature contains a digest of this file, so editing the YAML makes the
   next run switch to `full` by itself.

## Quality rules

| `rule` | Parameters | Fails when |
|---|---|---|
| `not_null` | `column` | any row is NULL (or an empty list) |
| `unique` | `column` | a value appears more than once |
| `allowed_values` | `column`, `values` | a non-empty value is outside the list |
| `max_null_fraction` | `column`, `max` | the share of NULLs is over the threshold |
| `min_rows` | `min` | there are fewer rows than the threshold (**full loads only**) |

`severity: error` (the default) fails the batch and does **not** advance the watermark, so the next
run reads the same window again. `severity: warn` records the finding in `ingest.quality_finding` and
carries on.
