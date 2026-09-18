# relex

A deterministic, no-model-at-query-time pipeline over a synthetic e-mail/report/transcript corpus:
split the corpus into atomic units, build a person/mention index, extract and link claims into
supersession chains with truth status, retrieve by hybrid BM25 + embedding search, and answer
questions by rendering the claim graph. It also implements right-to-erasure deletion of a person
from every derived store, with an auditable log and a full-text + structural verification sweep.

## Layout

- `corpus/` — supplied input corpus (raw e-mails, reports, transcripts). Not our work; **never
  committed**.
- `pipeline/` — the pipeline itself:
  - `split.py` — corpus files -> `data/units.jsonl` (one record per atomic unit: e-mail message,
    report entry, or transcript turn).
  - `people.py` — `data/units.jsonl` -> `data/people.json` (person registry) + `data/mentions.jsonl`
    (mention index).
  - (extraction of `data/claims_raw/*.jsonl` and `data/extract_input/*.txt` from units is a separate,
    non-deterministic LLM step outside this pipeline.)
  - `link.py` — `data/claims_raw/*.jsonl` -> `data/claims.jsonl` + `data/claim_groups.json`
    (supersession chains, truth status).
  - `index.py` — units + mentions -> `data/passages.jsonl` + `data/embeddings.npy` (retrieval index).
  - `retrieve.py` / `answer.py` — query-time hybrid retrieval and deterministic answer rendering.
    Both render citations from `data/units.jsonl`'s own stored (already-scrubbed) text and anchor
    fields — **never** re-read `corpus/` at query or render time. Any future UI must keep that
    property, or a deletion can be undone by simply clicking through to the source file.
  - `delete.py` — erases one person from every store in `data/`, renumbers surviving unit/claim ids
    so no gap or ordinal discloses that a deletion happened, and appends an audit record (with an
    `id_remap`, not the person's data) to `data/deletion_log.jsonl`.
  - `verify_*.py` — acceptance checks for each stage, runnable standalone.

## The one rule that matters most

**Nothing under `data/` or `corpus/` may ever enter git history.** `data/` is generated and, after a
deletion, `data/deletion_log.jsonl` is the *only* place erased text may legitimately appear — a
pre-deletion snapshot of any store landing in git history (a backup directory, a stray `git add -A`,
an accidentally-committed intermediate file) makes the deleted person permanently recoverable from
the repository, which defeats the deletion feature outright. `.gitignore` enforces this; don't
override it with `git add -f`.

## Working conventions

- Keep pipeline stages deterministic and re-runnable from `corpus/` (except the LLM extraction step).
- `delete.py` must keep working correctly when re-run for a second (or third) person: it composes
  id remaps and `head_removed`/`removed_statements` markers across runs without corrupting a prior
  deletion's record.
- Don't hardcode a specific corpus person's name in pipeline or verify-script source — a deleted
  person's name must not persist in versioned source either. Verify scripts that need "some person"
  or "the top speaker" should pick one dynamically from current data instead.
