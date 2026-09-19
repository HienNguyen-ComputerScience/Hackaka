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
  - `test_full.py` — the whole acceptance in one command (practice questions, unseen questions,
    deletion end to end with restore from the snapshot, hygiene). `python pipeline/test_full.py`;
    `--skip-deletion` leaves `data/` untouched.
  - `serve.py` — the judge-facing UI (standard library only): ask a question, expand citations to
    the stored unit text, delete a person. It calls `answer.py` in-process and runs `delete.py` as
    a subprocess. It sends the browser an allow-list of fields only: no unit id, claim id or turn
    index, and it never reads `corpus/`.

## Run the UI

```
pip install -r requirements.txt
python pipeline/serve.py --host 0.0.0.0 --port 8080
```

The first start loads the embedding model (about 20 s) and then prints `ready`. The page is at
`http://localhost:8080`; on the same network, judges use `http://<this machine's IP>:8080`. For
judges elsewhere, put a tunnel in front of the same port (for example `cloudflared tunnel --url
http://localhost:8080`, or ngrok) and send them the URL it prints, or run the same two commands on a
VM with port 8080 open. There is no login: anyone with the URL can delete a person.

`data/` must be present (the pipeline output). Deletion in the UI is real and has no undo, so keep a
pre-deletion copy of `data/` before judging and reset between runs. `test_full.py` defaults to
`relex-data-snapshot` **inside** the repo, where `.gitignore`'s `*snapshot*/` rule keeps it out of
git history; it falls back to the older beside-the-repo location, and `--snapshot PATH` overrides
both. Do not rename it to anything `*snapshot*/` would fail to catch.

```
# PowerShell, from the repo root
Remove-Item -Recurse -Force data; Copy-Item -Recurse relex-data-snapshot data
```

Restart `serve.py` after a reset: it holds the loaded index in memory.

## The one rule that matters most

**Nothing under `data/` or `corpus/` may ever enter git history.** `data/` is generated and, after a
deletion, `data/deletion_log.jsonl` is the *only* place erased text may legitimately appear — a
pre-deletion snapshot of any store landing in git history (a backup directory, a stray `git add -A`,
an accidentally-committed intermediate file) makes the deleted person permanently recoverable from
the repository, which defeats the deletion feature outright. `.gitignore` enforces this; don't
override it with `git add -f`.

## `split.py` is frozen — changing it destroys the claim layer

`claims_raw/` is keyed to the exact unit ids `split.py` emits: `link.py` builds every `claim_id` as
`unit_id#n` and **drops** any claim whose `unit_id` no longer resolves. It reports the count and
exits zero, so a change that orphans every claim looks like a clean run that happens to produce an
empty claim graph.

Extraction is not re-runnable in practice — re-running it yields different claims, which changes
answers, which invalidates the pass bars and the frozen question set. So there is no way back.
**Do not change how `split.py` chunks, merges, orders or numbers units.** If a split change ever
becomes genuinely necessary, it needs a matching id remap across `claims_raw/`, `extract_input/`
and claim ids — the same cascade `delete.py` already performs — not a re-extraction.

Bug fixes to `split.py` that leave unit ids untouched are fine. Check `link.py`'s `dropped:` count
after any change to it; a non-zero jump means claims have been orphaned.

## Working conventions

- Keep pipeline stages deterministic and re-runnable from `corpus/` (except the LLM extraction step).
- `delete.py` must keep working correctly when re-run for a second (or third) person: it composes
  id remaps and `head_removed`/`removed_statements` markers across runs without corrupting a prior
  deletion's record.
- Don't hardcode a specific corpus person's name in pipeline or verify-script source — a deleted
  person's name must not persist in versioned source either. Verify scripts that need "some person"
  or "the top speaker" should pick one dynamically from current data instead.
