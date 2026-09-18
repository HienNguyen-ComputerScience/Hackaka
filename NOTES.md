# Engineering notes

## Pipeline

Built the deterministic stages over the supplied corpus: `split.py` (units), `people.py` (registry +
mentions), `link.py` (claim supersession/truth-status graph over LLM-extracted `claims_raw`),
`index.py` (BM25 + MiniLM retrieval passages), `retrieve.py` / `answer.py` (query-time rendering).
Each stage has a standalone `verify_*.py` acceptance script.

## Deletion feature (`pipeline/delete.py`)

Implements right-to-erasure: remove one person's authored units, redact every other mention of them
across `units.jsonl` / `extract_input/*.txt` / `claims_raw/*.jsonl`, drop or reroute claims that
named or were about them, recompute supersession chains (a chain whose current statement was theirs
is marked `head_removed`, never silently re-resolved to a newer "current"), and rebuild every derived
store (mentions, passages, embeddings, claims, claim groups) from the scrubbed inputs. A full-text
sweep over `data/` confirms no form of the person's name survives outside the audit log.

**Bug found and fixed:** the original version removed authored units but never renumbered the
survivors. `split.py` bakes each unit's ordinal into both its id (`m05`, `t007`) and its rendered
anchor ("message 4 of 5", thread totals). Deleting a unit without renumbering left the *id* still
carrying the old ordinal while the *rendered position* (and thread total) shifted underneath it —
a visible gap that discloses a deletion happened even though the sweep found no name. Fixed by
having `delete.py` renumber every surviving unit contiguously per source file/thread (e-mail/report,
by original chronological position) or per transcript (by original turn order) as part of rebuild,
in `split.py`'s own id format, and cascading that remap into `claims_raw`, `extract_input` block
headers, and claim ids (`unit_id#n`) before everything downstream is rebuilt from them. Added a
structural verification pass (`verify_structure`) alongside the text sweep: no ordinal gaps, and
every claim/mention/passage/chain-member reference resolves to a unit or claim that still exists.
The id remap itself (not the erased content) is recorded on the deletion log entry, so a rerun is
traceable.

Confirmed separately: `retrieve.py`/`answer.py` render citations from `data/units.jsonl`'s own
stored text, never from `corpus/` — so there's no click-through path back to a deleted person's
original text. This must hold for the UI too.

## Git

Discovered the working checkout's `.git` lived at the drive root (`C:/`) rather than inside the
project folder — an empty, no-history stray `git init` (verified: no refs, no packed-refs, no log,
no objects) — meaning ordinary git commands were scanning the whole drive. Removed it and
initialized a fresh repository rooted at the project folder instead.

Added `.gitignore` before the first commit: excludes `data/` (generated, and after a deletion the
only place erased content may legitimately persist is `deletion_log.jsonl` — it must never reach git
history), `corpus/` (supplied input, not our work), and any backup/snapshot directory. Also had to
scrub a handful of hardcoded references to the corpus's deletion-test subject out of pipeline source
(`people.py`'s registry, a few docstring examples, and two `verify_retrieve.py` checks that hardcoded
their name) before the first commit, since the same reasoning applies to source as to data: their
name shouldn't persist in version-controlled text either. `verify_retrieve.py`'s checks now pick a
"top speaker" dynamically instead of naming a specific person, which also makes it keep working
correctly regardless of who gets deleted next.

First commit: `pipeline/`, `CLAUDE.md`, `NOTES.md`, `.gitignore` only. Not `data/`, not `corpus/`.
