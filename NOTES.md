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

## UI (`pipeline/serve.py`)

Standard-library HTTP server, one page, three controls: question box, expandable citations,
person picker + delete. It calls `answer.py` in-process and runs `delete.py` as a subprocess, then
reloads the index. The JSON sent to the browser is built from an allow-list (no unit id, claim id or
turn index; internal-transcript anchors have "turn N" stripped), and citations are the unit text
stored in `data/units.jsonl`, never `corpus/`. Deletion confirmation is an in-page second click, not
a native dialog, and the post-deletion summary deliberately does not repeat the erased name.

Two `answer.py` changes were needed for the UI acceptance. (1) Group selection: window
co-occurrence used to count every claim in a retrieved window, so a chatty window could outrank a
chain whose fact key literally named the question (P2's "pilot service level target" lost to an
unrelated transcript). Now a group counts once per hit, fact-key similarity has a lower threshold,
and a word-overlap channel on fact keys was added; the attribution pool also admits any top group
that holds both a proposal and its answer. (2) Relevance gate: a chain is shown only if its fact keys
relate to the question by meaning or by words, so an off-topic question yields no chains and the
renderer says the archive does not contain it, instead of surfacing whatever the windows touched.

## Full test (`pipeline/test_full.py`)

One command, four parts, a pass/fail table, exit code. Part 1 renders the nine practice questions
against the full archive and checks each against its stated bar. Part 2 asks fifteen unseen
questions (three outside the archive) and flags any unsourced statement or invented figure. Part 3
deletes one person for real (whoever authored most of P1's chain statements, chosen from the data,
never named in source), reruns P1 and four neighbours, runs `delete.py`'s text and structural
sweeps, restores from the snapshot beside the repo and byte-compares. Part 4 checks every rendered
answer: every statement cited, every figure present in its cited unit (dates the extractor resolved
from headers are not figures), no value on a truncated claim, no id or turn index in the text.

Making it green needed four deterministic additions to `answer.py`:

- **Date scope** ("September 2024", "Q1 2026", "2025"): retrieval is also run restricted to the
  period, chains with statements in it are favoured, and when the question asks for figures, chains
  carrying a number then are favoured and the group limit rises. This is what P1 needed: nothing but
  the date tied the question to the assessment figures.
- **Agreed-and-not-done trails** (P8): chains where a commitment is followed by a later statement
  that it is outstanding, with no later statement that it was done. The renderer shows the
  commitment, the latest statement, the gap in days, who committed and who raised it, and says
  explicitly that the archive does not record whether it was ever done.
- **Status-report flag** (P9): a statement taken from `reports/` is labelled as evidence of what was
  reported, not of the underlying state.
- **Answer rule**: a chain on the same topic is not an answer. Outside a date scope a chain must
  share two content words with the question or contain a statement semantically close to it;
  question words the archive never uses anywhere are stated as such, and when they make up a third
  of the question and the rest is not covered either, the answer is "not contained" with the nearest
  topics named. Known limit: a question about an attribute the archive never mentions of a topic it
  discusses at length ("which language is the DC-2 middleware written in") is caught only through
  the never-used words; a paraphrase that avoids new words would show the topic's chains as context.

**Direct-answer label.** Every non-scoped answer is tested for whether a single statement in the chosen
chains addresses the question as asked (semantically close, or covering half the question's content
words, or a third of them while moderately close). If none does, or if words the archive never uses
make up a third of the question, the answer opens with "NO DIRECT ANSWER" and every chain is headed
"related context". Related context never reads as an answer. Residual gap: a question about an
attribute the archive never mentions, phrased with words the archive does use elsewhere ("what version
of Windows does the middleware run on"), can pass the test through the topic's own statements.

**verify_claims.py is state-aware.** It reads the deletion log to decide whether `data/` is the full
archive or post-deletion, and whether the shelf-life chain was touched, and asserts the matching
expectation. Checked: ALL PASS on the full archive, ALL PASS after a deletion that beheads the chain,
FAIL when the head_removed flag is cleared post-deletion, FAIL when it is faked on the full archive.

**Head-loss counts.** Deleting the technical consultant yields 71 chains that lost their head;
deleting the project manager yields 171. Both were verified independently by counting the old groups
whose current claim is among the removed claims: the sets match the log exactly, no double counting.

Of the nine, P8 and P9 are the two without a clean answer, and the system says so in both: P8's
trails end with "no later statement in the archive says it was done", and P9's chain ends with the
statement that "completed" meant the job ran and the file landed and that the content was never
checked, with every report line flagged as evidence of reporting only.

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
