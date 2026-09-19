"""Acceptance checks for retrieval + person-mention index. Usage: python verify_retrieve.py [seed]"""
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "corpus" / "acme"
DATA = ROOT / "data"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(10 ** 6)
random.seed(seed)
ok = True
r = Retriever()
people = json.loads((DATA / "people.json").read_text(encoding="utf-8"))
mentions = [json.loads(l) for l in (DATA / "mentions.jsonl").open(encoding="utf-8")]
by_unit = defaultdict(set)
for m in mentions:
    by_unit[m["unit_id"]].add((m["person_id"], m["role"]))

# 1. OP_ID question -> the originating email is in the top 3, cited at message level with lines
hits = r.search("who proposed removing the operator ID field from the data extract and who agreed", k=3)
pins = [h["pin"]["unit_id"] for h in hits]
p1 = "email:07_op-id-field-exclusion:m01" in pins
print("[1] OP_ID query: emails/07 message 1 in top 3 ->", "PASS" if p1 else f"FAIL {pins}")
ok &= p1
for h in hits:
    print(f"     {h['pin']['where']}  lines {h['pin']['line_start']}-{h['pin']['line_end']}")

# 2. shelf-life proportion -> top 6 pins span at least 3 distinct dates (currency needs the series)
hits = r.search("what proportion of articles had shelf-life data populated", k=6)
dates = {h["pin"]["date"][:10] for h in hits}
p2 = len(dates) >= 3
print(f"[2] shelf-life query: {len(dates)} distinct dates in top 6 ->", "PASS" if p2 else "FAIL")
ok &= p2
for h in hits:
    print(f"     {h['pin']['date'][:10]}  {h['pin']['where']}")

# 3. person index completeness, for whichever registered person speaks the most units (whoever
#    that is: the check must keep working no matter who has since been deleted)
speaker_counts = defaultdict(int)
for u in r.units.values():
    speaker_counts[u["speaker"]] += 1
top_speaker = max(speaker_counts, key=speaker_counts.get)
kw = next((p["person_id"] for p in people.values() if p["name"] == top_speaker), None)
spoken = [u for u in r.units.values() if u["speaker"] == top_speaker]
missing = [u["unit_id"] for u in spoken if (kw, "speaker") not in by_unit[u["unit_id"]]]
p3a = kw is not None and not missing
print(f"[3a] every unit spoken by the top speaker ({len(spoken)}) has a speaker row ->", "PASS" if p3a else f"FAIL {missing[:5]}")
p3b = any(role == "text_mention" for pairs in by_unit.values() for (_, role) in pairs)
print("[3b] at least one body-text mention carries a text_mention row ->", "PASS" if p3b else "FAIL")
p3c = not [p for p in people.values() if "(" in p["name"] or ")" in p["name"]]
print("[3c] no person record has a broken parenthesised name ->", "PASS" if p3c else "FAIL")
ok &= p3a and p3b and p3c

# 4. person filter returns only passages where the person is speaker/recipient/attendee/mentioned
hits = r.search("data extract landing zone", k=10, person=kw)
bad = [h["passage_id"] for h in hits if not any(kw == pid for u in h["units"] for (pid, _) in by_unit[u["unit_id"]])]
p4 = not bad and len(hits) > 0
print(f"[4] person-filtered search ({len(hits)} hits) all involve the top speaker ->", "PASS" if p4 else f"FAIL {bad[:3]}")
ok &= p4

# 5. index integrity: every passage unit exists, embeddings aligned
p5 = all(uid in r.units for p in r.passages for uid in p["units"]) and r.emb.shape[0] == len(r.passages)
print(f"[5] {len(r.passages)} passages reference existing units, embeddings aligned ->", "PASS" if p5 else "FAIL")
ok &= p5

# 6. a random hit's pin resolves to the source lines
TS_LINE = re.compile(r"^\d+:\d\d\d+:\d\d$")
INITIALS = re.compile(r"^[A-Z][A-Z0-9]{0,3}$")   # 'KB', and 'G1' for an unnamed 'Guest 1'
LAPTOP_LABEL = re.compile(r"^(Me|Them):\s*")


def spoken_words(unit, pin, lines):
    """The spoken words the pin's line range covers, with speaker chrome removed.

    A transcript turn merges consecutive utterances by one speaker, and the source interleaves
    speaker chrome between them, so a raw join of the range does not reproduce the stored text.
    Matching against the raw join fails on ~14% of turns that are correctly anchored. The two
    transcript formats hide the text differently:

    teams           chrome occupies whole lines above the text (speaker name, doubled timestamp,
                    initials, marker) — and not always all four, so drop by shape, not by offset.
    internal_laptop the marker line *is* the text line, behind a "Me:" / "Them:" label — dropping
                    it would discard the words themselves, so strip the label instead.
    """
    anchor = (unit or {}).get("anchor") or {}
    segments = anchor.get("segments") or []
    if not segments:
        return " ".join(w for l in lines[pin["line_start"] - 1:pin["line_end"]] for w in l.split())

    span = [lines[n - 1] for n in range(pin["line_start"], pin["line_end"] + 1)]
    if anchor.get("format") == "internal_laptop":
        keep = [LAPTOP_LABEL.sub("", l) for l in span]
    else:
        markers = {s["line"] for s in segments}
        speaker = anchor.get("speaker")
        keep = []
        for n, line in zip(range(pin["line_start"], pin["line_end"] + 1), span):
            stripped = line.strip()
            if n in markers or stripped == speaker or TS_LINE.match(stripped) or INITIALS.match(stripped):
                continue
            keep.append(line)
    return " ".join(w for l in keep for w in l.split())


q = random.choice(["bakery inside the fresh workstream", "UAT sign-off scope", "nightly article extract completed with no errors",
                   "twelve months return on investment", "pilot store group", "hypercare incidents DC2"])
h = random.choice(r.search(q, k=5))
pin = h["pin"]
lines = (CORPUS / pin["source_file"]).read_text(encoding="utf-8").split("\n")
src = spoken_words(r.units.get(pin["unit_id"]), pin, lines)
first_words = " ".join(pin["text"].split()[:6])
p6 = first_words in src
print(f"[6] random pin resolves to source (seed {seed}, query {q!r}) ->", "PASS" if p6 else "FAIL")
print(f"     {pin['where']}")
print(f"     record : {pin['text'][:110]!r}")
print(f"     source {pin['line_start']}: {lines[pin['line_start'] - 1][:110]!r}")
ok &= p6

print("ALL PASS" if ok else "SOME FAILED")
