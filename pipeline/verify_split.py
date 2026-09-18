"""Acceptance checks for the splitter output. Usage: python verify_split.py [seed]"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "corpus" / "acme"
units = [json.loads(l) for l in (ROOT / "data" / "units.jsonl").open(encoding="utf-8")]
ok = True

# 1. emails/07 -> exactly 4 records, 4 distinct dates, 3 distinct senders
#    (brief said 4 senders; the source has Priya Nair at lines 6 and 76, a third sender at 25, Nadia at 51)
u07 = [u for u in units if u["source_file"] == "emails/07_op-id-field-exclusion.txt"]
dates = {u["date"] for u in u07}
senders = {u["speaker"] for u in u07}
p1 = (len(u07), len(dates), len(senders)) == (4, 4, 3)
print(f"[1] emails/07: {len(u07)} records, {len(dates)} distinct dates, {len(senders)} distinct senders ->",
      "PASS" if p1 else "FAIL")
ok &= p1
for u in u07:
    a = u["anchor"]
    print(f"     {u['unit_id']}  {u['date']}  {u['speaker']:14s}  lines {a['line_start']}-{a['line_end']}  "
          f"text[:60]={u['text'][:60]!r}")

# 2. every transcript turn's anchor resolves to the exact marker line in the source
seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(10 ** 6)
random.seed(seed)
turns = [u for u in units if u["unit_type"] == "transcript_turn"]
cache = {}
bad = []
for u in turns:
    lines = cache.setdefault(u["source_file"], (CORPUS / u["source_file"]).read_text(encoding="utf-8").split("\n"))
    a = u["anchor"]
    marker = lines[a["line_start"] - 1]
    if a["format"] == "teams":
        m, s = divmod(a["offset_seconds"], 60)
        mm = f"{m} minute{'s' if m != 1 else ''}"
        ss = f"{s} second{'s' if s != 1 else ''}"
        forms = {f"{u['speaker']} {mm} {ss}"}
        if m == 0:
            forms.add(f"{u['speaker']} {ss}")
        if s == 0:
            forms.add(f"{u['speaker']} {mm}")
        good = marker in forms
    else:
        good = marker.startswith(f"{u['speaker']}:")
    span = " ".join(l.strip() for l in lines[a["line_start"] - 1:a["line_end"]])
    good &= all(piece in span for piece in u["text"].split(" ")[:5])
    if not good:
        bad.append(u["unit_id"])
print(f"[2] all {len(turns)} transcript turns resolve to their marker line ->",
      "PASS" if not bad else f"FAIL ({len(bad)}: {bad[:5]})")
ok &= not bad
pick = random.choice([u for u in turns if u["anchor"]["format"] == "teams" and len(u["text"]) > 40])
lines = cache[pick["source_file"]]
a = pick["anchor"]
print(f"     random pick (seed {seed}): {a['human']}")
print(f"       record text : {pick['text'][:120]!r}")
for ln in range(a["line_start"], min(a["line_end"], a["line_start"] + 3) + 1):
    print(f"       source {ln:5d}: {lines[ln - 1]!r}")

# 3. no placeholders / banner / synthetic header survive in text or signature
def leaks(u):
    blob = u["text"] + "\n" + (u.get("signature") or "")
    if any(t in blob for t in ["[Image removed", "[cid:", "Bild borttagen", "originated from outside", "*** SYNTHETIC"]):
        return True
    return any(l.strip() == "Image" or l.endswith(" Image") for l in blob.split("\n"))

leak = [u["unit_id"] for u in units if leaks(u)]
print(f"[3] placeholder/banner/header leakage: {len(leak)} ->", "PASS" if not leak else f"FAIL {leak[:5]}")
ok &= not leak

# 4. required fields present, ids unique
req = ["unit_id", "source_file", "unit_type", "date", "speaker", "recipients", "text", "anchor"]
missing = [u["unit_id"] for u in units if any(u.get(k) in (None, "", []) for k in req)]
print(f"[4] required fields present on all {len(units)} units ->", "PASS" if not missing else f"FAIL {missing[:8]}")
ok &= not missing
ids = [u["unit_id"] for u in units]
p5 = len(ids) == len(set(ids))
print("[5] unit_ids unique ->", "PASS" if p5 else "FAIL")
ok &= p5

# 5. email/report dates strictly decrease from top of thread (reverse-chronological as stated)
disorder = []
for f in {u["source_file"] for u in units if u["anchor"]["kind"] == "email"}:
    ms = sorted((u for u in units if u["source_file"] == f), key=lambda u: u["anchor"]["position_from_top"])
    for x, y in zip(ms, ms[1:]):
        if not x["date"] > y["date"]:
            disorder.append((f, x["unit_id"], y["unit_id"]))
print(f"[6] threads reverse-chronological ->", "PASS" if not disorder else f"WARN {disorder}")

# 7. no speaker-time marker leaked into any transcript turn's text
import re
marker_re = re.compile(r"(\d+ minutes?( \d+ seconds?)?|\d+ seconds?)$")
leak2 = []
for u in turns:
    for name in {x["name"] for x in u["attendees"]} | {u["speaker"]}:
        if re.search(rf"{re.escape(name)} \d+ (minutes?|seconds?)", u["text"]):
            leak2.append(u["unit_id"]); break
print(f"[7] speaker-time markers leaked into turn text: {len(leak2)} ->", "PASS" if not leak2 else f"FAIL {leak2[:5]}")
ok &= not leak2

print("ALL PASS" if ok else "SOME FAILED")
