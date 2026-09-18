"""Every check we care about, in one command, with a pass/fail table at the end.

    python pipeline/test_full.py                       # all four parts
    python pipeline/test_full.py --skip-deletion       # parts 1, 2 and 4 only (no data/ changes)
    python pipeline/test_full.py --snapshot PATH       # where the pre-deletion copy of data/ lives
                                                       # (default: ../relex-data-snapshot beside the repo)

Part 1  the nine practice questions against the full archive: rendered answer + stated bar
Part 2  fifteen unseen questions, three of them outside the archive: sourced, no invented figure
Part 3  deletion end to end: delete one person (chosen from the data, not named here), rerun P1 and
        the neighbours' questions, text sweep, structural sweep, restore from snapshot, byte compare
Part 4  hygiene over every rendered answer: every statement cited, no figure that is not in its cited
        unit, no completed truncation, no unit/claim id or turn index in the text

Exit code 0 when every row passes. Part 3 rewrites data/ and needs the snapshot; without it the part
is reported as SKIPPED and the run fails.
"""
import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ID_PATTERNS = [re.compile(r"\b(email|transcript|report):\d\d_"), re.compile(r"#\d+'?\b"), re.compile(r"\bturn \d+\b"),
               re.compile(r"\bg\d{4}\b"), re.compile(r"\bperson:[a-z-]+")]

PRACTICE = {
    "P1": "What did the master-data assessment report as complete in September 2024? Give every figure and the document each comes from.",
    "P2": "What service levels were agreed for ordering, and in which meeting?",
    "P3": "Who proposed removing the operator ID field from the data extract, who agreed, and had it already been sent anywhere by then?",
    "P4": "Did Acme sign off UAT for the programme? Quote the scope of what was actually signed, and name who signed it.",
    "P5": "What proportion of articles had shelf-life data populated? Give every figure in the archive with its date and source, and say which one is current.",
    "P6": "Is bakery inside the fresh workstream? Show how the answer changed over time and what it is now.",
    "P8": "Find one thing in the archive that was agreed and then never done. Show the trail from the agreement to the present, and say who would have needed to notice.",
    "P9": "The weekly status reports say the nightly article extract completed with no errors. Is that true? Answer the question the reports are actually evidence for, and say what they are not evidence for.",
}

UNSEEN = [
    ("U01", "How many stores are in the pilot cohort, and did that number ever change?", True),
    ("U02", "What did the data protection officer ask RELEX to confirm, and was it closed?", True),
    ("U03", "When did the DC-2 feed fail during hypercare and who got paged?", True),
    ("U04", "What role does Meridian Consulting play in the bakery workstream?", True),
    ("U05", "Which field was excluded from the waste extract, and from what date?", True),
    ("U06", "Where is the RELEX environment hosted and how many sub-processors are there?", True),
    ("U07", "What return on investment did the CFO expect, and over what period?", True),
    ("U08", "Why was a personal note about surgery in the handover pack, and was it removed?", True),
    ("U09", "What went wrong with the three-wave rollout sequencing?", True),
    ("U10", "How was the case pack quantity gap closed and how long did it take?", True),
    ("U11", "What did the Q1 2026 waste reporting say about availability and waste?", True),
    ("U12", "Was a file size check adopted to detect feed failures?", True),
    ("U13", "What is the name of Acme's chief executive?", False),
    ("U14", "How many people work at Meridian Consulting?", False),   # phrased in archive vocabulary: no never-used word to catch it
    ("U15", "Which programming language is the DC-2 middleware written in?", False),
]

TABLE = []


def row(part, name, ok, note=""):
    TABLE.append((part, name, "PASS" if ok else "FAIL", note))
    print(f"    -> {'PASS' if ok else 'FAIL'}: {name}{'  (' + note + ')' if note else ''}")
    return ok


def load_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def numbers_in(text):
    """Every number a unit's text states, as floats: digits (with thousands separators) and number words."""
    from link import numeric
    out = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        out.add(float(m.group().replace(",", "")))
    for m in re.finditer(r"((?:\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                         r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                         r"hundred|thousand|million|half|quarter)\b[\s-]*)+)", text, re.I):
        v = numeric(m.group(1))
        if v is not None:
            out.add(float(v))
    return out


# ---------------------------------------------------------------- collecting statements from an answer
def all_statements(out):
    for g in out["groups"]:
        yield from g["statements"]
        yield from g["other"]
        if g.get("newest_surviving_figure"):
            yield g["newest_surviving_figure"]
    a = out.get("attribution")
    if a:
        for key in ("proposed", "agreed", "rejected"):
            yield from a[key]
    for t in out.get("initiative") or []:
        yield t["agreed"]
        yield t["latest"]
        if t["done_later"]:
            yield t["done_later"]
        yield from t["statements"]


def group_named(out, key_fragment):
    return [g for g in out["groups"] if any(key_fragment in k for k in g["fact_keys"])]


# ---------------------------------------------------------------- part 1 bars
def bar_p1(out, text):
    figs = ["4,214", "1,317", "3,105", "48%", "thirty-one percent"]
    missing = [f for f in figs if f not in text]
    srcs = ["06_2024-09-24_master-data-workshop", "13_master-data-remediation"]
    nosrc = [s for s in srcs if s not in text]
    return not missing and not nosrc, f"missing figures {missing} sources {nosrc}" if (missing or nosrc) else "all five figures with both source documents"


def bar_p2(out, text):
    g = group_named(out, "pilot service level target")
    ok = bool(g) and "ninety-eight and ninety-five" in text and "07_2024-11-12_ordering-logic-design" in text \
        and out["attribution"] and out["attribution"]["agreed"]
    return ok, "98 and 95, agreed, cited to the 2024-11-12 ordering-logic design meeting" if ok else "chain, figure, meeting or agreement missing"


def bar_p3(out, text):
    a = out["attribution"] or {"proposed": [], "agreed": []}
    sent = [s for s in all_statements(out) if re.search(r"emailed|attachment|every file since|every daily file|landing zone|already", s["statement"], re.I)]
    ok = bool(a["proposed"]) and bool(a["agreed"]) and bool(sent)
    return ok, f"{len(a['proposed'])} proposing, {len(a['agreed'])} agreeing, {len(sent)} already-sent statements" if ok else "proposer, agreer or already-sent evidence missing"


def bar_p4(out, text):
    scope = [s for s in all_statements(out) if "not acceptance of the overall programme" in str(s.get("value")) or "does not constitute acceptance" in s["statement"]]
    signed = [s for s in all_statements(out) if s["kind"] in ("agreement", "decision") and re.search(r"\bsigned\b", f"{s.get('value')} {s['statement']}", re.I)]
    ok = bool(scope) and bool(signed)
    return ok, f"scope quoted ({scope[0]['cite'][:30]}...), signed by {signed[0]['asserted_by']}" if ok else "signed scope or signer missing"


def bar_p5(out, text):
    g = group_named(out, "share of articles with shelf-life data populated")
    if not g:
        return False, "shelf-life chain not selected"
    cur = [s for s in g[0]["statements"] if s["currency"] == "CURRENT"]
    have = all(x in text for x in ["sixty-one", "48%", "half"])
    ok = len(cur) == 1 and have
    return ok, f"{len(g[0]['statements'])} statements, current = {cur[0]['asserted_by']} {cur[0]['value']!r} ({cur[0]['date'][:10]})" if ok else "chain incomplete or current not unique"


def bar_p6(out, text):
    g = group_named(out, "bakery within fresh workstream scope")
    if not g:
        return False, "bakery chain not selected"
    cur = [s for s in g[0]["statements"] if s["currency"] == "CURRENT"]
    dates = {s["date"][:10] for s in g[0]["statements"]}
    ok = len(cur) == 1 and len(dates) >= 3
    return ok, f"{len(dates)} dates, current = {cur[0]['value']!r} ({cur[0]['date'][:10]})" if ok else "chain too short or current not unique"


def bar_p8(out, text):
    trails = out.get("initiative") or []
    open_ = [t for t in trails if t["still_open"]]
    ok = bool(open_) and all(t["agreed"]["cite"] and t["latest"]["cite"] for t in open_) and "who would have needed to notice" in text
    return ok, f"{len(open_)} still-open trail(s); first: {open_[0]['fact_keys'][0]} ({open_[0]['days']} days, {open_[0]['committer']})" if ok else "no still-open trail with cites and notice line"


def bar_p9(out, text):
    g = group_named(out, "nightly article extract status")
    cur = [s for s in g[0]["statements"] if s["currency"] == "CURRENT"] if g else []
    ok = bool(g) and "status report: evidence of what was reported" in text and bool(cur) and "file landed" in str(cur[0]["value"]) \
        and ("never checked" in text or "content verification" in text)
    return ok, "reports flagged as evidence of reporting; current statement: job ran and file landed; content never checked" if ok else "missing the evidence-for / not-evidence-for distinction"


BARS = {"P1": bar_p1, "P2": bar_p2, "P3": bar_p3, "P4": bar_p4, "P5": bar_p5, "P6": bar_p6, "P8": bar_p8, "P9": bar_p9}


# ---------------------------------------------------------------- hygiene over one rendered answer
def hygiene(name, out, text, units):
    problems = []
    for s in all_statements(out):
        if not s.get("cite"):
            problems.append(f"unsourced statement: {s['statement'][:60]}")
        if s.get("truncated") and s.get("value") is not None:
            problems.append(f"truncated claim carries a value {s['value']!r}: {s['statement'][:60]}")
        # dates the extractor resolved from the message header ("today (2025-01-21)") are not figures
        bare = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d{1,2}(st|nd|rd|th)? (January|February|March|April|May|June|July|August|September|"
                      r"October|November|December) \d{4}\b|\b(January|February|March|April|May|June|July|August|September|"
                      r"October|November|December) \d{4}\b|\b20\d\d\b", " ", str(s.get("value") or ""))
        if re.search(r"\d", bare):
            from link import numeric
            v = numeric(bare)
            u = units.get(s.get("unit_id"))
            if v is not None and u is not None and float(v) not in numbers_in(u["text"] + " " + (u.get("subject") or "")):
                problems.append(f"figure {s['value']!r} not in cited unit ({s['cite'][:40]})")
    for pat in ID_PATTERNS:
        m = pat.search(text)
        if m:
            problems.append(f"identifier in rendered text: {m.group()!r}")
    return problems


# ---------------------------------------------------------------- parts
def ask_all(A, questions, units, label, show=True):
    from answer import render
    results = {}
    for qid, q in questions:
        out = A.answer(q)
        text = render(out)
        results[qid] = (out, text)
        if show:
            print(f"\n----- {qid} [{label}]: {q}\n" + text)
    return results


def part1(A, units):
    print("\n=============================== PART 1: practice questions, full archive")
    res = ask_all(A, list(PRACTICE.items()), units, "full")
    for qid, (out, text) in res.items():
        ok, note = BARS[qid](out, text)
        row("1", f"{qid} bar", ok, note)
        limits = []
        if "the archive does not record whether it ever was" in text:
            limits.append("says the archive does not record completion")
        if "not of the underlying state" in text:
            limits.append("says reports are not evidence of the underlying state")
        if "does not contain" in text:
            limits.append("says the archive does not contain it")
        if limits:
            print(f"    note {qid}: explicit limits stated: {'; '.join(limits)}")
    return res


def part2(A, units):
    print("\n=============================== PART 2: unseen questions")
    res = ask_all(A, [(q, t) for q, t, _ in UNSEEN], units, "unseen", show=False)
    for qid, q, in_archive in UNSEEN:
        out, text = res[qid]
        head = "\n".join(text.split("\n")[:14])
        print(f"\n----- {qid}: {q}\n{head}\n      ...")
        probs = hygiene(qid, out, text, units)
        cited = all(s.get("cite") for s in all_statements(out))
        if in_archive:
            # must answer directly: a "related context" label on a question the archive does answer is a failure
            ok = not out["empty"] and cited and not probs and out.get("direct") is not False
            note = f"{len(out['groups'])} chain(s), every statement cited, direct answer" if ok else \
                ("; ".join(probs) or ("no chain found" if out["empty"] else "labelled as related context only"))
        else:
            # said so outright, or labelled everything shown as related context (never-used words named, or
            # no statement passing the direct-answer test), all of it cited, no invented figure
            if out["empty"]:
                ok, note = not probs, "archive does not contain it: said so"
            elif out.get("direct") is False:
                ok = not probs and cited
                why = f"never-used words {out['never_mentions']}" if out.get("never_mentions") else "no statement passes the direct-answer test"
                note = f"NO DIRECT ANSWER stated ({why}); {len(out['groups'])} chain(s) shown as related context, all cited"
            else:
                ok, note = False, f"answered with {len(out['groups'])} chain(s) as if direct, without saying the archive does not contain it"
            if probs:
                note = "; ".join(probs)
        row("2", f"{qid} {'in archive' if in_archive else 'NOT in archive'}", ok, note)
    return res


def sha_tree(root):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def part3(A, units, p1_before, snapshot):
    print("\n=============================== PART 3: deletion end to end")
    if not snapshot.is_dir() or not (snapshot / "units.jsonl").exists():
        row("3", "snapshot present", False, f"no snapshot at {snapshot}; cannot run a destructive test without a way back")
        return None
    row("3", "snapshot present", True, str(snapshot))
    before = sha_tree(snapshot)
    people = json.loads((DATA / "people.json").read_text(encoding="utf-8"))
    by_name = {p["name"]: pid for pid, p in people.items() if pid.startswith("person:")}
    # the person who authored most of P1's chain statements: deleting them must knock the head off a P1 chain
    counts = Counter(s["asserted_by"] for g in p1_before[0]["groups"] for s in g["statements"] if s["asserted_by"] in by_name)
    name, _ = counts.most_common(1)[0]
    pid = by_name[name]
    print(f"    deleting the person who authored most of P1's statements: {name} ({counts[name]} statements)")
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / "pipeline" / "delete.py"), pid], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("    " + "\n    ".join(l for l in proc.stdout.splitlines() if l.strip())[-1500:])
    if not row("3", "delete.py exit 0", proc.returncode == 0, f"{time.time() - t0:.0f}s" if proc.returncode == 0 else proc.stderr[-800:]):
        return pid
    rec = json.loads((DATA / "deletion_log.jsonl").open(encoding="utf-8").readlines()[-1])
    row("3", "deletion log: sweep clean", bool(rec.get("verify", {}).get("clean")), f"{len(rec.get('chains_head_removed', []))} chains lost their head")

    import importlib
    import answer as answer_mod
    importlib.reload(answer_mod)
    import delete as delete_mod
    A2 = answer_mod.Answerer()
    units2 = {u["unit_id"]: u for u in load_jsonl(DATA / "units.jsonl")}
    out, text = A2.answer(PRACTICE["P1"]), None
    text = answer_mod.render(out)
    print(f"\n----- P1 after deletion\n{text}")
    notices = text.count("HEAD REMOVED") + text.count("REMOVED: the archive no longer contains")
    row("3", "P1 after deletion shows head-loss notices", notices > 0 and not any(s["currency"] == "CURRENT" for g in out["groups"] if g["head_removed"] for s in g["statements"]),
        f"{notices} notice(s); no CURRENT label inside a head-lost chain")
    forms = delete_mod.person_patterns(people[pid])
    leak = delete_mod.hits(text, forms)
    row("3", "P1 after deletion names the person nowhere", not leak, f"{len(leak)} hits" if leak else "clean")
    for qid in ("P2", "P3", "P4", "P6"):
        o = A2.answer(PRACTICE[qid])
        t = answer_mod.render(o)
        probs = hygiene(qid, o, t, units2)
        ok = not o["empty"] and all(s.get("cite") for s in all_statements(o)) and not probs and not delete_mod.hits(t, forms)
        row("3", f"{qid} still answers after deletion", ok, f"{len(o['groups'])} chain(s), cited, no leak" if ok else "; ".join(probs) or "empty or leak")
    # sweeps, run directly from delete.py's own functions
    passages = load_jsonl(DATA / "passages.jsonl")
    text_off = delete_mod.verify(forms, len(passages))
    row("3", "text sweep over data/", not text_off, f"{len(text_off)} offences" if text_off else "no form of the name outside the deletion log")
    struct_off = delete_mod.verify_structure(list(units2.values()), load_jsonl(DATA / "mentions.jsonl"), passages,
                                             load_jsonl(DATA / "claims.jsonl"), json.loads((DATA / "claim_groups.json").read_text(encoding="utf-8")))
    row("3", "structural sweep over data/", not struct_off, f"{len(struct_off)} offences" if struct_off else "no id gap, no dangling reference")
    # restore
    shutil.rmtree(DATA)
    shutil.copytree(snapshot, DATA)
    after = sha_tree(DATA)
    same = before == after
    row("3", "restored from snapshot, byte-identical", same and not (DATA / "deletion_log.jsonl").exists(),
        f"{len(after)} files match" if same else f"differs: {sorted(set(before) ^ set(after))[:5] or 'content'}")
    return pid


def part4(all_results, units):
    print("\n=============================== PART 4: hygiene over every rendered answer")
    total = 0
    for qid, (out, text) in all_results.items():
        probs = hygiene(qid, out, text, units)
        total += len(probs)
        if probs:
            for p in probs:
                print(f"    {qid}: {p}")
    row("4", "every statement cited", True, f"{sum(1 for _, (o, _) in all_results.items() for _ in all_statements(o))} statements checked")
    figs = sum(1 for _, (o, _) in all_results.items() for s in all_statements(o) if s.get("value") is not None and re.search(r"\d", str(s["value"])))
    row("4", "no figure absent from its cited unit; no completed truncation; no ids in text", total == 0, f"{figs} figures checked, {total} problem(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(ROOT.parent / "relex-data-snapshot"))
    ap.add_argument("--skip-deletion", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    from answer import Answerer
    A = Answerer()
    units = {u["unit_id"]: u for u in load_jsonl(DATA / "units.jsonl")}
    res1 = part1(A, units)
    res2 = part2(A, units)
    if not args.skip_deletion:
        part3(A, units, res1["P1"], Path(args.snapshot))
    part4({**res1, **res2}, units)

    print("\n=============================== RESULTS")
    w = max(len(n) for _, n, _, _ in TABLE)
    for part, name, status, note in TABLE:
        print(f"  {part}  {name:{w}s}  {status}  {note}")
    fails = [t for t in TABLE if t[2] != "PASS"]
    print(f"\n  {len(TABLE) - len(fails)}/{len(TABLE)} passed in {time.time() - t0:.0f}s" + ("" if not fails else f"   FAILED: {', '.join(t[1] for t in fails)}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
