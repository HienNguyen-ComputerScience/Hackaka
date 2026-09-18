"""Acceptance checks for the supersession / truth-status model."""
import json
import logging
import sys
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer import Answerer  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
C = [json.loads(l) for l in (DATA / "claims.jsonl").open(encoding="utf-8")]
by_id = {c["claim_id"]: c for c in C}
ok = True


def group_of(key):
    gid = next(c["group_id"] for c in C if c["fact_key_norm"] == key)
    return sorted([c for c in C if c["group_id"] == gid], key=lambda c: (c["date"], c["turn_index"] or 0))


def chain(members):
    return [c for c in members if c["chain_pos"]]


# P5 shelf-life series. State-aware: data/ is either the full archive (no deletion log) or post-deletion
# (a log exists). The expectation follows the state, and the state is read from the data, never assumed.
#   full archive    one CURRENT, the four dated statements the corpus carries, nothing flagged head_removed
#   post-deletion   if the log says this chain lost its head: head_removed on the group and on exactly one
#                   newest-surviving claim, the removed member ids gone, removed count matching the log;
#                   if the log does not touch the chain: intact, one CURRENT, no head_removed flag
GROUPS = json.loads((DATA / "claim_groups.json").read_text(encoding="utf-8"))
LOG = DATA / "deletion_log.jsonl"
log_records = [json.loads(l) for l in LOG.open(encoding="utf-8") if l.strip()] if LOG.exists() else []
g = chain(group_of("share of articles with shelf-life data populated"))
grp = GROUPS[g[0]["group_id"]]
cur = [c for c in g if c["is_current"]]
dates = {c["date"][:10] for c in g}
chain_ids = {c["claim_id"] for c in g}
if not log_records:
    state = "full archive"
    need = {"2024-04-18", "2024-09-24", "2024-09-30", "2025-09-12"}
    p5 = (len(cur) == 1 and need <= dates and not grp.get("head_removed") and not any(c.get("head_removed") for c in g)
          and all(c["superseded_by"] for c in g if not c["is_current"]) and all(c["where"] for c in g))
    detail = f"{len(cur)} current, required dates present={need <= dates}, no head_removed flag={not grp.get('head_removed')}"
else:
    state = f"post-deletion ({len(log_records)} log record(s))"
    touching = [r for rec in log_records for r in rec.get("chains_head_removed", [])
                if r.get("new_group") == g[0]["group_id"] or chain_ids & set(r.get("members_surviving", []))]
    removed_ids = {m for r in touching for m in r.get("members_removed", [])}
    if touching:
        expected_removed = sum(r.get("removed_statements_total", len(r.get("members_removed", []))) for r in touching[-1:])
        flagged = [c for c in g if c.get("head_removed")]
        p5 = (grp.get("head_removed") is True and grp.get("removed_statements") == expected_removed
              and len(cur) == 1 and flagged == cur
              and not (chain_ids & removed_ids) and all(c["where"] for c in g))
        detail = (f"chain lost its head: head_removed={grp.get('head_removed')} removed={grp.get('removed_statements')} "
                  f"(log says {expected_removed}), newest survivor flagged={flagged == cur}, removed ids absent={not (chain_ids & removed_ids)}")
    else:
        p5 = (len(cur) == 1 and not grp.get("head_removed") and not any(c.get("head_removed") for c in g)
              and all(c["superseded_by"] for c in g if not c["is_current"]) and all(c["where"] for c in g))
        detail = f"chain untouched by the deletion: {len(cur)} current, no head_removed flag={not grp.get('head_removed')}"
print(f"[P5] shelf-life chain, state = {state}: {detail} ->", "PASS" if p5 else "FAIL")
for c in g:
    lab = ("NEWEST SURVIVING" if c.get("head_removed") else "CURRENT") if c["is_current"] else "superseded"
    print(f"      {c['date'][:10]} {c['asserted_by']:15s} {lab:16s} {c['truth_status']:15s} {c.get('value')!r:34s} {c['where'][:60]}")
ok &= bool(p5)

# P6 bakery scope
g = chain(group_of("bakery within fresh workstream scope"))
cur = [c for c in g if c["is_current"]]
p6 = len(cur) == 1 and len({c["date"][:10] for c in g}) >= 3
print(f"[P6] bakery-in-fresh chain: {len(g)} statements over {len({c['date'][:10] for c in g})} dates, {len(cur)} current ->", "PASS" if p6 else "FAIL")
for c in g:
    print(f"      {c['date'][:10]} {c['asserted_by']:15s} {'CURRENT' if c['is_current'] else 'superseded':10s} {c['truth_status']:15s} {c.get('value')!r:34s} {c['where'][:60]}")
ok &= p6

# never_true 1: Marco's 40-store pilot figure, corrected next day in emails/09
g = group_of("pilot cohort store count")
m40 = [c for c in g if c["asserted_by"] == "Marco Rossi" and c["date"].startswith("2025-05-20")]
nt1 = m40 and all(c["truth_status"] == "never_true" and c["corrected_by"] and
                  by_id[c["corrected_by"]]["source_file"].endswith("09_pilot-store-group.txt") and
                  by_id[c["corrected_by"]]["date"].startswith("2025-05-21") for c in m40)
print(f"[NT1] Marco Rossi 2025-05-20 '40 stores' flagged never_true with emails/09 correction cited ->", "PASS" if nt1 else "FAIL")
for c in m40:
    print(f"      {c['truth_status']}  {c['statement'][:90]}  <- {c['truth_reason'][:110]}")
    if c["corrected_by"]:
        print(f"      corrected by: {by_id[c['corrected_by']]['where']}")
ok &= bool(nt1)

# never_true 2: Tomas's file-size-check claim, withdrawn in the hypercare review (transcript 11)
g = group_of("file size check as feed failure detection")
nt = [c for c in g if c["truth_status"] == "never_true"]
nt2 = nt and all(by_id[c["corrected_by"]]["source_file"].startswith("transcripts/11_") and by_id[c["corrected_by"]]["kind"] in ("withdrawal", "correction") for c in nt)
print(f"[NT2] file-size-check claim(s) flagged never_true with the transcript-11 withdrawal cited ({len(nt)}) ->", "PASS" if nt2 else "FAIL")
for c in nt:
    print(f"      {c['date'][:10]} {c['asserted_by']} [{c['kind']}] {c['statement'][:100]}")
    print(f"      withdrawn by: {by_id[c['corrected_by']]['where']}  \"{by_id[c['corrected_by']]['statement'][:100]}\"")
print("      note: earlier statements in this group and their status:")
for c in g:
    print(f"        {c['date'][:10]} {c['asserted_by']:15s} [{c['kind']:12s}] {c['truth_status']:15s} {c['statement'][:80]}")
ok &= bool(nt2)

# P3 attribution
A = Answerer()
out = A.answer("Who proposed removing the operator ID field from the data extract, who agreed, and had it already been sent anywhere by then?")
att = out["attribution"]
prop_units = {p["cite"] for p in att["proposed"]}
agr_units = {a["cite"] for a in att["agreed"]}
sent = [c for c in C if c["group_id"] in {c2["group_id"] for c2 in C if "op_id" in c2["fact_key_norm"]}
        and any(w in c["statement"].lower() for w in ("emailed", "attachment", "every file since", "every daily file", "distribution list", "already delivered", "since february"))]
p3 = bool(att["proposed"]) and bool(att["agreed"]) and not (prop_units & agr_units) and bool(sent)
print(f"[P3] proposer turns={len(att['proposed'])} agreeing turns={len(att['agreed'])} disjoint cites={not (prop_units & agr_units)} already-sent evidence={len(sent)} ->", "PASS" if p3 else "FAIL")
for p in att["proposed"][:6]:
    print(f"      PROPOSED {p['date'][:10]} {p['asserted_by']}: {p['statement'][:90]}  | {p['cite'][:70]}")
for a in att["agreed"][:6]:
    print(f"      AGREED   {a['date'][:10]} {a['asserted_by']}: {a['statement'][:90]}  | {a['cite'][:70]}")
for s in sent[:4]:
    print(f"      SENT?    {s['date'][:10]} {s['asserted_by']}: {s['statement'][:100]}  | {s['where'][:60]}")
ok &= p3

# same-day
sd = [c for c in C if c["order_confidence"] == "same_day_unknown"]
print(f"[SD] links with order_confidence=same_day_unknown: {len(sd)} ->", "PASS" if sd else "FAIL")
for c in sd[:6]:
    n = by_id[c["superseded_by"]]
    print(f"      {c['date'][:10]}  {c['where'][:55]}  ->  {n['where'][:55]}")
ok &= bool(sd)

print("ALL PASS" if ok else "SOME FAILED")
