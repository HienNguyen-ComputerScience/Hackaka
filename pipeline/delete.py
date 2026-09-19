"""Erase one person from every store, then rebuild everything derived. Deterministic, no model calls
except the local MiniLM re-embedding of the rebuilt passages.

    python pipeline/delete.py person:<id>            # do it
    python pipeline/delete.py person:<id> --dry-run  # plan only, write nothing

What "erase" means here, store by store:

  units.jsonl          units the person authored (speaker) are removed. Units by others that name
                       the person are kept but redacted: the person is dropped from to/cc/attendee
                       lists and every name / alias / e-mail / initials form in any string field is
                       replaced with [redacted]. So "Alex is checking" becomes "[redacted] is checking".
  extract_input/*.txt  same treatment, block by block (these are the extractor's inputs).
  claims_raw/*.jsonl   claims from removed units, claims asserted by the person, and claims whose
                       text (statement, entities, corrects, responds_to, value, fact_key) names
                       the person are removed. A claim about the person is evidence about them.
  people.json          the registry row goes, including aliases, e-mail forms and initials.
  mentions.jsonl       rebuilt from the scrubbed units with the person excluded from the registry.
  passages.jsonl       rebuilt from the scrubbed units and mentions (transcript windows re-stride).
  embeddings.npy       re-encoded from the rebuilt passages. Nothing is masked or zeroed.
  claims.jsonl         re-grouped and re-linked from the scrubbed claims_raw: supersession chains,
  claim_groups.json    merged claim groups, truth status and "current" pointers are recomputed, so a
                       chain whose current statement was the person's is marked head_removed with
                       the count of removed statements; its newest survivor is never called current.
                       Chains with no survivor stay as tombstone groups so they answer "removed".
  ids                  every surviving unit is renumbered contiguously per source file/thread (e-mail
                       and report position_chrono) or per transcript (turn_index), in split.py's own
                       id format, so no rendered ordinal or gap discloses that something was removed.
                       The remap cascades into claims_raw, extract_input block headers, claim ids
                       (unit_id#n), and everything rebuilt afterwards (mentions, passages, claims,
                       claim_groups). The remap itself is recorded on the deletion log entry.
  caches               any answers/cache/summaries directory or *.cache file under data/ is deleted.

Then a full-text sweep of every file under data/ checks that no form of the name survives, and a
structural sweep checks every id is gapless and every reference to a unit or claim id resolves. The
only place the name or an old id may appear afterwards is data/deletion_log.jsonl.

There is no tombstone and no filter: the person is not hidden, the data is gone. Getting them back
means re-running the pipeline from the corpus (split -> extract -> people -> link -> index).
"""
import argparse
import glob
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import people as people_mod  # noqa: E402
import index as index_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = DATA / "deletion_log.jsonl"
REDACTED = "[redacted]"
CACHE_GLOBS = ["answers*", "cache*", "summaries*", "*.cache", "*_cache", "*.pkl"]
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml"}


# ---------------------------------------------------------------- matching
def person_patterns(p):
    """Every textual form of the person that must not survive.

    Returns [(label, compiled_regex)]. Names are word-bounded and case-insensitive; e-mail addresses
    and their local parts are matched anywhere; initials must stand alone as a token (KB, not KBytes).
    """
    pats = []
    # longest forms first: the whole address must go before the surname inside it is redacted,
    # otherwise "a.surname@x" would leave "a.[redacted]@x" behind
    for e in p["emails"]:
        pats.append(("email", re.compile(re.escape(e), re.I)))
        local = e.split("@")[0]
        pats.append(("email_local", re.compile(rf"(?<![\w.])(?:{re.escape(local)})(?![\w])", re.I)))
    names = set(p["aliases"]) | {p["name"], people_mod.fold(p["name"])}
    for part in p["name"].split():
        names |= {part, people_mod.fold(part)}
    alt = "|".join(sorted((re.escape(n) for n in names if n), key=len, reverse=True))
    pats.append(("name", re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])", re.I)))
    if p.get("initials") and len(p["initials"]) >= 2:
        pats.append(("initials", re.compile(rf"(?<![A-Za-z0-9]){re.escape(p['initials'])}(?![A-Za-z0-9])")))
    pats.append(("person_id", re.compile(re.escape(p["person_id"]))))
    return pats


def hits(text, pats):
    return [(label, m.group()) for label, pat in pats for m in pat.finditer(text)]


def scrub_string(s, pats):
    for _, pat in pats:
        s = pat.sub(REDACTED, s)
    return s


def scrub_obj(obj, pats, changed):
    """Recursively redact a JSON object in place. Dicts with a matching name/email inside a list
    (recipients, attendees) are dropped; any other string is redacted. `changed` collects field notes."""
    if isinstance(obj, str):
        if hits(obj, pats):
            changed.append("string")
            return scrub_string(obj, pats)
        return obj
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict) and any(isinstance(v, str) and hits(v, pats) for k, v in item.items() if k in ("name", "email")):
                changed.append("list_entry_dropped")
                continue
            out.append(scrub_obj(item, pats, changed))
        return out
    if isinstance(obj, dict):
        for k in list(obj):
            before = len(changed)
            obj[k] = scrub_obj(obj[k], pats, changed)
            if len(changed) > before:
                changed.append(f"field:{k}")
        return obj
    return obj


def load_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- stages
def plan_units(units, mentions, person, pats):
    """Which units go entirely (person is the author) and which get redacted (person named by others)."""
    authored = {m["unit_id"] for m in mentions if m["person_id"] == person["person_id"] and m["role"] == "speaker"}
    for u in units:  # belt and braces: header forms the mention index could have missed
        if pats[0][1].fullmatch(u["speaker"] or "") or (u.get("speaker_email") or "").lower() in {e.lower() for e in person["emails"]}:
            authored.add(u["unit_id"])
    touched = {m["unit_id"] for m in mentions if m["person_id"] == person["person_id"]} - authored
    for u in units:
        if u["unit_id"] not in authored and hits(json.dumps(u, ensure_ascii=False), pats):
            touched.add(u["unit_id"])
    return authored, touched


def scrub_units(units, authored, touched, pats):
    kept, redactions = [], {}
    for u in units:
        if u["unit_id"] in authored:
            continue
        if u["unit_id"] in touched:
            changed = []
            u = scrub_obj(u, pats, changed)
            redactions[u["unit_id"]] = sorted({c for c in changed if c.startswith("field:")} | {c for c in changed if c == "list_entry_dropped"})
        kept.append(u)
    return kept, redactions


def scrub_claims_raw(authored, pats, dry_run):
    """Remove claims from deleted units, authored by the person, or naming the person. Returns log rows."""
    removed = []
    for f in sorted((DATA / "claims_raw").glob("*.jsonl")):
        keep_lines = []
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                keep_lines.append(line)  # link.py drops it with a reason; not ours to judge
                continue
            reason = None
            if c.get("unit_id") in authored:
                reason = "unit_deleted"
            elif pats[0][1].fullmatch(c.get("asserted_by") or ""):
                reason = "authored"
            elif hits(json.dumps(c, ensure_ascii=False), pats):
                reason = "names_person"
            if reason:
                removed.append({"file": f.name, "claim_id": c.get("claim_id"), "unit_id": c.get("unit_id"),
                                "asserted_by": c.get("asserted_by"), "fact_key": c.get("fact_key"),
                                "value": c.get("value"), "reason": reason})
            else:
                keep_lines.append(line)
        if not dry_run:
            f.write_text("".join(keep_lines), encoding="utf-8")
    return removed


BLOCK_HEAD = re.compile(r"^\[(?P<uid>[^\]\n]+)\] ", re.M)


def scrub_extract_input(authored, pats, dry_run):
    """extract_input/*.txt are the extractor's rendered inputs: one block per unit. Drop the person's
    blocks, redact the rest. Other files there (SCHEMA.md) are just redacted."""
    log = []
    for f in sorted((DATA / "extract_input").iterdir()):
        if f.suffix not in TEXT_SUFFIXES:
            continue
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".txt" and BLOCK_HEAD.search(text):
            heads = list(BLOCK_HEAD.finditer(text))
            out, dropped, redacted = [text[:heads[0].start()]], 0, 0
            for i, h in enumerate(heads):
                end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
                block = text[h.start():end]
                if h["uid"] in authored:
                    dropped += 1
                    continue
                if hits(block, pats):
                    redacted += 1
                    block = scrub_string(block, pats)
                out.append(block)
            new = "".join(out)
        else:
            dropped, redacted = 0, int(bool(hits(text, pats)))
            new = scrub_string(text, pats)
        if dropped or redacted:
            log.append({"file": f.name, "blocks_dropped": dropped, "blocks_redacted": redacted})
            if not dry_run:
                f.write_text(new, encoding="utf-8")
    return log


def rebuild_people_and_mentions(units, person_id):
    people = people_mod.build_registry(units, exclude={person_id})
    mentions = people_mod.build_mentions(units, people)
    return people, mentions


def rebuild_index(units, mentions):
    passages = index_mod.build_passages(units, mentions)
    emb = index_mod.embed([p["text"] for p in passages])
    return passages, emb


def rebuild_claims(units):
    """Re-run link.py's grouping and supersession over the scrubbed claims_raw. Imported late because
    link.py reads people.json at import time and it must see the rebuilt registry."""
    import importlib
    import link as link_mod
    importlib.reload(link_mod)
    umap = {u["unit_id"]: u for u in units}
    claims, dropped = link_mod.load_claims(umap)
    link_mod.group_claims(claims)
    groups = link_mod.link_and_flag(claims)
    claims = sorted(claims, key=link_mod.order_key)
    for c in claims:
        c.pop("_unit_text", None)
    return claims, groups, dropped


def annotate_head_loss(old_claims, old_groups, new_claims, new_groups, removed_ids, pats, claim_id_remap):
    """Chains that lost statements are marked, not silently re-resolved.

    For every old group that lost members now (or was already marked by an earlier deletion):
      head_removed        true when the chain's current statement was removed (now or before). The
                          newest survivor is NOT the head; the renderer must say so.
      removed_statements  running count of removed members across deletions.
    A group with no survivors is kept as a tombstone (members: [], current: null) under
    '<old id>-removed' so a question that lands on it gets "the archive no longer contains a
    statement on this" instead of silence. Tombstone fact keys are redacted like everything else.
    The newest surviving claim of a head-lost chain also carries head_removed / removed_statements.

    old_groups' members are pre-renumber claim ids; new_claims/new_groups carry the renumbered
    ones, so survivor lookups go through claim_id_remap (identity for ids that didn't move).
    Returns log rows for the deletion log."""
    new_by_id = {c["claim_id"]: c for c in new_claims}
    by_new_group = defaultdict(list)
    for c in new_claims:
        by_new_group[c["group_id"]].append(c)
    old_by_id = {c["claim_id"]: c for c in old_claims}
    log = []
    for gid, g in old_groups.items():
        removed_now = [m for m in g["members"] if m in removed_ids]
        prior_removed = int(g.get("removed_statements") or 0)
        if not removed_now and not prior_removed:
            continue
        head_lost_now = bool(g.get("current")) and g["current"] in removed_ids
        head_removed = head_lost_now or bool(g.get("head_removed"))
        removed_total = prior_removed + len(removed_now)
        survivors = [claim_id_remap.get(m, m) for m in g["members"]]
        survivors = [m for m in survivors if m in new_by_id]
        fact_keys = sorted({scrub_string(k, pats) for k in g["fact_keys"]})
        if survivors:
            ngid = new_by_id[survivors[0]]["group_id"]
            ng = new_groups[ngid]
            ng["head_removed"] = bool(ng.get("head_removed")) or head_removed
            ng["removed_statements"] = int(ng.get("removed_statements") or 0) + removed_total
            cur = [c for c in by_new_group[ngid] if c["is_current"]]
            if ng["head_removed"]:
                for c in cur:
                    c["head_removed"] = True
                    c["removed_statements"] = ng["removed_statements"]
            new_cur = ({k: cur[0].get(k) for k in ("claim_id", "date", "asserted_by", "kind", "value", "truth_status", "where")}
                       if cur else None)
            target = ngid
        else:
            target = gid if gid.endswith("-removed") else f"{gid}-removed"
            new_groups[target] = {"fact_keys": fact_keys, "size": 0, "current": None, "members": [],
                                  "head_removed": True, "removed_statements": removed_total}
            new_cur = None
        if not head_removed:
            continue   # lost a non-head member only: counted on the group, not a head-loss event
        old_cur = old_by_id.get(g.get("current") or "")
        log.append({
            "old_group": gid, "new_group": target, "fact_keys": fact_keys[:4],
            "old_current": ({k: old_cur.get(k) for k in ("claim_id", "date", "asserted_by", "value")} if old_cur else None),
            "members_removed": removed_now, "members_surviving": survivors,
            "removed_statements_total": removed_total, "head_removed": True,
            "newest_surviving": new_cur,
            "outcome": ("newest survivor is not the head: marked head_removed" if new_cur
                        else "no surviving statement: tombstone group kept, marked head_removed"),
        })
    return log


def renumber_units(units):
    """Reassign ordinals and unit ids over the surviving units so nothing about the numbering
    discloses a deletion happened: 'message 4 of 5' with the true 5th message gone, or unit id
    m05 still calling itself m05 when it now renders in position 4, are both a leak.

    E-mail/report units: renumbered per (source_file, thread) by original position_chrono, so
    thread totals and 'message N of M' describe the surviving thread only. Transcript units:
    renumbered per source_file by original turn_index, so turn numbering has no gap either.
    Ids are rebuilt in split.py's own format ('{prefix}:{stem}:m{chrono:02d}' /
    '...:t{n:03d}') by replacing just the trailing ordinal, so prefix and stem are untouched.

    Mutates unit_id and the anchor fields that encode position, in place. Returns
    {old_unit_id: new_unit_id} for every unit whose id actually changed."""
    remap = {}
    threads = defaultdict(list)
    transcripts = defaultdict(list)
    for u in units:
        if u["anchor"]["kind"] == "email":
            threads[(u["source_file"], u["anchor"]["thread"])].append(u)
        else:
            transcripts[u["source_file"]].append(u)

    for (src, _), us in threads.items():
        us.sort(key=lambda u: u["anchor"]["position_chrono"])
        n = len(us)
        name = Path(src).name
        for i, u in enumerate(us, 1):
            a = u["anchor"]
            a.update({"position_chrono": i, "position_from_top": n - i + 1, "thread_size": n, "declared_thread_size": n})
            if a.get("label"):
                a["human"] = f"{name} · {a['label']} · sent {a['sent']} by {u['speaker']}"
            else:
                a["human"] = f"{name} · message {i} of {n} (position {n - i + 1} from top) · sent {a['sent']} by {u['speaker']}"
            old_id = u["unit_id"]
            new_id = f"{old_id.rsplit(':', 1)[0]}:m{i:02d}"
            if new_id != old_id:
                remap[old_id] = new_id
                u["unit_id"] = new_id

    for src, us in transcripts.items():
        us.sort(key=lambda u: u["anchor"]["turn_index"])
        name = Path(src).name
        for i, u in enumerate(us, 1):
            a = u["anchor"]
            a["turn_index"] = i
            if a.get("offset_seconds") is not None:
                a["human"] = f"{name} · {a['speaker']} @ {a['timestamp']} (line {a['line_start']})"
            else:
                a["human"] = f"{name} · {a['speaker']} turn {i} (line {a['line_start']})"
            old_id = u["unit_id"]
            new_id = f"{old_id.rsplit(':', 1)[0]}:t{i:03d}"
            if new_id != old_id:
                remap[old_id] = new_id
                u["unit_id"] = new_id
    return remap


def remap_claims_raw(id_remap, dry_run):
    """Cascade the unit-id remap into claims_raw: rewrite unit_id, and the unit-id prefix of
    claim_id ('{unit_id}#{n}'), on every surviving claim whose unit moved. Returns
    {old_claim_id: new_claim_id} for every claim whose id changed."""
    claim_remap = {}
    for f in sorted((DATA / "claims_raw").glob("*.jsonl")):
        rows = load_jsonl(f)
        changed = False
        for c in rows:
            new_uid = id_remap.get(c.get("unit_id"))
            if not new_uid:
                continue
            old_cid, old_uid = c.get("claim_id"), c["unit_id"]
            c["unit_id"] = new_uid
            if old_cid and old_cid.startswith(old_uid + "#"):
                new_cid = new_uid + old_cid[len(old_uid):]
                c["claim_id"] = new_cid
                if new_cid != old_cid:
                    claim_remap[old_cid] = new_cid
            changed = True
        if changed and not dry_run:
            write_jsonl(f, rows)
    return claim_remap


def remap_extract_input(id_remap, dry_run):
    """Cascade the unit-id remap into extract_input/*.txt block headers ('[unit_id] ...')."""
    n = 0
    for f in sorted((DATA / "extract_input").iterdir()):
        if f.suffix != ".txt":
            continue
        text = f.read_text(encoding="utf-8")

        def sub(m):
            nonlocal n
            new = id_remap.get(m["uid"])
            if not new:
                return m.group()
            n += 1
            return f"[{new}] "

        new_text = BLOCK_HEAD.sub(sub, text)
        if new_text != text and not dry_run:
            f.write_text(new_text, encoding="utf-8")
    return n


def purge_caches(dry_run):
    gone = []
    for pattern in CACHE_GLOBS:
        for p in DATA.glob(pattern):
            gone.append(str(p.relative_to(ROOT)))
            if not dry_run:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    return gone


def verify(pats, passages_n):
    """Full-text sweep of every file under data/ except the deletion log. Returns a list of offences."""
    offences = []
    for f in sorted(DATA.rglob("*")):
        if not f.is_file() or f == LOG:
            continue
        if f.suffix in TEXT_SUFFIXES:
            text = f.read_text(encoding="utf-8", errors="replace")
            if f == DATA / "people.json":
                # another person's registry row may legitimately carry the same initials as a structural
                # field; only that field is exempt, and only from the initials pattern: names, e-mails and
                # the person id are still swept over the whole registry, and initials in prose everywhere
                rows = json.loads(text)
                sans_initials = json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "initials"} for k, v in rows.items()}, ensure_ascii=False)
                h = hits(text, [p for p in pats if p[0] != "initials"]) + hits(sans_initials, [p for p in pats if p[0] == "initials"])
            else:
                h = hits(text, pats)
            if h:
                offences.append({"file": str(f.relative_to(ROOT)), "hits": len(h), "sample": sorted({x[1] for x in h})[:5]})
        elif f.suffix == ".npy":
            n = np.load(f, mmap_mode="r").shape[0]
            if n != passages_n:
                offences.append({"file": str(f.relative_to(ROOT)), "hits": 1, "sample": [f"rows {n} != passages {passages_n}"]})
        else:
            raw = f.read_bytes()
            for label, pat in pats:
                if pat.pattern and re.search(pat.pattern.encode(), raw, re.I if pat.flags & re.I else 0):
                    offences.append({"file": str(f.relative_to(ROOT)), "hits": 1, "sample": [label]})
                    break
    return offences


def verify_structure(units, mentions, passages, claims, groups):
    """Structural acceptance checks, independent of the text sweep: every source file/thread and
    transcript is gaplessly numbered with unit ids that agree with their rendered position, and
    every claim/mention/passage/chain-member reference resolves to a unit or claim that still
    exists. Returns a list of offences in the same shape as verify()'s."""
    offences = []
    uids = {u["unit_id"] for u in units}
    cids = {c["claim_id"] for c in claims}

    threads, transcripts = defaultdict(list), defaultdict(list)
    for u in units:
        (threads if u["anchor"]["kind"] == "email" else transcripts)[u["source_file"]].append(u)
    for src, us in threads.items():
        positions = sorted(u["anchor"]["position_chrono"] for u in us)
        if positions != list(range(1, len(us) + 1)):
            offences.append({"file": src, "hits": 1, "sample": [f"position_chrono gap: {positions}"]})
        for u in us:
            want = f"m{u['anchor']['position_chrono']:02d}"
            if not u["unit_id"].endswith(":" + want):
                offences.append({"file": src, "hits": 1, "sample": [f"{u['unit_id']} disagrees with position_chrono {u['anchor']['position_chrono']}"]})
    for src, us in transcripts.items():
        positions = sorted(u["anchor"]["turn_index"] for u in us)
        if positions != list(range(1, len(us) + 1)):
            offences.append({"file": src, "hits": 1, "sample": [f"turn_index gap: {positions}"]})
        for u in us:
            want = f"t{u['anchor']['turn_index']:03d}"
            if not u["unit_id"].endswith(":" + want):
                offences.append({"file": src, "hits": 1, "sample": [f"{u['unit_id']} disagrees with turn_index {u['anchor']['turn_index']}"]})

    for c in claims:
        if c["unit_id"] not in uids:
            offences.append({"file": "claims.jsonl", "hits": 1, "sample": [f"{c['claim_id']} -> missing unit {c['unit_id']}"]})
        for k in ("superseded_by", "supersedes", "corrected_by"):
            if c.get(k) and c[k] not in cids:
                offences.append({"file": "claims.jsonl", "hits": 1, "sample": [f"{c['claim_id']}.{k} -> missing claim {c[k]}"]})
    for m in mentions:
        if m["unit_id"] not in uids:
            offences.append({"file": "mentions.jsonl", "hits": 1, "sample": [f"missing unit {m['unit_id']}"]})
    for p in passages:
        for uid in p["units"]:
            if uid not in uids:
                offences.append({"file": "passages.jsonl", "hits": 1, "sample": [f"{p['passage_id']} -> missing unit {uid}"]})
    for gid, g in groups.items():
        for m in g.get("members", []):
            if m not in cids:
                offences.append({"file": "claim_groups.json", "hits": 1, "sample": [f"{gid} member -> missing claim {m}"]})
        if g.get("current") and g["current"] not in cids:
            offences.append({"file": "claim_groups.json", "hits": 1, "sample": [f"{gid} current -> missing claim {g['current']}"]})
    return offences


# ---------------------------------------------------------------- main
def run(person_id, dry_run=False):
    t0 = time.time()
    people = json.loads((DATA / "people.json").read_text(encoding="utf-8"))
    if person_id not in people:
        sys.exit(f"{person_id} is not in data/people.json (already deleted, or never registered). "
                 f"Known: {', '.join(sorted(people))}")
    person = people[person_id]
    pats = person_patterns(person)
    units = load_jsonl(DATA / "units.jsonl")
    mentions = load_jsonl(DATA / "mentions.jsonl")
    old_claims = load_jsonl(DATA / "claims.jsonl")
    old_groups = json.loads((DATA / "claim_groups.json").read_text(encoding="utf-8"))
    old_passages_n = sum(1 for _ in (DATA / "passages.jsonl").open(encoding="utf-8"))

    # 1-3. units: delete authored, redact the rest
    authored, touched = plan_units(units, mentions, person, pats)
    kept_units, redactions = scrub_units(units, authored, touched, pats)
    print(f"units: {len(units)} -> {len(kept_units)}  (deleted {len(authored)} authored, redacted {len(redactions)} that named them)")

    # claims_raw + extract_input
    removed_claims = scrub_claims_raw(authored, pats, dry_run)
    by_reason = defaultdict(int)
    for r in removed_claims:
        by_reason[r["reason"]] += 1
    print(f"claims_raw: removed {len(removed_claims)} claims  {dict(by_reason)}")
    ei = scrub_extract_input(authored, pats, dry_run)
    print(f"extract_input: {sum(x['blocks_dropped'] for x in ei)} blocks dropped, {sum(x['blocks_redacted'] for x in ei)} redacted in {len(ei)} files")

    if dry_run:
        print("dry run: nothing written")
        return

    id_remap = renumber_units(kept_units)
    claim_id_remap = remap_claims_raw(id_remap, dry_run)
    ei_remapped = remap_extract_input(id_remap, dry_run)
    print(f"ids: {len(id_remap)} unit ids renumbered to close gaps left by the deletion   "
          f"({len(claim_id_remap)} claim ids, {ei_remapped} extract_input headers cascaded)")
    write_jsonl(DATA / "units.jsonl", kept_units)

    # 4. registry + mentions, then retrieval index with fresh embeddings
    new_people, new_mentions = rebuild_people_and_mentions(kept_units, person_id)
    (DATA / "people.json").write_text(json.dumps(new_people, ensure_ascii=False, indent=1), encoding="utf-8")
    write_jsonl(DATA / "mentions.jsonl", new_mentions)
    print(f"people: {len(people)} -> {len(new_people)}   mentions: {len(mentions)} -> {len(new_mentions)}")
    passages, emb = rebuild_index(kept_units, new_mentions)
    write_jsonl(DATA / "passages.jsonl", passages)
    np.save(DATA / "embeddings.npy", emb)
    print(f"passages: {old_passages_n} -> {len(passages)}   embeddings re-encoded: {emb.shape}")

    # 5. derived claim graph
    new_claims, new_groups, dropped = rebuild_claims(kept_units)
    removed_ids = {r["claim_id"] for r in removed_claims}
    # claim ids can gain a trailing ' on de-dup; match on the raw id too
    removed_ids |= {c["claim_id"] for c in old_claims if c["claim_id"].rstrip("'") in removed_ids or c["unit_id"] in authored}
    rechains = annotate_head_loss(old_claims, old_groups, new_claims, new_groups, removed_ids, pats, claim_id_remap)
    write_jsonl(DATA / "claims.jsonl", new_claims)
    (DATA / "claim_groups.json").write_text(json.dumps(new_groups, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"claims: {len(old_claims)} -> {len(new_claims)}   groups: {len(old_groups)} -> {len(new_groups)} (incl. tombstones)   "
          f"chains that lost their head: {len(rechains)} "
          f"({sum(1 for r in rechains if r['newest_surviving'])} with a newest survivor, "
          f"{sum(1 for r in rechains if not r['newest_surviving'])} empty)")
    caches = purge_caches(dry_run)
    if caches:
        print("caches removed:", ", ".join(caches))

    # 6. sweep: text (name survives nowhere) + structure (ids gapless, no dangling reference)
    text_offences = verify(pats, len(passages))
    struct_offences = verify_structure(kept_units, new_mentions, passages, new_claims, new_groups)
    offences = text_offences + struct_offences
    record = {
        "deleted_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "person_id": person_id, "name": person["name"],
        "forms_erased": {"aliases": person["aliases"], "emails": person["emails"], "initials": person.get("initials")},
        "units_deleted": sorted(authored), "units_redacted": redactions,
        "claims_raw_removed": removed_claims, "extract_input": ei,
        "people_rows": [len(people), len(new_people)], "mentions_rows": [len(mentions), len(new_mentions)],
        "passages": [old_passages_n, len(passages)], "claims": [len(old_claims), len(new_claims)],
        "groups": [len(old_groups), len(new_groups)], "link_dropped": dropped,
        "chains_head_removed": rechains, "id_remap": {"units": id_remap, "claims": claim_id_remap},
        "caches_removed": caches,
        "verify": {"clean": not offences, "offences": text_offences, "structure_offences": struct_offences},
        "seconds": round(time.time() - t0, 1),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if offences:
        print("VERIFY FAILED:", file=sys.stderr)
        for o in offences:
            print(f"  {o['file']}: {o['hits']} hits {o['sample']}", file=sys.stderr)
        sys.exit(2)
    print(f"verify: clean - no form of {person['name']!r} and no id gap or dangling reference in any file "
          f"under data/ except {LOG.name}   ({record['seconds']}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("person_id", help="canonical id from data/people.json, e.g. person:<slug>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.person_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
