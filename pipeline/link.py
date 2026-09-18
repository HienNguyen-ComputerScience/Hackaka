"""Supersession + truth-status model over extracted claims.

Reads  data/claims_raw/*.jsonl (extractor output, schema in data/extract_input/SCHEMA.md)
       data/units.jsonl
Writes data/claims.jsonl        every claim, enriched with:
         date, source_file, where (human anchor), has_time
         group_id              claims about the same underlying question (fact_key)
         chain_pos             position in the group's date-ordered chain of statements
         superseded_by         claim_id of the next later statement of the same fact, or null
         supersedes            claim_id of the previous statement, or null
         order_confidence      'dated' | 'same_day_unknown' | 'same_transcript_turn_order'
         is_current            true for the newest survivor in its group
         truth_status          'true_as_stated' | 'never_true' | 'unverified'
         truth_reason          why
         corrected_by          claim_id of the correction, when never_true
       data/claim_groups.json  group_id -> fact keys, members, current claim

The two things are independent: a claim can be current and unverified, or superseded and
true_as_stated. Corrected claims are never_true, not merely old.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHAIN_KINDS = {"fact", "correction", "decision", "recollection", "agreement", "withdrawal", "action"}
SIM_THRESHOLD = 0.85
CHANGE_WORDS = re.compile(r"\b(reduc\w*|increas\w*|chang\w*|updat\w*|revis\w*|no longer|since then|agreed to (cut|reduce|drop))\b", re.I)
WRONG_WORDS = re.compile(r"\b(was wrong|were wrong|incorrect|not correct|mistake\w*|misstat\w*|withdr\w*|in error|not true|never (was|happened))\b", re.I)

NUMWORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
NUMWORDS.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
                 "hundred": 100, "thousand": 1000, "million": 1_000_000, "half": 0.5, "quarter": 0.25})


def numeric(value):
    """Best-effort number from a value string: '48%', 'forty-eight percent', '25 stores', '1,317'."""
    if value is None:
        return None
    s = str(value).lower().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        return float(m.group())
    words = re.findall(r"[a-z]+", s.replace("-", " "))
    total, cur, seen = 0, 0, False
    for w in words:
        if w in NUMWORDS:
            n = NUMWORDS[w]
            seen = True
            if n in (100, 1000, 1_000_000):
                cur = max(cur, 1) * n
                if n >= 1000:
                    total += cur
                    cur = 0
            else:
                cur += n
    return (total + cur) if seen else None


def norm_key(k):
    return re.sub(r"\s+", " ", (k or "").lower().strip().rstrip("."))


def load_claims(units):
    claims, dropped = [], []
    for f in sorted((DATA / "claims_raw").glob("*.jsonl")):
        for n, line in enumerate(f.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                dropped.append(f"{f.name}:{n} bad json {e}")
                continue
            u = units.get(c.get("unit_id"))
            if not u:
                dropped.append(f"{f.name}:{n} unknown unit_id {c.get('unit_id')!r}")
                continue
            if not c.get("statement"):
                dropped.append(f"{f.name}:{n} no statement")
                continue
            if c.get("truncated"):
                c["value"] = None
            c.setdefault("claim_id", f"{c['unit_id']}#{n}")
            c["date"] = u["date"]
            c["has_time"] = "T" in u["date"]
            c["source_file"] = u["source_file"]
            c["where"] = u["anchor"]["human"]
            c["speaker"] = u["speaker"]
            c["turn_index"] = u["anchor"].get("turn_index")
            c["_unit_text"] = u["text"]
            c["fact_key_norm"] = norm_key(c.get("fact_key"))
            claims.append(c)
    # de-duplicate claim ids (extractors number per unit; collisions are possible across files)
    seen = set()
    for c in claims:
        while c["claim_id"] in seen:
            c["claim_id"] += "'"
        seen.add(c["claim_id"])
    return claims, dropped


def group_claims(claims):
    """Union-find over fact keys: exact match or embedding cosine >= SIM_THRESHOLD."""
    keys = sorted({c["fact_key_norm"] for c in claims if c["fact_key_norm"]})
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(keys, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sims = emb @ emb.T
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if sims[i, j] >= SIM_THRESHOLD:
                parent[find(i)] = find(j)
    key_group = {k: f"g{find(i):04d}" for i, k in enumerate(keys)}
    for c in claims:
        c["group_id"] = key_group.get(c["fact_key_norm"], "g-none-" + c["claim_id"])
    return key_group


def order_key(c):
    return (c["date"][:10], c["date"][11:] if c["has_time"] else "", c["source_file"], c["turn_index"] or 0)


def order_confidence(a, b):
    if a["date"][:10] != b["date"][:10]:
        return "dated"
    if a["source_file"] == b["source_file"] and a["turn_index"] is not None and b["turn_index"] is not None:
        return "same_transcript_turn_order"
    if a["has_time"] and b["has_time"]:
        return "dated"
    return "same_day_unknown"


PEOPLE = [p["name"] for p in json.loads((DATA / "people.json").read_text(encoding="utf-8")).values()
          if not p["person_id"].startswith("unresolved:")]


def link_and_flag(claims):
    by_group = defaultdict(list)
    for c in claims:
        by_group[c["group_id"]].append(c)
    for gid, members in by_group.items():
        chain = sorted([c for c in members if c.get("kind") in CHAIN_KINDS], key=order_key)
        # one utterance split by an interjection: same speaker, same transcript, within 3 turns
        # -> one chain node (the earlier turn keeps the value; the later statement is appended)
        merged = []
        for c in chain:
            c.setdefault("merged_claims", [])
            prev = merged[-1] if merged else None
            if (prev and prev["source_file"] == c["source_file"] and prev["asserted_by"] == c["asserted_by"]
                    and prev["turn_index"] is not None and c["turn_index"] is not None
                    and 0 < c["turn_index"] - prev["turn_index"] <= 3):
                prev["merged_claims"].append(c["claim_id"])
                prev["statement"] = prev["statement"].rstrip(".") + "; " + c["statement"][0].lower() + c["statement"][1:]
                if prev.get("value") is None and not prev.get("truncated"):
                    prev["value"] = c.get("value")
                c["merged_into"] = prev["claim_id"]
                continue
            merged.append(c)
        chain = merged
        for c in members:
            c.update({"chain_pos": None, "superseded_by": None, "supersedes": None,
                      "order_confidence": None, "is_current": False})
        for i, c in enumerate(chain):
            c["chain_pos"] = i + 1
            if i + 1 < len(chain):
                nxt = chain[i + 1]
                c["superseded_by"] = nxt["claim_id"]
                nxt["supersedes"] = c["claim_id"]
                c["order_confidence"] = order_confidence(c, nxt)
        if chain:
            chain[-1]["is_current"] = True
        # truth status
        for c in members:
            reasons = []
            if c.get("truncated"):
                reasons.append("sentence stops before the value")
            if c.get("hedged"):
                reasons.append("hedged wording")
            if c.get("secondhand"):
                reasons.append("second-hand or recalled")
            if c.get("kind") in ("proposal", "question", "recollection"):
                reasons.append(f"kind={c['kind']}")
            c["truth_status"] = "unverified" if reasons else "true_as_stated"
            c["truth_reason"] = "; ".join(reasons) if reasons else "stated as fact by a first-hand source; no correction found"
            c["corrected_by"] = None
        for corr in [c for c in members if c.get("kind") in ("correction", "withdrawal")]:
            # a "correction" that reports the fact CHANGED ("Lena agreed to reduce it") is supersession, not falsity
            ctx = f"{corr.get('statement','')} {corr.get('corrects') or ''} {corr.get('_unit_text','')}"
            if CHANGE_WORDS.search(ctx) and not WRONG_WORDS.search(ctx):
                continue
            cv = numeric(corr.get("value")) if corr.get("kind") == "correction" else None   # a withdrawal carries the withdrawn value, not the right one
            earlier = [c for c in members if c is not corr and order_key(c) < order_key(corr)
                       and c.get("kind") not in ("correction", "withdrawal", "question")]
            # stale vs never-true: a statement made BEFORE the fact last changed to the corrected value was
            # true at the time. Only statements made after the corrected value was already on record are hit.
            t0 = None
            if cv is not None:
                agreeing = [c for c in earlier if numeric(c.get("value")) == cv]
                if agreeing:
                    t0 = order_key(min(agreeing, key=order_key))
            # who is being corrected: a person named in `corrects`, else the correcting speaker themself
            target = corr.get("asserted_by")
            if corr.get("kind") == "correction":      # a withdrawal is always of one's own statement
                for name in PEOPLE:
                    if name.lower() in str(corr.get("corrects") or "").lower():
                        target = name
                        break
            for c in earlier:
                v = numeric(c.get("value"))
                if cv is not None and v is not None:
                    hit = v != cv and (t0 is None or order_key(c) >= t0)
                else:
                    hit = c.get("asserted_by") == target
                if hit:
                    c["truth_status"] = "never_true"
                    c["truth_reason"] = f"corrected by {corr['asserted_by']} on {corr['date'][:10]}: {corr['statement'][:120]}"
                    c["corrected_by"] = corr["claim_id"]
    groups = {}
    for gid, members in by_group.items():
        cur = [c for c in members if c["is_current"]]
        groups[gid] = {
            "fact_keys": sorted({c.get("fact_key") for c in members if c.get("fact_key")}),
            "size": len(members),
            "current": cur[0]["claim_id"] if cur else None,
            "members": [c["claim_id"] for c in sorted(members, key=order_key)],
        }
    return groups


def main():
    units = {u["unit_id"]: u for u in map(json.loads, (DATA / "units.jsonl").open(encoding="utf-8"))}
    claims, dropped = load_claims(units)
    group_claims(claims)
    groups = link_and_flag(claims)
    with (DATA / "claims.jsonl").open("w", encoding="utf-8") as f:
        for c in sorted(claims, key=order_key):
            c.pop("_unit_text", None)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (DATA / "claim_groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
    st = defaultdict(int)
    for c in claims:
        st[c["truth_status"]] += 1
    oc = defaultdict(int)
    for c in claims:
        if c["order_confidence"]:
            oc[c["order_confidence"]] += 1
    multi = [g for g in groups.values() if g["size"] > 1]
    print(f"claims: {len(claims)}  dropped: {len(dropped)}  groups: {len(groups)}  multi-claim groups: {len(multi)}")
    print("truth status:", dict(st))
    print("link order confidence:", dict(oc))
    big = sorted(groups.items(), key=lambda kv: -kv[1]["size"])[:8]
    for gid, g in big:
        print(f"  {gid} size={g['size']} keys={g['fact_keys'][:4]}")
    for d in dropped[:15]:
        print("  dropped:", d)


if __name__ == "__main__":
    main()
