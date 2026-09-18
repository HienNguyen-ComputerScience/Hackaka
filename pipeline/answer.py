"""Deterministic answer renderer over the claim graph. No LLM at query time.

    python pipeline/answer.py "What proportion of articles had shelf-life data populated?"

Steps: retrieve windows -> claims on the retrieved units -> expand each touched group to its
full chain -> render every statement with date, source, supersession label and truth status.
Attribution questions list proposing turns and agreeing turns separately, each cited.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import Retriever  # noqa: E402
from link import numeric  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ATTRIB_WORDS = re.compile(r"\b(who|propos|suggest|agree|accept|sign|commit|decid|approv|confirm)", re.I)


class Answerer:
    def __init__(self):
        self.r = Retriever()
        self.claims = [json.loads(l) for l in (DATA / "claims.jsonl").open(encoding="utf-8")]
        self.by_id = {c["claim_id"]: c for c in self.claims}
        self.by_unit = defaultdict(list)
        self.by_group = defaultdict(list)
        for c in self.claims:
            self.by_unit[c["unit_id"]].append(c)
            self.by_group[c["group_id"]].append(c)
        self.groups = json.loads((DATA / "claim_groups.json").read_text(encoding="utf-8"))
        # direct channel: query -> fact keys (so a chain is reachable even when no window is retrieved)
        self.gids = [g for g in self.groups if self.groups[g]["fact_keys"]]
        self.gkeys = ["; ".join(self.groups[g]["fact_keys"][:4]) for g in self.gids]
        import numpy as np
        self.gemb = self.r._embed_batch(self.gkeys) if hasattr(self.r, "_embed_batch") else None

    def answer(self, query, k=8, max_groups=4):
        hits = self.r.search(query, k=k)
        retrieved_units = []
        for h in hits:
            for u in h["units"]:
                if u["unit_id"] not in retrieved_units:
                    retrieved_units.append(u["unit_id"])
        # score groups by how many retrieved claims they contain, weighted by hit rank
        gscore = defaultdict(float)
        for rank, h in enumerate(hits):
            for u in h["units"]:
                for c in self.by_unit[u["unit_id"]]:
                    gscore[c["group_id"]] += 1.0 / (1 + rank)
        # fuse with fact-key similarity: a group whose key matches the question outranks window co-occurrence
        qv = self.r._embed_query(query)
        sims = self.gemb @ qv
        for i in sims.argsort()[::-1][:5]:
            if sims[i] >= 0.45:
                gscore[self.gids[i]] += 2.0 * float(sims[i])
        top_groups = [g for g, _ in sorted(gscore.items(), key=lambda kv: -kv[1])[:max_groups]]
        out = {"query": query, "groups": [], "attribution": None, "hits": [h["pin"]["where"] for h in hits]}
        for gid in top_groups:
            members = sorted(self.by_group[gid], key=lambda c: (c["date"], c["turn_index"] or 0))
            chain = [c for c in members if c["chain_pos"]]
            g = self.groups[gid]
            out["groups"].append({
                "group_id": gid, "fact_keys": g["fact_keys"],
                "current": None if g.get("head_removed") else g["current"],
                "head_removed": bool(g.get("head_removed")), "removed_statements": int(g.get("removed_statements") or 0),
                "newest_surviving_figure": self.newest_figure(chain) if g.get("head_removed") else None,
                "statements": [self.fmt(c) for c in chain],
                "other": [self.fmt(c) for c in members if not c["chain_pos"]],
            })
        if ATTRIB_WORDS.search(query):
            out["attribution"] = self.attribution(top_groups, retrieved_units)
        return out

    def newest_figure(self, chain):
        """Newest surviving statement that carries a number and was not corrected; else the newest survivor.
        Used only when the chain lost its head: it is the best the archive still has, not the answer."""
        # a figure is an explicit number ("48%", "1,317"), not a word like "half", and not a recollection
        figs = [c for c in chain if c.get("value") and re.search(r"\d", str(c["value"])) and numeric(c["value"]) is not None
                and c["truth_status"] != "never_true" and c["kind"] != "recollection"]
        pick = (figs or chain or [None])[-1]
        return self.fmt(pick) if pick else None

    def fmt(self, c):
        head_removed = self.groups.get(c["group_id"], {}).get("head_removed")
        label = ("NEWEST SURVIVING" if head_removed else "CURRENT") if c["is_current"] else ("SUPERSEDED" if c["superseded_by"] else "")
        sup = None
        if c["superseded_by"]:
            n = self.by_id[c["superseded_by"]]
            sup = f"superseded by {n['asserted_by']} on {n['date'][:10]} ({n['where']})"
            if c["order_confidence"] == "same_day_unknown":
                sup += "  [order_confidence: same_day_unknown - same day, no clock time on one side]"
        corr = None
        if c["corrected_by"]:
            x = self.by_id[c["corrected_by"]]
            corr = f"corrected by {x['asserted_by']} on {x['date'][:10]}: \"{x['statement']}\" ({x['where']})"
        return {
            "claim_id": c["claim_id"], "date": c["date"], "asserted_by": c["asserted_by"], "kind": c["kind"],
            "value": c.get("value"), "truncated": bool(c.get("truncated")), "statement": c["statement"],
            "currency": label, "supersession": sup, "truth_status": c["truth_status"], "truth_reason": c["truth_reason"],
            "correction": corr, "order_confidence": c["order_confidence"], "cite": c["where"],
        }

    def attribution(self, gids, retrieved_units):
        pool = [c for g in gids[:2] for c in self.by_group[g]]   # top two groups only: keeps neighbouring turns' unrelated proposals out
        seen, uniq = set(), []
        for c in pool:
            if c["claim_id"] not in seen:
                seen.add(c["claim_id"])
                uniq.append(c)
        uniq.sort(key=lambda c: (c["date"], c["turn_index"] or 0))
        props = [self.fmt(c) | {"responds_to": c.get("responds_to")} for c in uniq if c["kind"] == "proposal"]
        agrees = [self.fmt(c) | {"responds_to": c.get("responds_to")} for c in uniq if c["kind"] in ("agreement", "decision")]
        rejects = [self.fmt(c) | {"responds_to": c.get("responds_to")} for c in uniq if c["kind"] == "rejection"]
        return {
            "proposed": props, "agreed": agrees, "rejected": rejects,
            "verdict": ("nobody agreed: no agreement or decision claim found in the retrieved evidence"
                        if props and not agrees else
                        ("no proposal found in the retrieved evidence" if not props else "see agreed list")),
        }


def render(out):
    lines = [f"Q: {out['query']}", ""]
    for g in out["groups"]:
        lines.append(f"== {', '.join(g['fact_keys'][:3])} ==")
        if g.get("head_removed"):
            n = g["removed_statements"]
            if g["statements"]:
                f = g["newest_surviving_figure"]
                val = f"{f['asserted_by']}'s {f['value']!r}" if f["value"] is not None else f"{f['asserted_by']}'s statement"
                lines.append(f"  HEAD REMOVED: the most recent statement in this chain was removed ({n} statement{'s' if n != 1 else ''} removed by deletion). "
                             f"The newest surviving figure is {val} from {f['date'][:10]} ({f['cite']}), and it may have been "
                             f"superseded by a later statement that no longer exists. No statement below is current.")
            else:
                lines.append(f"  REMOVED: the archive no longer contains a statement on this. {n} statement{'s' if n != 1 else ''} "
                             f"existed and {'were' if n != 1 else 'was'} removed by deletion; none survive.")
        for s in g["statements"]:
            flag = f"[{s['currency']}]" if s["currency"] else "[unlinked]"
            val = f" value={s['value']!r}" if s["value"] is not None else (" value=TRUNCATED IN SOURCE" if s["truncated"] else "")
            lines.append(f"  {s['date'][:16]}  {s['asserted_by']:16s} {flag:12s} {s['truth_status']:15s}{val}")
            lines.append(f"      \"{s['statement']}\"")
            lines.append(f"      cite: {s['cite']}")
            if s["supersession"]:
                lines.append(f"      {s['supersession']}")
            if s["correction"]:
                lines.append(f"      NEVER TRUE: {s['correction']}")
            elif s["truth_status"] == "unverified":
                lines.append(f"      unverified: {s['truth_reason']}")
        if g["other"]:
            lines.append("  related (proposals/questions, not part of the chain):")
            for s in g["other"]:
                lines.append(f"    {s['date'][:10]} {s['asserted_by']} [{s['kind']}] \"{s['statement']}\"  cite: {s['cite']}")
        lines.append("")
    a = out["attribution"]
    if a:
        lines.append("== attribution ==")
        for name, items in (("PROPOSED", a["proposed"]), ("AGREED / DECIDED", a["agreed"]), ("REJECTED", a["rejected"])):
            lines.append(f"  {name}:")
            for s in items:
                lines.append(f"    {s['date'][:16]} {s['asserted_by']}: \"{s['statement']}\"")
                if s.get("responds_to"):
                    lines.append(f"        in response to: {s['responds_to']}")
                lines.append(f"        cite: {s['cite']}")
            if not items:
                lines.append("    (none)")
        lines.append(f"  verdict: {a['verdict']}")
        lines.append("")
    lines.append("retrieved windows: " + "; ".join(out["hits"][:5]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = Answerer().answer(args.query, k=args.k)
    print(json.dumps(out, ensure_ascii=False, indent=1) if args.json else render(out))


if __name__ == "__main__":
    main()
