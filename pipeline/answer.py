"""Deterministic answer renderer over the claim graph. No LLM at query time.

    python pipeline/answer.py "What proportion of articles had shelf-life data populated?"

Steps: retrieve windows -> claims on the retrieved units -> expand each touched group to its
full chain -> render every statement with date, source, supersession label and truth status.
Attribution questions list proposing turns and agreeing turns separately, each cited.

Group selection has three bounded channels (window co-occurrence, fact-key embedding similarity,
fact-key word overlap weighted by rarity) and a relevance gate: a chain is shown only when its fact
keys relate to the question, so a question the archive does not cover yields no chains and the
renderer says so. Two question shapes get extra, still deterministic, handling:
  * a date scope ("September 2024", "Q1 2026", "2025") restricts retrieval to that period, favours
    chains with statements in it, and, when the question asks for figures, favours chains that
    carry a number then;
  * an "agreed and never done" question runs the trail detector: chains where a commitment is
    followed by a later statement that it is outstanding, with no later statement that it was done.
Statements taken from status reports are flagged: they are evidence of what was reported, not of
the underlying state.
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import Retriever  # noqa: E402
from link import numeric  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOTHING = "The archive does not contain a statement on this."
RETRACTING = ("withdrawal", "correction")
ATTRIB_WORDS = re.compile(r"\b(who|propos|suggest|agree|accept|sign|commit|decid|approv|confirm)", re.I)
INITIATIVE_WORDS = re.compile(r"\b(never (been )?(done|delivered|happened|finished|completed|received)|not done|"
                              r"agreed (and|but|then)|still (open|outstanding|waiting)|outstanding|"
                              r"follow(ed|s)?[- ]?(up|through)|forgotten|slipped|fell through|dropped the ball)\b", re.I)
UNDONE = re.compile(r"\b(not (been |yet )?(received|done|delivered|sent|finished|happened|provided|set up|written|"
                    r"resolved|closed|answered|come back|completed)|never (received|done|delivered|arrived|happened|"
                    r"finished|got|came)|still (open|not|outstanding|waiting|missing|pending)|had not|has not|"
                    r"hasn't|hadn't|no (reply|answer|response)|not yet|remains? (open|outstanding)|chased|chasing)\b", re.I)
DONE = re.compile(r"\b(done|delivered|received|sent|completed|finished|removed|resolved|closed|fixed|answered)\b", re.I)
NUMERIC_INTENT = re.compile(r"\b(figure|number|how many|how much|proportion|percent|percentage|share|count|total|rate"
                            r"|charge|cost|price|fee|invoice|pay|paid)s?\b", re.I)
EVERY = re.compile(r"\b(every|all|each|list)\b", re.I)
TURN_INDEX = re.compile(r" turn \d+ \(line")   # internal-transcript anchors carry a turn number; never rendered
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
STOP = set("a an the and or of for in on at to by with from as is are was were be been did do does had has have it its this that "
           "these those what which who whom when where why how give every each any some say said tell name show quote there "
           "then than into out up not no yes one thing find archive".split())


def content_map(s):
    """stem -> the word as written, for content words (a crude plural/tense strip, stop words dropped)."""
    out = {}
    for w in re.findall(r"[a-z0-9]+", s.lower()):
        t = w
        if t in STOP or len(t) < 3:
            continue
        for suf in ("ings", "ing", "ies", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.setdefault(t, w)
    return out


def content_tokens(s):
    """Lower-cased content words with a crude plural/tense strip, for fact-key overlap scoring."""
    return set(content_map(s))


def anchor(where):
    return TURN_INDEX.sub(" (line", where or "")


def date_scope(query):
    """(date_from, date_to, label) when the question names a month, quarter or year; else None."""
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(20\d\d)\b", query, re.I)
    if m:
        y, mo = int(m[2]), MONTHS[m[1][:3].lower()]
        return f"{y}-{mo:02d}-01", f"{y}-{mo:02d}-31", m[0]
    m = re.search(r"\bQ([1-4])\s+(20\d\d)\b", query)
    if m:
        q, y = int(m[1]), int(m[2])
        return f"{y}-{3 * q - 2:02d}-01", f"{y}-{3 * q:02d}-31", m[0]
    m = re.search(r"\b(20\d\d)\b", query)
    if m:
        return f"{m[1]}-01-01", f"{m[1]}-12-31", m[0]
    return None


DATE_TOKENS = re.compile(
    r"\b\d{4}-\d\d-\d\d(?:t\d\d:\d\d)?\b"                                    # ISO date, optional clock
    r"|\b(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d\d\b"   # 5 March 2026 / March 2026
    r"|\bweek\s+\d{1,2}\b"                                                     # Week 48
    r"|\bq[1-4]\s+(?:19|20)\d\d\b"                                             # Q1 2026
    r"|\b(?:19|20)\d\d\b",                                                     # bare year
    re.I)


def carries_figure(value):
    """A value holds a figure once its date tokens are stripped; a value that is only a date does not."""
    if value is None:
        return False
    return numeric(DATE_TOKENS.sub(" ", str(value))) is not None


def has_number(value):
    return value is not None and re.search(r"\d", str(value)) is not None and numeric(value) is not None


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
        self.gtok = [content_tokens(k) for k in self.gkeys]
        df = defaultdict(int)
        for toks in self.gtok:
            for t in toks:
                df[t] += 1
        n = max(1, len(self.gtok))
        self.idf = {t: math.log(1 + n / d) for t, d in df.items()}
        self.gemb = self.r._embed_batch(self.gkeys)
        # statement embeddings: a chain is only shown if one of its statements actually addresses the question
        self.semb = self.r._embed_batch([c["statement"] for c in self.claims])
        self.sidx = {c["claim_id"]: i for i, c in enumerate(self.claims)}
        # retractions whose target link.py resolved to a claim in the archive (the target carries corrected_by)
        self.resolved_retractions = {c["corrected_by"] for c in self.claims if c.get("corrected_by")}
        # every content word the archive uses anywhere: texts, subjects, meeting names, names in headers.
        # A question word outside this set is a concept the archive never mentions.
        self.vocab = set().union(*self.gtok) if self.gtok else set()
        for u in self.r.units.values():
            names = [u.get("speaker") or ""] + [a.get("name", "") for a in u.get("attendees", [])]
            rec = u.get("recipients")
            if isinstance(rec, dict):
                for side in ("to", "cc"):
                    names += [r.get("name", "") for r in rec.get(side, [])]
            self.vocab |= content_tokens(" ".join([u["text"], u.get("subject") or "", u.get("meeting") or "", u.get("signature") or ""] + names))
        self._chain_tokens = {}

    def statement_cos(self, gid, qv):
        rows = [self.sidx[c["claim_id"]] for c in self.by_group[gid]]
        return float((self.semb[rows] @ qv).max()) if rows else 0.0

    def chain_tokens(self, gid):
        """Content words of everything a chain says: fact keys, statements, values."""
        if gid not in self._chain_tokens:
            text = " ".join([" ".join(self.groups[gid]["fact_keys"])] + [f"{c['statement']} {c.get('value') or ''}" for c in self.by_group[gid]])
            self._chain_tokens[gid] = content_tokens(text)
        return self._chain_tokens[gid]

    # ------------------------------------------------------------ selection
    def answer(self, query, k=8, max_groups=4):
        scope = date_scope(query)
        numeric_intent = bool(NUMERIC_INTENT.search(query))
        hits = self.r.search(query, k=k)
        # inside a named period there are few passages, so look at most of them
        scoped_hits = self.r.search(query, k=max(k, 20), date_from=scope[0], date_to=scope[1]) if scope else []
        if scope and numeric_intent and EVERY.search(query):
            max_groups = max(max_groups, 12)   # "every figure" in a period: show every numeric chain of that period
        retrieved_units = []
        for h in hits + scoped_hits:
            for u in h["units"]:
                if u["unit_id"] not in retrieved_units:
                    retrieved_units.append(u["unit_id"])
        # (a) window co-occurrence: a group counts once per hit, weighted by hit rank
        gscore, co, co_scoped, sem, lex = defaultdict(float), defaultdict(float), defaultdict(float), {}, {}
        for rank, h in enumerate(hits):
            for gid in {c["group_id"] for u in h["units"] for c in self.by_unit[u["unit_id"]]}:
                gscore[gid] += 1.0 / (1 + rank)
                co[gid] += 1.0 / (1 + rank)
        for rank, h in enumerate(scoped_hits):
            for gid in {c["group_id"] for u in h["units"] for c in self.by_unit[u["unit_id"]]}:
                gscore[gid] += 1.0 / (1 + rank)
                co_scoped[gid] += 1.0 / (1 + rank)
        # (b) fact-key embedding similarity
        qv = self.r._embed_query(query)
        sims = self.gemb @ qv
        for i in sims.argsort()[::-1][:8]:
            if sims[i] >= 0.40:
                gscore[self.gids[i]] += 2.0 * float(sims[i])
                sem[self.gids[i]] = float(sims[i])
        # (c) fact-key word overlap, weighted by how rare the word is across fact keys
        qtok = content_tokens(query)
        if qtok:
            qw = sum(self.idf.get(t, 1.0) for t in qtok)
            for gid, toks in zip(self.gids, self.gtok):
                ov = qtok & toks
                if ov:
                    gscore[gid] += 2.0 * sum(self.idf.get(t, 1.0) for t in ov) / qw
                    lex[gid] = len(ov)
        # (d) date scope: chains with a statement in the period, especially a numeric one when figures are asked for
        in_scope, num_in_scope = {}, {}
        if scope:
            for gid in set(gscore) | set(self.gids):
                ms = [c for c in self.by_group[gid] if scope[0] <= c["date"][:10] <= scope[1]]
                in_scope[gid] = bool(ms)
                num_in_scope[gid] = any(has_number(c.get("value")) and not c.get("truncated") for c in ms)
            for gid in list(gscore):
                if in_scope.get(gid) and co_scoped.get(gid, 0) > 0:
                    gscore[gid] += 1.0
                    if numeric_intent and num_in_scope.get(gid):
                        gscore[gid] += 2.0
                elif not in_scope.get(gid):
                    gscore[gid] *= 0.5

        # relevance gate: a chain is shown only if its fact keys relate to the question (by meaning or by
        # words), or, under a date scope, if it has a statement in that period inside a retrieved window.
        def related(gid):
            return (gid in sem or lex.get(gid, 0) >= 2 or (lex.get(gid, 0) >= 1 and co.get(gid, 0) > 0)
                    or (scope is not None and in_scope.get(gid) and co_scoped.get(gid, 0) > 0))

        top_groups = [g for g, _ in sorted(gscore.items(), key=lambda kv: (-round(kv[1], 6), kv[0])) if related(g)][:max_groups]
        # answer rule: a chain on the same topic is not an answer. Outside a date scope, each chain must either
        # share two content words with the question or contain a statement semantically close to it
        # (cosine >= 0.45). If nothing survives, the answer is "not contained" and the nearest topics are named.
        nearest = None
        if not scope and top_groups:
            keep = [g for g in top_groups if lex.get(g, 0) >= 2 or self.statement_cos(g, qv) >= 0.45]
            if not keep:
                nearest = {"topics": [self.groups[g]["fact_keys"][:2] for g in top_groups[:3]]}
            top_groups = keep
        # words the archive never uses: literally true, and stated. When they make up a third or more of the
        # question and the chosen chains do not cover the rest of it either, nothing here answers it.
        qmap = content_map(query)
        never_stems = {t for t in qtok if t not in self.vocab}
        never = sorted(qmap[t] for t in never_stems)
        if never and not scope and top_groups and 3 * len(never) >= len(qtok):
            rest = qtok - never_stems
            covered = rest & set().union(*(self.chain_tokens(g) for g in top_groups))
            if covered != rest:
                nearest = {"topics": [self.groups[g]["fact_keys"][:2] for g in top_groups[:3]]}
                top_groups = []
        # direct-answer test: does some single statement in the chosen chains address the question as asked?
        # A statement qualifies when it is semantically close to the question, or covers half its content
        # words, or covers a third of them while being moderately close. If no statement qualifies, the
        # chains are labelled related context, not an answer. When the question uses a word the archive
        # never uses, word coverage cannot qualify a statement (the remaining words are the easy ones to
        # cover; "charge" struck from "what did X charge for Y" leaves X and Y); only closeness can, and
        # never-words that are the bulk of the ask veto even that.
        direct = None
        if top_groups and not scope:
            direct = False
            rest = qtok - never_stems
            # a question asking for a figure is not answered by chains that hold none, however close their wording
            no_figure = numeric_intent and not any(carries_figure(c.get("value")) for g in top_groups for c in self.by_group[g])
            if 3 * len(never) < len(qtok) and not no_figure:
                for gid in top_groups:
                    for c in self.by_group[gid]:
                        cos = float(self.semb[self.sidx[c["claim_id"]]] @ qv)
                        cov = len(rest & content_tokens(f"{c['statement']} {c.get('value') or ''}")) / len(rest) if rest else 0.0
                        if cos >= 0.48 or (not never and (cov >= 0.5 or (cov >= 0.33 and cos >= 0.40))):
                            direct = True
                            break
                    if direct:
                        break
        if direct is False:
            # related context must be about the question's named subject: a chain admitted only through common
            # words (bakery, workstream) whose fact keys never name the proper noun asked about is off-subject
            proper = set(content_map(" ".join(w for w in query.split() if w[:1].isupper()))) & self.vocab
            if proper:
                on_subject = [g for g in top_groups if proper & content_tokens(" ".join(self.groups[g]["fact_keys"]))]
                if on_subject:
                    top_groups = on_subject
        attrib = bool(ATTRIB_WORDS.search(query))
        if attrib:
            # a proposal that restates the question anchors the attribution block even when its chain was not
            # selected; the chain is rendered too, so the answer is visible above the block, not only inside it
            ssims = self.semb @ qv
            restating = [self.claims[i]["group_id"] for i in ssims.argsort()[::-1][:5]
                         if ssims[i] >= 0.75 and self.claims[i]["kind"] == "proposal"]
            top_groups = top_groups + [g for g in dict.fromkeys(restating) if g not in top_groups]
        out = {"query": query, "scope": scope[2] if scope else None, "never_mentions": never, "direct": direct, "groups": [],
               "attribution": None, "initiative": None, "nearest": nearest, "hits": [h["pin"]["where"] for h in hits]}
        for gid in top_groups:
            members = sorted(self.by_group[gid], key=lambda c: (c["date"], c["turn_index"] or 0))
            chain = [c for c in members if c["chain_pos"]]
            g = self.groups[gid]
            head = None if g.get("head_removed") else self.head_of(chain)
            out["groups"].append({
                "group_id": gid, "fact_keys": g["fact_keys"],
                "current": None if g.get("head_removed") else (head["claim_id"] if head else g["current"]),
                "head_removed": bool(g.get("head_removed")), "removed_statements": int(g.get("removed_statements") or 0),
                "newest_surviving_figure": self.newest_figure(chain) if g.get("head_removed") else None,
                "statements": [self.fmt(c, head) for c in self.chain_order(chain, head)],
                "other": [self.fmt(c) for c in members if not c["chain_pos"]],
            })
        if attrib:
            out["attribution"] = self.attribution(top_groups, retrieved_units, qtok, qv)
        if INITIATIVE_WORDS.search(query):
            out["initiative"] = self.initiative()
        out["empty"] = not out["groups"] and not out["initiative"]
        return out

    # ------------------------------------------------------------ pieces
    def newest_figure(self, chain):
        """Newest surviving statement that carries a number and was not corrected; else the newest survivor.
        Used only when the chain lost its head: it is the best the archive still has, not the answer."""
        # a figure is an explicit number ("48%", "1,317"), not a word like "half", and not a recollection
        figs = [c for c in chain if has_number(c.get("value")) and c["truth_status"] != "never_true" and c["kind"] != "recollection"]
        pick = (figs or chain or [None])[-1]
        return self.fmt(pick) if pick else None

    @staticmethod
    def head_of(chain):
        """The chain's current statement: its newest factual claim. A withdrawal or correction is about an
        earlier statement and never carries the current value itself, so it cannot be the head."""
        factual = [c for c in chain if c["kind"] not in RETRACTING]
        return (factual or chain or [None])[-1]

    @staticmethod
    def chain_order(chain, head):
        """Chronological, except that withdrawals newer than the head are lifted to sit directly above
        it, so a chain reads: history, labelled retraction(s), then the statement that is current."""
        if head is None or head not in chain:
            return chain
        i = chain.index(head)
        later = chain[i + 1:]
        return chain[:i] + [c for c in later if c["kind"] == "withdrawal"] + [head] + [c for c in later if c["kind"] != "withdrawal"]

    def match_retraction(self, c):
        """The claim a retraction's free-text `corrects` describes, found at render time when link.py left
        no corrected_by pointer. Strict: an earlier, non-retracting claim in the same chain that the text
        names by speaker, dates, and whose value (every number) or wording (half its content words) it
        repeats. Exactly one such claim, else None; nothing is written back."""
        text = c.get("corrects") or ""
        dates = {f"{y}-{m}-{d}" for y, m, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", text)}
        months = set()
        for d, mon, y in re.findall(r"\b(\d{1,2})?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(20\d\d)\b", text, re.I):
            mm = MONTHS[mon[:3].lower()]
            (dates if d else months).add(f"{y}-{mm:02d}-{int(d):02d}" if d else f"{y}-{mm:02d}")
        if not dates and not months:
            return None
        ttok = content_tokens(text)
        tnums = set(re.findall(r"\d+", text))
        members = sorted(self.by_group[c["group_id"]], key=lambda x: (x["chain_pos"] or 0))
        hits = []
        for x in members:
            if x["claim_id"] == c["claim_id"] or x["kind"] in RETRACTING or not x["chain_pos"] or x["chain_pos"] >= (c["chain_pos"] or 0):
                continue
            if x["asserted_by"] not in text or not (x["date"][:10] in dates or x["date"][:7] in months):
                continue
            vnums = set(re.findall(r"\d+", str(x.get("value") or "")))
            vtok = content_tokens(str(x.get("value") or ""))
            stok = content_tokens(x["statement"])
            value_ok = bool(vnums or vtok) and vnums <= tnums and vtok <= ttok
            wording_ok = bool(stok) and len(stok & ttok) / len(stok) >= 0.5
            if value_ok or wording_ok:
                hits.append(x)
        return hits[0] if len(hits) == 1 else None

    def fmt(self, c, head=None):
        head_removed = self.groups.get(c["group_id"], {}).get("head_removed")
        is_head = c["is_current"] if head is None else c["claim_id"] == head["claim_id"]
        if c["kind"] == "withdrawal":
            label = "WITHDRAWN"
        elif c["kind"] == "correction":
            label = "CORRECTION"
        elif is_head:
            label = "NEWEST SURVIVING" if head_removed else "CURRENT"
        else:
            label = "SUPERSEDED" if c["superseded_by"] else ""
        retraction = None
        if c["kind"] in RETRACTING and c.get("corrects"):
            resolved = c["claim_id"] in self.resolved_retractions
            if c["kind"] == "withdrawal":
                retraction = f"WITHDRAWN: retracts {c['corrects']}"
            elif not resolved:
                retraction = f"CORRECTS: {c['corrects']}"
            if retraction and not resolved:
                m = self.match_retraction(c)
                if m:
                    retraction += (f" (link not established automatically; matching claim: {m['asserted_by']} on {m['date'][:10]}, "
                                   f"\"{m['statement']}\" ({anchor(m['where'])}))")
                else:
                    retraction += " (the retracted statement is not held in the archive as a claim)"
        sup = None
        # the head's stored successor can only be a retraction (see head_of); a current statement is not superseded
        if c["superseded_by"] and not (is_head and not head_removed):
            n = self.by_id[c["superseded_by"]]
            sup = f"superseded by {n['asserted_by']} on {n['date'][:10]} ({anchor(n['where'])})"
            if c["order_confidence"] == "same_day_unknown":
                sup += "  [order_confidence: same_day_unknown - same day, no clock time on one side]"
        corr = None
        if c["corrected_by"]:
            x = self.by_id[c["corrected_by"]]
            corr = f"corrected by {x['asserted_by']} on {x['date'][:10]}: \"{x['statement']}\" ({anchor(x['where'])})"
        return {
            "claim_id": c["claim_id"], "unit_id": c["unit_id"], "date": c["date"], "asserted_by": c["asserted_by"], "kind": c["kind"],
            "value": c.get("value"), "truncated": bool(c.get("truncated")), "statement": c["statement"],
            "currency": label, "supersession": sup, "truth_status": c["truth_status"], "truth_reason": c["truth_reason"],
            "correction": corr, "retraction": retraction, "order_confidence": c["order_confidence"], "cite": anchor(c["where"]),
            "reported": c["source_file"].startswith("reports/"),
        }

    def attribution(self, gids, retrieved_units, qtok=frozenset(), qv=None):
        # top two groups, plus any other top group that holds a proposal AND its answer (a self-contained
        # exchange); keeps neighbouring turns' unrelated proposals out without dropping the real one
        chosen = list(gids[:2])
        for g in gids[2:]:
            kinds = {c["kind"] for c in self.by_group[g]}
            if "proposal" in kinds and kinds & {"agreement", "decision", "rejection"}:
                chosen.append(g)
        pool = [c for g in chosen for c in self.by_group[g]]
        seen, uniq = set(), []
        for c in pool:
            if c["claim_id"] not in seen:
                seen.add(c["claim_id"])
                uniq.append(c)
        uniq.sort(key=lambda c: (c["date"], c["turn_index"] or 0))
        props_c = [c for c in uniq if c["kind"] == "proposal"]
        acts_c = [c for c in uniq if c["kind"] in ("agreement", "decision")]
        rejects_c = [c for c in uniq if c["kind"] == "rejection"]

        def relevance(p):
            cos = float(self.semb[self.sidx[p["claim_id"]]] @ qv) if qv is not None else 0.0
            return cos + 0.1 * len(set(qtok) & content_tokens(p["statement"]))

        # what a reply answers: its responds_to text names the proposer; among that person's earlier claims in
        # the pool, the one sharing most content words with the text (ties: the head proposal, then a non-proposal)
        def target(c, head=None):
            text = c.get("responds_to") or ""
            if not text:
                return None
            ttok = content_tokens(text)
            when = (c["date"], c["turn_index"] or 0)
            best, best_key = None, None
            for x in uniq:
                if x is c or x["asserted_by"] not in text or (x["date"], x["turn_index"] or 0) > when:
                    continue
                key = (len(ttok & content_tokens(x["statement"])), x is head, x["kind"] != "proposal")
                if best is None or key > best_key:
                    best, best_key = x, key
            return best

        if props_c:
            answered = {p["claim_id"]: 0 for p in props_c}
            for a in acts_c:
                t = target(a)
                if t is not None and t["claim_id"] in answered:
                    answered[t["claim_id"]] += 1
            head = max(props_c, key=lambda p: (answered[p["claim_id"]], relevance(p)))
            props_c = [head] + [p for p in props_c if p is not head]
            # a reply to a different proposal in the same chains is not agreement with this one, and nothing said
            # before the proposal (an earlier day, or an earlier turn of the same source that day) can agree to it
            def before(a):
                if a["date"][:10] != head["date"][:10]:
                    return a["date"][:10] < head["date"][:10]
                return a["source_file"] == head["source_file"] and (a["turn_index"] or 0) < (head["turn_index"] or 0)
            kept = []
            for a in acts_c:
                t = target(a, head)
                if before(a) or (t is not None and t["kind"] == "proposal" and t is not head):
                    continue
                kept.append(a)
            acts_c = kept
        props = [self.fmt(c) | {"responds_to": c.get("responds_to")} for c in props_c]
        agrees = [self.fmt(c) | {"responds_to": c.get("responds_to")} for c in acts_c]
        rejects = [self.fmt(c) | {"responds_to": c.get("responds_to")} for c in rejects_c]
        return {
            "proposed": props, "agreed": agrees, "rejected": rejects,
            "verdict": ("nobody agreed: no agreement or decision claim found in the retrieved evidence"
                        if props and not agrees else
                        ("no proposal found in the retrieved evidence" if not props else "see agreed list")),
        }

    def initiative(self, limit=3):
        """Chains where a commitment is followed by a later statement that it is outstanding. A trail is
        'still open' when no later statement in the chain says it was done. Everything named is a cited
        claim: the committer is the commitment's speaker, the people who would have needed to notice are
        the committer and whoever raised it afterwards."""
        trails = []
        for gid, members in self.by_group.items():
            ms = sorted(members, key=lambda c: (c["date"], c["turn_index"] or 0))
            commits = [c for c in ms if c["kind"] in ("agreement", "action", "decision")]
            if not commits:
                continue
            first = commits[0]
            undone = [c for c in ms if c["date"] > first["date"] and c["kind"] != "question" and UNDONE.search(c["statement"])]
            if not undone:
                continue
            last = undone[-1]
            done_later = [c for c in ms if c["date"] > last["date"] and DONE.search(c["statement"]) and not UNDONE.search(c["statement"])]
            days = (date.fromisoformat(last["date"][:10]) - date.fromisoformat(first["date"][:10])).days
            chasers = []
            for c in undone:
                if c["asserted_by"] != first["asserted_by"] and c["asserted_by"] not in [x[0] for x in chasers]:
                    chasers.append((c["asserted_by"], c["date"][:10]))
            trails.append({
                "group_id": gid, "fact_keys": self.groups[gid]["fact_keys"][:3],
                "agreed": self.fmt(first), "latest": self.fmt(last), "days": days,
                "still_open": not done_later, "done_later": self.fmt(done_later[0]) if done_later else None,
                "committer": first["asserted_by"], "raised_by": chasers,
                "statements": [self.fmt(c) for c in ms],
            })
        trails.sort(key=lambda t: (not t["still_open"], -t["days"]))
        return trails[:limit]


# ---------------------------------------------------------------- text rendering
def _statement_lines(s, indent="  "):
    flag = f"[{s['currency']}]" if s["currency"] else "[unlinked]"
    val = f" value={s['value']!r}" if s["value"] is not None else (" value=TRUNCATED IN SOURCE" if s["truncated"] else "")
    lines = [f"{indent}{s['date'][:16]}  {s['asserted_by']:16s} {flag:12s} {s['truth_status']:15s}{val}",
             f"{indent}    \"{s['statement']}\"",
             f"{indent}    cite: {s['cite']}"]
    if s.get("retraction"):
        lines.append(f"{indent}    {s['retraction']}")
    if s.get("reported"):
        lines.append(f"{indent}    status report: evidence of what was reported at the time, not of the underlying state")
    if s["supersession"]:
        lines.append(f"{indent}    {s['supersession']}")
    if s["correction"]:
        lines.append(f"{indent}    NEVER TRUE: {s['correction']}")
    elif s["truth_status"] == "unverified":
        lines.append(f"{indent}    unverified: {s['truth_reason']}")
    return lines


def render(out):
    lines = [f"Q: {out['query']}", ""]
    if out.get("scope"):
        lines.append(f"(scoped to {out['scope']}: retrieval limited to that period; chains with statements then are favoured)")
    if out.get("never_mentions"):
        lines.append(f"The archive never uses the word(s): {', '.join(out['never_mentions'])}.")
    if out.get("empty"):
        lines.append(NOTHING)
        if out.get("nearest"):
            lines.append("  The nearest chains, which do not answer this, are about: "
                         + "; ".join(", ".join(t) for t in out["nearest"]["topics"]) + ".")
    context_only = out["groups"] and out.get("direct") is False
    if context_only:
        lines.append("NO DIRECT ANSWER: no statement in the archive directly answers the question as asked. "
                     "The chains below are related context, not an answer.")
    for g in out["groups"]:
        lines.append(f"== {'related context: ' if context_only else ''}{', '.join(g['fact_keys'][:3])} ==")
        if g.get("head_removed"):
            n = g["removed_statements"]
            if g["statements"]:
                f = g["newest_surviving_figure"]
                what = "figure" if has_number(f["value"]) else "statement"
                val = f"{f['asserted_by']}'s {f['value']!r}" if f["value"] is not None else f"{f['asserted_by']}'s statement"
                lines.append(f"  HEAD REMOVED: the most recent statement in this chain was removed ({n} statement{'s' if n != 1 else ''} removed by deletion). "
                             f"The newest surviving {what} is {val} from {f['date'][:10]} ({f['cite']}), and it may have been "
                             f"superseded by a later statement that no longer exists. No statement below is current.")
            else:
                lines.append(f"  REMOVED: the archive no longer contains a statement on this. {n} statement{'s' if n != 1 else ''} "
                             f"existed and {'were' if n != 1 else 'was'} removed by deletion; none survive.")
        for s in g["statements"]:
            lines += _statement_lines(s)
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
    if out.get("initiative") is not None:
        lines.append("== agreed and not done ==")
        if not out["initiative"]:
            lines.append("  no chain in the archive has a commitment followed by a statement that it is outstanding")
        for i, t in enumerate(out["initiative"], 1):
            a, l = t["agreed"], t["latest"]
            lines.append(f"  {i}. {', '.join(t['fact_keys'])}")
            lines.append(f"     COMMITTED: {a['date'][:10]} {a['asserted_by']} [{a['kind']}] \"{a['statement']}\"  cite: {a['cite']}")
            lines.append(f"     LATEST ON RECORD: {l['date'][:10]} {l['asserted_by']} \"{l['statement']}\"  cite: {l['cite']}  ({t['days']} days later)")
            if t["still_open"]:
                lines.append("     status: no later statement in the archive says it was done; the archive does not record whether it ever was")
            else:
                d = t["done_later"]
                lines.append(f"     status: later reported done: {d['date'][:10]} {d['asserted_by']} \"{d['statement']}\"  cite: {d['cite']}")
            who = [f"{t['committer']} (made the commitment)"] + [f"{n} (raised it on {d})" for n, d in t["raised_by"]]
            lines.append(f"     who would have needed to notice: {'; '.join(who)}")
            lines.append("     trail:")
            for s in t["statements"]:
                lines.append(f"       {s['date'][:10]} {s['asserted_by']} [{s['kind']}] \"{s['statement']}\"  cite: {s['cite']}")
        lines.append("")
    lines.append("retrieved windows: " + "; ".join(anchor(h) for h in out["hits"][:5]))
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
