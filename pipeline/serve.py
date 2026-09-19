"""Judge-facing UI. Standard library only; one process; no login; no styling beyond legibility.

    python pipeline/serve.py                      # http://localhost:8080
    python pipeline/serve.py --host 0.0.0.0       # reachable from other machines on the network

Three things and nothing else:
  1. Ask a question -> answer.py's rendered claim graph (dates, sources, currency label, truth
     status, correcting document, order confidence, head-removed / removed notices).
  2. Clickable citations -> the human anchor string expands to the unit text stored in
     data/units.jsonl. corpus/ is never read at request time: it holds the unscrubbed original.
  3. A deletion control -> runs the real pipeline/delete.py as a subprocess, then reloads.

No unit id, claim id or turn index reaches the browser: the JSON sent to the page is built from an
allow-list of fields, and anchor strings have the internal-transcript turn number stripped.
"""
import argparse
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TURN_INDEX = re.compile(r" turn \d+ \(line")   # "Them turn 22 (line 38)" -> "Them (line 38)"
NOTHING = "The archive does not contain a statement on this."


def anchor(where):
    return TURN_INDEX.sub(" (line", where or "")


class State:
    """Answerer + unit store, rebuilt after a deletion. One lock: a deletion blocks questions."""

    def __init__(self):
        self.lock = threading.Lock()
        self.answerer = None
        self.units = None

    def load(self):
        from answer import Answerer
        self.answerer = Answerer()
        self.units = {u["unit_id"]: u for u in map(json.loads, (DATA / "units.jsonl").open(encoding="utf-8"))}

    def people(self):
        p = json.loads((DATA / "people.json").read_text(encoding="utf-8"))
        return sorted(({"person_id": k, "name": v["name"], "org": v.get("org"), "role": v.get("role")}
                       for k, v in p.items() if k.startswith("person:")), key=lambda x: x["name"])

    # ---- citations: stored unit text only, never corpus/
    def cite(self, claim_id):
        c = self.answerer.by_id.get(claim_id)
        u = self.units.get(c["unit_id"]) if c else None
        if not u:
            return None
        heading = u.get("subject") or u.get("meeting") or ""
        return {"label": anchor(u["anchor"]["human"]), "heading": heading, "speaker": u["speaker"],
                "date": u["date"], "text": u["text"]}

    def statement(self, s):
        """Allow-listed view of one answer.py statement, with citation objects instead of ids."""
        raw = self.answerer.by_id.get(s["claim_id"], {})
        out = {k: s.get(k) for k in ("date", "asserted_by", "kind", "value", "truncated", "statement", "currency",
                                     "truth_status", "truth_reason", "order_confidence", "responds_to", "reported", "retraction")}
        out["cite"] = self.cite(s["claim_id"])
        out["supersession"] = anchor(s["supersession"]) if s.get("supersession") else None
        out["supersession_cite"] = self.cite(raw["superseded_by"]) if raw.get("superseded_by") else None
        out["correction"] = anchor(s["correction"]) if s.get("correction") else None
        out["correction_cite"] = self.cite(raw["corrected_by"]) if raw.get("corrected_by") else None
        return out

    def ask(self, question):
        with self.lock:
            if self.answerer is None:
                self.load()
            out = self.answerer.answer(question)
            groups = []
            for g in out["groups"]:
                groups.append({
                    "fact_keys": g["fact_keys"][:3],
                    "head_removed": g["head_removed"], "removed_statements": g["removed_statements"],
                    "newest_surviving_figure": self.statement(g["newest_surviving_figure"]) if g.get("newest_surviving_figure") else None,
                    "statements": [self.statement(s) for s in g["statements"]],
                    "other": [self.statement(s) for s in g["other"]],
                })
            att = out["attribution"]
            if att:
                att = {"proposed": [self.statement(s) for s in att["proposed"]],
                       "agreed": [self.statement(s) for s in att["agreed"]],
                       "rejected": [self.statement(s) for s in att["rejected"]],
                       "verdict": att["verdict"]}
            initiative = None
            if out.get("initiative") is not None:
                initiative = [{
                    "fact_keys": t["fact_keys"], "days": t["days"], "still_open": t["still_open"],
                    "committer": t["committer"], "raised_by": t["raised_by"],
                    "agreed": self.statement(t["agreed"]), "latest": self.statement(t["latest"]),
                    "done_later": self.statement(t["done_later"]) if t["done_later"] else None,
                    "statements": [self.statement(s) for s in t["statements"]],
                } for t in out["initiative"]]
            empty = not groups and not initiative
            nothing = None
            if empty:
                nothing = NOTHING
                if out.get("nearest"):
                    nothing += (" The nearest chains, which do not answer this, are about: "
                                + "; ".join(", ".join(t) for t in out["nearest"]["topics"]) + ".")
            return {"question": question, "scope": out.get("scope"), "never_mentions": out.get("never_mentions") or [],
                    "direct": out.get("direct"),
                    "groups": groups, "attribution": att, "initiative": initiative, "empty": empty, "nothing": nothing}

    def delete(self, person_id):
        with self.lock:
            known = {p["person_id"] for p in self.people()}
            if person_id not in known:
                return {"ok": False, "error": "not in the registry (already deleted?)"}
            proc = subprocess.run([sys.executable, str(ROOT / "pipeline" / "delete.py"), person_id],
                                  cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
            self.answerer = None   # every store changed: reload on next question
            self.units = None
            if proc.returncode != 0:
                return {"ok": False, "error": f"delete.py exited {proc.returncode}",
                        "detail": proc.stderr[-2000:]}
            rec = json.loads((DATA / "deletion_log.jsonl").open(encoding="utf-8").readlines()[-1])
            chains = rec.get("chains_head_removed", [])
            summary = {
                "units_removed": len(rec.get("units_deleted", [])),
                "units_redacted": len(rec.get("units_redacted", {})),
                "claims_removed": len(rec.get("claims_raw_removed", [])),
                "chains_lost_head": len(chains),
                "chains_with_newest_survivor": sum(1 for r in chains if r.get("newest_surviving")),
                "chains_now_empty": sum(1 for r in chains if not r.get("newest_surviving")),
                "passages": rec.get("passages"), "claims": rec.get("claims"),
                "verify_clean": bool(rec.get("verify", {}).get("clean")),
            }
            self.load()   # warm, so the next question does not pay the model load
            return {"ok": True, "summary": summary, "people": self.people()}


STATE = State()

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ledger</title>
<style>
 :root{
   --bg:#f4f5f7; --paper:#ffffff; --paper-soft:#f7f8fa;
   --text:#15181d; --muted:#5b6470; --line:#e4e7eb; --line-strong:#7f8791;
   --accent:#0f5f8f; --accent-hover:#0c4f78; --accent-soft:#e7f0f7;
   --danger:#a6301f; --danger-soft:#fbe9e5;
   --shadow:0 1px 2px rgba(20,30,45,.06), 0 8px 24px -12px rgba(20,30,45,.18);
   --tag-current-bg:#e3f3e6; --tag-current-fg:#14603a; --tag-current-bd:#8fcf9d;
   --tag-superseded-bg:#eef0f3; --tag-superseded-fg:#4b5158; --tag-superseded-bd:#c3c9d1;
   --tag-newest-bg:#fbecc4; --tag-newest-fg:#6b4900; --tag-newest-bd:#e2b94f;
   --tag-nevertrue-bg:#f9dcd9; --tag-nevertrue-fg:#8a2015; --tag-nevertrue-bd:#e3897c;
   --tag-nodirect-bg:#e7e1f4; --tag-nodirect-fg:#452e82; --tag-nodirect-bd:#b8a3e0;
   --error-bg:#fbe4e1; --error-bd:#e7b6ae; --error-fg:#8a2a1c;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,system-ui,sans-serif;
   --r:.6rem;
   --s1:.5rem; --s2:1rem; --s3:1.5rem;
 }
 *{box-sizing:border-box}
 html{-webkit-text-size-adjust:100%}
 body{margin:0;min-height:100vh;color:var(--text);background:var(--bg);font:16px/1.55 var(--sans);overflow-wrap:break-word}
 /* the wash: soft blue atmosphere fixed behind everything; pure CSS gradients, never a text background */
 body::before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;background:
   radial-gradient(55% 45% at 82% 4%, rgba(133,171,209,.42), transparent 68%),
   radial-gradient(48% 40% at 8% 20%, rgba(150,196,196,.30), transparent 70%),
   radial-gradient(42% 38% at 60% 70%, rgba(230,184,146,.22), transparent 72%),
   linear-gradient(175deg, #eef1f5 0%, #e9edf2 100%)}
 a{color:var(--accent)}
 /* top bar */
 .topbar{background:rgba(255,255,255,.85);border-bottom:1px solid var(--line)}
 .topbar-in{max-width:46rem;margin:0 auto;padding:.85rem 1.1rem;display:flex;align-items:baseline;justify-content:space-between;gap:1rem;flex-wrap:wrap}
 .brand{font-size:1.35rem;font-weight:800;letter-spacing:-.02em;color:var(--text);margin:0}
 .topbar-sub{color:var(--muted);font-size:.92rem}
 .wrap{max-width:46rem;margin:0 auto;padding:var(--s3) 1rem 4rem}
 /* intro */
 .intro{margin:0 0 var(--s3)}
 .intro h1{font-size:1.6rem;line-height:1.25;font-weight:750;letter-spacing:-.015em;margin:0 0 var(--s1)}
 .intro p{margin:0;color:var(--muted)}
 /* cards */
 .card{background:var(--paper);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:var(--s3);margin:0 0 var(--s2)}
 .field-label{display:block;font-weight:600;font-size:.95rem;margin-bottom:var(--s1)}
 textarea{display:block;width:100%;min-height:3.4em;font:inherit;line-height:1.5;padding:.7em .85em;border:1px solid var(--line-strong);border-radius:var(--r);
   background:var(--paper);color:var(--text);resize:none;overflow:hidden;-webkit-appearance:none;appearance:none}
 textarea::placeholder,input::placeholder{color:#8a919b}
 input[type=text]{display:block;width:100%;min-height:2.75rem;font:inherit;line-height:1.5;padding:.55em .85em;border:1px solid var(--line-strong);border-radius:var(--r);background:var(--paper);color:var(--text);-webkit-appearance:none;appearance:none}
 textarea:focus,input:focus,select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
 button:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
 .helper{color:var(--muted);font-size:.86rem;margin:var(--s1) 0 0}
 .row{display:flex;align-items:center;gap:var(--s1) var(--s2);margin-top:var(--s2);flex-wrap:wrap}
 button{font:600 .95rem/1 var(--sans);min-height:2.75rem;padding:.6em 1.25em;border-radius:var(--r);border:1px solid var(--accent);
   background:var(--accent);color:#fff;cursor:pointer;touch-action:manipulation;transition:background-color .12s ease,border-color .12s ease}
 button:hover{background:var(--accent-hover);border-color:var(--accent-hover)}
 button:disabled{opacity:.5;cursor:not-allowed}
 button.btn-quiet{background:var(--paper);color:var(--accent)}
 button.btn-quiet:hover{background:var(--accent-soft)}
 button.btn-danger{background:var(--danger);border-color:var(--danger)}
 button.btn-danger:hover{background:#8b2818;border-color:#8b2818}
 button.btn-danger-quiet{background:var(--paper);border-color:var(--danger);color:var(--danger)}
 button.btn-danger-quiet:hover{background:var(--danger-soft)}
 .skip{position:absolute;left:-999px;top:.5rem;background:var(--paper);color:var(--accent);padding:.5em .8em;border-radius:.4rem;box-shadow:var(--shadow);z-index:10;font:600 .95rem/1 var(--sans)}
 .skip:focus{left:.75rem}
 .examples{display:flex;flex-wrap:wrap;gap:var(--s1);align-items:center;margin-top:var(--s2);padding-top:var(--s2);border-top:1px solid var(--line)}
 .examples-label{color:var(--muted);font-size:.88rem;margin-right:.2rem}
 .chip{min-height:2.25rem;padding:.4em .85em;font:500 .88rem/1.3 var(--sans);background:var(--paper);color:var(--text);
   border:1px solid var(--line);border-radius:999px;cursor:pointer;text-align:left}
 .chip:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
 /* legend */
 .card-quiet{background:var(--paper);border:1px solid var(--line);border-radius:var(--r);padding:var(--s1) var(--s3);margin:0 0 var(--s2)}
 .card-quiet > summary{padding:.55em 0;font-weight:600;color:var(--text)}
 .legend-list{margin:var(--s1) 0 var(--s2);display:grid;grid-template-columns:auto 1fr;gap:.6em 1rem;align-items:start;font-size:.92rem}
 .legend-list dt{margin:0;display:flex;gap:.3em;flex-wrap:wrap}
 .legend-list dd{margin:0;color:var(--muted);line-height:1.45}
 @media (max-width:520px){.legend-list{grid-template-columns:1fr}.legend-list dd{margin-bottom:.4em}}
 /* answer */
 #answer:not(:empty){margin:0 0 var(--s2)}
 .answer-head{display:flex;justify-content:space-between;align-items:flex-end;gap:var(--s1) var(--s2);flex-wrap:wrap;margin:0 0 var(--s2);color:var(--muted);font-size:.9rem}
 .answer-head .q{display:block;color:var(--text);font-weight:650;font-size:1.15rem;line-height:1.35}
 .link-btn{background:none;border:none;color:var(--accent);font:600 .88rem/1 var(--sans);padding:.6em .4em;min-height:2.25rem;cursor:pointer;text-decoration:underline}
 .link-btn:hover{background:var(--accent-soft)}
 .eyebrow{display:block;font:700 .74rem/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .5em}
 .g{background:var(--paper);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:var(--s3);margin:0 0 var(--s2)}
 .lbl{font:650 1.08rem/1.35 var(--sans);color:var(--text);margin:0 0 var(--s1)}
 h3.lbl{font-size:.95rem;margin:.9rem 0 .1rem;color:var(--muted)}
 .lead-note{font:600 .76rem/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--tag-current-fg);margin:.6rem 0 .5rem}
 .claim{margin:.6rem 0;padding:.1rem 0 .1rem .9rem;border-left:3px solid var(--line)}
 .claim.lead{border-left-color:var(--tag-current-bd);background:var(--paper-soft);border-radius:0 var(--r) var(--r) 0;padding:.75rem 1rem .6rem 1rem}
 .claim-text{margin:0 0 .45rem;font-size:1.02rem;line-height:1.55}
 .claim.lead .claim-text{font-size:1.08rem}
 .claim-meta{display:flex;align-items:center;gap:.4em .7em;flex-wrap:wrap;font-size:.86rem;margin:0 0 .35rem;color:var(--muted)}
 .claim-who{color:var(--muted)}
 .kind{font:600 .74rem/1 var(--sans);letter-spacing:.04em;text-transform:uppercase;color:var(--muted);border:1px dashed var(--line-strong);border-radius:.3rem;padding:.3em .5em}
 .claim-anchor{font-size:.88rem;margin:0}
 .claim-extra{font-size:.86rem;color:var(--muted);line-height:1.5;margin-top:.25rem}
 .claim-extra > div{margin:.15em 0}
 details.more{margin:.2rem 0 0}
 details.more > summary{font-size:.8rem;color:var(--muted);font-weight:500}
 .role-label{font:700 .74rem/1 var(--sans);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
 .tag{display:inline-flex;align-items:center;gap:.3em;font:700 .7rem/1 var(--sans);letter-spacing:.04em;text-transform:uppercase;
   padding:.32em .55em;border-radius:.3rem;border:1.5px solid transparent;white-space:nowrap}
 .tag-current{background:var(--tag-current-bg);color:var(--tag-current-fg);border-color:var(--tag-current-bd)}
 .tag-superseded{background:var(--tag-superseded-bg);color:var(--tag-superseded-fg);border-color:var(--tag-superseded-bd);border-style:dashed}
 .tag-newest{background:var(--tag-newest-bg);color:var(--tag-newest-fg);border-color:var(--tag-newest-bd);border-style:dashed}
 .tag-nevertrue{background:var(--tag-nevertrue-bg);color:var(--tag-nevertrue-fg);border-color:var(--tag-nevertrue-bd);border-width:2px}
 .tag-withdrawn,.tag-correction{background:var(--tag-superseded-bg);color:var(--tag-superseded-fg);border-color:var(--tag-superseded-bd);border-style:dotted}
 .tag-nodirect{background:var(--tag-nodirect-bg);color:var(--tag-nodirect-fg);border-color:var(--tag-nodirect-bd);border-style:dotted;border-width:2px}
 .retraction{font-style:italic}
 .corrector-label{font-size:.86rem;color:var(--muted)}
 .corrector-label.never{color:var(--tag-nevertrue-fg);font-weight:600}
 .sysnote{background:var(--accent-soft);border-left:3px solid var(--accent);border-radius:.3rem;padding:.65em .85em;margin:var(--s1) 0;font-size:.95rem;line-height:1.5}
 .sysnote .tag{margin-right:.4em}
 .error{background:var(--error-bg);border:1px solid var(--error-bd);color:var(--error-fg);border-radius:.4rem;padding:.7em .85em;margin:var(--s1) 0;font-size:.95rem}
 .loading{color:var(--muted);padding:.8rem 0}
 .loading::after{content:"";display:inline-block;width:1.1em;text-align:left;animation:ellipsis 1.2s steps(4,end) infinite}
 @keyframes ellipsis{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}
 @media (prefers-reduced-motion:reduce){*{transition:none !important;animation:none !important}.loading::after{content:"…"}}
 details{margin:.25em 0}
 summary{cursor:pointer;color:var(--accent);font:600 .9rem/1.4 var(--sans);display:list-item;overflow-wrap:anywhere;padding:.3em 0;touch-action:manipulation}
 summary:hover{text-decoration:underline}
 details.history{margin:.5rem 0 .2rem}
 details > div.cite-meta{color:var(--muted);font-size:.85rem;margin:.4em 0 .3em}
 pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--paper-soft);border:1px solid var(--line);border-radius:.45rem;
   padding:.7em .85em;margin:.3em 0 .5em;font:.95rem/1.55 var(--sans);color:var(--text);max-height:22em;overflow-y:auto}
 hr.section-break{border:none;border-top:1px solid var(--line);margin:var(--s2) 0}
 /* deletion */
 .quiet-panel{background:var(--paper-soft);border:1px solid var(--line);border-radius:var(--r);padding:var(--s3)}
 .quiet-title{font-size:1rem;font-weight:650;margin:0 0 var(--s1)}
 .quiet-intro{color:var(--muted);font-size:.95rem;margin:0 0 var(--s2)}
 .fineprint{color:var(--text);font-size:.92rem;line-height:1.5;margin:var(--s2) 0 0}
 .confirm-box{background:var(--danger-soft);border:1px solid var(--tag-nevertrue-bd);border-radius:.45rem;padding:.8em 1em;margin-top:.8em}
 .confirm-result{background:var(--tag-current-bg);border:1px solid var(--tag-current-bd);border-radius:.45rem;padding:.85em 1em;margin-top:.8em;font-size:.95rem;line-height:1.55}
 .confirm-result ul{margin:.4em 0 0;padding-left:1.2em}
 .del-hint{font-size:.88em}
 .muted{color:var(--muted)} .small{font-size:.87em} .grow{flex:1 1 12rem;min-width:0}
</style></head><body>
<a class="skip" href="#main">Skip to the question</a>
<div class="topbar"><div class="topbar-in"><p class="brand">Ledger</p><span class="topbar-sub">Every answer cited. Every person erasable.</span></div></div>
<div class="wrap">

<div class="intro">
<h1>Ask the archive anything.</h1>
<p>Answers are built only from statements stored in the archive, shown with the history of what was said, corrected or withdrawn.</p>
</div>

<main id="main">
<section class="card" aria-labelledby="ask-label">
<label class="field-label" id="ask-label" for="q">Your question</label>
<textarea id="q" aria-describedby="q-help" placeholder="e.g. What service levels were agreed for ordering, and in which meeting?"></textarea>
<p class="helper" id="q-help">Ctrl+Enter also asks. Not sure where to start? Try an example.</p>
<div class="row"><button id="ask" type="button">Ask</button></div>
<div class="examples" id="examples" aria-label="Example questions">
  <span class="examples-label">Try:</span>
  <button type="button" class="chip" data-q="What service levels were agreed for ordering, and in which meeting?">Service levels agreed for ordering</button>
  <button type="button" class="chip" data-q="How many stores ended up in the pilot group, and who changed the number?">Pilot store count and who changed it</button>
  <button type="button" class="chip" data-q="Who proposed taking bakery out of the fresh go-live, and who agreed?">Who took bakery out of the go-live</button>
  <button type="button" class="chip" data-q="Find one thing in the archive that was agreed and then never done. Show the trail from the agreement to the last mention.">Something agreed and never done</button>
</div>
</section>

<details class="legend card-quiet" open>
<summary>How to read the labels</summary>
<dl class="legend-list">
  <dt><span class="tag tag-current"><span aria-hidden="true">●</span>CURRENT</span></dt><dd>The latest statement on this topic that still stands.</dd>
  <dt><span class="tag tag-superseded"><span aria-hidden="true">→</span>SUPERSEDED</span></dt><dd>Was true when said; a later statement replaced it. The replacing document is named beside it.</dd>
  <dt><span class="tag tag-nevertrue"><span aria-hidden="true">✕</span>NEVER TRUE</span></dt><dd>Later corrected: this was wrong when it was said. The correction is quoted.</dd>
  <dt><span class="tag tag-correction"><span aria-hidden="true">✎</span>CORRECTION</span> <span class="tag tag-withdrawn"><span aria-hidden="true">↩</span>WITHDRAWN</span></dt><dd>A statement that corrects or retracts an earlier one.</dd>
  <dt><span class="tag tag-newest"><span aria-hidden="true">◐</span>NEWEST SURVIVING</span></dt><dd>After a deletion removed the latest statement, this is the newest one left. It may itself have been out of date.</dd>
  <dt><span class="tag tag-nodirect"><span aria-hidden="true">?</span>NO DIRECT ANSWER</span></dt><dd>Nothing in the archive answers the question as asked; what is shown is related context only.</dd>
</dl>
</details>

<div id="answer" role="region" aria-label="Answer" aria-live="polite" aria-busy="false"></div>
</main>

<hr class="section-break">

<section class="quiet-panel" aria-labelledby="del-title">
<h2 class="quiet-title" id="del-title">Delete a person from the archive</h2>
<p class="quiet-intro">Pick a name, confirm, and their statements, mentions and every derived record are removed. Then ask your question again to see what the answer lost.</p>
<label class="field-label" for="person">Person to erase</label>
<div class="row">
  <input id="person" class="grow" type="text" list="people-list" autocomplete="off" placeholder="Type or pick a name…" aria-describedby="delhint">
  <datalist id="people-list"></datalist>
  <button id="del" class="btn-danger-quiet" type="button">Delete</button>
  <span id="delhint" class="muted del-hint" role="status"></span>
</div>
<div id="delconfirm" class="confirm-box" role="alertdialog" aria-labelledby="delwho" hidden><span id="delwho"></span> will be erased from every store. This cannot be undone from the interface.
 <div class="row"><button id="delyes" class="btn-danger" type="button">Confirm delete</button> <button id="delno" class="btn-quiet" type="button">Cancel</button></div>
</div>
<p class="fineprint">This runs the real deletion pipeline and rebuilds the search index. It takes about half a minute. There is no undo.</p>
<div id="delresult" role="status" aria-live="polite"></div>
</section>

</div>
<script>
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
// a leading glyph on every tag: a third, non-colour cue (with the border style in CSS) so the
// five currency states stay distinguishable from each other in grayscale or low light, not just by hue.
const TAG_GLYPH = {current:"●", superseded:"→", newest:"◐", nevertrue:"✕", nodirect:"?", withdrawn:"↩", correction:"✎"};
function tag(kind, label){
  return `<span class="tag tag-${kind}"><span aria-hidden="true">${TAG_GLYPH[kind]||""}</span>${esc(label)}</span>`;
}
function currencyTag(currency){
  if(currency==="CURRENT") return tag("current", "CURRENT");
  if(currency==="SUPERSEDED") return tag("superseded", "SUPERSEDED");
  if(currency==="NEWEST SURVIVING") return tag("newest", "NEWEST SURVIVING");
  if(currency==="WITHDRAWN") return tag("withdrawn", "WITHDRAWN");
  if(currency==="CORRECTION") return tag("correction", "CORRECTION");
  return "";
}
function cite(c, prefix){
  if(!c) return "";
  const head = c.heading ? esc(c.heading) + " — " : "";
  return `<details><summary>${esc(prefix||"source")}: ${esc(c.label)}</summary>` +
         `<div class="cite-meta">${head}${esc(c.speaker)}, ${esc(c.date)}</div><pre>${esc(c.text)}</pre></details>`;
}
// s.supersession/s.correction are full sentences ("superseded by X on DATE (anchor...)" /
// "corrected by X on DATE: "..." (anchor...)"). For the tag row we need only the name + date so
// it reads as one short phrase next to the tag; the anchor is shown separately, as a citation.
function correctorName(text){
  const m = /^(superseded by|corrected by) (.+?) on (\d{4}-\d{2}-\d{2})/.exec(text || "");
  return m ? `${m[1]} ${m[2]}, ${m[3]}` : (text || "");
}
function correctionQuote(text){
  const m = /corrected by .+? on \d{4}-\d{2}-\d{2}:\s*"([\s\S]*)"\s*\([^)]*\)\s*$/.exec(text || "");
  return m ? m[1] : "";
}
// One claim = one block, fixed reading order: text, currency tag, correcting document
// (named right next to the tag), then this claim's own citation anchor. Nothing here needs
// expanding to be seen; expanding a <details> only reveals the source excerpt, never the tag
// or the correcting document's name.
function claim(s, lead){
  const val = s.value!==null&&s.value!==undefined ? `value: ${esc(s.value)}` : (s.truncated ? "value: TRUNCATED IN SOURCE" : "");
  let h = `<div class="claim${lead ? " lead" : ""}">`;
  h += `<p class="claim-text">“${esc(s.statement)}”</p>`;
  const ct = currencyTag(s.currency);
  h += `<div class="claim-meta">` + (ct || `<span class="kind">${esc(s.kind || "statement")}</span>`);
  if(s.correction){
    h += tag("nevertrue","NEVER TRUE") + `<span class="corrector-label never">${esc(correctorName(s.correction))}</span>` + cite(s.correction_cite, "correcting document");
  } else if(s.supersession){
    h += `<span class="corrector-label">${esc(correctorName(s.supersession))}</span>` + cite(s.supersession_cite, "superseding document");
  }
  h += `<span class="claim-who">${esc(s.asserted_by)} · ${esc(s.date.slice(0,10))}</span></div>`;
  h += `<div class="claim-anchor">` + cite(s.cite, "source") + `</div>`;
  // what a reader needs stays visible; bookkeeping (truth status, ordering confidence) sits behind "details"
  let extra = "", more = "";
  if(val) extra += `<div>${esc(val)}</div>`;
  if(s.retraction) extra += `<div class="retraction">${esc(s.retraction)}</div>`;
  if(s.reported) extra += `<div>From a status report: evidence of what was reported at the time, not of the underlying state.</div>`;
  if(s.correction){ const q = correctionQuote(s.correction); if(q) extra += `<div>What was actually true: “${esc(q)}”</div>`; }
  if(s.responds_to) extra += `<div>In response to: ${esc(s.responds_to)}</div>`;
  if(s.truth_status==="unverified" && s.truth_reason && !String(s.truth_reason).startsWith("kind=")) extra += `<div>Unverified: ${esc(s.truth_reason)}</div>`;
  if(!s.correction) more += `<div>truth status: ${esc(String(s.truth_status).replace(/_/g," "))}</div>`;
  if(s.order_confidence) more += `<div>order confidence: ${esc(s.order_confidence)}</div>`;
  if(extra) h += `<div class="claim-extra">${extra}</div>`;
  if(more) h += `<details class="more"><summary>details</summary><div class="claim-extra">${more}</div></details>`;
  return h + "</div>";
}
// Same visual language as claim(), with a role label (COMMITTED / LATEST ON RECORD / DONE LATER)
// standing in for the group heading, used in the "agreed and not done" trail.
function initiativeItem(roleLabel, s){
  let h = `<div class="claim">`;
  h += `<p class="claim-text">“${esc(s.statement)}”</p>`;
  h += `<div class="claim-meta"><span class="role-label">${esc(roleLabel)}</span>` + currencyTag(s.currency);
  if(s.correction) h += tag("nevertrue","NEVER TRUE") + `<span class="corrector-label never">${esc(correctorName(s.correction))}</span>` + cite(s.correction_cite, "correcting document");
  else if(s.supersession) h += `<span class="corrector-label">${esc(correctorName(s.supersession))}</span>` + cite(s.supersession_cite, "superseding document");
  h += `<span class="claim-who">${esc(s.asserted_by)} · ${esc(s.date.slice(0,10))}</span></div>`;
  h += `<div class="claim-anchor">` + cite(s.cite, "source") + `</div>`;
  return h + "</div>";
}
function renderAnswer(a){
  let h = "";
  if(a.never_mentions && a.never_mentions.length)
    h += `<div class="sysnote">The archive never uses the word${a.never_mentions.length>1?"s":""} <b>${esc(a.never_mentions.join(", "))}</b>, so nothing here can be matched on ${a.never_mentions.length>1?"them":"it"}.</div>`;
  if(a.empty) return h + `<div class="sysnote">${esc(a.nothing)}</div>`;
  if(a.scope) h += `<p class="muted small">Limited to ${esc(a.scope)}: only statements from that period were searched; topics with statements then come first.</p>`;
  const contextOnly = a.groups.length && a.direct === false;
  if(contextOnly) h += `<div class="sysnote">${tag("nodirect","NO DIRECT ANSWER")}Nothing in the archive answers this question as asked. Below is the closest related context, shown so you can judge for yourself; it is not an answer.</div>`;
  if(a.initiative){
    h += `<div class="g"><h2 class="eyebrow">agreed and not done</h2>`;
    if(!a.initiative.length) h += `<div class="claim muted">no chain in the archive has a commitment followed by a statement that it is outstanding</div>`;
    a.initiative.forEach((t, i) => {
      h += `<h3 class="lbl">${i+1}. ${esc(t.fact_keys.join(", "))}</h3>`;
      h += initiativeItem("COMMITTED", t.agreed);
      h += initiativeItem(`LATEST ON RECORD (${t.days} days later)`, t.latest);
      if(t.still_open) h += `<div class="sysnote">No later statement in the archive says it was done; the archive does not record whether it ever was.</div>`;
      else h += initiativeItem("DONE LATER", t.done_later);
      const who = [`${esc(t.committer)} (made the commitment)`].concat(t.raised_by.map(([n,d]) => `${esc(n)} (raised it on ${esc(d)})`));
      h += `<p class="muted small">Who would have needed to notice: ${who.join("; ")}</p>`;
      h += `<details><summary>full trail (${t.statements.length} statements)</summary>` + t.statements.map(claim).join("") + `</details>`;
    });
    h += `</div>`;
  }
  // for a who-proposed / who-agreed question the attribution block is the answer: it goes first
  const attribFirst = !!a.attribution && /(^|[^a-z])who([^a-z]|$)/i.test(a.question || "");
  if(attribFirst) h += attributionBlock(a.attribution);
  for(const g of a.groups){
    h += `<div class="g">${contextOnly ? '<span class="eyebrow">related context</span>' : ""}<h2 class="lbl">${esc(g.fact_keys.join(", "))}</h2>`;
    if(g.head_removed){
      const n = g.removed_statements, pl = n===1?"":"s";
      if(g.statements.length){
        const f = g.newest_surviving_figure;
        const isFig = /\d/.test(String(f.value ?? ""));
        const what = isFig ? "figure" : "statement";
        const fv = f.value!==null&&f.value!==undefined ? `${esc(f.asserted_by)}'s “${esc(f.value)}”` : `${esc(f.asserted_by)}'s statement`;
        h += `<div class="sysnote">${tag("newest","NEWEST SURVIVING")}The most recent statement in this chain was removed (${n} statement${pl} removed by deletion). ` +
             `The newest surviving ${what} is ${fv} from ${esc(f.date.slice(0,10))}, and it may have been superseded by a later ` +
             `statement that no longer exists. No statement below is current.</div>` + cite(f.cite, "newest surviving " + what);
      } else {
        h += `<div class="sysnote">The archive no longer contains a statement on this. ${n} statement${pl} existed and ` +
             `${n===1?"was":"were"} removed by deletion; none survive.</div>`;
      }
    }
    // the statement that stands today leads; everything it replaced sits behind one disclosure
    const st = g.statements;
    const leadIdx = st.findIndex(s => s.currency === "CURRENT" || s.currency === "NEWEST SURVIVING");
    const lead = leadIdx >= 0 ? st[leadIdx] : null;
    const rest = lead ? st.filter((_, i) => i !== leadIdx) : st;
    if(lead){
      h += `<p class="lead-note">${lead.currency === "CURRENT" ? "What stands today" : "Newest statement still in the archive"}</p>` + claim(lead, true);
    }
    if(rest.length){
      const open = rest.length <= 2 ? " open" : "";
      h += `<details class="history"${open}><summary>${lead ? "How it got here" : "Statements"}: ${rest.length} earlier statement${rest.length===1?"":"s"}, oldest first</summary>` +
           rest.map(s => claim(s)).join("") + `</details>`;
    }
    if(g.other.length){
      h += `<details class="history"><summary>Related proposals and questions (${g.other.length}), not part of the chain</summary>` + g.other.map(s => claim(s)).join("") + `</details>`;
    }
    h += "</div>";
  }
  if(a.attribution && !attribFirst) h += attributionBlock(a.attribution);
  return h;
}
function attributionBlock(at){
  let h = `<div class="g"><h2 class="eyebrow">Who proposed, who agreed</h2>`;
  for(const [name, items, emptyMsg] of [
    ["Proposed by", at.proposed, "No statement in the archive proposed this."],
    ["Agreed or decided by", at.agreed, "No statement in the archive agreed to or decided this. A proposal alone is not a decision."],
    ["Rejected by", at.rejected, "No statement in the archive rejected this."],
  ]){
    if(name.startsWith("Rejected") && !items.length) continue;
    h += `<h3 class="lbl">${name}</h3>` + (items.length ? items.map((s, i) => claim(s, i === 0 && name.startsWith("Proposed"))).join("") : `<div class="sysnote">${esc(emptyMsg)}</div>`);
  }
  const verdict = at.verdict.startsWith("nobody agreed") ? "Nobody agreed: the archive holds the proposal but no agreement or decision on it."
                : at.verdict.startsWith("no proposal") ? "No proposal found in the retrieved evidence." : "";
  if(verdict) h += `<div class="sysnote">${esc(verdict)}</div>`;
  return h + `</div>`;
}
async function post(url, body){
  const r = await fetch(url, {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify(body)});
  return r.json();
}
let PEOPLE_BY_LABEL = {};
function personLabel(p){
  return p.name + (p.org ? " — " + p.org : "") + (p.role ? ", " + p.role : "");
}
function fillPeople(list){
  const dl = document.getElementById("people-list"); dl.innerHTML = "";
  PEOPLE_BY_LABEL = {};
  for(const p of list){
    const label = personLabel(p);
    PEOPLE_BY_LABEL[label] = p.person_id;
    const o = document.createElement("option"); o.value = label;
    dl.appendChild(o);
  }
}
const qEl = document.getElementById("q"), askBtn = document.getElementById("ask"), answerEl = document.getElementById("answer");
let lastQuestion = "";
function answerHead(q, a){
  const n = a.groups ? a.groups.length : 0;
  const what = a.empty ? "no matching statements" : `${n} topic${n===1?"":"s"}`;
  return `<div class="answer-head"><span><span class="eyebrow">Answer</span><span class="q">${esc(q)}</span></span>` +
         `<span>${what}${n ? ` · <button type="button" class="link-btn" id="expand-all">expand all history and sources</button>` : ""}</span></div>`;
}
askBtn.onclick = async () => {
  const q = qEl.value.trim(); if(!q) return;
  lastQuestion = q;
  askBtn.disabled = true; answerEl.setAttribute("aria-busy", "true");
  answerEl.innerHTML = `<div class="loading">Searching the archive</div>`;
  try {
    const a = await post("/api/ask", {question:q});
    answerEl.innerHTML = a.error ? `<div class="error" role="alert">${esc(a.error)}</div>` : answerHead(q, a) + renderAnswer(a);
    const ex = document.getElementById("expand-all");
    if(ex) ex.onclick = () => {
      const all = answerEl.querySelectorAll("details"); const anyClosed = [...all].some(d => !d.open);
      all.forEach(d => d.open = anyClosed); ex.textContent = anyClosed ? "collapse history and sources" : "expand all history and sources";
    };
    answerEl.scrollIntoView({block:"start", behavior:"smooth"});
  }
  catch(e){ answerEl.innerHTML = `<div class="error" role="alert">The request failed (${esc(e)}). Check that the server is still running, then ask again.</div>`; }
  askBtn.disabled = false; answerEl.setAttribute("aria-busy", "false");
};
function grow(){ qEl.style.height = "auto"; qEl.style.height = qEl.scrollHeight + "px"; }
qEl.addEventListener("input", grow); window.addEventListener("resize", grow); grow();
document.querySelectorAll(".chip").forEach(c => c.onclick = () => { qEl.value = c.dataset.q; grow(); askBtn.click(); });
qEl.addEventListener("keydown", e => { if(e.key === "Enter" && (e.ctrlKey || e.metaKey)) askBtn.click(); });
const personEl = document.getElementById("person"), delHint = document.getElementById("delhint");
document.getElementById("del").onclick = () => {
  const pid = PEOPLE_BY_LABEL[personEl.value];
  if(!pid){ delHint.textContent = "Type or pick a full name from the list."; return; }
  delHint.textContent = "";
  document.getElementById("delwho").textContent = personEl.value;
  document.getElementById("delconfirm").hidden = false; document.getElementById("delresult").innerHTML = "";
};
document.getElementById("delno").onclick = () => { document.getElementById("delconfirm").hidden = true; };
document.getElementById("delyes").onclick = async () => {
  const pid = PEOPLE_BY_LABEL[personEl.value]; if(!pid) return;
  document.getElementById("delconfirm").hidden = true; document.getElementById("delwho").textContent = "";
  delHint.textContent = "deleting and rebuilding every store (about half a minute)…";
  document.getElementById("del").disabled = true; document.getElementById("delresult").innerHTML = "";
  try {
    const r = await post("/api/delete", {person_id:pid});
    if(!r.ok){ document.getElementById("delresult").innerHTML = `<div class="error" role="alert">${esc(r.error)}<br>${esc(r.detail||"")}</div>`; }
    else {
      const s = r.summary;
      document.getElementById("delresult").innerHTML =
        `<div class="confirm-result"><strong>Deleted.</strong> Removed from the archive:<ul>` +
        `<li>${s.units_removed} messages / turns they authored removed; ${s.units_redacted} others that named them redacted</li>` +
        `<li>${s.claims_removed} claims removed (theirs, and others' claims about them)</li>` +
        `<li>${s.chains_lost_head} chains lost their most recent statement: ${s.chains_with_newest_survivor} now show a newest surviving statement, ${s.chains_now_empty} are empty</li>` +
        `<li>retrieval index rebuilt: ${s.passages[0]} → ${s.passages[1]} passages, embeddings re-encoded; claims ${s.claims[0]} → ${s.claims[1]}</li>` +
        `<li>full-text and structural sweep: ${s.verify_clean ? "clean" : "NOT CLEAN — see deletion log"}</li></ul>` +
        (lastQuestion ? `<div class="row"><button type="button" id="askagain">Ask your last question again</button><span class="muted small">to see what the answer lost</span></div>` : "") + `</div>`;
      if(!s.verify_clean) document.getElementById("delresult").innerHTML += `<div class="error" role="alert">Verification sweep did not come back clean — see the deletion log.</div>`;
      const again = document.getElementById("askagain");
      if(again) again.onclick = () => { qEl.value = lastQuestion; grow(); askBtn.click(); document.getElementById("main").scrollIntoView({behavior:"smooth"}); };
      personEl.value = ""; fillPeople(r.people);
    }
  } catch(e){ document.getElementById("delresult").innerHTML = `<div class="error">request failed: ${esc(e)}</div>`; }
  delHint.textContent = ""; document.getElementById("del").disabled = false;
};
fetch("/api/people").then(r => r.json()).then(fillPeople);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):   # quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/people":
            return self._send(200, STATE.people())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad json"})
        try:
            if self.path == "/api/ask":
                q = (body.get("question") or "").strip()
                if not q:
                    return self._send(400, {"error": "empty question"})
                return self._send(200, STATE.ask(q))
            if self.path == "/api/delete":
                pid = (body.get("person_id") or "").strip()
                if not pid.startswith("person:"):
                    return self._send(400, {"ok": False, "error": "pick a person"})
                return self._send(200, STATE.delete(pid))
        except Exception as e:   # surface, never fabricate
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print("loading index and claim graph ...", flush=True)
    STATE.load()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ready: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
