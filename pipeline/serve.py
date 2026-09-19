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
<title>relex archive</title>
<style>
 :root{
   /* the wash: atmosphere only, never a text background */
   --wash-base:#eef1f5; --wash-base-2:#e9edf2;
   --wash-cool-1:rgba(133,171,209,.50); --wash-cool-2:rgba(150,196,196,.38); --wash-warm:rgba(230,184,146,.34);

   /* every panel that holds text is opaque */
   --paper:#ffffff; --paper-soft:#f6f7f9;
   --text:#1b1e23; --muted:#585f68; --border:#e2e5ea; --border-soft:#edeff2;
   --shadow:0 1.5rem 3rem -1.75rem rgba(25,35,55,.28), 0 .25rem .6rem -.3rem rgba(25,35,55,.12);
   --shadow-quiet:0 .6rem 1.6rem -1rem rgba(25,35,55,.16);

   --accent:#2f6690; --accent-soft:#e4eef5;
   --heading:#2a2e35;

   --tag-current-bg:#e2f1e2; --tag-current-fg:#186238; --tag-current-bd:#8fcf9d;
   --tag-superseded-bg:#eceef1; --tag-superseded-fg:#4b5158; --tag-superseded-bd:#c3c9d1;
   --tag-newest-bg:#fbecc4; --tag-newest-fg:#6b4900; --tag-newest-bd:#e2b94f;
   --tag-nevertrue-bg:#f9dcd9; --tag-nevertrue-fg:#8a2015; --tag-nevertrue-bd:#e3897c;
   --tag-nodirect-bg:#e7e1f4; --tag-nodirect-fg:#452e82; --tag-nodirect-bd:#b8a3e0;
   --error-bg:#fbe4e1; --error-bd:#e7b6ae; --error-fg:#8a2a1c;

   --serif:Georgia,"Iowan Old Style","Palatino Linotype",Cambria,serif;
   --sans:-apple-system,"Segoe UI",system-ui,sans-serif;
 }
 *{box-sizing:border-box}
 html{-webkit-text-size-adjust:100%}
 body{
   margin:0; width:100%; min-height:100vh; color:var(--text);
   font:1rem/1.55 var(--sans); overflow-wrap:break-word;
   background:var(--wash-base);
 }
 /* the hazy wash: pure CSS gradients, fixed behind everything, no blur/filter/animation/image */
 body::before{
   content:""; position:fixed; inset:0; z-index:-1;
   background:
     radial-gradient(52% 42% at 80% 6%, var(--wash-cool-1), transparent 68%),
     radial-gradient(46% 38% at 10% 22%, var(--wash-cool-2), transparent 70%),
     radial-gradient(40% 36% at 62% 68%, var(--wash-warm), transparent 72%),
     linear-gradient(175deg, var(--wash-base) 0%, var(--wash-base-2) 100%);
 }
 .wrap{width:100%;max-width:46rem;margin:0 auto;padding:clamp(1.25rem,4vw,2.5rem) 1.1rem clamp(3rem,8vw,5rem)}

 /* the only things allowed to sit directly on the wash: a big decorative heading + a short label */
 .hero-row{display:flex;align-items:baseline;justify-content:space-between;gap:1rem 1.5rem;flex-wrap:wrap}
 h1{font-size:clamp(1.7rem,4vw,2.3rem);font-weight:800;letter-spacing:-.02em;color:var(--heading);margin:0}
 .hero-meta{color:var(--muted);font:600 .82rem/1.4 var(--sans);text-align:right;text-transform:uppercase;letter-spacing:.04em}
 hr.hairline{border:none;border-top:1px solid rgba(25,35,55,.16);margin:1rem 0 1.75rem}

 /* everything else: opaque panels, floating on the haze via shadow, never a border-only edge */
 .card{background:var(--paper);border-radius:1rem;box-shadow:var(--shadow);padding:1.4rem 1.5rem;margin:0 0 1.5rem}
 .tagline{color:var(--muted);margin:0 0 1.25rem;font-size:1rem;max-width:60ch}
 .field-label{display:block;font-weight:600;font-size:1rem;margin-bottom:.5em}
 textarea{width:100%;min-height:4.5em;font-family:inherit;font-size:1.05rem;line-height:1.5;padding:.6em .7em;
   border:1px solid var(--border);border-radius:.5rem;background:var(--paper);color:var(--text);resize:vertical}
 input[type=text]{width:100%;font-family:inherit;font-size:1rem;padding:.55em .7em;border:1px solid var(--border);
   border-radius:.5rem;background:var(--paper);color:var(--text)}
 textarea:focus,input:focus,select:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
 .row{display:flex;align-items:center;gap:.6em .8em;margin-top:.8em;flex-wrap:wrap}
 button{font:600 1rem/1 inherit;padding:.55em 1.1em;border-radius:.5rem;border:1px solid var(--accent);
   background:var(--accent);color:#fff;cursor:pointer}
 button:hover{filter:brightness(1.1)}
 button:disabled{opacity:.55;cursor:default}
 button.btn-quiet{background:var(--paper);color:var(--accent)}
 select{font:inherit;padding:.5em .6em;border:1px solid var(--border);border-radius:.5rem;background:var(--paper);color:var(--text)}
 .muted{color:var(--muted)}
 .small{font-size:.87em}
 .grow{flex:1 1 12rem;min-width:0}

 /* answer: one opaque panel, reading typography, capped at a comfortable measure.
    :not(:empty) keeps it invisible until the first question lands an answer or loading state. */
 #answer:not(:empty){background:var(--paper);border-radius:1rem;box-shadow:var(--shadow);
   padding:1.5rem 1.6rem;margin:0 0 1.5rem}
 #answer{font-family:var(--serif);font-size:1.15rem;line-height:1.65;max-width:100%}
 #answer > *{max-width:70ch}
 #answer > p:first-child,#answer > .loading:first-child{margin-top:0}

 .eyebrow{display:block;font:700 .76rem/1 var(--sans);letter-spacing:.08em;
   text-transform:uppercase;color:var(--muted);margin-bottom:.35em}
 .lbl{font:700 1.05em/1.4 var(--sans);color:var(--text)}
 .g{border-top:1px solid var(--border);padding:1.2em 0}
 .g:first-child{border-top:none;padding-top:0}

 /* the claim: one distinct block, fixed reading order (text, tag, corrector, anchors) */
 .claim{margin:1em 0;padding:.15em 0 .15em 1em;border-left:3px solid var(--border-soft)}
 .claim-text{margin:0 0 .5em;max-width:70ch}
 .claim-meta{display:flex;align-items:center;gap:.5em .7em;flex-wrap:wrap;font:1rem/1.35 var(--sans);margin:0 0 .5em}
 .claim-who{color:var(--muted);font-size:.92em}
 .claim-anchor{font-size:.95rem;margin:0 0 .3em}
 .claim-anchor + .claim-anchor{margin-top:.2em}
 .claim-extra{font-size:.9rem;color:var(--muted);line-height:1.5}
 .claim-extra > div{margin:.25em 0}
 .role-label{font:700 .76rem/1 var(--sans);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}

 /* status tags: opaque flat colour (no gradient) + a glyph + the word — three independent cues,
    so the five stay distinguishable from each other by shape/label even in grayscale or on a
    washed-out projector, not colour alone. */
 .tag{display:inline-flex;align-items:center;gap:.3em;font:700 .72rem/1 var(--sans);letter-spacing:.03em;
   text-transform:uppercase;padding:.34em .6em;border-radius:.3rem;border:1.5px solid transparent;vertical-align:middle;
   white-space:nowrap}
 .tag-current{background:var(--tag-current-bg);color:var(--tag-current-fg);border-color:var(--tag-current-bd);border-style:solid}
 .tag-superseded{background:var(--tag-superseded-bg);color:var(--tag-superseded-fg);border-color:var(--tag-superseded-bd);border-style:dashed}
 .tag-newest{background:var(--tag-newest-bg);color:var(--tag-newest-fg);border-color:var(--tag-newest-bd);border-style:dashed}
 .tag-nevertrue{background:var(--tag-nevertrue-bg);color:var(--tag-nevertrue-fg);border-color:var(--tag-nevertrue-bd);border-style:solid;border-width:2px}
 .tag-withdrawn,.tag-correction{background:var(--tag-superseded-bg);color:var(--tag-superseded-fg);border-color:var(--tag-superseded-bd);border-style:dotted}
 .retraction{font-style:italic}
 .tag-nodirect{background:var(--tag-nodirect-bg);color:var(--tag-nodirect-fg);border-color:var(--tag-nodirect-bd);border-style:dotted;border-width:2px}

 /* the correcting/superseding document: named right beside the tag it belongs to */
 .corrector-label{font-size:.92rem;color:var(--muted);white-space:nowrap}
 .corrector-label.never{color:var(--tag-nevertrue-fg);font-weight:600}

 /* deliberate system statements: distinct from tags, not alarming */
 .sysnote{background:var(--accent-soft);border:1px solid var(--border);border-left:3px solid var(--accent);
   border-radius:.3rem;padding:.7em .85em;margin:.7em 0;font-size:1rem;line-height:1.55;max-width:70ch}
 .sysnote .tag{margin-right:.4em}

 /* real problems: kept visually separate from system statements */
 .error{background:var(--error-bg);border:1px solid var(--error-bd);color:var(--error-fg);border-radius:.3rem;
   padding:.7em .85em;margin:.7em 0;font-size:1rem;max-width:70ch}

 /* loading placeholder: sits exactly where the answer will land */
 .loading{color:var(--muted);font-style:italic;padding:.4em 0}
 .loading::after{content:"";display:inline-block;width:1.1em;text-align:left;
   animation:ellipsis 1.2s steps(4,end) infinite}
 @keyframes ellipsis{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}

 details{margin:.3em 0}
 summary{cursor:pointer;color:var(--accent);font:600 .95rem/1.35 var(--sans);
   display:list-item;overflow-wrap:anywhere}
 summary:hover{text-decoration:underline}
 details > div.cite-meta{color:var(--muted);font-size:.88em;margin:.5em 0 .3em}
 pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--paper-soft);border:1px solid var(--border-soft);border-radius:.5rem;
   padding:.75em .85em;margin:.3em 0;font:1.05rem/1.55 var(--serif);color:var(--text);
   max-height:22em;overflow-y:auto}

 hr.section-break{border:none;border-top:1px solid rgba(25,35,55,.14);margin:2.25rem 0}

 /* deletion: opaque like everything else, just a visually quieter panel — smaller shadow,
    muted heading — clearly a separate region from the answer flow */
 .quiet-panel{background:var(--paper);border-radius:1rem;box-shadow:var(--shadow-quiet);padding:1.15rem 1.4rem}
 .quiet-title{font-size:.95rem;color:var(--muted);font-weight:600;margin:0 0 .5em;text-transform:uppercase;
   letter-spacing:.04em}
 .fineprint{color:var(--muted);font-size:.85rem;line-height:1.5;margin:.8em 0 0;max-width:60ch}
 .confirm-box{background:var(--tag-newest-bg);border:1px solid var(--tag-newest-bd);border-radius:.4rem;
   padding:.8em 1em;margin-top:.8em}
 .confirm-box .row{margin-top:.7em}
 .confirm-result{background:var(--tag-current-bg);border:1px solid var(--tag-current-bd);border-radius:.4rem;
   padding:.85em 1em;margin-top:.8em;font-size:1rem;line-height:1.55}
 .confirm-result ul{margin:.4em 0 0;padding-left:1.2em}
 .del-hint{font-size:.88em}
</style></head><body>
<div class="wrap">

<header class="hero-row">
<h1>relex archive</h1>
<div class="hero-meta">right-to-erasure demo<br>no login</div>
</header>
<hr class="hairline">

<section class="card">
<p class="tagline">Ask a question; the answer is rendered from the claim graph, with sources you can expand.</p>
<label class="field-label" for="q">Ask a question</label>
<textarea id="q" placeholder="e.g. What service levels were agreed for ordering, and in which meeting?"></textarea>
<div class="row"><button id="ask">Ask</button></div>
</section>

<div id="answer"></div>

<hr class="section-break">

<section class="quiet-panel">
<h2 class="quiet-title">Delete a person from the archive</h2>
<div class="row">
  <input id="person" class="grow" type="text" list="people-list" autocomplete="off" placeholder="Type a name…">
  <datalist id="people-list"></datalist>
  <button id="del" class="btn-quiet">Delete</button>
  <span id="delhint" class="muted del-hint"></span>
</div>
<div id="delconfirm" class="confirm-box" hidden><span id="delwho"></span> will be erased from every store. This cannot be undone from the interface.
 <div class="row"><button id="delyes">Confirm delete</button> <button id="delno" class="btn-quiet">Cancel</button></div>
</div>
<p class="fineprint">This runs the deletion pipeline: their statements, statements about them, their registry row, the
retrieval index and embeddings, and every derived chain are rebuilt without them. There is no undo.</p>
<div id="delresult"></div>
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
  return `<span class="muted small">unlinked</span>`;
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
function claim(s){
  const val = s.value!==null&&s.value!==undefined ? `value: ${esc(s.value)}` : (s.truncated ? "value: TRUNCATED IN SOURCE" : "");
  let h = `<div class="claim">`;
  h += `<p class="claim-text">“${esc(s.statement)}”</p>`;
  h += `<div class="claim-meta">` + currencyTag(s.currency);
  if(s.correction){
    h += tag("nevertrue","NEVER TRUE") + `<span class="corrector-label never">${esc(correctorName(s.correction))}</span>` + cite(s.correction_cite, "correcting document");
  } else if(s.supersession){
    h += `<span class="corrector-label">${esc(correctorName(s.supersession))}</span>` + cite(s.supersession_cite, "superseding document");
  }
  h += `<span class="claim-who">${esc(s.asserted_by)} · ${esc(s.date.slice(0,16))}</span></div>`;
  h += `<div class="claim-anchor">` + cite(s.cite, "source") + `</div>`;
  let extra = "";
  if(val) extra += `<div>${esc(val)}</div>`;
  if(s.retraction) extra += `<div class="retraction">${esc(s.retraction)}</div>`;
  if(!s.correction) extra += `<div>${esc(String(s.truth_status).replace(/_/g," "))}</div>`;
  if(s.reported) extra += `<div>status report: evidence of what was reported at the time, not of the underlying state</div>`;
  if(s.order_confidence) extra += `<div>order confidence: ${esc(s.order_confidence)}</div>`;
  if(s.correction){ const q = correctionQuote(s.correction); if(q) extra += `<div>what was actually true: “${esc(q)}”</div>`; }
  else if(s.truth_status==="unverified") extra += `<div>unverified: ${esc(s.truth_reason)}</div>`;
  if(s.responds_to) extra += `<div>in response to: ${esc(s.responds_to)}</div>`;
  if(extra) h += `<div class="claim-extra">${extra}</div>`;
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
    h += `<div class="sysnote">The archive never uses the word(s): ${esc(a.never_mentions.join(", "))}.</div>`;
  if(a.empty) return h + `<div class="sysnote">${esc(a.nothing)}</div>`;
  if(a.scope) h += `<p class="muted small">scoped to ${esc(a.scope)}: retrieval limited to that period; chains with statements then are favoured</p>`;
  const contextOnly = a.groups.length && a.direct === false;
  if(contextOnly) h += `<div class="sysnote">${tag("nodirect","NO DIRECT ANSWER")}No statement in the archive directly answers the question as asked. The chains below are related context, not an answer.</div>`;
  if(a.initiative){
    h += `<div class="g"><span class="eyebrow">agreed and not done</span>`;
    if(!a.initiative.length) h += `<div class="claim muted">no chain in the archive has a commitment followed by a statement that it is outstanding</div>`;
    a.initiative.forEach((t, i) => {
      h += `<div class="lbl">${i+1}. ${esc(t.fact_keys.join(", "))}</div>`;
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
  for(const g of a.groups){
    h += `<div class="g">${contextOnly ? '<span class="eyebrow">related context</span>' : ""}<div class="lbl">${esc(g.fact_keys.join(", "))}</div>`;
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
    for(const s of g.statements) h += claim(s);
    if(g.other.length){
      h += `<p class="muted small">related (proposals / questions, not part of the chain):</p>`;
      for(const s of g.other) h += claim(s);
    }
    h += "</div>";
  }
  const at = a.attribution;
  if(at){
    h += `<div class="g"><span class="eyebrow">attribution</span>`;
    for(const [name, items, emptyMsg] of [
      ["PROPOSED", at.proposed, "No statement in the archive proposed this."],
      ["AGREED / DECIDED", at.agreed, "No statement in the archive agreed to or decided this — a proposal alone is not a decision."],
      ["REJECTED", at.rejected, "No statement in the archive rejected this."],
    ]){
      h += `<div class="lbl">${name}</div>` + (items.length ? items.map(claim).join("") : `<div class="sysnote">${esc(emptyMsg)}</div>`);
    }
    h += `<p class="muted">verdict: ${esc(at.verdict)}</p></div>`;
  }
  return h;
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
askBtn.onclick = async () => {
  const q = qEl.value.trim(); if(!q) return;
  askBtn.disabled = true;
  answerEl.innerHTML = `<div class="loading">Searching the archive</div>`;
  try { const a = await post("/api/ask", {question:q}); answerEl.innerHTML = a.error ? `<div class="error">${esc(a.error)}</div>` : renderAnswer(a); }
  catch(e){ answerEl.innerHTML = `<div class="error">request failed: ${esc(e)}</div>`; }
  askBtn.disabled = false;
  qEl.focus(); qEl.select();   // next question takes no thought: just start typing over this one
};
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
    if(!r.ok){ document.getElementById("delresult").innerHTML = `<div class="error">${esc(r.error)}<br>${esc(r.detail||"")}</div>`; }
    else {
      const s = r.summary;
      document.getElementById("delresult").innerHTML =
        `<div class="confirm-result"><strong>Deleted.</strong> Removed from the archive:<ul>` +
        `<li>${s.units_removed} messages / turns they authored removed; ${s.units_redacted} others that named them redacted</li>` +
        `<li>${s.claims_removed} claims removed (theirs, and others' claims about them)</li>` +
        `<li>${s.chains_lost_head} chains lost their most recent statement: ${s.chains_with_newest_survivor} now show a newest surviving statement, ${s.chains_now_empty} are empty</li>` +
        `<li>retrieval index rebuilt: ${s.passages[0]} → ${s.passages[1]} passages, embeddings re-encoded; claims ${s.claims[0]} → ${s.claims[1]}</li>` +
        `<li>full-text and structural sweep: ${s.verify_clean ? "clean" : "NOT CLEAN — see deletion log"}</li></ul></div>`;
      if(!s.verify_clean) document.getElementById("delresult").innerHTML += `<div class="error">Verification sweep did not come back clean — see the deletion log.</div>`;
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
