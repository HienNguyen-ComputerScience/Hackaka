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
                                     "truth_status", "truth_reason", "order_confidence", "responds_to", "reported")}
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
   --bg:#faf6ee; --paper:#fffdf9; --text:#25201a; --muted:#71675a; --border:#e6dcc7; --border-soft:#efe8d8;
   --accent:#8a4a2b; --accent-soft:#f1e2d4;
   --tag-current-bg:#e2f1e2; --tag-current-fg:#1f6d3d; --tag-current-bd:#b9ddb9;
   --tag-superseded-bg:#ece7dc; --tag-superseded-fg:#5c5548; --tag-superseded-bd:#d7cdb8;
   --tag-newest-bg:#fbecc4; --tag-newest-fg:#7a5300; --tag-newest-bd:#e9cd83;
   --tag-nevertrue-bg:#f8dcd8; --tag-nevertrue-fg:#992a1e; --tag-nevertrue-bd:#eab3ac;
   --tag-nodirect-bg:#e7e0f2; --tag-nodirect-fg:#4f3487; --tag-nodirect-bd:#cdbdea;
   --error-bg:#fbe4e1; --error-bd:#e7b6ae; --error-fg:#8a2a1c;
 }
 *{box-sizing:border-box}
 body{
   margin:0; background:var(--bg); color:var(--text);
   font:16px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;
 }
 .wrap{max-width:42rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
 h1{font-size:1.3rem;letter-spacing:.02em;margin:0 0 .15em}
 .tagline{color:var(--muted);margin:0 0 2rem;font-size:.95rem}
 .card{background:var(--paper);border:1px solid var(--border);border-radius:.6rem;padding:1.25rem 1.4rem;margin:0 0 1.75rem}
 .field-label{display:block;font-weight:600;font-size:.9rem;margin-bottom:.5em}
 textarea{width:100%;min-height:4.5em;font-family:inherit;font-size:15px;line-height:1.5;padding:.6em .7em;
   border:1px solid var(--border);border-radius:.4rem;background:#fff;color:var(--text);resize:vertical}
 textarea:focus,select:focus,button:focus{outline:2px solid var(--accent);outline-offset:1px}
 .row{display:flex;align-items:center;gap:.7em;margin-top:.8em;flex-wrap:wrap}
 button{font:600 .92rem/1 inherit;padding:.55em 1.1em;border-radius:.4rem;border:1px solid var(--accent);
   background:var(--accent);color:#fff;cursor:pointer}
 button:hover{filter:brightness(1.08)}
 button:disabled{opacity:.55;cursor:default}
 button.btn-quiet{background:transparent;color:var(--accent)}
 select{font:inherit;padding:.5em .6em;border:1px solid var(--border);border-radius:.4rem;background:#fff;color:var(--text)}
 .muted{color:var(--muted)}
 .small{font-size:.85em}

 /* answer: reading typography */
 #answer{font-family:Georgia,"Iowan Old Style","Palatino Linotype",Cambria,serif;font-size:17px;line-height:1.7}
 #answer > p:first-child{margin-top:0}

 .eyebrow{display:block;font:600 .74rem/1 -apple-system,"Segoe UI",system-ui,sans-serif;letter-spacing:.08em;
   text-transform:uppercase;color:var(--muted);margin-bottom:.3em}
 .lbl{font:700 1.02em/1.4 -apple-system,"Segoe UI",system-ui,sans-serif;color:var(--text)}
 .g{border-top:1px solid var(--border);padding:1.1em 0}
 .g:first-child{border-top:none;padding-top:0}
 .s{margin:.9em 0 .9em 0;padding-left:1em;border-left:2px solid var(--border-soft)}

 /* status tags: colour + word, never colour alone */
 .tag{display:inline-block;font:700 .68rem/1 -apple-system,"Segoe UI",system-ui,sans-serif;letter-spacing:.03em;
   text-transform:uppercase;padding:.32em .55em;border-radius:.3rem;border:1px solid transparent;vertical-align:middle}
 .tag-current{background:var(--tag-current-bg);color:var(--tag-current-fg);border-color:var(--tag-current-bd)}
 .tag-superseded{background:var(--tag-superseded-bg);color:var(--tag-superseded-fg);border-color:var(--tag-superseded-bd)}
 .tag-newest{background:var(--tag-newest-bg);color:var(--tag-newest-fg);border-color:var(--tag-newest-bd)}
 .tag-nevertrue{background:var(--tag-nevertrue-bg);color:var(--tag-nevertrue-fg);border-color:var(--tag-nevertrue-bd)}
 .tag-nodirect{background:var(--tag-nodirect-bg);color:var(--tag-nodirect-fg);border-color:var(--tag-nodirect-bd)}

 .s-meta{display:flex;align-items:center;gap:.55em;flex-wrap:wrap;font:14px/1.3 -apple-system,"Segoe UI",system-ui,sans-serif}
 .s-date{font-weight:600}
 .s-text{margin:.4em 0}

 /* deliberate system statements: distinct from tags, not alarming */
 .sysnote{background:var(--accent-soft);border:1px solid var(--border);border-left:3px solid var(--accent);
   border-radius:.3rem;padding:.65em .8em;margin:.6em 0;font-size:15px;line-height:1.55}
 .sysnote .tag{margin-right:.4em}

 /* real problems: kept visually separate from system statements */
 .error{background:var(--error-bg);border:1px solid var(--error-bd);color:var(--error-fg);border-radius:.3rem;
   padding:.65em .8em;margin:.6em 0;font-size:15px}

 details{margin:.35em 0}
 summary{cursor:pointer;color:var(--accent);font:600 .88rem/1.3 -apple-system,"Segoe UI",system-ui,sans-serif;
   display:list-item}
 summary:hover{text-decoration:underline}
 details > div.cite-meta{color:var(--muted);font-size:.85em;margin:.4em 0 .2em}
 pre{white-space:pre-wrap;background:#fff;border:1px solid var(--border-soft);border-radius:.35rem;
   padding:.7em .8em;margin:.3em 0;font:15px/1.55 Georgia,"Iowan Old Style",serif;color:var(--text)}

 hr.section-break{border:none;border-top:1px solid var(--border);margin:2.5rem 0}

 /* deletion: visually quieter than the answer */
 .quiet-panel{background:transparent;border:1px solid var(--border-soft);border-radius:.6rem;padding:1.1rem 1.3rem}
 .quiet-title{font-size:.95rem;color:var(--muted);font-weight:600;margin:0 0 .3em;text-transform:uppercase;
   letter-spacing:.04em}
 .fineprint{color:var(--muted);font-size:.83rem;line-height:1.5;margin:.8em 0 0}
 .confirm-box{background:var(--tag-newest-bg);border:1px solid var(--tag-newest-bd);border-radius:.4rem;
   padding:.8em 1em;margin-top:.8em}
 .confirm-box .row{margin-top:.7em}
 .confirm-result{background:var(--tag-current-bg);border:1px solid var(--tag-current-bd);border-radius:.4rem;
   padding:.8em 1em;margin-top:.8em;font-size:14.5px;line-height:1.55}
 .confirm-result ul{margin:.4em 0 0;padding-left:1.2em}
</style></head><body>
<div class="wrap">

<header>
<h1>relex archive</h1>
<p class="tagline">Ask a question; the answer is rendered from the claim graph, with sources you can expand.</p>
</header>

<section class="card">
<label class="field-label" for="q">Ask a question</label>
<textarea id="q" placeholder="e.g. What service levels were agreed for ordering, and in which meeting?"></textarea>
<div class="row"><button id="ask">Ask</button> <span id="askstate" class="muted small"></span></div>
</section>

<div id="answer"></div>

<hr class="section-break">

<section class="quiet-panel">
<h2 class="quiet-title">Delete a person from the archive</h2>
<div class="row"><select id="person"></select> <button id="del" class="btn-quiet">Delete</button> <span id="delstate" class="muted small"></span></div>
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
function tag(kind, label){
  return `<span class="tag tag-${kind}">${esc(label)}</span>`;
}
function currencyTag(currency){
  if(currency==="CURRENT") return tag("current", "CURRENT");
  if(currency==="SUPERSEDED") return tag("superseded", "SUPERSEDED");
  if(currency==="NEWEST SURVIVING") return tag("newest", "NEWEST SURVIVING");
  return `<span class="muted small">unlinked</span>`;
}
function cite(c, prefix){
  if(!c) return "";
  const head = c.heading ? esc(c.heading) + " — " : "";
  return `<details><summary>${esc(prefix||"cite")}: ${esc(c.label)}</summary>` +
         `<div class="cite-meta">${head}${esc(c.speaker)}, ${esc(c.date)}</div><pre>${esc(c.text)}</pre></details>`;
}
function statement(s){
  const val = s.value!==null&&s.value!==undefined ? ` · value: ${esc(s.value)}` : (s.truncated ? " · value: TRUNCATED IN SOURCE" : "");
  let h = `<div class="s"><div class="s-meta"><span class="s-date">${esc(s.date.slice(0,16))}</span> · ${esc(s.asserted_by)} · ` +
          `${currencyTag(s.currency)} · <span class="muted">${esc(String(s.truth_status).replace(/_/g," "))}</span>${val}</div>`;
  h += `<div class="s-text">“${esc(s.statement)}”</div>` + cite(s.cite);
  if(s.reported) h += `<div class="muted small">status report: evidence of what was reported at the time, not of the underlying state</div>`;
  if(s.supersession) h += `<div class="muted small">${esc(s.supersession)}</div>` + cite(s.supersession_cite, "superseding document");
  if(s.order_confidence) h += `<div class="muted small">order confidence: ${esc(s.order_confidence)}</div>`;
  if(s.correction) h += `<div class="sysnote">${tag("nevertrue","NEVER TRUE")}${esc(s.correction)}</div>` + cite(s.correction_cite, "correcting document");
  else if(s.truth_status==="unverified") h += `<div class="muted small">unverified: ${esc(s.truth_reason)}</div>`;
  if(s.responds_to) h += `<div class="muted small">in response to: ${esc(s.responds_to)}</div>`;
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
    if(!a.initiative.length) h += `<div class="s muted">no chain in the archive has a commitment followed by a statement that it is outstanding</div>`;
    a.initiative.forEach((t, i) => {
      const ag = t.agreed, l = t.latest;
      h += `<div class="s"><div class="eyebrow">${i+1}. ${esc(t.fact_keys.join(", "))}</div>` +
           `<div>COMMITTED: ${esc(ag.date.slice(0,10))} · ${esc(ag.asserted_by)} · “${esc(ag.statement)}”</div>` + cite(ag.cite) +
           `<div>LATEST ON RECORD (${t.days} days later): ${esc(l.date.slice(0,10))} · ${esc(l.asserted_by)} · “${esc(l.statement)}”</div>` + cite(l.cite);
      if(t.still_open) h += `<div class="sysnote">No later statement in the archive says it was done; the archive does not record whether it ever was.</div>`;
      else h += `<div>Later reported done: ${esc(t.done_later.date.slice(0,10))} · ${esc(t.done_later.asserted_by)} · “${esc(t.done_later.statement)}”</div>` + cite(t.done_later.cite);
      const who = [`${esc(t.committer)} (made the commitment)`].concat(t.raised_by.map(([n,d]) => `${esc(n)} (raised it on ${esc(d)})`));
      h += `<div class="muted small">Who would have needed to notice: ${who.join("; ")}</div>`;
      h += `<details><summary>full trail (${t.statements.length} statements)</summary>` + t.statements.map(statement).join("") + `</details></div>`;
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
    for(const s of g.statements) h += statement(s);
    if(g.other.length){
      h += `<p class="muted small">related (proposals / questions, not part of the chain):</p>`;
      for(const s of g.other) h += statement(s);
    }
    h += "</div>";
  }
  const at = a.attribution;
  if(at){
    h += `<div class="g"><span class="eyebrow">attribution</span>`;
    for(const [name, items] of [["PROPOSED", at.proposed], ["AGREED / DECIDED", at.agreed], ["REJECTED", at.rejected]]){
      h += `<div class="lbl">${name}</div>` + (items.length ? items.map(statement).join("") : `<div class="s muted">(none)</div>`);
    }
    h += `<p class="muted">verdict: ${esc(at.verdict)}</p></div>`;
  }
  return h;
}
async function post(url, body){
  const r = await fetch(url, {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify(body)});
  return r.json();
}
function fillPeople(list){
  const sel = document.getElementById("person"); sel.innerHTML = "";
  for(const p of list){
    const o = document.createElement("option"); o.value = p.person_id;
    o.textContent = p.name + (p.org ? " — " + p.org : "") + (p.role ? ", " + p.role : "");
    sel.appendChild(o);
  }
}
document.getElementById("ask").onclick = async () => {
  const q = document.getElementById("q").value.trim(); if(!q) return;
  document.getElementById("askstate").textContent = "…"; document.getElementById("answer").innerHTML = "";
  try { const a = await post("/api/ask", {question:q}); document.getElementById("answer").innerHTML = a.error ? `<div class="error">${esc(a.error)}</div>` : renderAnswer(a); }
  catch(e){ document.getElementById("answer").innerHTML = `<div class="error">request failed: ${esc(e)}</div>`; }
  document.getElementById("askstate").textContent = "";
};
document.getElementById("del").onclick = () => {
  const sel = document.getElementById("person"); if(!sel.value) return;
  document.getElementById("delwho").textContent = sel.options[sel.selectedIndex].textContent;
  document.getElementById("delconfirm").hidden = false; document.getElementById("delresult").innerHTML = "";
};
document.getElementById("delno").onclick = () => { document.getElementById("delconfirm").hidden = true; };
document.getElementById("delyes").onclick = async () => {
  const sel = document.getElementById("person"); const pid = sel.value; if(!pid) return;
  document.getElementById("delconfirm").hidden = true; document.getElementById("delwho").textContent = "";
  document.getElementById("delstate").textContent = "deleting and rebuilding every store (about half a minute)…";
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
      fillPeople(r.people);
    }
  } catch(e){ document.getElementById("delresult").innerHTML = `<div class="error">request failed: ${esc(e)}</div>`; }
  document.getElementById("delstate").textContent = ""; document.getElementById("del").disabled = false;
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
