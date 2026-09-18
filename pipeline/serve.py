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
<html lang="en"><head><meta charset="utf-8"><title>relex archive</title>
<style>
 body{font:15px/1.45 system-ui,sans-serif;max-width:60em;margin:1.5em auto;padding:0 1em;color:#111;background:#fff}
 textarea{width:100%;height:4em;font:inherit}
 button{font:inherit;padding:.3em .9em}
 fieldset{margin:1.2em 0;padding:.8em 1em}
 .g{border-top:1px solid #999;padding:.6em 0}
 .s{margin:.7em 0 .7em 1em}
 .lbl{font-weight:600}
 .cur{background:#dfd}.sup{background:#eee}.nsv{background:#ffd}
 .note{background:#fee;padding:.4em .6em;margin:.4em 0}
 details{margin:.2em 0 .2em 1em}summary{cursor:pointer;color:#036}
 pre{white-space:pre-wrap;background:#f6f6f6;padding:.5em;margin:.3em 0}
 .muted{color:#555}
</style></head><body>
<h1>relex archive</h1>

<fieldset><legend>Ask a question</legend>
<textarea id="q" placeholder="e.g. What service levels were agreed for ordering, and in which meeting?"></textarea>
<p><button id="ask">Ask</button> <span id="askstate" class="muted"></span></p>
<div id="answer"></div>
</fieldset>

<fieldset><legend>Delete a person from the archive</legend>
<p><select id="person"></select> <button id="del">Delete</button> <span id="delstate" class="muted"></span></p>
<p id="delconfirm" class="note" hidden><span id="delwho"></span> will be erased from every store. This cannot be undone from the interface.
 <button id="delyes">Confirm delete</button> <button id="delno">Cancel</button></p>
<p class="muted">This runs the deletion pipeline: their statements, statements about them, their registry row, the
retrieval index and embeddings, and every derived chain are rebuilt without them. There is no undo.</p>
<div id="delresult"></div>
</fieldset>

<script>
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function cite(c, prefix){
  if(!c) return "";
  const head = c.heading ? esc(c.heading) + " — " : "";
  return `<details><summary>${esc(prefix||"cite")}: ${esc(c.label)}</summary>` +
         `<div class="muted">${head}${esc(c.speaker)}, ${esc(c.date)}</div><pre>${esc(c.text)}</pre></details>`;
}
function statement(s){
  const cls = s.currency==="CURRENT"?"cur":(s.currency==="SUPERSEDED"?"sup":(s.currency==="NEWEST SURVIVING"?"nsv":""));
  const val = s.value!==null&&s.value!==undefined ? ` · value: ${esc(s.value)}` : (s.truncated ? " · value: TRUNCATED IN SOURCE" : "");
  let h = `<div class="s"><div><span class="lbl">${esc(s.date.slice(0,16))}</span> · ${esc(s.asserted_by)} · ` +
          `<span class="lbl ${cls}">[${esc(s.currency||"unlinked")}]</span> · ${esc(s.truth_status)}${val}</div>`;
  h += `<div>“${esc(s.statement)}”</div>` + cite(s.cite);
  if(s.reported) h += `<div class="muted">status report: evidence of what was reported at the time, not of the underlying state</div>`;
  if(s.supersession) h += `<div class="muted">${esc(s.supersession)}</div>` + cite(s.supersession_cite, "superseding document");
  if(s.order_confidence) h += `<div class="muted">order confidence: ${esc(s.order_confidence)}</div>`;
  if(s.correction) h += `<div class="note">NEVER TRUE: ${esc(s.correction)}</div>` + cite(s.correction_cite, "correcting document");
  else if(s.truth_status==="unverified") h += `<div class="muted">unverified: ${esc(s.truth_reason)}</div>`;
  if(s.responds_to) h += `<div class="muted">in response to: ${esc(s.responds_to)}</div>`;
  return h + "</div>";
}
function renderAnswer(a){
  let h = "";
  if(a.never_mentions && a.never_mentions.length)
    h += `<p class="note">The archive never uses the word(s): ${esc(a.never_mentions.join(", "))}.</p>`;
  if(a.empty) return h + `<p class="note">${esc(a.nothing)}</p>`;
  if(a.scope) h += `<p class="muted">scoped to ${esc(a.scope)}: retrieval limited to that period; chains with statements then are favoured</p>`;
  const contextOnly = a.groups.length && a.direct === false;
  if(contextOnly) h += `<p class="note">NO DIRECT ANSWER: no statement in the archive directly answers the question as asked. The chains below are related context, not an answer.</p>`;
  if(a.initiative){
    h += `<div class="g"><div class="lbl">agreed and not done</div>`;
    if(!a.initiative.length) h += `<div class="s muted">no chain in the archive has a commitment followed by a statement that it is outstanding</div>`;
    a.initiative.forEach((t, i) => {
      const ag = t.agreed, l = t.latest;
      h += `<div class="s"><div class="lbl">${i+1}. ${esc(t.fact_keys.join(", "))}</div>` +
           `<div>COMMITTED: ${esc(ag.date.slice(0,10))} · ${esc(ag.asserted_by)} · “${esc(ag.statement)}”</div>` + cite(ag.cite) +
           `<div>LATEST ON RECORD (${t.days} days later): ${esc(l.date.slice(0,10))} · ${esc(l.asserted_by)} · “${esc(l.statement)}”</div>` + cite(l.cite);
      if(t.still_open) h += `<div class="note">No later statement in the archive says it was done; the archive does not record whether it ever was.</div>`;
      else h += `<div>Later reported done: ${esc(t.done_later.date.slice(0,10))} · ${esc(t.done_later.asserted_by)} · “${esc(t.done_later.statement)}”</div>` + cite(t.done_later.cite);
      const who = [`${esc(t.committer)} (made the commitment)`].concat(t.raised_by.map(([n,d]) => `${esc(n)} (raised it on ${esc(d)})`));
      h += `<div>Who would have needed to notice: ${who.join("; ")}</div>`;
      h += `<details><summary>full trail (${t.statements.length} statements)</summary>` + t.statements.map(statement).join("") + `</details></div>`;
    });
    h += `</div>`;
  }
  for(const g of a.groups){
    h += `<div class="g"><div class="lbl">${contextOnly ? "related context: " : ""}${esc(g.fact_keys.join(", "))}</div>`;
    if(g.head_removed){
      const n = g.removed_statements, pl = n===1?"":"s";
      if(g.statements.length){
        const f = g.newest_surviving_figure;
        const isFig = /\d/.test(String(f.value ?? ""));
        const what = isFig ? "figure" : "statement";
        const fv = f.value!==null&&f.value!==undefined ? `${esc(f.asserted_by)}'s “${esc(f.value)}”` : `${esc(f.asserted_by)}'s statement`;
        h += `<div class="note">The most recent statement in this chain was removed (${n} statement${pl} removed by deletion). ` +
             `The newest surviving ${what} is ${fv} from ${esc(f.date.slice(0,10))}, and it may have been superseded by a later ` +
             `statement that no longer exists. No statement below is current.</div>` + cite(f.cite, "newest surviving " + what);
      } else {
        h += `<div class="note">The archive no longer contains a statement on this. ${n} statement${pl} existed and ` +
             `${n===1?"was":"were"} removed by deletion; none survive.</div>`;
      }
    }
    for(const s of g.statements) h += statement(s);
    if(g.other.length){
      h += `<div class="muted">related (proposals / questions, not part of the chain):</div>`;
      for(const s of g.other) h += statement(s);
    }
    h += "</div>";
  }
  const at = a.attribution;
  if(at){
    h += `<div class="g"><div class="lbl">attribution</div>`;
    for(const [name, items] of [["PROPOSED", at.proposed], ["AGREED / DECIDED", at.agreed], ["REJECTED", at.rejected]]){
      h += `<div class="lbl">${name}</div>` + (items.length ? items.map(statement).join("") : `<div class="s muted">(none)</div>`);
    }
    h += `<div>verdict: ${esc(at.verdict)}</div></div>`;
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
  try { const a = await post("/api/ask", {question:q}); document.getElementById("answer").innerHTML = a.error ? `<p class="note">${esc(a.error)}</p>` : renderAnswer(a); }
  catch(e){ document.getElementById("answer").innerHTML = `<p class="note">request failed: ${esc(e)}</p>`; }
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
    if(!r.ok){ document.getElementById("delresult").innerHTML = `<p class="note">${esc(r.error)}<br>${esc(r.detail||"")}</p>`; }
    else {
      const s = r.summary;
      document.getElementById("delresult").innerHTML =
        `<p class="note">Deleted. Removed from the archive:</p><ul>` +
        `<li>${s.units_removed} messages / turns they authored removed; ${s.units_redacted} others that named them redacted</li>` +
        `<li>${s.claims_removed} claims removed (theirs, and others' claims about them)</li>` +
        `<li>${s.chains_lost_head} chains lost their most recent statement: ${s.chains_with_newest_survivor} now show a newest surviving statement, ${s.chains_now_empty} are empty</li>` +
        `<li>retrieval index rebuilt: ${s.passages[0]} → ${s.passages[1]} passages, embeddings re-encoded; claims ${s.claims[0]} → ${s.claims[1]}</li>` +
        `<li>full-text and structural sweep: ${s.verify_clean ? "clean" : "NOT CLEAN — see deletion log"}</li></ul>`;
      fillPeople(r.people);
    }
  } catch(e){ document.getElementById("delresult").innerHTML = `<p class="note">request failed: ${esc(e)}</p>`; }
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
