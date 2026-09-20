"""Judge-facing UI. Standard library only; one process; no login; no styling beyond legibility.

    python pipeline/serve.py                      # http://localhost:8080
    python pipeline/serve.py --host 0.0.0.0       # reachable from other machines on the network

Three things and nothing else:
  1. Ask a question -> answer.py's rendered claim graph (dates, sources, currency label, truth
     status, correcting document, order confidence, head-removed / removed notices).
  2. Clickable citations -> the human anchor string expands to the unit text stored in
     data/units.jsonl. corpus/ is never read at request time: it holds the unscrubbed original.
  3. A deletion control -> runs the real pipeline/delete.py as a subprocess, then reloads.
  4. A reset control, for between judging runs -> replaces data/ with a copy of relex-data-snapshot/
     (the pre-deletion pipeline output) and reloads in-process, so the port and any tunnel in front
     of it survive. Click only: never on start-up, never on a timer. The snapshot is read, never
     served, listed or written.

No unit id, claim id or turn index reaches the browser: the JSON sent to the page is built from an
allow-list of fields, and anchor strings have the internal-transcript turn number stripped.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOT = ROOT / "relex-data-snapshot"   # pre-deletion output; copied from, never written to or served
STAGING = ROOT / "data.restoring"         # a complete copy of the snapshot, built before data/ is touched
STALE = ROOT / "data.stale"               # the old data/, set aside in one rename and then removed
TURN_INDEX = re.compile(r" turn \d+ \(line")   # "Them turn 22 (line 38)" -> "Them (line 38)"
NOTHING = "The archive does not contain a statement on this."


def anchor(where):
    return TURN_INDEX.sub(" (line", where or "")


def _locked_file(root):
    """Name one file under root that another process holds open, or None. A rename out and back
    fails on Windows exactly when a handle without FILE_SHARE_DELETE is open on it."""
    for f in sorted(root.rglob("*")):
        if f.is_file():
            probe = f.with_name(f.name + ".probe")
            try:
                os.rename(f, probe)
            except OSError:
                return f
            os.rename(probe, f)
    return None


def _rmtree(path):
    """Remove a tree, naming the first file that will not go."""
    for p in sorted(path.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        try:
            p.rmdir() if p.is_dir() else p.unlink()
        except OSError as e:
            raise OSError(f"{p.relative_to(ROOT).as_posix()}: {e.strerror or e}") from None
    path.rmdir()


def restore_data():
    """Replace data/ with relex-data-snapshot/ (never merge: a stale file must not survive).

    data/ is either the old tree or the complete new one at every step: the snapshot is copied to
    a staging directory first, then the two are swapped by rename. A locked file blocks the swap
    before anything is lost and is named in the error. Returns a warning string or None."""
    if not SNAPSHOT.is_dir():
        raise OSError(f"{SNAPSHOT.name}/ is missing; nothing to restore from")
    for leftover in (STAGING, STALE):
        if leftover.exists():
            _rmtree(leftover)
    shutil.copytree(SNAPSHOT, STAGING)
    if DATA.exists():
        try:
            os.rename(DATA, STALE)
        except OSError as e:
            _rmtree(STAGING)
            held = _locked_file(DATA)
            what = f"{held.relative_to(ROOT).as_posix()} is open in another program" if held else (e.strerror or str(e))
            raise OSError(f"{what}; data/ was left as it was") from None
    try:
        os.rename(STAGING, DATA)
    except OSError as e:
        if STALE.exists() and not DATA.exists():
            os.rename(STALE, DATA)
        raise OSError(f"could not move the restored copy into place ({e.strerror or e}); data/ was left as it was") from None
    try:
        if STALE.exists():
            _rmtree(STALE)
    except OSError as e:
        return f"data/ is fully restored, but the previous archive could not be removed: {e}. Delete {STALE.name}/ by hand."
    return None


class State:
    """Answerer + unit store, rebuilt after a deletion or a reset. One lock: either blocks questions."""

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

    def reset(self):
        with self.lock:
            try:
                warning = restore_data()
            except OSError as e:
                return {"ok": False, "error": f"Reset failed: {e}"}
            self.answerer = None
            self.units = None
            self.load()
            claims = sum(1 for l in (DATA / "claims.jsonl").open(encoding="utf-8") if l.strip())
            log = DATA / "deletion_log.jsonl"
            entries = sum(1 for l in log.open(encoding="utf-8") if l.strip()) if log.exists() else 0
            return {"ok": True, "claims": claims, "deletion_log_entries": entries, "warning": warning,
                    "people": self.people()}


STATE = State()

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ledger</title>
<meta name="description" content="Every answer cited. Every person erasable.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@1,700&display=swap">
<style>
 :root{
   --bg:#F8FAFC; --card:#FFFFFF; --fg:#0F172A; --muted:#475569; --muted-bg:#E9EEF5; --line:#CBD5E1; --line-soft:#E2E8F0;
   --primary:#1E3A5F; --primary-hover:#17304F; --secondary:#2563EB; --accent:#16A34A; --accent-hover:#15803D;
   --danger:#DC2626; --danger-hover:#B91C1C; --danger-soft:#FEE2E2; --ring:#1E3A5F;
   --xs:.25rem; --sm:.5rem; --md:1rem; --lg:1.5rem; --xl:2rem; --xxl:3rem;
   --sh-sm:0 1px 2px rgba(0,0,0,.05); --sh-md:0 4px 6px rgba(0,0,0,.1);
   --r:8px; --r-sm:4px;
   --ui:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
   --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
   --display:"Playfair Display",Georgia,"Times New Roman",serif;
   /* stamps: each label is its own colour, border style and glyph */
   --st-current-bg:#DCFCE7;   --st-current-fg:#14532D;   --st-current-bd:#16A34A;
   --st-superseded-bg:#E9EEF5;--st-superseded-fg:#334155;--st-superseded-bd:#94A3B8;
   --st-nevertrue-bg:#FEE2E2; --st-nevertrue-fg:#7F1D1D; --st-nevertrue-bd:#DC2626;
   --st-correction-bg:#DBEAFE;--st-correction-fg:#1E3A8A;--st-correction-bd:#2563EB;
   --st-withdrawn-bg:#F1F5F9; --st-withdrawn-fg:#334155; --st-withdrawn-bd:#64748B;
   --st-newest-bg:#FEF3C7;    --st-newest-fg:#78350F;    --st-newest-bd:#D97706;
   --st-nodirect-bg:#EDE9FE;  --st-nodirect-fg:#4C1D95;  --st-nodirect-bd:#7C3AED;
 }
 *{box-sizing:border-box}
 html{-webkit-text-size-adjust:100%;scroll-padding-top:var(--md)}
 body{margin:0;min-height:100vh;display:flex;flex-direction:column;color:var(--fg);background:var(--bg);font:16px/1.6 var(--ui);overflow-wrap:break-word}
 a{color:var(--secondary)}
 :focus-visible{outline:2px solid var(--ring);outline-offset:2px}
 textarea:focus,input:focus{outline:none;border-color:var(--ring);box-shadow:0 0 0 3px rgba(30,58,95,.13)}
 .sr{position:absolute;left:-999px;top:.5rem;background:var(--card);color:var(--primary);padding:.5em .8em;border-radius:var(--r);box-shadow:var(--sh-md);z-index:30;font-weight:600}
 .sr:focus{left:.75rem}

 /* masthead */
 .masthead{background:var(--card);border-bottom:1px solid var(--line)}
 .masthead-in{max-width:80rem;margin:0 auto;padding:var(--md) var(--lg);display:flex;align-items:baseline;gap:var(--sm) var(--lg);flex-wrap:wrap}
 .wordmark{margin:0;font:italic 700 1.9rem/1 var(--display);color:var(--primary);letter-spacing:-.01em}
 .tagline{margin:0;color:var(--muted);font-size:.95rem}
 .masthead-note{margin-left:auto;color:var(--muted);font-size:.85rem}

 /* workbench: control rail + ledger */
 .workbench{width:100%;max-width:80rem;margin:0 auto;padding:var(--lg);display:grid;grid-template-columns:minmax(17rem,21rem) minmax(0,1fr);gap:var(--xl);align-items:start;flex:1}
 .rail{position:sticky;top:var(--md);display:grid;gap:var(--md);max-height:calc(100vh - var(--xl));overflow-y:auto;padding-right:var(--xs)}
 .ledger{min-width:0}
 @media (max-width:900px){
   .workbench{grid-template-columns:1fr;gap:var(--md);padding:var(--md)}
   .rail{display:contents}
   .panel-ask{order:1}.ledger{order:2}.panel-erase{order:3}
   .masthead-in{padding:var(--md)}
 }

 /* panels */
 .panel{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:var(--lg);box-shadow:var(--sh-sm)}
 .panel-erase{border-top:4px solid var(--danger)}
 .panel h2{margin:0 0 var(--xs);font:600 1.05rem/1.3 var(--ui)}
 .panel .step{display:inline-block;font:500 .7rem/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:var(--sm)}
 .panel p.lede{margin:0 0 var(--md);color:var(--muted);font-size:.92rem}
 label.lbl{display:block;font-weight:600;font-size:.9rem;margin:0 0 var(--xs)}
 textarea,input[type=text]{display:block;width:100%;font:inherit;line-height:1.5;padding:.65em .8em;border:1px solid var(--line);border-radius:var(--r);background:var(--card);color:var(--fg);-webkit-appearance:none;appearance:none;transition:border-color .2s ease,box-shadow .2s ease}
 textarea{min-height:4.6em;resize:none;overflow:hidden}
 input[type=text]{min-height:2.75rem}
 ::placeholder{color:#64748B}
 .row{display:flex;align-items:center;gap:var(--sm) var(--md);margin-top:var(--md);flex-wrap:wrap}
 .hint{color:var(--muted);font-size:.84rem}
 button{font:600 .95rem/1 var(--ui);min-height:2.75rem;padding:.6em 1.2em;border-radius:var(--r);border:1px solid var(--primary);background:var(--primary);color:#fff;cursor:pointer;touch-action:manipulation;transition:background-color .2s ease,border-color .2s ease,color .2s ease}
 button:hover{background:var(--primary-hover);border-color:var(--primary-hover)}
 button:disabled{opacity:.5;cursor:not-allowed}
 button.go{background:var(--accent);border-color:var(--accent);color:#052E16;padding-inline:1.6em}
 button.go:hover{background:var(--accent-hover);border-color:var(--accent-hover);color:#fff}
 button.quiet{background:var(--card);color:var(--primary)}
 button.quiet:hover{background:var(--muted-bg)}
 button.danger{background:var(--danger);border-color:var(--danger)}
 button.danger:hover{background:var(--danger-hover);border-color:var(--danger-hover)}
 button.danger-quiet{background:var(--card);border-color:var(--danger);color:var(--danger)}
 button.danger-quiet:hover{background:var(--danger-soft)}
 button.link{background:none;border:none;color:var(--secondary);text-decoration:underline;padding:.5em .3em;min-height:2.25rem;font-weight:600;font-size:.88rem}
 button.link:hover{background:var(--muted-bg)}
 .chips{list-style:none;margin:var(--md) 0 0;padding:0;display:flex;flex-wrap:wrap;gap:var(--sm);justify-content:center}
 .chip{min-height:2.4rem;padding:.5em .85em;font:500 .88rem/1.35 var(--ui);background:var(--card);color:var(--primary);border:1px solid var(--line);border-radius:var(--r-sm);cursor:pointer;text-align:left;transition:background-color .2s ease,border-color .2s ease}
 .chip:hover{background:var(--muted-bg);border-color:var(--primary)}
 details.people{margin-top:var(--sm)}
 details.people > summary{font-size:.86rem;color:var(--secondary);font-weight:600}
 .people-list{list-style:none;margin:var(--xs) 0 0;padding:0;display:grid;gap:2px;max-height:14rem;overflow-y:auto;border:1px solid var(--line-soft);border-radius:var(--r-sm);padding:var(--xs)}
 .people-list button{width:100%;text-align:left;background:none;border:none;color:var(--fg);font:400 .86rem/1.4 var(--ui);min-height:2rem;padding:.35em .5em;border-radius:var(--r-sm)}
 .people-list button:hover{background:var(--muted-bg)}
 .people-list b{font-weight:600}
 .confirm{background:var(--danger-soft);border:1px solid var(--st-nevertrue-bd);border-radius:var(--r);padding:var(--md);margin-top:var(--md);font-size:.95rem}
 .confirm .row{margin-top:var(--sm)}
 .done{background:var(--st-current-bg);border:1px solid var(--st-current-bd);border-radius:var(--r);padding:var(--md);margin-top:var(--md);font-size:.92rem;line-height:1.5}
 .done ul{margin:var(--xs) 0 0;padding-left:1.1em}
 .done .row{margin-top:var(--sm)}
 .error{background:var(--danger-soft);border:1px solid #FCA5A5;color:#7F1D1D;border-radius:var(--r);padding:.7em .85em;margin:var(--sm) 0;font-size:.95rem}

 /* reset strip: between judging runs, kept apart from the ask and erase controls */
 .foot{border-top:1px solid var(--line);background:var(--card);margin-top:var(--xl)}
 .foot-in{max-width:80rem;margin:0 auto;padding:var(--lg);display:flex;align-items:flex-start;gap:var(--sm) var(--lg);flex-wrap:wrap}
 .foot h2{margin:0 0 var(--xs);font:600 .95rem/1.3 var(--ui)}
 .foot p{margin:0;color:var(--muted);font-size:.88rem;max-width:44rem}
 .foot-ctl{margin-left:auto;display:grid;gap:var(--sm);justify-items:end;min-width:min(100%,22rem)}
 .foot .confirm,.foot .done{margin-top:0;width:100%}
 #resetstatus{font-size:.9rem}
 @media (max-width:900px){.foot-in{padding:var(--md)}.foot-ctl{margin-left:0;justify-items:start;width:100%}}

 /* the ledger column */
 .ledger-head{display:flex;justify-content:space-between;align-items:flex-end;gap:var(--sm) var(--md);flex-wrap:wrap;margin:0 0 var(--md)}
 .ledger-head h1{margin:0;font:600 1.25rem/1.3 var(--ui)}
 .ledger-head .count{color:var(--muted);font:.85rem/1.6 var(--mono)}
 .empty{background:var(--card);border:1px dashed var(--line);border-radius:var(--r);padding:var(--xl) var(--lg);color:var(--muted);text-align:center}
 .empty b{color:var(--fg);display:block;font-size:1.05rem;margin-bottom:var(--xs)}
 details.key{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:0 var(--lg);margin:0 0 var(--md)}
 details.key > summary{padding:.8em 0;font-weight:600;cursor:pointer}
 .key-grid{display:grid;grid-template-columns:auto 1fr;gap:.55em var(--md);align-items:start;margin:0 0 var(--md);font-size:.88rem}
 .key-grid dt{margin:0;display:flex;gap:var(--xs);flex-wrap:wrap}
 .key-grid dd{margin:0;color:var(--muted);line-height:1.45}
 @media (max-width:560px){.key-grid{grid-template-columns:1fr}.key-grid dd{margin-bottom:var(--xs)}}

 /* stamps: word + glyph + border style + colour, never colour alone */
 .stamp{display:inline-flex;align-items:center;gap:.35em;font:700 .7rem/1 var(--ui);letter-spacing:.06em;text-transform:uppercase;padding:.35em .6em;border-radius:var(--r-sm);border:1.5px solid;white-space:nowrap;vertical-align:middle;flex:0 0 auto}
 .stamp svg{width:.8em;height:.8em;flex:none}
 .st-current{background:var(--st-current-bg);color:var(--st-current-fg);border-color:var(--st-current-bd);border-style:solid}
 .st-superseded{background:var(--st-superseded-bg);color:var(--st-superseded-fg);border-color:var(--st-superseded-bd);border-style:dashed}
 .st-nevertrue{background:var(--st-nevertrue-bg);color:var(--st-nevertrue-fg);border-color:var(--st-nevertrue-bd);border-style:solid;border-width:2.5px}
 .st-correction{background:var(--st-correction-bg);color:var(--st-correction-fg);border-color:var(--st-correction-bd);border-style:double;border-width:3px}
 .st-withdrawn{background:var(--st-withdrawn-bg);color:var(--st-withdrawn-fg);border-color:var(--st-withdrawn-bd);border-style:dotted;border-width:2px;text-decoration:line-through;text-decoration-thickness:1px}
 .st-newest{background:var(--st-newest-bg);color:var(--st-newest-fg);border-color:var(--st-newest-bd);border-style:dashed;border-width:2px}
 .st-nodirect{background:var(--st-nodirect-bg);color:var(--st-nodirect-fg);border-color:var(--st-nodirect-bd);border-style:dotted;border-width:2.5px}
 .kind{font:600 .7rem/1 var(--ui);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);border:1px solid var(--line);border-radius:var(--r-sm);padding:.35em .55em;white-space:nowrap}

 /* topics and entries */
 .topic{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:var(--lg);margin:0 0 var(--md);box-shadow:var(--sh-sm)}
 .topic > h2{margin:0 0 var(--sm);font:600 1.05rem/1.35 var(--ui)}
 .eyebrow{display:block;font:500 .7rem/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 var(--sm)}
 h3.sub{margin:var(--md) 0 var(--xs);font:600 .8rem/1.3 var(--ui);letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
 .standing{margin:var(--sm) 0 var(--xs);font:600 .72rem/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--st-current-fg)}
 .standing.newest{color:var(--st-newest-fg)}
 .entry{display:grid;grid-template-columns:7rem minmax(0,1fr);gap:var(--xs) var(--md);padding:var(--md) 0 var(--sm);border-top:1px solid var(--line-soft)}
 .entry:first-child{border-top:none}
 .entry-date{font:.78rem/1.5 var(--mono);color:var(--muted)}
 .entry-date time{display:block;color:var(--fg);font-weight:500}
 .entry-text{margin:0 0 var(--sm);font-size:1rem;line-height:1.55}
 .entry.lead{grid-template-columns:7rem minmax(0,1fr);background:var(--bg);border:1px solid var(--line);border-left:4px solid var(--st-current-bd);border-radius:0 var(--r) var(--r) 0;padding:var(--md)}
 .entry.lead.newest{border-left-color:var(--st-newest-bd)}
 .entry.lead .entry-text{font-size:1.08rem;font-weight:500}
 .entry-status{display:flex;align-items:center;gap:var(--xs) var(--sm);flex-wrap:wrap;margin:0 0 var(--xs);font-size:.86rem;color:var(--muted)}
 .corr{color:var(--muted)} .corr.never{color:var(--st-nevertrue-fg);font-weight:600}
 .entry-source{font-size:.86rem}
 .entry-extra{font-size:.85rem;color:var(--muted);line-height:1.5;margin-top:var(--xs)}
 .entry-extra > div{margin:.15em 0}
 .entry-extra .retraction{font-style:italic}
 details.more{margin-top:var(--xs)} details.more > summary{font-size:.78rem;color:var(--muted);font-weight:500}
 @media (max-width:560px){.entry,.entry.lead{grid-template-columns:1fr}.entry-date{display:flex;gap:var(--sm)}.entry-date time{display:inline}}

 /* disclosures: citations and history */
 details{margin:.2em 0}
 summary{cursor:pointer;color:var(--secondary);font:600 .88rem/1.45 var(--ui);display:list-item;overflow-wrap:anywhere;padding:.25em 0;touch-action:manipulation;transition:color .2s ease}
 summary:hover{color:var(--primary);text-decoration:underline}
 .cite-meta{color:var(--muted);font-size:.84rem;margin:.35em 0 .3em}
 pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--bg);border:1px solid var(--line);border-radius:var(--r-sm);padding:.75em .9em;margin:.3em 0 .5em;font:.95rem/1.55 var(--ui);color:var(--fg);max-height:22em;overflow-y:auto}
 details.history{margin:var(--sm) 0 0;padding-top:var(--xs);border-top:1px dashed var(--line)}
 details.history > summary{padding:.5em 0}
 .note{background:var(--muted-bg);border-left:3px solid var(--primary);border-radius:var(--r-sm);padding:.65em .85em;margin:var(--sm) 0;font-size:.93rem;line-height:1.5}
 .note .stamp{margin-right:.4em}
 .loading{color:var(--muted);padding:var(--md) 0}
 .loading::after{content:"";display:inline-block;width:1.1em;text-align:left;animation:dots 1.2s steps(4,end) infinite}
 @keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}
 @media (prefers-reduced-motion:reduce){*{transition:none !important;animation:none !important;scroll-behavior:auto !important}.loading::after{content:"…"}}
 .muted{color:var(--muted)} .small{font-size:.86rem}
</style></head><body>
<a class="sr" href="#q">Skip to the question</a>

<header class="masthead"><div class="masthead-in">
  <p class="wordmark">Ledger</p>
  <p class="tagline">Every answer cited. Every person erasable.</p>
  <span class="masthead-note">No login. Deletion is real and permanent.</span>
</div></header>

<div class="workbench">
<div class="rail">

<section class="panel panel-ask" aria-labelledby="ask-h">
  <span class="step">Step 1</span>
  <h2 id="ask-h">Ask the archive</h2>
  <label class="lbl" for="q">Your question</label>
  <textarea id="q" aria-describedby="q-hint" placeholder="e.g. What service levels were agreed for ordering, and in which meeting?"></textarea>
  <div class="row"><button id="ask" class="go" type="button">Ask</button><span class="hint" id="q-hint">Ctrl+Enter also asks</span></div>
</section>

<section class="panel panel-erase" aria-labelledby="del-h">
  <span class="step">Step 2, when you are ready</span>
  <h2 id="del-h">Erase a person</h2>
  <p class="lede">Their statements, mentions and every derived record are removed for good. Then ask again and watch what the answer loses.</p>
  <label class="lbl" for="person">Person</label>
  <div class="row" style="margin-top:0">
    <input id="person" type="text" list="people-list" autocomplete="off" placeholder="Type a name…" aria-describedby="delhint" style="flex:1 1 10rem;min-width:0">
    <datalist id="people-list"></datalist>
    <button id="del" class="danger-quiet" type="button">Delete</button>
  </div>
  <span id="delhint" class="hint" role="status"></span>
  <details class="people"><summary id="people-sum">Show everyone in the archive</summary><ul class="people-list" id="people-ul"></ul></details>
  <div id="delconfirm" class="confirm" role="alertdialog" aria-labelledby="delwho" hidden><b id="delwho"></b> will be erased from every store. This cannot be undone.
    <div class="row"><button id="delyes" class="danger" type="button">Confirm delete</button><button id="delno" class="quiet" type="button">Cancel</button></div>
  </div>
  <div id="delresult" role="status" aria-live="polite"></div>
</section>

</div>

<main class="ledger" id="main">
  <div class="ledger-head"><h1 id="ledger-title">The ledger</h1><span class="count" id="ledger-count">nothing asked yet</span></div>
  <details class="key" id="key" open>
    <summary>Key to the labels</summary>
    <dl class="key-grid" id="key-grid"></dl>
  </details>
  <div class="empty" id="empty"><b>Your answer will appear here.</b>Every statement in it is quoted from a stored document you can open. Ask a question, or try one of these:
    <ul class="chips" aria-label="Example questions">
      <li><button type="button" class="chip" data-q="What service levels were agreed for ordering, and in which meeting?">Service levels agreed for ordering</button></li>
      <li><button type="button" class="chip" data-q="How many stores ended up in the pilot group, and who changed the number?">Pilot store count, and who changed it</button></li>
      <li><button type="button" class="chip" data-q="Who proposed taking bakery out of the fresh go-live, and who agreed?">Who took bakery out of the go-live</button></li>
    </ul>
  </div>
  <div id="answer" role="region" aria-label="Answer" aria-live="polite" aria-busy="false"></div>
</main>
</div>

<footer class="foot" aria-labelledby="reset-h">
  <div class="foot-in">
    <div>
      <h2 id="reset-h">Between judging runs</h2>
      <p>Deletions are real. Reset puts the archive back to its pre-deletion state so the next judge starts from the full record. Questions and deletions wait while it runs.</p>
    </div>
    <div class="foot-ctl">
      <button id="reset" class="quiet" type="button">Reset archive</button>
      <div id="resetconfirm" class="confirm" role="alertdialog" aria-labelledby="reset-h" hidden>Every deletion made so far will be undone and the full archive restored.
        <div class="row"><button id="resetyes" type="button">Confirm reset</button><button id="resetno" class="quiet" type="button">Cancel</button></div>
      </div>
      <div id="resetstatus" role="status" aria-live="polite"></div>
    </div>
  </div>
</footer>

<script>
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
// Every label carries a glyph (SVG, decorative) and its own border style in CSS, so the seven
// stay distinguishable from one another without colour.
const GLYPH = {
  current:    '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><circle cx="6" cy="6" r="4.5" fill="currentColor"/></svg>',
  superseded: '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M1.5 6h8M6.5 2.5L10 6l-3.5 3.5" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>',
  nevertrue:  '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M2.5 2.5l7 7M9.5 2.5l-7 7" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
  correction: '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M2 10l1-3 5.5-5.5 2 2L5 9z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>',
  withdrawn:  '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M4.5 3L2 5.5 4.5 8M2 5.5h5a2.5 2.5 0 010 5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>',
  newest:     '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><circle cx="6" cy="6" r="4.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M6 1.5a4.5 4.5 0 010 9z" fill="currentColor"/></svg>',
  nodirect:   '<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M4 4.5a2 2 0 113 1.7c-.7.4-1 .8-1 1.6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="6" cy="9.7" r=".9" fill="currentColor"/></svg>',
};
function stamp(kind, label){ return `<span class="stamp st-${kind}">${GLYPH[kind]||""}${esc(label)}</span>`; }
function currencyStamp(currency){
  if(currency==="CURRENT") return stamp("current", "CURRENT");
  if(currency==="SUPERSEDED") return stamp("superseded", "SUPERSEDED");
  if(currency==="NEWEST SURVIVING") return stamp("newest", "NEWEST SURVIVING");
  if(currency==="WITHDRAWN") return stamp("withdrawn", "WITHDRAWN");
  if(currency==="CORRECTION") return stamp("correction", "CORRECTION");
  return "";
}
const KEY = [
  ["current","CURRENT","The latest statement on this topic that still stands."],
  ["superseded","SUPERSEDED","Was true when said; a later statement replaced it. The replacing document is named beside it."],
  ["nevertrue","NEVER TRUE","Later corrected: wrong when it was said. The correction is quoted."],
  ["correction","CORRECTION","A statement that corrects an earlier one."],
  ["withdrawn","WITHDRAWN","A statement that retracts an earlier one."],
  ["newest","NEWEST SURVIVING","A deletion removed the latest statement; this is the newest one left, and may itself have been out of date."],
  ["nodirect","NO DIRECT ANSWER","Nothing in the archive answers the question as asked; what is shown is related context only."],
];
document.getElementById("key-grid").innerHTML = KEY.map(([k,l,d]) => `<dt>${stamp(k,l)}</dt><dd>${esc(d)}</dd>`).join("");

function cite(c, prefix){
  if(!c) return "";
  const head = c.heading ? esc(c.heading) + " — " : "";
  return `<details><summary>${esc(prefix||"source")}: ${esc(c.label)}</summary>` +
         `<div class="cite-meta">${head}${esc(c.speaker)}, ${esc(c.date)}</div><pre>${esc(c.text)}</pre></details>`;
}
// s.supersession / s.correction are full sentences; the status line needs only who + when.
function correctorName(text){
  const m = /^(superseded by|corrected by) (.+?) on (\d{4}-\d{2}-\d{2})/.exec(text || "");
  return m ? `${m[1]} ${m[2]}, ${m[3]}` : (text || "");
}
function correctionQuote(text){
  const m = /corrected by .+? on \d{4}-\d{2}-\d{2}:\s*"([\s\S]*)"\s*\([^)]*\)\s*$/.exec(text || "");
  return m ? m[1] : "";
}
// One ledger entry: date rail on the left; statement, status line (label + correcting document),
// source, then anything extra on the right. Nothing needs expanding to be seen except source text.
function entry(s, opts){
  opts = opts || {};
  const val = s.value!==null&&s.value!==undefined ? `Value: ${esc(s.value)}` : (s.truncated ? "Value: truncated in the source" : "");
  const cls = "entry" + (opts.lead ? " lead" + (s.currency==="NEWEST SURVIVING" ? " newest" : "") : "");
  let h = `<article class="${cls}">`;
  h += `<div class="entry-date"><time datetime="${esc(s.date)}">${esc(s.date.slice(0,10))}</time><span>${esc(s.asserted_by)}</span></div>`;
  h += `<div class="entry-body">`;
  h += `<p class="entry-text">“${esc(s.statement)}”</p>`;
  const cs = currencyStamp(s.currency);
  h += `<div class="entry-status">` + (opts.role ? `<span class="kind">${esc(opts.role)}</span>` : "") + (cs || (opts.role ? "" : `<span class="kind">${esc(s.kind || "statement")}</span>`));
  if(s.correction){
    h += stamp("nevertrue","NEVER TRUE") + `<span class="corr never">${esc(correctorName(s.correction))}</span>`;
  } else if(s.supersession){
    h += `<span class="corr">${esc(correctorName(s.supersession))}</span>`;
  }
  h += `</div>`;
  if(s.correction) h += `<div class="entry-source">` + cite(s.correction_cite, "correcting document") + `</div>`;
  else if(s.supersession) h += `<div class="entry-source">` + cite(s.supersession_cite, "superseding document") + `</div>`;
  h += `<div class="entry-source">` + cite(s.cite, "source") + `</div>`;
  let extra = "", more = "";
  if(val) extra += `<div>${val}</div>`;
  if(s.retraction) extra += `<div class="retraction">${esc(s.retraction)}</div>`;
  if(s.reported) extra += `<div>From a status report: evidence of what was reported at the time, not of the underlying state.</div>`;
  if(s.correction){ const q = correctionQuote(s.correction); if(q) extra += `<div>What was actually true: “${esc(q)}”</div>`; }
  if(s.responds_to) extra += `<div>In response to: ${esc(s.responds_to)}</div>`;
  if(s.truth_status==="unverified" && s.truth_reason && !String(s.truth_reason).startsWith("kind=")) extra += `<div>Unverified: ${esc(s.truth_reason)}</div>`;
  if(!s.correction) more += `<div>truth status: ${esc(String(s.truth_status).replace(/_/g," "))}</div>`;
  if(s.order_confidence) more += `<div>order confidence: ${esc(s.order_confidence)}</div>`;
  if(extra) h += `<div class="entry-extra">${extra}</div>`;
  if(more) h += `<details class="more"><summary>details</summary><div class="entry-extra">${more}</div></details>`;
  return h + `</div></article>`;
}
function renderAnswer(a){
  let h = "";
  if(a.never_mentions && a.never_mentions.length)
    h += `<div class="note">The archive never uses the word${a.never_mentions.length>1?"s":""} <b>${esc(a.never_mentions.join(", "))}</b>, so nothing here can be matched on ${a.never_mentions.length>1?"them":"it"}.</div>`;
  if(a.empty) return h + `<div class="note">${esc(a.nothing)}</div>`;
  if(a.scope) h += `<p class="muted small">Limited to ${esc(a.scope)}: only statements from that period were searched; topics with statements then come first.</p>`;
  const contextOnly = a.groups.length && a.direct === false;
  if(contextOnly) h += `<div class="note">${stamp("nodirect","NO DIRECT ANSWER")}Nothing in the archive answers this question as asked. Below is the closest related context, shown so you can judge for yourself; it is not an answer.</div>`;
  if(a.initiative){
    h += `<section class="topic"><span class="eyebrow">Agreed and not done</span>`;
    if(!a.initiative.length) h += `<p class="muted">No chain in the archive has a commitment followed by a statement that it is outstanding.</p>`;
    a.initiative.forEach((t, i) => {
      h += `<h2>${i+1}. ${esc(t.fact_keys.join(", "))}</h2>`;
      h += entry(t.agreed, {role:"COMMITTED"});
      h += entry(t.latest, {role:`LATEST ON RECORD (${t.days} days later)`});
      if(t.still_open) h += `<div class="note">No later statement in the archive says it was done; the archive does not record whether it ever was.</div>`;
      else h += entry(t.done_later, {role:"DONE LATER"});
      const who = [`${esc(t.committer)} (made the commitment)`].concat(t.raised_by.map(([n,d]) => `${esc(n)} (raised it on ${esc(d)})`));
      h += `<p class="muted small">Who would have needed to notice: ${who.join("; ")}</p>`;
      h += `<details class="history"><summary>Full trail: ${t.statements.length} statements, oldest first</summary>` + t.statements.map(s => entry(s)).join("") + `</details>`;
    });
    h += `</section>`;
  }
  const attribFirst = !!a.attribution && /(^|[^a-z])who([^a-z]|$)/i.test(a.question || "");
  if(attribFirst) h += attributionBlock(a.attribution);
  for(const g of a.groups){
    h += `<section class="topic">${contextOnly ? '<span class="eyebrow">Related context</span>' : ""}<h2>${esc(g.fact_keys.join(", "))}</h2>`;
    if(g.head_removed){
      const n = g.removed_statements, pl = n===1?"":"s";
      if(g.statements.length){
        const f = g.newest_surviving_figure;
        const what = /\d/.test(String(f.value ?? "")) ? "figure" : "statement";
        const fv = f.value!==null&&f.value!==undefined ? `${esc(f.asserted_by)}'s “${esc(f.value)}”` : `${esc(f.asserted_by)}'s statement`;
        h += `<div class="note">${stamp("newest","NEWEST SURVIVING")}The most recent statement in this chain was removed (${n} statement${pl} removed by deletion). ` +
             `The newest surviving ${what} is ${fv} from ${esc(f.date.slice(0,10))}, and it may have been superseded by a later ` +
             `statement that no longer exists. No statement below is current.</div>` + cite(f.cite, "newest surviving " + what);
      } else {
        h += `<div class="note">The archive no longer contains a statement on this. ${n} statement${pl} existed and ${n===1?"was":"were"} removed by deletion; none survive.</div>`;
      }
    }
    const st = g.statements;
    const leadIdx = st.findIndex(s => s.currency === "CURRENT" || s.currency === "NEWEST SURVIVING");
    const lead = leadIdx >= 0 ? st[leadIdx] : null;
    const rest = lead ? st.filter((_, i) => i !== leadIdx) : st;
    if(lead){
      h += `<p class="standing${lead.currency==="CURRENT" ? "" : " newest"}">${lead.currency === "CURRENT" ? "Standing today" : "Newest statement still in the archive"}</p>` + entry(lead, {lead:true});
    }
    if(rest.length){
      const open = rest.length <= 3 ? " open" : "";
      h += `<details class="history"${open}><summary>${lead ? "How it got here" : "Statements"}: ${rest.length} earlier statement${rest.length===1?"":"s"}, oldest first</summary>` +
           rest.map(s => entry(s)).join("") + `</details>`;
    }
    if(g.other.length){
      h += `<details class="history"><summary>Related proposals and questions (${g.other.length}), not part of the chain</summary>` + g.other.map(s => entry(s)).join("") + `</details>`;
    }
    h += `</section>`;
  }
  if(a.attribution && !attribFirst) h += attributionBlock(a.attribution);
  return h;
}
function attributionBlock(at){
  let h = `<section class="topic"><span class="eyebrow">Who proposed, who agreed</span>`;
  for(const [name, items, emptyMsg] of [
    ["Proposed by", at.proposed, "No statement in the archive proposed this."],
    ["Agreed or decided by", at.agreed, "No statement in the archive agreed to or decided this. A proposal alone is not a decision."],
    ["Rejected by", at.rejected, "No statement in the archive rejected this."],
  ]){
    if(name.startsWith("Rejected") && !items.length) continue;
    h += `<h3 class="sub">${name}</h3>` + (items.length ? items.map((s, i) => entry(s, {lead: i === 0 && name.startsWith("Proposed")})).join("") : `<div class="note">${esc(emptyMsg)}</div>`);
  }
  const verdict = at.verdict.startsWith("nobody agreed") ? "Nobody agreed: the archive holds the proposal but no agreement or decision on it."
                : at.verdict.startsWith("no proposal") ? "No proposal found in the retrieved evidence." : "";
  if(verdict) h += `<div class="note">${esc(verdict)}</div>`;
  return h + `</section>`;
}
async function post(url, body){
  const r = await fetch(url, {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify(body)});
  return r.json();
}
let PEOPLE_BY_LABEL = {};
function personLabel(p){ return p.name + (p.org ? " — " + p.org : "") + (p.role ? ", " + p.role : ""); }
const personEl = document.getElementById("person"), delHint = document.getElementById("delhint");
function fillPeople(list){
  const dl = document.getElementById("people-list"), ul = document.getElementById("people-ul"); dl.innerHTML = ""; ul.innerHTML = "";
  PEOPLE_BY_LABEL = {};
  for(const p of list){
    const label = personLabel(p);
    PEOPLE_BY_LABEL[label] = p.person_id;
    const o = document.createElement("option"); o.value = label; dl.appendChild(o);
    const li = document.createElement("li"), b = document.createElement("button"); b.type = "button";
    b.innerHTML = `<b>${esc(p.name)}</b>${p.org || p.role ? `<span class="muted"> — ${esc([p.org, p.role].filter(Boolean).join(", "))}</span>` : ""}`;
    b.onclick = () => { personEl.value = label; delHint.textContent = ""; document.getElementById("del").focus(); };
    li.appendChild(b); ul.appendChild(li);
  }
  document.getElementById("people-sum").textContent = `Show everyone in the archive (${list.length})`;
}
const qEl = document.getElementById("q"), askBtn = document.getElementById("ask"), answerEl = document.getElementById("answer");
const emptyEl = document.getElementById("empty"), keyEl = document.getElementById("key"), titleEl = document.getElementById("ledger-title"), countEl = document.getElementById("ledger-count");
let lastQuestion = "";
askBtn.onclick = async () => {
  const q = qEl.value.trim(); if(!q) return;
  lastQuestion = q;
  askBtn.disabled = true; answerEl.setAttribute("aria-busy", "true");
  emptyEl.hidden = true; keyEl.open = false;
  titleEl.textContent = q; countEl.textContent = "searching";
  answerEl.innerHTML = `<div class="loading">Searching the archive</div>`;
  try {
    const a = await post("/api/ask", {question:q});
    if(a.error){ answerEl.innerHTML = `<div class="error" role="alert">${esc(a.error)}</div>`; countEl.textContent = "error"; }
    else {
      const n = a.groups ? a.groups.length : 0;
      countEl.innerHTML = (a.empty ? "no matching statements" : `${n} topic${n===1?"":"s"}`) +
        (n ? ` · <button type="button" class="link" id="expand-all">expand all history and sources</button>` : "");
      answerEl.innerHTML = renderAnswer(a);
      const ex = document.getElementById("expand-all");
      if(ex) ex.onclick = () => {
        const all = answerEl.querySelectorAll("details"); const anyClosed = [...all].some(d => !d.open);
        all.forEach(d => d.open = anyClosed); ex.textContent = anyClosed ? "collapse history and sources" : "expand all history and sources";
      };
    }
    document.getElementById("main").scrollIntoView({block:"start", behavior:"smooth"});
  }
  catch(e){ answerEl.innerHTML = `<div class="error" role="alert">The request failed (${esc(e)}). Check that the server is still running, then ask again.</div>`; countEl.textContent = "error"; }
  askBtn.disabled = false; answerEl.setAttribute("aria-busy", "false");
};
function grow(){ qEl.style.height = "auto"; qEl.style.height = qEl.scrollHeight + "px"; }
qEl.addEventListener("input", grow); window.addEventListener("resize", grow); grow();
document.querySelectorAll(".chip").forEach(c => c.onclick = () => { qEl.value = c.dataset.q; grow(); askBtn.click(); });
qEl.addEventListener("keydown", e => { if(e.key === "Enter" && (e.ctrlKey || e.metaKey)) askBtn.click(); });
document.getElementById("del").onclick = () => {
  const pid = PEOPLE_BY_LABEL[personEl.value];
  if(!pid){ delHint.textContent = "Type or pick a full name from the list."; return; }
  delHint.textContent = "";
  document.getElementById("delwho").textContent = personEl.value;
  document.getElementById("delconfirm").hidden = false; document.getElementById("delresult").innerHTML = "";
  document.getElementById("delyes").focus();
};
document.getElementById("delno").onclick = () => { document.getElementById("delconfirm").hidden = true; };
document.getElementById("delyes").onclick = async () => {
  const pid = PEOPLE_BY_LABEL[personEl.value]; if(!pid) return;
  document.getElementById("delconfirm").hidden = true; document.getElementById("delwho").textContent = "";
  delHint.textContent = "Deleting and rebuilding every store (about half a minute)…";
  document.getElementById("del").disabled = true; document.getElementById("delresult").innerHTML = "";
  try {
    const r = await post("/api/delete", {person_id:pid});
    if(!r.ok){ document.getElementById("delresult").innerHTML = `<div class="error" role="alert">${esc(r.error)}<br>${esc(r.detail||"")}</div>`; }
    else {
      const s = r.summary;
      document.getElementById("delresult").innerHTML =
        `<div class="done"><strong>Deleted.</strong> Removed from the archive:<ul>` +
        `<li>${s.units_removed} messages / turns they authored removed; ${s.units_redacted} others that named them redacted</li>` +
        `<li>${s.claims_removed} claims removed (theirs, and others' claims about them)</li>` +
        `<li>${s.chains_lost_head} chains lost their most recent statement: ${s.chains_with_newest_survivor} now show a newest surviving statement, ${s.chains_now_empty} are empty</li>` +
        `<li>search index rebuilt: ${s.passages[0]} → ${s.passages[1]} passages; claims ${s.claims[0]} → ${s.claims[1]}</li>` +
        `<li>full-text and structural sweep: ${s.verify_clean ? "clean" : "NOT CLEAN — see deletion log"}</li></ul>` +
        (lastQuestion ? `<div class="row"><button type="button" id="askagain" class="quiet">Ask your last question again</button></div>` : "") + `</div>`;
      if(!s.verify_clean) document.getElementById("delresult").innerHTML += `<div class="error" role="alert">Verification sweep did not come back clean — see the deletion log.</div>`;
      const again = document.getElementById("askagain");
      if(again) again.onclick = () => { qEl.value = lastQuestion; grow(); askBtn.click(); };
      personEl.value = ""; fillPeople(r.people);
    }
  } catch(e){ document.getElementById("delresult").innerHTML = `<div class="error">request failed: ${esc(e)}</div>`; }
  delHint.textContent = ""; document.getElementById("del").disabled = false;
};
const resetBtn = document.getElementById("reset"), resetStatus = document.getElementById("resetstatus");
resetBtn.onclick = () => { document.getElementById("resetconfirm").hidden = false; resetStatus.innerHTML = ""; document.getElementById("resetyes").focus(); };
document.getElementById("resetno").onclick = () => { document.getElementById("resetconfirm").hidden = true; };
document.getElementById("resetyes").onclick = async () => {
  document.getElementById("resetconfirm").hidden = true;
  resetBtn.disabled = true; document.getElementById("del").disabled = true; askBtn.disabled = true;
  resetStatus.innerHTML = `<div class="loading">Restoring the archive and reloading the index</div>`;
  try {
    const r = await post("/api/reset", {});
    if(!r.ok){ resetStatus.innerHTML = `<div class="error" role="alert">${esc(r.error)}</div>`; }
    else {
      resetStatus.innerHTML = `<div class="done"><strong>Archive reset.</strong> ${r.claims} claims, ${r.deletion_log_entries} deletion log ${r.deletion_log_entries===1?"entry":"entries"}.` +
        (r.warning ? `<div class="error" role="alert">${esc(r.warning)}</div>` : "") + `</div>`;
      personEl.value = ""; delHint.textContent = ""; document.getElementById("delresult").innerHTML = ""; fillPeople(r.people);
    }
  } catch(e){ resetStatus.innerHTML = `<div class="error" role="alert">request failed: ${esc(e)}</div>`; }
  resetBtn.disabled = false; document.getElementById("del").disabled = false; askBtn.disabled = false;
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
            if self.path == "/api/reset":
                return self._send(200, STATE.reset())
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
