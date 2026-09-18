"""Person registry + mention index.

Reads data/units.jsonl, writes:
  data/people.json     person_id -> canonical name, org, role, aliases
  data/mentions.jsonl  one row per (unit_id, person_id, role, evidence)

Roles: speaker | to | cc | attendee | text_mention | signature_mention
Unresolved speaker labels (Me, Them, Guest 1, Unknown Speaker, phone numbers) get their
own person_id prefixed 'unresolved:' so deletion can report them instead of ignoring them.
"""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# canonical registry: README people + names that appear in bodies
REGISTRY = [
    # name, org, role
    ("Lena Fischer", "Acme", "Head of Supply Chain"),
    ("Robert Kahn", "Acme", "CFO"),
    ("Sofia Almeida", "Acme", "Category Manager, Fresh"),
    ("Priya Nair", "Acme", "IT Integration Lead"),
    ("Jonas Weiss", "Acme", "Category Manager, Ambient"),
    ("Katarina Voss", "Acme", "Data Protection Officer"),
    ("Marco Rossi", "RELEX", "Account Executive"),
    ("Ana Duarte", "RELEX", "Project Manager"),
    ("Nadia Haddad", "RELEX", "Solution Consultant"),
    ("Tomas Lindholm", "RELEX", "Solution Architect"),
    ("Charlotte Meyer", "RELEX", "Service Delivery"),
    ("Henrik Sørensen", "RELEX", "Account Director"),
    ("Ivan Petrov", "Meridian Consulting", None),
    ("Ruth Oyelaran", "Meridian Consulting", None),
    # people who only appear inside message bodies
    ("Nils Ackermann", "Acme", "warehouse system owner (logistics IT)"),
    ("Osman Yildirim", "Acme store staff", None),
    ("Marika Lindqvist", "Acme store staff", None),
    ("Heidi Salminen", "Acme store staff", None),
    ("Elin Bergqvist", None, None),
]
UNRESOLVED_SPEAKERS = re.compile(r"^(Me|Them|Guest \d+|Unknown Speaker|\+\d[\d ]+)$")


def fold(s):
    """ASCII-fold for alias matching (Sørensen -> Sorensen)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.replace("ø", "o").replace("Ø", "O"))
                   if not unicodedata.combining(c))


def pid(name):
    return "person:" + re.sub(r"[^a-z]+", "-", fold(name).lower()).strip("-")


def build_registry(units, exclude=()):
    """exclude: person_ids that must not be registered (delete.py). Their REGISTRY entry is skipped and
    any header/attendee name that resolves to them is not harvested, so no row, alias or e-mail survives."""
    exclude = set(exclude)
    people = {}
    for name, org, role in REGISTRY:
        if pid(name) in exclude:
            continue
        first, last = name.split()[0], name.split()[-1]
        people[pid(name)] = {
            "person_id": pid(name), "name": name, "org": org, "role": role,
            "aliases": sorted({name, fold(name), first, last, fold(last)}),
            "emails": set(), "initials": "".join(w[0] for w in name.split()),
        }
    # harvest e-mail addresses and any header/attendee names not yet registered
    for u in units:
        cands = []
        if u["anchor"]["kind"] == "email":
            cands.append((u["speaker"], u.get("speaker_email")))
            for r in u["recipients"]["to"] + u["recipients"]["cc"]:
                cands.append((r["name"], r["email"]))
        else:
            cands += [(a["name"], None) for a in u["attendees"]]
            cands.append((u["speaker"], None))
        for name, email in cands:
            if UNRESOLVED_SPEAKERS.match(name):
                key = f"unresolved:{name}"
                people.setdefault(key, {"person_id": key, "name": name, "org": None, "role": "UNRESOLVED speaker label",
                                        "aliases": [name], "emails": set(), "initials": None})
                continue
            key = pid(name)
            if key in exclude:
                continue
            if key not in people:
                people[key] = {"person_id": key, "name": name, "org": None, "role": None,
                               "aliases": sorted({name, fold(name), name.split()[0], name.split()[-1]}),
                               "emails": set(), "initials": "".join(w[0] for w in name.split())}
            if email:
                people[key]["emails"].add(email)
    for p in people.values():
        p["emails"] = sorted(p["emails"])
    return people


def alias_lookup(people):
    """alias string -> person_id, for header names (exact, folded)."""
    lk = {}
    for p in people.values():
        for a in p["aliases"]:
            lk[fold(a).lower()] = p["person_id"]
        for e in p["emails"]:
            lk[e.lower()] = p["person_id"]
    return lk


def text_patterns(people):
    """Compiled regexes for text mentions: full name, surname, first name, email."""
    pats = []
    for p in people.values():
        if p["person_id"].startswith("unresolved:"):
            continue
        parts = p["name"].split()
        variants = {p["name"], fold(p["name"]), parts[-1], fold(parts[-1]), parts[0]}
        alt = "|".join(sorted((re.escape(v) for v in variants), key=len, reverse=True))
        pats.append((p["person_id"], re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])")))
        for e in p["emails"]:
            pats.append((p["person_id"], re.compile(re.escape(e), re.I)))
    return pats


def resolve(name, lk):
    if UNRESOLVED_SPEAKERS.match(name):
        return f"unresolved:{name}"
    return lk.get(fold(name).lower())


def build_mentions(units, people):
    lk = alias_lookup(people)
    pats = text_patterns(people)
    rows = []
    for u in units:
        seen = set()

        def add(person_id, role, evidence):
            if person_id and (person_id, role) not in seen:
                seen.add((person_id, role))
                rows.append({"unit_id": u["unit_id"], "person_id": person_id, "role": role, "evidence": evidence})

        add(resolve(u["speaker"], lk), "speaker", u["speaker"])
        if u["anchor"]["kind"] == "email":
            for r in u["recipients"]["to"]:
                add(resolve(r["name"], lk), "to", r["name"])
            for r in u["recipients"]["cc"]:
                add(resolve(r["name"], lk), "cc", r["name"])
        else:
            for a in u["attendees"]:
                add(resolve(a["name"], lk), "attendee", a["name"])
        for person_id, pat in pats:
            m = pat.search(u["text"])
            if m:
                s = max(0, m.start() - 30)
                add(person_id, "text_mention", u["text"][s:m.end() + 30].replace("\n", " "))
            sig = u.get("signature") or ""
            if sig and pat.search(sig) and person_id != resolve(u["speaker"], lk):
                add(person_id, "signature_mention", sig[:60])
    return rows


def main():
    units = [json.loads(l) for l in (DATA / "units.jsonl").open(encoding="utf-8")]
    people = build_registry(units)
    rows = build_mentions(units, people)
    (DATA / "people.json").write_text(json.dumps(people, ensure_ascii=False, indent=1), encoding="utf-8")
    with (DATA / "mentions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per[r["person_id"]][r["role"]] += 1
    print(f"people: {len(people)}   mention rows: {len(rows)}")
    for k in sorted(per, key=lambda k: -sum(per[k].values())):
        print(f"  {people[k]['name']:24s} " + "  ".join(f"{r}={n}" for r, n in sorted(per[k].items())))


if __name__ == "__main__":
    main()
