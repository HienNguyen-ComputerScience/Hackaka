"""Splitter: corpus files -> one JSON record per atomic unit.

Atomic units:
  email_message    one message in an email thread (threads are reverse-chronological)
  report_weekly    one weekly update in reports/01 (same email structure, plus a week label)
  report_monthly   one monthly steering report in reports/02 (plus a month label)
  transcript_turn  one speaker turn in a Teams transcript, consecutive same-speaker
                   segments merged; INTERNAL laptop transcripts use Me:/Them: lines.

Anchors resolve to 1-based line numbers in the source file (line_start/line_end).
"""
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "corpus" / "acme"
OUT = ROOT / "data" / "units.jsonl"

# ---------------------------------------------------------------- helpers
MONTHS = {
    # english
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    # german
    "januar": 1, "februar": 2, "märz": 3, "mai": 5, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
    # swedish
    "januari": 1, "februari": 2, "mars": 3, "maj": 5, "augusti": 8,
}
DATE_PATTERNS = [
    # Monday, November 24, 2025 17:55  /  ... 11:44 AM
    re.compile(r"^\w+,\s+(?P<mon>\w+)\s+(?P<d>\d{1,2}),\s+(?P<y>\d{4})\s+(?P<h>\d{1,2}):(?P<mi>\d{2})\s*(?P<ampm>AM|PM)?$"),
    # Donnerstag, 30. April 2026 09:20
    re.compile(r"^\w+,\s+(?P<d>\d{1,2})\.\s+(?P<mon>\w+)\s+(?P<y>\d{4})\s+(?P<h>\d{1,2}):(?P<mi>\d{2})$"),
    # den 20 augusti 2024 09:05
    re.compile(r"^den\s+(?P<d>\d{1,2})\s+(?P<mon>\w+)\s+(?P<y>\d{4})\s+(?P<h>\d{1,2}):(?P<mi>\d{2})$"),
]


def parse_date(s):
    s = s.strip()
    for pat in DATE_PATTERNS:
        m = pat.match(s)
        if m:
            mon = MONTHS[m["mon"].lower()]
            h = int(m["h"])
            mi = int(m["mi"])
            ampm = m.groupdict().get("ampm")
            if ampm == "PM" and h != 12:
                h += 12
            if ampm == "AM" and h == 12:
                h = 0
            return datetime(int(m["y"]), mon, int(m["d"]), h, mi).strftime("%Y-%m-%dT%H:%M")
    raise ValueError(f"unparsed date: {s!r}")


def parse_people(s):
    """'Name <addr>; Name <addr>' -> [{'name','email'}]"""
    out = []
    for part in re.split(r";\s*", s.strip()):
        if not part:
            continue
        m = re.match(r"^(?P<name>.*?)\s*<(?P<email>[^>]+)>$", part)
        if m:
            out.append({"name": m["name"].strip(), "email": m["email"].strip()})
        else:
            out.append({"name": part.strip(), "email": None})
    return out


# image placeholders: whole-line and inline forms, incl. the Swedish one
PLACEHOLDER_INLINE = re.compile(r"\s*(\[Image removed by sender\]|\[cid:image\d+\.\w+\]|Bild borttagen av avsändaren\.?)")
PLACEHOLDER_BARE_LINE = re.compile(r"^\s*Image\s*$")
PLACEHOLDER_TRAILING_IMAGE = re.compile(r"(?<=\S)\s+Image$")   # 'Project Manager Image'
EXTERNAL_BANNER = "This email originated from outside of RELEX."


def strip_placeholders(lines):
    out = []
    for l in lines:
        if PLACEHOLDER_BARE_LINE.match(l):
            continue
        l = PLACEHOLDER_INLINE.sub("", l)
        l = PLACEHOLDER_TRAILING_IMAGE.sub("", l)
        if l.strip() == "":
            l = ""
        out.append(l)
    return out


def skip_synthetic_header(lines):
    """Index (0-based) of the first line after the leading '*** ...' block and its blank."""
    i = 0
    while i < len(lines) and lines[i].startswith("***"):
        i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return i


def squeeze_blank(lines):
    out = []
    for l in lines:
        if l == "" and out and out[-1] == "":
            continue
        out.append(l)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


# ---------------------------------------------------------------- emails / reports
HDR_KEYS = {
    "from": ("From", "Von", "Från"),
    "sent": ("Sent", "Date", "Gesendet", "Skickat"),
    "to": ("To", "An", "Till"),
    "cc": ("Cc", "Kopia"),
    "subject": ("Subject", "Betreff", "Ämne"),
    "count": ("Messages in thread",),
}
KEY_LOOKUP = {k: canon for canon, ks in HDR_KEYS.items() for k in ks}
FROM_LINE = re.compile(r"^(From|Von|Från): .+<[^>]+>\s*$")
WEEK_LABEL = re.compile(r"^Update (\d{2})-(\d{2})-(\d{4}) \(Week (\d+)\)\s*$")
MONTH_LABEL = re.compile(r"^Monthly steering report, ([A-Z][a-z]+ \d{4})\.?\s*$")


def parse_email_file(path, kind):
    lines = path.read_text(encoding="utf-8").split("\n")
    start = skip_synthetic_header(lines)
    # message boundaries: a From-line preceded by a blank line starts a quoted message;
    # the top message header starts at `start` (its first line is Subject:)
    bounds = [start]
    for i in range(start + 1, len(lines)):
        if FROM_LINE.match(lines[i]) and lines[i - 1].strip() == "":
            bounds.append(i)
    bounds.append(len(lines))

    raw_msgs = []
    for bi in range(len(bounds) - 1):
        a, b = bounds[bi], bounds[bi + 1]
        block = lines[a:b]
        hdr = {}
        j = 0
        while j < len(block):
            m = re.match(r"^([A-Za-zÅÄÖåäö ]+): ?(.*)$", block[j])
            if not m or m.group(1) not in KEY_LOOKUP:
                break
            hdr[KEY_LOOKUP[m.group(1)]] = m.group(2).strip()
            j += 1
        raw_msgs.append((a + 1, b, hdr, block[j:], a + j + 1))

    stem = path.stem
    total = len(raw_msgs)
    declared = int(raw_msgs[0][2].get("count", "0") or 0)
    thread_subject = raw_msgs[0][2].get("subject")
    prefix = "report" if kind == "report" else "email"
    rel = f"{path.parent.name}/{path.name}"
    units = []
    for top_pos, (l1, l2, hdr, body, body_l1) in enumerate(raw_msgs, start=1):
        sender = parse_people(hdr["from"])[0]
        to = parse_people(hdr.get("to", ""))
        cc = parse_people(hdr.get("cc", ""))
        sent = parse_date(hdr["sent"])
        body = [l.rstrip() for l in body]
        body = [l for l in body if not l.startswith(EXTERNAL_BANNER)]
        body = strip_placeholders(body)
        # signature = from the first line that is exactly the sender's display name
        sig_idx = next((k for k, l in enumerate(body) if l.strip() == sender["name"]), None)
        signature = None
        if sig_idx is not None:
            signature = "\n".join(squeeze_blank(body[sig_idx:]))
            body = body[:sig_idx]
        body = squeeze_blank(body)
        text = "\n".join(body)
        chrono_pos = total - top_pos + 1
        unit_type = "email_message"
        label = None
        if kind == "report":
            for l in body:
                mw = WEEK_LABEL.match(l)
                mm = MONTH_LABEL.match(l)
                if mw:
                    label = f"Week {int(mw.group(4))} ({mw.group(3)}-{mw.group(2)}-{mw.group(1)})"
                    unit_type = "report_weekly"
                    break
                if mm:
                    label = mm.group(1)
                    unit_type = "report_monthly"
                    break
            if label is None:
                unit_type = "report_unlabelled"
        line_end = l2
        while line_end > l1 and lines[line_end - 1].strip() == "":
            line_end -= 1
        if label:
            human = f"{path.name} · {label} · sent {sent} by {sender['name']}"
        else:
            human = (f"{path.name} · message {chrono_pos} of {total} "
                     f"(position {top_pos} from top) · sent {sent} by {sender['name']}")
        units.append({
            "unit_id": f"{prefix}:{stem}:m{chrono_pos:02d}",
            "source_file": rel,
            "unit_type": unit_type,
            "date": sent,
            "speaker": sender["name"],
            "speaker_email": sender["email"],
            "recipients": {"to": to, "cc": cc},
            "subject": hdr.get("subject"),
            "thread_subject": thread_subject,
            "text": text,
            "signature": signature,
            "anchor": {
                "kind": "email",
                "thread": stem,
                "position_from_top": top_pos,
                "position_chrono": chrono_pos,
                "thread_size": total,
                "declared_thread_size": declared,
                "label": label,
                "sent": sent,
                "line_start": l1,
                "body_line_start": body_l1,
                "line_end": line_end,
                "human": human,
            },
        })
    units.sort(key=lambda u: u["anchor"]["position_chrono"])
    return units, declared, total


# ---------------------------------------------------------------- transcripts
TS_LINE = re.compile(r"^\d+:\d\d\d+:\d\d$")


def speaker_time_re(names):
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"^(?P<name>{alt})(?: (?P<m>\d+) minutes?)?(?: (?P<s>\d+) seconds?)?$")


def parse_meta(lines, start):
    meta = {}
    i = start
    while i < len(lines) and re.match(r"^(Meeting|Customer|Date|Phase|Attendees): ", lines[i]):
        k, v = lines[i].split(": ", 1)
        meta[k.lower()] = v.strip()
        i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    att = []
    # split on commas that are not inside parentheses: "Robert Kahn (Acme CFO, joins late)"
    for part in re.split(r",\s*(?![^()]*\))", meta.get("attendees", "")):
        m = re.match(r"^(?P<name>.+?)\s*\((?P<org>[^)]*)\)$", part.strip())
        att.append({"name": m["name"], "org": m["org"]} if m else {"name": part.strip(), "org": None})
    meta["attendees"] = att
    return meta, i


def parse_transcript(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    start = skip_synthetic_header(lines)
    meta, i = parse_meta(lines, start)
    stem = path.stem
    rel = f"{path.parent.name}/{path.name}"
    base = {
        "source_file": rel, "unit_type": "transcript_turn", "date": meta["date"],
        "meeting": meta.get("meeting"), "phase": meta.get("phase"),
        "attendees": meta["attendees"], "recipients": [a["name"] for a in meta["attendees"]],
    }
    segments = []
    is_internal = any(re.match(r"^(Me|Them): ", l) for l in lines[i:i + 5])
    if is_internal:
        fmt = "internal_laptop"
        for ln in range(i, len(lines)):
            l = lines[ln]
            m = re.match(r"^(Me|Them): ?(.*)$", l)
            if m:
                segments.append({"speaker": m.group(1), "offset": None,
                                 "marker_line": ln + 1, "text": [(ln + 1, m.group(2))]})
            elif l.strip() and segments:
                segments[-1]["text"].append((ln + 1, l))
    else:
        fmt = "teams"
        # speaker names = the line above every mm:ssmm:ss line
        names = {lines[k - 1] for k in range(i, len(lines)) if TS_LINE.match(lines[k])}
        st_re = speaker_time_re(names)
        skip = set()
        for k in range(i, len(lines)):
            if TS_LINE.match(lines[k]):
                skip.update({k - 1, k, k + 1})   # name line / mm:ss line / initials line
        for ln in range(i, len(lines)):
            if ln in skip:
                continue
            l = lines[ln]
            m = st_re.match(l)
            if m and (m["m"] is not None or m["s"] is not None):
                off = int(m["m"] or 0) * 60 + int(m["s"] or 0)
                segments.append({"speaker": m["name"], "offset": off, "marker_line": ln + 1, "text": []})
            elif l.strip():
                if not segments:
                    raise RuntimeError(f"{path}: text before first speaker at line {ln + 1}: {l!r}")
                segments[-1]["text"].append((ln + 1, l))
    # merge consecutive same-speaker segments into one turn
    turns = []
    for s in segments:
        if turns and turns[-1]["speaker"] == s["speaker"]:
            turns[-1]["segments"].append(s)
        else:
            turns.append({"speaker": s["speaker"], "segments": [s]})
    units = []
    for n, t in enumerate(turns, start=1):
        first = t["segments"][0]
        all_text = [(ln, tx) for s in t["segments"] for (ln, tx) in s["text"]]
        text = " ".join(tx.strip() for _, tx in all_text)
        last_line = max([first["marker_line"]] + [ln for ln, _ in all_text])
        off = first["offset"]
        mmss = f"{off // 60}:{off % 60:02d}" if off is not None else None
        if mmss:
            human = f"{path.name} · {t['speaker']} @ {mmss} (line {first['marker_line']})"
        else:
            human = f"{path.name} · {t['speaker']} turn {n} (line {first['marker_line']})"
        units.append({
            "unit_id": f"transcript:{stem}:t{n:03d}",
            **base,
            "speaker": t["speaker"],
            "text": text,
            "anchor": {
                "kind": "transcript", "format": fmt, "meeting_file": stem,
                "turn_index": n, "speaker": t["speaker"],
                "offset_seconds": off, "timestamp": mmss,
                "line_start": first["marker_line"], "text_line_start": first["marker_line"] + 1,
                "line_end": last_line,
                "segments": [{"offset_seconds": s["offset"], "line": s["marker_line"]} for s in t["segments"]],
                "human": human,
            },
        })
    return units


# ---------------------------------------------------------------- main
def main():
    units = []
    problems = []
    for p in sorted((CORPUS / "emails").glob("*.txt")):
        u, declared, total = parse_email_file(p, "email")
        if declared != total:
            problems.append(f"{p.name}: header says {declared} messages, found {total}")
        units += u
    for p in sorted((CORPUS / "reports").glob("*.txt")):
        u, declared, total = parse_email_file(p, "report")
        if declared != total:
            problems.append(f"{p.name}: header says {declared} messages, found {total}")
        units += u
    for p in sorted((CORPUS / "transcripts").glob("*.txt")):
        units += parse_transcript(p)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    c = Counter(u["unit_type"] for u in units)
    print(f"wrote {len(units)} units -> {OUT}")
    for k, v in sorted(c.items()):
        print(f"  {k:18s} {v}")
    print(f"  files: {len({u['source_file'] for u in units})}")
    if problems:
        print("declared-vs-found mismatches:")
        for x in problems:
            print("  " + x)


if __name__ == "__main__":
    main()
