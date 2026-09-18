"""Build retrieval passages + hybrid index (BM25 + local MiniLM embeddings).

Reads  data/units.jsonl, data/mentions.jsonl
Writes data/passages.jsonl   one retrieval passage per email/report message or transcript window
       data/embeddings.npy   float32 [n_passages, 384], row i = passages[i]

Passages never lose the unit boundary: each passage lists its member unit_ids, so every hit
can be cited at unit level (message, or speaker + timestamp + line).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WINDOW_TURNS = 8      # transcript window size in turns
WINDOW_STRIDE = 4
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def build_passages(units, mentions):
    people_of = defaultdict(set)
    for m in mentions:
        people_of[m["unit_id"]].add(m["person_id"])
    passages = []
    by_file = defaultdict(list)
    for u in units:
        by_file[u["source_file"]].append(u)
    for f, us in by_file.items():
        if us[0]["anchor"]["kind"] == "email":
            for u in us:
                passages.append({
                    "passage_id": "p:" + u["unit_id"], "source_file": f, "kind": u["unit_type"],
                    "date": u["date"], "units": [u["unit_id"]],
                    "text": (u.get("subject") or "") + "\n" + u["text"],
                    "people": sorted(people_of[u["unit_id"]]),
                })
        else:
            us = sorted(us, key=lambda u: u["anchor"]["turn_index"])
            n = len(us)
            starts = list(range(0, max(1, n - WINDOW_TURNS + 1), WINDOW_STRIDE))
            if starts[-1] + WINDOW_TURNS < n:
                starts.append(n - WINDOW_TURNS)
            for s in starts:
                win = us[s:s + WINDOW_TURNS]
                ppl = set()
                for u in win:
                    ppl |= people_of[u["unit_id"]]
                passages.append({
                    "passage_id": f"p:{win[0]['unit_id']}..{win[-1]['anchor']['turn_index']:03d}",
                    "source_file": f, "kind": "transcript_window", "date": win[0]["date"],
                    "units": [u["unit_id"] for u in win],
                    "text": (win[0].get("meeting") or "") + "\n" + "\n".join(f"{u['speaker']}: {u['text']}" for u in win),
                    "people": sorted(ppl),
                })
    return passages


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype("float32")


def main():
    units = load_jsonl(DATA / "units.jsonl")
    mentions = load_jsonl(DATA / "mentions.jsonl")
    passages = build_passages(units, mentions)
    with (DATA / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    emb = embed([p["text"] for p in passages])
    np.save(DATA / "embeddings.npy", emb)
    kinds = defaultdict(int)
    for p in passages:
        kinds[p["kind"]] += 1
    print(f"passages: {len(passages)}  embeddings: {emb.shape}")
    for k, v in sorted(kinds.items()):
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
