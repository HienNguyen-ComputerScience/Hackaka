"""Hybrid retrieval with unit-level citations.

    python pipeline/retrieve.py "what proportion of articles had shelf-life data" -k 5
    python pipeline/retrieve.py "..." --person person:<slug>   (only passages mentioning them)

API: Retriever().search(query, k=8) -> list of hits; each hit has the passage, its score,
the member units with human anchors, and a `pin`: the single unit inside the passage that
best matches the query, which is what an answer should cite.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(s):
    return TOKEN.findall(s.lower())


class Retriever:
    def __init__(self, data=DATA):
        self.units = {u["unit_id"]: u for u in map(json.loads, (data / "units.jsonl").open(encoding="utf-8"))}
        self.passages = [json.loads(l) for l in (data / "passages.jsonl").open(encoding="utf-8")]
        self.emb = np.load(data / "embeddings.npy")
        assert len(self.passages) == self.emb.shape[0], "passages/embeddings out of sync: rebuild index"
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in self.passages])
        self._model = None

    def _embed_query(self, q):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return self._model.encode([q], normalize_embeddings=True)[0].astype("float32")

    def _embed_batch(self, texts):
        self._embed_query("warm")
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")

    def search(self, query, k=8, person=None, kinds=None, date_from=None, date_to=None, pool=60):
        qtok = tokenize(query)
        bm = self.bm25.get_scores(qtok)
        dense = self.emb @ self._embed_query(query)
        # reciprocal-rank fusion of the two rankings
        order_bm = np.argsort(-bm)
        order_de = np.argsort(-dense)
        rrf = np.zeros(len(self.passages))
        for rank, i in enumerate(order_bm[:pool]):
            if bm[i] > 0:
                rrf[i] += 1 / (60 + rank)
        for rank, i in enumerate(order_de[:pool]):
            rrf[i] += 1 / (60 + rank)
        hits = []
        pinned = set()   # overlapping windows: keep the best hit per pinned unit
        for i in np.argsort(-rrf):
            if rrf[i] == 0:
                break
            p = self.passages[i]
            if person and person not in p["people"]:
                continue
            if kinds and p["kind"] not in kinds:
                continue
            if date_from and p["date"] < date_from:
                continue
            if date_to and p["date"] > date_to:
                continue
            h = self._hit(p, float(rrf[i]), float(bm[i]), float(dense[i]), qtok)
            if h["pin"]["unit_id"] in pinned:
                continue
            pinned.add(h["pin"]["unit_id"])
            hits.append(h)
            if len(hits) >= k:
                break
        return hits

    def _hit(self, p, score, bm, dense, qtok):
        members = [self.units[uid] for uid in p["units"]]
        qset = set(qtok)

        def overlap(u):
            t = tokenize(u["text"])
            return (len(qset & set(t)), len(t))

        pin = max(members, key=overlap)
        return {
            "passage_id": p["passage_id"], "score": round(score, 5), "bm25": round(bm, 2), "cosine": round(dense, 3),
            "source_file": p["source_file"], "kind": p["kind"], "date": p["date"], "people": p["people"],
            "pin": cite(pin), "units": [cite(u) for u in members],
            "text": p["text"],
        }


def cite(u):
    """Unit-level citation: everything an answer needs to point at the source."""
    a = u["anchor"]
    return {
        "unit_id": u["unit_id"], "source_file": u["source_file"], "date": u["date"], "speaker": u["speaker"],
        "where": a["human"], "line_start": a["line_start"], "line_end": a["line_end"],
        "text": u["text"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--person")
    ap.add_argument("--kind", action="append")
    ap.add_argument("--full", action="store_true", help="print whole passage text")
    args = ap.parse_args()
    r = Retriever()
    for n, h in enumerate(r.search(args.query, k=args.k, person=args.person, kinds=args.kind), 1):
        print(f"#{n}  rrf={h['score']}  bm25={h['bm25']}  cos={h['cosine']}  [{h['kind']}] {h['date']}")
        print(f"    CITE: {h['pin']['where']}")
        print(f"          \"{h['pin']['text'][:200]}\"")
        if h["kind"] == "transcript_window":
            print(f"    window: {h['units'][0]['where'].split(' · ')[1]} .. {h['units'][-1]['where'].split(' · ')[1]}")
        print(f"    people: {', '.join(x.split(':',1)[1] for x in h['people'])}")
        if args.full:
            print("    ---\n    " + h["text"].replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
