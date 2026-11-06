"""Catalogue loading, structured filtering, and BM25 / dense / hybrid retrieval.

Two index granularities are supported so the eval can compare them:
  * "product": one document per parent product (the design we keep)
  * "sku":     one document per variant SKU (the design we reject; see README)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .parser import Constraints

DATA = Path(__file__).resolve().parent.parent / "data" / "catalogue.json"

STOPWORDS = set("""a an the and or for to of in on with my me i im i'm we you your is are be
something some any anything need want looking look find show get got buy like would could
please thanks hi hello do does have has sell stock carry what which who can also just
really good nice new pair one ones that this these those it its it's under below over above
than less more between max up budget about around at by from into as so too very
""".split())

TOKEN_RE = re.compile(r"[a-z0-9£']+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t.strip("'") for t in TOKEN_RE.findall(text.lower()) if t.strip("'")]


class Catalogue:
    def __init__(self, path: Path = DATA):
        self.products: list[dict] = json.loads(Path(path).read_text())
        self.by_id = {p["product_id"]: p for p in self.products}
        self.skus = [(p["product_id"], v) for p in self.products for v in p["variants"]]

    @staticmethod
    def product_doc(p: dict) -> str:
        colours = sorted({v["colour"] for v in p["variants"]})
        return (f"{p['name']}. {p['category']}. {p['gender']}. {p['material']}. "
                f"{p['description']} Colours: {', '.join(colours)}.")

    @staticmethod
    def sku_doc(p: dict, v: dict) -> str:
        return f"{p['name']} in {v['colour']}, size {v['size']}. {p['category']}. {p['description']}"

    def matching_variants(self, pid: str, c: Constraints) -> list[dict]:
        """In-stock variants of a product satisfying the variant-level constraints."""
        out = []
        for v in self.by_id[pid]["variants"]:
            if not v["in_stock"]:
                continue
            if c.colour and v["colour"] != c.colour:
                continue
            if c.size and v["size"].upper() != c.size:
                continue
            if c.max_price is not None and v["price"] > c.max_price:
                continue
            if c.min_price is not None and v["price"] < c.min_price:
                continue
            out.append(v)
        return out

    def satisfies(self, pid: str, c: Constraints) -> bool:
        p = self.by_id[pid]
        if c.category and p["category"] != c.category:
            return False
        if c.gender and p["gender"] not in (c.gender, "unisex"):
            return False
        if c.brand and p["brand"] != c.brand:
            return False
        return bool(self.matching_variants(pid, c))


# ---------------------------------------------------------------- embedders
class LSAEmbedder:
    """Offline dense baseline: TF-IDF (word + char n-grams) -> truncated SVD.

    This is a stand-in for a neural embedder. It captures co-occurrence ("trail" near
    "hiking") but not true paraphrase. The interface matches FastEmbedEmbedder and a
    Bedrock Titan embedder, so swapping is one line; the eval then tells you whether
    the better embedder actually paid for itself.
    """
    name = "lsa-256"

    def __init__(self, dim: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline, make_union
        self.pipe = make_pipeline(
            make_union(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
            ),
            TruncatedSVD(n_components=dim, random_state=0),
        )

    def fit(self, docs: list[str]) -> np.ndarray:
        return self._norm(self.pipe.fit_transform(docs))

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._norm(self.pipe.transform(texts))

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU). Needs HuggingFace access once."""
    name = "bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def fit(self, docs: list[str]) -> np.ndarray:
        return self.embed(docs)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts)))


def make_embedder(name: str = "lsa"):
    if name == "lsa":
        return LSAEmbedder()
    if name == "bge":
        return FastEmbedEmbedder()
    raise ValueError(name)


# ---------------------------------------------------------------- retriever
@dataclass
class Hit:
    product_id: str
    score: float


class Retriever:
    def __init__(self, cat: Catalogue, granularity: str = "product", embedder: str = "lsa"):
        self.cat = cat
        self.granularity = granularity
        if granularity == "product":
            self.doc_pids = [p["product_id"] for p in cat.products]
            docs = [cat.product_doc(p) for p in cat.products]
        else:
            self.doc_pids = [pid for pid, _ in cat.skus]
            docs = [cat.sku_doc(cat.by_id[pid], v) for pid, v in cat.skus]
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.embedder = make_embedder(embedder)
        self.doc_vecs = self.embedder.fit(docs)
        self.vocab = {t for d in docs for t in tokenize(d)}

    # raw scorers return one score per document
    def _bm25(self, q: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(q)))

    def _dense(self, q: str) -> np.ndarray:
        return self.doc_vecs @ self.embedder.embed([q])[0]

    @staticmethod
    def _rrf(*score_lists: np.ndarray, k: int = 60) -> np.ndarray:
        """Reciprocal-rank fusion. Uses ranks only, so BM25 and cosine need no calibration."""
        fused = np.zeros_like(score_lists[0], dtype=float)
        for s in score_lists:
            ranks = np.empty_like(s, dtype=int)
            ranks[np.argsort(-s)] = np.arange(len(s))
            fused += 1.0 / (k + ranks + 1)
        return fused

    def search(self, query: str, constraints: Constraints | None = None, mode: str = "hybrid",
               k: int = 10) -> list[Hit]:
        if mode == "bm25":
            scores = self._bm25(query)
        elif mode == "dense":
            scores = self._dense(query)
        elif mode == "hybrid":
            scores = self._rrf(self._bm25(query), self._dense(query))
        else:
            raise ValueError(mode)

        order = np.argsort(-scores)
        hits: list[Hit] = []
        seen: set[str] = set()
        if self.granularity == "sku":
            # A SKU index returns k *SKUs*; the user sees however many distinct products those are.
            for i in order[:k]:
                pid = self.doc_pids[i]
                if pid not in seen:
                    seen.add(pid)
                    hits.append(Hit(pid, float(scores[i])))
            return hits

        for i in order:
            pid = self.doc_pids[i]
            if pid in seen:
                continue
            if constraints is not None and not self.cat.satisfies(pid, constraints):
                continue
            seen.add(pid)
            hits.append(Hit(pid, float(scores[i])))
            if len(hits) == k:
                break
        return hits

    def unknown_terms(self, query: str, extra_vocab: set[str]) -> list[str]:
        """Content words the catalogue has never seen, e.g. 'ps5', 'gucci'."""
        out = []
        for t in tokenize(query):
            if t in STOPWORDS or t in self.vocab or t in extra_vocab:
                continue
            if re.fullmatch(r"£?\d+(\.\d+)?", t):
                continue
            out.append(t)
        return out
