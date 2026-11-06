"""The conversational loop: parse -> (carry state) -> gate -> retrieve -> generate -> validate."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace

from .generation import Card, TemplateGenerator, GenResult
from .parser import CATEGORY_SYNONYMS, Constraints, parse
from .retrieval import Catalogue, Retriever, STOPWORDS, tokenize
from .synonyms import SHOPPER_VOCAB, expand
from .validator import Validation, validate


@dataclass
class TurnResult:
    text: str
    product_ids: list[str]            # what retrieval handed the generator, in rank order
    constraints: Constraints
    action: str                       # answer | refuse_out_of_catalogue | no_match | validation_fallback
    unknown_terms: list[str]
    disclosed: list[str]
    validation: Validation | None
    gen: GenResult | None
    timings_ms: dict = field(default_factory=dict)


@dataclass
class Config:
    mode: str = "hybrid"          # bm25 | dense | hybrid
    use_filters: bool = True
    use_expansion: bool = True
    stateful: bool = True
    k: int = 5                    # cards passed to the generator
    disclosure: str = "brand_like"  # any_unknown (v1) | brand_like (v2): when to say "we don't carry X"


def brand_like_terms(message: str, unknown: list[str]) -> list[str]:
    """Unknown words that sit where a brand would: capitalised, not sentence-initial,
    and within three words before a product noun ("any Dr Martens boots?")."""
    words = re.findall(r"[A-Za-z0-9£'&-]+", message)
    low = [w.lower().strip("'") for w in words]
    nouns = {w for syns in CATEGORY_SYNONYMS.values() for s in syns for w in s.split()}
    out = []
    for i, w in enumerate(words):
        if i == 0 or low[i] not in unknown or not w[0].isupper():
            continue
        if any(low[j] in nouns for j in range(i + 1, min(len(words), i + 4))):
            out.append(w)
    return out


class Assistant:
    def __init__(self, cat: Catalogue, retriever: Retriever, generator=None, config: Config | None = None):
        self.cat, self.retriever = cat, retriever
        self.gen = generator or TemplateGenerator()
        self.cfg = config or Config()
        self.reset()

    def reset(self):
        self.history: list[str] = []
        self.state = Constraints()

    def _cards(self, pids: list[str], c: Constraints) -> list[Card]:
        cards = []
        for pid in pids:
            p = self.cat.by_id[pid]
            vs = self.cat.matching_variants(pid, c) or [v for v in p["variants"] if v["in_stock"]]
            if not vs:
                continue
            cards.append(Card(pid, p["name"], p["brand"], p["category"], min(v["price"] for v in vs),
                              sorted({v["colour"] for v in vs}),
                              sorted({v["size"] for v in vs}, key=lambda s: (len(s), s)),
                              p["description"]))
        return cards

    def _why_empty(self, c: Constraints) -> str:
        """Name the constraint that blocks results, so the failure is explicit."""
        for field_name in ("colour", "max_price", "size", "brand", "gender", "min_price"):
            if getattr(c, field_name) is None:
                continue
            relaxed = replace(c, **{field_name: None})
            if self.retriever.search("", relaxed, mode="bm25", k=1):
                val = getattr(c, field_name)
                label = {"max_price": f"under £{val:g}" if field_name == "max_price" else "",
                         "min_price": f"over £{val:g}" if field_name == "min_price" else "",
                         "colour": f"in {val}", "size": f"in size {val}"}.get(field_name) or f"for {field_name} {val}"
                return f"Nothing is in stock {label}; dropping that gives options."
        return "Nothing in our range matches all of that."

    def turn(self, message: str) -> TurnResult:
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        parsed = parse(message)
        c = self.state.merge(parsed.constraints) if self.cfg.stateful else parsed.constraints
        unknown = self.retriever.unknown_terms(message, SHOPPER_VOCAB)
        content = [t for t in tokenize(message) if t not in STOPWORDS and not t.replace(".", "").lstrip("£").isdigit()]
        known_content = [t for t in content if t not in unknown]
        tm["parse"] = (time.perf_counter() - t0) * 1000

        self.history.append(message)
        if self.cfg.stateful:
            self.state = c

        # Gate: nothing we recognise and no category -> we don't sell it. Refuse, don't retrieve.
        if c.category is None and unknown and not known_content:
            text = (f"Sorry, we don't sell {' '.join(unknown)}. We're a fashion retailer: "
                    "clothing, shoes, bags and watches.")
            return TurnResult(text, [], c, "refuse_out_of_catalogue", unknown, [], None, None, tm)

        t1 = time.perf_counter()
        q_turns = self.history[-3:] if self.cfg.stateful else [message]
        q = " ".join(q_turns)
        if self.cfg.use_expansion:
            q = expand(q)
        hits = self.retriever.search(q, c if self.cfg.use_filters else None, mode=self.cfg.mode, k=self.cfg.k)
        pids = [h.product_id for h in hits]
        tm["retrieve"] = (time.perf_counter() - t1) * 1000

        notes = []
        flagged = unknown if self.cfg.disclosure == "any_unknown" else brand_like_terms(message, unknown)
        if flagged and c.category:
            notes.append(f"We don't carry {' '.join(flagged)}, so these are the closest items from our own brands.")
        action = "answer"
        if not pids:
            notes.append(self._why_empty(c))
            action = "no_match"

        cards = self._cards(pids, c)
        g = self.gen.generate(self.history, c, cards, notes)
        tm["generate"] = g.latency_s * 1000

        t3 = time.perf_counter()
        v = validate(g.text, self.cat, {cd.product_id for cd in cards},
                     tuple(x for x in (c.max_price, c.min_price) if x is not None))
        tm["validate"] = (time.perf_counter() - t3) * 1000
        text = g.text
        if not v.ok or g.refused_by_model:
            # Fail loudly: log the issues, never ship the unverified text.
            action = "validation_fallback"
            safe = TemplateGenerator().generate(self.history, c, cards, notes)
            text = "[answer failed grounding check; showing verified catalogue data]\n" + safe.text
        tm["total"] = (time.perf_counter() - t0) * 1000
        return TurnResult(text, pids, c, action, unknown, flagged if c.category else [], v, g, tm)
