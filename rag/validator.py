"""Post-generation grounding check. Runs on every answer, whichever backend wrote it.

The generator is non-deterministic; this is not. Every claim we can check mechanically
(product ids, prices, colours) is checked against the catalogue and against the cards
the model was actually shown. A failed check is surfaced, logged and replaced by a safe
answer: the system fails loudly rather than shipping a plausible-looking wrong price.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import COLOURS
from .retrieval import Catalogue

ID_RE = re.compile(r"\bP\d{4}\b")
PRICE_RE = re.compile(r"£\s?(-?\d+(?:\.\d{1,2})?)")


@dataclass
class Validation:
    ok: bool
    issues: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)


def _segments(text: str) -> list[str]:
    # one product claim per line or sentence
    parts = []
    for line in text.splitlines():
        parts.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line) if s.strip())
    return parts


def validate(text: str, cat: Catalogue, context_ids: set[str],
             budget_prices: tuple[float, ...] = ()) -> Validation:
    """budget_prices: the shopper's own budget figures, which the answer may repeat."""
    issues: list[str] = []
    cited: list[str] = []
    for seg in _segments(text):
        ids = ID_RE.findall(seg)
        prices = [float(p) for p in PRICE_RE.findall(seg)]
        for pid in ids:
            if pid not in cat.by_id:
                issues.append(f"unknown_product:{pid}")
            elif pid not in context_ids:
                issues.append(f"not_in_context:{pid}")
            cited.append(pid)
        known = [pid for pid in ids if pid in cat.by_id]
        # exact match only: a £119.99 product must not hide behind a £120 budget
        prices = [x for x in prices if not any(abs(x - b) < 0.001 for b in budget_prices)]
        if prices and not ids:
            issues.append(f"unattributed_price:{seg.strip()[:60]}")
        for price in prices:
            if known and not any(abs(v["price"] - price) < 0.011 for pid in known
                                 for v in cat.by_id[pid]["variants"]):
                issues.append(f"price_mismatch:{known[0]}:£{price}")
        colours_said = [c for c in COLOURS if re.search(rf"\b{c}\b", seg.lower())]
        for c in colours_said:
            if known and not any(v["colour"] == c and v["in_stock"] for pid in known
                                 for v in cat.by_id[pid]["variants"]):
                issues.append(f"colour_unavailable:{known[0]}:{c}")
    return Validation(ok=not issues, issues=issues, cited_ids=list(dict.fromkeys(cited)))
