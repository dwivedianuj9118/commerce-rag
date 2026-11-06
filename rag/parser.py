"""Regex-based parser: shopper text -> hard constraints (category, colour, price...).

Using rules instead of an LLM call here because it's fast, free and easy to test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

# shopper word -> catalogue category
CATEGORY_SYNONYMS = {
    "dress": ["dress", "dresses", "frock", "gown"],
    "shirt": ["shirt", "shirts", "button-up", "button-down", "blouse"],
    "t-shirt": ["t-shirt", "t-shirts", "tee", "tees", "tshirt"],
    "jeans": ["jeans", "denims"],
    "trousers": ["trousers", "pants", "chinos", "slacks", "joggers"],
    "jacket": ["jacket", "jackets", "blazer", "bomber"],
    "coat": ["coat", "coats", "overcoat", "parka", "puffer", "trench"],
    "knitwear": ["jumper", "jumpers", "sweater", "sweaters", "cardigan", "knit", "knitwear"],
    "trainers": ["trainers", "sneakers", "kicks", "running shoes", "trainer"],
    "boots": ["boots", "boot"],
    "heels": ["heels", "pumps", "stilettos"],
    "loafers": ["loafers", "loafer"],
    "sandals": ["sandals", "sliders", "slides", "flip flops"],
    "handbag": ["handbag", "handbags", "bag", "purse", "tote", "clutch", "crossbody"],
    "backpack": ["backpack", "rucksack", "daypack"],
    "watch": ["watch", "watches", "timepiece"],
}

COLOURS = ["black", "white", "navy", "grey", "beige", "brown", "olive", "burgundy",
           "red", "pink", "blue", "green", "cream", "tan"]
COLOUR_SYNONYMS = {"gray": "grey", "maroon": "burgundy", "khaki": "olive", "ivory": "cream",
                   "camel": "tan", "nude": "beige"}

GENDER_WORDS = {
    "women": ["women", "women's", "womens", "ladies", "for her", "my wife", "my girlfriend", "my mum"],
    "men": ["men", "men's", "mens", "for him", "my husband", "my boyfriend", "my dad"],
}

BRANDS = ["Northvale", "Atelier Rue", "Loom Street", "Mariner & Co", "Solace",
          "Brightline", "Harrow & Finch", "Palma", "Kestrel"]

_NUM = r"£?\s?(\d+(?:\.\d{1,2})?)"
PRICE_MAX = [rf"under\s+{_NUM}", rf"below\s+{_NUM}", rf"less than\s+{_NUM}", rf"max(?:imum)?\s+{_NUM}",
             rf"up to\s+{_NUM}", rf"no more than\s+{_NUM}", rf"budget(?: is| of)?\s+{_NUM}",
             rf"{_NUM}\s+or less", rf"cheaper than\s+{_NUM}"]
PRICE_MIN = [rf"over\s+{_NUM}", rf"above\s+{_NUM}", rf"more than\s+{_NUM}", rf"at least\s+{_NUM}"]
PRICE_BETWEEN = rf"between\s+{_NUM}\s+and\s+{_NUM}"


@dataclass
class Constraints:
    category: str | None = None
    colour: str | None = None
    max_price: float | None = None
    min_price: float | None = None
    gender: str | None = None
    brand: str | None = None
    size: str | None = None

    def merge(self, newer: "Constraints") -> "Constraints":
        # newer values win. if the category changes, drop colour/size
        # ("black boots" -> "what about a bag?" shouldn't keep black)
        out = Constraints(**asdict(self))
        if newer.category and newer.category != self.category:
            out.colour, out.size = None, None
        for k, v in asdict(newer).items():
            if v is not None:
                setattr(out, k, v)
        return out

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ParsedQuery:
    text: str
    constraints: Constraints = field(default_factory=Constraints)


def _find_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None


def parse(text: str) -> ParsedQuery:
    t = text.lower()
    c = Constraints()

    # longest match wins, so "running shoes" beats "shoes"
    best = None
    for cat, syns in CATEGORY_SYNONYMS.items():
        for s in syns:
            if _find_word(t, s) and (best is None or len(s) > len(best[1])):
                best = (cat, s)
    if best:
        c.category = best[0]

    for col in COLOURS:
        if _find_word(t, col):
            c.colour = col
            break
    else:
        for syn, col in COLOUR_SYNONYMS.items():
            if _find_word(t, syn):
                c.colour = col
                break

    m = re.search(PRICE_BETWEEN, t)
    if m:
        c.min_price, c.max_price = float(m.group(1)), float(m.group(2))
    else:
        for pat in PRICE_MAX:
            m = re.search(pat, t)
            if m:
                c.max_price = float(m.group(1))
                break
        for pat in PRICE_MIN:
            m = re.search(pat, t)
            if m:
                c.min_price = float(m.group(1))
                break

    for g, words in GENDER_WORDS.items():
        if any(_find_word(t, w) for w in words):
            c.gender = g
            break

    for b in BRANDS:
        if b.lower() in t:
            c.brand = b
            break

    m = re.search(r"\bsize\s+(xs|s|m|l|xl|\d{1,2})\b", t)
    if m:
        c.size = m.group(1).upper()

    return ParsedQuery(text=text, constraints=c)
