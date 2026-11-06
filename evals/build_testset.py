"""Builds labelled test queries from catalogue attributes.

grade 2 = matches all hard constraints (category, gender, in-stock colour, budget) + all soft ones (occasion, style)
grade 1 = hard constraints + at least one soft one

Half the soft phrases are held out, i.e. not in rag/synonyms.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from rag.parser import CATEGORY_SYNONYMS
from rag.retrieval import Catalogue

OUT = Path(__file__).resolve().parent.parent / "data" / "testset.json"
SEED = 11

OCCASION_SEEN = {
    "wedding": ["a wedding"], "office": ["the office", "work"], "gym": ["the gym"],
    "beach holiday": ["a beach holiday"], "hiking": ["hiking"], "date night": ["a dinner date"],
    "everyday": ["everyday wear"], "party": ["a party"],
}
OCCASION_HELDOUT = {
    "wedding": ["my friend's nuptials", "a summer ceremony"], "office": ["client meetings", "my nine-to-five"],
    "gym": ["lifting weights", "spin class"], "beach holiday": ["a trip to Ibiza", "lounging by the pool"],
    "hiking": ["hillwalking in the Lakes", "a long trek"], "date night": ["a romantic evening", "drinks on Friday night"],
    "everyday": ["running errands", "lazy weekends"], "party": ["a birthday bash", "new year's eve"],
}
STYLE_SEEN = {
    "minimalist": ["minimal"], "classic": ["classic"], "casual": ["casual"], "sporty": ["sporty"],
    "bohemian": ["boho"], "edgy": ["edgy"], "preppy": ["preppy"], "outdoor": ["rugged"],
}
STYLE_HELDOUT = {
    "minimalist": ["nothing flashy", "Scandi-style"], "classic": ["old-school elegant", "something that won't date"],
    "casual": ["chilled", "low-key"], "sporty": ["athleisure"], "bohemian": ["hippie-ish", "festival vibe"],
    "edgy": ["a bit punk", "grungy"], "preppy": ["ivy league look"], "outdoor": ["tough enough for bad weather"],
}
GENDER_PHRASE = {"women": ["for my wife", "women's"], "men": ["for my husband", "men's"]}

OUT_OF_CATALOGUE = ["a PS5", "a lawn mower", "an iPhone case", "garden furniture", "dog food", "a sofa",
                    "car tyres", "noise-cancelling headphones", "a kettle", "protein powder", "board games",
                    "a mattress", "a laptop", "a fishing rod", "a pram", "paint", "vitamins", "a drone"]
OOC_TEMPLATES = ["do you sell {x}?", "I'm looking for {x}", "{x} under £50", "can I get {x}"]
FOREIGN_BRANDS = [("Gucci", "handbag"), ("Prada", "handbag"), ("Nike", "trainers"), ("Adidas", "trainers"),
                  ("Levi's", "jeans"), ("Zara", "dress"), ("Rolex", "watch"), ("Dr Martens", "boots"),
                  ("Ray-Ban", "watch"), ("Birkenstock", "sandals"), ("Uniqlo", "t-shirt"), ("Barbour", "jacket")]


def grade(cat: Catalogue, p: dict, hard: dict, soft: dict) -> int:
    if hard.get("category") and p["category"] != hard["category"]:
        return 0
    if hard.get("gender") and p["gender"] not in (hard["gender"], "unisex"):
        return 0
    ok_variant = any(v["in_stock"]
                     and (not hard.get("colour") or v["colour"] == hard["colour"])
                     and (hard.get("max_price") is None or v["price"] <= hard["max_price"])
                     for v in p["variants"])
    if not ok_variant:
        return 0
    wants = [("occasions", soft.get("occasion")), ("styles", soft.get("style"))]
    wants = [(k, v) for k, v in wants if v]
    if not wants:
        return 2
    hit = sum(v in p[k] for k, v in wants)
    return 2 if hit == len(wants) else (1 if hit else 0)


def labels(cat: Catalogue, hard: dict, soft: dict) -> dict:
    out = {}
    for p in cat.products:
        g = grade(cat, p, hard, soft)
        if g:
            out[p["product_id"]] = g
    return out


def phrase_single(rng, hard, soft, heldout) -> str:
    parts = []
    if soft.get("style"):
        parts.append(rng.choice((STYLE_HELDOUT if heldout else STYLE_SEEN)[soft["style"]]))
    if hard.get("colour"):
        parts.append(hard["colour"])
    if hard.get("category"):
        parts.append(rng.choice(CATEGORY_SYNONYMS[hard["category"]]))
    else:
        parts.append("something")
    s = " ".join(parts)
    if hard.get("gender"):
        g = rng.choice(GENDER_PHRASE[hard["gender"]])
        s = f"{g} {s}" if g.endswith("'s") else f"{s} {g}"
    if soft.get("occasion"):
        s += " for " + rng.choice((OCCASION_HELDOUT if heldout else OCCASION_SEEN)[soft["occasion"]])
    if hard.get("max_price"):
        s += f" under £{hard['max_price']:g}"
    return rng.choice(["I need {}", "looking for {}", "{}", "any {}?", "can you find me {}"]).format(s)


def build() -> dict:
    rng = random.Random(SEED)
    cat = Catalogue()
    single, multi = [], []

    tries = 0
    while len(single) < 240 and tries < 20000:
        tries += 1
        anchor = rng.choice(cat.products)  # anchor on a real product so the query is answerable
        v = rng.choice([x for x in anchor["variants"] if x["in_stock"]] or anchor["variants"])
        vague = rng.random() < 0.25
        hard = {"category": None if vague else anchor["category"]}
        if rng.random() < 0.5:
            hard["colour"] = v["colour"]
        if rng.random() < 0.6:
            hard["max_price"] = float(int(v["price"] / 10 + 1) * 10 + rng.choice([0, 10, 20]))
        if not vague and anchor["gender"] != "unisex" and rng.random() < 0.3:
            hard["gender"] = anchor["gender"]
        soft = {}
        if vague or rng.random() < 0.6:
            soft["occasion"] = rng.choice(anchor["occasions"])
        if rng.random() < (0.5 if vague else 0.35):
            soft["style"] = rng.choice(anchor["styles"])
        if vague and not soft:
            continue
        rel = labels(cat, hard, soft)
        n2 = sum(1 for g in rel.values() if g == 2)
        if not 1 <= n2 <= 25:
            continue
        heldout = bool(soft) and len(single) % 2 == 1
        single.append({"id": f"S{len(single):03d}", "type": "single", "turns": [phrase_single(rng, hard, soft, heldout)],
                       "hard": {k: v for k, v in hard.items() if v is not None}, "soft": soft,
                       "phrasing": "heldout" if heldout else ("seen" if soft else "no_soft"),
                       "vague": vague, "relevance": rel})

    while len(multi) < 60:
        anchor = rng.choice(cat.products)
        stock = [x for x in anchor["variants"] if x["in_stock"]]
        if not stock:
            continue
        v = rng.choice(stock)
        occ = rng.choice(anchor["occasions"])
        hard = {"category": anchor["category"], "max_price": float(int(v["price"] / 10 + 1) * 10), "colour": v["colour"]}
        soft = {"occasion": occ}
        rel = labels(cat, hard, soft)
        if not 1 <= sum(1 for g in rel.values() if g == 2) <= 25:
            continue
        noun = rng.choice(CATEGORY_SYNONYMS[anchor["category"]])
        turns = [f"I need some {noun} for {OCCASION_SEEN[occ][0]}",
                 rng.choice(["ideally under £{p:g}", "my budget is £{p:g}", "can we keep it under £{p:g}?"]).format(p=hard["max_price"]),
                 rng.choice(["any in {c}?", "do you have them in {c}", "{c} would be best"]).format(c=v["colour"])]
        multi.append({"id": f"M{len(multi):03d}", "type": "multi", "turns": turns, "hard": hard, "soft": soft,
                      "phrasing": "seen", "vague": False, "relevance": rel})

    ooc = [{"id": f"O{i:03d}", "type": "out_of_catalogue",
            "turns": [OOC_TEMPLATES[i % len(OOC_TEMPLATES)].format(x=x)], "expect": "refuse_out_of_catalogue"}
           for i, x in enumerate(OUT_OF_CATALOGUE)]
    brands = [{"id": f"B{i:03d}", "type": "foreign_brand", "turns": [f"have you got any {b} {CATEGORY_SYNONYMS[c][0]}?"],
               "brand": b, "hard": {"category": c}, "expect": "answer_with_disclosure"}
              for i, (b, c) in enumerate(FOREIGN_BRANDS)]
    impossible = []
    for i, c in enumerate(list(CATEGORY_SYNONYMS)[:16]):
        cheapest = min(v["price"] for p in cat.products if p["category"] == c for v in p["variants"])
        impossible.append({"id": f"I{i:03d}", "type": "impossible",
                           "turns": [f"{CATEGORY_SYNONYMS[c][0]} under £{max(1, int(cheapest * 0.3)):g}"],
                           "hard": {"category": c, "max_price": float(max(1, int(cheapest * 0.3)))},
                           "expect": "no_match"})

    return {"single": single, "multi": multi, "out_of_catalogue": ooc, "foreign_brand": brands,
            "impossible": impossible}


def main():
    ts = build()
    OUT.write_text(json.dumps(ts, indent=1))
    print({k: len(v) for k, v in ts.items()}, "->", OUT)


if __name__ == "__main__":
    main()
