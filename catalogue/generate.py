"""Generates a fake fashion catalogue (seeded, so it's the same every run).

product -> brand, category, gender, material, style/occasion tags, description
variant -> colour x size, price, in_stock

Descriptions use different words than shoppers would ("pared-back" not "minimal",
"ceremony-ready" not "wedding") so plain keyword matching isn't enough.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 7
OUT = Path(__file__).resolve().parent.parent / "data" / "catalogue.json"

# (name, price multiplier, preferred styles)
BRANDS = [
    ("Northvale", 1.1, ["outdoor", "casual"]),
    ("Atelier Rue", 1.8, ["classic", "minimalist"]),
    ("Loom Street", 0.6, ["casual", "minimalist"]),
    ("Mariner & Co", 1.0, ["preppy", "classic"]),
    ("Solace", 1.3, ["minimalist", "bohemian"]),
    ("Brightline", 0.9, ["sporty", "casual"]),
    ("Harrow & Finch", 1.6, ["classic", "preppy"]),
    ("Palma", 1.0, ["bohemian", "casual"]),
    ("Kestrel", 1.2, ["edgy", "sporty"]),
]

# category -> (gender options, base price range GBP, size scheme, materials, noun variants, occasions it suits)
CATEGORIES = {
    "dress": (["women"], (35, 160), "alpha", ["cotton", "linen", "silk", "viscose", "satin"],
              ["Midi Dress", "Wrap Dress", "Slip Dress", "Shirt Dress", "Maxi Dress"],
              ["wedding", "date night", "party", "beach holiday", "everyday", "office"]),
    "shirt": (["men", "women"], (25, 90), "alpha", ["cotton", "linen", "oxford cotton", "silk"],
              ["Oxford Shirt", "Linen Shirt", "Poplin Shirt", "Overshirt"],
              ["office", "everyday", "date night", "wedding"]),
    "t-shirt": (["men", "women", "unisex"], (12, 45), "alpha", ["organic cotton", "cotton", "merino"],
                ["Crew Tee", "Pocket Tee", "Boxy Tee", "Performance Tee"],
                ["everyday", "gym", "beach holiday"]),
    "jeans": (["men", "women"], (40, 140), "waist", ["denim", "stretch denim", "selvedge denim"],
              ["Straight Jeans", "Slim Jeans", "Wide-Leg Jeans", "Tapered Jeans"],
              ["everyday", "date night"]),
    "trousers": (["men", "women"], (35, 130), "waist", ["wool", "cotton twill", "linen", "tech fabric"],
                 ["Tailored Trousers", "Chinos", "Pleated Trousers", "Joggers"],
                 ["office", "wedding", "everyday", "gym"]),
    "jacket": (["men", "women", "unisex"], (60, 280), "alpha", ["leather", "wool", "nylon", "denim", "suede"],
               ["Biker Jacket", "Blazer", "Bomber Jacket", "Rain Shell", "Trucker Jacket"],
               ["office", "wedding", "everyday", "hiking", "date night"]),
    "coat": (["men", "women"], (90, 380), "alpha", ["wool", "cashmere blend", "down", "waxed cotton"],
             ["Overcoat", "Trench Coat", "Puffer Coat", "Parka"],
             ["office", "everyday", "hiking"]),
    "knitwear": (["men", "women", "unisex"], (30, 180), "alpha", ["merino", "cashmere", "cotton", "lambswool"],
                 ["Crew Jumper", "Cardigan", "Roll-Neck", "Cable Knit"],
                 ["office", "everyday", "date night"]),
    "trainers": (["men", "women", "unisex"], (40, 170), "shoe", ["leather", "mesh", "canvas", "suede"],
                 ["Court Trainers", "Running Trainers", "Low-Top Trainers", "Retro Runners"],
                 ["everyday", "gym"]),
    "boots": (["men", "women"], (70, 260), "shoe", ["leather", "suede", "waterproof leather"],
              ["Chelsea Boots", "Hiking Boots", "Ankle Boots", "Lace-Up Boots"],
              ["everyday", "hiking", "office", "date night"]),
    "heels": (["women"], (45, 220), "shoe", ["leather", "satin", "suede", "patent leather"],
              ["Block Heels", "Stiletto Pumps", "Slingback Heels", "Strappy Sandals Heels"],
              ["wedding", "party", "date night", "office"]),
    "loafers": (["men", "women"], (55, 210), "shoe", ["leather", "suede"],
                ["Penny Loafers", "Horsebit Loafers", "Chunky Loafers"],
                ["office", "wedding", "everyday"]),
    "sandals": (["women", "men", "unisex"], (20, 110), "shoe", ["leather", "rubber", "raffia"],
                ["Slide Sandals", "Strappy Sandals", "Sport Sandals"],
                ["beach holiday", "everyday"]),
    "handbag": (["women"], (40, 320), "one", ["leather", "canvas", "vegan leather", "raffia"],
                ["Tote Bag", "Crossbody Bag", "Clutch", "Shoulder Bag"],
                ["office", "wedding", "party", "everyday", "beach holiday"]),
    "backpack": (["unisex"], (35, 160), "one", ["nylon", "canvas", "recycled polyester", "leather"],
                 ["Daypack", "Rolltop Backpack", "Laptop Backpack"],
                 ["hiking", "everyday", "office"]),
    "watch": (["men", "women", "unisex"], (60, 400), "one", ["stainless steel", "leather strap", "titanium"],
              ["Field Watch", "Dress Watch", "Sport Watch"],
              ["office", "wedding", "gym", "everyday"]),
}

SIZES = {
    "alpha": ["XS", "S", "M", "L", "XL"],
    "waist": ["28", "30", "32", "34", "36"],
    "shoe": ["4", "5", "6", "7", "8", "9", "10", "11"],
    "one": ["One Size"],
}

COLOURS = ["black", "white", "navy", "grey", "beige", "brown", "olive", "burgundy",
           "red", "pink", "blue", "green", "cream", "tan"]

# Catalogue-side phrasing for tags. Shopper-side phrasing lives in evals/build_testset.py
STYLE_COPY = {
    "minimalist": ["clean lines", "understated", "pared-back design"],
    "classic": ["timeless", "heritage-inspired", "a wardrobe staple"],
    "casual": ["relaxed", "easy-going", "laid-back fit"],
    "sporty": ["athletic", "performance-led", "built for movement"],
    "bohemian": ["free-spirited", "boho detailing", "flowing silhouette"],
    "edgy": ["bold hardware", "statement", "rock-inspired"],
    "preppy": ["collegiate", "polished", "smart-casual"],
    "outdoor": ["trail-ready", "weather-resistant", "rugged"],
}
OCCASION_COPY = {
    "wedding": ["ceremony-ready", "perfect as a wedding guest", "for formal celebrations"],
    "office": ["workwear essential", "desk-to-dinner", "for the working week"],
    "gym": ["training-ready", "sweat-wicking", "for workouts"],
    "beach holiday": ["resort-ready", "for sunny getaways", "poolside"],
    "hiking": ["for the trail", "made for hillwalking", "for long days outdoors"],
    "date night": ["evening-ready", "for nights out", "dinner-date worthy"],
    "everyday": ["everyday wear", "for daily rotation", "weekend-friendly"],
    "party": ["party-ready", "made for celebrations", "after-dark glamour"],
}

N_PRODUCTS = 600


def build(seed: int = SEED, n_products: int = N_PRODUCTS) -> list[dict]:
    rng = random.Random(seed)
    products = []
    cats = list(CATEGORIES)
    for i in range(n_products):
        cat = cats[i % len(cats)]
        genders, (lo, hi), size_scheme, materials, nouns, occasions = CATEGORIES[cat]
        brand, mult, brand_styles = rng.choice(BRANDS)
        gender = rng.choice(genders)
        material = rng.choice(materials)
        noun = rng.choice(nouns)
        styles = sorted({rng.choice(brand_styles), rng.choice(list(STYLE_COPY))})[: rng.choice([1, 2])]
        occ = sorted(rng.sample(occasions, k=min(len(occasions), rng.choice([1, 2]))))
        base = round(rng.uniform(lo, hi) * mult / 5) * 5 - 0.01  # e.g. 79.99
        colours = rng.sample(COLOURS, k=rng.choice([1, 2, 3, 4]))
        sizes = SIZES[size_scheme]

        pid = f"P{i + 1:04d}"
        variants = []
        for c in colours:
            # some colours carry a small premium, to make "price" a variant-level fact
            cprice = round(base + rng.choice([0, 0, 0, 5, 10]), 2)
            for s in sizes:
                variants.append({
                    "sku": f"{pid}-{c[:3].upper()}-{s.replace(' ', '')}",
                    "colour": c,
                    "size": s,
                    "price": cprice,
                    "in_stock": rng.random() > 0.2,
                })

        style_words = [rng.choice(STYLE_COPY[s]) for s in styles]
        occ_words = [rng.choice(OCCASION_COPY[o]) for o in occ]
        name = f"{brand} {material.title()} {noun}"
        description = (
            f"A {', '.join(style_words)} {noun.lower()} in {material}. "
            f"{' '.join(w.capitalize() + '.' for w in occ_words)}"
        )
        products.append({
            "product_id": pid,
            "name": name,
            "brand": brand,
            "category": cat,
            "gender": gender,
            "material": material,
            "styles": styles,
            "occasions": occ,
            "description": description,
            "base_price": base,
            "variants": variants,
        })
    return products


def main() -> None:
    products = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(products, indent=1))
    n_sku = sum(len(p["variants"]) for p in products)
    print(f"wrote {len(products)} products / {n_sku} SKUs -> {OUT}")


if __name__ == "__main__":
    main()
