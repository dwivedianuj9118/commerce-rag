"""Merchandising synonym table: shopper phrasing -> catalogue phrasing.

In production this comes from search logs and the merch team. Here it is hand-written,
which makes it a source of eval leakage: the test set therefore also uses *held-out*
phrasings that are NOT in this table, and the eval reports both slices separately.
"""
from .parser import CATEGORY_SYNONYMS, COLOURS, COLOUR_SYNONYMS

EXPANSIONS = {
    # style
    "minimal": "clean lines understated pared-back",
    "simple": "clean lines understated pared-back",
    "timeless": "timeless heritage-inspired wardrobe staple",
    "classic": "timeless heritage-inspired wardrobe staple",
    "relaxed": "relaxed easy-going laid-back",
    "casual": "relaxed easy-going laid-back",
    "sporty": "athletic performance-led movement",
    "athletic": "athletic performance-led movement",
    "boho": "free-spirited boho flowing",
    "bohemian": "free-spirited boho flowing",
    "edgy": "bold hardware statement rock-inspired",
    "preppy": "collegiate polished smart-casual",
    "smart": "collegiate polished smart-casual",
    "outdoorsy": "trail-ready weather-resistant rugged",
    "rugged": "trail-ready weather-resistant rugged",
    # occasion
    "wedding": "ceremony-ready wedding guest formal celebrations",
    "work": "workwear desk-to-dinner working week",
    "office": "workwear desk-to-dinner working week",
    "gym": "training-ready sweat-wicking workouts",
    "workout": "training-ready sweat-wicking workouts",
    "holiday": "resort-ready sunny getaways poolside",
    "beach": "resort-ready sunny getaways poolside",
    "hiking": "trail hillwalking outdoors",
    "hike": "trail hillwalking outdoors",
    "date": "evening-ready nights out dinner-date",
    "dinner": "evening-ready nights out dinner-date",
    "everyday": "everyday daily rotation weekend-friendly",
    "party": "party-ready celebrations after-dark glamour",
}

# Words the assistant treats as "known" when deciding whether a request is out of catalogue.
SHOPPER_VOCAB: set[str] = set(EXPANSIONS)
for syns in CATEGORY_SYNONYMS.values():
    for s in syns:
        SHOPPER_VOCAB.update(s.split())
SHOPPER_VOCAB.update(COLOURS)
SHOPPER_VOCAB.update(COLOUR_SYNONYMS)
SHOPPER_VOCAB.update("women women's womens ladies men men's mens her him wife husband girlfriend "
                     "boyfriend mum dad gift size shoes clothes outfit wear style colour color "
                     "cheap affordable price priced".split())


def expand(text: str) -> str:
    low = text.lower()
    extra = [v for k, v in EXPANSIONS.items() if k in low.split() or f" {k} " in f" {low} "]
    return text + (" " + " ".join(extra) if extra else "")
