"""Fast unit tests for the deterministic parts. Run: python -m pytest -q"""
import pytest

from rag.assistant import Assistant, brand_like_terms
from rag.parser import Constraints, parse
from rag.retrieval import Catalogue, Retriever
from rag.validator import validate


@pytest.fixture(scope="module")
def cat():
    return Catalogue()


@pytest.fixture(scope="module")
def asst(cat):
    return Assistant(cat, Retriever(cat))


def test_parser_extracts_hard_constraints():
    c = parse("women's black boots under £150 for hiking").constraints
    assert (c.category, c.colour, c.max_price, c.gender) == ("boots", "black", 150.0, "women")


def test_parser_between_and_synonyms():
    c = parse("sneakers between £40 and £90 in gray").constraints
    assert (c.category, c.colour, c.min_price, c.max_price) == ("trainers", "grey", 40.0, 90.0)


def test_new_category_resets_colour_but_keeps_budget():
    old = Constraints(category="boots", colour="black", max_price=100)
    new = old.merge(Constraints(category="handbag"))
    assert new.category == "handbag" and new.colour is None and new.max_price == 100


def test_filters_never_return_constraint_violations(cat, asst):
    asst.reset()
    r = asst.turn("black boots under £150")
    assert r.product_ids
    for pid in r.product_ids:
        assert cat.by_id[pid]["category"] == "boots"
        assert cat.matching_variants(pid, r.constraints)


def test_out_of_catalogue_is_refused(asst):
    asst.reset()
    assert asst.turn("do you sell a PS5?").action == "refuse_out_of_catalogue"


def test_impossible_request_shows_nothing(asst):
    asst.reset()
    r = asst.turn("jeans under £3")
    assert r.action == "no_match" and r.product_ids == []


def test_multi_turn_carries_state(asst):
    asst.reset()
    asst.turn("I need trainers")
    asst.turn("under £80")
    r = asst.turn("any in white?")
    assert r.constraints.as_dict() == {"category": "trainers", "max_price": 80.0, "colour": "white"}


def test_brand_like_terms():
    assert brand_like_terms("any Nike trainers?", ["nike"]) == ["Nike"]
    assert brand_like_terms("something for my cousin's wedding", ["cousin's"]) == []


def test_validator_passes_clean_and_catches_fakes(cat):
    p = cat.products[0]
    v = next(v for v in p["variants"] if v["in_stock"])
    ok = f"1. {p['name']} ({p['product_id']}): £{v['price']:.2f}, in {v['colour']}."
    ctx = {p["product_id"]}
    assert validate(ok, cat, ctx).ok
    assert not validate(ok.replace(p["product_id"], "P9999"), cat, ctx).ok          # invented product
    assert not validate(ok.replace(f"{v['price']:.2f}", "1.99"), cat, ctx).ok        # wrong price
    assert not validate(ok + "\nAlso the Nike Air Max at £119.99.", cat, ctx).ok      # uncited product
    # a £119.99 product must not hide behind a £120 budget (bug found by the stress test)
    assert not validate("Try the Air Max at £119.99.", cat, ctx, budget_prices=(120.0,)).ok
