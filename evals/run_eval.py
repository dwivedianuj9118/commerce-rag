"""Run every evaluation and write results/results.json + results/report.md.

    python -m evals.run_eval                 # offline, deterministic generator
    python -m evals.run_eval --generator bedrock --limit 40   # real LLM, spends money

Sections
  1. Retrieval ablation   (which architecture, measured on the same labelled queries)
  2. Multi-turn           (does conversation state actually help?)
  3. Behaviour            (refuse what we don't sell, disclose foreign brands, fail loudly on impossible asks)
  4. Constraint parser    (per-field extraction accuracy)
  5. Grounding validator  (does the guardrail catch corrupted answers? false alarms?)
  6. Cost & latency       (tokens/turn, £ per session, k vs quality trade-off)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from rag.assistant import Assistant, Config
from rag.generation import make_generator
from rag.parser import parse
from rag.retrieval import Catalogue, Retriever
from rag.validator import validate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# example $ per 1M tokens. check https://aws.amazon.com/bedrock/pricing/ for actual bedrock rates
PRICE_IN_PER_M = 4.00
PRICE_OUT_PER_M = 20.00


# ------------------------------------------------------------------ metrics
def recall_at(ranked, rel, k):
    ideal = [p for p, g in rel.items() if g == 2]
    if not ideal:
        return None
    return sum(1 for p in ranked[:k] if rel.get(p) == 2) / min(k, len(ideal))


def ndcg_at(ranked, rel, k):
    dcg = sum((2 ** rel.get(p, 0) - 1) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(ranked, rel):
    for i, p in enumerate(ranked):
        if rel.get(p) == 2:
            return 1 / (i + 1)
    return 0.0


def violation_rate(ranked, cat, hard, k):
    """Share of shown products that break a hard constraint (the shopper would see a wrong item)."""
    shown = ranked[:k]
    if not shown:
        return None
    from evals.build_testset import grade
    return sum(1 for p in shown if grade(cat, cat.by_id[p], hard, {}) == 0) / len(shown)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None


# ------------------------------------------------------------------ runners
def run_case(asst: Assistant, case: dict):
    asst.reset()
    res = None
    for t in case["turns"]:
        res = asst.turn(t)
    return res


def retrieval_block(asst, cases, cat, k=10):
    rows = {"recall@5": [], "recall@10": [], "ndcg@10": [], "mrr": [], "violations@5": [],
            "distinct_products_shown": [], "empty": []}
    for c in cases:
        r = run_case(asst, c)
        ranked = r.product_ids
        rows["recall@5"].append(recall_at(ranked, c["relevance"], 5))
        rows["recall@10"].append(recall_at(ranked, c["relevance"], 10))
        rows["ndcg@10"].append(ndcg_at(ranked, c["relevance"], 10))
        rows["mrr"].append(mrr(ranked, c["relevance"]))
        rows["violations@5"].append(violation_rate(ranked, cat, c["hard"], 5))
        rows["distinct_products_shown"].append(len(ranked))
        rows["empty"].append(1.0 if not ranked else 0.0)
    return {k_: mean(v) for k_, v in rows.items()} | {"n": len(cases)}


def section_retrieval(cat, retr_p, retr_s, ts, gen):
    configs = [
        ("A  BM25 only", retr_p, Config(mode="bm25", use_filters=False, use_expansion=False, k=10)),
        ("B  Dense only (LSA)", retr_p, Config(mode="dense", use_filters=False, use_expansion=False, k=10)),
        ("C  Hybrid (RRF)", retr_p, Config(mode="hybrid", use_filters=False, use_expansion=False, k=10)),
        ("D  Hybrid + synonyms", retr_p, Config(mode="hybrid", use_filters=False, use_expansion=True, k=10)),
        ("E  Hybrid + synonyms + filters", retr_p, Config(mode="hybrid", use_filters=True, use_expansion=True, k=10)),
        ("X  SKU-level index (rejected)", retr_s, Config(mode="hybrid", use_filters=False, use_expansion=True, k=10)),
    ]
    single = ts["single"]
    out = {}
    for name, retr, cfg in configs:
        a = Assistant(cat, retr, gen, cfg)
        out[name] = {
            "all": retrieval_block(a, single, cat),
            "seen_phrasing": retrieval_block(a, [c for c in single if c["phrasing"] == "seen"], cat),
            "heldout_phrasing": retrieval_block(a, [c for c in single if c["phrasing"] == "heldout"], cat),
            "vague_no_category": retrieval_block(a, [c for c in single if c["vague"]], cat),
        }
        print(f"  {name:34s} {out[name]['all']}")
    return out


def section_multiturn(cat, retr, ts, gen):
    out = {}
    for name, stateful in [("stateful (carry constraints)", True), ("stateless (last message only)", False)]:
        a = Assistant(cat, retr, gen, Config(stateful=stateful, k=10))
        out[name] = retrieval_block(a, ts["multi"], cat)
        print(f"  {name:34s} {out[name]}")
    return out


def section_behaviour(cat, retr, ts, gen, disclosure="brand_like"):
    a = Assistant(cat, retr, gen, Config(disclosure=disclosure))
    ooc = [run_case(a, c) for c in ts["out_of_catalogue"]]
    brands = [(c, run_case(a, c)) for c in ts["foreign_brand"]]
    imp = [run_case(a, c) for c in ts["impossible"]]
    legit = [(c, run_case(a, c)) for c in ts["single"] + ts["multi"]]
    false_refusals = [c["turns"][-1] for c, r in legit if r.action == "refuse_out_of_catalogue"]
    brand_ok = [r.action != "refuse_out_of_catalogue" and bool(r.disclosed)
                and all(cat.by_id[p]["category"] == c["hard"]["category"] for p in r.product_ids)
                for c, r in brands]
    missed_ooc = [c["turns"][0] for c, r in zip(ts["out_of_catalogue"], ooc) if r.action != "refuse_out_of_catalogue"]
    spurious_disclosure = [(c["turns"][-1], r.disclosed) for c, r in legit if r.disclosed]
    out = {
        "out_of_catalogue_refusal_rate": mean([1.0 if r.action == "refuse_out_of_catalogue" else 0.0 for r in ooc]),
        "missed_out_of_catalogue": missed_ooc,
        "false_refusal_rate_on_legit": round(len(false_refusals) / len(legit), 3),
        "false_refusals": false_refusals,
        "foreign_brand_disclosed_and_on_category": mean([1.0 if x else 0.0 for x in brand_ok]),
        "impossible_no_match_rate": mean([1.0 if r.action == "no_match" else 0.0 for r in imp]),
        "impossible_products_shown": sum(len(r.product_ids) for r in imp),
        "spurious_disclosure_rate_on_legit": round(len(spurious_disclosure) / len(legit), 3),
        "spurious_disclosure_examples": spurious_disclosure[:8],
    }
    for k, v in out.items():
        if not isinstance(v, list):
            print(f"  {k:42s} {v}")
    return out


def section_parser(ts):
    fields = ["category", "colour", "max_price", "gender"]
    hits = {f: [] for f in fields}
    errors = []
    for c in ts["single"]:
        got = parse(c["turns"][0]).constraints.as_dict()
        for f in fields:
            ok = got.get(f) == c["hard"].get(f)
            hits[f].append(1.0 if ok else 0.0)
            if not ok and len(errors) < 10:
                errors.append({"query": c["turns"][0], "field": f, "gold": c["hard"].get(f), "got": got.get(f)})
    out = {f"{f}_accuracy": mean(v) for f, v in hits.items()} | {"all_fields_exact": mean(
        [1.0 if all(hits[f][i] for f in fields) else 0.0 for i in range(len(ts["single"]))]), "errors": errors}
    print("  ", {k: v for k, v in out.items() if k != "errors"})
    return out


CORRUPTIONS = ["fake_id", "swap_to_uncited_real_product", "wrong_price", "unavailable_colour", "uncited_product_with_price"]


def corrupt(text, kind, cat, rng, context_ids):
    import re
    ids = re.findall(r"\bP\d{4}\b", text)
    if not ids:
        return None
    pid = ids[0]
    p = cat.by_id[pid]
    if kind == "fake_id":
        return text.replace(pid, "P9" + pid[2:], 1)
    if kind == "swap_to_uncited_real_product":
        other = rng.choice([q for q in cat.by_id if q not in context_ids])
        return text.replace(pid, other, 1)
    if kind == "wrong_price":
        m = re.search(rf"{pid}\): £(\d+\.\d\d)", text)
        prices = {v["price"] for v in p["variants"]}
        bad = round(float(m.group(1)) * 0.8) + 0.99   # plausible-looking, e.g. £79.99 -> £64.99
        while any(abs(bad - x) < 0.02 for x in prices):
            bad += 5
        return text[:m.start(1)] + f"{bad:.2f}" + text[m.end(1):]
    if kind == "unavailable_colour":
        avail = {v["colour"] for v in p["variants"] if v["in_stock"]}
        missing = [c for c in ["black", "white", "navy", "red", "green", "pink"] if c not in avail]
        if not missing:
            return None
        return text.replace(f"({pid}):", f"({pid}): also comes in {missing[0]}.", 1)
    if kind == "uncited_product_with_price":
        return text + "\n4. You might also like the Nike Air Max 90 at £119.99."
    raise ValueError(kind)


def section_validator(cat, retr, ts, gen):
    rng = random.Random(3)
    a = Assistant(cat, retr, gen, Config())
    clean_ok, detected = [], {k: [] for k in CORRUPTIONS}
    for c in ts["single"][:150]:
        r = run_case(a, c)
        if not r.product_ids:
            continue
        ctx = set(r.product_ids)
        budget = tuple(x for x in (r.constraints.max_price, r.constraints.min_price) if x is not None)
        clean_ok.append(1.0 if validate(r.gen.text, cat, ctx, budget).ok else 0.0)
        for kind in CORRUPTIONS:
            bad = corrupt(r.gen.text, kind, cat, rng, ctx)
            if bad is not None:
                detected[kind].append(0.0 if validate(bad, cat, ctx, budget).ok else 1.0)
    out = {"clean_answers_passing (1 - false alarm rate)": mean(clean_ok), "n_clean": len(clean_ok)}
    out |= {f"detect_{k}": mean(v) for k, v in detected.items()}
    for k, v in out.items():
        print(f"  {k:42s} {v}")
    return out


def section_cost(cat, retr, ts, gen):
    """Tokens and latency per turn at different k (cards sent to the LLM), against recall@k."""
    out = {}
    cases = ts["single"] + ts["multi"]
    for k in (3, 5, 10):
        a = Assistant(cat, retr, gen, Config(k=k))
        tin, tout, lat, stages = [], [], [], {"parse": [], "retrieve": [], "generate": [], "validate": []}
        recalls = []
        sessions = []
        for c in cases:
            a.reset()
            s_in = s_out = 0
            for t in c["turns"]:
                r = a.turn(t)
                if r.gen:
                    tin.append(r.gen.input_tokens)
                    tout.append(r.gen.output_tokens)
                    s_in += r.gen.input_tokens
                    s_out += r.gen.output_tokens
                lat.append(r.timings_ms.get("total", 0))
                for st in stages:
                    if st in r.timings_ms:
                        stages[st].append(r.timings_ms[st])
            sessions.append((s_in, s_out))
            recalls.append(recall_at(r.product_ids, c["relevance"], k))
        cost_turn = statistics.fmean(tin) * PRICE_IN_PER_M / 1e6 + statistics.fmean(tout) * PRICE_OUT_PER_M / 1e6
        cost_session = statistics.fmean(s[0] * PRICE_IN_PER_M / 1e6 + s[1] * PRICE_OUT_PER_M / 1e6 for s in sessions)
        out[f"k={k}"] = {
            f"recall@{k}": mean(recalls),
            "input_tokens_per_turn": round(statistics.fmean(tin)),
            "output_tokens_per_turn": round(statistics.fmean(tout)),
            "usd_per_turn": round(cost_turn, 5),
            "usd_per_session": round(cost_session, 5),
            "usd_per_1k_sessions": round(cost_session * 1000, 2),
            "latency_ms_p50": pct(lat, 0.5), "latency_ms_p95": pct(lat, 0.95),
            "stage_ms_p50": {s: pct(v, 0.5) for s, v in stages.items()},
        }
        print(f"  k={k}: {out[f'k={k}']}")
    return out


# ------------------------------------------------------------------ report
def fmt_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join("" if x is None else str(x) for x in r) + " |\n"
    return s


def write_report(res, meta):
    L = [f"# Evaluation report\n\nGenerated {meta['when']} · generator `{meta['generator']}` · "
         f"embedder `{meta['embedder']}` · {meta['n_products']} products / {meta['n_skus']} SKUs · "
         f"eval wall time {meta['seconds']}s\n"]
    L.append("## 1. Retrieval ablation (240 single-turn queries, top-10)\n")
    rows = [[n, v["all"]["recall@10"], v["all"]["ndcg@10"], v["all"]["mrr"], v["all"]["violations@5"],
             v["heldout_phrasing"]["ndcg@10"], v["vague_no_category"]["ndcg@10"], v["all"]["distinct_products_shown"]]
            for n, v in res["retrieval"].items()]
    L.append(fmt_table(["config", "Recall@10", "nDCG@10", "MRR", "constraint violations@5",
                        "nDCG@10 held-out phrasing", "nDCG@10 vague (no category)", "distinct products in top 10"], rows))
    L.append("\n## 2. Multi-turn (60 three-turn conversations, scored after the last turn)\n")
    L.append(fmt_table(["mode", "Recall@10", "nDCG@10", "violations@5", "empty results"],
                       [[n, v["recall@10"], v["ndcg@10"], v["violations@5"], v["empty"]] for n, v in res["multiturn"].items()]))
    b = res["behaviour"]
    L.append("\n## 3. Behaviour\n")
    L.append(fmt_table(["check", "result"], [
        ["Out-of-catalogue requests refused", b["out_of_catalogue_refusal_rate"]],
        ["False refusals on legitimate queries", b["false_refusal_rate_on_legit"]],
        ["Foreign brand: disclosed + on-category alternatives", b["foreign_brand_disclosed_and_on_category"]],
        ["Impossible request -> explicit no-match", b["impossible_no_match_rate"]],
        ["Products shown for impossible requests", b["impossible_products_shown"]],
        ["Spurious 'we don't carry X' on legitimate queries", b["spurious_disclosure_rate_on_legit"]],
    ]))
    v1 = res.get("behaviour_v1_any_unknown")
    if v1:
        L.append(f"\nv1 disclosure rule (flag every unknown word) for comparison: spurious disclosure "
                 f"{v1['spurious_disclosure_rate_on_legit']}, foreign brands caught {v1['foreign_brand_disclosed_and_on_category']}.\n")
    if b["missed_out_of_catalogue"]:
        L.append(f"\nMissed out-of-catalogue: {b['missed_out_of_catalogue']}\n")
    if b["false_refusals"]:
        L.append(f"\nFalse refusals: {b['false_refusals'][:10]}\n")
    if b["spurious_disclosure_examples"]:
        L.append(f"\nSpurious disclosure examples: {b['spurious_disclosure_examples']}\n")
    p = res["parser"]
    L.append("\n## 4. Constraint parser accuracy (240 queries)\n")
    L.append(fmt_table(["field", "accuracy"], [[k, v] for k, v in p.items() if k != "errors"]))
    if p["errors"]:
        L.append("\nSample errors:\n\n" + "\n".join(f"- `{e['query']}`: {e['field']} gold={e['gold']} got={e['got']}" for e in p["errors"]) + "\n")
    L.append("\n## 5. Grounding validator (corrupted-answer stress test)\n")
    L.append(fmt_table(["check", "rate"], [[k, v] for k, v in res["validator"].items()]))
    L.append("\n## 6. Cost and latency by k (cards sent to the generator)\n")
    L.append(f"Token counts are estimates (~4 chars/token) of the prompt the Bedrock backend would send; "
             f"prices are illustrative (${PRICE_IN_PER_M}/${PRICE_OUT_PER_M} per 1M in/out). Latency excludes the LLM call.\n\n")
    L.append(fmt_table(["k", "recall@k", "in tok/turn", "out tok/turn", "$/session", "$/1k sessions", "p50 ms", "p95 ms"],
                       [[k, v[f"recall@{k.split('=')[1]}"], v["input_tokens_per_turn"], v["output_tokens_per_turn"],
                         v["usd_per_session"], v["usd_per_1k_sessions"], v["latency_ms_p50"], v["latency_ms_p95"]]
                        for k, v in res["cost"].items()]))
    (RESULTS / "report.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="template")
    ap.add_argument("--embedder", default="lsa")
    args = ap.parse_args()
    t0 = time.time()
    cat = Catalogue()
    ts = json.loads((ROOT / "data" / "testset.json").read_text())
    gen = make_generator(args.generator)
    print("building indexes...")
    retr_p = Retriever(cat, "product", args.embedder)
    retr_s = Retriever(cat, "sku", args.embedder)
    res = {}
    print("1. retrieval ablation"); res["retrieval"] = section_retrieval(cat, retr_p, retr_s, ts, gen)
    print("2. multi-turn"); res["multiturn"] = section_multiturn(cat, retr_p, ts, gen)
    print("3. behaviour (v1 disclosure rule, kept for comparison)")
    res["behaviour_v1_any_unknown"] = section_behaviour(cat, retr_p, ts, gen, "any_unknown")
    print("3. behaviour"); res["behaviour"] = section_behaviour(cat, retr_p, ts, gen)
    print("4. parser"); res["parser"] = section_parser(ts)
    print("5. validator"); res["validator"] = section_validator(cat, retr_p, ts, gen)
    print("6. cost"); res["cost"] = section_cost(cat, retr_p, ts, gen)
    RESULTS.mkdir(exist_ok=True)
    meta = {"when": time.strftime("%Y-%m-%d %H:%M"), "generator": args.generator, "embedder": args.embedder,
            "n_products": len(cat.products), "n_skus": len(cat.skus), "seconds": round(time.time() - t0, 1)}
    (RESULTS / "results.json").write_text(json.dumps({"meta": meta, **res}, indent=1))
    write_report(res, meta)
    print(f"done in {meta['seconds']}s -> results/report.md")


if __name__ == "__main__":
    main()
