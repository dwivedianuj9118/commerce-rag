# commerce-rag

**A grounded shopping assistant over a product catalogue, with an evaluation harness that measures every design choice.**

It answers questions like *"black boots under £150 for hiking"*, remembers context across turns (*"any in brown?"*), refuses to invent products, and says plainly when nothing matches. Each part of the pipeline was compared against the alternatives on the same 346 labelled test cases, and this README reports what the numbers showed, including the unflattering parts.

- Runs fully offline in under a minute, with no API keys
- Optional Claude-on-Amazon-Bedrock generator, switched on with one flag
- ~750 lines of plain Python, no RAG framework, every stage timed and swappable

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Offline](https://img.shields.io/badge/runs-offline-lightgrey)

---

## Contents

- [Quickstart](#quickstart)
- [Example session](#example-session)
- [How it works](#how-it-works)
- [Results](#results)
- [What the evaluation showed](#what-the-evaluation-showed)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Project layout](#project-layout)
- [Further docs](#further-docs)

## Quickstart

Requires Python 3.10 or newer.

```bash
git clone https://github.com/dwivedianuj9118/commerce-rag.git
cd commerce-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

./run_all.sh          # build data, run tests, run the full evaluation
python chat.py --debug
```

Or step by step:

```bash
python -m catalogue.generate     # synthetic catalogue: 600 products, 7,567 SKUs, 9 brands (seeded)
python -m evals.build_testset    # 346 labelled test cases
python -m pytest -q              # unit tests
python -m evals.run_eval         # writes results/report.md and results/results.json
python chat.py --debug           # interactive CLI; shows constraints, ids, timings, grounding checks
```

### Using Claude on Amazon Bedrock (optional, costs money)

```bash
pip install "anthropic[bedrock]"
export AWS_REGION=us-east-1                       # default
export BEDROCK_MODEL_ID=anthropic.claude-opus-5-5 # default
python chat.py --generator bedrock
python -m evals.run_eval --generator bedrock
```

You need AWS credentials and model access enabled in the Bedrock console.

### Using a neural embedder (optional)

```bash
pip install fastembed
python chat.py --embedder bge     # BAAI/bge-small-en-v1.5, downloads weights once
```

## Example session

```text
you> black boots under £150 for hiking
assistant> Here are 3 options from our range:
1. <product name> (<product id>): £<price>, in black; sizes <in-stock sizes>.
...

you> any in brown?
           ↳ still boots, still under £150; only the colour changes

you> do you sell a PS5?
assistant> Sorry, we don't sell ps5. We're a fashion retailer: clothing, shoes, bags and watches.

you> jeans under £16
assistant> Nothing is in stock under £16; dropping that gives options.
```

Type `reset` to start a new conversation and `quit` to exit. `--debug` prints the parsed constraints, retrieved product ids, per-stage timings and the grounding check for every turn.

## How it works

```
shopper message
   │
   ▼
1. Constraint parser (rules, ~0.2 ms)       "black boots under £150 for hiking"
   │                                        -> {category: boots, colour: black, max_price: 150}
   ▼
2. Conversation state                       new constraints merge over old ones;
   │                                        a new category resets colour and size
   ▼
3. Out-of-catalogue gate                    nothing recognised and no category -> refuse
   ▼
4. Hybrid retrieval over PARENT products    BM25 + dense, fused with RRF; synonym expansion;
   │                                        hard filters keep only products with an in-stock
   │                                        variant matching colour, size and budget
   ▼
5. Variant resolution                       one card per product, built from matching in-stock SKUs
   ▼
6. Generator (template | Claude on Bedrock) may only cite products from the cards, by id
   ▼
7. Grounding validator                      every id, price and colour is checked against the
                                            catalogue and the cards; on failure the answer is
                                            replaced with verified data and flagged
```

| Stage | File |
|---|---|
| Constraint parser | `rag/parser.py` |
| Retrieval, filters, embedders | `rag/retrieval.py` |
| Synonym table | `rag/synonyms.py` |
| Conversation loop and state | `rag/assistant.py` |
| Generators and prompt | `rag/generation.py` |
| Grounding validator | `rag/validator.py` |

## Results

Full numbers are in [`results/report.md`](results/report.md). The test set has 240 single-turn queries, 60 three-turn conversations, 18 out-of-catalogue requests, 12 requests for brands the store doesn't carry, and 16 impossible requests.

**Retrieval (240 queries, top 10)**

| Config | nDCG@10 | Recall@10 | Items in top 5 breaking a stated constraint |
|---|---|---|---|
| A. BM25 only | 0.22 | 0.24 | 64% |
| B. Dense only (LSA) | 0.30 | 0.31 | 61% |
| C. Hybrid (RRF) | 0.28 | 0.31 | 61% |
| D. Hybrid + synonym expansion | 0.34 | 0.39 | 65% |
| **E. Hybrid + synonyms + structured filters (shipped)** | **0.81** | **0.85** | **0%** |
| X. One document per SKU (rejected) | 0.22 | 0.20 | 54%, with only 2.4 distinct products in the top 10 |

**Behaviour**

| Check | Result |
|---|---|
| Out-of-catalogue requests refused | 17/18 (94%) |
| Legitimate queries wrongly refused | 1% |
| Unstocked brand ("any Nike trainers?"): disclosed, then real alternatives shown | 12/12 |
| Impossible request ("jeans under £16"): explicit no-match, nothing shown | 16/16 |
| Validator catches deliberately corrupted answers (5 corruption types) | 100% |
| Validator wrongly flags clean answers | 0% |
| Follow-up turns, stateful vs last-message-only (Recall@10) | 1.00 vs 0.09 |

**Cost and latency.** With 5 product cards, a turn is about 416 input and 71 output tokens, roughly $4.30 per 1,000 sessions at an illustrative $4 / $20 per million tokens. Pipeline latency without the LLM call is 6 ms p50 and 8 ms p95. Going from 5 to 10 cards adds 35% input tokens for 4 points of recall.

## What the evaluation showed

1. **Structured filters did most of the work.** Adding them (D → E) more than doubled nDCG, from 0.34 to 0.81. "Under £150" and "in black" are facts; a ranking model treats them as hints. Without filters, about 60% of what the shopper sees breaks a constraint they stated.
2. **The headline number hides a gap.** The shipped system scores nDCG 0.96 on queries whose style and occasion words are in the synonym table, and **0.59** on held-out phrasings ("my friend's nuptials" instead of "a wedding"). The synonym table, not the retriever, carries the semantic load. The held-out slice is the only reason this is visible.
3. **Hybrid didn't beat dense here.** RRF with a weak BM25 signal slightly diluted the dense ranking (0.28 vs 0.30). The dense model is an offline LSA stand-in, so this says more about the embedder than about hybrid search.
4. **Indexing every SKU is a trap.** With one document per variant, the same boot in eight sizes crowds out everything else: 2.4 distinct products in the top 10. Indexing parent products and resolving variants afterwards fixes this and shrinks the index 12x.
5. **Conversation state beats any retrieval tweak for follow-ups.** "Any in black?" means nothing without the previous turns.

### Fixes the evaluation drove

| Change | Before | After |
|---|---|---|
| "We don't carry X" only for capitalised unknown words before a product noun | 38% of legitimate queries got a false "we don't carry 'cousin's'" | 1% |
| Validator ignores a price only when it exactly equals the shopper's budget | A hallucinated £119.99 slipped through under a £120 budget | Caught |
| Stress test keeps corrupted prices positive and plausible | Produced "£-0.01", which the price regex missed | Fixed |

Both versions of the disclosure rule remain in the code (`Config.disclosure`) so the comparison can be reproduced.

### Designs considered and rejected

- **One document per SKU**: collapses result diversity and recall (see X above).
- **Pure vector search**: "under £50" is a number, not a vibe.
- **An LLM call to extract constraints every turn**: adds a round-trip and tokens to every turn and is hard to test. The rule parser is a drop-in replacement point behind the same `Constraints` type.
- **Trusting the prompt to prevent hallucination**: the prompt is a request; the deterministic validator is the guarantee.
- **Silently relaxing constraints when nothing matches**: the shopper is told which constraint blocks results instead.

More detail, with the evidence for each choice, is in [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## Limitations

- **Synthetic data.** The catalogue and queries are generated; relevance labels come from catalogue attributes, so they are exact but narrow.
- **The parser's 100% accuracy is partly circular.** Test queries use words from the parser's own synonym lists. Real search logs would give an honest number.
- **No real LLM in the published numbers.** The default generator is deterministic, so its hallucination rate is zero by construction; that's why the validator is tested on deliberately corrupted answers.
- **Token counts are estimates** (~4 characters per token). Check [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) before quoting costs.
- **The default dense model is LSA** (TF-IDF + SVD), so everything runs offline. `--embedder bge` switches to a neural model.
- **Unstocked-brand detection relies on capitalisation**: "any gucci bags" in lowercase isn't flagged.

## Roadmap

- [ ] Re-run the eval with a neural embedder (bge-small, or Titan/Cohere on Bedrock) and see whether held-out nDCG moves from 0.59 toward 0.96
- [ ] Run the Bedrock generator on a fixed slice; report real hallucination-caught rate, refusals, tokens and latency
- [ ] Add an LLM judge for tone and helpfulness, calibrated against hand labels
- [ ] Nightly eval runs that alert on metric drops, not just errors
- [ ] Export the catalogue as schema.org `Product` / `ProductGroup` JSON-LD with `hasVariant` offers

## Project layout

```
catalogue/generate.py    synthetic catalogue generator (seeded)
data/                    catalogue.json, testset.json
rag/                     parser, retrieval, synonyms, generation, validator, assistant
evals/build_testset.py   labelled queries, with seen and held-out phrasings
evals/run_eval.py        ablations, behaviour, parser, validator stress test, cost, latency
results/                 report.md, results.json, run_log.txt
tests/                   unit tests for parser, filters, state and validator
docs/                    run guide and design decisions
chat.py                  interactive CLI
run_all.sh               build, test and evaluate in one command
```

## Further docs

- [How to run it](docs/HOW_TO_RUN.md): setup on macOS, Linux and Windows, what each step does, and a demo script
- [Design decisions](docs/DESIGN_DECISIONS.md): each choice, what was rejected, and the evidence

## License

[MIT](LICENSE) © 2026 Anuj Dwivedi
