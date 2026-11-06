# Grounded shopping assistant: a measured RAG demo

A conversational-commerce assistant that answers over a product catalogue with variant explosion,
refuses to invent products, and comes with an evaluation harness that says how well each part
works and where it breaks.

The assistant is the small part. The evaluation is the point: every architecture choice below
was compared against the alternatives on the same labelled queries, and the README records what
the numbers showed, including the parts that are not flattering.

Everything runs offline in about 40 seconds with no API keys. A Claude-on-Bedrock generator is
included and switches on with one flag.

## Documentation

- [How to run it](docs/HOW_TO_RUN.md): setup on macOS, Linux and Windows, what each step does, and a live demo script
- [Design decisions](docs/DESIGN_DECISIONS.md): 12 choices, what was rejected, and the evidence for each
- [Interview guide](docs/INTERVIEW_GUIDE.md): pitch, concepts, likely questions with answers, and what not to claim

## Quickstart

```bash
pip install -r requirements.txt
python -m catalogue.generate        # 600 products, 7,567 SKUs, 9 brands (seeded)
python -m evals.build_testset       # 346 labelled cases
python -m pytest -q                 # 9 unit tests
python -m evals.run_eval            # writes results/report.md and results/results.json
python chat.py --debug              # talk to it; shows constraints, ids, timings, grounding checks
```

With AWS credentials and Bedrock model access (this costs money):

```bash
pip install "anthropic[bedrock]"
python chat.py --generator bedrock
python -m evals.run_eval --generator bedrock
```

## Architecture

```
shopper message
   │
   ▼
1. Constraint parser (rules, ~0.2 ms)      "black boots under £150 for hiking"
   │   -> {category: boots, colour: black, max_price: 150} + free text
   ▼
2. Conversation state                       new constraints merge over old ones;
   │                                        a new category resets colour and size
   ▼
3. Out-of-catalogue gate                    no known category and no known words -> refuse
   │                                        ("do you sell a PS5?")
   ▼
4. Hybrid retrieval over PARENT products    BM25 + dense (RRF fusion), synonym expansion,
   │                                        hard filters: only products with an in-stock
   │                                        variant matching colour/size/budget survive
   ▼
5. Variant resolution                       builds a card per product from the matching
   │                                        in-stock SKUs (real price, colours, sizes)
   ▼
6. Generator (template | Claude on Bedrock) may only cite products from the cards, by id
   ▼
7. Grounding validator                      every id, price and colour in the answer is
                                            checked against the catalogue and the cards.
                                            Fail -> the answer is replaced with verified
                                            data and flagged. It fails loudly.
```

Code: `rag/parser.py`, `rag/retrieval.py`, `rag/assistant.py`, `rag/generation.py`, `rag/validator.py`.

## Headline results

Full numbers are in [`results/report.md`](results/report.md). The test set has 240 single-turn
queries, 60 three-turn conversations, 18 out-of-catalogue requests, 12 foreign-brand requests
and 16 impossible requests.

| Retrieval config (240 queries) | nDCG@10 | Recall@10 | Wrong items shown in top 5 |
|---|---|---|---|
| A. BM25 only | 0.22 | 0.25 | 64% |
| B. Dense only (LSA) | 0.30 | 0.31 | 61% |
| C. Hybrid (RRF) | 0.28 | 0.31 | 61% |
| D. Hybrid + synonym expansion | 0.34 | 0.39 | 64% |
| **E. Hybrid + synonyms + structured filters (shipped)** | **0.81** | **0.85** | **0%** |
| X. SKU-level index (rejected) | 0.22 | 0.20 | 54%, and only 2.4 distinct products in the top 10 |

| Behaviour | Result |
|---|---|
| Out-of-catalogue requests refused | 17/18 (94%) |
| Legitimate queries wrongly refused | 1% |
| Foreign brand ("any Nike trainers?"): disclosed, then real alternatives shown | 12/12 |
| Impossible request ("jeans under £16"): explicit no-match, no products shown | 16/16 |
| Grounding validator: corrupted answers caught (5 corruption types) | 100% |
| Grounding validator: clean answers wrongly flagged | 0% |
| Multi-turn, stateful vs last-message-only (Recall@10) | 1.00 vs 0.09 |

Cost at k=5 cards: about 416 input and 71 output tokens per turn, roughly **$4.30 per 1,000
sessions** at the illustrative rate of $4/$20 per million tokens. Latency without the LLM call
is 6 ms p50 and 8 ms p95. Going from k=5 to k=10 adds 35% input tokens for +4 points of recall.

## What the evaluation told me

**1. Structured filters did almost all of the work.** Going from D to E more than doubles nDCG
(0.34 to 0.81). Hard constraints like "under £150" and "in black" are facts, and a ranking model
of any kind treats them as hints. Without filters, about 60% of what the shopper sees breaks a
constraint they stated. With them, it's 0% by construction.

**2. The headline number hides a big gap, and the held-out slice exposed it.** The shipped
system scores nDCG 0.96 on queries whose style and occasion words appear in the synonym table,
and **0.59** on queries phrased differently ("my friend's nuptials" instead of "a wedding",
"nothing flashy" instead of "minimal"). The synonym table, not the retriever, is carrying the
semantic part. I only know this because the test set was built with held-out phrasings from the
start. Without that split the report would say 0.81 and look finished.

**3. Hybrid did not beat dense here.** RRF with a weak BM25 signal slightly diluted the dense
ranking (0.28 vs 0.30). The dense model is an offline LSA stand-in, so this says more about the
embedder than about hybrid search. It's the next experiment (see below), not a conclusion.

**4. Indexing every SKU is a trap.** With one document per variant, the top 10 results average
**2.4 distinct products**: the same boot in 8 sizes crowds everything else out. Recall drops
below even BM25. Indexing parent products and resolving variants afterwards fixes this, and it
also shrinks the index 12x.

**5. Conversation state matters more than any retrieval tweak for follow-ups.** "Any in black?"
means nothing without the previous two turns. Stateless recall is 0.09.

## What I changed after measuring

| Change | Before | After | How I found it |
|---|---|---|---|
| "We don't carry X" disclosure: only flag capitalised, unknown words sitting before a product noun, instead of every unknown word | 38% of legitimate queries got a false "we don't carry 'cousin's'" | 1% | Behaviour eval, spurious-disclosure metric |
| Validator: budget figures are exempt from price checks only on an exact match | A hallucinated £119.99 product slipped through when the budget was £120 | Caught | Corrupted-answer stress test (95% to 100%) |
| Stress-test generator: wrong prices stay positive and plausible | The test produced "£-0.01", which the price regex didn't even see | Fixed; validator regex also accepts negatives | Reading the misses by hand |

Both versions of the disclosure rule stay in the code (`Config.disclosure`), and the report
prints the v1 numbers next to v2 so the comparison can be reproduced.

## Designs I rejected, and why

- **One document per SKU.** Measured above: it collapses result diversity and recall. Variants
  are a structured lookup after retrieval, not a retrieval problem.
- **Pure vector search.** Embeddings blur exact constraints: "under £50" is a number, not a
  vibe. Measured above: about 60% of shown items break a stated constraint.
- **An LLM call to extract constraints every turn.** It would add a model round-trip and tokens
  to every turn, and it's hard to test. The rule parser takes 0.2 ms and its accuracy is
  measured. The honest caveat is that its 100% score is partly circular (next section). If real
  queries show it missing things, the LLM parser is a drop-in replacement behind the same
  `Constraints` type.
- **Trusting the prompt to prevent hallucination.** The system prompt forbids inventing
  products, but the prompt is a request, not a guarantee. The validator is the guarantee: it's
  deterministic, runs on every answer, and replaces anything it can't verify.
- **Silently relaxing constraints when nothing matches.** A shopper who asked for jeans under
  £16 is shown nothing, plus a sentence saying which constraint blocks results. Showing £40
  jeans without saying so is the "degrades quietly" failure the JD warns about.

## What these numbers don't cover

- **The catalogue and queries are synthetic.** Relevance labels come from catalogue attributes,
  so they are exact but narrow. Real shoppers are messier, and real relevance is partly taste.
- **The parser's 100% is circular.** Test queries use category words from the parser's own
  synonym list. It measures that the regexes work, not that they cover real vocabulary. Real
  search logs would give the honest number.
- **No real LLM was used for the published numbers.** The default generator is deterministic,
  so its hallucination rate is zero by construction. That's why the validator is tested on
  deliberately corrupted answers instead. Measuring Claude's actual hallucination and refusal
  rates needs `--generator bedrock`, which costs money and wasn't run.
- **Token counts are estimates** (about 4 characters per token) of the prompt the Bedrock
  backend would send. Prices are illustrative first-party list rates. Bedrock pricing is set by
  AWS, so check https://aws.amazon.com/bedrock/pricing/ before quoting a figure.
- **The dense model is LSA** (TF-IDF plus SVD), because this sandbox couldn't download neural
  embedding weights. `--embedder bge` switches to `BAAI/bge-small-en-v1.5` where HuggingFace is
  reachable.
- **Foreign-brand detection relies on capitalisation.** "any gucci bags" in lowercase would not
  be flagged. "Scandi-style dress" is flagged wrongly. Both are visible in the report.

## What I'd do next

1. **Swap in a neural embedder** (bge-small locally, Titan or Cohere embeddings on Bedrock), then
   re-run. The question is whether held-out nDCG moves from 0.59 towards the 0.96 seen-phrasing
   number. If it does, the synonym table can shrink. If it doesn't, hybrid isn't worth its
   complexity.
2. **Run the Bedrock generator on a fixed slice** and report real hallucination-caught rate,
   refusal behaviour, tokens and latency. Add an LLM judge for tone and helpfulness, calibrated
   against a small set of hand labels before trusting it.
3. **Track drift over time:** re-run the same eval nightly as the catalogue changes, and alert
   when a metric drops rather than when something errors.
4. **The second system in the JD (AEO).** The same catalogue can be exported as schema.org
   `Product`/`ProductGroup` JSON-LD with `hasVariant` offers, which is the machine-readable feed
   external agents read. The parent/variant modelling here maps onto it directly.

## How this maps to the role

| JD asks for | Where it is here |
|---|---|
| Retrieval over ~100K SKUs with variant explosion | Parent-level index plus variant resolution; the SKU-level design measured and rejected |
| Grounded generation that can't hallucinate a product | Id-only citations plus a deterministic validator, stress-tested on 5 corruption types |
| Context across turns | Constraint state with category reset; stateful vs stateless measured |
| "Work out why the wrong thing came back" | Per-slice metrics (seen, held-out, vague), constraint-violation rate, sample failures in the report |
| Instrument latency, token cost per session, answer quality | Per-stage timings, tokens per turn, $ per session, k vs recall trade-off |
| "An agent that degrades quietly is worse than one that fails loudly" | Validator fallback, explicit no-match, out-of-catalogue refusal |
| Bedrock, Python | `BedrockGenerator` in `rag/generation.py` (Anthropic SDK Mantle client) |

## Layout

```
catalogue/generate.py   synthetic catalogue (seeded)
rag/                    parser, retrieval, synonyms, generation, validator, assistant
evals/build_testset.py  labelled queries from catalogue attributes (seen and held-out phrasing)
evals/run_eval.py       ablations, behaviour, parser, validator stress test, cost and latency
results/report.md       latest numbers
chat.py                 interactive CLI
tests/                  unit tests for parser, filters, state, validator
docs/                   run guide, design decisions, interview guide
run_all.sh              build data, test, evaluate in one command
```
