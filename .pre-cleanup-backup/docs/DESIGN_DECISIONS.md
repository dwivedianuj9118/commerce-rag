# Design decisions: what I chose, what I rejected, and the evidence

Each decision follows the same shape: **the problem**, **what I chose**, **what I rejected**,
**the evidence** (a number from `results/report.md` where there is one), and **the trade-off I
accepted**. If an interviewer asks "why X?", the answer is one of these.

---

## 1. Index parent products, not SKUs

**Problem.** A retailer with 100K SKUs might have only 10K real products. The rest are the same
boot in 8 sizes and 3 colours ("variant explosion").

**Chose.** One search document per parent product. After retrieval, a structured lookup finds
which variants are in stock and match the shopper's colour, size and budget
(`Catalogue.matching_variants` in `rag/retrieval.py`).

**Rejected.** One document per SKU.

**Evidence.** With a SKU index, the top 10 results averaged **2.4 distinct products**, and
Recall@10 fell to 0.20, lower than plain BM25 (0.25). Near-identical variants crowd each other
into the top slots.

**Trade-off.** Variant-specific text (e.g. "this colourway is limited edition") has to live on
the parent document or in the structured data, not in its own search document.

---

## 2. Hard constraints are filters, not ranking signals

**Problem.** "Under £150" and "in black" are facts. Search ranking of any kind (keyword or
vector) treats them as fuzzy hints.

**Chose.** A rule-based parser pulls out category, colour, budget, gender, brand and size.
Products that don't have an in-stock variant satisfying them are removed before ranking.

**Rejected.** Pure vector search, or letting the LLM "pay attention" to the budget.

**Evidence.** Without filters, **about 60%** of products in the top 5 broke a constraint the
shopper stated. With filters it's 0% (by construction). nDCG@10 went from 0.34 to **0.81**, the
biggest single gain in the project.

**Trade-off.** If the parser misreads a constraint, the filter confidently removes the right
products. Parser accuracy must be measured, which is why it has its own eval section.

---

## 3. A rule-based parser, not an LLM call per turn

**Chose.** Regexes and synonym lists (`rag/parser.py`). About 0.2 ms, free, and unit-tested.

**Rejected.** Asking the LLM to extract constraints as JSON on every turn. That adds a model
round-trip and tokens to every turn, and its behaviour can drift silently when the model changes.

**Honest caveat.** The parser scores 100%, but the test queries use the same category words as
the parser's synonym list, so that number mostly proves the regexes work. Real shopper
vocabulary would score lower. The fix path: log misses in production, and if they're common,
swap in an LLM parser that returns the same `Constraints` object. Nothing downstream changes.

---

## 4. Hybrid retrieval (BM25 + dense, fused with RRF)

**Why both.** BM25 is good at exact words (product names, materials). Dense embeddings are good
at meaning ("trail-ready" is close to "hiking").

**Why RRF (reciprocal rank fusion).** It combines rankings, not raw scores. BM25 scores and
cosine similarities are on different scales, and RRF avoids having to calibrate them. The
constant k=60 is the standard value from the original RRF paper.

**Evidence, and it is not flattering.** Hybrid (0.28) did *not* beat dense alone (0.30). The
dense model here is LSA (TF-IDF plus SVD), a weak offline stand-in, because the sandbox couldn't
download neural weights. So the conclusion is "unproven with this embedder", not "hybrid is
useless". The next experiment is to rerun with a neural embedder.

---

## 5. Synonym expansion, with a held-out test to keep it honest

**Chose.** A merchandising synonym table (`rag/synonyms.py`) that maps shopper words to
catalogue words ("wedding" to "ceremony-ready"). Real retailers maintain these from search logs.

**The risk.** I wrote both the table and the test queries, so the table could simply memorise
the test.

**How I controlled for it.** Half the test queries use *held-out* phrasings that are not in the
table ("my friend's nuptials", "nothing flashy").

**Evidence.** nDCG@10 is **0.96** on seen phrasing and **0.59** on held-out. That gap is the most
important finding in the project: the synonym table, not the retriever, is doing the semantic
work. Without the split, I'd have reported 0.81 and believed it.

---

## 6. Conversation state: merge constraints, and reset on a new category

**Chose.** Each turn's constraints are merged over the previous ones. A new category clears
colour and size, because "black" said about boots shouldn't carry over to "what about a bag?".
The budget carries over.

**Evidence.** On 60 three-turn conversations, stateful Recall@10 was **1.00** and stateless was
**0.09**. ("Any in black?" means nothing on its own.)

**Caveat.** 1.00 is high because the synthetic conversations are tidy and the filters are exact.
Real conversations include corrections ("actually, not black") that this merge rule doesn't
handle yet.

---

## 7. Grounding: the prompt asks, the validator enforces

**Problem.** The JD says the assistant "cannot hallucinate a product we do not sell". A prompt
instruction can't guarantee that, because LLM output is non-deterministic.

**Chose.** Two layers:
1. The generator only sees product cards and must cite products by id, e.g. `(P0123)`.
2. A deterministic validator (`rag/validator.py`) checks every answer. It checks that each id
   exists and was in the cards it was shown, that each price matches a real variant price, that
   each colour is in stock, and that no price appears without a product id. If any check fails,
   the answer is replaced with verified catalogue data and flagged.

**Evidence.** On 147 answers × 5 kinds of deliberate corruption (fake id, real-but-unshown
product, wrong price, unavailable colour, uncited product), it caught **100%**, with **0%** false
alarms on clean answers.

**What the stress test taught me.** Its first run caught only 95%. A made-up "£119.99 Nike Air
Max" slipped through because the validator let any price within 1p of the shopper's £120 budget
pass. I tightened that to an exact match.

**Caveat.** It checks facts it can parse (ids, prices, colours). It can't catch a wrong
*description* ("waterproof" when it isn't). That needs an LLM judge or attribute-level checks.

---

## 8. Fail loudly, never degrade quietly

The JD says "an agent that degrades quietly is worse than one that fails loudly". Three places
implement that here:

- **Out-of-catalogue gate.** No known category and no known words means refuse ("we don't sell
  PS5"), not recommend the nearest jumper. Result: 17/18 refused. The miss is "laptop", because
  the catalogue has "laptop backpacks". There were 3 false refusals out of 300 legitimate queries.
- **Impossible requests.** "Jeans under £16" returns no products and a sentence naming the
  constraint that blocks results. It never quietly shows £40 jeans. 16/16.
- **Validator fallback.** An unverifiable answer is replaced and flagged, not shipped.

---

## 9. "We don't carry X": a rule I shipped, measured and fixed

**v1.** Flag any word the catalogue has never seen. It caught every foreign brand, but it also
told **38%** of normal shoppers "we don't carry cousin's" or "we don't carry nuptials".

**v2.** Flag only unknown words that are capitalised, not the first word, and within three words
before a product noun ("any **Nike** trainers"). Spurious disclosures fell to **1%**, and it still
catches 12/12 foreign brands.

**Known gaps.** Lowercase "gucci bag" is missed, and "Scandi-style dress" is flagged wrongly.
Both are listed in the report. Both versions stay in the code (`Config.disclosure`), so the
comparison can be reproduced.

---

## 10. Cost: k (how many products go to the LLM) is the main lever

| k | Recall@k | Input tokens per turn | $ per 1K sessions |
|---|---|---|---|
| 3 | 0.80 | 342 | 3.89 |
| 5 | 0.83 | 416 | 4.30 |
| 10 | 0.88 | 563 | 5.12 |

I chose **k=5**. Going to 10 costs 35% more input tokens for 4 points of recall, and shoppers
rarely read past 3 to 5 suggestions anyway. The prices are illustrative ($4/$20 per million
tokens, Anthropic's list price for Claude Opus 5.5). AWS sets Bedrock pricing separately, so it
must be checked before quoting a number. Other cost levers to measure next are a smaller model
on the same eval, prompt caching of the fixed system prompt, and shorter product cards.

---

## 11. No framework (LangChain etc.)

About 750 lines of plain Python for the assistant itself (`rag/`), plus about 800 for data and evals. Every stage is timed separately, can be swapped behind a small
interface, and fits in my head. A framework would hide exactly the steps the JD asks about
("work out why the wrong thing came back"). In a team I'd reconsider it for orchestration
across several agents, but not for a single retrieval pipeline.

---

## 12. Offline by default, Bedrock behind a flag

The template generator makes the whole eval free, deterministic and repeatable in 40 seconds, so
it can run on every change. `BedrockGenerator` uses the Anthropic SDK's Bedrock client and is the
same one-line swap. It hasn't been run for the published numbers, because that costs money.
This is stated in the README rather than hidden.
