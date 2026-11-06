# Interview guide

How to present and defend this project. Learn the pitch word for word. For the questions, learn
the *reasoning*, then say it in your own words. Every number here comes from `results/report.md`.

---

## Part 1: The 30-second pitch

> "I built a grounded shopping assistant over a catalogue with variant explosion: 600 products
> and 7,500 SKUs. The assistant is the small part. Most of the work is the evaluation harness:
> 346 labelled test cases, six retrieval configurations compared on the same queries, a stress
> test for the hallucination guard, and cost per session. The most useful thing it told me was
> uncomfortable: my system scores 0.96 nDCG on phrasing my synonym table knows, but 0.59 on
> phrasing it doesn't. So the synonym table, not the retriever, was doing the semantic work. I
> only caught that because I held out phrasings when I built the test set."

## Part 2: The 2-minute walkthrough (draw the pipeline as you talk)

1. **Parse.** Rules pull out hard constraints: category, colour, budget, gender, size.
   It takes 0.2 ms, costs nothing, and is unit-tested.
2. **Remember.** Constraints merge across turns, so "any in white?" still means trainers under
   £80. Stateful recall is 1.00, stateless 0.09.
3. **Gate.** If I recognise nothing ("do you sell a PS5?"), refuse instead of retrieving.
4. **Retrieve.** Hybrid BM25 plus dense over *parent products*, with hard filters. Filters took
   nDCG from 0.34 to 0.81.
5. **Resolve variants.** Build a card per product from its matching in-stock SKUs.
6. **Generate.** The LLM only sees the cards and must cite product ids.
7. **Validate.** Deterministic checks on every id, price and colour. If a check fails, the answer
   is replaced with verified data and flagged. It fails loudly.

Close with: "The prompt *asks* the model not to hallucinate. The validator *enforces* it."

---

## Part 3: Concepts to know cold

| Term | Plain-English meaning | Where it shows up here |
|---|---|---|
| **RAG** | Retrieve relevant documents, then have the LLM answer using only those | The whole pipeline |
| **BM25** | Classic keyword ranking: rare words that match count more; very long documents count less | `rank-bm25` in `retrieval.py` |
| **Embedding / dense retrieval** | Turn text into a vector so similar meanings are close together | LSA here; bge or Titan next |
| **RRF** | Merge several rankings by summing 1/(60 + rank). It uses ranks, so the scores don't need calibrating | `Retriever._rrf` |
| **Recall@k** | Of the truly relevant items, how many made the top k (capped at k) | Report sections 1, 2 and 6 |
| **nDCG@k** | Rewards putting the *most* relevant items *highest*. 1.0 is a perfect ordering | Main ranking metric |
| **MRR** | 1 divided by the position of the first correct result, averaged over queries | Report section 1 |
| **Grounding** | Every claim in the answer traces back to the source data | `validator.py` |
| **Hallucination** | The model states something not supported by the data (a fake product or price) | What the validator catches |
| **Ablation** | Remove or add one component at a time to see what each one contributes | Configs A to E and X |
| **Held-out set** | Test data deliberately unlike what the system was tuned on | Held-out phrasings |
| **Variant explosion** | One product becomes many SKUs (sizes × colours) | 600 products, 7,567 SKUs |
| **Bedrock** | AWS's managed service for calling foundation models such as Claude | `BedrockGenerator` |

---

## Part 4: Likely questions, with strong answers

### About measurement (the JD's main signal)

**Q: How do you know it works?**
A: "Three kinds of evidence. First, retrieval metrics on 240 labelled queries whose answers come
from catalogue attributes, so they were fixed before any system saw them. Second, behaviour
checks: does it refuse what we don't sell, disclose foreign brands, and fail loudly on
impossible requests. Third, a stress test of the guardrail itself: I corrupted 147 answers in
five ways and checked it catches every one without flagging clean answers. And I'm explicit
about what it *doesn't* prove: no real LLM was in the loop for those numbers."

**Q: Where did your evaluation tell you something you didn't want to hear?**
A: "Two places. The held-out phrasing gap, 0.96 vs 0.59, which means my synonym table was doing
the work. And my first 'we don't carry X' rule, which told 38% of normal shoppers 'we don't carry
cousin's'. I changed the rule to only flag capitalised unknown words before a product noun, and
false alarms fell to 1% with every foreign brand still caught."

**Q: What did you change after measuring?**
A: "The disclosure rule (38% to 1% false alarms). The validator: the stress test showed a fake
£119.99 product slipping past a £120 budget because I allowed a 1p tolerance, so I made it an
exact match. And I found a bug in my own test generator that produced negative prices. Both
versions of the disclosure rule are still in the code, so anyone can reproduce the comparison."

**Q: Isn't your ground truth circular, since you generated both the data and the labels?**
A: "Partly, and I say so. The labels come from structured attributes, so they're exact but
narrow. The parser's 100% is circular, because the queries use its own synonym words. I
controlled for the worst leak, the synonym table, with held-out phrasings. With real data I'd
label from click and purchase logs plus a small hand-labelled set."

**Q: Your multi-turn recall is 1.00. Isn't that suspicious?**
A: "Yes, and I wouldn't claim it holds in production. The synthetic conversations are tidy and
the filters are exact. Real conversations include corrections like 'actually, not black', which
my merge rule doesn't handle. The honest takeaway is the contrast with stateless (0.09), not the
1.00."

### About architecture

**Q: Why not just put everything in a vector database?**
A: "Budgets and colours are facts, not similarities. Without structured filters, about 60% of
what the shopper saw broke a constraint they'd stated. Adding filters took nDCG from 0.34 to
0.81, the biggest gain in the whole project."

**Q: Why index products and not SKUs?**
A: "I measured it. A SKU-level index put the same boot in 8 sizes into the top results, so the
top 10 averaged 2.4 distinct products and recall fell below plain BM25. Variants are a structured
lookup after retrieval."

**Q: Did hybrid search help?**
A: "Not here. Hybrid was 0.28 and dense alone 0.30. But my dense model is LSA, a weak offline
stand-in, so I'd call it unproven rather than disproven. Rerunning with a neural embedder is my
next experiment. If held-out nDCG moves towards the 0.96 seen number, I can shrink the synonym
table."

**Q: Why a rule parser instead of asking the LLM to extract JSON?**
A: "It's 0.2 ms, free and testable, and it removes a model call from every turn. It returns a
`Constraints` object, so an LLM parser is a drop-in replacement if real queries show the rules
missing things. I'd decide that from logged misses, not up front."

**Q: How do you stop hallucinated products?**
A: "Three layers. The model only sees retrieved cards. It must cite every product by id. And a
deterministic validator checks every id, price and colour against the catalogue on every answer.
If a check fails, the shopper gets verified data, not the model's text. The prompt is a request;
the validator is the guarantee."

**Q: What can your validator NOT catch?**
A: "Claims it can't parse. If the model calls a jacket 'waterproof' and it isn't, the id and
price are right, so it passes. Fixing that needs attribute-level checks or an LLM judge
calibrated against human labels."

**Q: Why no LangChain?**
A: "The pipeline is about 750 lines, and every stage is timed and swappable. The JD asks me to
'work out why the wrong thing came back', and that's easier when I can see each step. For
orchestration across several specialised agents, I'd evaluate a framework on its merits."

**Q: Where would orchestration across agents beat a single model here?**
A: "Probably for separate jobs with different tools and budgets: for example a sizing or
returns-policy agent next to the product finder. For the core 'find me X' loop, a single
pipeline with deterministic steps is cheaper, faster and easier to evaluate. I'd only split it
if the eval showed one prompt doing two jobs badly."

### About scale, AWS and cost

**Q: How does this scale to 100K SKUs?**
A: "The parent-product design helps: 100K SKUs might be 10–15K products. In-memory BM25 and a
brute-force vector search still work at that size, but in production I'd use OpenSearch, which
does BM25, vector search and filtering in one engine, or Bedrock Knowledge Bases. The
architecture stays the same: filters before ranking, variants resolved after."

**Q: What does it cost?**
A: "About 416 input and 71 output tokens per turn at k=5, which is roughly $4.30 per 1,000
sessions at an illustrative $4/$20 per million tokens. Bedrock pricing is set by AWS, so I'd
check it before quoting. The main lever is k: going from 5 to 10 products costs 35% more input
for 4 points of recall. Next I'd test a smaller model against the same eval and cache the fixed
system prompt."

**Q: Which model would you use on Bedrock, and why?**
A: "The code defaults to Claude Opus 5.5 at low effort, since the answers are short. But that's a
starting point, not a decision. The right choice is the cheapest model that holds the eval
numbers, so I'd run the same eval on a smaller model and compare hallucination-caught rate,
refusal accuracy and cost per session."

**Q: How would you monitor this in production?**
A: "Re-run the fixed eval on every change and nightly as the catalogue changes. Alert on metric
drops, not just errors. Log the validator's failure rate per day, because a rising rate means
the model or the data changed. Track p95 latency and token cost per session."

### About the second system (AEO, machine-readable discovery)

**Q: How does this relate to making the catalogue readable by ChatGPT or Perplexity?**
A: "The same parent/variant modelling maps directly onto schema.org `ProductGroup` with
`hasVariant` offers in JSON-LD, which is what external agents read. The hard part the JD
mentions, knowing whether the response was any good, is the same discipline: build the eval
first. For citation share, I'd run a fixed set of shopper prompts against each assistant on a
schedule and measure how often our products are cited."

### Curveballs

**Q: What would you rebuild?**
A: "I'd build the held-out test split *before* the synonym table, not alongside it, so I couldn't
unconsciously tune to it. And I'd get a real LLM run earlier, because a deterministic generator
makes the hallucination numbers trivially perfect. That's why I had to test the validator with
corrupted answers instead."

**Q: What's the weakest part?**
A: "Semantic retrieval on unseen phrasing: 0.59 nDCG. The embedder is a weak stand-in, and the
synonym table is covering for it."

**Q: What did you learn?**
A: "That the headline number hides the story. 0.81 overall looked finished. Slicing by phrasing
showed it was 0.96 and 0.59 mixed together."

---

## Part 5: Things NOT to claim

- Don't say "it doesn't hallucinate". Say "every id, price and colour is verified, and on a
  stress test the validator caught 100% of five corruption types".
- Don't say "it works with Claude". Say "the Bedrock backend is written and uses the same
  validator, but the published numbers use the deterministic generator".
- Don't present $4.30 per 1,000 sessions as a Bedrock quote. It's an illustrative rate, and token
  counts are estimates.
- Don't say "hybrid search is better". Here it wasn't, with this embedder.
- Don't oversell the parser's 100% or the multi-turn 1.00. Explain why they're high.

## Part 6: Before the interview

1. Run `./run_all.sh` and check your numbers match the README.
2. Practise the demo script in `docs/HOW_TO_RUN.md` with `--debug` on.
3. Be able to open and explain `rag/validator.py` and `Retriever.search` line by line. They're
   short.
4. If you can, run the neural-embedder experiment (`--embedder bge`) on your own machine. Whatever
   happens to held-out nDCG, it's a fresh result you can talk about.
