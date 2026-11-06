# commerce-rag

A small shopping assistant built on RAG, plus an eval setup to check whether each piece actually helps.

You ask it things like "black boots under £150 for hiking", it finds matching products from a catalogue, keeps track of what you said in earlier messages ("any in brown?"), and won't make up products or prices. If nothing matches, it says so.

I built it mainly to answer one question: in a product search assistant, which parts matter? So most of the code is the evaluation, not the assistant itself. Everything runs offline, no API keys needed. There's an optional Claude (AWS Bedrock) backend too.

## Running it

Python 3.10+.

```bash
git clone https://github.com/dwivedianuj9118/commerce-rag.git
cd commerce-rag
python -m venv .venv
source .venv/bin/activate        # windows: .venv\Scripts\activate
pip install -r requirements.txt

./run_all.sh                     # generate data, run tests, run evals
python chat.py --debug           # chat with it
```

`run_all.sh` just runs these in order:

```bash
python -m catalogue.generate     # fake catalogue: 600 products, 7,567 SKUs, 9 brands
python -m evals.build_testset    # 346 test cases
python -m pytest -q
python -m evals.run_eval         # writes results/report.md
```

In the chat, `reset` starts a new conversation and `quit` exits. `--debug` prints the parsed constraints, retrieved ids, timings and the validation result for each turn.

To use Claude on Bedrock instead of the template answers (needs AWS credentials, and it costs money):

```bash
pip install "anthropic[bedrock]"
python chat.py --generator bedrock
```

To use a real embedding model instead of the offline one: `pip install fastembed`, then pass `--embedder bge`.

## What it looks like

```
you> black boots under £150 for hiking
assistant> Here are 3 options from our range:
1. Mariner & Co Waterproof Leather Hiking Boots (P0314): £124.99, in black; sizes 4, 5, 7, 8, 9.
2. Palma Leather Ankle Boots (P0170): £114.99, in black; sizes 4, 5, 6, 7, 8, 9, 10, 11.
3. Palma Waterproof Leather Chelsea Boots (P0250): £129.99, in black; sizes 4, 6, 8, 9, 10, 11.

you> any in brown?
assistant> Here's 1 option from our range:
1. Loom Street Waterproof Leather Lace-Up Boots (P0410): £84.99, in brown; sizes 4, 6, 7, 8, 9, 10, 11.

you> do you sell a PS5?
assistant> Sorry, we don't sell ps5. We're a fashion retailer: clothing, shoes, bags and watches.

you> jeans under £16
assistant> Nothing is in stock under £16; dropping that gives options.
I couldn't find anything in our range that matches that.
```

(The default generator is a simple template. With `--generator bedrock` Claude writes the reply, but only from the same product data.)

## How it works

```
message
  -> parse constraints (regex)      category, colour, price range, size, gender, brand
  -> merge with previous turns      new category clears colour/size
  -> out-of-catalogue check         "PS5" -> refuse instead of searching
  -> search parent products         BM25 + dense, merged with RRF, synonym expansion,
                                    then hard filters on in-stock variants
  -> build product cards            real price / colours / sizes from matching SKUs
  -> generate answer                template or Claude, can only cite card ids
  -> validate                       every id, price and colour checked against the catalogue;
                                    if anything is wrong the answer is replaced
```

Code is in `rag/`: `parser.py`, `retrieval.py`, `synonyms.py`, `assistant.py`, `generation.py`, `validator.py`.

## Results

Full output: [results/report.md](results/report.md). Test set: 240 single queries, 60 three-turn conversations, 18 "we don't sell that" queries, 12 queries for brands not in the catalogue, 16 impossible ones.

Retrieval on 240 queries:

| Setup | nDCG@10 | Recall@10 | Top-5 items that break a constraint |
|---|---|---|---|
| BM25 only | 0.22 | 0.24 | 64% |
| Dense only (LSA) | 0.30 | 0.31 | 61% |
| Hybrid (RRF) | 0.28 | 0.31 | 61% |
| Hybrid + synonyms | 0.34 | 0.39 | 65% |
| Hybrid + synonyms + filters (what I use) | 0.81 | 0.85 | 0% |
| One doc per SKU (dropped) | 0.22 | 0.20 | 54% |

Other checks:

| Check | Result |
|---|---|
| "We don't sell that" queries refused | 17/18 |
| Normal queries wrongly refused | 1% |
| Unknown brand ("any Nike trainers?"): says so, then shows alternatives | 12/12 |
| Impossible request ("jeans under £16"): says no match, shows nothing | 16/16 |
| Validator catches corrupted answers (5 kinds of corruption) | 100% |
| Validator flags a correct answer by mistake | 0% |
| Follow-up questions, with vs without conversation state (Recall@10) | 1.00 vs 0.09 |

Cost: with 5 product cards a turn is about 414 input and 71 output tokens, around $3.80 per 1,000 conversations at $4 / $20 per million tokens (example rates, check Bedrock pricing). Without the LLM call, a turn takes about 6.5 ms (p50).

## Things I learned

**Filters did most of the work.** Adding hard filters took nDCG from 0.34 to 0.81. "Under £150" is a fact, not a ranking signal. Without filters, around 60% of the results broke something the user asked for.

**The overall number was hiding a problem.** With phrasing that's in my synonym table, nDCG is 0.96. With phrasing that isn't ("my friend's nuptials" instead of "wedding"), it drops to 0.59. So the synonym table is doing the semantic work, not the retriever. I only noticed because I held back some phrasings when building the test set.

**Hybrid didn't beat dense.** 0.28 vs 0.30. But the dense model here is LSA, not a real embedding model, so I don't trust this result much yet.

**Don't index every SKU.** With one document per size/colour, the same boot in 8 sizes fills the top 10 (only 2.4 different products on average). Indexing products and picking variants afterwards fixed it, and the index got 12x smaller.

**Conversation state matters a lot for follow-ups.** "Any in black?" means nothing on its own. Recall without state: 0.09.

Bugs the evals caught:

- The "we don't carry X" message fired on any unknown word, so 38% of normal queries got things like "we don't carry 'cousin's'". Now it only fires for capitalised words right before a product noun: 1%.
- A made-up £119.99 price got through the validator when the budget was £120, because budget amounts were skipped too loosely. Now only an exact match is skipped.
- The stress test generated "£-0.01" as a wrong price, which the price regex didn't even match. Fixed both.

The old version of the brand rule is still in the code (`Config.disclosure`) so you can compare.

More on why I picked what I picked: [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md).

## Limitations

- The catalogue and queries are generated, so the labels are clean but not realistic.
- The parser scores 100%, but the test queries use words from the parser's own lists, so that number doesn't mean much.
- The published numbers use the template generator, not a real LLM. That's why the validator is tested on deliberately broken answers.
- Token counts are estimated (~4 chars per token).
- Brand detection depends on capitalisation, so "any gucci bags" isn't caught.

## Next

- Run the evals with a real embedding model (`--embedder bge`) and see if the 0.59 on unseen phrasing goes up
- Run the Bedrock generator on a fixed set of queries and measure how often the validator catches it
- Run the evals on a schedule and alert when a number drops
- Export the catalogue as schema.org Product / ProductGroup JSON-LD

## Folder structure

```
catalogue/    fake catalogue generator
data/         catalogue.json, testset.json
rag/          the assistant
evals/        test set builder + eval runner
results/      latest eval output
tests/        unit tests
docs/         setup guide, design decisions
chat.py       CLI
run_all.sh    runs everything
```

## License

MIT
