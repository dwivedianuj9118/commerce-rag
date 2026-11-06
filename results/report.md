# Evaluation report

Generated 2026-11-06 09:19 · generator `template` · embedder `lsa` · 600 products / 7567 SKUs · eval wall time 42.7s

## 1. Retrieval ablation (240 single-turn queries, top-10)

| config | Recall@10 | nDCG@10 | MRR | constraint violations@5 | nDCG@10 held-out phrasing | nDCG@10 vague (no category) | distinct products in top 10 |
|---|---|---|---|---|---|---|---|
| A  BM25 only | 0.242 | 0.216 | 0.278 | 0.644 | 0.182 | 0.164 | 9.875 |
| B  Dense only (LSA) | 0.309 | 0.296 | 0.41 | 0.607 | 0.251 | 0.214 | 9.875 |
| C  Hybrid (RRF) | 0.31 | 0.284 | 0.378 | 0.61 | 0.241 | 0.205 | 9.875 |
| D  Hybrid + synonyms | 0.392 | 0.335 | 0.446 | 0.648 | 0.24 | 0.307 | 9.875 |
| E  Hybrid + synonyms + filters | 0.846 | 0.807 | 0.816 | 0.0 | 0.587 | 0.62 | 7.596 |
| X  SKU-level index (rejected) | 0.199 | 0.224 | 0.438 | 0.534 | 0.149 | 0.205 | 2.442 |


## 2. Multi-turn (60 three-turn conversations, scored after the last turn)

| mode | Recall@10 | nDCG@10 | violations@5 | empty results |
|---|---|---|---|---|
| stateful (carry constraints) | 1.0 | 0.986 | 0.0 | 0.0 |
| stateless (last message only) | 0.097 | 0.084 | 0.96 | 0.0 |


## 3. Behaviour

| check | result |
|---|---|
| Out-of-catalogue requests refused | 0.944 |
| False refusals on legitimate queries | 0.01 |
| Foreign brand: disclosed + on-category alternatives | 1.0 |
| Impossible request -> explicit no-match | 1.0 |
| Products shown for impossible requests | 0 |
| Spurious 'we don't carry X' on legitimate queries | 0.01 |


v1 disclosure rule (flag every unknown word) for comparison: spurious disclosure 0.38, foreign brands caught 1.0.


Missed out-of-catalogue: ['do you sell a laptop?']


False refusals: ['looking for chilled something for a birthday bash', "low-key something for my friend's nuptials under £120", 'looking for a bit punk something for client meetings under £350']


Spurious disclosure examples: [('any Scandi-style dress for my nine-to-five?', ['Scandi-style']), ('any Scandi-style beige kicks?', ['Scandi-style']), ('any Scandi-style blouse for a summer ceremony under £110?', ['Scandi-style'])]


## 4. Constraint parser accuracy (240 queries)

| field | accuracy |
|---|---|
| category_accuracy | 1.0 |
| colour_accuracy | 1.0 |
| max_price_accuracy | 1.0 |
| gender_accuracy | 1.0 |
| all_fields_exact | 1.0 |


## 5. Grounding validator (corrupted-answer stress test)

| check | rate |
|---|---|
| clean_answers_passing (1 - false alarm rate) | 1.0 |
| n_clean | 147 |
| detect_fake_id | 1.0 |
| detect_swap_to_uncited_real_product | 1.0 |
| detect_wrong_price | 1.0 |
| detect_unavailable_colour | 1.0 |
| detect_uncited_product_with_price | 1.0 |


## 6. Cost and latency by k (cards sent to the generator)

Token counts are estimates (~4 chars/token) of the prompt the Bedrock backend would send; prices are illustrative ($4.0/$20.0 per 1M in/out). Latency excludes the LLM call.


| k | recall@k | in tok/turn | out tok/turn | $/session | $/1k sessions | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| k=3 | 0.807 | 341 | 71 | 0.00344 | 3.44 | 6.5 | 7.53 |
| k=5 | 0.836 | 414 | 71 | 0.0038 | 3.8 | 6.48 | 7.84 |
| k=10 | 0.877 | 555 | 71 | 0.00449 | 4.49 | 6.63 | 7.9 |
