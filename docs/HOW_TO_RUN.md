# How to run it

Everything below works offline with no API keys. A full run takes under a minute on a laptop.

## 1. Prerequisites

- Python 3.10 or newer (`python --version`)
- About 300 MB of disk space for the dependencies

## 2. Set up a virtual environment

macOS or Linux:

```bash
cd commerce-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd commerce-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Run everything with one command

macOS or Linux: `./run_all.sh`

Windows: run the four commands from step 4 one after another.

## 4. What each step does

| Command | What it does | Output |
|---|---|---|
| `python -m catalogue.generate` | Builds the synthetic catalogue: 600 products, 7,567 SKUs, 9 brands. It is seeded, so it comes out the same every time. | `data/catalogue.json` |
| `python -m evals.build_testset` | Builds 346 labelled test cases from catalogue attributes | `data/testset.json` |
| `python -m pytest -q` | 9 unit tests for the deterministic parts (parser, filters, state, validator) | `9 passed` |
| `python -m evals.run_eval` | Runs all six evaluation sections | `results/report.md`, `results/results.json` |
| `python chat.py --debug` | Interactive assistant | Terminal chat |

## 5. Reading the report

`results/report.md` has six sections:

1. **Retrieval ablation:** the same 240 queries against six configurations. Look at the jump from row D to row E (structured filters), and at row X (the rejected SKU-level index).
2. **Multi-turn:** stateful vs stateless on 60 three-turn conversations.
3. **Behaviour:** out-of-catalogue refusals, foreign-brand disclosure, impossible requests, and false alarms.
4. **Parser accuracy:** per field. It is 100%, and partly circular (see the design decisions guide).
5. **Validator stress test:** clean answers vs deliberately corrupted ones.
6. **Cost and latency:** tokens and dollars per session at k = 3, 5 and 10.

## 6. A two-minute live demo script

Run `python chat.py --debug` and type these in order. Each one shows a different part of the design.

| Type this | What it shows |
|---|---|
| `I need trainers for the gym` | Category parsed, hybrid retrieval, grounded answer with product ids |
| `under £80` | Conversation state: the budget is added to trainers |
| `any in white?` | State again: the colour is added, and variant resolution shows only white, in-stock sizes |
| `reset`, then `do you sell a PS5?` | Out-of-catalogue gate: refuses instead of recommending a jumper |
| `reset`, then `any Nike trainers?` | Foreign brand: "we don't carry Nike", then real alternatives |
| `reset`, then `jeans under £5` | Fails loudly: explicit no-match that names the blocking constraint |

With `--debug`, each turn also prints the parsed constraints, the retrieved ids, per-stage timings in ms, the grounding check result and the token counts.

## 7. Optional: Claude on Amazon Bedrock (costs money)

1. Install the Bedrock client: `pip install "anthropic[bedrock]"`
2. Set up AWS credentials: `aws configure`, or set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables.
3. Enable Claude model access in the Bedrock console for your region.
4. Optionally set `AWS_REGION` (default `us-east-1`) and `BEDROCK_MODEL_ID` (default `anthropic.claude-opus-5-5`).
5. Run `python chat.py --generator bedrock --debug`

The validator runs on Claude's answers exactly as it does on the template's. If Claude cites a product it wasn't shown or misquotes a price, you'll see `grounding ok=False` and the verified fallback answer.

## 8. Optional: a neural embedder

`python -m evals.run_eval --embedder bge` uses `BAAI/bge-small-en-v1.5` through fastembed (`pip install fastembed`). It downloads the model weights from HuggingFace the first time. This is the most useful next experiment: does held-out nDCG rise above 0.59?

## Troubleshooting

- **`ModuleNotFoundError: rag`:** run commands from inside the `commerce-rag` folder, using `python -m ...` as shown.
- **`data/testset.json` not found:** run the two generate steps first.
- **Numbers differ slightly from the README:** latency varies by machine. Everything else is seeded and should match exactly.
