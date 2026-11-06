"""Interactive shopping assistant.

    python chat.py                      # offline, deterministic generator
    python chat.py --generator bedrock  # Claude on Amazon Bedrock (needs AWS credentials)
    python chat.py --debug              # show constraints, retrieved ids, timings, validation
Type 'reset' to start a new conversation, 'quit' to exit.
"""
import argparse

from rag.assistant import Assistant
from rag.generation import make_generator
from rag.retrieval import Catalogue, Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="template")
    ap.add_argument("--embedder", default="lsa")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cat = Catalogue()
    asst = Assistant(cat, Retriever(cat, "product", args.embedder), make_generator(args.generator))
    print("Shopping assistant ready. Try: 'black boots under £150 for hiking'")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("quit", "exit"):
            break
        if msg == "reset":
            asst.reset()
            print("(new conversation)")
            continue
        r = asst.turn(msg)
        print(f"\nassistant> {r.text}")
        if args.debug:
            print(f"  action={r.action} constraints={r.constraints.as_dict()} ids={r.product_ids}")
            print(f"  unknown={r.unknown_terms} disclosed={r.disclosed} timings_ms="
                  f"{ {k: round(v, 2) for k, v in r.timings_ms.items()} }")
            if r.validation:
                print(f"  grounding ok={r.validation.ok} issues={r.validation.issues}")
            if r.gen:
                print(f"  tokens in/out={r.gen.input_tokens}/{r.gen.output_tokens} backend={r.gen.backend}")


if __name__ == "__main__":
    main()
