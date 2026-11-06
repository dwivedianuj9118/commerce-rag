"""Answer generation.

TemplateGenerator works offline (no API key). BedrockGenerator calls Claude on Bedrock.
Both build the same prompt, so the token counts are comparable.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .parser import Constraints

SYSTEM_PROMPT = """You are the shopping assistant for a fashion retailer.
Rules that are not negotiable:
1. Recommend ONLY products from the PRODUCT CARDS below. Never mention any other product or brand as something we sell.
2. Every product you mention must include its id in brackets, e.g. (P0123).
3. Only quote prices, colours and sizes exactly as written on the card.
4. If no card fits the request, say so plainly and suggest how the shopper could relax their request.
5. If the shopper asks for a brand or item we don't carry, say we don't carry it before offering alternatives.
Keep answers under 120 words, warm and concise, in British English."""


@dataclass
class Card:
    product_id: str
    name: str
    brand: str
    category: str
    price: float
    colours: list[str]
    sizes: list[str]
    blurb: str

    def render(self) -> str:
        return (f"[{self.product_id}] {self.name} | {self.category} | £{self.price:.2f} | "
                f"colours: {', '.join(self.colours)} | sizes: {', '.join(self.sizes)} | {self.blurb}")


@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    backend: str
    refused_by_model: bool = False


def est_tokens(s: str) -> int:
    return max(1, round(len(s) / 4))


def build_user_prompt(history: list[str], constraints: Constraints, cards: list[Card],
                      notes: list[str]) -> str:
    convo = "\n".join(f"Shopper: {h}" for h in history[-3:])
    card_txt = "\n".join(c.render() for c in cards) or "(no products matched)"
    note_txt = "\n".join(f"- {n}" for n in notes)
    return (f"CONVERSATION SO FAR:\n{convo}\n\n"
            f"UNDERSTOOD CONSTRAINTS: {constraints.as_dict()}\n"
            f"NOTES:\n{note_txt or '- none'}\n\n"
            f"PRODUCT CARDS:\n{card_txt}\n\n"
            "Reply to the shopper's latest message.")


class TemplateGenerator:
    name = "template"

    def generate(self, history, constraints, cards, notes) -> GenResult:
        t0 = time.perf_counter()
        prompt = SYSTEM_PROMPT + build_user_prompt(history, constraints, cards, notes)
        lines = []
        for n in notes:
            lines.append(n)
        if not cards:
            lines.append("I couldn't find anything in our range that matches that.")
        else:
            n = min(3, len(cards))
            lines.append("Here's 1 option from our range:" if n == 1 else f"Here are {n} options from our range:")
            for i, c in enumerate(cards[:3], 1):
                lines.append(f"{i}. {c.name} ({c.product_id}): £{c.price:.2f}, "
                             f"in {', '.join(c.colours)}; sizes {', '.join(c.sizes)}.")
        text = "\n".join(lines)
        return GenResult(text, est_tokens(prompt), est_tokens(text), time.perf_counter() - t0, self.name)


class BedrockGenerator:
    # needs `pip install "anthropic[bedrock]"` + AWS creds
    # env: AWS_REGION (default us-east-1), BEDROCK_MODEL_ID
    name = "bedrock"

    def __init__(self, model: str | None = None, region: str | None = None):
        from anthropic import AnthropicBedrockMantle
        self.client = AnthropicBedrockMantle(aws_region=region or os.getenv("AWS_REGION", "us-east-1"))
        self.model = model or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-opus-5-5")

    def generate(self, history, constraints, cards, notes) -> GenResult:
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2000,  # answers are capped at ~120 words by the prompt
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},  # replies are short
            messages=[{"role": "user", "content": build_user_prompt(history, constraints, cards, notes)}],
        )
        refused = resp.stop_reason == "refusal"
        text = "".join(b.text for b in resp.content if b.type == "text")
        return GenResult(text, resp.usage.input_tokens, resp.usage.output_tokens,
                         time.perf_counter() - t0, self.name, refused_by_model=refused)


def make_generator(name: str = "template"):
    return {"template": TemplateGenerator, "bedrock": BedrockGenerator}[name]()
