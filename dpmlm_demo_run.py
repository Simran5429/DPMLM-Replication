import argparse
import re
import math
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Sensitive span detection

SIMPLE_CITY_LIST = {
    "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
    "Canberra", "Hobart", "Darwin", "Auckland", "Wellington",
    "London", "New York", "San Francisco", "Tokyo", "Delhi", "Mumbai"
}

def find_sensitive_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return character spans likely to be sensitive:
    - long numbers (>=3 digits)
    - phone-like patterns
    - emails
    - simple IDs
    - capitalized names and common city names
    """
    spans = []

    # numbers (>=3 digits)
    for m in re.finditer(r"\b\d{3,}\b", text):
        spans.append((m.start(), m.end()))

    # phone numbers or general phone patterns
    phone_patterns = [
        r"\b0\d{1,2}\s?\d{3}\s?\d{3}\b",       # e.g., 0412 345 678
        r"\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b"
    ]
    for pat in phone_patterns:
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end()))

    # emails
    for m in re.finditer(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", text):
        spans.append((m.start(), m.end()))

    # simple IDs like "account number", "ID", "ssn", etc. (mask the following token if present)
    for m in re.finditer(r"\b(account|acct|id|ssn|pass(?:port)?|license)\b[:\s]*([A-Za-z0-9\-_/]{2,})?", text, flags=re.I):
        if m.group(2):
            s, e = m.span(2)
            spans.append((s, e))

    # capitalized tokens that look like names or cities
    for m in re.finditer(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*\b", text):
        token = m.group(0)
        if token in SIMPLE_CITY_LIST or len(token.split()) <= 3:
            spans.append((m.start(), m.end()))

    # merge overlapping spans
    spans.sort()
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


# Mask-and-fill with RoBERTa


class MaskFiller:
    def __init__(self, model_name: str = "roberta-base", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.mask_token = self.tokenizer.mask_token  # "<mask>" for RoBERTa

    @torch.no_grad()
    def fill_one(self, text: str, epsilon: float, top_k: int = 20) -> str:
        """
        Replace exactly one <mask> with a token sampled from the model distribution.
        epsilon controls randomness via temperature: temp = clamp(1.5/epsilon, 0.7..3.0)
        """
        if self.mask_token not in text:
            return text

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        mask_positions = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            return text

        outputs = self.model(**inputs)
        logits = outputs.logits  # [batch, seq, vocab]
        b, pos = mask_positions[0].tolist()
        vocab_logits = logits[b, pos, :]  # [vocab_size]

        # temperature from epsilon (smaller epsilon → higher temp → more noise)
        eps = max(1e-6, float(epsilon))
        temp = max(0.7, min(3.0, 1.5 / eps))
        probs = torch.softmax(vocab_logits / temp, dim=-1)

        # restrict to top_k
        top_k = max(5, int(top_k))
        topk_probs, topk_idx = torch.topk(probs, k=top_k)
        topk_probs = topk_probs / topk_probs.sum()  # renormalize

        # sample
        choice = torch.multinomial(topk_probs, num_samples=1).item()
        token_id = topk_idx[choice].item()

        # decode chosen token alone
        token_str = self.tokenizer.decode([token_id]).strip()

        # simple cleanup for RoBERTa byte-level merges (leading 'Ġ' means space)
        token_str = token_str.replace("Ġ", " ")
        # compact spacing around punctuation
        token_str = re.sub(r"\s+([,.!?;:])", r"\1", token_str)

        # replace first mask occurrence in raw text
        new_text = text.replace(self.mask_token, token_str, 1)
        return new_text

    def rewrite(self, text: str, epsilon: float, max_masks: int = 8, top_k: int = 20) -> str:
        """
        Iteratively fill all <mask> tokens (up to max_masks).
        """
        out = text
        for _ in range(max_masks):
            if self.mask_token not in out:
                break
            out = self.fill_one(out, epsilon=epsilon, top_k=top_k)
        return out


def mask_sensitive_spans(raw: str, mask_token: str) -> str:
    spans = find_sensitive_spans(raw)
    if not spans:
        return raw
    # apply from end to start so offsets don't move
    s = raw
    for start, end in reversed(spans):
        s = s[:start] + mask_token + s[end:]
    # normalize spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    ap = argparse.ArgumentParser(description="Minimal privacy-style rewriting demo (mask-and-fill).")
    ap.add_argument("--input", required=True, help="Input text")
    ap.add_argument("--epsilon", type=float, default=1.0, help="Privacy-strength control (smaller → noisier)")
    ap.add_argument("--topk", type=int, default=20, help="Top-k sampling for each mask")
    args = ap.parse_args()

    mf = MaskFiller("roberta-base")

    original = args.input.strip()
    masked = mask_sensitive_spans(original, mf.mask_token)
    rewritten = mf.rewrite(masked, epsilon=args.epsilon, top_k=args.topk)

    print("Original:", original)
    print("Masked:  ", masked)
    print(f"Rewritten (epsilon={args.epsilon}):", rewritten)


if __name__ == "__main__":
    main()
