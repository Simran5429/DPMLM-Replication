import argparse
import csv
import re
import sys
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Ensure UTF-8 output on Windows terminals / redirection
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Simple gazetteer to catch common proper nouns 
CITIES = {
    "Sydney","Melbourne","Brisbane","Perth","Adelaide","Canberra","Hobart","Darwin",
    "Auckland","Wellington","London","New","York","San","Francisco","Tokyo","Delhi","Mumbai"
}

def find_sensitive_spans(text: str) -> List[Tuple[int, int]]:
    """Return character spans of likely sensitive substrings."""
    spans: List[Tuple[int, int]] = []

    # URLs, @handles, hashtags
    for pat in [r"https?://\S+", r"@\w+", r"#\w+"]:
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end()))

    # Emails
    for m in re.finditer(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b", text):
        spans.append((m.start(), m.end()))

    # Numbers (>= 3 digits)
    for m in re.finditer(r"\b\d{3,}\b", text):
        spans.append((m.start(), m.end()))

    # Phone-like
    phone = [
        r"\b0\d{1,2}\s?\d{3}\s?\d{3,4}\b",
        r"\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b",
    ]
    for pat in phone:
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end()))

    # Capitalised tokens (names/simple multiword names) and city tokens
    for m in re.finditer(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,}){0,2}\b", text):
        token = m.group(0)
        # treat short proper names and city tokens as sensitive
        if token in CITIES or len(token.split()) <= 3:
            spans.append((m.start(), m.end()))

    # Merge overlapping spans
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged

def mask_text(text: str, mask_token: str) -> str:
    spans = find_sensitive_spans(text)
    if not spans:
        return text
    s = text
    for a, b in reversed(spans):
        s = s[:a] + mask_token + s[b:]
    # tidy whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

class MaskFiller:
    def __init__(self, model_name: str = "roberta-base", device: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.mask = self.tok.mask_token  # "<mask>"

    @torch.no_grad()
    def fill_once(self, text: str, epsilon: float, top_k: int = 20) -> str:
        if self.mask not in text:
            return text
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        # positions of mask tokens
        mask_pos = (inputs["input_ids"] == self.tok.mask_token_id).nonzero(as_tuple=False)
        if mask_pos.numel() == 0:
            return text
        outputs = self.model(**inputs).logits
        b, pos = mask_pos[0].tolist()
        logits = outputs[b, pos, :]

        # Temperature from epsilon: lower eps -> more randomness/obfuscation
        eps = max(1e-6, float(epsilon))
        temp = max(0.7, min(3.0, 1.5 / eps))
        probs = torch.softmax(logits / temp, dim=-1)

        top_k = max(5, int(top_k))
        tkp, tki = torch.topk(probs, k=top_k)
        tkp = tkp / tkp.sum()
        idx = torch.multinomial(tkp, 1).item()
        tok_id = tki[idx].item()

        tok_str = self.tok.decode([tok_id]).replace("Ġ", " ").strip()
        tok_str = re.sub(r"\s+([,.!?;:])", r"\1", tok_str)  # clean spacing before punct
        return text.replace(self.mask, tok_str, 1)

    def rewrite(self, text: str, epsilon: float, max_masks: int = 8, top_k: int = 20) -> str:
        out = text
        for _ in range(max_masks):
            if self.mask not in out:
                break
            out = self.fill_once(out, epsilon, top_k)
        return out

def process_csv(path: str, limit: int, epsilon: float, out_csv: str | None):
    mf = MaskFiller("roberta-base")
    # Sentiment140 style files are often latin-1 encoded
    try:
        f = open(path, newline="", encoding="utf-8")
        rows = list(csv.reader(f))
        f.close()
    except UnicodeDecodeError:
        f = open(path, newline="", encoding="latin-1")
        rows = list(csv.reader(f))
        f.close()

    rows = rows[:limit] if limit > 0 else rows

    writer = None
    outf = None
    if out_csv:
        outf = open(out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(outf)
        writer.writerow(["index", "original", "masked", "rewritten", "epsilon"])

    for i, row in enumerate(rows, start=1):
        if not row:
            continue
        text = row[-1].strip()
        masked = mask_text(text, mf.mask)
        rewritten = mf.rewrite(masked, epsilon=epsilon)

        print(f"\nTweet {i}: {text}")
        print(f"Masked:  {masked}")
        print(f"Rewritten (ε={epsilon}): {rewritten}")

        if writer:
            writer.writerow([i, text, masked, rewritten, epsilon])

    if outf:
        outf.close()

def main():
    ap = argparse.ArgumentParser(description="Mask-and-fill rewriting demo over a CSV dataset (Sentiment140, etc.)")
    ap.add_argument("--csv", required=True, help="Path to CSV dataset (e.g., Sentiment140 subset)")
    ap.add_argument("--limit", type=int, default=25, help="How many rows to process")
    ap.add_argument("--epsilon", type=float, default=1.0, help="Privacy-like knob: lower => more obfuscation")
    ap.add_argument("--out_csv", type=str, default="", help="Optional path to save side-by-side results as CSV")
    args = ap.parse_args()

    out_csv = args.out_csv if args.out_csv else None
    process_csv(args.csv, args.limit, args.epsilon, out_csv)

if __name__ == "__main__":
    main()
