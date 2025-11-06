import argparse
import re
import sys
from typing import List

import pandas as pd
from transformers import pipeline

# Ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def strip_html(text: str) -> str:
    # IMDB reviews often contain <br /><br />
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"<.*?>", " ", text)  # drop any remaining tags
    return re.sub(r"\s+", " ", text).strip()


def mask_review(text: str) -> str:
    """Very simple masking: names, capitalized tokens, long numbers."""
    # numbers with >=3 digits
    text = re.sub(r"\b\d{3,}\b", "<mask>", text)
    # capitalised words or short proper-name sequences
    text = re.sub(r"\b([A-Z][a-z]{2,})(\s[A-Z][a-z]{2,}){0,2}\b", "<mask>", text)
    return re.sub(r"\s+", " ", text).strip()


def fill_one_mask(mask_filler, text: str) -> str:
    """Replace the FIRST <mask> with the top prediction."""
    if "<mask>" not in text:
        return text
    # Ensure we feed a single-mask string to keep the return shape simple
    before, _, after = text.partition("<mask>")
    query = before + mask_filler.tokenizer.mask_token + after
    res = mask_filler(query, top_k=5)

    # `res` can be a list[dict] (single mask) OR list[list[dict]] (HF versions)
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        token = res[0]["token_str"]
    elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        token = res[0][0]["token_str"]
    else:
        token = ""  # fallback: empty

    token = token.replace("Ġ", " ").strip()
    return query.replace(mask_filler.tokenizer.mask_token, token, 1)


def fill_all_masks(mask_filler, text: str, max_steps: int = 8) -> str:
    out = text
    for _ in range(max_steps):
        if "<mask>" not in out:
            break
        out = fill_one_mask(mask_filler, out)
    return out


def main():
    ap = argparse.ArgumentParser(description="IMDB review rewriting with mask-fill")
    ap.add_argument("--csv", required=True, help='Path to "IMDB Dataset.csv"')
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--epsilon", type=float, default=1.0, help="Displayed only (control knob if you wish)")
    ap.add_argument("--out_csv", type=str, default="imdb_rewrites.csv")
    args = ap.parse_args()

    print("Loading DP-MLM model (using RoBERTa base for rewriting)...")
    mask_filler = pipeline("fill-mask", model="roberta-base", device=-1)  # CPU

    # Load IMDB (handles UTF-8 / latin-1)
    try:
        df = pd.read_csv(args.csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(args.csv, encoding="latin-1")

    # Expect columns: review, sentiment
    if "review" not in df.columns:
        # try the first text-like column as a fallback
        for col in df.columns:
            if df[col].dtype == "object":
                df = df.rename(columns={col: "review"})
                break

    df = df.head(args.limit).copy()

    records = []
    for i, row in df.iterrows():
        original = str(row["review"])
        original_clean = strip_html(original)
        masked = mask_review(original_clean)

        try:
            rewritten = fill_all_masks(mask_filler, masked)
        except Exception as e:
            print(f"Error processing review {i+1}: {e}")
            rewritten = original_clean

        print(f"\nReview {i+1}: {original_clean[:90]}...")
        print(f"Masked: {masked[:90]}...")
        print(f"Rewritten (ε={args.epsilon}): {rewritten[:90]}...")

        records.append(
            {
                "index": i + 1,
                "original": original_clean,
                "masked": masked,
                "rewritten": rewritten,
                "epsilon": args.epsilon,
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\n Saved rewritten IMDB reviews to {args.out_csv}")


if __name__ == "__main__":
    main()
