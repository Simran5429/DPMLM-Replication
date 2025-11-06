import argparse
import torch
from transformers import RobertaTokenizer, RobertaModel
from DPMLM import DPMLM  # import from the original project

def rewrite_with_any_api(model, input_text, epsilon):
    """
    Try to run the privatization. If the [MASK] token is missing,
    automatically insert one and retry (without editing DPMLM.py).
    """
    try:
        return model.privatize(input_text, epsilon)
    except ValueError as e:
        if "not in list" in str(e):
            print(" Warning: No [MASK] token found — inserting one automatically.")
            # Add a mask token near the end of the sentence
            masked_text = input_text.strip()
            if not masked_text.endswith('.'):
                masked_text += '.'
            mask_token = getattr(model.tokenizer, 'mask_token', '[MASK]')
            masked_text = masked_text.replace('.', f' {mask_token}.', 1)
            return model.privatize(masked_text, epsilon)
        else:
            raise

def main():
    parser = argparse.ArgumentParser(description="Run DP-MLM text privatization")
    parser.add_argument("--input", type=str, required=True, help="Input text to privatize")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Differential privacy epsilon")
    args = parser.parse_args()

    print(" Loading DP-MLM model...")
    model = DPMLM()

    print(f"\n Original: {args.input}")
    rewritten = rewrite_with_any_api(model, args.input, args.epsilon)
    print(f" Rewritten (ε={args.epsilon}): {rewritten}\n")

if __name__ == "__main__":
    main()
