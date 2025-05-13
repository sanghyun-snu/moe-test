#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.utils.prune as prune

def prune_unstructured_model(model: torch.nn.Module, amount: float):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

def prune_structured_model(model: torch.nn.Module, amount: float, axis: int):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # prune entire rows (axis=0) or columns (axis=1)
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=axis)
            prune.remove(module, 'weight')

def main():
    parser = argparse.ArgumentParser(
        description="Download a HF causal LM, prune it, and save the pruned model"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Hugging Face model identifier (e.g. gpt2, facebook/opt-1.3b)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the pruned model"
    )
    parser.add_argument(
        "--method", choices=["unstructured","structured"],
        default="unstructured",
        help="Pruning method"
    )
    parser.add_argument(
        "--amount", type=float, default=0.2,
        help="Fraction of weights (or channels) to prune"
    )
    parser.add_argument(
        "--axis", type=int, default=0,
        help="Axis for structured pruning: 0=row, 1=column"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to load model on (cpu or cuda)"
    )
    args = parser.parse_args()

    # 1) Load model & tokenizer
    print(f"Loading model '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()

    # 2) Apply pruning
    print(f"Pruning ({args.method}, amount={args.amount}, axis={args.axis})...")
    if args.method == "unstructured":
        prune_unstructured_model(model, args.amount)
    else:
        prune_structured_model(model, args.amount, args.axis)

    # 3) (Optional) run one forward pass to flush masks
    dummy_input = tokenizer("Hello world", return_tensors="pt").to(args.device)
    with torch.no_grad():
        _ = model(**dummy_input)

    # 4) Save pruned model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving pruned model to '{out_dir}'...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
