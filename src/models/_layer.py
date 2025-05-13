#!/usr/bin/env python3
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SparseLinear(nn.Module):
    def __init__(self, dense: nn.Linear):
        super().__init__()
        w = dense.weight.data.float()
        self.in_features = dense.in_features
        self.out_features = dense.out_features
        self.weight = w.to_sparse()
        self.bias = dense.bias.data if dense.bias is not None else None

    def forward(self, x: torch.Tensor):
        orig = x.shape
        flat = x.reshape(-1, self.in_features)
        idx, vals = self.weight.indices(), self.weight.values()
        wt_t = torch.sparse_coo_tensor(
            idx[[1,0]], vals, (self.in_features, self.out_features)
        )
        out = torch.sparse.mm(flat, wt_t)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig[:-1], self.out_features)

def convert_to_sparse(m: nn.Module):
    for name, child in list(m.named_children()):
        if isinstance(child, nn.Linear):
            setattr(m, name, SparseLinear(child))
        else:
            convert_to_sparse(child)

def profile_model(model_dir, prompt, prefill_tokens, decode_length, num_threads, device):
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device).eval()

    convert_to_sparse(model)

    if prefill_tokens > 0:
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        ids = torch.full((1, prefill_tokens), pad, dtype=torch.long).to(device)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    layers = model.transformer.h if hasattr(model, "transformer") else model.model.layers
    N = len(layers)
    params = [sum(p.numel() for p in l.parameters())/1e6 for l in layers]

    # prefill
    start = [0.0]*N
    pre_times = [0.0]*N
    hooks = []
    for i, l in enumerate(layers):
        hooks += [
            l.register_forward_pre_hook(lambda m, x, i=i: start.__setitem__(i, time.time())),
            l.register_forward_hook(lambda m, x, y, i=i: pre_times.__setitem__(i, (time.time()-start[i])*1000))
        ]
    with torch.no_grad():
        _ = model(**{**inputs, "use_cache": False})
    for h in hooks: h.remove()

    # decode
    start = [0.0]*N
    recs = [[] for _ in range(N)]
    hooks = []
    for i, l in enumerate(layers):
        hooks += [
            l.register_forward_pre_hook(lambda m, x, i=i: start.__setitem__(i, time.time())),
            l.register_forward_hook(lambda m, x, y, i=i: recs[i].append((time.time()-start[i])*1000))
        ]
    with torch.no_grad():
        out = model(**{**inputs, "use_cache": True})
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    for _ in range(decode_length):
        with torch.no_grad():
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1, keepdim=True)
    for h in hooks: h.remove()

    print(f"Device={device} Threads={num_threads}")
    print("Layer | Params(M) | Prefill(ms) | Decode μ±σ(ms)")
    print("----- | --------- | ---------- | -------------")
    for i in range(N):
        m = np.mean(recs[i]) if recs[i] else 0.0
        s = np.std(recs[i])  if recs[i] else 0.0
        print(f"{i:5d} | {params[i]:9.3f} | {pre_times[i]:10.3f} | {m:7.3f}±{s:6.3f}")

import argparse
import subprocess
import shlex
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# (이전과 동일한 SparseLinear / convert_to_sparse / profile_model 함수는 생략)

def run_with_perf(script, args, perf_events):
    """
    perf stat -e <events> -- python <script> <args...>
    """
    perf_cmd = (
        f"perf stat -e {','.join(perf_events)} -- "
        f"python3 {script} {shlex.join(args)}"
    )
    print(f"[Perf] running: {perf_cmd}\n")
    p = subprocess.run(
        shlex.split(perf_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # perf 결과는 stderr 에 뜹    print(p.stdout)
    print("\n--- perf stat ---")
    print(p.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--prefill_tokens", type=int, default=0)
    parser.add_argument("--decode_length",  type=int, default=10)
    parser.add_argument("--num_threads",    type=int, default=4)
    parser.add_argument("--device",         choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--perf",           action="store_true",
                        help="wrap the profiling run under perf stat")
    args, unknown = parser.parse_known_args()

    script_args = [
        "--model_dir",    args.model_dir,
        "--prompt",       args.prompt,
        "--prefill_tokens", str(args.prefill_tokens),
        "--decode_length",  str(args.decode_length),
        "--num_threads",    str(args.num_threads),
        "--device",         args.device
    ]

    if args.perf:
        perf_events = [
            "cache-references",
            "cache-misses",
            "LLC-loads",
            "LLC-load-misses"
        ]
        run_with_perf(__file__, script_args, perf_events)
    else:
        # 직접 프로파일링 수행
        profile_model(
            args.model_dir,
            args.prompt,
            args.prefill_tokens,
            args.decode_length,
            args.num_threads,
            args.device
        )

