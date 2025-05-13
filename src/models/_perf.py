#!/usr/bin/env python3
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
