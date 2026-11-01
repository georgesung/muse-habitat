"""LoRA with speculative decoding (DFlash) in vLLM vs the same LoRA without it, on real PARTNR prompts.

For the base model and each adapter (loaded into both servers):
1. greedy: are the generated tokens identical with and without speculative decoding?
2. sampling: tokens sampled at temperature 1 on the spec server, their logprobs as reported there vs the same
   tokens' prompt logprobs on the reference server;
3. speed: <conc> concurrent temperature-1 requests on the spec server: tokens/s (acceptance is in its log).

    python scripts/lora_specdec_check.py <traj jsonl> --spec-port 8031 --ref-port 8021 \
        --adapter weak=runs/lora_check_b0.02/adapter --adapter strong=runs/lora_check_b0.2/adapter
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read()


def complete(port, model, prompt, **kw):
    body = {"model": model, "prompt": prompt, "return_token_ids": True, "skip_special_tokens": False, **kw}
    return json.loads(post(port, "/v1/completions", body))["choices"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--spec-port", type=int, default=8031)
    ap.add_argument("--ref-port", type=int, default=8021)
    ap.add_argument("--adapter", action="append", default=[])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--conc", type=int, default=12)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.traj)]
    random.Random(0).shuffle(recs)
    prompts = [r["prompt_token_ids"] for r in recs[: a.n]]
    models = ["glimmer"]
    for spec in a.adapter:
        name, path = spec.split("=")
        for port in (a.spec_port, a.ref_port):
            try:
                post(port, "/v1/unload_lora_adapter", {"lora_name": name})
            except Exception:
                pass
            post(port, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(Path(path).resolve())})
        models.append(name)
    out = {}
    for m in models:
        same, first_div, lp_diffs = 0, [], []
        for i, p in enumerate(prompts):
            g1 = complete(a.spec_port, m, p, max_tokens=a.max_tokens, temperature=0)["token_ids"]
            g2 = complete(a.ref_port, m, p, max_tokens=a.max_tokens, temperature=0)["token_ids"]
            same += g1 == g2
            first_div.append(next((j for j, (x, y) in enumerate(zip(g1, g2)) if x != y), min(len(g1), len(g2))))
            s = complete(a.spec_port, m, p, max_tokens=a.max_tokens, temperature=1.0, seed=i, logprobs=0)
            toks, lps = s["token_ids"], s["logprobs"]["token_logprobs"]
            r = complete(a.ref_port, m, p + toks, max_tokens=1, temperature=0, prompt_logprobs=0)
            ref = [next(iter(d.values()))["logprob"] if len(d) == 1 else d[str(t)]["logprob"]
                   for d, t in zip(r["prompt_logprobs"][len(p):], toks)]
            lp_diffs += [abs(x - y) for x, y in zip(lps, ref)]
        lp_diffs.sort()
        out[m] = {"greedy_identical": f"{same}/{len(prompts)}", "greedy_first_divergence": first_div,
                  "sample_logprob_absdiff_mean": sum(lp_diffs) / len(lp_diffs),
                  "sample_logprob_absdiff_p99": lp_diffs[int(0.99 * (len(lp_diffs) - 1))],
                  "tokens": len(lp_diffs)}
        # speed: concurrent temperature-1 requests
        work = [(m, prompts[i % len(prompts)], 1000 + i) for i in range(a.conc)]
        t0 = time.time()
        with ThreadPoolExecutor(a.conc) as ex:
            n = sum(len(c["token_ids"]) for c in ex.map(
                lambda w: complete(a.spec_port, w[0], w[1], max_tokens=a.max_tokens, temperature=1.0, seed=w[2]), work))
        out[m]["spec_tok_per_s_conc"] = n / (time.time() - t0)
        print(m, json.dumps(out[m]), flush=True)


if __name__ == "__main__":
    main()
