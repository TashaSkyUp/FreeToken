# Experimental Qwen3.8 tensor parallelism on R9700

This branch investigates TP=1/TP=2 greedy-output divergence for
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).
It adapts the intermediate-axis TP design from
[gdevenyi's PR #385](https://github.com/FlashML-org/FreeToken/pull/385)
(head `d2ad39b8d55550938741f7788183557fd2524591`) and adds an opt-in precision
profile. The upstream issue is
[#62](https://github.com/FlashML-org/FreeToken/issues/62); the alternative
owner-local expert-parallel design in
[#447](https://github.com/FlashML-org/FreeToken/pull/447) is separate work.

The tested path uses BF16 dense weights, native NVFP4 routed experts, the Triton
expert backend, host expert offload, and torch.distributed with RCCL. Other
checkpoints, quantized dense projections, CPU/hybrid experts, speculative
decoding, other TP degrees, and NVIDIA hardware have not been qualified by this
experiment. Cross-TP text and logit equality remain unresolved in general.

## Changes

Dense/GDN projections and the embedding/head are sharded consistently with the
checkpoint loader. The QSA indexer, router, norms, hyper-connections and PLE stay
replicated. NVFP4 gate/up rows and down columns are sliced along each expert's
intermediate dimension, with matching packed-code and scale-bank geometry.
Shared and routed expert partial outputs are combined before one all-reduce.

The precision settings are disabled by default. The measured profile enables
all five in both TP modes; the experiment does not establish the quality or
performance of arbitrary subsets. Set each variable to `1` to enable it and
unset it to disable it (the implementation checks for a nonempty value).

| Variable | Behavior |
| --- | --- |
| `FREETOKEN_TP_PRECISE` | Keep BF16 row/output-projection products, NVFP4 expert down outputs, expert sums and untied BF16 LM-head logits in FP32 through their relevant reductions. |
| `FREETOKEN_GDN_SNAPSHOT_FP32` | Store chunk boundary snapshots in the recurrent-state pool's dtype; an FP32 pool no longer receives a BF16-rounded snapshot. |
| `FREETOKEN_ROUTER_FP32` | Produce Qwen4 router logits in FP32 before expert selection. |
| `FREETOKEN_STABLE_LINEAR` | Use fixed 16x64x64 Triton tiles for unquantized half-precision dense products, with FP32 accumulation, to reduce changes in arithmetic with prefill batch shape. |
| `FREETOKEN_RESIDUAL_FP32` | Carry Qwen4 hyper-connection residual streams in FP32, casting normalized projection inputs and PLE inputs to their weight dtype. |

These switches reach shared linear/GDN/NVFP4 code. Enable them only for this
experimental profile until other model/backend combinations have been checked.
Fixed tiling does not guarantee identical arithmetic across TP degrees.

## Hardware and source

- GPU: 2 x AMD Radeon AI PRO R9700, 32 GiB each, gfx1201. An additional
  RX 9070 XT 16 GiB was installed but excluded with `HIP_VISIBLE_DEVICES=0,1`.
- CPU: AMD Ryzen 9 3900X; system RAM: 109.375 GiB visible.
- OS: Ubuntu 24.04.4, Linux `7.0.14-12-pve`, in the existing container environment.
- Python 3.12.3, PyTorch `2.14.1+rocm7.2`, HIP `7.2.53211`, ROCm SDK 7.2.3,
  triton-rocm 3.8.0. This differs from upstream's documented ROCm 7.14 stack.
- Checkpoint revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- RCCL: task-local `2.27.7-HEAD:96a25b5+`, with SONAME changed to `librccl.so`
  to prevent a second bundled RCCL from loading. Source and modified-library
  hashes are recorded in the evidence archive's `environment.json`.
- The original 100-prompt run used the adapted source based on
  `e5856ac471ec4747ada27f0ba506b816f1506593`. The PR preparation rebases that
  experiment onto `14119b8cbb8bd0a1e20410d5f1371f2412a6f338`, removes scratch
  trace/replay imports and the global timeout change, and adds an exact
  divisibility check to NVFP4 slicing. The 100-prompt run was not repeated on
  this cleaned branch; fresh checks are recorded separately in the PR body.

## Server and request settings

Activate the local ROCm environment, then use the following command for each TP
mode. `MODEL_DIR`, `TORCH_HIP_LIBRARY` and `RCCL_LIBRARY` must name your local
checkpoint and libraries; they are not files supplied by this branch.

```bash
export HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export LD_PRELOAD="$TORCH_HIP_LIBRARY:$RCCL_LIBRARY"
export FREETOKEN_TP_PRECISE=1 FREETOKEN_GDN_SNAPSHOT_FP32=1
export FREETOKEN_ROUTER_FP32=1 FREETOKEN_STABLE_LINEAR=1
export FREETOKEN_RESIDUAL_FP32=1
PYTHONPATH=python python -m freetoken.cli serve \
  --model "$MODEL_DIR" --served-model-name qwen38-repro \
  --tp-size 2 --disable-pynccl --moe-strategy offload \
  --expert-load parallel --quant-backend moe.nvfp4=triton \
  --ple-backend disk --text-model-only --moe-cache-auto \
  --kv-reserve-tokens 16384 --num-tokens 16384 \
  --max-seq-len-override 16384 --memory-ratio 0.90 \
  --max-running-requests 1 --cuda-graph-max-bs 1 \
  --max-prefill-length 1024 --enable-timings-report --enable-cache-report \
  --reasoning-parser off --host 127.0.0.1 --port 2959
```

Use `--tp-size 1` for TP=1. For the main baseline unset all five precision
variables. Current main rejects Qwen4 TP=2; there is no main TP=2 performance
baseline. Requests use the frozen chat-templated prompts via `/v1/completions`,
`temperature=0`, `top_k=1`, `top_p=1`, no penalties and `max_tokens=128`.
The prompt manifest preserves exact text and token counts.

Building with the older local HIP SDK required
`CFLAGS=-DhipStreamWriteValueDefault=0` and
`CXXFLAGS=-DhipStreamWriteValueDefault=0` before
`python setup.py build_ext --inplace`. An initial build without these definitions
failed. The successful rebuild log is retained; no C++ source compatibility
change is included here.

## Existing evidence

The initial three regression prompts were each requested three times (one cold,
two repeats). Before the precision profile, 3/9 cross-TP responses differed;
with the full profile, 0/9 differed and the long-context cold/cache mismatch
disappeared. These are results against the original TP adaptation, not against
unmodified main at TP=2.

The expanded run has 100 distinct prompts, ten per category, plus ten long
repeat attempts per TP mode: 220 responses total. Input lengths are 25-13,967
tokens. All 100 core requests were uncached in both modes.

- 74/100 core outputs match byte for byte. Two more differ only in JSON spacing;
  24 have other wording, format or content differences.
- 100/100 finish reasons agree; 85/100 completion-token counts agree.
  Seventeen responses per mode hit the 128-token cap.
- Strict known-answer/format checks pass 35/42 in each mode. Matching text is
  not evidence of a correct answer: common wrong and incomplete answers occur.
- Nine of ten repeat attempts actually hit the prefix cache in each mode.
  Initial/repeated text matches in 7/10 TP=1 and 8/10 TP=2 attempts; the
  differences preserve parsed JSON values. The remaining attempt misses the
  cache in both modes and is not a cache-hit validation.
- No unpatched 100-prompt control was run. The 9-request improvement does not
  demonstrate a 50% divergence reduction on the larger set.

For the 66 short pairs whose delivered text matches, the historical run measured:

| Client/engine measurement | TP=1 p50 / p95 | TP=2 p50 / p95 |
| --- | --- | --- |
| Client time to first nonempty text, seconds | 11.136 / 11.248 | 5.785 / 5.906 |
| Client request to SSE completion, seconds | 12.421 / 19.415 | 6.360 / 9.324 |
| Engine per-request mean decode, ms/token | 77.750 / 159.148 | 32.154 / 40.206 |

The decode percentiles summarize per-request means, not individual inter-token
gaps. Auto expert cache capacity differs: 6,829 slots at TP=1 versus 16,275 per
rank at TP=2. The observed speed includes the resulting residency benefit.
This was one sequential pass, with existing disk/JIT caches, not a repeated
statistical estimate or a pure communication-scaling benchmark.

The cleaned branch also has a separate, fresh three-short-prompt A/B against
main at `14119b8`, with all requests uncached:

| Configuration | Median decode, tok/s | Median client TTFT, s | Median response, s |
| --- | ---: | ---: | ---: |
| Main TP=1, precision off | 14.632 | 11.228 | 19.286 |
| Branch TP=1, precision on | 5.913 | 11.233 | 32.214 |
| Branch TP=2, precision on | 24.638 | 5.799 | 9.954 |

All three branch TP=1/TP=2 output pairs match exactly. This small comparison
exposes a roughly 60% TP=1 decode-throughput cost for the precision profile;
TP=2 is about 1.68x main TP=1. There was no timed warmup phase. Auto sizing
resolved 6,807 / 6,829 / 16,275 expert slots, respectively, and 256 / 257 / 257
KV pages. These are observations from one pass, with changing cache residency,
not estimates of settled-cache performance. The archived controller records
every command, response, source hash and timing.

A secondary blinded judge experiment made 200 accepted Codex CLI evaluations:
100 A/B pairs, then the same 100 in reversed submission and A/B order, up to 20
concurrent, `gpt-6.1-terra` at `max` reasoning. Its three 0-4 dimensions were
correctness, instruction following/completeness, and semantic similarity. Mean
correctness was 3.720 (TP=1) / 3.710 (TP=2); completeness was 3.505 / 3.470.
These model judgments are not independent accuracy ground truth. Each order
was judged once, so order sensitivity is confounded with judge variance.
The archive discloses 242 main CLI invocations for 200 accepted results: a
parser error unnecessarily retried 42 completed calls and was fixed afterward.

The [evidence archive](../benchmarks/evidence/qwen4-tp-r9700.tar.gz) includes frozen prompts, responses, timings,
server logs, source hashes, the historical patch, judge ratings and their
methodology. It excludes model weights, authentication material and raw Codex
CLI event logs. Its manifest hashes every member.
