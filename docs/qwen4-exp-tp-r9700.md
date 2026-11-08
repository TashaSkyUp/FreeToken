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

The audited 1,024-token follow-up records **100/100 exact V2 TP=1/TP=2 text pairs**, compared with 77/100 for V1 on the same revised prompts. V2 strict checks pass 38/42 at TP=1 and 38/42 at TP=2. Prompt specificity, tokenizer normalization, completion limits and the judging rubric were audited before the final run. The historical Unicode-copy failure had an invalid answer key under NFC input normalization and is corrected explicitly below. Output agreement remains separate from answer quality; the precision switches remain experimental and disabled by default.

## Changes

Dense/GDN projections and the embedding/head are sharded consistently with the
checkpoint loader. The QSA indexer, router, norms, hyper-connections and PLE stay
replicated. NVFP4 gate/up rows and down columns are sliced along each expert's
intermediate dimension, with matching packed-code and scale-bank geometry.
Shared and routed expert partial outputs are combined before one all-reduce.

The precision settings are disabled by default. The initial v1 profile enables
the first five settings below. The follow-up v2 profile enables all seven in both
TP modes; arbitrary subsets have not been broadly qualified. Set each variable to `1` to enable it and
unset it to disable it (the implementation checks for a nonempty value).

| Variable | Behavior |
| --- | --- |
| `FREETOKEN_TP_PRECISE` | Keep BF16 row/output-projection products, NVFP4 expert down outputs, expert sums and untied BF16 LM-head logits in FP32 through their relevant reductions (the v2 reduction flag widens the affected row/expert paths further). |
| `FREETOKEN_GDN_SNAPSHOT_FP32` | Store chunk boundary snapshots in the recurrent-state pool's dtype; an FP32 pool no longer receives a BF16-rounded snapshot. |
| `FREETOKEN_ROUTER_FP32` | Produce Qwen4 router logits in FP32 before expert selection. |
| `FREETOKEN_STABLE_LINEAR` | Use fixed 16x64x64 Triton tiles for unquantized half-precision dense products, with FP32 accumulation, to reduce changes in arithmetic with prefill batch shape. |
| `FREETOKEN_RESIDUAL_FP32` | Carry Qwen4 hyper-connection residual streams in FP32, casting normalized projection inputs and PLE inputs to their weight dtype. |
| `FREETOKEN_TP_REDUCTION_FP64` | Accumulate fixed 64-wide BF16 dot-product tiles and NVFP4 down-projection partials in FP64; retain FP64 through expert weighting, shared/routed combination and TP reduction, then return FP32 residual updates. |
| `FREETOKEN_STABLE_ATTENTION` | Select QSA attention tiles and per-head normalization layouts from the original head counts, so TP sharding does not change those choices. |

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
  divisibility check to NVFP4 slicing. The v1 100-prompt run was not repeated
  on that cleaned branch. The v2 run below uses cleaned commit `61b0189` plus
  the archived candidate patch; its settings record every Python source hash.

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
export FREETOKEN_TP_REDUCTION_FP64=1 FREETOKEN_STABLE_ATTENTION=1
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

Use `--tp-size 1` for TP=1. To reproduce v1, unset the two v2 additions.
For the main baseline unset all seven precision variables. Main at `14119b8`
rejects Qwen4 TP=2; there is no main TP=2 performance baseline. Requests use the frozen chat-templated prompts via `/v1/completions`,
`temperature=0`, `top_k=1`, `top_p=1`, no penalties and `max_tokens=128` in the historical runs. The audited v3 rerun below uses `max_tokens=1024`.
The prompt manifests preserve exact text and token counts.

Building with the older local HIP SDK required
`CFLAGS=-DhipStreamWriteValueDefault=0` and
`CXXFLAGS=-DhipStreamWriteValueDefault=0` before
`python setup.py build_ext --inplace`. An initial build without these definitions
failed. The successful rebuild log is retained; no C++ source compatibility
change is included here.

## Historical v1 evidence (128-token cap)

This section retains the original prompts and 128-token measurements. The latest
revised-prompt results are in [the audited 1,024-token rerun](#audited-prompts-and-1024-token-rerun).

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
- Strict known-answer/format checks pass **35/41 valid cases** in each mode.
  The frozen summary originally reported 35/42; its Unicode-copy case had an
  invalid NFD key for NFC-normalized input and is excluded from the denominator.
  Matching text is not evidence of a correct answer: common wrong and incomplete
  answers occur.
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
has branch TP=1 decode throughput roughly 60% below main in that pass;
TP=2 is about 1.68x main TP=1. It does not isolate a settled precision cost. There was no timed warmup phase. Auto sizing
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

## Historical v2 experiment (128-token cap)

The first remaining difference was an FP32 attention output projection. Small
rounding differences then crossed BF16 normalization boundaries and grew through
the decoder. Wider accumulation made the four short prefill traces identical
until QSA layers. Two additional choices depended on local head counts: QSA
attention tile/split selection and the per-head RMSNorm warp count. V2 uses the
unsharded head counts for both choices. All switches remain opt-in.

The operator probes use FP64 arithmetic as an independent numerical reference.
Before the final FP32 cast, a tested 6,144-wide BF16 row projection RMS error fell from about
`1.2e-6` to `5.8e-8`; split and unsplit results agreed after FP32 rounding.
The NVFP4 down/decode probe also reduced error and eliminated the split
difference after FP32 rounding. These are representative operator checks,
not an independent full-model reference-engine accuracy benchmark.

Four difficult short prompts were replayed along one fixed token path. Metrics
below cover 305 delivered prediction positions; the raw traces also retain
4 extra queued final forwards. Final TP=1 and TP=2 runs both enable all seven flags.

| Profile | Greedy disagreements | Mean KL, nats | Centered logit RMS |
| --- | ---: | ---: | ---: |
| V1 | 8/305 | 0.00161089931 | 0.0637994979 |
| V1 + FP64 reductions | 2/305 | 0.000344091699 | 0.0319741129 |
| V2, including stable attention | 0/305 | 0 | 0 |

The final raw FP32 logit bit patterns match at all 309 recorded dispatches.
The source hashes and model settings are retained for each run.

The original 100 prompts and ten cache-repeat attempts per mode were then run
with tracing and replay disabled, at the same greedy settings and 128-token cap.

| Measurement | V1 | V2 |
| --- | ---: | ---: |
| Exact core text pairs | 74/100 | 100/100 |
| Divergent core pairs | 26 | 0 |
| Strict known-answer/format checks, TP=1, valid cases | 35/41 | 34/41 |
| Strict known-answer/format checks, TP=2, valid cases | 35/41 | 34/41 |

The frozen original summaries reported 35/42 and 34/42. Both counted the invalid
Unicode-copy case; the table excludes it. Those historical archives are unchanged.
The revised NFC-stable cohort below has 42 valid checks and is a separate comparison.

Divergent-output count fell by 100.0%. All core requests were uncached.
V2 also changes TP=1 arithmetic: 35/100 TP1 texts change between profiles. V2 TP2 matches the original V1 TP1 text in 65/100 cases. The 100-pair agreement figure compares TP modes with V2 enabled in both.
Finish reasons match in 100/100 pairs and completion-token counts in 100/100.
Core finish reasons: {"tp1": {"length": 17, "stop": 83}, "tp2": {"length": 17, "stop": 83}}. Total completion tokens across the 220 responses: 11132.
Agreement is a separate measurement from answer correctness. Common wrong or
incomplete answers remain. In `normalize-records`, V2 wraps the correct JSON
data in Markdown fences despite a return-only-JSON instruction; this remains a
strict format-check failure, not a pass obtained by stripping the fences.

Newly failing strict checks relative to V1: {"tp1": ["normalize-records"], "tp2": ["normalize-records"]}.
TP=1: 9/10 repeats actually hit the cache; 8/10 initial/repeat texts match.
TP=2: 9/10 repeats actually hit the cache; 8/10 initial/repeat texts match.

Cached/repeated texts match across TP in 10/10 attempts.

A concurrent ComfyUI workload overlapped host paging during part of the TP=1
100-prompt pass. Those full-run latency observations are preserved as contended
data, and are not used to isolate the cost of V2. A separate same-source A/B
uses identical delivered text/token workloads, three warmups followed by three
timed requests per configuration, and cold KV caches. Idle service/GPU and RAM
preflight observations are retained. The ordering is V1 TP1, V2 TP1, V2 TP2,
V1 TP2; this is one small sequential comparison, not a confidence interval.
The controller paused between the first two legs while GPU memory was being
released after shutdown. The completed V1 TP1 leg was preserved, and the
remaining legs resumed after idle checks. Both harness versions are retained.

| Configuration | Median TTFT, s | Median response, s | Median decode, ms/token | Median decode, tok/s |
| --- | ---: | ---: | ---: | ---: |
| V1-TP1 | 11.096 | 32.286 | 166.845 | 5.994 |
| V2-TP1 | 11.136 | 33.502 | 176.111 | 5.678 |
| V1-TP2 | 5.772 | 8.997 | 25.251 | 39.603 |
| V2-TP2 | 5.802 | 9.218 | 26.259 | 38.082 |

Median decode time per token rises 5.55% at TP1 and 3.99% at TP2. Median response time rises 3.77% and 2.46%, respectively.
Resolved expert-cache slots are unchanged between profiles within each TP mode: 6,824 at TP1, 15,728 per rank at TP2, with 256 KV pages in every leg.

The SSE delivery logs additionally preserve client text-chunk arrival times.
Chunks can contain partial or multiple tokens; their gaps are not presented as
individual GPU token-generation gaps. Auto expert-cache residency remains part
of these end-to-end measurements.

One old-profile TP2 timing attempt timed out in the cache-control endpoint before
any inference response. Its logs are retained. That leg was restarted after an
idle preflight with a three-second control-plane settling delay outside measured
request intervals; the three completed legs were preserved.

The existing blinded Terra rubric was also run on **V1 TP=1 versus V2 TP=1**
(arithmetic-profile quality, not a TP1/TP2 comparison). Both original source
files retain their actual TP=1 identity. One hundred pairs were judged, then
resubmitted with reversed sample and A/B order, at up to 20 concurrent calls,
`gpt-6.1-terra`, `max` reasoning. The previous parser correction is included.

| Mean judge score, 0-4 | V1 TP=1 | V2 TP=1 |
| --- | ---: | ---: |
| Correctness | 3.695 | 3.655 |
| Instruction following/completeness | 3.515 | 3.490 |

Accepted evaluations: 200; main CLI invocations: 200.
Identical-output control errors: 0.
These ratings are secondary evidence. One judgment per order confounds order
effects with judge variance, and the purpose-built prompt set does not establish
general model capability or quality noninferiority.

Correctness outcomes stable across both judging orders: {"order_sensitive": 3, "profile1_better": 4, "profile2_better": 1, "tie": 92}.
Instruction/completeness outcomes stable across both orders: {"profile1_better": 2, "profile2_better": 1, "tie": 97}.
The judges consistently favored V1 on `bash-spaces` (V2 reaches the cap before
finishing the shell code) and `ordered-steps` (V2 recommends SHA-256 commands
after mentioning other checksum algorithms), as well as two smaller explanatory
precision differences. The raw outputs and rationales are retained; exact TP
agreement should not be described as a general quality improvement.

Fifteen targeted precision-profile tests pass, including dense-oracle QSA,
chunked prefill, decode/verify graph replay, GDN, NVFP4 and hyper-connections.
With the precision flags disabled, the QSA file has nine passes, one skip and
one failure at `test_chunked_prefill_matches_one_shot[unaligned]`. The same
assertion independently fails on the unmodified `61b0189` draft tree; the
all-seven-flags run passes that case. No test or tolerance was changed.

The [v2 evidence archive](../benchmarks/evidence/qwen4-tp-r9700-v2.tar.gz)
contains the frozen 100-prompt runs, operator and aligned-logit summaries,
fresh timing/SSE data, profile-quality judgments, executed harnesses, settings,
source hashes and test logs. Large NumPy logit/layer arrays remain local and
are represented by hashes; weights, authentication data and raw Codex event
logs are excluded. Other hardware, communication backends and checkpoints
remain unqualified by this experiment.

## Audited prompts and 1,024-token rerun

The entire 100-prompt matrix was rerun at cleaned commit `4aed3f5c7ae3054fb3813de304ef2b820c804642`,
with identical hashes for all 503 Python engine files in all four configurations.
V1 enables the original five precision flags; V2 adds FP64 reductions and stable attention.
Each configuration runs 100 uncached core requests and ten cache-repeat attempts:
**440 inference responses**, followed by **400 blinded evaluations**. No engine code changed for this rerun.

A fresh-context agent first audited all 100 user requests without seeing outputs, scores,
answer keys or the conversation. It identified 17 material ambiguities in the original set.
After revision, another fresh-context audit reviewed the exact decoded user turns and found
a remaining long-context shortcut: supplied formulas allowed record retrieval to be bypassed.
The formulas were removed, structural validation was specified, and the final delta was
independently verified. The final audit has no unresolved material finding.
Arithmetic keys were independently recomputed. All 100 rendered inputs round-trip through
the checkpoint tokenizer exactly; the final user-turn SHA256 is
`008c85a8b27c24b2dd163b6f29f284d7f771ef4e45c6f0f9fa05111f2045dbbc`.

The tokenizer normalizes input to NFC. The historical `unicode-preserve` key required
NFD `e` plus a combining accent even though the model received precomposed `é`.
That historical failure must be excluded: the previously reported V1 35/42 and V2 34/42
become **35/41 and 34/41 valid cases**. The old archives remain unchanged.
The new cohort uses an NFC-stable payload and corrected key, retaining 42 checks;
the other 41 keys are unchanged. An initial revised attempt was stopped when this issue
was found; its partial responses are excluded from the final cohort and preserved locally.

Requests retain the same greedy settings and explicit `enable_thinking=False` chat template.
The rendered generation prefix ends with an empty closed think block. `/v1/completions`
returns raw delivered text, with the reasoning parser off; no generated text is stripped
as inferred thinking. Raw SSE records reconstruct all 440 saved outputs exactly.
A cap-only check of the eight historical strict failures, using
their original prompts, found seven unchanged natural-stop answers. The fraction answer
continued from 128 to 381 tokens and reached 29/36, but still violated the answer-only format.
The larger cap matters for completion; it does not explain all historical failures.

| Configuration | Exact TP pairs | Strict passes | Capped core responses | Largest output, tokens |
| --- | ---: | ---: | ---: | ---: |
| V1-TP1 | 77/100 | 38/42 | 0/100 | 317 |
| V1-TP2 | 77/100 | 38/42 | 0/100 | 266 |
| V2-TP1 | 100/100 | 38/42 | 0/100 | 615 |
| V2-TP2 | 100/100 | 38/42 | 0/100 | 615 |

V2 changes 23/100 TP=1 texts relative to V1. V2 TP=2 matches the V1 TP=1 text in 77/100 cases. The 100/100 agreement compares TP modes with the same V2 profile enabled in both.

The shared remaining fixed-check failures are delivered content errors, not cutoffs:

| Case | Expected | V1 TP1 / TP2 | V2 TP1 / TP2 |
| --- | --- | --- | --- |
| `fraction-sum` | `29/36` | `41/36` / `41/36` | `13/36` / `13/36` |
| `modular-power` | `3` | `9` / `9` | `9` / `9` |
| `weighted-average` | `8` | `7` / `7` | `7` / `7` |
| `quoted-instruction` | `6` | `5` / `5` | `5` / `5` |

All fixed-check failures, with prompts, expected values and all four actual outputs,
are retained in `all-fixed-check-failures.json` and the matching HTML catalog in the archive.
Fixed checks assess 42 specified answers/formats; the judges assess all 100 requests.
The revised contracts accept equivalent numeric spellings for the weighted average and
enforce the explicitly requested ordering/spacing in two other cases. Legacy checker
totals are retained too. JSON fences and altered Unicode output remain failures.

For the 60 short prompts with identical delivered text, token count and finish reason in all four configurations:

| Configuration | TTFT p50 / p95, s | Request p50 / p95, s | Decode p50 / p95, ms/token | Decode p50, tok/s |
| --- | ---: | ---: | ---: | ---: |
| V1-TP1 | 11.133 / 11.230 | 12.004 / 16.632 | 75.746 / 152.826 | 13.203 |
| V1-TP2 | 5.819 / 5.903 | 6.512 / 10.447 | 54.775 / 63.215 | 18.257 |
| V2-TP1 | 11.134 / 11.223 | 12.027 / 16.828 | 78.929 / 154.156 | 12.673 |
| V2-TP2 | 5.825 / 5.911 | 6.550 / 10.808 | 59.179 / 67.934 | 16.898 |

Startup-to-healthy seconds, kept separate from request latency: V1-TP1 70.074, V1-TP2 68.071, V2-TP1 70.071, V2-TP2 66.078.
Resolved expert slots per rank / KV pages: V1-TP1 6,824 / 256, V1-TP2 15,652 / 256, V2-TP1 6,824 / 256, V2-TP2 15,652 / 256.

This is one sequential pass, ordered V2 TP2, V1 TP2, V1 TP1, V2 TP1, with existing
disk/JIT caches and no timed warmup phase. The same-output subset controls delivered
workload, but order and cache state remain possible influences. Auto expert cache
residency differs between TP modes. Decode percentiles summarize per-request means;
SSE chunk arrival gaps are retained separately and are not individual GPU token times.
All 440 timing rows, long-input and cache-hit distributions, resource observations and
idle preflights are included. ComfyUI was observed running or queued in 0/517 sampled checks; 0 observations failed.

| Configuration | Actual cache hits | Cold/repeat exact matches | Exact or equal JSON values | Repeat fixed checks |
| --- | ---: | ---: | ---: | ---: |
| V1-TP1 | 10/10 | 9/10 | 9/10 | 9/9 |
| V1-TP2 | 10/10 | 9/10 | 9/10 | 9/9 |
| V2-TP1 | 10/10 | 9/10 | 10/10 | 9/9 |
| V2-TP2 | 10/10 | 9/10 | 10/10 | 9/9 |

The V1 cold/repeat differences are in the final validation explanation; the requested
warehouse/quantity/checksum fields stay unchanged. The V2 repeat difference preserves
the parsed JSON values. Every repeated fixed-answer lookup passes in all four modes.

The judging rubric was independently audited before any new evaluations. It separates
correctness from completeness/format, prevents double counting, defines partial-answer
handling and semantic-difference classes, and does not penalize a length flag by itself.
Each comparison has 100 forward evaluations, then 100 in reversed sample and A/B order,
using `codex exec`, `gpt-6.1-terra`, `max` reasoning and at most 20 concurrent calls.
Judges see the original request, response texts and finish reasons; source identities
and performance data are hidden. Both comparison sets use this same audited cohort.

| Comparison | Correctness, left / right | Instruction/completeness, left / right | Semantic similarity |
| --- | ---: | ---: | ---: |
| V1 TP1 / V2 TP1 | 3.595 / 3.605 | 3.970 / 3.970 | 3.755 |
| V2 TP1 / V2 TP2 | 3.630 / 3.630 | 3.965 / 3.965 | 4.000 |

Across both presentations, profile correctness preferences have 93 ties, 2 favoring V1 and 3 favoring V2; 2 differ between presentations. All scores use 0-4 anchors.

V2 TP1 appears in both comparisons with the same saved outputs. Its absolute
ratings differ because each comparison used independent judge calls; scores
from separate comparisons should not be treated as one calibrated quality scale.

Accepted evaluations / actual CLI invocations / identical-output control errors: profile-quality: 200 / 200 / 0; tp-quality: 200 / 200 / 0.
Model judgments are secondary evidence. One evaluation per presentation confounds order
effects with judge variance. New prompts, a larger cap and a revised rubric prevent
direct quality-score comparisons with the historical cohort. This purpose-built set
does not establish general capability or independent reference-engine accuracy.

All 69 existing targeted engine checks passed on the same source in this task:
54 configuration/weight checks and 15 precision-path checks. Their actual command and
result logs are retained; they were not redundantly repeated after wording-only edits.
A local launch without the ROCm activation environment failed before any response;
the failure log is retained, then the full matrix used the corrected launch environment.

The report formatter was corrected to display an empty differing-output group, and
historical subset labels were removed. No inference or judging call was repeated;
numeric results are unchanged. Original scripts and a recovery receipt are archived.

The [v3 evidence archive](../benchmarks/evidence/qwen4-tp-r9700-v3.tar.gz) includes
all frozen prompts, audits, corrected-key provenance, 440 responses, 400 accepted
judgments, every invocation receipt, timing/SSE data, source/library hashes and logs.
Its manifest verifies every payload. Weights, credentials and raw Codex CLI events
are omitted. All profiles remain opt-in, and the scope limitations above still apply.

### Statistics audit

An independent recomputation on November 8, 2026 checked the published counts,
fixed-check results, all latency/resource distributions and both sets of judge
ratings directly against frozen records, without importing the report helpers.
It reconstructed all 440 responses from raw SSE, verified all 400 A/B mappings
and final ratings, and checked every one of the archive's 2,612 payload hashes.
The current numerical summaries agree with that recomputation. The historical
text/format totals, published timing tables and mean judge ratings were also
recomputed; the corrected historical denominators now appear beside those results.

The [audit evidence](../benchmarks/evidence/qwen4-tp-r9700-v3-statistics-audit.tar.gz)
contains the independent script, detailed results, source hashes and review scope.
This was a record audit; no inference or judging calls were repeated. Historical
aligned-logit metrics were recomputed from the saved FP32 arrays. Operator
measurements were traced to the saved probe records without rerunning GPU kernels.
The original v3 archive remains unchanged at SHA256
`47ce5129da08d105d05afc39fa69192464e445a1f8bf65422fecd38fd7efc2f3`.
