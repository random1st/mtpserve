# Inference performance objective and evidence

Status: **in progress; >90% of the theoretical ceiling is not demonstrated.**

The working scope is this repository's single-sequence text inference on the
existing M3 Max (40 GPU cores, 128 GiB), Qwen3.8-27B 4-bit affine/group64 weights
and BF16 MTP head. Preserve greedy token correctness. Changing model quality,
buying hardware or reducing the target to the current implementation's ceiling
is not a substitute for the objective.

## Target architecture

One speculative verifier pass advances the main model. An accepted draft keeps
the resulting state. A rejected draft retains the valid primary token and
restores the recurrent/convolution state at that boundary without repeating the
whole transformer. All MTP history positions advance their cache, while only
needed vocabulary outputs are projected. The existing asynchronous draft/verify
pipeline remains intact. Unsupported recovery cases retain the reference path.

## Hardware screening bound

Apple specifies [400 GB/s for the 40-core M3 Max](https://support.apple.com/en-euro/117737).
Local safetensors headers, including packed weights, quantization scales and
affine biases, give:

| Weight component | Bytes |
|---|---:|
| Text layers and final norm | 13,702,478,848 |
| Main output projection | 715,161,600 |
| BF16 MTP | 849,398,784 |

An embedding lookup reads selected rows rather than the entire embedding table;
the text loader excludes vision weights. Define `W = 14,417,640,448` bytes for
the main matrices, and `D = 1,564,560,384` bytes for MTP plus another read of the
shared output projection. Small embedding-row traffic is omitted below, making
the bound slightly more optimistic.

For acceptance fraction `a`, ideal depth-1 weight traffic per emitted token is
`(W + D) / (1 + a)`, giving `400e9 * (1 + a) / (W + D)` tokens/s. This assumes
matrix reuse across both verifier positions. It is a weight-streaming upper
bound, not a measured attainable throughput or a claim about unlimited
speculation depth.

- Ordinary decode upper bound: approximately **27.74 tokens/s**.
- Depth-1 at the previously measured aggregate `a = 0.83105`: approximately
  **45.83 tokens/s**, with a 90% screening threshold above **41.24 tokens/s**.
- Depth-1 at perfect acceptance: approximately **50.06 tokens/s**; the stricter
  90% screening threshold is above **45.05 tokens/s**. Passing only the
  acceptance-conditioned threshold does not establish this stricter target.

Full-model reject replay adds `(1 - a) * W` bytes per round. This is avoidable
work and is **not included in the ideal denominator**. At `a < 0.87683`, that
extra traffic alone prevents reaching 90% of the no-replay weight bound.

The complete model also needs recurrent state, convolution state, attention KV,
intermediate tensors and dequantization/compute. Recurrent read/write traffic
is approximately 302 MB per sweep; main attention KV reads approximately
`65,536 * context_length` bytes, plus MTP KV. Thus long-context bounds differ.
MLX 0.32.2's installed affine `qmv_wide` kernel dequantizes weights once and
reuses them for both input rows at M=2 (`nv_2`, `kl_8`). Thus the two-position
weight-reuse assumption is supported by source, although runtime dispatch and
attainable bandwidth still need measurement.

Those terms and measured hardware bandwidth need explicit accounting; current
low throughput must never be used to lower the target by definition.

## Required evidence before completion

1. Reproducible roofline calculations tied to actual weights, context lengths,
   speculation depth and acceptance; distinguish ideal from avoidable traffic.
2. A profile of the dependent decode pipeline, including verifier, reject work,
   draft and CPU/GPU idle. Independent kernel throughput is insufficient.
3. Correct outputs and cache state across accepted and rejected drafts, mixed
   sequences, long/chunked prefill and the supported reuse paths.
4. Identical workloads and weights in reference/candidate measurements. Report
   aggregate tokens divided by aggregate generation time, not only medians of
   ratios. Report per-prompt outcomes too.
5. Sustained measurements with GPU frequency and CLTM/power state observed.
   A cold burst or GPU active residency percentage does not prove the target.

## Evidence so far

- Last-position MTP projection is implemented and verified. Its isolated
  2048-position component improved from 0.969 s to 0.218 s; full generation gain
  was within noise. Details and raw data:
  `/private/tmp/mtpserve-drift-20260905/OPTIMIZATION.md`.
- Sustained generation has fallen together with explicit CLTM GPU restrictions.
  Both fans reach their reported maxima; High Power is already selected. The
  responsible physical sensor/constraint is unresolved. At 09:36:55 UTC on
  2026-09-05, a fresh read-only snapshot reported `NO_CLTM` and `NO_ZONE` again.
  Prior evidence: `/private/tmp/mtpserve-drift-20260905/REPORT.md`.
- Current experiment: retain the recurrent state after P inside the two-token
  verification kernel. This avoids recomputation using a different batch shape.
  It remains opt-in until full-model/cache validation and pipeline timing pass.

No completion claim is based on component speedups, passing unit checks alone,
or a ceiling that includes the current implementation's avoidable replay.

## Dependent-pipeline profiling

The prompt-2 reference generated the same 200-token hash and 115 attempts /
85 accepts in two observations. The first GPU trace missed decode entirely
because attachment/handshake exceeded its 15-second window; its GPU data are
not evidence about inference. The CPU event recorder still measured 8.562 s
generation. The corrected 40-second trace overlapped decode and generation
took 24.588 s. Later untraced runs were also slow under explicit SLOWCLTM
limits, so profiling overhead and changing controller state are confounded;
the 2.87x timing ratio cannot be attributed to profiling alone. Neither
observation is an optimization result.

The CPU events expose lazy evaluation: in the first observation, the 29
`async_eval` calls immediately following reject replay consumed 1.465 s
(median 47.65 ms); the other 86 consumed 28.82 ms in total. In the traced
observation the corresponding totals were 2.814 s and 46.73 ms. Building a
replay graph itself took only about 0.09 s across 30 calls; that is not its
execution cost. This supports removing full replay from the dependency chain.

Artifacts: `/private/tmp/mtpserve-recovery-20260905/profile-baseline.json` and
`profile-baseline-v2.json`. Raw Instruments traces are private diagnostic
artifacts and must not be shared: Instruments includes process environment
metadata. Read only the explicitly exported performance tables.

## SSM-only recovery experiment: correctness failure

`ssm_recovery=False` remains the default. On real Q4 weights, the first three
64-token coding cases matched reference token IDs and acceptance counts in
ABBA/BAAB runs. Repeated reference runs had bit-identical logical caches.
Candidate caches differed: maximum relative L-infinity differences were about
1.59%, 1.97%, and 2.11% respectively. These numerical differences were recorded,
not treated as tolerance passes.

The fourth prompt changed output at zero-based token index 48 and changed
attempt/accept counts (reference 34/31; candidate 36/29). The test stopped
after writing diagnostics. This is **not an accepted optimization**.
It requires first-reject numerical localization before any default enablement.

Raw comparison: `/private/tmp/mtpserve-recovery-20260905/recovery-correctness.jsonl`.
A simultaneous read-only snapshot again showed `GPU_CLTM` limited to P1/P2,
`SLOWCLTM` and `PWR_ZONE` at 100%; performance under these conditions does not
establish the hardware target.

### First-reject localization

The first rejected draft of prompt 3 occurs in verifier round 11. The primary
token and immediate next argmax match between two-position verification and
one-position reference replay. Input to recurrent layer 0 is bit-identical;
input to layer 1 already differs (maximum relative L-infinity 0.4065%).
Final hidden and logits differ by 0.9494% and 0.7514% respectively.

All 48 control recoveries using the reference replay's own layer inputs match
the reference states bit-for-bit. Previous recurrent states and KV prefixes
restore exactly, every KV offset is correct, and MTP offsets remain unchanged
through verify/replay. Thus this experiment's mismatch is localized to using
two-position hidden inputs in place of the reference's one-position inputs,
not a demonstrated cache-trimming or recurrent-recovery bookkeeping error.
The exact operation producing the first difference is not yet isolated.

This proves a greedy-output compatibility difference; it does not by itself
prove degraded model quality. The opt-in experiment remains disabled by default.
Artifact: `/private/tmp/mtpserve-recovery-20260905/first-reject-p3.json`.

### GPU interval coverage

In the second trace, the generation window spans 24.58835 s. Target-process
Compute active intervals cover approximately 23.705 s (96.4%). An independent
root check restricted to depth-zero Active intervals measured 23.704754 s.
The union of captured activity covers 98.5–98.6%, depending on whether nested
Active intervals are included; roughly 0.34–0.37 s has no captured activity. Other
process activity overlaps target activity and cannot be subtracted as stolen
time. These intervals do not measure shader occupancy, bandwidth efficiency
or theoretical throughput. They limit the likely benefit of eliminating
CPU submission gaps in this observed run, while controller state and tracing
remain confounded.


## Direct verification checkpoint (experimental)

`ssm_checkpoint=False` is the default. This alternative keeps the state after P
inside the same scalar gated-delta Metal loop that verifies [P,D], and retains
the corresponding convolution prefix. On rejection, KV is trimmed by one,
these checkpoints replace recurrent states, and the first verifier logits and
hidden state are retained. Unsupported cache contracts use the original replay.
The `ssm_recovery` and `ssm_checkpoint` flags are mutually exclusive.

The new kernel preserves upstream arithmetic and adds one FP32 checkpoint write:
150,994,944 bytes across 48 recurrent layers per verification, plus convolution
snapshots. This replaces avoidable reject replay; it is not a proven speedup yet.
At real head dimensions (Hk=16, Hv=48, Dk=Dv=128), three seeds in FP32/BF16
produced 18/18 exact GPU comparisons: outputs and final states against upstream
T=2, and checkpoints against upstream T=1 on the same prepared q/k/v/g/beta.
Four kernel guards passed. Six real-small-GatedDeltaNet tests passed separately
on CPU and GPU, including full outputs/cache, both checkpoints, and guard
non-mutation. Seven CPU control-flow tests with fake arithmetic and real cache
classes passed: mixed accept/reject, continuation, last reject, complete accept,
incomplete capture fallback, unsupported contract, and flag mutual exclusion.
These do not claim real-model numerical equivalence. Temporary tests:
`/private/tmp/mtpserve-recovery-20260905/checkpoint_kernel_check.py` and
`test_qwen_checkpoint.py` in the same directory.

### Ordinary greedy / reference MTP / SSM-recompute diagnostic

`modes-gab-64.jsonl` compared G=ordinary greedy, A=reference MTP, and B=SSM-only
recompute across four canonical prompts, with two A and B observations each.
A/A and B/B repeat outputs were identical. For prompts 0 and 2, all methods
matched the first 64 tokens; the MTP loop's existing two-token overshoot returned
65. For prompt 1, A and B matched each other but both differed from G at index
22. For prompt 3, G and A matched the first 64, while B diverged at index 48.
Thus the previous B/A mismatch is real, while ordinary greedy and existing MTP
are themselves not universally token-identical in BF16. This is a numerical
compatibility diagnostic, not a model-quality evaluation.

Aggregate throughput was G=17.685, A=23.242, B=24.681 tok/s, but these figures
are thermally confounded. During prompt 0, observed GPU clock was about
1347-1375 MHz; by the final B call it was about 698 MHz. GPU temperature
reached roughly 101 C earlier in the run. The first prompt's balanced A/B
aggregate was 26.703/30.322 tok/s; it is only a short observation, not sustained
hardware-target evidence. Before the checkpoint tests, a fresh macmon snapshot
again reported NO_CLTM and NO_ZONE. No cooling or power settings were changed.


Full-model T=2 control passed on the actual Q4 model: logits, hidden and all
main-cache arrays were bit-identical to upstream, first after prefill and then
from a restored checkpoint with D trimmed. Both pairs had rejected draft IDs;
all 48 checkpoint pairs (96 tensors) were finite and matched the expected
shapes/dtypes. The second comparison starts both implementations from the same
checkpoint state, so it proves unchanged next-verification behavior, not
bit-equivalence with M=1 replay. Artifact:
`/private/tmp/mtpserve-recovery-20260905/checkpoint-verify-v1.jsonl`.


### Checkpoint generation diagnostic

`checkpoint-gac-64.jsonl` exercised 42 actual rejected drafts with checkpoint
restoration and zero fallback replays (8 C calls, 518 output tokens, 280
attempts / 238 accepts). The corresponding 20-call G/A/C series measured
G=18.744, A=26.687, C=29.078 tok/s in aggregate. C/A is +8.96% in this
observation; changing GPU clocks and temperatures prevent a sustained-speed
claim. Per-prompt A/C aggregates were 26.506/30.825, 26.821/30.207,
28.605/29.524, and 25.055/26.231 tok/s. C/C and A/A repeats were stable.

C differs from A at token index 45 on prompt 0. For prompt 1, A/C match but
both differ from ordinary greedy at index 22. Prompts 2 and 3 match A/C;
prompt 3 has different attempt/accept counts (A=34/31, C=35/30) despite equal
output tokens. The prior SSM-recompute p3 output difference is absent here.
These facts do not constitute a model-quality evaluation or justify default
enablement. Both recovery experiments remain off by default.

At this C acceptance of 0.85, the ideal weight-only depth-one roofline is
46.302 tok/s, and its 90% threshold is 41.671 tok/s. Observed aggregate C is
62.8% of that roofline. Checkpoint/state/context traffic still require explicit
accounting; they do not justify treating current throughput as the ceiling.

## Q4 dependent streaming diagnostic

The temporary `q4_stream_probe.py` loads 128 distinct actual MLP up/down Q4
matrices (6,417,285,120 bytes including scale/bias), chaining actual GPU output
into the next kernel. Q performs stock M=2 quantized matmul; R reads packed
weights and metadata into a retained checksum with the same dispatch layout;
L provides a launch/dependency control without weight reads. This synthetic
zero-seeded MLP chain is not full inference. R includes checksum instructions;
L cannot simply be subtracted as an exact overhead correction.

All three GPU perturbation controls passed (packed word, scale and bias;
exact raw uint32 checksum changes only in the expected output row). The first
Metal compile failed because BF16 output requires an explicit cast; this was
fixed with the upstream-style static_cast, without changing indexing.

Without sustained warmup, v2 Q/R/L means were 52.286/26.048/4.661 ms and Q
fell from 60.3 to 42.6 ms within the short probe. These do not establish a 2x
compute bottleneck. With three seconds of dependent Q warmup, v3 means became
24.407/21.752/2.191 ms. Its monitor stopped before measurement because sample
count was mistaken for duration; no simultaneous-frequency claim is made.
The v4 repeat with a long enough 100-ms monitor measured
23.148/21.499/3.200 ms. Two monitor samples overlap its 0.207-second measurement
window (900 and 1015 MHz, GPU mean temperature about 64.2 C). A preceding
snapshot reported NO_CLTM/NO_ZONE. This is short component evidence, not a
sustained inference result or proof that the memory interface reaches 400 GB/s.

The warm Q/R gap is about 7.7%, substantially smaller than the un-warmed
observation. A fused two-vector version of stock M1 qmv_fast is being explored
for numerical-path consistency. Subsequent correctness and timing evidence
is recorded below; this is not an accepted throughput optimization.
Artifacts: `/private/tmp/mtpserve-recovery-20260905/q4-stream-v{2,3,4}.jsonl`.


## Prompt snapshot reuse fix

Extended CPU tests exposed a pre-existing aliasing defect independent of either
optimization: initial decode -> exact reuse -> exact reuse changed output in
both checkpoint=False and True cases. Restoring `cache[i].state = snap` shared
the saved snapshot list with ArraysCache, whose `__setitem__` replaces list
entries during generation. Both exact and extension restore sites now assign
`list(snap)`; tensor contents remain shared safely, while list mutation no
longer corrupts the prompt boundary. The complete 11-test CPU suite now passes,
including repeated reuse, extension and chunked prefill plus continuation.
The MTP head had a second baseline reuse defect: its generated tail was not
trimmed before exact or extension reuse. State now records the actual
`prompt_mtp_offsets` after prefill, and reuse restores those boundaries before
further forward calls. Canonical pin restoration supplies the same metadata.
Legacy/malformed/unreachable boundaries select cold prefill without mutating
the old caches; offsets are never inferred from token count, because extension
prefill skips its first MTP position.

Both fixes passed six full-model comparisons with MTP history enabled and
checkpoint=False/True: initial vs first and second exact reuse, and equivalent
segmented extension after zero generation vs after a generated tail. Tokens,
attempt/accept counters, all main/MTP cache arrays and offsets were exact.
Exact reuse called neither prefill; extension prefilled only the five-token
tail and kept the actual MTP prompt offset at 69 for a 71-token prompt.
Artifact: `/private/tmp/mtpserve-recovery-20260905/real-reuse-v1.jsonl`.

The CPU regression suites are now checked into the workspace under `tests/`
(no commit made). Canonical local check: `uv run python -m unittest discover
-s tests -v`. All 18 tests pass, including legacy-state fallback and validating
all head offsets before changing any cache. Ruff and py_compile also pass.


### M1-arithmetic Q4 pair prototype

The temporary `qmv_fast_pair.py` specializes upstream qmv_fast for BF16,
Q4/group64, K divisible by 512 and N divisible by 8. Its M2 path shares packed
weight loads while keeping two independent M1 accumulations/reductions.
On four real matrices (layers 0/63, MLP up/down), three nonzero random seeds
and both input rows, 24 M1-clone comparisons against stock M1 and 12 fused-M2
comparisons against two stock M1 calls were bit-exact and finite. The stock
M2 comparison is diagnostic, not the reference arithmetic for this prototype.
No full-model or throughput claim follows from this gate.

An agent's first launch failed during `import mlx.core` with no Metal device;
root executed the same script successfully without that error. No kernel ran
in the failed launch. Artifact:
`/private/tmp/mtpserve-recovery-20260905/qmv-pair-check-v2.jsonl`.


### Paired Q4 throughput and full-model prefix controls

The warmed dependent chain compared stock M2 Q against the M1-arithmetic
paired kernel F using QFFQ repeated four times. Mean wall time was
20.546 ms for Q and 22.007 ms for F: F was approximately 7.1% slower. Mean
synchronized evaluation time was 20.418 versus 21.425 ms; graph construction
was 0.127 versus 0.582 ms. All retained zero-output hashes matched. The
paired prototype has no demonstrated component speed benefit.
Artifact: `/private/tmp/mtpserve-recovery-20260905/q4-pair-v1.jsonl`.

Full-model controls on canonical prompts 0 and 1 compared stock M1(P) with
paired M2(P,D) after restoring all 48 linear-attention checkpoints and trimming
one KV position. All 497 Q4 projections used the paired kernel, with no
fallbacks. First-position logits, hidden state and every main cache tensor
were bit-exact and finite. Temporary classes were restored and parameter
objects remained identical. A further prompt-0 control also compared the
second-position logits/hidden and final cache with two sequential stock M1
calls; that passed exactly as well. These fixed-token controls do not yet
establish correctness throughout generation or an end-to-end speed benefit.
Artifacts: `pair-model-v1.jsonl`, `pair-model-p1-v1.jsonl` and
`pair-model-sequential-p0-v1.jsonl` in the same private directory.


### Paired full-generation control

`pair-decode-v1.jsonl` ran 28 measured calls across all four canonical prompts,
64 requested tokens each, using balanced G/A/F/P orders. G is ordinary greedy,
A stock MTP with replay, F paired MTP with replay, and P paired MTP with
checkpoints. Every G/F/P requested prefix matched; every F/P full token array
and attempt/accept count matched. P emitted 520 tokens over eight calls,
accepted 240/280 drafts, and restored all 40 rejects without replay. Existing
one-token limit overshoot was reported separately. The paired adapter's Python
counters were included in these diagnostic timings; F/P share that overhead.

Per-prompt A/F/P rates were 27.379/26.385/29.421,
25.755/25.546/28.359, 24.728/24.917/25.899, and
15.413/16.610/18.465 tok/s. Weighted totals were A=22.128, F=22.550 and
P=24.688 tok/s: observed P/A +11.57%, P/F +9.48%. These are not stable-clock
speed claims. GPU samples fell from 1365 to 618 MHz across the measured
prompt groups, reached a 395 MHz minimum, and GPU temperature peaked near
101.5 C. Before and after snapshots reported NO_CLTM/NO_ZONE; they do not
establish controller residency during generation. At P acceptance 6/7, the
weight-only bound is 46.480 tok/s, its 90% threshold 41.832, and this P
observation reaches only 53.1% of that bound.

### Threadgroup sweep

The paired kernel's unchanged M1 arithmetic was tested at G=1/2/4 SIMD groups
per threadgroup, keeping four rows per SIMD and 32-lane reduction. All 72
tuned comparisons against two stock M1s and validated G2, plus 12 G2 controls,
were bit-exact and finite. A warmed palindromic dependent-chain sweep measured
Q=33.119, G1=33.024, G2=31.882, G4=32.994 ms. Neither alternative beat G2;
the validated two-group layout is retained. These short component rates with
changing hardware state are not directly comparable to earlier 20-ms probes.
Artifacts: `qmv-pair-tune-v1.jsonl` and corresponding macmon output.


## Durable experimental path

`mtpserve/q4_pair.py` now contains the validated G2 kernel and a reversible
adapter. Its Metal HEADER/SOURCE match the validated temporary prototype
exactly. All observed QuantizedLinear modules are checked before installation,
including root and intermediate modules via MLX named_modules. Unsupported
weights/subclasses fail before mutation. Default timing has no diagnostic
projection counters. Class and parameter-object identity are verified on exit.

`uv run python bench.py --checkpoint-mtp` selects paired checkpoint verification
while leaving the canonical prompts, 200-token limit and 3x4 order unchanged.
An excluded counting warmup must exercise every patched projection and every
actual reject must use its checkpoint; otherwise no benchmark result is
reported. Measurement uses a separate context with counters disabled. Both
reference and experimental output now include weighted generation throughput,
acceptance and raw totals in addition to the existing medians.

The adapter's `verification_only=True` scope is essential: globally pairing
all M2 calls would also change a two-token prefill or a two-token chunk/extension
tail, changing the starting state relative to ordinary greedy. In the benchmark,
only a root model call carrying `ssm_checkpoints` activates pairing. Other calls
use stock projections, and incompatible active inputs raise instead of silently
falling back. The default benchmark/server path keeps stock MTP verification.

Root GPU controls passed after promotion: both positions and final/restored
main cache against two sequential stock M1 calls, plus six exact repeated-reuse
and segmented-extension controls with counters disabled. The final scoped
adapter also passed four stock-prefill identity controls: a two-token prompt,
short chunk tail of two, 2050-token prompt with a 2048+2 split, and a two-token
extension. All logits/hidden/cache arrays matched exactly, with zero pair calls
during prefill. Generation after the short chunked prefill matched greedy for
64 tokens, exercised all 497 paired projections and restored all four rejects.
Artifacts: `repo-pair-sequential-v2.jsonl`, `repo-pair-reuse-v1.jsonl` and
`repo-pair-scope-v1.jsonl` in the recovery directory.

All 37 durable CPU tests pass (engine/cache reuse, adapter lifecycle/dispatch,
and benchmark accounting/coverage); Ruff and diff whitespace checks pass.
Neither this correctness evidence nor the experimental flag establishes the
90% performance objective. The complete canonical sustained run is recorded below.


### Complete canonical checkpoint benchmark

`uv run python bench.py --checkpoint-mtp` completed successfully, including
its counting warmup and all 12 measured 200-token calls. Measurement emitted
2406 tokens in 317.184140 seconds of generation: weighted throughput
**7.586 tok/s** (reported 7.59), median 7.63, range 6.03–9.01. Series medians
were 6.91, 7.97 and 8.84. It accepted 1056/1350 drafts (0.782222) and
checkpointed all 294 actual rejects; the effective-mode guard passed every
call. Peak MLX memory was 17.2 GB. This verifies the canonical experimental
entry point, not token equality beyond the separate controls described above.

The observer covered the entire 362.285-second run. Its measured-call window
(including prefill, excluding warmup) had 1293 GPU samples: frequency
min/median/max 338/351/1235 MHz, GPU temperature 62.45–68.63 C. A snapshot
taken during generation explicitly showed GPU_CLTM P1, SLOWCLTM and PWR_ZONE.
Read-only power checks confirmed AC supply and powermode=2. The responsible
sensor/physical constraint remains unknown; these facts do not establish that
the currently reported GPU temperature alone explains the restriction.

At this run's acceptance, the same ideal weight-only roofline is 44.605 tok/s
and its 90% screening threshold is **40.145 tok/s**. Observed throughput is
17.0% of that bound. The objective is not achieved; do not substitute the
limited current frequency or current implementation rate for the ideal bound.
This constrained run is not a controlled speed comparison with earlier runs.
Artifacts: `bench-checkpoint-v1.out`, `.events.jsonl`, `.macmon.jsonl` and
`bench-checkpoint-during.debug` in the recovery directory. The observer stopped
its own macmon process on completion.


### Local server verification

`uv run mtpserve --model /path/to/model --checkpoint-mtp` selects the same
verification-only paired checkpoint path. It is mutually exclusive with
`--no-mtp`; startup validates model support and all paired projections before
binding. Each decode checks effective checkpoint mode and reject accounting
before retaining the session cache. Experimental failures clear that cache.
The paired adapter stays installed until active request handlers have finished.
Experimental-mode socket waits have a 30-second timeout to bound idle HTTP/1.1
keepalive during shutdown; generation has no added deadline.

A temporary localhost server using the actual model passed health and two
32-token requests. Both response texts matched exactly; the second request
reused all 26 prompt tokens. Metrics recorded 64 generated tokens and 32/32
accepted drafts. This API smoke did not exercise rejected drafts; reject
recovery is covered by the engine GPU controls and 294-reject canonical run
above. The test closed its connections and stopped its own server with SIGINT.
Artifacts: `server-checkpoint-v1.jsonl`, `.server.out` and isolated `.cache`
in the recovery directory.

Final validation: **47 CPU tests passed**, including ten server tests for CLI
constraints, startup validation, checkpoint wiring, cache invalidation and
shutdown behavior. Ruff on all changed Python paths and `git diff --check`
also passed. These changes remain experimental and opt-in.


## Follow-up optimization probes

### Packed-vector loads: no demonstrated improvement

Two temporary G2 variants changed only packed integer loads: V16 uses
`packed_ushort4`; V32 uses `packed_uint2` and extracts four 16-bit halves.
Floating-point expressions and reduction order were retained. Both runs passed
48 exact/finite candidate comparisons and 12 existing-pair controls against
separate stock M1 projections (four real matrices, three seeds).

The first 12-measurement dependent-chain sweep gave F/V16/V32 =
90.342/95.612/77.404 ms, but individual times varied from about 70 to 124 ms.
GPU samples in that window were 338–366 MHz. A repeat with 30 measurements
gave 25.617/25.421/25.559 ms, with GPU samples 876–947 MHz. Both alternatives
were within 0.8% of the existing pair in the repeat. No meaningful repeatable
improvement was established; neither is promoted. These component measurements
do not establish sustained inference performance. Artifacts:
`qmv-pair-vector-v1.*` and `qmv-pair-vector-v2.*` in the recovery directory.

### Same-input gate/up fusion

A temporary two-buffer kernel prepares the two input vectors once and retains
independent gate/up accumulators and output arrays. Weight objects are neither
repacked nor changed. All 24 nonzero output comparisons against separate pair
and two stock M1 calls passed on layers 0/63 and seeds 1/7/23.

A dependent 64-MLP synthetic chain reads 9,625,927,680 weight bytes and uses
the same stock SwiGLU and paired down projections in both variants. Three
S/F/F/S blocks measured separate=50.407 ms and fused=48.897 ms, a 3.0%
reduction; mean graph construction was 0.924/0.829 ms. Sampled GPU frequency
fell from about 704 to 673 MHz. This is a component observation requiring
full-model/cache and generation verification before integration. Artifacts:
`qmv-pair-gateup-v1.*`.

### Depth-two acceptance economics

The existing `mtp_depth=2` branch was measured without changing its behavior.
An isolated AST copy adds scalar recording immediately after the existing
`mx.eval(p0,p1,d1,d2,primary)`; removing that recording reproduces the original
AST. Original engine source bytes were checked unchanged. Counting every
computed D2 avoids hiding the second head's cost when D1 is rejected.

Prompt 3, excluded G/A/B 16-token warmups, then G/A/B/B/A at 64 requested
tokens: every requested prefix matched greedy and depth-one reference. Existing
limit overshoot returned 65 tokens for depth one and 66 for depth two. The two
depth-two calls had 50 total rounds, 46 accepted D1 and 36 accepted D2
conditional on D1 acceptance: conditional D2 acceptance was 78.26%. Four
rounds replayed P and ten replayed P,D1. Weighted G/A/B rates were
18.714/27.292/25.732 tok/s; varying GPU frequencies and diagnostic scalar
recording preclude a sustained comparison or attributing the difference to
replay alone. Artifacts: `depth2-p3-v1.*`.

This establishes a reason to test checkpointed depth two. The target is one
three-position verifier, with exact recoverable recurrent/convolution states
after P and after P,D1. Rejection restores the corresponding prefix and trims
KV; it must not replay the full model. Three-row Q4 projections must preserve
each stock M1's arithmetic, while prefill and the draft head keep their current
behavior. Unsupported cases keep the reference path. Kernel-level numerical
gates precede model/cache controls and generation comparisons.

At this diagnostic's observed 2.64 emitted tokens per round, the optimistic
weight-only depth-two bound is `400e9 * 2.64 / (W + 2D)` = 60.182 tok/s
and its 90% threshold is 54.164 tok/s. This assumes one weight read across
all three verifier rows; it is neither a measured attainable ceiling nor
evidence that the new path exists or reaches the objective.


### Long-context generation and forced rejection control

The current durable depth-one checkpoint path matched greedy through EOS
(37 emitted tokens) after a synthetic 2050-token prompt with the default
2048+2 prefill split. An additional correctness-only run rotated every third
MTP output's vocabulary logits by one position to exercise rejected drafts.
It still matched all 37 greedy tokens: 22 attempts, 14 accepts and all eight
rejects restored from checkpoints. All 497 paired projections were exercised;
model/projection classes and parameter identities were restored. The artificial
drafter perturbation is explicitly excluded from performance claims. These
controls cover one synthetic long prompt, not arbitrary long-context output.
Artifacts: `pair-long-generation-v1.*` and
`pair-long-generation-forced-v1.*` in the recovery directory.


### Gate/up fusion full-model result: not promoted

The temporary fusion adapter passed exact numerical comparisons of full/first
logits and hidden states, all main cache arrays, 48 recurrent/convolution
checkpoint pairs and restored-prefix caches, including another verification
after restoration. These array comparisons do not distinguish signed zeros.
A two-token call without checkpoint capture exercised zero fused MLPs and kept
stock output/state. Checkpoint calls exercised all 64 fused MLPs plus 369
remaining paired projections; gate/up intentionally bypass their original
per-projection counters. Classes and parameter identities were restored.

A prompt-3 P/F/F/P generation probe (excluded P/F warmups of 16, then 64
requested tokens per call; counters disabled during measurement) matched full
token arrays and attempt/accept/checkpoint counts. Each mode emitted 130
tokens, accepted 60/70 drafts and checkpointed all ten rejects. Weighted
throughput was P=32.071 and F=31.380 tok/s: fusion was 2.15% slower in this
observation. P endpoints were 32.085 and 32.057; F was 31.512 and 31.249.
Median GPU frequencies across the calls were 1376/1376/1353/1375 MHz. The
component improvement did not transfer to this full-generation probe, so fusion
is not promoted. Artifact: `qmv-pair-gateup-decode-v1.*`.

### Three-position component gates

The initial custom M3 projection shares scalar packed integer loads among
three independent M1-ordered accumulators. All 24 candidate comparisons and
12 existing-pair controls passed, exact and finite, on real up/down matrices
from layers 0/63 and seeds 1/7/23. Its dependent 128-matrix ABBA sweep was
slower than stock M3: 34.888 versus 23.785 ms. Every measured block showed
the same direction. It is not integrated; reducing temporary live values and
changing threadgroup size are separate hypotheses to test. Artifact:
`qmv-triple-v1.*`.

The T3 recurrent kernel retains states after both P and P,D1 while preserving
the previous scalar arithmetic. On real head dimensions B=1,Hk=16,Hv=48,
Dk=Dv=128, BF16/FP32 and seeds 1/7/23, all 24 GPU comparisons passed:
output/final state against upstream T3, and each prefix state against upstream
T1/T2 on the same prepared inputs. Four GPU input-contract guards also passed.
This establishes the component contract, not full-model equivalence or speed.
Artifact: `checkpoint3-kernel-gpu-v1.*`.

Two exact M3 tuning variants also passed 48 candidate comparisons plus 12
validated-M3 controls. G1 uses one SIMD group per threadgroup; OD prepares
third-row values at use sites to shorten their lifetime. A balanced
B/G1/OD/OD/G1/B sweep repeated three times measured 32.070/31.054/30.359 ms
respectively. OD reduced component wall time by 5.34% relative to the exact
base M3 in this run. Six GPU samples overlapping the 0.59-second measurement
window were 1343–1365 MHz (median 1358.5); GPU temperature was 70.30–71.87 C.
This is a short component result, not sustained generation or a same-window
comparison with stock M3. Neither variant is promoted. Artifact:
`qmv-triple-tune-v1.*`.

### Current depth-one CPU submission observation

A temporary P/I/I/P prompt-3 probe used the durable depth-one checkpoint mode;
I wraps existing base/head/eval/async_eval Python calls with clock recording and
adds no GPU synchronization. After an excluded 16-token warmup, all four calls
emitted the same 65 tokens, accepted 30/35 drafts and checkpointed all five
rejects. Rates were 31.755/31.834/31.926/31.751 tok/s. GPU samples in each
call were 1362–1379 MHz overall, with per-call medians 1375/1375/1373/1370.5.
This is a short diagnostic, not a sustained benchmark.

After excluding prefill events, the two instrumented calls recorded 35 main
verifier graph submissions: 186.50/190.66 ms total, medians 5.235/5.351 ms.
The 35 draft graph submissions used 3.86/3.79 ms total; 40 async_eval calls
used 11.21/10.75 ms total. Existing blocking eval calls occupied
1832.67/1823.46 ms of Python wall time. These are CPU-call durations around
lazy execution, not GPU kernel durations; graph construction can overlap GPU
work and cannot simply be subtracted from generation time as a speedup.
Artifacts: `profile-checkpoint-v2.jsonl`, `.out`, and the overlapping
`profile-checkpoint-v1.macmon.jsonl`. The first runner attempt stopped before
warmup because its decode call omitted the required `use_mtp` argument; it
provides no inference measurement.

### Depth-two full-model checkpoint control

The isolated T3 adapter passed a full-model GPU control after prompt 3: three
sequential stock M1 calls versus one checkpointed T3 call using the exact base
M3 projection. Every position's logits/hidden, the complete final cache, and
both retained prefix caches matched exactly and were finite. Restoring P trims
two KV positions; restoring P,D1 trims one. The control materialized independent
cache snapshots before subsequent mutable KV writes.

All 497 projections used M3 once, with zero fallback; all 48 recurrent layers
provided both convolution/recurrent prefix pairs. Model/projection classes,
module hooks and parameter identities were restored. Equality is numerical and
does not distinguish signed zeros. This validates one full-model verification
and both restore boundaries; generation and speed remain separate gates.
Artifact: `checkpoint3-model-v1.jsonl` / `.out`.

The separate serial-input M3 candidate failed its first real-weight numerical
gate before timing: layer-0 up projection, seed 1, 17408/52224 output elements
differed, maximum absolute error 4221.8125. Base M3 and OD controls passed in
the same process. The serial candidate is excluded; the cause is under isolated
diagnosis. Artifact: `qmv-triple-serial-v1.jsonl` / `.out`.

### Depth-two generation result: correct, but not promoted

The private AST-copy decode removed both full-model replay branches and used
the validated T3 prefix states. Prompt-3 correctness runs matched the requested
64-token greedy prefix. A separate fault-injection run exercised seven D1 and
seven D2 rejects; all 14 restored checkpoints. Normal runs also matched, and
same-mode full tokens/acceptance/checkpoint counts repeated exactly. The
original engine and model-gate source hashes remained unchanged. Existing
max-token overshoot is retained: P returned 65 tokens and C returned 66.

After excluded A/P/C 16-token warmups, measured P/C/C/P gave weighted rates
**P=31.966 and C=29.017 tok/s**: depth two was 9.23% slower in this short
observation. P emitted 130 tokens in 4.066768 seconds; C emitted 132 in
4.549078 seconds. C had 50 verifier rounds, accepted D1 in 44 and D2 in 38;
all six D1 and six D2 rejects restored without replay. The stronger conditional
D2 acceptance (38/44) did not by itself produce a throughput gain.

Per-call GPU frequency medians were 1366/1351/1342/1348 MHz; sampled
temperature rose from 87.31 to 96.56 C. P endpoints were 31.925/32.008 and
C was 29.060/28.973 tok/s. This is not sustained throughput or an isolation of
the remaining cost. The candidate is not integrated. Artifacts:
`checkpoint3-decode-v1.jsonl`, `.out`, `.macmon.jsonl`.

### Depth-two cost diagnosis

A separate instrumented P/C/C/P observation reproduced the direction:
P=31.910/32.085 and C=29.472/29.284 tok/s, with per-call median GPU
frequencies 1375/1360/1360/1373.5 MHz. Existing main-model Python calls
averaged 5.30–5.36 ms in P and 5.19–5.33 ms in C; they do not explain the
extra C time by larger graph-construction duration. Blocking eval calls
averaged approximately 50.5–50.7 ms in P and 80.6–81.0 ms in C (including
each generation's final eval). These are CPU submission/wait durations, not
exclusive GPU kernel measurements. Both modes retained their exact repeated
token/count results and shared requested prefixes. Artifact:
`profile-checkpoint3-v1.*`.

Read source also shows that depth two does not submit `async_eval(d2)` before
constructing the verifier graph. Testing an asynchronous submission and testing
last-position-only draft vocabulary projection are separate pending hypotheses;
neither is included in the recorded candidate or claimed as a gain.

The serial-input diagnostic localized its corruption to vector 0 on the
17408x5120 real matrix; vectors 1/2 matched. Splitting the sum initializer,
using separate input arrays, or returning to device weight reads did not fix
that failure. Every variant matched on an 8x512 real-weight slice. These tests
do not establish a compiler cause or isolate N versus K. This branch is stopped
with cause unknown; no serial kernel is promoted. Artifact:
`qmv-triple-serial-diagnose-v1.*`.

### Last-position draft head at depth two: no demonstrated gain

A separate private `_mtp_step` AST copy sliced only the input to final norm/head;
the returned full hidden sequence and every decoder/cache position remained.
Real prompt/MTP-prefix controls with one, two and three carry positions matched
full hidden arrays, KV arrays and all offsets exactly and finitely. Last logits
were identical for one position; two/three positions differed in 173489/160012
vocabulary elements while retaining the same argmax in these controls. Draft
logit differences are permitted diagnostics, not an equality pass.

Generation matched the requested 64-token greedy prefix, including a separate
control with seven rejects of each draft stage. Excluded C/L16 warmups then
C/L/L/C64 gave **C=29.136 and L=29.053 tok/s** (a 0.28% difference), with
identical full tokens, attempts/accepts, 12 checkpointed rejects per mode and
MTP cache offsets. Per-call GPU medians were 1359/1323/1351/1333 MHz. This
short observation demonstrates no speed gain; the change is not promoted.
Artifact: `checkpoint3-last-head-v1.*`.

### Early asynchronous D2 submission: short-run improvement

An isolated AST copy added only `mx.async_eval(d2)` after D2 argmax and before
MTP trim/verifier graph construction. The existing blocking eval, head, state
recovery and counters remained unchanged. Excluded C/X16 warmups then C/X/X/C64
matched the frozen normal C oracle's complete 66 tokens and all counts, with
both reject paths exercised and every context restored.

Weighted rates were **C=29.345 and X=31.489 tok/s**, a 7.31% increase. C
endpoints were 29.323/29.367; X was 31.498/31.481. Per-call median GPU
frequencies were 1364/1359/1362/1358 MHz. Each mode emitted 132 tokens with
82/94 accepts and all 12 rejects checkpointed. This supports earlier GPU
submission for this short workload; it does not establish sustained speed or
superiority to the durable depth-one path. Artifact: `checkpoint3-async-v1.*`.

### Two output rows per SIMD: component improvement

A separate M3 layout retained the validated helper/arithmetic and all three
independent input buffers, reducing output rows per SIMD from four to two.
R2G2/R2G4 vary the number of SIMD groups per threadgroup. Both passed all 48
candidate comparisons and 12 base controls against three stock M1 calls on
real up/down projections, layers 0/63 and seeds 1/7/23.

The balanced B/R2G2/R2G4/R2G4/R2G2/B dependent-chain sweep repeated three
times measured **31.402/24.594/24.479 ms**, approximately 22% less component
wall time. This changes six persistent output accumulators versus twelve; an
occupancy explanation remains a hypothesis without hardware allocation proof.
Full-model prefix-state and generation controls are pending before integration.
Artifact: `qmv-triple-rows2-v1.*`.


### R2 full-model and combined generation gates

Both actual R2G2 and R2G4 kernels passed fresh three-stock-M1 controls: all
three logits/hidden arrays, the full cache, both restored prefixes, all 48
recurrent layers and all 497 projections were exact and finite. Kernel hashes,
actual dispatch, helper/class/parameter cleanup were recorded independently.
No full-token or count difference remained between X and either R2 variant.

Excluded G64 and P/X/G2/G4 warm16, then P/X/G2/G4/G4/G2/X/P64, gave weighted
**P=29.379, X=28.676, R2G2=32.619, R2G4=32.436 tok/s**. G2 was 11.03% above
P in this short observation. All 12 actual rejects per depth-two mode restored
checkpoints. Generation GPU frequency medians were 1364/1354/1342.5/1249/1140/
1125/1312/1327 MHz; G2 endpoints were 34.900/30.618. Frequency drift prevents a
clean choice between G2 and G4. G2 is selected provisionally for the bounded
integration; no occupancy cause or sustained gain is claimed.
Artifacts: `checkpoint3-rows2-v1.*`.

The follow-up ran excluded G64 plus P/R16 warmup and measured P/R/R/P64 for
each of all four canonical prompts. All requested greedy prefixes and all
same-mode full tokens/counts matched. P emitted 520 tokens in 17.475190 s;
R2G2 emitted 524 in 16.968233 s: **29.756 versus 30.881 tok/s, +3.78%**.
R2 had 28 first-draft and 26 second-draft rejects, all 54 checkpointed.
These are short 64-token requests, not the canonical sustained 200-token run.
Artifact: `rows2-prompts-v1.*`.

### Stock depth-two asynchronous submission control

The sole early `mx.async_eval(d2)` addition also passed a separate existing
stock-depth-two control, without recurrent checkpoint or Q4 adapters. Excluded
A/X16 warmup and G64, measured A/X/X/A64: **A=30.260, X=31.642 tok/s (+4.57%)**.
Full tokens/counts, final main/MTP caches, prompt snapshots/logits/hidden/offsets,
and exact-prompt reuse were exact and finite. Each mode had 14 replayed rejects;
reuse skipped all 63 prefill tokens and matched fresh warm generation. Generation
GPU medians were 1325.5/1343/1350.5/1338 MHz. This validates a short scheduling
improvement and correctness, not a sustained-rate claim.
Artifact: `stock-depth2-async-v1.*`.

The native integration adds `--mtp-depth 2` to benchmark/server while retaining
one as the default. It uses R2G2 only for checkpoint-marked T3 verification and
stores both rejection prefixes directly in the recurrent kernel. The native
D2 attempt counter now increments after the D1 EOS guard, so accepted D1 EOS
no longer counts an untested D2 as a reject. Native full-model/generation and
sustained checks are pending; private-probe evidence alone does not certify
this integration.


Native integration verification now passed its fresh full-model gate: all three
outputs, full caches and both restored prefixes were exact and finite, with
48 recurrent layers, 497 actual triple calls and complete class/parameter
cleanup. The native R2G2 Metal text hash matches the tested private kernel
(`9b021c51ee32061ee3fa181455c2b05fdb08f690c19371a6a56f208f05906b64`), and the
native T3 recurrent module is byte-identical to the validated private module.
All 67 CPU tests, focused Ruff and diff whitespace checks passed. New control
coverage includes both rejection branches, all-accept, primary/D1/D2 EOS,
invalid unused prefix and unreachable KV offsets, atomic replay fallback,
repeat prompt reuse, CLI depth routing and three-row warmup coverage.
Artifacts: `integrated-depth2-model-v1.*`, `repo-depth2-tests-v1.out`.


The native depth-two canonical benchmark completed: excluded 200-token warmup,
then the unchanged three series of four 200-token requests. It emitted **2409
tokens in 82.417506 seconds of generation = 29.229 tok/s**, median 29.34,
range 24.53–33.75, series medians 31.26/29.51/28.31. All **471** actual
rejects restored checkpoints; 1365/1836 drafts were accepted. Warmup coverage
exercised all projections and was excluded from the totals; measured counters
were disabled. Peak memory was 17.4 GB. Process exited zero and every watched
runtime source hash was unchanged.

Whole-call GPU median frequencies across the 12 measured calls were
1340.5/1291/1217/1183.5/1179/1158/1101.5/1102/1092/1073/1068/1027 MHz;
GPU temperature at call ends ranged from 87.4 to 99.5 C. These include prefill,
so they are not exclusive generation-frequency samples. This observed decline
and the earlier 338 MHz run prevent attributing the difference from the old
7.586 tok/s benchmark entirely to code.

With 2409/(2409-1365)=2.30747 output tokens per verification round, the same
400 GB/s weight-traffic model gives a depth-two upper bound of **52.602 tok/s**
and a 90% threshold of **47.341 tok/s**. The measured 29.229 is approximately
55.57% of that bound. No lower-frequency ceiling is substituted and the >90%
goal is not achieved. Artifacts: `bench-checkpoint-depth2-v1.out`,
`.events.jsonl`, `.macmon.jsonl`.


Native generation integration passed the frozen experimental R2G2 oracle and
full main/MTP/prompt state comparison at prompt 3, including exact reuse. All
four native 64-token outputs/counts matched their saved R2 and requested greedy
oracles; every 16-token exact reuse matched a fresh call's tokens/counts and
complete state. The 2050-token synthetic prompt (2048+2 prefill) produced the
same 37-token EOS-terminated greedy output normally and with forced drafts.
The injected run restored six P prefixes and six P,D1 prefixes, with all 12
actual rejects checkpointed. Class, parameter identity, hooks, engine helpers,
private callable and source-hash cleanup passed. These controls were correctness
runs, not additional throughput benchmarks. Artifact: `integrated-depth2-v1.*`.

A localhost native server smoke control at port 19236 also passed health plus
two identical nonstreaming 64-token requests: identical choices, 65 output
tokens each, second request reused all 23 prompt tokens. Metrics recorded
130 generated, 88 attempted, 86 accepted and two total rejected drafts; server
success required each reject to be checkpointed. The owned server was stopped
with SIGINT after verification. Artifacts: `api-depth2-v1.json`, `.out`.

### Further output-row layout probes

The next component gate changed only output ownership, accumulator-array sizes
and output-row loop bounds; floating-point expressions, input preparation and
packed-weight loads stayed unchanged. It used the same dependent 128-matrix,
6.417 GB MLP chain and balanced forward/reverse order, with compilation and
warmup excluded. Nonzero real-matrix checks passed 48 exact finite candidate
comparisons plus 12 native controls separately for M2 and M3.

For M2, native R4G2 averaged **21.241 ms**, R2G2 **19.408 ms** and R1G2
**21.002 ms**. R2 reduced component wall time by 8.63%; full-generation benefit
is pending. The measured window contained four GPU samples at 1353–1367 MHz
and 66.65–68.32 C. Artifact: `qmv-pair-rows-v1.*`.

For M3, native R2G2 averaged **24.097 ms**, R1G2 **27.251 ms** and R1G4
**27.331 ms**. Both one-row candidates are rejected as slower; no full-generation
test or runtime change is justified by these results. Four GPU samples in the
measured window were 1294–1347 MHz and 73.20 C. These subsecond component
measurements do not establish sustained throughput or actual register occupancy.
Artifact: `qmv-triple-rows1-v1.*`.

Keeping M3's native two output rows but reducing the threadgroup to one SIMD
group also passed its 24 candidate comparisons and 12 native controls. Balanced
B/G1/G1/B blocks averaged **28.716 ms native versus 29.392 ms G1**. This
candidate is not promoted. Absolute times from separate probe runs are not
used to claim a cross-run regression. Artifact: `qmv-triple-rows2-g1-v1.*`.

The remaining M2 R2 candidate passed native depth-one full generation: excluded
B/R16 warmup covered all 497 projections, then ordinary G64 supplied the greedy
reference and measured B/R/R/B64 ran with projection counters disabled. Full
tokens/counts and final main/MTP/prompt state were exact and finite; all ten
actual rejects per mode restored checkpoints, and class/parameter/hook/source
cleanup passed. Each mode emitted 130 tokens: native B took **4.310785 s
(30.157 tok/s)**, R2 took **4.239300 s (30.665 tok/s)**, a short-run difference
of **+1.69%**. This small result does not establish a sustained benefit; the
native pair kernel is unchanged. Artifact: `native-pair-rows-gen-v1.*`.

### Native depth-two checkpoints with stock projections

A correctness-first screening used the current native depth-two engine with
checkpoints enabled and no Q4 projection adapter. On canonical prompt 0 its
requested 64-token prefix differed from ordinary greedy at zero-based token
index **45**. All ten actual rejects used checkpoints. The probe stopped at
that first mismatch; classes, parameter identity and runtime source hashes
were unchanged. This combination is rejected as a greedy-compatible
optimization, and its unbalanced timings are not performance evidence.
Artifact: `stock-checkpoint-depth2-v1.*`.
