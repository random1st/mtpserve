# mtpserve

OpenAI-compatible inference server for MLX models on Apple Silicon, built for
coding agents that hammer the same system prefix over and over.

Three things make it fast:

- **MTP speculative decoding** — the model's own multi-token-prediction head
  drafts the next token; ~1.5x generation at 90%+ acceptance on Qwen3.5/3.8
  dense models (works on hybrid attention/SSM architectures where external
  draft models can't).
- **Pinned prefix cache** — the stable prefix shared by agent sessions
  (system prompt + tool schemas) is detected automatically as the longest
  common prefix of consecutive session-initial prompts, prefilled once,
  snapshotted, and persisted to disk. New sessions start in ~0.3s instead of
  re-prefilling thousands of tokens. Survives restarts.
- **Exact/extension prefix reuse** — within a session every agent step reuses
  the previous KV + recurrent state (hybrid models can't trim backwards, so
  the server snapshots recurrent layers at the prompt boundary and rolls
  forward only).

Also: real SSE streaming, tool calling in both Qwen dialects (Hermes JSON and
XML `<function=...>`), `reasoning_content` extraction, `/metrics`.

## Setup

One command from scratch (installs uv if missing, downloads the model
~16 GB, builds the MTP head ~3 GB download, runs a smoke test):

```sh
sh install.sh
# already have the MLX model locally?
MTPSERVE_MODEL_DIR=/path/to/your-mlx-model sh install.sh
```

Or manually:

```sh
uv sync

# build the MTP head from the original checkpoint (one-time, ~3 GB download):
uv run python scripts/add_mtp_weights.py \
    --mlx-model-path /path/to/your-mlx-model \
    --source-model Qwen/Qwen3.8-27B --no-quantize

uv run mtpserve --model /path/to/your-mlx-model --port 19234
```

The MTP head must stay in BF16 (`--no-quantize`): quantizing it collapses
draft acceptance to zero.

## Endpoints

- `POST /v1/chat/completions` — JSON or SSE (`stream: true`), tools supported
- `GET /v1/models`, `GET /health`, `GET /metrics`

## Numbers (M3 Max, Qwen3.8-27B 4-bit)

| | |
|---|---|
| generation | ~33 tok/s (~20 without MTP), acceptance ~92% |
| session-initial request | ~1.2s with pin vs ~10s cold |
| end-to-end agent task (read + edit) | ~5s |

Single sequence, single process; requests are serialized. That is the
intended use: one local agent, minimum latency.


## Experimental checkpoint mode

```sh
uv run python bench.py --checkpoint-mtp
uv run mtpserve --model /path/to/your-mlx-model --checkpoint-mtp
# Optional depth-two verifier with both reject checkpoints:
uv run python bench.py --checkpoint-mtp --mtp-depth 2
uv run mtpserve --model /path/to/your-mlx-model --checkpoint-mtp --mtp-depth 2
```

This opt-in path shares Q4 weight reads using the ordinary one-token arithmetic
and retains the recurrent states needed when an MTP draft is rejected. Sharing
is limited to checkpoint verification; prefill and the draft head keep their
original execution. Depth defaults to one; `--mtp-depth 2` enables two draft
tokens, exact three-row Q4 verification, and separate states after P and P,D1
for either rejection. Both depths remain experimental. The benchmark uses the
same weights and prompts; its excluded warmup verifies actual projection coverage, then diagnostic counters are disabled
for measurement. The server flag selects the same verification path and is
mutually exclusive with `--no-mtp`.
The supported kernel is BF16, affine Q4/group64 with K divisible by 512 and N
by 8. Unsupported quantized projections are rejected before model mutation.
The adapter restores the original model classes when the benchmark or server
stops. In the experimental server mode, socket I/O waits are limited to
30 seconds so idle HTTP connections cannot hold shutdown indefinitely; this
does not impose a generation deadline.

Both depths matched ordinary greedy output on four short prompts. Depth two
also passed full-model state, prompt reuse, 2050-token context and forced
rejection controls. Its canonical 2409-token run measured 29.23 tok/s on the
tested M3 Max; a sustained speedup over depth one and the 90% hardware target
remain unproven. See [performance evidence](PERFORMANCE.md)
for timing, thermal limits and validation scope. The default benchmark remains
`uv run python bench.py`.
