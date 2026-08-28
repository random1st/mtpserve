#!/bin/sh
# Установка mtpserve с нуля на Apple Silicon.
#
#   sh install.sh                 — модель по умолчанию (Qwen3.8-27B, ~16 GB
#                                   весов + ~3 GB шард для MTP-головы)
#   MTPSERVE_MODEL_DIR=/path ...  — использовать уже скачанную MLX-модель
#
# Переменные:
#   MTPSERVE_MODEL       HF-репозиторий MLX-модели (orcarouter/Qwen3.8-27B-Uncensored-MLX)
#   MTPSERVE_MTP_SOURCE  оригинальный чекпойнт с mtp-тензорами (Qwen/Qwen3.8-27B)
#   MTPSERVE_MODEL_DIR   готовый локальный каталог модели (пропускает скачивание)
#   MTPSERVE_MODELS      куда качать модели (~/.cache/mtpserve/models)

set -eu

REPO_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_HF="${MTPSERVE_MODEL:-orcarouter/Qwen3.8-27B-Uncensored-MLX}"
SOURCE_HF="${MTPSERVE_MTP_SOURCE:-Qwen/Qwen3.8-27B}"
MODELS_DIR="${MTPSERVE_MODELS:-$HOME/.cache/mtpserve/models}"

say() { printf '\033[1m== %s\033[0m\n' "$*"; }
die() { printf 'ошибка: %s\n' "$*" >&2; exit 1; }

# --- платформа ---------------------------------------------------------------
[ "$(uname -s)" = Darwin ] || die "mtpserve работает только на macOS (MLX)"
[ "$(uname -m)" = arm64 ] || die "нужен Apple Silicon"

# --- uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    say "ставлю uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv не встал; поставь вручную: https://docs.astral.sh/uv/"
fi

# --- зависимости -------------------------------------------------------------
say "зависимости (uv sync)"
cd "$REPO_DIR"
uv sync

# --- модель ------------------------------------------------------------------
if [ -n "${MTPSERVE_MODEL_DIR:-}" ]; then
    MODEL_DIR="$MTPSERVE_MODEL_DIR"
    [ -f "$MODEL_DIR/config.json" ] || die "в $MODEL_DIR нет config.json"
    say "модель: $MODEL_DIR (локальная)"
else
    MODEL_DIR="$MODELS_DIR/$(basename "$MODEL_HF")"
    if [ -f "$MODEL_DIR/config.json" ]; then
        say "модель уже скачана: $MODEL_DIR"
    else
        say "качаю $MODEL_HF -> $MODEL_DIR (~16 GB)"
        mkdir -p "$MODELS_DIR"
        uv run python - "$MODEL_HF" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
    fi
fi

# --- MTP-голова --------------------------------------------------------------
if [ -f "$MODEL_DIR/mtp/weights.safetensors" ]; then
    say "MTP-голова уже собрана"
else
    say "собираю MTP-голову из $SOURCE_HF (BF16; квантование ломает acceptance)"
    uv run python scripts/add_mtp_weights.py \
        --mlx-model-path "$MODEL_DIR" --source-model "$SOURCE_HF" --no-quantize
fi

# --- смоук-тест --------------------------------------------------------------
say "смоук-тест (загрузка модели + один запрос)"
SMOKE_PORT=19299
nohup uv run mtpserve --model "$MODEL_DIR" --port "$SMOKE_PORT" \
    > /tmp/mtpserve-install-smoke.log 2>&1 &
SMOKE_PID=$!
i=0
until curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$SMOKE_PORT/health"; do
    i=$((i + 1))
    [ "$i" -gt 90 ] && { kill "$SMOKE_PID" 2>/dev/null || true; \
        die "сервер не поднялся, см. /tmp/mtpserve-install-smoke.log"; }
    sleep 1
done
ANSWER=$(curl -s --max-time 120 "http://127.0.0.1:$SMOKE_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"Reply with exactly: ok"}]}' \
    | uv run python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
kill "$SMOKE_PID" 2>/dev/null || true
[ -n "${ANSWER}" ] || die "пустой ответ смоук-теста, см. /tmp/mtpserve-install-smoke.log"
say "смоук-тест пройден: модель ответила: ${ANSWER}"

# --- итог --------------------------------------------------------------------
cat <<DONE

Готово. Запуск сервера:

    cd $REPO_DIR && uv run mtpserve --model "$MODEL_DIR" --port 19234

API: OpenAI chat completions на http://127.0.0.1:19234/v1
(эндпоинты: /v1/chat/completions, /v1/models, /health, /metrics)
DONE
