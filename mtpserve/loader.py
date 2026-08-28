# SPDX-License-Identifier: Apache-2.0
"""Загрузка MLX-модели с инжекцией MTP-головы.

mlx_lm грузит модель (в т.ч. со смешанным квантованием), после чего
qwen3_5_mtp.inject_mtp_support навешивает MTP-модуль, если в каталоге модели
лежит mtp/weights.safetensors (собирается scripts/add_mtp_weights.py).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model(model_path: str, tokenizer_config: dict | None = None):
    """Возвращает (model, tokenizer); model развёрнута до уровня с MTP.

    mlx_lm для семейства qwen3_5 оборачивает текстовую модель во внешний
    Model, а инжектор патчит внутренний language_model — возвращаем тот
    уровень, у которого есть mtp_forward/return_hidden (qwen3_next обёртки
    не имеет, там патчится сама модель).
    """
    from mlx_lm import load

    model, tokenizer = load(model_path, tokenizer_config=tokenizer_config or {})

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        try:
            from .qwen3_5_mtp import inject_mtp_support

            inject_mtp_support(model, model_path, config)
        except Exception:  # noqa: BLE001 — MTP это ускорение, не обязанность
            logger.exception("MTP injection failed; continuing without drafter")

    if (not hasattr(model, "mtp_forward")
            and hasattr(model, "language_model")
            and hasattr(model.language_model, "mtp_forward")):
        model = model.language_model
    return model, tokenizer
