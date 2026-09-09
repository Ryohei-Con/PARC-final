"""上流に届かない 1 点だけを monkeypatch する（計画 §3 の例外規定）。

計画 §3 の方針は overlay（公開拡張点だけを使い、上流を 1 バイトも編集しない）である。
その例外規定に「どうしても monkeypatch が要る場合は **この 1 ファイルに集約**し、
適用時に ``logging.warning`` で必ず名前を出す」と書いてある。本モジュールがそれである。

**なぜ必要か（VLM の学習率 0.1 倍のため）**

``lerobot/optim/factory.py:37`` は::

    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    optimizer = cfg.optimizer.build(params)

であり、どちらの分岐でも **パラメータ名が失われる**（``InternVLAA15Policy.get_optim_params()``
は ``modeling_internvla_a1_5.py:1439`` で ``self.parameters()`` を返すだけ）。名前が無いと
「どれが VLM か」を判定できないので、param group を分けられない。

さらに ``configs/train.py:115-119`` は ``use_policy_training_preset=True`` のとき
``cfg.optimizer = policy.get_optimizer_preset()`` で **CLI の ``--optimizer.*`` を上書きする**。
``False`` にすると今度は ``policy.parameters()`` 側（名前なし）に落ちるので、CLI だけでは
どうやっても届かない。

よって次の 2 点だけを差し替える:

P1. ``InternVLAA15Policy.get_optim_params`` → ``dict(self.named_parameters())``
    （AdamW は dict も iterable も受けるので、上流の既定 optimizer に戻しても壊れない）
P2. ``InternVLAA15Config.get_optimizer_preset`` → :class:`ParcAdamWVlmScaledConfig`
    （lr / betas / eps / weight_decay / grad_clip_norm は上流の preset と同じ config
    フィールドから引く。**変わるのは param group の分割だけ**）

scheduler preset は触らない。``LambdaLR`` は各グループの ``initial_lr`` に同じ係数を
掛けるので、0.1 の比は warmup / cosine decay を通して保たれる。

``IVLA_VLM_LR_SCALE`` で倍率を変えられる（既定 0.1）。値は
:class:`ParcAdamWVlmScaledConfig` のフィールドとして ``train_config.json`` に記録される
ので、後から「その run が本当に 0.1 だったか」を checkpoint から確認できる。
``IVLA_VLM_LR_SCALE=1.0`` にすれば分割は残したまま上流と同じ挙動になる（A/B 用）。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: 倍率の環境変数と既定値（ユーザー指定: バックボーン VLM は 0.1 倍）
ENV_VLM_LR_SCALE = "IVLA_VLM_LR_SCALE"
DEFAULT_VLM_LR_SCALE = 0.1

_APPLIED: list[str] = []


def vlm_lr_scale() -> float:
    raw = os.environ.get(ENV_VLM_LR_SCALE)
    if raw is None or raw.strip() == "":
        return DEFAULT_VLM_LR_SCALE
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_VLM_LR_SCALE} を float として読めない: {raw!r}") from exc
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{ENV_VLM_LR_SCALE} は (0, 1] の範囲であること: {value}")
    return value


def apply() -> list[str]:
    """monkeypatch を適用する（冪等）。適用したものの名前を返す。"""
    if _APPLIED:
        return list(_APPLIED)

    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy

    from .optim_vlm_lr import ParcAdamWVlmScaledConfig

    # --- P1: named_parameters() を optimizer まで届ける -----------------------
    def get_optim_params(self) -> dict:
        return dict(self.named_parameters())

    get_optim_params.__doc__ = (
        "lerobot_policy_parc の差し替え。param group を名前で分けるために "
        "named_parameters() の dict を返す（上流は self.parameters()）。"
    )
    InternVLAA15Policy.get_optim_params = get_optim_params
    _APPLIED.append("InternVLAA15Policy.get_optim_params -> dict(named_parameters())")

    # --- P2: optimizer preset を VLM 分割版に差し替える ------------------------
    def get_optimizer_preset(self) -> ParcAdamWVlmScaledConfig:
        return ParcAdamWVlmScaledConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            vlm_lr_scale=vlm_lr_scale(),
        )

    get_optimizer_preset.__doc__ = (
        "lerobot_policy_parc の差し替え。バックボーン VLM だけ lr を "
        f"{ENV_VLM_LR_SCALE}（既定 {DEFAULT_VLM_LR_SCALE}）倍にする。"
    )
    InternVLAA15Config.get_optimizer_preset = get_optimizer_preset
    _APPLIED.append(
        f"InternVLAA15Config.get_optimizer_preset -> ParcAdamWVlmScaledConfig"
        f"(vlm_lr_scale={vlm_lr_scale()})"
    )

    # 計画 §3 の例外規定どおり、適用したことを **必ず warning で** 出す。
    # 「上流をそのまま使っているつもり」で読むと事故るため、目立たせる。
    for name in _APPLIED:
        logger.warning("lerobot_policy_parc._monkeypatch applied: %s", name)
    return list(_APPLIED)


def applied() -> list[str]:
    return list(_APPLIED)
