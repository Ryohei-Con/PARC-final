"""バックボーン VLM だけ学習率を下げる optimizer config（overlay）。

**なぜ上流の ``xvla-adamw`` を使わないのか。**
上流にも同趣旨の :class:`XVLAAdamWConfig`（``optim/optimizers.py:107``）があり、
docstring どおり「VLM を lr*0.1」にする。しかしその振り分けは::

    if "vlm" in name.lower():

であり、**InternVLA-A1.5 のパラメータ名には ``vlm`` という語が 1 つも現れない**。
実際の階層は::

    InternVLAA15Policy.model                       -> InternVLAA15                (modeling:539)
      .qwen3_5_with_expert                         -> InternVLAA15WithExpertModel (modeling:550)
        .qwen3_5                                   -> Qwen3_5ForConditionalGeneration（= VLM 本体、modeling:372）
        .action_expert                             -> Qwen3_5TextModel（= action expert、modeling:409）

つまり VLM のパラメータ名は ``model.qwen3_5_with_expert.qwen3_5.*`` である。
``xvla-adamw`` をそのまま使うと VLM グループが**空**になり、全パラメータが
``other``（フル lr）に落ちる。**例外も警告も出ないまま「0.1 倍したつもり」で
学習が進む**ので、移植手順書 §6 のサイレント失敗そのものになる。

そこで本モジュールは:

1. 振り分けを**名前の前方一致**で行う（``vlm_param_prefixes``）
2. **VLM グループと other グループが両方とも非空であることを assert する**。
   上流のリファクタで属性名が変われば、静かに無効化されるのではなく起動時に落ちる
3. 実際に何個のパラメータがどちらに入ったかを ``logging.info`` で必ず出す

``action_expert`` は ``qwen3_5_with_expert`` の下にあるが ``qwen3_5.`` 直下では
ないので、前方一致では VLM 側に混ざらない（``model.qwen3_5_with_expert.qwen3_5.``
という末尾のドットまで含めて一致を見る）。

スケジューラとの関係: 上流の :class:`CosineDecayWithWarmupSchedulerConfig` は
``LambdaLR`` を返す。``LambdaLR`` は各 param group の ``initial_lr`` に同じ係数を
掛けるので、**グループ間の比 0.1 は warmup / cosine decay を通して保たれる**
（peak も decay 後の下限も VLM 側だけ 0.1 倍になる）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from lerobot.optim.optimizers import OptimizerConfig

logger = logging.getLogger(__name__)

#: choice registry へ登録する名前。``train_config.json`` にこの名前で残る。
OPTIMIZER_CHOICE = "parc_adamw_vlm_scaled"

#: VLM 本体のパラメータ名プレフィックス。末尾のドットまで含めることで
#: ``qwen3_5_with_expert`` 自体や ``action_expert`` と混ざらないようにする。
DEFAULT_VLM_PREFIXES = ["model.qwen3_5_with_expert.qwen3_5."]


@OptimizerConfig.register_subclass(OPTIMIZER_CHOICE)
@dataclass
class ParcAdamWVlmScaledConfig(OptimizerConfig):
    """AdamW。バックボーン VLM のみ ``lr * vlm_lr_scale`` で学習する。

    Attributes:
        vlm_lr_scale: VLM グループの学習率倍率。**ユーザー指定で 0.1**。
        vlm_weight_decay_scale: VLM グループの weight decay 倍率。既定 1.0
            （= 変えない）。上流 ``xvla-adamw`` は 0.1 にしているが、今回の指示は
            学習率のみなので既定では触らない。A/B 用に残してある。
        vlm_param_prefixes: VLM と判定するパラメータ名の前方一致リスト。
    """

    lr: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0
    vlm_lr_scale: float = 0.1
    vlm_weight_decay_scale: float = 1.0
    vlm_param_prefixes: list[str] = field(default_factory=lambda: list(DEFAULT_VLM_PREFIXES))

    def build(self, params) -> torch.optim.Optimizer:
        """``named_parameters()`` の dict を受け取り、2 グループの AdamW を作る。"""
        if not isinstance(params, dict):
            raise TypeError(
                f"{OPTIMIZER_CHOICE} は named_parameters() の dict を要求する（受領: "
                f"{type(params).__name__}）。上流の InternVLAA15Policy.get_optim_params() は "
                "self.parameters() を返すので、lerobot_policy_parc._monkeypatch が適用されて "
                "いるか確認すること。"
            )

        prefixes = tuple(self.vlm_param_prefixes)
        vlm_params, other_params = [], []
        vlm_names, other_names = [], []
        for name, p in params.items():
            if not p.requires_grad:
                continue
            if name.startswith(prefixes):
                vlm_params.append(p)
                vlm_names.append(name)
            else:
                other_params.append(p)
                other_names.append(name)

        # ★ サイレント失敗ガード。片方が空なら「0.1 倍したつもり」で全パラメータが
        #   同じ lr で回る／VLM しか学習しない、のどちらかになる。起動時に落とす。
        if not vlm_params:
            raise RuntimeError(
                f"{OPTIMIZER_CHOICE}: VLM グループが空。prefixes={prefixes} に一致する "
                f"パラメータが無い（全 {len(params)} 個）。上流のモジュール階層が変わった "
                "可能性がある。名前の例: " + ", ".join(list(params)[:5])
            )
        if not other_params:
            raise RuntimeError(
                f"{OPTIMIZER_CHOICE}: VLM 以外のグループが空。prefixes={prefixes} が広すぎる。"
            )

        vlm_lr = self.lr * self.vlm_lr_scale
        # ★ group 0 を "other"（基準 lr）にしておく。lerobot_train.py:130 が
        #   `train_metrics.lr = optimizer.param_groups[0]["lr"]` でログ用の lr を取るので、
        #   vlm を先頭に置くとログと W&B の "lr" が 0.1 倍の値になり、学習率設定を
        #   読み違える。上流 xvla-adamw は vlm を先頭に置いているが、ここでは入れ替える。
        param_groups = [
            {
                "params": other_params,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "name": "other",
            },
            {
                "params": vlm_params,
                "lr": vlm_lr,
                "weight_decay": self.weight_decay * self.vlm_weight_decay_scale,
                "name": "vlm",
            },
        ]

        n_vlm = sum(p.numel() for p in vlm_params)
        n_other = sum(p.numel() for p in other_params)
        logger.info(
            "%s: vlm group %d tensors / %.1fM params @ lr=%.3g (= %.3g * %.3g) | "
            "other group %d tensors / %.1fM params @ lr=%.3g",
            OPTIMIZER_CHOICE,
            len(vlm_params), n_vlm / 1e6, vlm_lr, self.lr, self.vlm_lr_scale,
            len(other_params), n_other / 1e6, self.lr,
        )
        logger.info("%s: vlm 側の例 %s", OPTIMIZER_CHOICE, vlm_names[:2])
        logger.info("%s: other 側の例 %s", OPTIMIZER_CHOICE, other_names[:4])

        return torch.optim.AdamW(param_groups, betas=self.betas, eps=self.eps)
