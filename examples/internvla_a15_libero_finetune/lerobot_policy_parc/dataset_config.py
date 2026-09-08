"""PARC 2026 用の DatasetConfig（overlay）。

上流 ``InternVLAA15DatasetConfig`` / ``InternVLAA15VQADatasetConfig`` を継承し、
``RenderDownsampleFn`` を **``ResizeImagesWithPadFn`` の直前に 1 個だけ**挿入する
（計画 §5.1）。上流ファイルは 1 バイトも編集しない。

なぜ「直前」か:

  RenderDownsample(256 -> 128) -> ResizeImagesWithPad(128 -> 224)

  という順序でのみ、推論経路（env の native 128 -> resize_with_pad(224)）と
  同じ画像統計になる。逆順や resize の後に置くと 224 -> 128 -> 224 になって
  情報が二重に落ちる。

VQA チェーンについて（計画 §5.1 の注記 / D5）:
  現行の libero レシピは VQA データセットを与えていないので、
  ``InternVLAA15ParcVQADatasetConfig`` は **dead path** である。実装はするが、
  本走では効かない。合成データの単体テストまでが動作確認の範囲。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (
    InternVLAA15DatasetConfig,
    InternVLAA15VQADatasetConfig,
)
from lerobot.transforms.core import (
    DeltaActionTransformFn,
    ResizeImagesWithPadFn,
    ResizeVQAImagesWithPadFn,
)

from .transforms_render import RenderDownsampleFn

logger = logging.getLogger(__name__)


def _build_render_fn(cfg) -> RenderDownsampleFn:
    return RenderDownsampleFn(
        target_h=int(cfg.render_target),
        target_w=int(cfg.render_target),
        p_box=float(cfg.render_p_box),
        p_triangle=float(cfg.render_p_triangle),
        p_cubic=float(cfg.render_p_cubic),
        p_nearest=float(cfg.render_p_nearest),
    )


def _insert_before(
    inputs: list,
    new_transform,
    anchor_type: type,
    *,
    remove_type: type,
    label: str,
) -> list:
    """``anchor_type`` の最初の要素の直前に ``new_transform`` を 1 個だけ置く。

    冪等性のため、まず ``remove_type`` のインスタンスを全部取り除いてから挿入する。
    ``new_transform`` が None なら「取り除くだけ」。
    """
    cleaned = [t for t in inputs if not isinstance(t, remove_type)]

    anchors = [i for i, t in enumerate(cleaned) if isinstance(t, anchor_type)]
    if new_transform is None:
        return cleaned

    if len(anchors) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one {anchor_type.__name__} in data_transforms.inputs, "
            f"found {len(anchors)}. Transform chain: {[type(t).__name__ for t in cleaned]}"
        )

    cleaned.insert(anchors[0], new_transform)
    return cleaned


@DatasetConfig.register_subclass("internvla_a1_5_parc")
@dataclass
class InternVLAA15ParcDatasetConfig(InternVLAA15DatasetConfig):
    """robot データ用。``RenderDownsampleFn`` を挿入する以外は上流と同一。"""

    render_downsample: bool = True
    render_target: int = 128
    render_p_box: float = 0.45
    render_p_triangle: float = 0.30
    render_p_cubic: float = 0.15
    render_p_nearest: float = 0.10

    def __post_init__(self) -> None:
        super().__post_init__()

        inputs = list(self.data_transforms.inputs)
        render_fn = _build_render_fn(self) if self.render_downsample else None
        inputs = _insert_before(
            inputs,
            render_fn,
            ResizeImagesWithPadFn,
            remove_type=RenderDownsampleFn,
            label="InternVLAA15ParcDatasetConfig",
        )

        n_render = sum(isinstance(t, RenderDownsampleFn) for t in inputs)
        expected = 1 if self.render_downsample else 0
        if n_render != expected:
            raise RuntimeError(
                f"InternVLAA15ParcDatasetConfig: expected {expected} RenderDownsampleFn, got {n_render}"
            )

        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "abs" and has_delta:
            # 上流 __post_init__ で除去済みのはず。ここは保険。
            raise RuntimeError(
                "InternVLAA15ParcDatasetConfig: action_mode='abs' but DeltaActionTransformFn remains"
            )

        self.data_transforms = replace(self.data_transforms, inputs=inputs)
        logger.info(
            "InternVLAA15ParcDatasetConfig transform chain: %s",
            [type(t).__name__ for t in inputs],
        )


@VQADatasetConfig.register_subclass("internvla_a1_5_parc")
@dataclass
class InternVLAA15ParcVQADatasetConfig(InternVLAA15VQADatasetConfig):
    """VQA データ用。**現行レシピでは dead path**（計画 §5.1 注記 / D5）。"""

    render_downsample: bool = True
    render_target: int = 128
    render_p_box: float = 0.45
    render_p_triangle: float = 0.30
    render_p_cubic: float = 0.15
    render_p_nearest: float = 0.10

    def __post_init__(self) -> None:
        super().__post_init__()

        inputs = list(self.data_transforms.inputs)
        render_fn = _build_render_fn(self) if self.render_downsample else None
        inputs = _insert_before(
            inputs,
            render_fn,
            ResizeVQAImagesWithPadFn,
            remove_type=RenderDownsampleFn,
            label="InternVLAA15ParcVQADatasetConfig",
        )

        n_render = sum(isinstance(t, RenderDownsampleFn) for t in inputs)
        expected = 1 if self.render_downsample else 0
        if n_render != expected:
            raise RuntimeError(
                f"InternVLAA15ParcVQADatasetConfig: expected {expected} RenderDownsampleFn, got {n_render}"
            )

        self.data_transforms = replace(self.data_transforms, inputs=inputs)
