"""256 -> 128 レンダリング解像度のダウンサンプル（カーネルランダム化）。

採点環境は LIBERO を native 128x128 でレンダリングする（計画 F11）。学習データ
``libero_combined_20hz`` は 256x256 で保存されているので、そのまま
``ResizeImagesWithPadFn(224, 224)`` に通すと「256 -> 224 の縮小」になり、推論時の
「128 -> 224 の拡大」と別物の画像統計になる。

そこで学習側で 256 -> 128 を挟む。ただし採点環境が実際にどのカーネルで 128 を
作っているかは（データ作成側のパイプラインも含めて）確定できないので、カーネルを
ランダムに振って頑健にする。

実装上の決めごと（計画 §5.1）:

- **カーネルはサンプル単位で 1 回だけ引く。** 入力は ``[T, C, H, W]``（T=5、
  ``image_delta_indices = [0, 12, 25, 37, 50]``、計画 F8）で来るので、フレームごとに
  引くと WAN の教師動画がちらついて video loss に偽の時間変化が入る。
  ``same_kernel_across_views=True``（既定）なら 2 カメラも同じカーネルを使う。
- ``box`` は ``F.avg_pool2d(x, k)`` で実装する。倍率がちょうど整数のとき box 平均は
  数学的に厳密で、``cv2.INTER_AREA`` とも ``F.interpolate(bilinear, antialias=False)``
  とも一致する（``tests/test_render_downsample.py`` で回帰テスト化）。
- ``nearest`` は ``x[..., oy::k, ox::k]`` のスライス。``oy`` / ``ox`` は独立に引く
  （半ピクセルの位相ジッタ）。
- ``cubic`` は overshoot するので ``.clamp_(0, 1)`` を必ず入れる。
- **``resize_with_pad`` を使ってはならない**（計画 F15）。``resize_with_pad`` は
  ``mode="bilinear"`` / ``antialias`` 指定なしで固定されており、2 倍縮小では box 平均に
  潰れてカーネルのバリエーションが消える。128 -> 224 の側は既存の
  ``ResizeImagesWithPadFn`` にそのまま任せる。
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F

from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES

logger = logging.getLogger(__name__)

#: サポートするカーネル名。抽選の順序でもある（RNG ストリームの再現性のため固定）。
KERNELS: tuple[str, ...] = ("box", "triangle", "cubic", "nearest")


def _is_image_key(key: str) -> bool:
    """上流 ``ResizeVQAImagesWithPadFn`` と同じ判定（``transforms/core.py:583-586``）。"""
    if "is_pad" in key or key.endswith("_mask"):
        return False
    return key.startswith(OBS_IMAGES) or key == OBS_IMAGE or "image" in key


@DataTransformFn.register_subclass("render_downsample")
@dataclass
class RenderDownsampleFn(DataTransformFn):
    """画像を ``target_h x target_w`` へランダムなカーネルでダウンサンプルする。

    Attributes:
        target_h / target_w: 出力解像度。既定 128（採点環境の native 解像度）。
        p_box: 2x2 box 平均（= INTER_AREA = antialias=False bilinear）の確率。
        p_triangle: 三角カーネル（antialias=True bilinear = PIL BILINEAR）の確率。
        p_cubic: bicubic antialias=True + clamp(0,1) の確率。
        p_nearest: 位相ジッタ付き点サンプルの確率。**ユーザー指定で 0.10 固定**。
        nearest_phase_jitter: nearest のときに開始画素を {0,1}^2 から引くか。
        same_kernel_across_views: 全カメラで同じカーネルを使うか。
        enabled: False なら恒等変換。
        keys: 対象キー。``hydrate()`` が schema の ``image_mapping.keys()`` を入れる。
        auto_detect_keys: ``keys`` が空のときにデータ側からキーを推定するか。
            **False にすると未 hydrate 時に黙って no-op になる**（計画 F6 と同じ
            サイレント失敗）ので既定は True。
    """

    target_h: int = 128
    target_w: int = 128
    p_box: float = 0.45
    p_triangle: float = 0.30
    p_cubic: float = 0.15
    p_nearest: float = 0.10
    nearest_phase_jitter: bool = True
    same_kernel_across_views: bool = True
    enabled: bool = True
    keys: list[str] = field(default_factory=list)
    auto_detect_keys: bool = True

    def __post_init__(self) -> None:
        probs = self.probabilities()
        for name, p in probs.items():
            if p < 0.0:
                raise ValueError(f"RenderDownsampleFn: p_{name}={p} must be >= 0")
        total = sum(probs.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"RenderDownsampleFn: kernel probabilities must sum to 1.0, got {total} ({probs})"
            )
        if self.target_h <= 0 or self.target_w <= 0:
            raise ValueError(
                f"RenderDownsampleFn: bad target size {self.target_h}x{self.target_w}"
            )

        # dataclass の field ではない内部状態（draccus のシリアライズ対象外）。
        self._rng: random.Random | None = None
        self._rng_worker: int | None = None
        #: 直近の抽選結果 ``(kernel, (oy, ox))``。テストとデバッグ用。
        self._last_draw: tuple[str, tuple[int, int]] | None = None
        self._warned_auto_detect = False

    # ------------------------------------------------------------------ #
    # 設定
    # ------------------------------------------------------------------ #
    def probabilities(self) -> dict[str, float]:
        return {
            "box": float(self.p_box),
            "triangle": float(self.p_triangle),
            "cubic": float(self.p_cubic),
            "nearest": float(self.p_nearest),
        }

    def hydrate(self, dataset) -> RenderDownsampleFn:
        """dataset の robot_type から対象画像キーを解決する。

        上流 ``ResizeImagesWithPadFn.hydrate`` と**同じ mapping**（``image_mapping`` の
        キー側 = データセット上のキー名）を使う。``RemapImageKeyTransformFn`` より
        手前に挿入されるため、image0/image1 ではなく元のキー名になる。
        """
        from lerobot.dataset_schemas import get_schema

        robot_type = dataset.meta.robot_type
        schema = get_schema(robot_type)
        keys = list(schema.image_mapping.keys())
        if not keys:
            raise RuntimeError(
                f"RenderDownsampleFn.hydrate: schema '{robot_type}' has an empty image_mapping. "
                "Downsampling would silently become a no-op."
            )
        return replace(self, keys=keys)

    # ------------------------------------------------------------------ #
    # RNG
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict:
        # DataLoader worker へ送るときに RNG を持ち越さない
        # （全 worker が同じ列を引く事故を防ぐ）。
        state = dict(self.__dict__)
        for key in ("_rng", "_rng_worker", "_last_draw", "_warned_auto_detect"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._rng = None
        self._rng_worker = None
        self._last_draw = None
        self._warned_auto_detect = False

    def _worker_rng(self) -> random.Random:
        """worker ごとに 1 個だけ ``random.Random`` を作りキャッシュする。"""
        worker_id = 0
        info = torch.utils.data.get_worker_info()
        if info is not None:
            worker_id = int(info.id)

        if self._rng is None or self._rng_worker != worker_id:
            # torch.initial_seed() は worker 内では base_seed + worker_id を返すが、
            # 実装に依存しないよう worker_id を明示的に混ぜる。
            seed = (int(torch.initial_seed()) ^ (worker_id * 0x9E3779B1)) & 0xFFFF_FFFF_FFFF_FFFF
            self._rng = random.Random(seed)
            self._rng_worker = worker_id
        return self._rng

    def _draw(self, rng: random.Random) -> tuple[str, tuple[int, int]]:
        """カーネルと位相を 1 回引く。"""
        probs = self.probabilities()
        r = rng.random()
        cumulative = 0.0
        kernel = KERNELS[-1]
        for name in KERNELS:
            cumulative += probs[name]
            if r < cumulative:
                kernel = name
                break

        if kernel == "nearest" and self.nearest_phase_jitter:
            phase = (rng.randint(0, 1), rng.randint(0, 1))
        else:
            phase = (0, 0)
        return kernel, phase

    # ------------------------------------------------------------------ #
    # 適用
    # ------------------------------------------------------------------ #
    def _resolve_keys(self, data: DataDict) -> list[str]:
        if self.keys:
            resolved = [k for k in self.keys if k in data]
            if not resolved:
                # hydrate 済みなのに 1 つも当たらない = schema の image_mapping と
                # サンプルのキーが食い違っている。これは計画 F6 の失敗形そのもの
                # （上流の推論バックエンドが no-op になっていたのと同じ理由）なので、
                # 黙って素通りさせず落とす。auto_detect_keys はここを覆わない
                # （keys が空のときしか作動しない）。
                raise KeyError(
                    "RenderDownsampleFn: none of the configured keys "
                    f"{list(self.keys)} are present in the sample "
                    f"(sample keys: {sorted(data)[:20]}). "
                    "Downsampling would silently become a no-op."
                )
            return resolved
        if not self.auto_detect_keys:
            return []
        detected = [k for k in data if _is_image_key(k) and isinstance(data[k], torch.Tensor)]
        if detected and not self._warned_auto_detect:
            self._warned_auto_detect = True
            logger.warning(
                "RenderDownsampleFn was not hydrated; falling back to key auto-detection %s. "
                "Prefer hydrate(dataset) so the schema decides which images to downsample.",
                detected,
            )
        return detected

    def _integer_step(self, h: int, w: int) -> int | None:
        """``h/target_h`` と ``w/target_w`` が同じ整数倍なら、その倍率を返す。"""
        if h % self.target_h != 0 or w % self.target_w != 0:
            return None
        ky = h // self.target_h
        kx = w // self.target_w
        if ky != kx or ky < 2:
            return None
        return ky

    def _apply(self, x: torch.Tensor, kernel: str, phase: tuple[int, int]) -> torch.Tensor:
        """``x``: ``[C, H, W]`` または ``[T, C, H, W]``。出力は同じ次元数。"""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"RenderDownsampleFn expects torch.Tensor, got {type(x)!r}")
        if x.ndim not in (3, 4):
            raise ValueError(
                f"RenderDownsampleFn expects [C,H,W] or [T,C,H,W], got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise TypeError(
                "RenderDownsampleFn expects float images in [0, 1]; "
                f"got dtype={x.dtype}. Convert before this transform."
            )

        h, w = int(x.shape[-2]), int(x.shape[-1])
        # 既に目標以下なら何もしない（データが最初から 128 でも壊れない）。
        if h <= self.target_h and w <= self.target_w:
            return x

        squeeze_back = x.ndim == 3
        x4 = x.unsqueeze(0) if squeeze_back else x

        exact_k = self._integer_step(h, w)

        if kernel == "box":
            if exact_k is not None:
                out = F.avg_pool2d(x4, exact_k)
            else:
                out = F.interpolate(
                    x4,
                    size=(self.target_h, self.target_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
        elif kernel == "triangle":
            out = F.interpolate(
                x4,
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        elif kernel == "cubic":
            out = F.interpolate(
                x4,
                size=(self.target_h, self.target_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            out = out.clamp_(0.0, 1.0)  # bicubic は overshoot する
        elif kernel == "nearest":
            if exact_k is not None:
                oy = min(int(phase[0]), exact_k - 1)
                ox = min(int(phase[1]), exact_k - 1)
                out = x4[..., oy::exact_k, ox::exact_k]
            else:
                out = F.interpolate(
                    x4, size=(self.target_h, self.target_w), mode="nearest-exact"
                )
        else:
            raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")

        if tuple(out.shape[-2:]) != (self.target_h, self.target_w):
            raise RuntimeError(
                f"RenderDownsampleFn produced {tuple(out.shape[-2:])}, "
                f"expected {(self.target_h, self.target_w)} (kernel={kernel})"
            )
        return out.squeeze(0) if squeeze_back else out

    def __call__(self, data: DataDict) -> DataDict:
        if not self.enabled:
            return data

        keys = self._resolve_keys(data)
        if not keys:
            return data

        rng = self._worker_rng()
        # ---- サンプル単位で 1 回だけ抽選する（計画 F8）----
        kernel, phase = self._draw(rng)
        self._last_draw = (kernel, phase)

        for index, key in enumerate(keys):
            if index > 0 and not self.same_kernel_across_views:
                kernel, phase = self._draw(rng)
            data[key] = self._apply(data[key], kernel, phase)
        return data
