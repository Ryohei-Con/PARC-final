"""学習 dataloader と推論前処理の突き合わせ（計画 P3-21 / P5-31）。

**F6 の回帰テスト。** 上流の推論バックエンドは画像をリサイズしていないので、学習は
224、推論は生解像度のまま Qwen の smart_resize に入り、``image_grid_thw``（視覚
トークン数）が別物になる。同一の生フレームを学習経路と推論経路に流して
``image_grid_thw`` が一致することを確認する。

2 つのモードがある:

``--mode mini``
    **手元で動く縮小版**（計画 §7.1 の最終項目）。Qwen の processor ファイル
    （22MB、重み不要）だけあればよい。合成 256x256 画像を
    「学習チェーン相当」= RenderDownsample(128) -> resize_with_pad(224) と
    「推論チェーン相当」= env 128 -> resize_with_pad(224) に流し、
    ``image_grid_thw`` の一致を確認する。

``--mode full``
    実データセット + 実チェックポイント（クラウド）。``make_dataset`` で 1 サンプル
    作り、``pixel_values`` / ``image_grid_thw`` / ``input_ids`` の長さ /
    ``observation.video_frames.shape`` / ``action.shape`` を記録する。

使い方::

    python parity_check.py --mode mini --vlm-dir ~/hf/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _synthetic_frame(size: int, seed: int = 0) -> np.ndarray:
    """向きと周波数成分のある合成画像（HWC uint8）。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    base = 0.35 * yy + 0.25 * xx
    stripes = 0.25 * np.sin(2.0 * np.pi * 16.0 * xx)  # 高周波（エイリアス確認用）
    blob = 0.4 * np.exp(-(((yy - 0.3) ** 2 + (xx - 0.7) ** 2) / 0.01))
    gray = np.clip(base + stripes + blob + 0.02 * rng.standard_normal((size, size)), 0.0, 1.0)
    return (np.stack([gray, gray * 0.8, gray * 0.6], axis=-1) * 255.0).astype(np.uint8)


def _to_chw_float01(image: np.ndarray):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image))
    tensor = tensor.float() / 255.0 if tensor.dtype == torch.uint8 else tensor.float()
    return tensor.permute(2, 0, 1).contiguous()


def run_mini(args: argparse.Namespace) -> int:
    """手元で動く縮小版。Qwen の processor だけ使う（重み不要）。"""
    import torch
    from transformers import AutoProcessor

    from lerobot.transforms.utils import resize_with_pad

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    processor = AutoProcessor.from_pretrained(args.vlm_dir)

    raw_256 = _synthetic_frame(256, seed=0)
    raw_128 = _synthetic_frame(128, seed=0)

    # --- 学習チェーン相当: 256 -> RenderDownsample -> 128 -> resize_with_pad -> 224 ---
    train_fn = RenderDownsampleFn(keys=["img"])
    train_data = {"img": _to_chw_float01(raw_256)}
    train_image = train_fn(train_data)["img"]
    train_image = resize_with_pad(train_image, 224, 224)

    # --- 推論チェーン相当: env の native 128 -> resize_with_pad -> 224 ---
    infer_image = resize_with_pad(_to_chw_float01(raw_128), 224, 224)

    result = {}
    for label, image in (("train", train_image), ("infer", infer_image)):
        inputs = processor(
            text=["<image>"] if args.use_image_token else ["describe"],
            images=[image],
            do_rescale=False,
            return_tensors="pt",
        )
        grid = inputs["image_grid_thw"]
        result[label] = {
            "image_shape": list(image.shape),
            "image_grid_thw": grid.tolist(),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
        }

    same = result["train"]["image_grid_thw"] == result["infer"]["image_grid_thw"]
    result["image_grid_thw_match"] = same
    print(json.dumps(result, indent=2))

    if not same:
        print(
            "FAIL: image_grid_thw differs between the training and inference chains. "
            "This is exactly the F6 failure mode."
        )
        return 1

    # 参考: リサイズを通さなかった場合との比較（F6 を再現して見せる）
    no_resize = processor(
        text=["describe"],
        images=[_to_chw_float01(raw_128)],
        do_rescale=False,
        return_tensors="pt",
    )["image_grid_thw"].tolist()
    print(
        f"[reference] image_grid_thw WITHOUT resize_with_pad = {no_resize} "
        f"(upstream inference backend does this -- F6)"
    )
    _ = torch  # noqa: B018 - 明示的に import を使う
    print("PASS")
    return 0


def run_full(args: argparse.Namespace) -> int:
    """実データセットで 1 サンプル作り、shape を記録する（クラウド）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import lerobot_policy_parc as overlay

    overlay.assert_installed()

    from lerobot.datasets.factory import make_dataset

    dataset = make_dataset(args.train_config)
    sample = dataset[0]

    report = {
        "keys": sorted(sample.keys()),
        "shapes": {
            key: list(getattr(value, "shape", []))
            for key, value in sample.items()
            if hasattr(value, "shape")
        },
    }
    print(json.dumps(report, indent=2))

    expectations = {
        "observation.video_frames": [5, 3, 224, 224],
        "action": [50, 32],
    }
    problems = []
    for key, expected in expectations.items():
        actual = report["shapes"].get(key)
        if actual != expected:
            problems.append(f"{key}: got {actual}, expected {expected}")
    if problems:
        print("FAIL: " + "; ".join(problems))
        return 1
    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mini", "full"], default="mini")
    parser.add_argument("--vlm-dir", default=None, help="Qwen3.5-2B の processor ディレクトリ")
    parser.add_argument("--train-config", default=None, help="full モードの train_config.json")
    parser.add_argument("--use-image-token", action="store_true")
    args = parser.parse_args()

    if args.mode == "mini":
        if not args.vlm_dir:
            parser.error("--vlm-dir is required for --mode mini")
        return run_mini(args)
    if not args.train_config:
        parser.error("--train-config is required for --mode full")
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
