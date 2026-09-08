"""データセットから agentview / wrist のフレームを PNG 保存する（計画 P2-17）。

``dump_env_frames.py`` の出力と ``orientation_match.py`` で突き合わせ、
``runtime_config.json`` の ``image_orientation`` を確定する。

**transform を一切通さない生フレーム**を保存する。resize / normalize が入ると
向きの判定には影響しないが、目視の手がかり（縦横比・黒帯）が失われるため。

実行はクラウド（torchcodec でデコードできる環境）。

使い方::

    python dump_dataset_frames.py --root ~/data/libero_combined_20hz --n-frames 8 \
                                  --out artifacts/facts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: LIBERO データセット側の画像キー（上流 libero.yaml の image_mapping のキー側）
DATASET_CAMERAS = {
    "agentview": "observation.images.image",
    "wrist": "observation.images.wrist_image",
}


def _save_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    image = np.asarray(array)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[0] != image.shape[2]:
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None, help="HF repo id またはローカルパス")
    parser.add_argument("--root", required=True, help="データセットのルート")
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=97, help="何サンプルおきに保存するか")
    parser.add_argument("--out", default="artifacts/facts")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root).expanduser()
    dataset = LeRobotDataset(repo_id=args.repo_id or root.name, root=str(root))

    out_dir = Path(args.out)
    metadata: dict = {
        "root": str(root),
        "robot_type": dataset.meta.robot_type,
        "fps": dataset.meta.fps,
        "num_frames": len(dataset),
        "frames": [],
    }

    for index in range(args.n_frames):
        sample = dataset[(index * args.stride) % len(dataset)]
        for name, key in DATASET_CAMERAS.items():
            if key not in sample:
                raise KeyError(f"dataset sample has no {key!r}; got {sorted(sample)[:20]}")
            array = np.asarray(sample[key])
            if array.ndim == 4:  # [T, C, H, W] -> 先頭フレーム
                array = array[0]
            path = out_dir / f"dataset_{name}_{index:02d}.png"
            _save_png(array, path)
            metadata["frames"].append(
                {
                    "camera": name,
                    "key": key,
                    "path": str(path),
                    "shape": list(np.asarray(sample[key]).shape),
                    "dtype": str(np.asarray(sample[key]).dtype),
                    "min": float(np.asarray(sample[key]).min()),
                    "max": float(np.asarray(sample[key]).max()),
                }
            )

    meta_path = out_dir / "dataset_frames.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    # torchcodec の ABI 不一致で黒画像が出る事故（計画 R4）をここでも検出する。
    zero_frames = [f for f in metadata["frames"] if f["max"] == f["min"]]
    if zero_frames:
        raise RuntimeError(
            f"{len(zero_frames)} decoded frames are constant (likely a torchcodec ABI "
            f"mismatch producing black images): {zero_frames[:3]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
