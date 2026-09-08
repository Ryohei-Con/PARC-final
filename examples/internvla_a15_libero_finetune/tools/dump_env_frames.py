"""採点 env を dry-run して**生**フレームを PNG 保存する（計画 P2-17）。

``pipeline/rollout.py:254-266`` の ``_capture_frame`` は保存の直前に ``[::-1]`` を
掛けている。向きの判定に使いたいのは**その手前の生 obs** なので、ここでは
``env.reset()`` / ``env.step()`` が返す辞書からそのまま取り出す。

実行はクラウド（LIBERO が入った環境）。手元では import できない。

使い方::

    python dump_env_frames.py --track track1 --n-frames 8 --out artifacts/facts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: 採点環境の観測キー（pipeline/config.py:52-54、LIBERO_EVAL_CAMERA 既定 128）
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


def _save_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    image = np.asarray(array)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="track1")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10, help="何 env ステップおきに保存するか")
    parser.add_argument("--out", default="artifacts/facts")
    args = parser.parse_args()

    # PARC-final のリポジトリルートを sys.path に入れる（pipeline を import するため）
    parc_root = Path(__file__).resolve().parents[3]
    if str(parc_root) not in sys.path:
        sys.path.insert(0, str(parc_root))

    from pipeline.config import EvalConfig, Track
    from pipeline.environment import EnvironmentManager

    config = EvalConfig()
    manager = EnvironmentManager(config)
    tasks = manager.load_tasks(Track(args.track))
    if not tasks:
        raise RuntimeError(f"no tasks for track={args.track}")
    task_info = tasks[args.task_index]

    env = manager.create_env(task_info)
    out_dir = Path(args.out)
    metadata: dict = {
        "track": args.track,
        "task": task_info.name,
        "camera_height": config.camera_height,
        "camera_width": config.camera_width,
        "note": (
            "raw obs frames, captured BEFORE pipeline/rollout.py::_capture_frame applies [::-1]"
        ),
        "frames": [],
    }

    try:
        obs = env.reset()
        saved = 0
        step = 0
        while saved < args.n_frames:
            if step % args.stride == 0:
                for camera in CAMERAS:
                    if camera not in obs:
                        raise KeyError(f"env obs has no {camera!r}; got {sorted(obs)}")
                    array = np.asarray(obs[camera])
                    path = out_dir / f"env_{camera}_{saved:02d}.png"
                    _save_png(array, path)
                    metadata["frames"].append(
                        {
                            "camera": camera,
                            "path": str(path),
                            "shape": list(array.shape),
                            "dtype": str(array.dtype),
                            "min": int(array.min()),
                            "max": int(array.max()),
                            "step": step,
                        }
                    )
                saved += 1
            obs, _reward, done, _info = env.step(np.zeros(7, dtype=np.float32))
            step += 1
            if done:
                obs = env.reset()
    finally:
        env.close()

    meta_path = out_dir / "env_frames.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
