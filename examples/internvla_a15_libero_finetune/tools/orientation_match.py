"""dataset 画像と env 画像の向きの食い違いを推定する（計画 P2-17 / U2）。

学習データと採点 env で画像の上下左右が違っていても**エラーは出ない**。静かに
精度だけ落ちる（上下逆でも「動くが掴めない」）。4 変換 {none, flip_ud, flip_lr,
rot180} それぞれについて、行平均・列平均の輝度プロファイルの相関を計算し argmax を
候補として出す。

**必ず目視でも確認すること。** テーブル面・アーム・グリッパの位置関係が一致するか。
カメラごとに独立に決める。結果は ``runtime_config.json`` の ``image_orientation`` に
焼く（コードに直書きしない）。

依存は numpy だけ。cv2 も PIL も要らない（解像度差はプロファイルの線形補間で吸収する）。

使い方::

    python orientation_match.py --dataset-png artifacts/facts/dataset_agentview_00.png \
                                --env-png     artifacts/facts/env_agentview_00.png \
                                --camera agentview
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import numpy as np

#: 候補の変換。``env`` 画像に掛けて ``dataset`` 画像に合わせる向き。
TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda a: a,
    "flip_ud": lambda a: a[::-1, ...],
    "flip_lr": lambda a: a[:, ::-1, ...],
    "rot180": lambda a: a[::-1, ::-1, ...],
}


def to_gray(image: np.ndarray) -> np.ndarray:
    """HWC (または HW) を float32 のグレースケール HW にする。"""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        if arr.shape[2] < 3:
            gray = arr[..., 0].astype(np.float32)
        else:
            # Rec.601 luma
            gray = (
                0.299 * arr[..., 0].astype(np.float32)
                + 0.587 * arr[..., 1].astype(np.float32)
                + 0.114 * arr[..., 2].astype(np.float32)
            )
    else:
        raise ValueError(f"expected HW or HWC image, got shape {arr.shape}")
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def profiles(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """行平均輝度プロファイルと列平均輝度プロファイルを返す。"""
    gray = to_gray(image)
    return gray.mean(axis=1).astype(np.float32), gray.mean(axis=0).astype(np.float32)


def _resample(profile: np.ndarray, length: int) -> np.ndarray:
    """プロファイルを長さ ``length`` に線形補間する（128 と 256 を比べるため）。"""
    profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    if profile.shape[0] == length:
        return profile
    src = np.linspace(0.0, 1.0, profile.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.interp(dst, src, profile).astype(np.float32)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson 相関。分散が 0 の場合は 0.0 を返す。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def score_transform(dataset_image: np.ndarray, env_image: np.ndarray, mode: str) -> float:
    """``mode`` を env 画像に掛けたときの、行/列プロファイル相関の平均。"""
    if mode not in TRANSFORMS:
        raise ValueError(f"unknown transform {mode!r}; expected one of {sorted(TRANSFORMS)}")
    transformed = TRANSFORMS[mode](np.asarray(env_image))

    ref_rows, ref_cols = profiles(dataset_image)
    cand_rows, cand_cols = profiles(transformed)

    row_score = correlation(ref_rows, _resample(cand_rows, ref_rows.shape[0]))
    col_score = correlation(ref_cols, _resample(cand_cols, ref_cols.shape[0]))
    return 0.5 * (row_score + col_score)


def match_orientation(dataset_image: np.ndarray, env_image: np.ndarray) -> dict:
    """4 変換のスコアと argmax を返す。

    Returns:
        ``{"scores": {mode: float}, "best": mode, "margin": float}``。
        ``margin`` は 1 位と 2 位の差。小さいときは判定が曖昧なので目視が必須。
    """
    scores = {mode: score_transform(dataset_image, env_image, mode) for mode in TRANSFORMS}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "scores": scores,
        "best": ordered[0][0],
        "margin": float(ordered[0][1] - ordered[1][1]),
    }


def _load_png(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-png", required=True, help="データセット側のフレーム PNG")
    parser.add_argument("--env-png", required=True, help="採点 env 側の生フレーム PNG")
    parser.add_argument("--camera", default="agentview", choices=["agentview", "wrist"])
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = match_orientation(_load_png(args.dataset_png), _load_png(args.env_png))
    result["camera"] = args.camera
    result["dataset_png"] = args.dataset_png
    result["env_png"] = args.env_png

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["margin"] < 0.05:
        print(
            "WARNING: the top-2 margin is small; the profile test is inconclusive. "
            "Decide by eye (table surface / arm / gripper) before writing runtime_config.json.",
        )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
