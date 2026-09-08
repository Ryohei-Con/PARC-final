"""データセットの事実収集（計画 P2-16 / SC-4）。

**推測で進めない。** 学習と推論の整合を決める値をすべてデータから読み出し、
``facts.json`` に書く。ここが埋まらないと U1〜U7（グリッパ規約・画像の向き・
state の中身・解像度・stats のキー構造・fps・正規化モード）が確定しない。

出力必須項目（SC-4）:
  ``fps`` / ``image_hw`` / ``state_dim`` / ``state_layout`` / ``action_dim`` /
  ``gripper_convention`` / ``stats_keys`` / ``robot_type_before``。
  ``robot_type_after`` は ``prepare_dataset.py``（P3-19）が埋める。

実行はクラウド。使い方::

    python inspect_dataset.py --root ~/data/libero_combined_20hz \
                              --out ../artifacts/facts/facts.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

#: 未確定の番人。facts.json に残っていたら SC-4 は未達。
TBD = "TBD"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_gripper_convention(low: float, high: float) -> str:
    """action dim6 の min/max から規約を推定する（最終判断は人間が下す）。"""
    if low >= -0.05 and high <= 1.05:
        return "zero_one"
    if low >= -1.05 and high <= 1.05 and low < -0.05:
        return "minus_one_one"
    return TBD


def _stats_entry(stats: dict, key: str) -> dict | None:
    """stats.json が「スイート別 1 段」でも「フラット」でも entry を引く（U5）。"""
    if key in stats and isinstance(stats[key], dict) and "mean" in stats[key]:
        return stats[key]
    for value in stats.values():
        if isinstance(value, dict) and key in value:
            entry = value[key]
            if isinstance(entry, dict) and "mean" in entry:
                return entry
    return None


def collect(root: Path, n_samples: int) -> dict:
    info = _load_json(root / "meta" / "info.json")
    stats_path = root / "meta" / "stats.json"
    stats = _load_json(stats_path) if stats_path.is_file() else {}

    features = info.get("features", {})
    image_features = {
        key: value
        for key, value in features.items()
        if str(value.get("dtype", "")) in ("video", "image")
    }

    facts: dict = {
        "root": str(root),
        "fps": info.get("fps", TBD),
        "robot_type_before": info.get("robot_type", TBD),
        "robot_type_after": TBD,  # prepare_dataset.py が埋める
        "total_episodes": info.get("total_episodes", TBD),
        "total_frames": info.get("total_frames", TBD),
        "image_features": {
            key: {
                "shape": value.get("shape"),
                "names": value.get("names"),
                "dtype": value.get("dtype"),
            }
            for key, value in image_features.items()
        },
        "state_dim": TBD,
        "state_layout": TBD,
        "action_dim": TBD,
        "gripper_convention": TBD,
        "stats_keys": sorted(stats.keys()) if stats else [],
        "stats_num_keys": len(stats),
        "normalization_mode_hint": (
            "read from train_config.json after training starts (plan U7)"
        ),
    }

    image_hw = sorted(
        {
            tuple(value.get("shape", [])[-3:-1]) if len(value.get("shape", [])) >= 3 else None
            for value in image_features.values()
        }
        - {None}
    )
    facts["image_hw"] = [list(hw) for hw in image_hw] if image_hw else TBD

    state_feature = features.get("observation.state", {})
    if state_feature.get("shape"):
        facts["state_dim"] = int(state_feature["shape"][0])
        facts["state_layout"] = state_feature.get("names", TBD)

    action_feature = features.get("action", {})
    if action_feature.get("shape"):
        facts["action_dim"] = int(action_feature["shape"][0])

    action_stats = _stats_entry(stats, "action")
    if action_stats is not None:
        low = np.asarray(action_stats["min"], dtype=np.float64).reshape(-1)
        high = np.asarray(action_stats["max"], dtype=np.float64).reshape(-1)
        mean = np.asarray(action_stats["mean"], dtype=np.float64).reshape(-1)
        facts["action_stats"] = {
            "min": low.tolist(),
            "max": high.tolist(),
            "mean": mean.tolist(),
        }
        if low.shape[0] > 6:
            facts["gripper_convention"] = _infer_gripper_convention(
                float(low[6]), float(high[6])
            )
            facts["gripper_dim6"] = {
                "min": float(low[6]),
                "max": float(high[6]),
                "mean": float(mean[6]),
            }

    state_stats = _stats_entry(stats, "observation.state")
    if state_stats is not None:
        facts["state_stats"] = {
            "min": np.asarray(state_stats["min"], dtype=np.float64).reshape(-1).tolist(),
            "max": np.asarray(state_stats["max"], dtype=np.float64).reshape(-1).tolist(),
            "mean": np.asarray(state_stats["mean"], dtype=np.float64).reshape(-1).tolist(),
        }

    # ---- 実サンプルで dim6 のヒストグラムを取る（規約の裏取り）----
    facts["gripper_histogram"] = _gripper_histogram(root, n_samples)
    return facts


def _gripper_histogram(root: Path, n_samples: int) -> dict:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover
        return {"error": f"lerobot unavailable: {type(exc).__name__}: {exc}"}

    try:
        dataset = LeRobotDataset(repo_id=root.name, root=str(root))
    except Exception as exc:  # pragma: no cover
        return {"error": f"could not open dataset: {type(exc).__name__}: {exc}"}

    total = len(dataset)
    stride = max(1, total // max(1, n_samples))
    counter: Counter = Counter()
    values: list[float] = []
    for index in range(0, total, stride):
        action = np.asarray(dataset[index]["action"], dtype=np.float32)
        if action.ndim == 2:
            action = action[0]  # [chunk, D] -> 先頭ステップ
        if action.shape[-1] <= 6:
            raise ValueError(f"action has only {action.shape[-1]} dims; expected >= 7")
        gripper = float(action[6])
        values.append(gripper)
        counter[round(gripper, 3)] += 1
        if len(values) >= n_samples:
            break

    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.shape[0]),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "most_common": counter.most_common(10),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="データセットのルート")
    parser.add_argument("--out", default="artifacts/facts/facts.json")
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()

    facts = collect(Path(args.root).expanduser(), args.n_samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(facts, indent=2, ensure_ascii=False))

    remaining = [key for key, value in facts.items() if value == TBD]
    if remaining:
        print(f"\nSTILL TBD: {remaining}")
        print("SC-4 is not satisfied until every one of these is resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
