"""データセットの ``meta/info.json`` の ``robot_type`` を書き換える（計画 P3-19）。

上流の libero launch script（``launch/internvla_a15_finetune_libero.sh``）は、4 つの
LIBERO サブセットに **スイート別の** robot_type（libero_10 / libero_goal / libero_object /
libero_spatial）を振る。理由は「サブセットごとの stats.json が混ざらないように」である。

**本レシピはこれを意図的に反転させ、単一の robot_type にする。**
配布データセット ``libero_combined_20hz`` は 4 スイートが 1 本に統合済みで stats.json も
1 セットしか無く、かつ**推論時にスイートを判別できない**（採点ハーネスはタスク名しか
渡さない）。robot_type をスイート別にすると、推論側でどの schema を引けばよいか決まらない。

冪等。書き換え前後を stdout に出す。既に目的の値なら何もしない。
``--restore`` で ``.bak`` から戻せる。

使い方::

    python scripts/prepare_dataset.py --root ~/data/libero_combined_20hz
    python scripts/prepare_dataset.py --root ~/data/libero_combined_20hz --check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

#: inspect_dataset.py が書く facts.json。robot_type_after をここが埋める（計画 SC-4）。
DEFAULT_FACTS = Path(__file__).resolve().parent.parent / "artifacts" / "facts" / "facts.json"

DEFAULT_ROBOT_TYPE = "libero_combined"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(root: Path, robot_type: str, *, check_only: bool = False) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"info.json が無い: {info_path}")

    info = _load(info_path)
    before = info.get("robot_type")

    # 実データの feature 名を確認する。schema の image_mapping の源キーはこれと
    # 一致していなければならない（不一致なら RemapImageKeyTransformFn が KeyError）。
    image_features = sorted(
        key for key, value in info.get("features", {}).items()
        if str(value.get("dtype", "")) in ("video", "image")
    )

    result = {
        "info_json": str(info_path),
        "robot_type_before": before,
        "robot_type_after": before,
        "changed": False,
        "image_features": image_features,
    }

    if before == robot_type:
        print(f"[prepare-dataset] robot_type は既に {robot_type!r}（変更なし）")
        return result

    if check_only:
        print(f"[prepare-dataset] --check: robot_type は {before!r}（{robot_type!r} であるべき）")
        result["robot_type_after"] = before
        return result

    backup = info_path.with_suffix(".json.parc-bak")
    if not backup.exists():
        shutil.copy2(info_path, backup)
        print(f"[prepare-dataset] 元ファイルを退避: {backup}")

    info["robot_type"] = robot_type
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[prepare-dataset] robot_type: {before!r} -> {robot_type!r}  ({info_path})")
    result["robot_type_after"] = robot_type
    result["changed"] = True
    return result


def update_facts(facts_path: Path, before: str | None, after: str | None) -> bool:
    """facts.json の robot_type_before / robot_type_after を埋める（計画 SC-4）。

    ``inspect_dataset.py`` を書き換え後に実行すると ``robot_type_before`` にも
    書き換え後の値が入ってしまうので、**退避ファイルにある元の値**を正とする。
    """
    if not facts_path.is_file():
        return False
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    if before is not None:
        facts["robot_type_before"] = before
    if after is not None:
        facts["robot_type_after"] = after
    facts_path.write_text(
        json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[prepare-dataset] facts.json を更新: {facts_path} "
          f"(before={before!r} after={after!r})")
    return True


def original_robot_type(root: Path) -> str | None:
    """退避ファイルから書き換え前の robot_type を読む。"""
    backup = root / "meta" / "info.json.parc-bak"
    if not backup.is_file():
        return None
    return json.loads(backup.read_text(encoding="utf-8")).get("robot_type")


def restore(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    backup = info_path.with_suffix(".json.parc-bak")
    if not backup.is_file():
        raise FileNotFoundError(f"退避ファイルが無い: {backup}")
    shutil.copy2(backup, info_path)
    print(f"[prepare-dataset] {backup} から復元した")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="データセットのルート")
    parser.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE)
    parser.add_argument("--check", action="store_true", help="書き換えず状態だけ見る")
    parser.add_argument("--restore", action="store_true", help=".parc-bak から復元する")
    parser.add_argument("--facts", default=str(DEFAULT_FACTS),
                        help="更新する facts.json（無ければ何もしない）")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if args.restore:
        restore(root)
        return 0

    result = prepare(root, args.robot_type, check_only=args.check)

    # 書き換え前の値は退避ファイルが正（2 回目以降は info.json が既に書き換わっている）。
    original = original_robot_type(root) or result["robot_type_before"]
    result["robot_type_before"] = original
    if not args.check:
        update_facts(Path(args.facts).expanduser(), original, args.robot_type)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # schema の源キーと実データの feature 名が食い違っていないかをここでも見ておく。
    expected = {"observation.images.front", "observation.images.wrist"}
    actual = set(result["image_features"])
    if actual != expected:
        print(
            f"[prepare-dataset] 警告: 画像 feature が想定と違う\n"
            f"  想定: {sorted(expected)}\n"
            f"  実際: {sorted(actual)}\n"
            f"  lerobot_policy_parc/schemas/libero_combined.yaml の image_mapping を"
            f"合わせること（不一致だと学習開始時に KeyError）。",
            file=sys.stderr,
        )
        return 2
    if args.check and result["robot_type_before"] != args.robot_type:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
