#!/usr/bin/env python
"""RoboInter-VQA の ``llava_format`` メタ (.json 配列) を ``VQADataset`` が読める
``.jsonl`` へ変換する（計画 §5.3 / PV1）。

上流 ``lerobot.datasets.vqa_dataset.VQADataset`` の jsonl スキーマ (V3):

  - ``image``      : str または str のリスト（相対 / 絶対パス）
  - ``conversations`` : LLaVA 形式（``from`` = human/gpt, ``value``）
  - ``source``     : 任意。ログ・混合比の可視化用
  - ``data_path``  : 任意。M1 形式互換の画像ルート

RoboInter-VQA ``llava_format`` レコードのキー (V11): ``id`` / ``task`` /
``conversations`` (from/value) / ``images`` (パスのリスト) / ``gt`` / ``h`` / ``w`` /
``new_h`` / ``new_w``。

この変換で行うこと:

  1. ``.json``（JSON 配列） → ``.jsonl``（1 行 1 オブジェクト）
  2. ``images``（リスト） → ``image``
  3. ``source`` を ``robointer_vqa/<category>`` で付与（連結後もカテゴリ別に残す）
  4. 画像パスを **出力 jsonl と同じディレクトリからの相対パス**へ正規化
     （``VQADataset._resolve_image_path`` の第 2 候補 ``jsonl_path.parent/image`` で
     解決させる。``root`` 不要 / V5）。``--pad-square`` 時は 256x256 の正方形へ
     再生成、それ以外は元画像へのシンボリックリンクを張る。
  5. ``--max-samples`` / ``--max-samples-per-cat`` で決定的サブセット（``--seed``）
  6. 画像が解決できない / ``conversations`` が空 のレコードはスキップし件数を stderr へ

``datasets`` ライブラリには依存しない（標準 ``json`` のみ）。冪等。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# 純関数（GPU 不要・ファイル I/O なしでテスト可能な部分）
# --------------------------------------------------------------------------- #

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def load_json_array(path: str | Path) -> list[dict[str, Any]]:
    """``llava_format`` の ``.json``（JSON 配列）を読む。配列でなければ ``ValueError``。"""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array in {path}, got {type(payload).__name__}")
    return payload


def _extract_image_list(rec: dict[str, Any]) -> list[str]:
    raw = rec.get("images", rec.get("image"))
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw if x is not None and str(x) != ""]
    else:
        return []
    return [x for x in items if x]


def _conversations_ok(rec: dict[str, Any]) -> bool:
    conv = rec.get("conversations")
    if not isinstance(conv, (list, tuple)) or len(conv) == 0:
        return False
    for turn in conv:
        if not isinstance(turn, dict):
            return False
        if "value" not in turn and "content" not in turn:
            return False
    return True


def llava_record_to_vqa(
    rec: dict[str, Any],
    *,
    category: str,
    image_paths: list[str] | None = None,
) -> dict[str, Any] | None:
    """1 レコードを ``VQADataset`` jsonl スキーマへ変換する。

    破損レコード（画像なし / ``conversations`` 空）は ``None`` を返す。

    ``image_paths`` を渡すとその値を ``image`` に入れる（呼び出し側でパス正規化済み）。
    渡さなければ元の ``images`` の値をそのまま使う。
    """
    if not _conversations_ok(rec):
        return None

    images = image_paths if image_paths is not None else _extract_image_list(rec)
    if not images:
        return None

    out: dict[str, Any] = {
        "image": images if len(images) > 1 else images[0],
        "conversations": list(rec["conversations"]),
        "source": f"robointer_vqa/{category}",
    }
    # 参考情報は残す（学習には使われないが後追い調査で有用）。
    for key in ("id", "task", "gt"):
        if key in rec:
            out[key] = rec[key]
    return out


def subsample(records: list[Any], max_samples: int | None, seed: int) -> list[Any]:
    """``max_samples`` 件を決定的に選ぶ（``VQADataset`` と同じ手法 / V1）。"""
    if not max_samples or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), max_samples))
    return [records[i] for i in indices]


def _rel_after_marker(raw: str) -> PurePosixPath:
    """``.../image/a/b.jpg`` -> ``a/b.jpg``。marker が無ければ basename だけ。"""
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    for marker in ("image", "images"):
        if marker in parts:
            idx = len(parts) - 1 - parts[::-1].index(marker)
            tail = parts[idx + 1 :]
            if tail:
                return PurePosixPath(*tail)
    return PurePosixPath(PurePosixPath(raw.replace("\\", "/")).name)


def pad_to_square(img, size: int = 256):
    """短辺基準でゼロ（黒）パディングして正方形にし、``size`` へリサイズする。

    ``RenderDownsampleFn`` の整数ステップ 2x 経路（256 -> 128）に載せ、robot データと
    同じ前処理へそろえる（計画 R3 / U3）。PIL ``Image`` を受け取り ``Image`` を返す。
    """
    from PIL import Image

    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    if side != size:
        canvas = canvas.resize((size, size), Image.LANCZOS)
    return canvas


# --------------------------------------------------------------------------- #
# 画像解決 / 書き出し
# --------------------------------------------------------------------------- #

def resolve_source_image(raw: str, src_dir: Path, category: str) -> Path | None:
    """展開済みツリーから 1 枚の元画像を探す。見つからなければ ``None``。"""
    rel = _rel_after_marker(raw)
    p = Path(raw.replace("\\", "/"))
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    candidates += [
        src_dir / raw,
        src_dir / "robotinter" / category / raw,
        src_dir / category / raw,
        src_dir / "robotinter" / category / "image" / rel,
        src_dir / category / "image" / rel,
        src_dir / "robotinter" / category / rel,
        src_dir / category / rel,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    # 最後の手段: basename でグロブ（遅いので 1 回だけ）
    hits = list((src_dir).rglob(rel.name))
    return hits[0] if len(hits) == 1 else None


@dataclass
class _CatResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    skipped_parse: int = 0
    skipped_image: int = 0


def _process_category(
    *,
    src_dir: Path,
    out_dir: Path,
    jsonl_dir: Path,
    category: str,
    pad_square: bool,
    max_samples: int | None,
    seed: int,
    force: bool,
) -> _CatResult:
    meta_files = _discover_meta_files(src_dir, category)
    if not meta_files:
        raise FileNotFoundError(
            f"no *llava*.json meta found for category {category!r} under {src_dir}"
        )

    raw_records: list[dict[str, Any]] = []
    for meta in meta_files:
        raw_records.extend(load_json_array(meta))

    result = _CatResult()
    # まず parse フェーズの破損を落としてからサブサンプルする（サブセットに穴を作らない）
    staged: list[tuple[dict[str, Any], list[str]]] = []
    for rec in raw_records:
        if not _conversations_ok(rec):
            result.skipped_parse += 1
            continue
        imgs = _extract_image_list(rec)
        if not imgs:
            result.skipped_parse += 1
            continue
        staged.append((rec, imgs))

    staged = subsample(staged, max_samples, seed)

    images_root = out_dir / "images" / category
    for rec, imgs in staged:
        rel_paths: list[str] = []
        ok = True
        for raw in imgs:
            src_img = resolve_source_image(raw, src_dir, category)
            if src_img is None:
                ok = False
                break
            rel = _rel_after_marker(raw)
            dest = images_root / rel
            if pad_square:
                if dest.suffix.lower() not in _IMAGE_SUFFIXES:
                    dest = dest.with_suffix(".png")
                elif dest.suffix.lower() in {".webp", ".tiff", ".tif", ".bmp"}:
                    dest = dest.with_suffix(".png")
                rel_out = dest.relative_to(out_dir)
                if force or not dest.exists():
                    _write_padded(src_img, dest)
            else:
                rel_out = dest.relative_to(out_dir)
                if force or not dest.exists():
                    _link_image(src_img, dest)
            # jsonl からの相対（jsonl は out_dir 直下 or merge 先。どちらも out_dir 直下）
            rel_paths.append(os.path.relpath(out_dir / rel_out, jsonl_dir))
        if not ok:
            result.skipped_image += 1
            continue
        converted = llava_record_to_vqa(rec, category=category, image_paths=rel_paths)
        if converted is None:
            result.skipped_image += 1
            continue
        result.records.append(converted)
    return result


def _discover_meta_files(src_dir: Path, category: str) -> list[Path]:
    seen: set[Path] = set()
    roots = [
        src_dir / "robotinter" / category / "meta",
        src_dir / category / "meta",
    ]
    for root in roots:
        if root.is_dir():
            for path in sorted(root.glob("*llava*.json")):
                seen.add(path.resolve())
    if not seen:
        for path in sorted(src_dir.rglob("*llava*.json")):
            if f"/{category}/" in str(path).replace("\\", "/"):
                seen.add(path.resolve())
    return sorted(seen)


def _write_padded(src_img: Path, dest: Path) -> None:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_img) as handle:
        out = pad_to_square(handle, size=256)
    if dest.suffix.lower() in {".jpg", ".jpeg"}:
        out.save(dest, quality=95)
    else:
        out.save(dest)


def _link_image(src_img: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(src_img.resolve(), dest)
    except OSError:
        import shutil

        shutil.copy2(src_img, dest)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


# --------------------------------------------------------------------------- #
# オーケストレーション
# --------------------------------------------------------------------------- #

def build_vqa_dataset(
    *,
    src: str | Path,
    out: str | Path,
    categories: list[str],
    max_samples: int | None = None,
    max_samples_per_cat: dict[str, int] | None = None,
    merge_into: str | None = None,
    pad_square: bool = True,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """全カテゴリを変換し jsonl を書き出す。サマリ dict を返す。冪等。"""
    src_dir = Path(src).expanduser()
    out_dir = Path(out).expanduser()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"--src not found: {src_dir}")

    # 出力ファイルパスを決める
    if merge_into:
        targets = {merge_into: categories}
    else:
        targets = {f"{cat.lower()}.jsonl": [cat] for cat in categories}

    # 冪等ガード: 全ターゲットが存在し **かつ 1 行以上** あり、--force でないなら検証のみ。
    # 0 行の jsonl（前回の中断で truncate された等）は再生成する。
    counts = {
        name: sum(1 for _ in (out_dir / name).open("r", encoding="utf-8"))
        for name in targets
        if (out_dir / name).is_file()
    }
    if not force and len(counts) == len(targets) and all(n > 0 for n in counts.values()):
        summary = {"status": "verified", "targets": {}}
        for name, n in counts.items():
            summary["targets"][str(out_dir / name)] = n
            print(f"[prepare-vqa] 既存: {out_dir / name} ({n} 行) -- 検証のみ", file=sys.stderr)
        return summary
    if counts and any(n == 0 for n in counts.values()):
        empties = [str(out_dir / name) for name, n in counts.items() if n == 0]
        print(f"[prepare-vqa] 0 行の出力を検出 -> 再生成する: {empties}", file=sys.stderr)

    # per-category の上限を決める（既定は max_samples の均等割り）
    per_cat: dict[str, int | None] = {}
    if max_samples_per_cat:
        for cat in categories:
            per_cat[cat] = max_samples_per_cat.get(cat)
    if max_samples:
        n_cats = len(categories)
        base = max_samples // n_cats
        rem = max_samples % n_cats
        for i, cat in enumerate(categories):
            if per_cat.get(cat) is None:
                per_cat[cat] = base + (1 if i < rem else 0)
    for cat in categories:
        per_cat.setdefault(cat, None)

    cat_results: dict[str, _CatResult] = {}
    for cat in categories:
        jsonl_dir = out_dir  # jsonl は必ず out_dir 直下
        cat_results[cat] = _process_category(
            src_dir=src_dir,
            out_dir=out_dir,
            jsonl_dir=jsonl_dir,
            category=cat,
            pad_square=pad_square,
            max_samples=per_cat[cat],
            seed=seed,
            force=force,
        )

    summary: dict[str, Any] = {"status": "built", "targets": {}, "skipped": {}}
    for name, cats in targets.items():
        merged: list[dict[str, Any]] = []
        for cat in cats:
            merged.extend(cat_results[cat].records)
        n = _write_jsonl(out_dir / name, merged)
        summary["targets"][str(out_dir / name)] = n

    total_parse = sum(r.skipped_parse for r in cat_results.values())
    total_image = sum(r.skipped_image for r in cat_results.values())
    for cat, r in cat_results.items():
        summary["skipped"][cat] = {"parse": r.skipped_parse, "image": r.skipped_image}
    if total_parse or total_image:
        print(
            f"[prepare-vqa] スキップ: parse={total_parse} image={total_image} "
            f"(カテゴリ別 {summary['skipped']})",
            file=sys.stderr,
        )
    return summary


# --------------------------------------------------------------------------- #
# CLI（薄い）
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="fetch_vqa_data.sh が置く raw ディレクトリ")
    parser.add_argument("--out", required=True, help="出力先（lerobot_vqa ディレクトリ）")
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help='カテゴリ名。空白区切り 1 引数 "Understanding Task_planning" でも複数引数でも可',
    )
    parser.add_argument("--max-samples", type=int, default=None, help="カテゴリ合計の上限")
    parser.add_argument(
        "--max-samples-per-cat",
        nargs="+",
        default=None,
        metavar="CAT=N",
        help="カテゴリ別上限。例: Understanding=20000 Task_planning=20000",
    )
    parser.add_argument("--merge-into", default=None, help="全カテゴリを 1 本に連結する出力名（例: all.jsonl）")
    parser.add_argument("--pad-square", dest="pad_square", action="store_true", default=True)
    parser.add_argument("--no-pad-square", dest="pad_square", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="既存出力を無視して再生成")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    categories = [c for item in args.categories for c in str(item).split()]
    if not categories:
        print("[prepare-vqa] --categories が空", file=sys.stderr)
        return 2

    per_cat: dict[str, int] | None = None
    if args.max_samples_per_cat:
        per_cat = {}
        for item in args.max_samples_per_cat:
            if "=" not in item:
                print(f"[prepare-vqa] --max-samples-per-cat は CAT=N 形式: {item!r}", file=sys.stderr)
                return 2
            key, _, value = item.partition("=")
            per_cat[key.strip()] = int(value)

    summary = build_vqa_dataset(
        src=args.src,
        out=args.out,
        categories=categories,
        max_samples=args.max_samples,
        max_samples_per_cat=per_cat,
        merge_into=args.merge_into,
        pad_square=args.pad_square,
        seed=args.seed,
        force=args.force,
    )
    for path, n in summary.get("targets", {}).items():
        print(f"[prepare-vqa] {summary['status']}: {path} ({n} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
