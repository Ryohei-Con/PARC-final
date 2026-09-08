"""提出パッケージ（``submission/``）の生成（計画 P6-32）。

実行はクラウド。ただし ``copy_safetensors_without()`` / ``find_tied_duplicates()`` /
``list_droppable_keys()`` は小さな合成 safetensors で手元テストできる
（``tests/test_export_submission.py``）。

手順:

1. :func:`list_droppable_keys` で ``model.wan_video_model.*`` が**実際に checkpoint に
   存在するか**を確認し、結果を MANIFEST に記録する（計画 F5 の実測）。
   上流の ``state_dict()`` は既に ``model.wan_video_model.*`` を除外しているので、
   通常はここが 0 件になる。残るのは ``model.learnable_to_wan_proj.*`` と
   buffer ``model._wan_grid_sizes``（数十 MB）。
2. それらを落とす。**``learnable_tokens`` / ``learnable_tokens_in_proj`` は絶対に
   落とさない。** foresight token の本体であり、推論でも使う。
3. tie 重複テンソルを SHA256 比較で検出し、あればストリーミングで除去（B5）。
4. VLM は config / tokenizer / processor だけコピー（B4）。
   **``model.safetensors.index.json`` は必ず除外する。** 残すと ``from_pretrained`` が
   存在しない shard を参照しに行く。
5. ``src/lerobot`` を vendor へコピーし、``libero_combined.yaml`` を「追加」する。
6. ``runtime_config.json`` を書く。
7. ``MANIFEST.json``（元 checkpoint / step / 上流 commit / 落としたキー /
   追加ファイル / 各ファイルの SHA256）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

#: 落としてよいキーの接頭辞（F5）
DROPPABLE_PREFIXES: tuple[str, ...] = (
    "model.wan_video_model.",
    "model.learnable_to_wan_proj.",
)
#: 落としてよい単独キー
DROPPABLE_EXACT: tuple[str, ...] = ("model._wan_grid_sizes",)

#: **絶対に落としてはいけない**キーの部分文字列（誤爆の番人）
PROTECTED_SUBSTRINGS: tuple[str, ...] = (
    "learnable_tokens",
    "learnable_tokens_in_proj",
)

#: VLM ディレクトリから持っていく拡張子（B4。重みは持っていかない）
VLM_PATTERNS: tuple[str, ...] = ("*.json", "*.txt", "*.model", "*.jinja")

#: VLM から必ず除外するファイル
VLM_EXCLUDE: tuple[str, ...] = ("model.safetensors.index.json", "pytorch_model.bin.index.json")

#: safetensors の dtype 文字列 -> 1 要素のバイト数
_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_header(path: Path) -> tuple[dict, int]:
    """safetensors のヘッダ（JSON）とデータ開始オフセットを返す。"""
    with open(path, "rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is not a safetensors file (truncated header length)")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return header, 8 + header_len


def list_droppable_keys(safetensors_path: str | Path) -> list[str]:
    """落としてよいキーを列挙する。保護対象に触れたら例外。"""
    header, _ = _read_header(Path(safetensors_path))
    keys = [k for k in header if k != "__metadata__"]

    droppable = [
        key
        for key in keys
        if key.startswith(DROPPABLE_PREFIXES) or key in DROPPABLE_EXACT
    ]

    for key in droppable:
        for protected in PROTECTED_SUBSTRINGS:
            if protected in key:
                raise RuntimeError(
                    f"refusing to drop {key!r}: it matches the protected substring "
                    f"{protected!r}. learnable_tokens must stay in the checkpoint."
                )
    return sorted(droppable)


def find_tied_duplicates(safetensors_path: str | Path) -> list[str]:
    """データ部が完全に一致するテンソルの「片方」を返す（B5）。

    Qwen3.5 は入力埋め込みと出力層で重みを共有する。checkpoint に同じ 1GB の
    テンソルが 2 つ入っていることがある。落とすのは ``lm_head`` 側（無ければ
    辞書順で後ろの方）にする。欠損は :func:`assert_checkpoint_covers_model` が
    起動時に検出するので、やり過ぎればそこで落ちる。
    """
    path = Path(safetensors_path)
    header, data_offset = _read_header(path)

    by_digest: dict[str, list[str]] = {}
    with open(path, "rb") as handle:
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            handle.seek(data_offset + start)
            digest = hashlib.sha256()
            remaining = end - start
            while remaining > 0:
                block = handle.read(min(remaining, 1 << 20))
                if not block:
                    raise ValueError(f"{path}: unexpected EOF while hashing {key}")
                digest.update(block)
                remaining -= len(block)
            by_digest.setdefault(digest.hexdigest(), []).append(key)

    duplicates: list[str] = []
    for keys in by_digest.values():
        if len(keys) < 2:
            continue
        keys = sorted(keys)
        preferred = [k for k in keys if "lm_head" in k]
        drop = preferred[:1] if preferred else keys[1:]
        # 全部は落とさない（必ず 1 本残す）
        keep = set(keys) - set(drop)
        if not keep:
            drop = keys[1:]
        duplicates.extend(drop)
    return sorted(set(duplicates))


def copy_safetensors_without(
    src: str | Path,
    dst: str | Path,
    drop_keys: Iterable[str],
    *,
    metadata: dict[str, str] | None = None,
) -> dict:
    """``drop_keys`` を除いた safetensors を**ストリーミングで**書き出す。

    6GB を一度にメモリへ載せない。1 テンソルずつ読み書きする。

    Returns:
        ``{"kept": [...], "dropped": [...], "bytes_before": int, "bytes_after": int}``
    """
    src = Path(src)
    dst = Path(dst)
    drop = set(drop_keys)

    header, data_offset = _read_header(src)
    original_metadata = header.get("__metadata__", {})
    keys = [k for k in header if k != "__metadata__"]

    unknown = drop - set(keys)
    if unknown:
        raise KeyError(f"cannot drop keys that are not in {src}: {sorted(unknown)}")
    for key in drop:
        for protected in PROTECTED_SUBSTRINGS:
            if protected in key:
                raise RuntimeError(f"refusing to drop protected key {key!r}")

    kept = [k for k in keys if k not in drop]
    if not kept:
        raise ValueError("refusing to write an empty safetensors file")

    # --- 新しいヘッダを組み立てる（データはまだ書かない）---
    new_header: dict = {}
    merged_metadata = dict(original_metadata)
    if metadata:
        merged_metadata.update({str(k): str(v) for k, v in metadata.items()})
    if merged_metadata:
        new_header["__metadata__"] = merged_metadata

    cursor = 0
    for key in kept:
        entry = header[key]
        start, end = entry["data_offsets"]
        size = end - start
        expected = _DTYPE_SIZES.get(entry["dtype"])
        if expected is not None:
            elements = 1
            for dim in entry["shape"]:
                elements *= int(dim)
            if elements * expected != size:
                raise ValueError(
                    f"{src}: inconsistent entry for {key}: shape={entry['shape']} "
                    f"dtype={entry['dtype']} bytes={size}"
                )
        new_header[key] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # safetensors は 8 バイト境界へのパディングを許す
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as reader, open(dst, "wb") as writer:
        writer.write(struct.pack("<Q", len(header_bytes)))
        writer.write(header_bytes)
        for key in kept:
            start, end = header[key]["data_offsets"]
            reader.seek(data_offset + start)
            remaining = end - start
            while remaining > 0:
                block = reader.read(min(remaining, 1 << 22))
                if not block:
                    raise ValueError(f"{src}: unexpected EOF while copying {key}")
                writer.write(block)
                remaining -= len(block)

    return {
        "kept": kept,
        "dropped": sorted(drop),
        "bytes_before": src.stat().st_size,
        "bytes_after": dst.stat().st_size,
    }


def copy_vlm_config_only(src: str | Path, dst: str | Path) -> list[str]:
    """VLM の config / tokenizer / processor だけコピーする（B4）。

    ``snapshot_download(allow_patterns=...)`` は「何をダウンロードするか」しか
    制御しない。学習で既に重みをキャッシュ済みだと snapshot に重みが入っているので、
    **コピー時にも同じ patterns で絞る**。
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for pattern in VLM_PATTERNS:
        for path in sorted(src.rglob(pattern)):
            if not path.is_file():
                continue
            if path.name in VLM_EXCLUDE:
                logger.info("skipping %s (would point at missing shards)", path.name)
                continue
            relative = path.relative_to(src)
            target = dst / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied.append(str(relative))

    leftovers = [p.name for p in dst.rglob("*.safetensors")] + [
        p.name for p in dst.rglob("*.bin")
    ]
    if leftovers:
        raise RuntimeError(f"VLM directory still contains weights: {leftovers}")
    return sorted(copied)


def copy_vendor_lerobot(upstream_src: str | Path, dst: str | Path, extra_schema: Path | None = None) -> dict:
    """``src/lerobot`` を丸ごと vendor へコピーし、SHA256 を記録する。

    ``libero_combined.yaml`` は ``dataset_schemas/configs/`` へ**追加**する。
    これが「上流に無い追加ファイル」であることを MANIFEST に明記する（計画 §3 例外規定）。
    """
    upstream_src = Path(upstream_src)
    source = upstream_src / "lerobot"
    if not (source / "__init__.py").is_file():
        raise FileNotFoundError(f"upstream lerobot not found at {source}")

    dst = Path(dst)
    target = dst / "lerobot"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    added: list[str] = []
    if extra_schema is not None:
        extra_schema = Path(extra_schema)
        schema_dst = target / "dataset_schemas" / "configs" / extra_schema.name
        if schema_dst.exists():
            raise RuntimeError(
                f"{schema_dst} already exists upstream; this must be an ADDED file, not an edit"
            )
        shutil.copyfile(extra_schema, schema_dst)
        added.append(str(schema_dst.relative_to(dst)))

    hashes = {
        str(path.relative_to(dst)): sha256_file(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    return {"files": len(hashes), "sha256": hashes, "added_files": added}


def write_runtime_config(dst: str | Path, cfg: dict) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def _git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def export(
    ckpt_dir: str | Path,
    vlm_src: str | Path,
    upstream_src: str | Path,
    out_dir: str | Path,
    *,
    drop_wan_keys: bool = True,
    dedupe_tied: bool = True,
    runtime_config: str | Path | None = None,
    extra_files: Iterable[Path] = (),
) -> dict:
    """提出ツリーを作り、``MANIFEST.json`` を返す。"""
    ckpt_dir = Path(ckpt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_weights = ckpt_dir / "model.safetensors"
    if not src_weights.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {src_weights}")

    manifest: dict = {
        "source_checkpoint": str(ckpt_dir.resolve()),
        "upstream_src": str(Path(upstream_src).resolve()),
        "upstream_commit": _git_commit(Path(upstream_src).parent),
        "vlm_source": str(Path(vlm_src).resolve()),
    }

    # --- 1/2. 落とすキーを決める（F5 の実測を記録する）---
    droppable = list_droppable_keys(src_weights) if drop_wan_keys else []
    manifest["wan_keys_present_in_checkpoint"] = [
        k for k in droppable if k.startswith("model.wan_video_model.")
    ]
    manifest["dropped_keys"] = list(droppable)

    # --- 3. tie 重複 ---
    tied: list[str] = []
    if dedupe_tied:
        tied = find_tied_duplicates(src_weights)
        if tied:
            manifest["tied_duplicates_dropped"] = tied
        else:
            manifest["tied_duplicates_dropped"] = []
            logger.info("no tied duplicate tensors found; skipping dedupe")

    checkpoint_out = out_dir / "model_weights" / "checkpoint"
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    copy_report = copy_safetensors_without(
        src_weights,
        checkpoint_out / "model.safetensors",
        set(droppable) | set(tied),
    )
    manifest["weights"] = {
        "bytes_before": copy_report["bytes_before"],
        "bytes_after": copy_report["bytes_after"],
        "kept_keys": len(copy_report["kept"]),
        "dropped_keys": copy_report["dropped"],
    }

    for name in ("config.json", "stats.json", "train_config.json"):
        candidate = ckpt_dir / name
        if candidate.is_file():
            shutil.copyfile(candidate, checkpoint_out / name)
        else:
            logger.warning("%s not found next to the checkpoint", name)

    # --- 4. VLM ---
    manifest["vlm_files"] = copy_vlm_config_only(vlm_src, out_dir / "model_weights" / "vlm")

    # --- 5. vendor lerobot ---
    schema_yaml = (
        Path(__file__).resolve().parent.parent
        / "lerobot_policy_parc"
        / "schemas"
        / "libero_combined.yaml"
    )
    manifest["vendor"] = copy_vendor_lerobot(upstream_src, out_dir / "vendor", schema_yaml)

    # --- 6. runtime_config.json + 提出必須ファイル ---
    inference_dir = Path(__file__).resolve().parent.parent / "inference"
    runtime_config_path = Path(runtime_config or (inference_dir / "runtime_config.json"))
    cfg = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    write_runtime_config(out_dir / "runtime_config.json", cfg)

    for name in ("policy_server.py", "internvla_runtime.py", "chunk_blending.py",
                 "verify_inference.py", "requirements.txt"):
        shutil.copyfile(inference_dir / name, out_dir / name)
    for extra in extra_files:
        shutil.copyfile(extra, out_dir / Path(extra).name)

    # --- 7. MANIFEST ---
    manifest["files"] = {
        str(path.relative_to(out_dir)): sha256_file(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    manifest["total_bytes"] = sum(
        path.stat().st_size for path in out_dir.rglob("*") if path.is_file()
    )

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--vlm-src", required=True)
    parser.add_argument("--upstream-src", required=True, help="上流リポジトリの src/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runtime-config", default=None)
    parser.add_argument("--no-drop-wan", action="store_true")
    parser.add_argument("--no-dedupe-tied", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    manifest = export(
        args.ckpt_dir,
        args.vlm_src,
        args.upstream_src,
        args.out_dir,
        drop_wan_keys=not args.no_drop_wan,
        dedupe_tied=not args.no_dedupe_tied,
        runtime_config=args.runtime_config,
    )
    summary = {k: v for k, v in manifest.items() if k not in ("files", "vendor")}
    summary["vendor_files"] = manifest["vendor"]["files"]
    summary["vendor_added_files"] = manifest["vendor"]["added_files"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"total size: {manifest['total_bytes'] / 2**30:.2f} GiB (limit 20 GiB zip / 40 GiB unpacked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
