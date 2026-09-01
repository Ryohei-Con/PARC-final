#!/usr/bin/env python3
"""LoRA アダプタをベース重みにマージし、提出用の単一チェックポイントを書き出す。

    python scripts/merge_lora.py \
        --adapter ~/pi05-ft-outputs/<RUN_NAME>/checkpoints/020000/pretrained_model \
        --out     ~/pi05-ft-outputs/<RUN_NAME>/merged

patches/pi05-config-defaults.patch が追加する学習専用の設定キーは出力から
取り除く。残すと、パッチを当てていない lerobot でのロードが失敗する。
（drop_n_first_frames は旧版のパッチが入れていたもので、念のため対象に含める。）
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# パッチが追加する学習専用キー。推論時は不要なので取り除く。
_TRAIN_ONLY_CONFIG_KEYS = ("drop_n_last_frames", "drop_n_first_frames")


def strip_train_only_keys(config_path: Path) -> list[str]:
    """config.json から学習専用キーを除く。取り除いたキー名を返す。"""
    cfg = json.loads(config_path.read_text())
    removed = [k for k in _TRAIN_ONLY_CONFIG_KEYS if k in cfg]
    for k in removed:
        del cfg[k]
    if removed:
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        required=True,
        help="学習が出力した LoRA チェックポイント（checkpoints/<step>/pretrained_model）",
    )
    ap.add_argument(
        "--base",
        default="lerobot/pi05_libero_base",
        help="ベース重み。Hub の repo id、またはローカルパス",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="マージ済みチェックポイントの出力先ディレクトリ",
    )
    ap.add_argument(
        "--tokenizer",
        default="google/paligemma-3b-pt-224",
        help="同梱するトークナイザ。gated repo のため HF_TOKEN が必要",
    )
    args = ap.parse_args()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from peft import PeftModel

    adapter = Path(args.adapter)
    out = Path(args.out)
    if not adapter.is_dir():
        raise SystemExit(f"adapter が見つからない: {adapter}")

    print(f"[merge] base    = {args.base}")
    print(f"[merge] adapter = {adapter}")
    base = PI05Policy.from_pretrained(args.base)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)

    # 正規化統計とプロセッサ設定は学習時のチェックポイント側にあるため、
    # save_pretrained が書かないものをコピーで補う。
    for name in ("train_config.json",):
        src = adapter / name
        if src.is_file():
            shutil.copy2(src, out / name)
    for pattern in ("*preprocessor*", "*postprocessor*", "*stats*"):
        for src in adapter.glob(pattern):
            dst = out / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    # トークナイザ（既定は google/paligemma-3b-pt-224）を同梱する。採点環境は
    # 外部通信を遮断しており、かつ gated repo なので、実行時には取得できない。
    from transformers import AutoTokenizer

    tok_dir = out / "tokenizer"
    AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(tok_dir)
    print(f"[merge] トークナイザを同梱: {tok_dir}（{args.tokenizer}）")

    config_path = out / "config.json"
    if config_path.is_file():
        removed = strip_train_only_keys(config_path)
        if removed:
            print(f"[merge] 学習専用キーを除去: {', '.join(removed)}")

    print(f"[merge] 出力: {out}")


if __name__ == "__main__":
    main()
