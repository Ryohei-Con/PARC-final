#!/usr/bin/env bash
# RoboInter-VQA（InternRobotics/RoboInter-VQA, HF dataset）の **サブセットだけ**を
# 取得して in-place 展開する（計画 §5.2 / PV1）。全体は 150GB あるので必ず絞る。
#
#   source env_train.sh && source env.vqa.sh   # env.vqa.example.sh を元に作る
#   bash scripts/fetch_vqa_data.sh
#
# 取得するもの:
#   1. meta : 各カテゴリの llava_format .json（smart_resize / origin 形式は取らない）
#   2. image: 各カテゴリの image zip（IVLA_VQA_IMAGE_INCLUDE で上書き可）
#
# zip は「その zip があるディレクトリ」へ `unzip -o` で in-place 展開し、展開後に
# 削除する（IVLA_VQA_KEEP_ZIP=1 で残す）。$IVLA_VQA_ROOT/raw/.fetch_done が
# あれば取得済みとみなし検証だけして exit 0（冪等）。
#
# 安全ガード: 取得予定パターンにマッチするファイルの合計が IVLA_VQA_MAX_GIB
# （既定 15）を超えたら **1 バイトも取得せず exit 1**。$HOME/data はエフェメラルで
# 既に libero + 重みで ~50GB 使っているため。
#
# オフライン再現性: 取得後は prep も本走もローカルファイルしか触らない
# （HF_HUB_OFFLINE=1 でも動く）。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

: "${IVLA_VQA_ROOT:=${HOME}/data/robointer_vqa}"
: "${IVLA_VQA_HF_REPO:=InternRobotics/RoboInter-VQA}"
: "${IVLA_VQA_CATEGORIES:=Understanding Task_planning}"   # 空白区切りで複数可（U2）
: "${IVLA_VQA_IMAGE_INCLUDE:=}"                            # 未指定なら "robotinter/<cat>/image/*.zip"
: "${IVLA_VQA_MAX_GIB:=15}"
: "${IVLA_VQA_KEEP_ZIP:=0}"

RAW_DIR="$IVLA_VQA_ROOT/raw"
DONE_MARKER="$RAW_DIR/.fetch_done"

say() { echo "[fetch-vqa] $*"; }

# 各カテゴリの llava_format .json が読め、先頭レコードの画像 1 件が展開後ツリーに
# 存在することを確認する。prepare_vqa_data.py の画像解決ロジックを流用する。
_verify_subset() {
    python - "$RAW_DIR" "$IVLA_VQA_CATEGORIES" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
from prepare_vqa_data import (  # noqa: E402
    _discover_meta_files,
    _extract_image_list,
    load_json_array,
    resolve_source_image,
)

raw_dir = Path(sys.argv[1])
categories = sys.argv[2].split()
problems = []
for cat in categories:
    metas = _discover_meta_files(raw_dir, cat)
    if not metas:
        problems.append(f"{cat}: llava_format .json が無い")
        continue
    records = load_json_array(metas[0])
    if not records:
        problems.append(f"{cat}: {metas[0]} が空")
        continue
    imgs = _extract_image_list(records[0])
    if not imgs:
        problems.append(f"{cat}: 先頭レコードに images が無い")
        continue
    resolved = resolve_source_image(imgs[0], raw_dir, cat)
    if resolved is None:
        problems.append(f"{cat}: 画像が展開後ツリーに無い: {imgs[0]!r}")
        continue
    print(f"[fetch-vqa]   OK {cat}: {metas[0].name} / {resolved}")

if problems:
    for p in problems:
        print(f"[fetch-vqa]   FAIL {p}", file=sys.stderr)
    sys.exit(1)
PY
}

_activate_conda

if ! command -v unzip >/dev/null 2>&1; then
    say "ERROR: unzip が無い（apt-get install unzip / conda install unzip）" >&2
    exit 1
fi

# --- allow_patterns を組み立てる ------------------------------------------------
META_PATTERNS=()
IMAGE_PATTERNS=()
for cat in $IVLA_VQA_CATEGORIES; do
    META_PATTERNS+=("robotinter/${cat}/meta/*llava*")
done
if [ -n "$IVLA_VQA_IMAGE_INCLUDE" ]; then
    # 空白区切りで複数パターン可
    # shellcheck disable=SC2206
    IMAGE_PATTERNS=($IVLA_VQA_IMAGE_INCLUDE)
else
    for cat in $IVLA_VQA_CATEGORIES; do
        IMAGE_PATTERNS+=("robotinter/${cat}/image/*.zip")
    done
fi
ALL_PATTERNS=("${META_PATTERNS[@]}" "${IMAGE_PATTERNS[@]}")

# --- 0. 冪等ガード ------------------------------------------------------------
if [ -f "$DONE_MARKER" ]; then
    say "取得済み: $DONE_MARKER"
    cat "$DONE_MARKER"
    say "検証（各カテゴリ先頭レコードの画像 1 件）"
    _verify_subset || {
        say "検証失敗。再取得するには $DONE_MARKER を削除すること" >&2
        exit 1
    }
    exit 0
fi

mkdir -p "$RAW_DIR"

# --- 1. ファイル一覧 + サイズ表 --------------------------------------------------
say "リポジトリのファイル一覧を取得: $IVLA_VQA_HF_REPO (repo_type=dataset)"
say "取得予定パターン: ${ALL_PATTERNS[*]}"
python - "$IVLA_VQA_HF_REPO" "$IVLA_VQA_MAX_GIB" "${ALL_PATTERNS[@]}" <<'PY'
import fnmatch
import sys

from huggingface_hub import HfApi

repo_id, max_gib, *patterns = sys.argv[1:]
max_bytes = float(max_gib) * (1024 ** 3)

api = HfApi()
info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)

rows = []
for sibling in info.siblings:
    size = sibling.size or 0
    rows.append((sibling.rfilename, size))
rows.sort()

matched_total = 0
matched_n = 0
print(f"{'size(MiB)':>12}  path")
for name, size in rows:
    is_match = any(fnmatch.fnmatch(name, pat) for pat in patterns)
    if is_match:
        matched_total += size
        matched_n += 1
    flag = " *" if is_match else "  "
    print(f"{size / 1048576:12.1f}{flag} {name}")

print()
print(f"[fetch-vqa] マッチ {matched_n} ファイル / 合計 {matched_total / 1073741824:.2f} GiB "
      f"(上限 {float(max_gib):.2f} GiB)")
if matched_total > max_bytes:
    print("[fetch-vqa] ERROR: 取得予定サイズが IVLA_VQA_MAX_GIB を超えている。"
          "IVLA_VQA_CATEGORIES / IVLA_VQA_IMAGE_INCLUDE を絞ること。", file=sys.stderr)
    sys.exit(1)
if matched_n == 0:
    print("[fetch-vqa] ERROR: パターンに 1 件もマッチしない。パターンを確認すること。", file=sys.stderr)
    sys.exit(1)
PY

# --- 2/3. meta + image を snapshot_download --------------------------------------
say "meta + image を取得 -> $RAW_DIR"
python - "$IVLA_VQA_HF_REPO" "$RAW_DIR" "${ALL_PATTERNS[@]}" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, local_dir, *patterns = sys.argv[1:]
path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=patterns,
    max_workers=8,
)
print(f"[fetch-vqa]   -> {path}")
PY

# --- 4. zip を in-place 展開 --------------------------------------------------
say "zip を in-place 展開"
zip_count=0
while IFS= read -r -d '' zipf; do
    zip_count=$((zip_count + 1))
    say "  unzip -o $zipf"
    unzip -o -q "$zipf" -d "$(dirname "$zipf")"
    if [ "$IVLA_VQA_KEEP_ZIP" != "1" ]; then
        rm -f "$zipf"
    fi
done < <(find "$RAW_DIR" -type f -name '*.zip' -print0)
say "  展開した zip: $zip_count"

# --- 5. .fetch_done マーカ ---------------------------------------------------
total_bytes=$(du -sb "$RAW_DIR" 2>/dev/null | cut -f1 || echo 0)
file_count=$(find "$RAW_DIR" -type f | wc -l | tr -d ' ')
{
    echo "categories: $IVLA_VQA_CATEGORIES"
    echo "patterns  : ${ALL_PATTERNS[*]}"
    echo "files     : $file_count"
    echo "bytes     : $total_bytes"
    echo "fetched_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$DONE_MARKER"
say "マーカ書き込み: $DONE_MARKER"
cat "$DONE_MARKER"

# --- 6. 検証 ---------------------------------------------------------------
say "検証（各カテゴリ先頭レコードの画像 1 件）"
_verify_subset

say "完了。次: python scripts/prepare_vqa_data.py --src $RAW_DIR --out $IVLA_VQA_ROOT/lerobot_vqa ..."
