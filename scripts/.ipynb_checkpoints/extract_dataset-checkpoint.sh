#!/usr/bin/env bash
# 演習環境の共有データセット（~/dataset、読み取り専用マウント）から、
# 使用する tar を ~/data/ 以下に展開する。
#
# ~/dataset は小さいファイルの大量読み込みが極端に遅く、直接読み込む学習は
# 実用にならない。使用するデータセットの tar を本スクリプトで ~/data/ に
# 展開してから学習に使用すること。
#
# 使い方:
#   bash scripts/extract_dataset.sh                                  # 展開できる tar の一覧を表示
#   bash scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar # 指定した tar を展開
#   bash scripts/extract_dataset.sh <tar> <tar> ...                  # 複数指定可
#
# 場所は環境変数 DATASET_DIR / DATA_DIR で変更できる（既定: ~/dataset, ~/data）。
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-$HOME/dataset}"
DATA_DIR="${DATA_DIR:-$HOME/data}"

gb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

if [ ! -d "$DATASET_DIR" ]; then
    echo "エラー: データセットディレクトリが見つからない: $DATASET_DIR" >&2
    echo "演習環境以外で使う場合は DATASET_DIR で tar の置き場所を指定すること。" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "使い方: bash $0 <tar のパス>..."
    echo
    echo "展開できる tar（$DATASET_DIR からの相対パスで指定する）:"
    (cd "$DATASET_DIR" && find . -name '*.tar' -printf '%s %P\n' | sort -k2) \
        | while read -r size path; do
            printf '  %6s GB  %s\n' "$(gb "$size")" "$path"
        done
    echo
    echo "例: bash $0 lerobot/libero_combined_20hz.tar"
    exit 0
fi

mkdir -p "$DATA_DIR"

for name in "$@"; do
    # $DATASET_DIR からの相対パスと絶対パスの両方を受け付ける（.tar は省略可）
    tar_path=""
    for cand in "$DATASET_DIR/$name" "$DATASET_DIR/$name.tar" "$name" "$name.tar"; do
        if [ -f "$cand" ]; then tar_path="$cand"; break; fi
    done
    if [ -z "$tar_path" ]; then
        echo "エラー: tar が見つからない: $name" >&2
        echo "引数なしで実行すると一覧を表示する。" >&2
        exit 1
    fi

    # 展開先 = tar の先頭エントリのトップディレクトリ（tar 名と同名のディレクトリになっている）
    topdir="$(tar -tf "$tar_path" 2>/dev/null | head -n 1 | cut -d/ -f1)" || true
    case "$topdir" in
        "" | "." | ".." | -* | */*)
            echo "エラー: tar の内容を確認できない: $tar_path" >&2
            exit 1
            ;;
    esac
    dest="$DATA_DIR/$topdir"
    if [ -e "$dest" ]; then
        echo "スキップ: $dest は既に存在する（展開し直す場合は先に削除すること）"
        continue
    fi

    # 空き容量の確認（tar サイズ分を必要量の目安とする）
    need="$(stat -c %s "$tar_path")"
    avail="$(( $(df --output=avail -k "$DATA_DIR" | tail -n 1) * 1024 ))"
    if [ "$avail" -lt "$need" ]; then
        echo "エラー: $DATA_DIR の空き容量が不足している（必要 $(gb "$need") GB / 空き $(gb "$avail") GB）" >&2
        echo "不要になったデータセットを削除してから再実行すること。" >&2
        exit 1
    fi

    echo "展開中: $tar_path -> $dest（$(gb "$need") GB）"
    tar -xf "$tar_path" -C "$DATA_DIR"
    echo "完了: $dest"
done
