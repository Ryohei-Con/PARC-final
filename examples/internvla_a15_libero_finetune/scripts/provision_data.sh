#!/usr/bin/env bash
# 再起動後にデータセットと重みを入れ直す（$HOME/data はエフェメラル）。
#
#   source env_train.sh
#   bash scripts/provision_data.sh
#
# **この機体の $HOME/data は再起動のたびに空になる。** 実測で 2 回消えている
# （/opt/dlami/nvme/vol/data.img が作り直される）。一方 /home/ryokondo は
# 再起動をまたいで残る。よって配置はこう決めてある:
#
#   $HOME/data     … データセット + HF 重み  → **消えてよい。ここで入れ直す**
#   $HOME/ivla-a15-outputs … チェックポイント → **消えては困る。永続側に置く**
#   $HOME/ivla-a15-logs    … ログ / VRAM CSV  → 同上
#   $HOME/dataset  … 配布 tar（読み取り専用マウント）→ 常にここから展開できる
#
# 冪等。既にあるものは飛ばす。所要時間は実測でデータセット約 2 分 + 重み約 5 分。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"
PARC_ROOT="$(cd "$IVLA_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

say() { echo "[provision] $*"; }

say "1/3 データセット"
if [ -f "$IVLA_DATASET_ROOT/meta/info.json" ]; then
    say "  展開済み: $IVLA_DATASET_ROOT"
else
    bash "$PARC_ROOT/scripts/extract_dataset.sh" lerobot/libero_combined_20hz.tar
fi

say "2/3 robot_type の書き換え（冪等）"
_activate_conda
python "$IVLA_DIR/scripts/prepare_dataset.py" --root "$IVLA_DATASET_ROOT" \
    --robot-type "${IVLA_ROBOT_TYPE:-libero_combined}"

say "3/3 重み"
if [ -f "$HF_HOME/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ] \
   && [ -f "$HF_HOME/InternVLA-A1.5-base/model.safetensors" ]; then
    say "  取得済み: $HF_HOME"
else
    bash "$IVLA_DIR/scripts/fetch_weights.sh"
fi

say "検証"
IVLA_DIR="$IVLA_DIR" python "$IVLA_DIR/scripts/verify_train_env.py"

# 4/4 VQA データ（任意。IVLA_VQA_ENABLE=1 のときだけ / 計画 §5.4）。
# 未設定なら echo 1 行のみで既存挙動は完全に不変。RoboInter-VQA は全体 150GB あるので
# fetch_vqa_data.sh がサブセットだけ取得し、IVLA_VQA_MAX_GIB 超過なら取得せず失敗する。
if [ "${IVLA_VQA_ENABLE:-0}" = "1" ]; then
    say "4/4 VQA サブセット（RoboInter-VQA）"
    bash "$IVLA_DIR/scripts/fetch_vqa_data.sh"
    python "$IVLA_DIR/scripts/prepare_vqa_data.py" \
        --src  "${IVLA_VQA_ROOT:-$HOME/data/robointer_vqa}/raw" \
        --out  "${IVLA_VQA_ROOT:-$HOME/data/robointer_vqa}/lerobot_vqa" \
        --categories "${IVLA_VQA_CATEGORIES:-Understanding Task_planning}" \
        --max-samples "${IVLA_VQA_MAX_SAMPLES:-40000}" \
        --merge-into all.jsonl \
        --pad-square
else
    say "4/4 VQA データはスキップ（IVLA_VQA_ENABLE=1 で有効化）"
fi

say "完了。学習を開始できる: bash scripts/tmux_train.sh"
