#!/usr/bin/env bash
# 短い学習を 1 本回して、レシピが通ることを確認する（スモークテスト）。
# 本番の学習を仕掛ける前と、パッチや lerobot を触ったあとに実行する。
#
#   source env_train.sh
#   bash scripts/smoke_pi05.sh
#
# 確認すること:
#   1. LoRA 学習が最後まで走る
#   2. cfg.steps がマイクロステップではなく実効ステップ数として扱われる
#      （保存されるステップ番号で確認する。ただしこれは整合性の確認であって、
#        勾配累積が実際に効いているかの証明にはならない。下記を参照）
#   3. チェックポイントが保存される
#   4. merge_lora.py でマージできる（提出物にするまでの経路）
#
# env: SMOKE_BS, SMOKE_GA, SMOKE_STEPS, SMOKE_SKIP_MERGE=1（マージを省略）

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./_train_common.sh
source "$HERE/_train_common.sh"

SMOKE_BS="${SMOKE_BS:-2}"
SMOKE_GA="${SMOKE_GA:-2}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
RUN_NAME="${RUN_NAME:-smoke_pi05}"

PI05_DATASET_REPO_ID="${PI05_DATASET_REPO_ID:-local/libero_combined_20hz}"
PI05_DATASET_ROOT="${PI05_DATASET_ROOT:-${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}/libero_combined_20hz}"

# 動画デコーダ。torchcodec は FFmpeg の共有ライブラリ（libavutil 等）を必要とし、
# 入っていないと DataLoader が落ちる。scripts/setup_train.sh がその ffmpeg を
# 入れるので、既定は速い torchcodec にしてある。入れられなかった環境（setup の
# 動作確認で「torchcodec: 使えない」と出る）では PI05_VIDEO_BACKEND=pyav に落と
# すこと。pyav は自前で FFmpeg を同梱するため追加の system 依存が要らない。
PI05_VIDEO_BACKEND="${PI05_VIDEO_BACKEND:-torchcodec}"

_resolve_paths
rm -rf "$OUT_DIR"          # スモークは毎回まっさらから
_require_hf_token
_activate_venv

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export LEROBOT_GRAD_ACCUM=$SMOKE_GA

_start_vram_sampler

echo "=== smoke :: bs=$SMOKE_BS ga=$SMOKE_GA eff=$((SMOKE_BS * SMOKE_GA)) steps=$SMOKE_STEPS ==="
START=$(date +%s)

lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_libero_base \
    --policy.dtype=bfloat16 \
    --policy.n_obs_steps=1 \
    --policy.n_action_steps=10 \
    --policy.optimizer_lr=5e-5 \
    --policy.num_inference_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --peft.method_type=LORA \
    --peft.r=16 \
    --dataset.repo_id="$PI05_DATASET_REPO_ID" \
    --dataset.root="$PI05_DATASET_ROOT" \
    --dataset.video_backend="$PI05_VIDEO_BACKEND" \
    --batch_size=$SMOKE_BS \
    --steps=$SMOKE_STEPS \
    --num_workers=2 \
    --save_freq=$SMOKE_STEPS \
    --log_freq=1 \
    --output_dir="$OUT_DIR" \
    --job_name="$RUN_NAME" \
    --seed=1000 \
    --wandb.enable=false \
    2>&1 | tee "$LOG_FILE"

_summarize_run "$START" "$(date +%s)"

# --- 確認 ---
CKPT="$OUT_DIR/checkpoints/last/pretrained_model"
if [[ ! -d "$CKPT" ]]; then
    echo "NG: チェックポイントが出ていない: $CKPT" >&2
    exit 1
fi
echo "OK: チェックポイント $CKPT"

# ディレクトリ名は保存時のステップ番号（桁数は lerobot が総ステップ数から決める）。
# これが SMOKE_STEPS と一致すれば、log/save の刻みが cfg.steps と揃っている。
#
# 注意: この確認は勾配累積が効いていることの証明にはならない。パッチ無し
# （1 ステップ = 1 マイクロステップ）でも同じ番号で保存されるため、両者を
# 区別できない。
STEP_DIR=$(basename "$(readlink -f "$OUT_DIR/checkpoints/last")")
if [[ ! "$STEP_DIR" =~ ^[0-9]+$ ]]; then
    STEP_DIR=$(find "$OUT_DIR/checkpoints" -maxdepth 1 -regex '.*/[0-9]+' -printf '%f\n' | sort | tail -1)
fi
if [[ "${STEP_DIR:-}" =~ ^[0-9]+$ ]] && (( 10#$STEP_DIR == SMOKE_STEPS )); then
    echo "OK: $SMOKE_STEPS 実効ステップで保存（GA=$SMOKE_GA）"
else
    echo "NG: 保存されたステップ番号が $SMOKE_STEPS と一致しない（$STEP_DIR）" >&2
    echo "    cfg.steps がマイクロステップとして扱われている可能性がある" >&2
    echo "    （grad-accum-env-var.patch を確認すること）" >&2
    exit 1
fi

if [[ -n "${SMOKE_SKIP_MERGE:-}" ]]; then
    echo "スキップ: LoRA のマージ（SMOKE_SKIP_MERGE）"
else
    MERGED="$OUT_DIR/merged"
    rm -rf "$MERGED"
    python "$HERE/merge_lora.py" --adapter "$CKPT" --out "$MERGED"
    python - "$MERGED" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads((Path(sys.argv[1]) / "config.json").read_text())
leftover = [k for k in ("drop_n_last_frames", "drop_n_first_frames") if k in cfg]
if leftover:
    sys.exit(f"NG: 学習専用キーが残っている: {leftover}")
print("OK: マージ済みチェックポイントから学習専用キーが除かれている")
PY
    echo "OK: マージ $MERGED"
fi

cat <<EOF

=== smoke 完了 ===

マージ済みチェックポイント: ${MERGED:-<SMOKE_SKIP_MERGE のためなし>}

提出物にまとめる手順はこのレシピでは扱わない。採点環境で実際に動作した構成が
サンプル提出物 pi05_step005000_submission_py310.zip として配布されているので、
そちらを参照すること。提出物としての確認は評価用の環境（ルートの setup.sh +
env.sh）で行うこと。学習用 venv では採点環境の依存を再現できない。
EOF
