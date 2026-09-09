#!/usr/bin/env bash
# InternVLA-A1.5 本学習ランチャー（単 GPU / overlay / 単一 robot_type）。
#
#   source env_train.sh
#   bash scripts/train_ivla_a15.sh                 # フォアグラウンド
#   bash scripts/tmux_train.sh                     # tmux 常駐（ssh が切れても継続）
#
# 上流 launch/internvla_a15_finetune_libero.sh の ARGS を土台に、次を変えている:
#   - 単 GPU（--multi_gpu を外し num_processes=1）
#   - 入口を scripts/train_entry.py にする（overlay を明示 import。F2 対策）
#   - dataset.type を internvla_a1_5_parc に（RenderDownsampleFn 入り）
#   - robot_type はスイート別ではなく単一（prepare_dataset.py が info.json を書き換える）
#   - **warmup / decay_steps を実 step 数に再スケール**（計画 F12）
#     上流の warmup=2000 / decay=100000 は 100k step 前提。30k step にそのまま使うと
#     warmup が全体の 6.7% を食い、cosine は 3/10 しか進まずに終わる。
#   - optimizer preset は overlay が差し替え、**バックボーン VLM だけ lr を 0.1 倍**する
#     （IVLA_VLM_LR_SCALE。scripts/train_entry.py が適用を検証する）
#
# 注意: 上流 lerobot_train.py には **gradient accumulation が無い**
#   （update_policy() が毎バッチ optimizer.step() する）。実効バッチ = batch_size。
#   よって IVLA_BS は probe（scripts/probe_ivla_bs.sh）で載る最大値を採る。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

: "${RUN_NAME:=ivla_a15_libero_combined}"
: "${IVLA_DATASET_REPO_ID:=libero_combined_20hz}"
: "${IVLA_PRETRAINED_PATH:=${HF_HOME}/InternVLA-A1.5-base}"
: "${IVLA_VLM_MODEL_PATH:=Qwen/Qwen3.5-2B}"

# --- ハイパーパラメータ（ユーザー指定の確定値。引き継ぎ書 §4）------------------
: "${IVLA_STEPS:=30000}"
: "${IVLA_WARMUP_STEPS:=600}"      # 上流比率 2000/100000 = 2% を 30k に再スケール
: "${IVLA_DECAY_STEPS:=30000}"     # steps と一致させる
: "${IVLA_LR:=5e-5}"
: "${IVLA_DECAY_LR:=5e-6}"
: "${IVLA_SAVE_FREQ:=5000}"        # steps // 6 -> 5〜6 個残る
: "${IVLA_LOG_FREQ:=50}"
: "${IVLA_SAVE_CHECKPOINT:=true}"   # probe / smoke では false にして保存を省く
: "${IVLA_SEED:=42}"
: "${IVLA_BS:=8}"                  # probe で確定する
: "${IVLA_NUM_WORKERS:=8}"
: "${IVLA_GRADIENT_CHECKPOINTING:=false}"
: "${IVLA_ACTION_LOSS_ONLY:=false}"   # false = 動画ヘッド on（WAN 重みが必要）
: "${IVLA_VLM_LR_SCALE:=0.1}"         # ★ バックボーン VLM の学習率倍率（ユーザー指定）
export IVLA_VLM_LR_SCALE

# --- ダウンサンプルのカーネル確率（ユーザー指定・p_nearest は固定）-------------
: "${IVLA_RENDER_TARGET:=128}"
: "${IVLA_RENDER_P_BOX:=0.45}"
: "${IVLA_RENDER_P_TRIANGLE:=0.30}"
: "${IVLA_RENDER_P_CUBIC:=0.15}"
: "${IVLA_RENDER_P_NEAREST:=0.10}"

_activate_conda
_resolve_paths
_assert_env

# データセットは HF_LEROBOT_HOME/<repo_id> として解決される（utils/constants.py:67,
# datasets/lerobot_dataset.py:681）。展開先の親をここで指す。
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$(dirname "$IVLA_DATASET_ROOT")}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# probe の実測で BS=8 は peak 80.6GB / 95.6GB（84%）と余裕が小さい。プロンプト長は
# タスク（111 種）によって変わるので、確保パターンの断片化だけで OOM になりうる。
# expandable_segments は確保済みブロックを伸縮させて断片化を減らす。数値結果は変わらない。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# robot_type を単一値へ（冪等）。schema の image_mapping と実データの feature 名の
# 突き合わせもここで行う。
python "$IVLA_DIR/scripts/prepare_dataset.py" --root "$IVLA_DATASET_ROOT" \
    --robot-type "${IVLA_ROBOT_TYPE:-libero_combined}"

_maybe_resume
_wandb_args

ARGS=(
    --num_processes=1
    --num_machines=1
    --mixed_precision=no          # policy.dtype=bfloat16 側で扱う
    "$IVLA_DIR/scripts/train_entry.py"

    # ---- Output ----
    --output_dir="$OUT_DIR"
    --job_name="$RUN_NAME"
    --num_workers="$IVLA_NUM_WORKERS"

    # ---- Policy ----
    --policy.type=internvla_a1_5
    --policy.repo_id=lerobot_lab/internvla_a1_5
    --policy.pretrained_path="$IVLA_PRETRAINED_PATH"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing="$IVLA_GRADIENT_CHECKPOINTING"
    --policy.dtype=bfloat16
    --policy.optimizer_lr="$IVLA_LR"
    --policy.scheduler_warmup_steps="$IVLA_WARMUP_STEPS"
    --policy.scheduler_decay_steps="$IVLA_DECAY_STEPS"
    --policy.scheduler_decay_lr="$IVLA_DECAY_LR"
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.vlm_model_name_or_path="$IVLA_VLM_MODEL_PATH"
    --policy.enable_vqa_loss=true
    --policy.tokenize_state=true
    --policy.knowledge_insulation=false
    --policy.video_loss_only=false
    --policy.video_loss_weight=1
    --policy.action_loss_only="$IVLA_ACTION_LOSS_ONLY"
    --policy.freeze_learnable_tokens=false
    --policy.num_learnable_tokens=50

    # ---- Dataset（overlay: RenderDownsampleFn 入り）----
    --dataset.type=internvla_a1_5_parc
    --dataset.repo_id="$IVLA_DATASET_REPO_ID"
    --dataset.action_mode=abs
    --dataset.use_external_stats=false
    --dataset.dist_loading=false
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=true
    --dataset.render_downsample=true
    --dataset.render_target="$IVLA_RENDER_TARGET"
    --dataset.render_p_box="$IVLA_RENDER_P_BOX"
    --dataset.render_p_triangle="$IVLA_RENDER_P_TRIANGLE"
    --dataset.render_p_cubic="$IVLA_RENDER_P_CUBIC"
    --dataset.render_p_nearest="$IVLA_RENDER_P_NEAREST"

    # ---- Training ----
    --seed="$IVLA_SEED"
    --batch_size="$IVLA_BS"
    --steps="$IVLA_STEPS"
    --save_freq="$IVLA_SAVE_FREQ"
    --log_freq="$IVLA_LOG_FREQ"
    --save_checkpoint="$IVLA_SAVE_CHECKPOINT"
)
[ ${#RESUME_ARGS[@]} -gt 0 ] && ARGS+=("${RESUME_ARGS[@]}")
ARGS+=("${WANDB_ARGS[@]}")

echo "=== $RUN_NAME ==="
echo "  out       : $OUT_DIR"
echo "  log       : $LOG_FILE"
echo "  dataset   : $HF_LEROBOT_HOME/$IVLA_DATASET_REPO_ID"
echo "  bs=$IVLA_BS steps=$IVLA_STEPS warmup=$IVLA_WARMUP_STEPS decay=$IVLA_DECAY_STEPS"
echo "  lr=$IVLA_LR  vlm_lr_scale=$IVLA_VLM_LR_SCALE (= $(python -c "print($IVLA_LR*$IVLA_VLM_LR_SCALE)"))"
echo "  action_loss_only=$IVLA_ACTION_LOSS_ONLY  gradient_checkpointing=$IVLA_GRADIENT_CHECKPOINTING"

# 新規実行ならログを退避してから始める。`tee -a` で追記していると、前回の実行の
# Traceback や step 行が混ざって「今の実行が失敗している」ように読めてしまう
# （実際に読み違えた）。resume のときは続きなので残す。
if [ ${#RESUME_ARGS[@]} -eq 0 ] && [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE%.log}.$(date +%Y%m%d_%H%M%S).log"
    echo "==> 前回のログを退避した: ${LOG_FILE%.log}.*.log"
fi

_start_vram_sampler
# 保存のたびに古い training_state（11GB/個）を落とす。放っておくと 6 個で 96GB になり
# /home に入らず、終盤の保存で落ちる（計画 R8）。
_start_checkpoint_janitor
START=$(date +%s)
set +e
accelerate launch "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
set -e
END=$(date +%s)
_prune_checkpoint_states   # 最後にもう一度掃除する
_summarize_run "$START" "$END"
echo "=== checkpoints: $(du -sh "$OUT_DIR/checkpoints" 2>/dev/null | cut -f1) / 空き $(df -h "$OUT_DIR" | awk 'NR==2{print $4}') ==="
echo "=== exit=$RC ==="
exit "$RC"
