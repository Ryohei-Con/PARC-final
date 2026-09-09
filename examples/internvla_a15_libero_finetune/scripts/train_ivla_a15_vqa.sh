#!/usr/bin/env bash
# InternVLA-A1.5 「VQA あり」本学習ランチャー（単 GPU / overlay / robot + VQA 混合）。
#
#   source env_train.sh && source env.vqa.sh      # env.vqa.example.sh を元に作る
#   bash scripts/train_ivla_a15_vqa.sh            # フォアグラウンド
#   bash scripts/tmux_train.sh                    # tmux 常駐（RUN_NAME を合わせること）
#
# これは scripts/train_ivla_a15.sh（VQA なしベースライン）の **コピー + 最小差分** で
# ある。ベースライン / _train_common.sh / env_train.sh / env.example.sh は 1 バイトも
# 触らない（計画 A3）。ベースラインとの差分は 3 点だけ:
#   1. RUN_NAME の既定を ivla_a15_libero_combined_vqa に（OUT_DIR / LOG_FILE が別になる）
#   2. _assert_vqa_data（prep 済み jsonl の存在チェック）をローカル定義して呼ぶ
#   3. ARGS に --vqa_dataset.* 群を追加（type/repo_id/root/weight/seed/render_*）
# 共通ヘルパー（_train_common.sh）は source して再利用する（無改変）。
# --policy.enable_vqa_loss=true はベースラインと同一（mixed collate はこれで有効。V8）。
#
# VQA データが無い場合は _assert_vqa_data で即失敗する。先に:
#   IVLA_VQA_ENABLE=1 bash scripts/provision_data.sh
#
# 注意: 上流 lerobot_train.py には **gradient accumulation が無い**
#   （update_policy() が毎バッチ optimizer.step() する）。実効バッチ = batch_size。
#   よって IVLA_BS は probe（scripts/probe_ivla_bs.sh）で載る最大値を採る。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

# ★ 差分 1: RUN_NAME の既定を VQA あり用に（ベースラインは ivla_a15_libero_combined）
: "${RUN_NAME:=${IVLA_VQA_RUN_NAME:-ivla_a15_libero_combined_vqa}}"
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

# ★ 差分 2a: VQA サブセット関連の既定（env.vqa.sh が設定。ベースラインには無い）
: "${IVLA_VQA_ROOT:=${HOME}/data/robointer_vqa}"
: "${IVLA_VQA_REPO_ID:=${IVLA_VQA_ROOT}/lerobot_vqa/all.jsonl}"
: "${IVLA_VQA_WEIGHT:=0.10}"                 # U4: 本走はこの 1 値のみ。A/B しない
: "${IVLA_VQA_RENDER_TARGET:=128}"
: "${IVLA_VQA_RENDER_P_BOX:=${IVLA_RENDER_P_BOX}}"          # 既定は robot 側と同一
: "${IVLA_VQA_RENDER_P_TRIANGLE:=${IVLA_RENDER_P_TRIANGLE}}"
: "${IVLA_VQA_RENDER_P_CUBIC:=${IVLA_RENDER_P_CUBIC}}"
: "${IVLA_VQA_RENDER_P_NEAREST:=${IVLA_RENDER_P_NEAREST}}"

# ★ 差分 2b: prep 済み VQA jsonl の存在チェック（_train_common.sh は無改変なのでローカル定義）
_assert_vqa_data() {
    local first
    first="$(echo "$IVLA_VQA_REPO_ID" | awk '{print $1}')"
    if [ ! -e "$first" ]; then
        echo "VQA jsonl が無い: $first" >&2
        echo "  IVLA_VQA_ENABLE=1 bash scripts/provision_data.sh を先に実行すること" >&2
        exit 1
    fi
}

_activate_conda
_resolve_paths
_assert_env
_assert_vqa_data                    # ★ 差分 2b: ベースラインには無い呼び出し

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

    # ---- ★ 差分 3: VQA データセット（overlay: RenderDownsampleFn 入り。ベースラインには無い）----
    --vqa_dataset.type=internvla_a1_5_parc
    --vqa_dataset.repo_id="$IVLA_VQA_REPO_ID"
    --vqa_dataset.root=""
    --vqa_dataset.weight="$IVLA_VQA_WEIGHT"
    --vqa_dataset.seed="$IVLA_SEED"
    --vqa_dataset.render_downsample=true
    --vqa_dataset.render_target="$IVLA_VQA_RENDER_TARGET"
    --vqa_dataset.render_p_box="$IVLA_VQA_RENDER_P_BOX"
    --vqa_dataset.render_p_triangle="$IVLA_VQA_RENDER_P_TRIANGLE"
    --vqa_dataset.render_p_cubic="$IVLA_VQA_RENDER_P_CUBIC"
    --vqa_dataset.render_p_nearest="$IVLA_VQA_RENDER_P_NEAREST"

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
echo "  vqa       : $IVLA_VQA_REPO_ID  (weight=$IVLA_VQA_WEIGHT render_target=$IVLA_VQA_RENDER_TARGET)"
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
