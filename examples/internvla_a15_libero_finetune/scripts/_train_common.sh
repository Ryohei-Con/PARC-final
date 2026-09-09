# InternVLA-A1.5 の学習スクリプトが source する共通ヘルパー。
#
# 提供する関数:
#   _ivla_find_conda_sh   conda.sh を候補から探す（$HOME/miniconda3 を決め打ちしない。手順書 §3.4）
#   _activate_conda       学習用 conda env を有効化する
#   _resolve_paths        $RUN_NAME から OUT_DIR / LOG_FILE / VRAM_FILE を決める
#   _wandb_args           WANDB_API_KEY があれば W&B を有効にする引数を組み立てる
#   _maybe_resume         $OUT_DIR/checkpoints/last から resume、無ければ出力先を wipe
#   _start_vram_sampler   nvidia-smi のバックグラウンドサンプラ + EXIT トラップ
#   _summarize_run S E    ウォールタイムとピーク VRAM を出力する
#   _assert_env           学習に必要なパス（上流 repo / データセット / 重み）の存在確認
#
# 呼び出し側は _resolve_paths の前に RUN_NAME を設定すること。
# 既定値はすべて env で上書きできる（env_train.sh が設定する）。

: "${IVLA_REPO:=${HOME}/InternVLA-A-series}"
: "${IVLA_CONDA_ENV:=internvla_a1_5}"
: "${IVLA_DATASET_ROOT:=${HOME}/data/libero_combined_20hz}"
: "${IVLA_OUTPUT_DIR:=${HOME}/ivla-a15-outputs}"
: "${IVLA_LOG_ROOT:=${HOME}/ivla-a15-logs}"
: "${HF_HOME:=${HOME}/hf}"

# このファイルの 1 つ上（= IVLA_DIR）。train_entry.py / lerobot_policy_parc の置き場。
IVLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# conda
# ---------------------------------------------------------------------------

# conda.sh の場所を返す。**$HOME/miniconda3 を決め打ちしない**（手順書 §3.4）。
# AWS Deep Learning AMI は /opt/conda、miniforge は $HOME/miniforge3 に入る。
_ivla_find_conda_sh() {
    local candidates=()
    # 1. 既に conda が有効なシェルなら CONDA_EXE から辿るのが最も確実
    [ -n "${CONDA_EXE:-}" ] && candidates+=("$(dirname "$(dirname "$CONDA_EXE")")")
    [ -n "${_CONDA_ROOT:-}" ] && candidates+=("$_CONDA_ROOT")
    [ -n "${IVLA_CONDA_ROOT:-}" ] && candidates+=("$IVLA_CONDA_ROOT")
    candidates+=("$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"
                 "/opt/conda" "/opt/miniconda3" "/usr/local/miniconda3")
    # 2. 最後の手段として conda 本体に聞く
    if command -v conda >/dev/null 2>&1; then
        candidates+=("$(conda info --base 2>/dev/null || true)")
    fi
    local root
    for root in "${candidates[@]}"; do
        [ -n "$root" ] && [ -f "$root/etc/profile.d/conda.sh" ] && { echo "$root/etc/profile.d/conda.sh"; return 0; }
    done
    return 1
}

# 学習用 conda env を有効化する。
# **`bash setup_train.sh` 内の conda activate はサブシェルにしか効かない**（手順書 §3.4）ので、
# 後続スクリプトは毎回これを呼んで自分で activate し直すこと。
_activate_conda() {
    local conda_sh
    if ! conda_sh="$(_ivla_find_conda_sh)"; then
        echo "conda が見つからない。scripts/setup_train.sh を実行するか、" >&2
        echo "IVLA_CONDA_ROOT=/path/to/conda を設定すること。" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$conda_sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$IVLA_CONDA_ENV"; then
        echo "conda env が無い: $IVLA_CONDA_ENV" >&2
        echo "scripts/setup_train.sh を実行すること。" >&2
        exit 1
    fi
    conda activate "$IVLA_CONDA_ENV"
}

# ---------------------------------------------------------------------------
# パス / 実行
# ---------------------------------------------------------------------------

_resolve_paths() {
    OUT_DIR="$IVLA_OUTPUT_DIR/$RUN_NAME"
    LOG_FILE="$IVLA_LOG_ROOT/$RUN_NAME.log"
    VRAM_FILE="$IVLA_LOG_ROOT/$RUN_NAME.vram.csv"
    mkdir -p "$IVLA_LOG_ROOT" "$IVLA_OUTPUT_DIR"
}

# 学習に必要なものが揃っているかを、学習を起動する前に落とす。
# 30k step の本走が「10 分後に重みが無くて落ちる」のを防ぐ。
_assert_env() {
    local missing=0
    if [ ! -d "$IVLA_REPO/src/lerobot" ]; then
        echo "上流リポジトリが無い: $IVLA_REPO" >&2; missing=1
    fi
    if [ ! -f "$IVLA_DATASET_ROOT/meta/info.json" ]; then
        echo "データセットが無い: $IVLA_DATASET_ROOT/meta/info.json" >&2
        echo "  bash ../../scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar" >&2
        missing=1
    fi
    # action_loss_only=false（動画ヘッド on）では WAN の重みが要る。
    # modeling_internvla_a1_5.py:576-584 が config.wan_checkpoint_path から構築する。
    if [ "${IVLA_ACTION_LOSS_ONLY:-false}" != "true" ] && [ ! -f "$HF_HOME/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ]; then
        echo "WAN の重みが無い: $HF_HOME/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" >&2
        echo "  bash scripts/fetch_weights.sh  （または IVLA_ACTION_LOSS_ONLY=true で動画ヘッドを切る）" >&2
        missing=1
    fi
    [ "$missing" = 0 ] || exit 1
}

# W&B は既定で offline。WANDB_API_KEY があるときだけ online にできる。
_wandb_args() {
    if [ "${WANDB_MODE:-offline}" = "offline" ] || [ -z "${WANDB_API_KEY:-}" ]; then
        WANDB_ARGS=(--wandb.enable=true --wandb.project="${WANDB_PROJECT:-parc2026-ivla-a15}" --wandb.mode=offline)
    else
        WANDB_ARGS=(--wandb.enable=true --wandb.project="${WANDB_PROJECT:-parc2026-ivla-a15}" --wandb.mode=online)
    fi
}

_maybe_resume() {
    RESUME_ARGS=()
    if [ -d "$OUT_DIR/checkpoints/last/pretrained_model" ]; then
        echo "==> $OUT_DIR/checkpoints/last から resume する"
        RESUME_ARGS=(--resume=true --config_path="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json")
    elif [ -d "$OUT_DIR" ]; then
        echo "==> 新規実行のため、既存の出力ディレクトリを削除する: $OUT_DIR"
        rm -rf "$OUT_DIR"
    fi
}

_SAMPLER_PID=""
_sampler_cleanup() {
    [ -n "$_SAMPLER_PID" ] && kill "$_SAMPLER_PID" 2>/dev/null || true
    _janitor_cleanup
}

_start_vram_sampler() {
    trap _sampler_cleanup EXIT
    (
        echo "ts,used_MiB,util_%" > "$VRAM_FILE"
        while true; do
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
                | tr -d ' ' | awk -v ts="$(date +%s)" -F',' '{print ts","$1","$2}' >> "$VRAM_FILE"
            sleep 5
        done
    ) &
    _SAMPLER_PID=$!
}

# --- チェックポイントの掃除 --------------------------------------------------
# 1 チェックポイント = pretrained_model 5.1GB + training_state 11GB = 16GB（実測）。
# steps=30000 / save_freq=5000 なら 6 個で 96GB になり、/home の空き（約 85GB）に
# 入らない。**学習が終盤で「保存できずに落ちる」のが一番損失が大きい**（計画 R8）。
#
# training_state（optimizer / scheduler / RNG）は **resume にしか使わない**ので、
# 最新の 1 個だけ残せばよい。評価と提出に要るのは pretrained_model の方だけ。
# これで 6 個 = 5×5.1 + 16 ≈ 42GB に収まる。
#
# `last` は最新チェックポイントへの symlink なので、その実体は必ず残す。
_prune_checkpoint_states() {
    local ckpt_root="$OUT_DIR/checkpoints"
    [ -d "$ckpt_root" ] || return 0
    local newest
    newest="$(readlink -f "$ckpt_root/last" 2>/dev/null || true)"
    if [ -z "$newest" ]; then
        # last が無ければ名前順で最後のものを最新とみなす
        newest="$(find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)"
    fi
    local dir
    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        [ "$(readlink -f "$dir")" = "$newest" ] && continue
        if [ -d "$dir/training_state" ]; then
            echo "[janitor] $(basename "$dir")/training_state を削除（resume には最新のみ必要）"
            rm -rf "$dir/training_state"
        fi
    done < <(find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d)
}

_JANITOR_PID=""
_janitor_cleanup() {
    [ -n "$_JANITOR_PID" ] && kill "$_JANITOR_PID" 2>/dev/null || true
}

# 学習中ずっと回して、保存されるたびに古い training_state を落とす。
_start_checkpoint_janitor() {
    (
        while true; do
            sleep 300
            _prune_checkpoint_states
        done
    ) &
    _JANITOR_PID=$!
}

_summarize_run() {
    local start=$1 end=$2
    echo
    echo "=== $RUN_NAME :: total_wall_sec=$((end - start)) ==="
    if [ -f "$VRAM_FILE" ]; then
        local peak
        peak=$(awk -F',' 'NR>1 {if ($2+0 > max) max=$2+0} END {print max+0}' "$VRAM_FILE")
        echo "=== peak VRAM=$peak MiB ==="
    fi
}
