#!/usr/bin/env bash
# VQA あり run の 20 step smoke（計画 §5.8 / SC-V4）。**これが PASS するまで本走しない。**
#
#   source env_train.sh && source env.vqa.sh
#   bash scripts/smoke_ivla_vqa.sh
#
# 確認すること（ベースライン smoke_ivla.sh の全項目 + VQA 固有）:
#   - MixedMultimodalDataset が構築される（factory.py:566）
#   - [make_vqa_dataset] all_repo_ids= に prep 済み jsonl パスが出る
#   - RenderDownsampleFn が VQA 画像キー（observation.images.image0 等）に作用する
#     （"was not hydrated; falling back to key auto-detection" の warning。V2）
#   - loss に nan/inf が無い（robot バッチと VQA バッチの両方が流れて有限）
#   - ベースライン smoke の既存チェック（monkeypatch / optimizer 2 グループ / WAN 重み）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

: "${IVLA_SMOKE_STEPS:=20}"
: "${IVLA_SMOKE_BS:=1}"
: "${IVLA_VQA_ROOT:=${HOME}/data/robointer_vqa}"
: "${IVLA_VQA_SMOKE_OUT:=${IVLA_VQA_ROOT}/lerobot_vqa_smoke}"
: "${IVLA_VQA_SMOKE_SAMPLES:=200}"
: "${IVLA_VQA_SMOKE_CATEGORIES:=Understanding}"

RUN_NAME="${IVLA_SMOKE_RUN_NAME:-smoke_ivla_vqa}"
LOG_FILE="$IVLA_LOG_ROOT/$RUN_NAME.log"
SMOKE_JSONL="$IVLA_VQA_SMOKE_OUT/all.jsonl"

_activate_conda

# --- 極小サブセット（200 件）を prep する（既にあれば再利用）------------------
if [ -f "$SMOKE_JSONL" ]; then
    echo "=== smoke VQA jsonl 再利用: $SMOKE_JSONL ==="
else
    echo "=== smoke VQA サブセットを prep（$IVLA_VQA_SMOKE_SAMPLES 件）==="
    python "$IVLA_DIR/scripts/prepare_vqa_data.py" \
        --src "${IVLA_VQA_ROOT}/raw" \
        --out "$IVLA_VQA_SMOKE_OUT" \
        --categories "$IVLA_VQA_SMOKE_CATEGORIES" \
        --max-samples "$IVLA_VQA_SMOKE_SAMPLES" \
        --merge-into all.jsonl \
        --pad-square
fi

echo "=== smoke: steps=$IVLA_SMOKE_STEPS bs=$IVLA_SMOKE_BS vqa=$SMOKE_JSONL ==="
rm -rf "${IVLA_OUTPUT_DIR:?}/$RUN_NAME"

set +e
RUN_NAME="$RUN_NAME" \
IVLA_VQA_RUN_NAME="$RUN_NAME" \
IVLA_VQA_REPO_ID="$SMOKE_JSONL" \
IVLA_VQA_WEIGHT="${IVLA_VQA_SMOKE_WEIGHT:-0.5}" \
IVLA_BS="$IVLA_SMOKE_BS" \
IVLA_STEPS="$IVLA_SMOKE_STEPS" \
IVLA_SAVE_FREQ="$IVLA_SMOKE_STEPS" \
IVLA_LOG_FREQ=1 \
IVLA_NUM_WORKERS="${IVLA_SMOKE_WORKERS:-2}" \
    bash "$IVLA_DIR/scripts/train_ivla_a15_vqa.sh"
rc=$?
set -e

echo
echo "=== smoke の検査 ==="
fail=0
# `set -e` の下で条件式を直に評価しない（smoke_ivla.sh と同じ理由）。必ず if で受ける。
chk() {  # <条件の終了コード> <説明>
    if [ "$1" = 0 ]; then echo "  OK   $2"; else echo "  FAIL $2" >&2; fail=1; fi
}
chk_cmd() {  # <説明> -- <コマンド...>
    local msg="$1"; shift; shift   # $2 は "--"
    if "$@" >/dev/null 2>&1; then chk 0 "$msg"; else chk 1 "$msg"; fi
}

chk "$([ "$rc" -eq 0 ] && echo 0 || echo 1)" "学習プロセスが exit 0 で終わった"

# --- ベースライン smoke の既存チェック（維持）--------------------------------
chk_cmd "overlay の monkeypatch が適用された"          -- grep -q "training patches:" "$LOG_FILE"
chk_cmd "optimizer が vlm / other の 2 グループに分かれた" -- grep -q "vlm group" "$LOG_FILE"
chk_cmd "WAN の重みがチェックポイントから読めた"        -- grep -q "Successfully loaded WAN weights" "$LOG_FILE"

if grep -q "Using random initialization instead" "$LOG_FILE"; then
    echo "  FAIL WAN の重みが読めずランダム初期化されている（wan_model.py:165-168）" >&2
    fail=1
else
    echo "  OK   WAN のランダム初期化警告は出ていない"
fi

# --- VQA 固有チェック -----------------------------------------------------
if grep -Eq "Mixed dataset created|MixedMultimodalDataset\(total=" "$LOG_FILE"; then
    echo "  OK   MixedMultimodalDataset が構築された"
else
    echo "  FAIL MixedMultimodalDataset のログが無い（VQA データが混ざっていない）" >&2
    fail=1
fi

if grep -q "\[make_vqa_dataset\] all_repo_ids=" "$LOG_FILE"; then
    if grep -q "$SMOKE_JSONL" "$LOG_FILE"; then
        echo "  OK   make_vqa_dataset が prep 済み jsonl を読んだ"
    else
        echo "  FAIL make_vqa_dataset のログに smoke jsonl パスが無い" >&2
        fail=1
    fi
else
    echo "  FAIL [make_vqa_dataset] all_repo_ids= のログが無い" >&2
    fail=1
fi

if grep -q "weight=" "$LOG_FILE" && grep -q "MixedMultimodalDataset(total=" "$LOG_FILE"; then
    echo "  OK   MixedMultimodalDataset の weight ログが出ている"
    grep -A3 "MixedMultimodalDataset(total=" "$LOG_FILE" | head -4 | sed 's/^/       /'
else
    echo "  FAIL MixedMultimodalDataset の weight ログが無い" >&2
    fail=1
fi

if grep -q "RenderDownsampleFn was not hydrated; falling back to key auto-detection" "$LOG_FILE" \
   && grep -q "auto-detection.*image0" "$LOG_FILE"; then
    echo "  OK   RenderDownsampleFn が VQA 画像キー（image0 等）に作用している（V2）"
else
    echo "  FAIL RenderDownsampleFn(VQA) の auto-detection ログが無い" >&2
    fail=1
fi

if grep -qE "loss:(nan|inf|-inf)" "$LOG_FILE"; then
    echo "  FAIL loss が nan/inf" >&2; fail=1
else
    echo "  OK   loss に nan/inf は無い"
fi

ckpt="$IVLA_OUTPUT_DIR/$RUN_NAME/checkpoints"
if [ -d "$ckpt" ]; then
    echo "  OK   チェックポイント保存: $ckpt（$(du -sh "$ckpt" | cut -f1)）"
else
    echo "  FAIL チェックポイントが保存されていない: $ckpt" >&2; fail=1
fi

echo
if [ "$fail" = 0 ]; then
    echo "=== smoke PASS。VQA あり本走を開始してよい: RUN_NAME 未設定で bash scripts/train_ivla_a15_vqa.sh ==="
else
    echo "=== smoke FAIL。ログ: $LOG_FILE ===" >&2
fi
exit "$fail"
