#!/usr/bin/env bash
# 20 step の smoke test（計画 P4-25）。**これが通るまで本走を始めない。**
#
#   source env_train.sh
#   bash scripts/smoke_ivla.sh
#
# 確認すること:
#   1. overlay が登録され、monkeypatch（VLM lr 0.1 倍）が適用される
#   2. データセットが読める（torchcodec のデコード込み）
#   3. 前方・後方計算が通り、loss が有限
#   4. チェックポイントが保存でき、サイズが分かる（ディスク見積り。計画 R8）
#   5. 学習ログに WAN のランダム初期化警告が出ていない（wan_model.py:165-168 の
#      `Using random initialization instead` はサイレント失敗）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

: "${IVLA_SMOKE_STEPS:=20}"
: "${IVLA_SMOKE_BS:=1}"
# env_train.sh が RUN_NAME を export しているので `${RUN_NAME:-...}` では拾えない。
# smoke の出力を本走のディレクトリ・ログに混ぜないよう、専用の名前を明示的に使う。
RUN_NAME="${IVLA_SMOKE_RUN_NAME:-smoke_ivla}"
LOG_FILE="$IVLA_LOG_ROOT/$RUN_NAME.log"

echo "=== smoke: steps=$IVLA_SMOKE_STEPS bs=$IVLA_SMOKE_BS ==="
rm -rf "${IVLA_OUTPUT_DIR:?}/$RUN_NAME"

set +e
RUN_NAME="$RUN_NAME" \
IVLA_BS="$IVLA_SMOKE_BS" \
IVLA_STEPS="$IVLA_SMOKE_STEPS" \
IVLA_SAVE_FREQ="$IVLA_SMOKE_STEPS" \
IVLA_LOG_FREQ=1 \
IVLA_NUM_WORKERS="${IVLA_SMOKE_WORKERS:-2}" \
    bash "$IVLA_DIR/scripts/train_ivla_a15.sh"
rc=$?
set -e

echo
echo "=== smoke の検査 ==="
fail=0
# **`cond; chk $?` と書いてはいけない。** `set -e` の下では条件が偽になった時点で
# `[` / `grep` 自体が失敗コマンドとみなされ、FAIL を記録する前にスクリプトごと落ちる
# （実際にそれで検査結果が 1 行も出なかった）。必ず if で受ける。
chk() {  # <条件の終了コード> <説明>
    if [ "$1" = 0 ]; then echo "  OK   $2"; else echo "  FAIL $2" >&2; fail=1; fi
}
chk_cmd() {  # <説明> -- <コマンド...>
    local msg="$1"; shift; shift   # $2 は "--"
    if "$@" >/dev/null 2>&1; then chk 0 "$msg"; else chk 1 "$msg"; fi
}

chk "$([ "$rc" -eq 0 ] && echo 0 || echo 1)" "学習プロセスが exit 0 で終わった"

chk_cmd "overlay の monkeypatch が適用された"          -- grep -q "training patches:" "$LOG_FILE"
chk_cmd "optimizer が vlm / other の 2 グループに分かれた" -- grep -q "vlm group" "$LOG_FILE"
chk_cmd "WAN の重みがチェックポイントから読めた"        -- grep -q "Successfully loaded WAN weights" "$LOG_FILE"

# WAN のランダム初期化はサイレント失敗（学習は進むが動画教師が無意味）。
if grep -q "Using random initialization instead" "$LOG_FILE"; then
    echo "  FAIL WAN の重みが読めずランダム初期化されている（wan_model.py:165-168）" >&2
    grep -n "Failed to load WAN checkpoint" "$LOG_FILE" | head -2 >&2
    fail=1
else
    echo "  OK   WAN のランダム初期化警告は出ていない"
fi

# loss が有限か（nan/inf なら学習が壊れている）
if grep -qE "loss:(nan|inf|-inf)" "$LOG_FILE"; then
    echo "  FAIL loss が nan/inf" >&2; fail=1
else
    echo "  OK   loss に nan/inf は無い"
fi

ckpt="$IVLA_OUTPUT_DIR/$RUN_NAME/checkpoints"
if [ -d "$ckpt" ]; then
    size=$(du -sh "$ckpt" | cut -f1)
    echo "  OK   チェックポイント保存: $ckpt（$size）"
    echo "       -> 本走で 6 個残すなら約 $(du -sm "$ckpt" | cut -f1) MiB x 6 が要る（計画 R8）"
else
    echo "  FAIL チェックポイントが保存されていない: $ckpt" >&2; fail=1
fi

echo
if [ "$fail" = 0 ]; then
    echo "=== smoke PASS。本走を開始してよい: bash scripts/tmux_train.sh ==="
else
    echo "=== smoke FAIL。ログ: $LOG_FILE ===" >&2
fi
exit "$fail"
