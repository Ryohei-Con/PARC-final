#!/usr/bin/env bash
# 再起動から学習を復旧する（1 コマンド）。
#
#   bash scripts/resume_after_reboot.sh
#
# **この機体は再起動が起きる。** 実測で 3 回起きており、そのたびに:
#   - $HOME/data（データセット + HF 重み）が空になる（エフェメラル領域）
#   - tmux セッションが消え、学習プロセスが死ぬ
# 一方、**$HOME 側は残る**:
#   - $HOME/ivla-a15-outputs（チェックポイント）… **これが残るのが肝心**
#   - $HOME/ivla-a15-logs、$HOME/miniforge3、$HOME/InternVLA-A-series
#
# よって復旧は「データを入れ直して、checkpoints/last から再開する」だけでよい。
# scripts/train_ivla_a15.sh の _maybe_resume が last を見て自動で --resume=true を付ける。
#
# 所要時間の実測: データセット約 2 分 + 重み約 5 分 + モデルロード約 2 分。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"

echo "[resume] $(date -Is)"
echo "[resume] uptime: $(uptime -p)"

# shellcheck disable=SC1091
source "$IVLA_DIR/env_train.sh"

CKPT="$IVLA_OUTPUT_DIR/${RUN_NAME:-ivla_a15_libero_combined}/checkpoints/last"
if [ -d "$CKPT/training_state" ]; then
    step="$(python -c "
import json,sys
try:
    print(json.load(open('$CKPT/training_state/training_step.json')).get('step','?'))
except Exception:
    print('?')
" 2>/dev/null || echo '?')"
    echo "[resume] 再開元: $CKPT（step=$step）"
elif [ -d "$CKPT" ]; then
    echo "[resume] 警告: $CKPT に training_state が無い。resume できず最初からになる。" >&2
    echo "[resume]        チェックポイント整理が最新の training_state まで消していないか確認すること。" >&2
else
    echo "[resume] チェックポイントが無い。新規実行として始める。"
fi

# @reboot の cron から呼ばれると、ネットワークがまだ上がっていないことがある。
# 重みの取得は HF への通信が要るので、疎通するまで少し待つ（最大 5 分）。
for _ in $(seq 1 30); do
    if curl -fsS --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then break; fi
    echo "[resume] ネットワーク待ち..."
    sleep 10
done

echo "[resume] 1/2 データと重みを入れ直す"
bash "$IVLA_DIR/scripts/provision_data.sh"

echo "[resume] 2/2 学習を tmux で起動する"
bash "$IVLA_DIR/scripts/tmux_train.sh" start

echo
echo "[resume] 完了。確認:"
echo "  bash scripts/tmux_train.sh status"
echo "  bash scripts/tmux_train.sh logs"
