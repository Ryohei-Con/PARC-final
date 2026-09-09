#!/usr/bin/env bash
# 学習を tmux セッションの中で起動する（ssh が切れても学習を止めないため）。
#
#   bash scripts/tmux_train.sh              # 起動（既に動いていれば何もしない）
#   bash scripts/tmux_train.sh attach       # 画面に接続（抜けるのは Ctrl-b d）
#   bash scripts/tmux_train.sh status       # 生存確認 + 直近ログ
#   bash scripts/tmux_train.sh logs         # ログを追尾（Ctrl-C で追尾だけ止める）
#   bash scripts/tmux_train.sh stop         # 学習を止める
#
# **なぜ tmux か**: ssh が切れると、そのシェルの子プロセスは SIGHUP で死ぬ。
# tmux のサーバープロセスは ssh セッションから切り離されているので、接続が切れても
# 中のプロセスは動き続け、再接続して `attach` すれば同じ画面に戻れる。
# nohup でも「死なない」だけは達成できるが、対話的に様子を見て止める手段が無い。
#
# 学習本体は scripts/train_ivla_a15.sh。中断からの再開（checkpoints/last）は
# そちらの _maybe_resume が自動で判定するので、落ちたら同じコマンドで再実行すればよい。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"

: "${IVLA_TMUX_SESSION:=ivla-a15}"
: "${RUN_NAME:=ivla_a15_libero_combined}"
: "${IVLA_LOG_ROOT:=${HOME}/ivla-a15-logs}"
LOG_FILE="$IVLA_LOG_ROOT/$RUN_NAME.log"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux が無い。次で導入すること:" >&2
    echo "  apt-get install -y tmux   （root でない場合は sudo を付ける）" >&2
    exit 1
fi

_alive() { tmux has-session -t "$IVLA_TMUX_SESSION" 2>/dev/null; }

case "${1:-start}" in
start)
    if _alive; then
        echo "既に動いている: tmux セッション '$IVLA_TMUX_SESSION'"
        echo "  接続: bash scripts/tmux_train.sh attach"
        exit 0
    fi
    mkdir -p "$IVLA_LOG_ROOT"
    # env_train.sh を tmux の中で source する（tmux は親シェルの env を継がない場合がある）。
    # `exec bash` で終わらせず、終了コードを画面に残してから待つ。落ちた理由を
    # attach して読めるようにするため。
    tmux new-session -d -s "$IVLA_TMUX_SESSION" -c "$IVLA_DIR" \
        "source '$IVLA_DIR/env_train.sh'; \
         RUN_NAME='$RUN_NAME' bash '$IVLA_DIR/scripts/train_ivla_a15.sh'; \
         rc=\$?; echo; echo '=== 学習プロセス終了 (exit='\$rc') ==='; \
         echo 'この画面は残してある。抜けるには Ctrl-b d、閉じるには exit。'; \
         exec bash"
    echo "起動した: tmux セッション '$IVLA_TMUX_SESSION'"
    echo "  接続  : bash scripts/tmux_train.sh attach"
    echo "  ログ  : tail -f $LOG_FILE"
    ;;
attach)
    _alive || { echo "セッション '$IVLA_TMUX_SESSION' は動いていない" >&2; exit 1; }
    exec tmux attach -t "$IVLA_TMUX_SESSION"
    ;;
status)
    if _alive; then
        echo "セッション '$IVLA_TMUX_SESSION': 動作中"
        tmux list-panes -t "$IVLA_TMUX_SESSION" -F '  pane #{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}'
    else
        echo "セッション '$IVLA_TMUX_SESSION': 動いていない"
    fi
    echo "--- GPU ---"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv || true
    echo "--- ログ末尾 ($LOG_FILE) ---"
    [ -f "$LOG_FILE" ] && tail -n 15 "$LOG_FILE" || echo "  （まだ無い）"
    ;;
logs)
    [ -f "$LOG_FILE" ] || { echo "ログがまだ無い: $LOG_FILE" >&2; exit 1; }
    exec tail -f "$LOG_FILE"
    ;;
stop)
    _alive || { echo "動いていない"; exit 0; }
    tmux send-keys -t "$IVLA_TMUX_SESSION" C-c
    sleep 5
    tmux kill-session -t "$IVLA_TMUX_SESSION" 2>/dev/null || true
    echo "停止した。checkpoints/last があれば同じコマンドで再開できる。"
    ;;
*)
    echo "使い方: bash scripts/tmux_train.sh [start|attach|status|logs|stop]" >&2
    exit 2
    ;;
esac
