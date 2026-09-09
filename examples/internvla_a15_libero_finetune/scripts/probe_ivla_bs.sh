#!/usr/bin/env bash
# batch size probe（計画 P4-23 / §8.1）。
#
#   source env_train.sh
#   bash scripts/probe_ivla_bs.sh                       # 既定ラダー
#   IVLA_PROBE_BS="1 2 4" IVLA_PROBE_GC="false" bash scripts/probe_ivla_bs.sh
#
# **probe の第 1 の目的は「WAN 分岐（action_loss_only=false）が 96GB に載るか」の判定**
# である（計画 R1）。載らない場合のメモリ削減ラダーは §8.1:
#   1. batch_size を下げる（上流 lerobot_train.py に gradient accumulation は無いので、
#      実効バッチ = batch_size。ここが小さいと step あたりの情報量がそのまま減る）
#   2. --policy.gradient_checkpointing=true（スループット -25〜35%）
#   3. --policy.action_loss_only=true で WAN 分岐ごと切る（動画教師と foresight token の
#      学習を失うので最後の手段。採用したら probe_report.md に理由を書くこと）
#   4. --num_workers を下げる（VRAM ではなく host RAM 対策）
#
# 出力: artifacts/probe/probe_report.md（BS / GC / peak VRAM / sec/step / OOM の表）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

# 実測の見積り: 学習可能パラメータは vlm 2.22B + other 0.47B = 2.69B。
# 固定費 ≈ 重み bf16 5.4GB + AdamW の状態 fp32 21.6GB + 凍結 WAN bf16 10GB ≈ 37GB。
# 96GB のうち活性化に使えるのは 55GB 程度なので、1 や 2 から始めても情報が薄い。
# 既定は 2 から始めて上に伸ばす。全部 OK なら IVLA_PROBE_BS を上へ足すこと。
: "${IVLA_PROBE_BS:=2 4 8 16}"
# gradient_checkpointing=true 側は、false が全部 OOM のときだけ意味がある（§8.1 の 2）。
# 既定では false のみを見て時間を節約する。両方見たいときは "false true" を渡す。
: "${IVLA_PROBE_GC:=false}"
: "${IVLA_PROBE_STEPS:=30}"
REPORT_DIR="$IVLA_DIR/artifacts/probe"
REPORT="$REPORT_DIR/probe_report.md"
mkdir -p "$REPORT_DIR"

_activate_conda

{
    echo "# probe_report — InternVLA-A1.5 batch size probe"
    echo
    echo "- 日時: $(date -Is)"
    echo "- GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "- steps/条件: ${IVLA_PROBE_STEPS}（sec/step は step 10 以降の中央値。最初の数 step は"
    echo "  cudnn ベンチマークとコンパイルで遅いので除く）"
    echo "- action_loss_only=${IVLA_ACTION_LOSS_ONLY:-false}（false = WAN 動画ヘッド on）"
    echo "- vlm_lr_scale=${IVLA_VLM_LR_SCALE:-0.1}"
    echo
    echo "| batch_size | gradient_checkpointing | peak VRAM (MiB) | sec/step | 結果 |"
    echo "|---:|:---|---:|---:|:---|"
} > "$REPORT"

for gc in $IVLA_PROBE_GC; do
  for bs in $IVLA_PROBE_BS; do
    run="probe_bs${bs}_gc${gc}"
    echo "=== $run ==="
    log="$IVLA_LOG_ROOT/$run.log"
    rm -rf "${IVLA_OUTPUT_DIR:?}/$run" "$log"

    # OOM は失敗として記録し、次の設定へ進む（§8.1）。
    set +e
    RUN_NAME="$run" \
    IVLA_BS="$bs" \
    IVLA_GRADIENT_CHECKPOINTING="$gc" \
    IVLA_STEPS="$IVLA_PROBE_STEPS" \
    IVLA_LOG_FREQ=1 \
    IVLA_SAVE_CHECKPOINT=false \
    IVLA_SAVE_FREQ=1000000 \
        bash "$IVLA_DIR/scripts/train_ivla_a15.sh" >/dev/null 2>&1
    rc=$?
    set -e

    vram_file="$IVLA_LOG_ROOT/$run.vram.csv"
    peak=$(awk -F',' 'NR>1 {if ($2+0 > max) max=$2+0} END {print (max+0)}' "$vram_file" 2>/dev/null || echo 0)
    # lerobot のメトリクス行から updt_s を拾い、step 10 以降の中央値を取る。
    # `|| true` が必須。grep がヒットしないと exit 1 を返し、pipefail + set -e で
    # 代入ごとスクリプトが落ちる（OOM した設定でまさにヒットしない）。
    secstep=$(grep -oE 'updt_s:[0-9.]+' "$log" 2>/dev/null | sed 's/updt_s://' \
        | awk 'NR>10' | sort -n \
        | awk '{a[NR]=$1} END {if (NR) printf "%.3f", (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2); else print "n/a"}' || true)
    [ -z "$secstep" ] && secstep="n/a"

    if [ "$rc" -eq 0 ]; then
        result="OK"
    elif grep -qiE "out of memory|CUDA out of memory" "$log" 2>/dev/null; then
        result="**OOM**"
    else
        result="失敗 (exit=$rc)"
    fi
    echo "| $bs | $gc | $peak | $secstep | $result |" >> "$REPORT"
    echo "  -> $result  peak=${peak}MiB  sec/step=$secstep"
    sleep 5   # GPU メモリの解放を待つ
  done
done

{
    echo
    echo "## 読み方"
    echo
    echo "- **OK の中で最大の batch_size** を本走に採る（実効バッチ = batch_size。"
    echo "  上流に gradient accumulation が無いため）。"
    echo "- gradient_checkpointing=true 側は、false が全部 OOM のときだけ使う。"
    echo "- 全部 OOM なら \`IVLA_ACTION_LOSS_ONLY=true\` で WAN 分岐を切る（計画 §8.1 の 3）。"
    echo "  その場合、動画教師と foresight token の学習を失う。採用理由をここに追記すること。"
    echo "- steps=30000 での所要時間 = sec/step × 30000 / 3600 時間。"
} >> "$REPORT"

echo
echo "=== probe 完了: $REPORT ==="
cat "$REPORT"
