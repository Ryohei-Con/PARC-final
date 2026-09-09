#!/usr/bin/env bash
# 学習に必要な重みを HF_HOME に取得する（計画 R5: 事前キャッシュ → 本走は HF_HUB_OFFLINE=1）。
#
#   source env_train.sh
#   bash scripts/fetch_weights.sh
#
# 取得するもの:
#   1. InternRobotics/InternVLA-A1.5-base  (~5.0 GiB)  ベース重み
#   2. Qwen/Qwen3.5-2B                     (~4.3 GiB)  VLM。A1.5 が実行時に FAST トークンを足す
#   3. physical-intelligence/fast          (~数 MB)    FAST action tokenizer
#   4. Wan-AI/Wan2.2-TI2V-5B               (~22 GiB)   動画ヘッド（action_loss_only=false のとき必須）
#
# 4 は **T5 テキストエンコーダ（models_t5_umt5-xxl-enc-bf16.pth, 10.8 GiB）を取らない。**
# InternVLA の動画分岐は条件付けを learnable_to_wan_proj 経由で行い、T5 を一切 import
# しない（src/lerobot/policies/internvla_a1_5/ に umt5/t5 の参照が無いことを確認済み）。
#
# ★ サイレント失敗の注意 ★
# wan_model.py:165-168 の from_pretrained は、チェックポイントの読み込みに失敗すると
#   `logger.warning("Using random initialization instead")`
# で**握り潰してランダム初期化のまま学習を続ける**。学習は普通に進むが動画教師が無意味に
# なる。本スクリプトは必要ファイルの存在を最後に検証するが、本走のログでもこの warning が
# 出ていないことを必ず確認すること。
#
# 環境変数: HF_HOME / SKIP_WAN=1（動画ヘッドを使わない場合）/ HF_TOKEN（不要だが尊重する）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"

: "${HF_HOME:=${HOME}/data/hf}"
export HF_HOME
mkdir -p "$HF_HOME/hub"

say() { echo "[fetch-weights] $*"; }

# huggingface_hub は上流の依存として入っている。conda env が有効でなければ有効化する。
if ! python -c "import huggingface_hub" 2>/dev/null; then
    # shellcheck disable=SC1091
    source "$IVLA_DIR/scripts/_train_common.sh"
    _activate_conda
fi

# 空き容量の確認。途中で落ちると中途半端なキャッシュが残って原因が分かりにくくなる。
need_gib=$([ "${SKIP_WAN:-0}" = "1" ] && echo 12 || echo 34)
avail_gib=$(( $(df --output=avail -k "$HF_HOME" | tail -n1) / 1048576 ))
say "HF_HOME=$HF_HOME 空き=${avail_gib}GiB 必要≈${need_gib}GiB"
[ "$avail_gib" -ge "$need_gib" ] || { say "ERROR: 空き容量が足りない" >&2; exit 1; }

dl() {  # <repo_id> <local_dir|-> [allow_patterns...]
    local repo="$1" dest="$2"; shift 2
    local args=(--repo-type model)
    [ "$dest" != "-" ] && args+=(--local-dir "$dest")
    [ $# -gt 0 ] && args+=(--include "$@")
    say "取得: $repo ${dest#-}"
    python - "$repo" "$dest" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest, *patterns = sys.argv[1:]
kwargs = {"repo_id": repo, "repo_type": "model", "max_workers": 8}
if dest != "-":
    kwargs["local_dir"] = dest
if patterns:
    kwargs["allow_patterns"] = patterns
path = snapshot_download(**kwargs)
print(f"  -> {path}")
PY
}

# 1. ベース重み。launch script は repo id を直接渡せるが、本走を HF_HUB_OFFLINE=1 で
#    回せるよう（R5）ローカルディレクトリに落として env_train.sh からそこを指す。
dl "InternRobotics/InternVLA-A1.5-base" "$HF_HOME/InternVLA-A1.5-base"

# 2. VLM。repo id のままで使うので通常のキャッシュに入れる。
dl "Qwen/Qwen3.5-2B" "-"

# 3. FAST action tokenizer（--dataset.use_fast_action_tokens=true で使う）。
#    transformers>=5 はオフライン時に repo id 解決へ失敗するので、上流 launch script の
#    コメントどおり ${HF_HOME}/hub/physical-intelligence/fast にファイルとしても置く。
dl "physical-intelligence/fast" "$HF_HOME/hub/physical-intelligence/fast"

# 4. WAN。T5 と assets/examples は取らない。
if [ "${SKIP_WAN:-0}" = "1" ]; then
    say "SKIP_WAN=1 のため WAN をスキップ（--policy.action_loss_only=true で回すこと）"
else
    dl "Wan-AI/Wan2.2-TI2V-5B" "$HF_HOME/hub/Wan2.2-TI2V-5B" \
        "config.json" "diffusion_pytorch_model*" "Wan2.2_VAE.pth"
fi

# ---------------------------------------------------------------------------
say "検証"
# ---------------------------------------------------------------------------
fail=0
chk() { if [ -e "$1" ]; then say "  OK   $1"; else say "  FAIL 見つからない: $1" >&2; fail=1; fi; }

chk "$HF_HOME/InternVLA-A1.5-base/config.json"
chk "$HF_HOME/InternVLA-A1.5-base/model.safetensors"
chk "$HF_HOME/InternVLA-A1.5-base/stats.json"
# train_config.json は **ベース重みには同梱されていない**（実測: config.json /
# model.safetensors / stats.json / README.md のみ）。--policy.pretrained_path は
# config.json しか読まないので不要。train_config.json が要るのは学習の resume
# （--config_path）とエクスポートした提出用 checkpoint の側だけ。
[ -f "$HF_HOME/InternVLA-A1.5-base/train_config.json" ] \
    && say "  OK   train_config.json（任意）" \
    || say "  --   train_config.json は無い（ベース重みには同梱されない。想定どおり）"
chk "$HF_HOME/hub/physical-intelligence/fast/tokenizer.json"
if [ "${SKIP_WAN:-0}" != "1" ]; then
    # configuration_internvla_a1_5.py:332-334 が既定で見るパスと完全に一致させる。
    chk "$HF_HOME/hub/Wan2.2-TI2V-5B/config.json"
    chk "$HF_HOME/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    chk "$HF_HOME/hub/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json"
fi
python -c "
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
print('[fetch-weights]   OK   Qwen/Qwen3.5-2B のトークナイザをキャッシュから構築できた')
" || fail=1

[ "$fail" = 0 ] || { say "ERROR: 取得に失敗したものがある" >&2; exit 1; }

say "完了。使用量:"
du -sh "$HF_HOME" 2>/dev/null || true
