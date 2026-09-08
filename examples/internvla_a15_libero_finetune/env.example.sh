# InternVLA-A1.5 追加学習レシピの環境変数（雛形）
#
#   cp env.example.sh env_train.sh && $EDITOR env_train.sh && source env_train.sh
#
# env_train.sh は .gitignore 対象にする（パスや token が入るため）。

# ---------------------------------------------------------------------------
# 1. パス
# ---------------------------------------------------------------------------

# 上流リポジトリ（InternVLA-A-series）。**読み取り専用で使う。**
# このレシピは overlay 方式で、上流を 1 バイトも書き換えない。
export IVLA_REPO="${HOME}/InternVLA-A-series"

# 学習用 conda env（評価用 venv とは別に作る）
export IVLA_CONDA_ENV="internvla_a1_5"

# データセット（scripts/extract_dataset.sh で展開したもの）
#   bash ../../scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar
export IVLA_DATASET_ROOT="${HOME}/data/libero_combined_20hz"
export IVLA_DATASET_REPO_ID="${IVLA_DATASET_ROOT}"

# 学習の出力先
export IVLA_OUTPUT_DIR="${HOME}/ivla-a15-outputs"
export RUN_NAME="ivla_a15_libero_combined"

# ベース重みと VLM
export IVLA_PRETRAINED_PATH="${HOME}/hf/InternVLA-A1.5-base"
export IVLA_VLM_MODEL_PATH="Qwen/Qwen3.5-2B"

# Hugging Face のキャッシュ。学習前にキャッシュしておき、本走は
# HF_HUB_OFFLINE=1 で再現性を確認する（計画 R5）。
export HF_HOME="${HOME}/hf"
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# 2. robot_type
# ---------------------------------------------------------------------------
# overlay が登録する robot_type。scripts/prepare_dataset.py が
# meta/info.json をこの値に書き換える（計画 P3-19）。
export IVLA_ROBOT_TYPE="libero_combined"

# ---------------------------------------------------------------------------
# 3. 学習ハイパーパラメータ（確定値 / 計画 §8.2）
# ---------------------------------------------------------------------------
# steps はユーザー指定で 30,000。warmup と decay_steps は **実 step 数に
# 再スケール済み**である（計画 F12）。上流 launch script の
# warmup=2000 / decay_steps=100000 は 100k step 前提の値なので、
# そのままコピーすると warmup が全体の 6.7% を食い、cosine は最初の 3/10 しか
# 進まずに終わる。
export IVLA_STEPS=30000
export IVLA_WARMUP_STEPS=600            # 上流比率 2000/100000 = 2% を 30k に再スケール
export IVLA_DECAY_STEPS=30000           # steps と一致させる
export IVLA_LR=5e-5                     # 上流どおり
export IVLA_DECAY_LR=5e-6               # 上流どおり
export IVLA_SAVE_FREQ=5000              # steps // 6 = 5,000（5〜6 個残る）
export IVLA_LOG_FREQ=50                 # 単 GPU なので上流の 200 より細かく

# 実効バッチ 32〜64 を狙う。**BS と grad accum は probe（P4-23）後に確定する。**
export IVLA_BS=""
export IVLA_GRAD_ACCUM=""
export IVLA_NUM_WORKERS=8
export IVLA_GRADIENT_CHECKPOINTING=false

# ---------------------------------------------------------------------------
# 4. ダウンサンプル（RenderDownsampleFn）の確率
# ---------------------------------------------------------------------------
# p_nearest は **ユーザー指定で 0.10 固定**。残りは A/B の対象（計画 D1）。
export IVLA_RENDER_TARGET=128
export IVLA_RENDER_P_BOX=0.45
export IVLA_RENDER_P_TRIANGLE=0.30
export IVLA_RENDER_P_CUBIC=0.15
export IVLA_RENDER_P_NEAREST=0.10

# ---------------------------------------------------------------------------
# 5. W&B（任意）
# ---------------------------------------------------------------------------
export WANDB_PROJECT="parc2026-ivla-a15"
export WANDB_MODE="offline"
# export WANDB_API_KEY=...
