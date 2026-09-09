# InternVLA-A1.5 「VQA ありファインチューニング」用の追加 env（雛形 / 計画 §5.6）
#
#   cp env.vqa.example.sh env.vqa.sh && $EDITOR env.vqa.sh
#   source env_train.sh && source env.vqa.sh
#
# **env.example.sh / env_train.sh は 1 文字も触らない。** VQA 専用の変数だけを
# この別ファイルに置き、ベースライン env の後に追加 source する運用にする。
#
# ここで有効化されるのは:
#   - scripts/provision_data.sh の「4/4 VQA」ブロック（IVLA_VQA_ENABLE=1 のとき）
#   - scripts/fetch_vqa_data.sh / scripts/prepare_vqa_data.py のサブセット指定
#   - scripts/train_ivla_a15_vqa.sh の --vqa_dataset.* 群

# --- provision / fetch / prepare -------------------------------------------
export IVLA_VQA_ENABLE=1
export IVLA_VQA_ROOT="${HOME}/data/robointer_vqa"
export IVLA_VQA_HF_REPO="InternRobotics/RoboInter-VQA"

# U2: Understanding + Task_planning の 2 カテゴリ（Generation は使わない）。空白区切り。
export IVLA_VQA_CATEGORIES="Understanding Task_planning"

# カテゴリ合計のサブセット上限（既定は均等割り = 20000 / 20000）。
# VQADataset は実行時に絞れない（V1）ので prep 時に小さい jsonl を書く。
export IVLA_VQA_MAX_SAMPLES=40000

# fetch がこの合計サイズ（GiB）を超える取得予定なら 1 バイトも取らず失敗する。
# $HOME/data はエフェメラルで既に libero + 重みで ~50GB 使用済み。
export IVLA_VQA_MAX_GIB=15

# 画像 zip の include パターン（未指定なら "robotinter/<cat>/image/*.zip"）。
# fetch のファイル一覧ステップで zip 粒度を見てから必要な zip だけに絞れる。
# export IVLA_VQA_IMAGE_INCLUDE="robotinter/Understanding/image/*.zip robotinter/Task_planning/image/part-0*.zip"

# 展開後に zip を消さず残す場合は 1。
# export IVLA_VQA_KEEP_ZIP=1

# --- train ---------------------------------------------------------------
# prep 出力（train が --vqa_dataset.repo_id に渡す）。
# U2: 2 カテゴリを 1 本に連結する（prepare_vqa_data.py --merge-into all.jsonl）。
# MultiVQADataset は per-dataset weight を持たない（V4 / R6）ので連結が必要。
export IVLA_VQA_REPO_ID="${IVLA_VQA_ROOT}/lerobot_vqa/all.jsonl"

# robot : VQA の混合比。U4: 本走はこの 1 値のみ。A/B はしない。
# 0.05 / 0.15 に上書きすれば別値でも回せるが計画上は回さない。
export IVLA_VQA_WEIGHT=0.10

# RenderDownsampleFn(VQA) の目標解像度（採点環境の native 128 に合わせる）。
export IVLA_VQA_RENDER_TARGET=128

# ダウンサンプルのカーネル確率。**ベースラインと同じ既定**（box/triangle/cubic/nearest）。
# A/B する場合のみここで上書きする。
# export IVLA_VQA_RENDER_P_BOX=0.45
# export IVLA_VQA_RENDER_P_TRIANGLE=0.30
# export IVLA_VQA_RENDER_P_CUBIC=0.15
# export IVLA_VQA_RENDER_P_NEAREST=0.10

# VQA あり run の名前（OUT_DIR / LOG_FILE が自動で別になる。ベースラインを汚さない）。
export IVLA_VQA_RUN_NAME="ivla_a15_libero_combined_vqa"
