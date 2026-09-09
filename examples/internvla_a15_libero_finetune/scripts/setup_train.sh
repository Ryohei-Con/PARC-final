#!/usr/bin/env bash
# InternVLA-A1.5 学習環境の構築（計画 P2-13/14、移植手順書 §3）。
#
#   bash scripts/setup_train.sh      # このディレクトリ（examples/internvla_a15_libero_finetune）から
#   source env_train.sh              # 以後、学習を回すシェルで毎回 source する
#
# ここで作るのは **学習専用の conda env（Python 3.11）** であり、採点・提出前チェック用の
# venv（リポジトリルートの setup.sh が作る Python 3.10 のもの）とは**別物**である。
# 手順書 §2 の設計原則: 学習と推論を同じ環境で動かそうとしない。
#   - 学習: Python 3.11 conda / 上流を pip install -e してそのまま使う
#   - 推論: Python 3.10 venv / vendor した src/lerobot を sys.path から import
#
# やること:
#   1. conda の確保（無ければ Miniconda を入れる。$HOME/miniconda3 を決め打ちしない = 手順書 §3.4）
#   2. conda env（Python 3.11）の作成 + ffmpeg/svt-av1
#   3. 上流 InternVLA-A-series の clone（既にあれば HEAD を照合するだけ。**編集しない**）
#   4. torch / torchvision / transformers / 上流本体（-e）のインストール
#   5. torchcodec を torch に合わせて**明示ピン**（手順書 §3.1。緩い依存を上書きする）
#   6. transformers 差し替え（Qwen3.5 / pi0 / pi05 の models を site-packages へコピー）
#   7. オプショナル依存（attention kernel）— 失敗しても止めない（手順書 §3.3）
#   8. env_train.sh の生成
#   9. 検証 — **ここが本体**。「import が通る」で満足しない:
#        - GPU で実際に行列積を回す（sm_120 のカーネルがあるか）
#        - transformers 差し替えがバイト一致で入っているか
#        - overlay（lerobot_policy_parc）が登録されるか
#        - **torchcodec が実データの mp4 を非ゼロにデコードするか + PNG 保存**（手順書 §3.1 / R4）
#
# 環境変数で上書きできるもの:
#   IVLA_REPO / IVLA_CONDA_ENV / IVLA_CONDA_ROOT / IVLA_PY / HF_HOME / IVLA_DATASET_ROOT
#   SKIP_OPTIONAL_KERNELS=1  で 7 をスキップ
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IVLA_DIR="$PWD"

# 秘密情報（HF_TOKEN / WANDB_API_KEY）は git 管理外の ~/.env から読む。
ENV_FILE="${ENV_FILE:-${HOME}/.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

: "${IVLA_REPO:=${HOME}/InternVLA-A-series}"
: "${IVLA_CONDA_ENV:=internvla_a1_5}"
: "${IVLA_PY:=3.11}"
# 重み・データセットは大きいので、既定を大容量ディスク（${HOME}/data）に置く。
: "${IVLA_DATA_ROOT:=${HOME}/data}"
: "${HF_HOME:=${IVLA_DATA_ROOT}/hf}"
: "${IVLA_DATASET_ROOT:=${IVLA_DATA_ROOT}/libero_combined_20hz}"
# 出力（チェックポイント）はホーム側。データセット + 重みと同じディスクを食い合わせない。
: "${IVLA_OUTPUT_DIR:=${HOME}/ivla-a15-outputs}"
: "${IVLA_LOG_ROOT:=${HOME}/ivla-a15-logs}"

# 上流が検証している組み合わせ（tutorials/installation.md: Python 3.11 / CUDA 12.8 / torch 2.10.0）。
# ここを動かすときは torchcodec の対応表も必ず一緒に動かすこと。
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
# ★ 上流の手順は cu128 だが、**この GPU では cu128 ビルドが使えない。**
#   RTX PRO 6000 Blackwell Server Edition（sm_120 / driver 595 系 = CUDA 13.2）で
#   torch 2.10.0+cu128 を入れると、`torch.cuda.is_available()` は True・fp32 の matmul も
#   通るのに、**fp16 / bf16 の GEMM だけ** `CUBLAS_STATUS_INVALID_VALUE` で落ちる
#   （cu128 同梱の cuBLAS がこの SKU の tensor core パスを持たない）。学習は
#   `--policy.dtype=bfloat16` なので致命的である。
#   torch の版は上流検証どおり 2.10.0 のまま、CUDA だけ 13 系ビルドに上げる。
#   これは配布環境の setup.sh が採点用 venv に cu130 を入れているのとも整合する。
#   scripts/verify_train_env.py の C2 が実際に bf16 の行列積を回してここを毎回検証する。
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.2.0}"
# ★ torch 2.10.x に対応する torchcodec は 0.10.x（手順書 §3.1）。
#   上流 pyproject は `torchcodec>=0.2.1` としか書いておらず、pip install -e は
#   PyPI 最新（torch>=2.11 必須）を引いてしまう。ABI が合わないと **import は通るのに**
#   動画デコードだけが例外→ゼロ埋め画像になり、学習が進んで見えるままカメラが黒くなる。
TORCHCODEC_SPEC="${TORCHCODEC_SPEC:-torchcodec==0.10.*}"
# torch が引く 13.1.0.3 は sm_120 で cuBLASLt が壊れている（4/9 のコメント参照）。
NVIDIA_CUBLAS_SPEC="${NVIDIA_CUBLAS_SPEC:-nvidia-cublas>=13.6.1.10}"
UPSTREAM_REF="${UPSTREAM_REF:-e6fc904f9edbfb14532e97095fc2372202517f76}"

say() { echo "[setup-train] $*"; }

# ---------------------------------------------------------------------------
say "1/9 conda の確保"
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$IVLA_DIR/scripts/_train_common.sh"

if ! CONDA_SH="$(_ivla_find_conda_sh)"; then
    CONDA_ROOT="${IVLA_CONDA_ROOT:-${HOME}/miniforge3}"
    if [ "${INSTALL_CONDA:-1}" != "1" ]; then
        say "ERROR: conda が見つからない。INSTALL_CONDA=1 で自動導入するか、"
        say "       IVLA_CONDA_ROOT=/path/to/conda を指定すること。" >&2
        exit 1
    fi
    # **Miniconda ではなく Miniforge を入れる。** Miniconda の既定チャンネル
    # （repo.anaconda.com/pkgs/main）は Terms of Service の同意を要求し、非対話の
    # conda create がそこで止まる。加えて商用利用のライセンス条件が付く。Miniforge は
    # conda-forge のみを既定チャンネルに持ち、どちらの問題も無い。上流の手順書が要求
    # しているのは「Python 3.11 の conda env」であってディストリビューションではない。
    say "  conda が無いので Miniforge を導入する -> $CONDA_ROOT"
    installer="$(mktemp -t miniforge-XXXXXX.sh)"
    curl -fsSL --retry 5 -o "$installer" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
    CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
CONDA_ROOT="$(cd "$(dirname "$(dirname "$CONDA_SH")")/.." && pwd)"
say "  conda=$(conda --version) root=$CONDA_ROOT"

# ---------------------------------------------------------------------------
say "2/9 conda env（Python $IVLA_PY）+ ffmpeg/svt-av1"
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$IVLA_CONDA_ENV"; then
    say "  既存の env を使う: $IVLA_CONDA_ENV"
else
    conda create -y -q --override-channels -c conda-forge -n "$IVLA_CONDA_ENV" "python=$IVLA_PY"
fi
conda activate "$IVLA_CONDA_ENV"
ACTUAL_PY="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$ACTUAL_PY" = "$IVLA_PY" ] || { say "ERROR: env の Python が $ACTUAL_PY（$IVLA_PY のはず）" >&2; exit 1; }
say "  $(python --version) @ $CONDA_PREFIX"

# 学習データが AV1 の場合、torchcodec は FFmpeg の共有ライブラリを dlopen する。
# apt の ffmpeg では版が合わないことがあるので conda-forge で env の中に入れる。
if [ -x "$CONDA_PREFIX/bin/ffmpeg" ]; then
    say "  ffmpeg 導入済み: $("$CONDA_PREFIX/bin/ffmpeg" -version 2>/dev/null | head -1)"
else
    conda install -y -q --override-channels -c conda-forge ffmpeg svt-av1
    say "  $("$CONDA_PREFIX/bin/ffmpeg" -version 2>/dev/null | head -1)"
fi

# ---------------------------------------------------------------------------
say "3/9 上流 InternVLA-A-series（読み取り専用で使う。**1 バイトも編集しない**）"
# ---------------------------------------------------------------------------
if [ ! -d "$IVLA_REPO/.git" ]; then
    git clone --filter=blob:none https://github.com/InternRobotics/InternVLA-A-series.git "$IVLA_REPO"
fi
HEAD_SHA="$(git -C "$IVLA_REPO" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$UPSTREAM_REF" ]; then
    say "  警告: 上流 HEAD が想定と違う"
    say "    想定: $UPSTREAM_REF"
    say "    実際: $HEAD_SHA"
    say "  overlay は lerobot_policy_parc/upstream_provenance.json のハッシュで照合する。"
    say "  9/9 の検証で MISMATCH が出たら、差分を確認してから先に進むこと。"
else
    say "  HEAD=$HEAD_SHA（想定どおり）"
fi
if [ -n "$(git -C "$IVLA_REPO" status --porcelain)" ]; then
    say "  警告: 上流の作業ツリーが clean でない。overlay 方針（計画 §3）に反する変更が無いか確認すること。" >&2
    git -C "$IVLA_REPO" status --short >&2
fi

# ---------------------------------------------------------------------------
say "4/9 torch $TORCH_VERSION / torchvision $TORCHVISION_VERSION / transformers $TRANSFORMERS_VERSION / 上流本体"
# ---------------------------------------------------------------------------
pip install -q --upgrade pip setuptools wheel
# 既存 env に別 CUDA 系の torch が入っている場合、nvidia wheel は同じパス
# （site-packages/nvidia/*/lib）へ上書きされるので、cu12 と cu13 が同居すると
# torch が読む cuDNN / cuBLAS が化ける。入れ替える前に古い系統を必ず消す。
# `|| true` が必須。クリーンな env では grep がヒットせず exit 1 を返し、
# pipefail + set -e でスクリプトごと静かに死ぬ。
_stale_nvidia="$(pip freeze | grep -oE '^nvidia-[a-z0-9-]+-cu1[23]' | sort -u | tr '\n' ' ' || true)"
if [ -n "$_stale_nvidia" ]; then
    say "  既存の nvidia wheel を除去してから入れ直す: $(echo "$_stale_nvidia" | wc -w) 個"
    # shellcheck disable=SC2086
    pip uninstall -y -q torch torchvision $_stale_nvidia >/dev/null 2>&1 || true
fi
pip install -q --timeout 300 --retries 10 \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
# 差し替え版 modeling_qwen3_5.py が transformers の内部 API を直接叩くので、版は完全固定する。
pip install -q --timeout 120 "transformers==$TRANSFORMERS_VERSION"
# 上流本体。internvla extra は flash-linear-attention を引くが、無くても純 torch に
# フォールバックする（手順書 §3.3）ので、ここでは素の依存だけ入れる。
pip install -q --timeout 300 --retries 5 -e "$IVLA_REPO"
pip install -q "peft>=0.14,<0.20"
# ninja が無いと torch の CUDAExtension は **直列コンパイル**にフォールバックする。
# flash-attn のビルドが数十分で終わるか数時間かかるかがこれで決まる（実測で並列度 1 に
# なっていた）。7/9 のビルドより前に必ず入れておくこと。
pip install -q ninja


# ---------------------------------------------------------------------------
say "5/9 torchcodec の ABI ピン（$TORCHCODEC_SPEC）"
# ---------------------------------------------------------------------------
# **必ず `pip install -e` の後**に実行する。先に入れても -e が最新版で上書きしてしまう。
pip install -q --timeout 120 --force-reinstall --no-deps "$TORCHCODEC_SPEC"
say "  torchcodec=$(python -c 'import torchcodec; print(torchcodec.__version__)' 2>/dev/null || echo '??')"

# ---------------------------------------------------------------------------
say "6/9 transformers 差し替え（Qwen3.5 / pi0 / pi05）"
# ---------------------------------------------------------------------------
TRANSFORMERS_DIR="$(python -c 'import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)')"
for policy in pi0 pi05 internvla_a1_5; do
    src="$IVLA_REPO/src/lerobot/policies/${policy}/transformers_replace/models"
    if [ -d "$src" ]; then
        cp -r "$src" "$TRANSFORMERS_DIR"
        say "  $policy -> $TRANSFORMERS_DIR/models"
    else
        say "  警告: $src が無い（上流の版を確認すること）" >&2
    fi
done

# ---------------------------------------------------------------------------
say "7/9 オプショナル依存（attention kernel）"
# ---------------------------------------------------------------------------
# ビルドが必要で環境依存に失敗する。無くても純 torch 実装にフォールバックする
# （Gated DeltaNet が数倍遅くなるだけ）ので、失敗を致命的にしない（手順書 §3.3）。
# 3 つとも性格が違う。**flash-attn だけは「あれば速い」ではなく必須**である。
#
#   flash-linear-attention : triton ベースで即入る。Gated DeltaNet の速度に効く
#                            （計画 R6）。無くても純 torch にフォールバックする。
#   flash-attn             : **動画ヘッド（--policy.action_loss_only=false）では必須。**
#                            WAN の DiT は wan/modules/model.py:249 が
#                            wan/modules/attention.py の flash_attention() を直接呼び、
#                            その中は `assert FLASH_ATTN_2_AVAILABLE` で始まる。同じ
#                            ファイルの attention() には sdpa フォールバックがあるが、
#                            model.py はそちらを通らない。無いと最初の video_loss 計算で
#                            AssertionError になる（実測）。
#                            CUDA カーネルをソースからビルドするので 30〜60 分かかる。
#   causal-conv1d          : 無くても純 torch にフォールバックする。ビルドが長いので既定で入れない。
#
# IVLA_ACTION_LOSS_ONLY=true（動画ヘッドを切る）なら flash-attn は不要。
if [ "${SKIP_OPTIONAL_KERNELS:-0}" = "1" ]; then
    say "  SKIP_OPTIONAL_KERNELS=1 のためスキップ"
else
    set +e
    timeout 900 pip install --timeout 300 "flash-linear-attention==0.5.0" \
        >/tmp/ivla_optional_kernels.log 2>&1
    fla_rc=$?
    [ $fla_rc -eq 0 ] && say "  flash-linear-attention: OK" \
                      || say "  flash-linear-attention: 失敗（純 torch にフォールバック。/tmp/ivla_optional_kernels.log）"

    # flash-attn: 動画ヘッドを使うなら必須なので既定でビルドする。
    if [ "${IVLA_ACTION_LOSS_ONLY:-false}" = "true" ]; then
        say "  flash-attn: スキップ（IVLA_ACTION_LOSS_ONLY=true なので WAN 分岐を使わない）"
    elif python -c "import flash_attn" 2>/dev/null; then
        say "  flash-attn: 導入済み"
    else
        # FLASH_ATTN_CUDA_ARCHS の既定は "80;90;100;120" で 4 アーキテクチャ分を
        # コンパイルする。この機体は sm_120 の 1 枚だけなので 120 に絞る。
        # コンパイル対象が 1/4 になり、ビルド時間もおおよそ 1/4 になる。
        # 別 GPU に env を持ち回るなら既定に戻すこと。
        say "  flash-attn をビルドする（**動画ヘッドに必須**。sm_120 のみ。十数分〜）"
        FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}" \
        MAX_JOBS="${MAX_JOBS:-$(nproc)}" timeout "${IVLA_KERNEL_BUILD_TIMEOUT:-7200}" \
            pip install --no-build-isolation "flash-attn==2.8.3" \
            >>/tmp/ivla_optional_kernels.log 2>&1
        if [ $? -eq 0 ]; then
            say "  flash-attn: OK"
        else
            say "  flash-attn: **失敗**。このままでは動画ヘッドの学習が AssertionError で落ちる。" >&2
            say "     ログ: /tmp/ivla_optional_kernels.log" >&2
            say "     回避するなら IVLA_ACTION_LOSS_ONLY=true（動画教師と foresight token の学習を失う）" >&2
        fi
    fi

    # causal-conv1d は任意。
    if [ "${IVLA_BUILD_CUDA_KERNELS:-0}" = "1" ]; then
        say "  causal-conv1d をビルドする"
        timeout "${IVLA_KERNEL_BUILD_TIMEOUT:-7200}" pip install --no-build-isolation \
            "causal-conv1d==1.6.1" >>/tmp/ivla_optional_kernels.log 2>&1
        [ $? -eq 0 ] && say "  causal-conv1d: OK" || say "  causal-conv1d: 失敗（純 torch にフォールバック）"
    else
        say "  causal-conv1d: スキップ（任意。IVLA_BUILD_CUDA_KERNELS=1 で入る）"
    fi
    set -e
fi

# ---------------------------------------------------------------------------
say "7.5/9 cuBLAS のピン（**この位置でなければならない**）"
# ---------------------------------------------------------------------------
# torch 2.10.0+cu130 が引く nvidia-cublas 13.1.0.3 は、この GPU
# （sm_120 / RTX PRO 6000 Blackwell Server Edition）で **cuBLASLt の経路だけ**が壊れている:
#     torch.nn.functional.linear(x, w, bias)  -> CUBLAS_STATUS_NOT_INITIALIZED
#         （cublasLtMatmulAlgoGetHeuristic で失敗）
#     x @ w.T / linear(x, w)（bias 無し）      -> 正常
# bias 付き Linear は VLM の qkv などモデル中に無数にあるので、学習は最初の forward で
# 落ちる。メモリは 94GiB 空いていて OOM ではない。13.6.1.10 に上げると解消する。
# （DISABLE_ADDMM_CUDA_LT=1 でも回避できるが、cuBLASLt の epilogue fusion を捨てることに
#  なるのでライブラリを直す方を採る。）
#
# ★ **必ず flash-attn より後に実行すること。** flash-attn は torch を依存に持ち、
#   torch は nvidia-cublas==13.1.0.3 をピンしている。先に上げておいても
#   `pip install flash-attn` が 13.6.1.10 を uninstall して 13.1.0.3 に巻き戻す
#   （実測でそうなった）。この 1 行の位置がそのまま学習の可否を決める。
pip install -q --no-deps "$NVIDIA_CUBLAS_SPEC"
say "  nvidia-cublas=$(pip show nvidia-cublas 2>/dev/null | awk '/^Version:/{print $2}')"

# 実際に入った版を記録する（再現用。版の確定はこのファイルを見ること）。
pip freeze --exclude-editable > "$IVLA_DIR/requirements-train.lock.txt"
say "  依存を requirements-train.lock.txt に記録"

# ---------------------------------------------------------------------------
say "8/9 env_train.sh の生成"
# ---------------------------------------------------------------------------
# env.example.sh の確定値をここに焼く。パスは実際に構築した場所を書く。
sed -e "s|^export IVLA_REPO=.*|export IVLA_REPO=\"$IVLA_REPO\"|" \
    -e "s|^export IVLA_CONDA_ENV=.*|export IVLA_CONDA_ENV=\"$IVLA_CONDA_ENV\"|" \
    -e "s|^export IVLA_DATASET_ROOT=.*|export IVLA_DATASET_ROOT=\"$IVLA_DATASET_ROOT\"|" \
    -e "s|^export IVLA_OUTPUT_DIR=.*|export IVLA_OUTPUT_DIR=\"$IVLA_OUTPUT_DIR\"|" \
    -e "s|^export IVLA_PRETRAINED_PATH=.*|export IVLA_PRETRAINED_PATH=\"$HF_HOME/InternVLA-A1.5-base\"|" \
    -e "s|^export HF_HOME=.*|export HF_HOME=\"$HF_HOME\"|" \
    env.example.sh > env_train.sh
cat >> env_train.sh <<EOF

# ---------------------------------------------------------------------------
# 6. setup_train.sh が確定させた値（自動生成。手で編集せず setup_train.sh を再実行する）
# ---------------------------------------------------------------------------
# conda は毎回 source し直す。**bash script 内の conda activate はサブシェルにしか
# 効かない**ので、後続スクリプトも _train_common.sh の _activate_conda を使うこと。
source "$CONDA_SH"
conda activate "$IVLA_CONDA_ENV"

export IVLA_CONDA_ROOT="$CONDA_ROOT"
export IVLA_LOG_ROOT="$IVLA_LOG_ROOT"
export HF_LEROBOT_HOME="\${HF_HOME}/lerobot"
# 上流 launch script が \$CONDA_PREFIX/lib を LD_LIBRARY_PATH に足しているのと同じ。
# conda-forge の ffmpeg 共有ライブラリを torchcodec が dlopen できるようにする。
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# 秘密情報は git 管理外の ~/.env から読む（このファイルには書かない）。
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
EOF
say "  $IVLA_DIR/env_train.sh"

# ---------------------------------------------------------------------------
say "9/9 検証"
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export IVLA_REPO IVLA_DATASET_ROOT
IVLA_DIR="$IVLA_DIR" python "$IVLA_DIR/scripts/verify_train_env.py"

cat <<EOF

[setup-train] 完了。次の手順:

  source env_train.sh
  bash scripts/fetch_weights.sh            # ベース重み / Qwen3.5-2B / WAN（動画ヘッド用）
  python scripts/inspect_dataset.py --root "$IVLA_DATASET_ROOT" \\
      --out artifacts/facts/facts.json     # 事実確定（P2-16。最優先）

学習用 conda env : $IVLA_CONDA_ENV（Python $IVLA_PY）
上流リポジトリ   : $IVLA_REPO（読み取り専用）
HF_HOME          : $HF_HOME
データセット     : $IVLA_DATASET_ROOT
出力             : $IVLA_OUTPUT_DIR

採点・提出前チェックはルートの setup.sh + env.sh（Python 3.10 venv）側で行うこと。
EOF
