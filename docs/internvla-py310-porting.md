# InternVLA-A1.5 を Python 3.10 の閉じた採点環境で動かす — 移植手順書

**この文書の読み手は Claude Code（またはそれに準ずるコーディングエージェント）である。**
別のリポジトリで InternVLA-A-series 系のモデルを「Python 3.10・外部通信なし・
`requirements.txt` に外部ソース指定不可」という環境へ載せる作業を、この手順で再現できる。

実装のリファレンスは `PARC2026/InternVLA-finetuning/` にある。本文中の
`→ inference/internvla_runtime.py` のような参照はそのツリー内の位置を指す。

---

## 0. 作業を始める前に確定させる入力

移植先で最初に確認する。ここが違うと以降の判断が全部ずれる。

| 変数 | 本キットでの値 | 確認方法 |
|---|---|---|
| 採点環境の Python | 3.10 | 採点側 `Dockerfile` の `python3.10` |
| ポリシーサーバーの起動上限 | 120 秒 | `evaluate.py` の `SERVER_TIMEOUT` |
| 1 リクエストの上限 | 10 秒 | 大会規定 / ハーネスの `/act` タイムアウト |
| requirements の禁止事項 | `git+` / `--index-url` / `-f` 等 | `validate_submission.py` の `BANNED_REQ_OPTIONS` / `BANNED_REQ_SCHEMES` |
| 推論 venv は分離されるか | される（`evaluate.py` が提出物用 venv を作る） | `evaluate.py` の `create_submission_venv` |
| 上流モデルの `requires-python` | `>=3.10` | `InternVLA-A-series/pyproject.toml` |

**最後の行が移植可否を決める。** InternVLA-A-series 本体が 3.10 を許容しているかを
最初に見ること。許容していなければこの手順は使えない。

---

## 1. なぜ素直に動かないのか（3 つの壁）

1. **`pip install lerobot` ができない。**
   上流 LeRobot は v0.5.0 以降 `requires-python >= 3.12`。3.10 の環境には入らない。
2. **`pip install git+https://github.com/InternRobotics/InternVLA-A-series` もできない。**
   採点環境は外部通信が遮断されており、`validate_submission.py` が
   `requirements.txt` 内の外部ソース指定（スキーム `git:` / `https:` / `--index-url` 等）を
   ERROR にする。
3. **`transformers` を素で入れても動かない。**
   InternVLA は `transformers/models/qwen3_5/modeling_qwen3_5.py` の**差し替え版**を
   配布しており（Gated DeltaNet と action expert のフック）、上流の手順では
   site-packages へコピーすることが前提になっている。提出 zip からこれをどうやるかを
   自分で決める必要がある。

**突破口**: InternVLA-A-series の `src/lerobot` は **107 ファイル / 1.9MB のピュア Python**
で、`requires-python` は `>=3.10`。つまり **pip を通さずソースを持ち込めば 3.10 で動く。**

---

## 2. 設計原則 — 学習と推論を同じ環境で動かそうとしない

これが一番重要な判断である。1 つの env で両方を満たそうとすると破綻する。

| 段階 | 環境 | Python | 方針 |
|---|---|---|---|
| 学習 | GPU サーバー / conda env | **3.11**（上流が検証しているバージョン） | InternVLA-A-series を `pip install -e` してそのまま使う |
| 推論 | 採点環境 / venv | **3.10**（固定） | `vendor/lerobot` を同梱し `sys.path` から import |
| 評価 | 採点と同じ Docker | 3.10 | ハーネスと推論を**別 venv**に分けて再現 |

学習側を無理に 3.10 に落とさないこと。上流が検証していない構成でハマると、
それが学習の失敗なのか環境の問題なのか切り分けられなくなる。
→ `env.sh` の `IVLA_PYTHON_VERSION=3.11`

---

## 3. 手順 A — 学習環境（Python 3.11・上流をそのまま使う）

参照実装: `setup/01_setup.sh`

```bash
git clone https://github.com/InternRobotics/InternVLA-A-series.git "${IVLA_REPO}"
conda create -y -n "${IVLA_CONDA_ENV}" python=3.11
conda activate "${IVLA_CONDA_ENV}"

conda install -y -c conda-forge ffmpeg svt-av1        # 学習データが AV1 の場合に必須
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.2.0
pip install -e "${IVLA_REPO}"
pip install "torchcodec==0.10.*"                      # ← 3.1 を必ず読むこと
pip install "peft>=0.14,<0.20"                        # LoRA 学習用（上流は peft 非依存）
```

### 3.1 torchcodec の ABI ピン（サイレント失敗の代表例）

InternVLA-A-series の `pyproject.toml` は `torchcodec>=0.2.1` としか固定していないため、
`pip install -e .` は PyPI 最新版（`torch>=2.11` 必須）を引く。`torch==2.10.0` と ABI が
合わず、ネイティブ `.so` がロードできない。

**厄介なのは `import` は通ってしまうこと。** 起動時には気付けず、学習中に動画を読むたびに
例外 → ゼロ埋め画像、という形でサイレントに失敗し続ける。ログ上は学習が進んで見えるのに、
実際はカメラ入力がずっと黒画像になる。

→ `torch 2.10.x` には `torchcodec==0.10.*` を明示的に固定する。
対応表: https://github.com/pytorch/torchcodec#installing-torchcodec

移植先で `pip install -e` する際は、**上流が緩く固定している C 拡張依存を全部洗い出す**こと。

### 3.2 transformers の差し替え（学習側）

```bash
TRANSFORMERS_DIR="$(python -c 'import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)')"
for policy in pi0 pi05 internvla_a1_5; do
    src="${IVLA_REPO}/src/lerobot/policies/${policy}/transformers_replace/models"
    [ -d "${src}" ] && cp -r "${src}" "${TRANSFORMERS_DIR}"
done
```

### 3.3 オプショナル依存は失敗を致命的にしない

`flash-linear-attention` / `causal-conv1d` / `flash-attn` はビルドが必要で環境依存に失敗する。
無くても純 torch 実装にフォールバックする（数倍遅いだけ）ので、`set +e` で囲んで
警告に留める。

### 3.4 conda の場所を決め打ちしない

`$HOME/miniconda3` を前提にすると AWS Deep Learning AMI（`/opt/conda`）等で落ちる。
`_ivla_find_conda_sh` のように候補を順に探索し、最後に `conda info --base` へ落とす。
→ `setup/_find_conda.sh`

また **`bash setup/01_setup.sh` 内の `conda activate` はサブシェル内でしか効かない。**
後続スクリプトは毎回自分で activate し直す設計にする（`setup/_activate_conda.sh`）。

---

## 4. 手順 B — 推論パッケージ（Python 3.10）

参照実装: `export/05_export_submission.py`（生成側）と `inference/internvla_runtime.py`（実行側）。

### 目標レイアウト

```
submission/
├── policy_server.py          # 提出必須。MyPolicy だけ差し替える
├── requirements.txt          # PyPI のみ・ピン留め
├── internvla_runtime.py      # 3.10 で動かすためのブートストラップ + 推論
├── verify_inference.py       # 提出前の自己チェック
├── runtime_config.json       # 環境変数を渡せないので設定をファイルに焼く
├── MANIFEST.json             # 何をどこから固めたかの記録
├── model_weights/
│   ├── checkpoint/           # config.json / model.safetensors / stats.json / train_config.json
│   └── vlm/                  # ★重み以外だけ（22MB）
└── vendor/
    ├── lerobot/              # src/lerobot をそのまま（107 files / 1.9MB）
    └── transformers_patch/models/qwen3_5/modeling_qwen3_5.py
```

### B1. vendor 同梱と `sys.path` 注入

`src/lerobot` を丸ごとコピーし、`import lerobot` より**前に** `sys.path` の先頭へ入れる。

```python
def install_vendored_lerobot(vendor_dir):
    vendor_dir = Path(vendor_dir).resolve()
    if not (vendor_dir / "lerobot" / "__init__.py").is_file():
        raise FileNotFoundError(...)

    # 既に別の lerobot が import 済みなら黙って上書きせず落とす
    if "lerobot" in sys.modules:
        loaded = Path(getattr(sys.modules["lerobot"], "__file__", "")).resolve()
        if vendor_dir not in loaded.parents:
            raise RuntimeError(f"別の lerobot が既に import 済み: {loaded}")
        return vendor_dir

    path_str = str(vendor_dir)
    while path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
    importlib.invalidate_caches()
    return vendor_dir
```

**「既に import 済みなら落とす」を必ず入れること。** 黙って先勝ちさせると、
どちらの `lerobot` が動いているか分からないまま精度だけ落ちる。

dataset schema の yaml など、pip では入らない設定ファイルも vendor 側へコピーする
（`vendor/lerobot/dataset_schemas/configs/*.yaml`）。

### B2. transformers の差し替え — コピー、駄目なら注入

site-packages に書ける環境（Docker で自分がオーナー）ならコピーが確実。
書けない環境（read-only FS・権限降格）では `sys.modules` へ直接流し込む。

```python
def patch_transformers_qwen3_5(vendor_dir):
    patch_file = Path(vendor_dir) / "transformers_patch/models/qwen3_5/modeling_qwen3_5.py"
    import transformers
    target = Path(transformers.__file__).parent / "models/qwen3_5/modeling_qwen3_5.py"

    if target.is_file() and target.read_bytes() == patch_file.read_bytes():
        return "already-patched"
    # 差し替え対象が import 済みだと手遅れ
    if "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        raise RuntimeError("モデルを import する前に呼んでください")

    try:
        shutil.copyfile(patch_file, target)
        importlib.invalidate_caches()
        return "copied"
    except OSError:
        _inject_qwen35_module(patch_file)   # importlib で spec を作り sys.modules に載せる
        return "injected"


def _inject_qwen35_module(patch_file):
    package_name = "transformers.models.qwen3_5"
    module_name = f"{package_name}.modeling_qwen3_5"
    package = importlib.import_module(package_name)
    spec = importlib.util.spec_from_file_location(module_name, patch_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)   # 半端な状態を残さない
        raise
    setattr(package, "modeling_qwen3_5", module)
```

B1 と B2 を `bootstrap()` に束ね、`MyPolicy.__init__` の先頭で呼ぶ。

Docker を自分で用意できるなら、ビルド時に `chmod -R a+rwX <transformers>/models/qwen3_5`
しておくとコピー経路を確実に通せる（→ `eval/Dockerfile.eval`）。

### B3. requirements.txt は PyPI のピン留めだけ

```
torch==2.10.0
torchvision==0.25.0
transformers==5.2.0     # ← 差し替えファイルが内部 API に依存するので固定必須
accelerate>=1.10.0,<2.0.0
diffusers>=0.27.2,<0.36.0
datasets>=4.0.0,<4.2.0
...
```

`transformers` を範囲指定にしないこと。差し替え版 `modeling_qwen3_5.py` は
`transformers.utils.generic.is_flash_attention_requested` のような**内部 API**を直接呼ぶため、
マイナーバージョンがずれるだけで起動時 `ImportError` になる。

vendor した `lerobot` が import する依存（`av`・`pandas`・`pyarrow`・`draccus`・
`deepdiff` など、推論では使わないが import されるもの）を**漏れなく**書き出すこと。
これは `verify_inference.py` をクリーン venv で回して潰すのが早い（→ 手順 C）。

### B4. VLM の重みを同梱しない（約 6GB 削減）

`InternVLAA15WithExpertModel` は VLM を `from_pretrained` で構築するが、その重みは
**直後にチェックポイントの strict ロードで全て上書きされる**。よって config・トークナイザ・
画像プロセッサだけ同梱し、`from_pretrained` を config 構築に差し替える。

```python
@contextlib.contextmanager
def _vlm_built_from_config_only(vlm_dir):
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from transformers.modeling_utils import no_init_weights

    original = Qwen3_5ForConditionalGeneration.from_pretrained

    def _from_config(pretrained_model_name_or_path=None, *args, **kwargs):
        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(
            pretrained_model_name_or_path or str(vlm_dir))
        with no_init_weights():
            return Qwen3_5ForConditionalGeneration(config)

    Qwen3_5ForConditionalGeneration.from_pretrained = staticmethod(_from_config)
    try:
        yield
    finally:
        Qwen3_5ForConditionalGeneration.from_pretrained = original
```

同梱するのは `*.json` / `*.txt` / `*.model` / `*.jinja` のみ（実測 22MB）。
**`model.safetensors.index.json` は消すこと。** 残すと `from_pretrained` が存在しない
shard を参照しに行く。

また `snapshot_download(allow_patterns=...)` は「何をダウンロードするか」しか制御しない。
学習で既に重みをキャッシュ済みだと snapshot に重みが入っているので、**コピー時にも
同じ patterns で絞る**こと。

等価性は `verify_inference.py --compare-vlm-weights <重み同梱版のvlmディレクトリ>` で確認する。

### B5. tie された重複テンソルを落とす（約 1GB 削減）

Qwen3.5 は入力埋め込みと出力層で重みを共有する（tie）ため、チェックポイントに
同じ 1GB のテンソルが 2 つ入っている。`lm_head.weight` と `embed_tokens.weight` の
データ部を SHA256 で比較し、一致したら safetensors を**ストリーミングで書き直して**除く
（6GB を一度にメモリへ載せない）。

→ `export/05_export_submission.py` の `find_tied_duplicates()` / `copy_safetensors_without()`

安全性は B6 の検証が担保する（本当に必要ならそこで落ちる）。

### B6. `strict=False` のサイレント失敗を自分で検出する

**`PreTrainedPolicy.from_pretrained` の `strict` は既定が False。**
キーが足りなくても例外を投げずログを出すだけなので、

- LoRA をマージし忘れたチェックポイント
- 分岐の有無が食い違う config
- VLM を config から構築したのに重みが一部欠けている

が**ランダム初期化のまま静かに動く**。行動が微妙におかしいだけで気付けない。
起動時に明示的に落とすこと。

```python
def assert_checkpoint_covers_model(policy, checkpoint_file):
    from safetensors import safe_open
    with safe_open(str(checkpoint_file), framework="pt") as f:
        ckpt_keys = set(f.keys())

    state = policy.state_dict()
    # tied weights は保存時に片方が落ちるので、同じストレージを指すキーが
    # チェックポイントにあれば欠損とみなさない
    by_storage = {}
    for name, tensor in state.items():
        by_storage.setdefault(tensor.data_ptr(), []).append(name)

    missing = []
    for name in sorted(set(state) - ckpt_keys):
        if any(twin in ckpt_keys for twin in by_storage.get(state[name].data_ptr(), [])):
            continue
        missing.append(name)
    if missing:
        raise RuntimeError(f"チェックポイントに {len(missing)} 個のパラメータが足りません: {missing[:10]}")
    return len(state)
```

**この関数が B5 の安全網でもある。** 重複削除がやり過ぎならここで落ちる。

### B7. 起動 120 秒制限に間に合わせる

action token を足すため `resize_token_embeddings(target_size)` が呼ばれる。
transformers の既定は `mean_resizing=True` で、既存 embedding の平均と共分散を求めて
多変量正規分布から新しい行をサンプリングする。`embed_tokens` は `[250368, 2048]` あるので
**この計算だけで数十秒**かかる。

しかもその結果は捨てられる。直後に `_init_new_rows()` が上書きし、さらにチェックポイントの
strict ロードで全パラメータが上書きされる。よって `mean_resizing=False` にしても
**最終的な重みは 1 ビットも変わらない。**

```python
@contextlib.contextmanager
def _fast_token_embedding_resize():
    from transformers.modeling_utils import PreTrainedModel
    original = PreTrainedModel.resize_token_embeddings

    def _patched(self, *args, **kwargs):
        kwargs.setdefault("mean_resizing", False)
        return original(self, *args, **kwargs)

    PreTrainedModel.resize_token_embeddings = _patched
    try:
        yield
    finally:
        PreTrainedModel.resize_token_embeddings = original
```

### B8. 設定は環境変数ではなくファイルに焼き込む

**採点環境は環境変数を渡してくれない。** 推論の設定（robot_type、replan_steps、
ノイズの決め方、どのランタイムを使うか）は `runtime_config.json` として提出物に入れる。

```json
{
  "runtime": "standard",
  "env": {
    "IVLA_ROBOT_TYPE": "libero_plus",
    "IVLA_NOISE_MODE": "fixed",
    "IVLA_NOISE_SEED": "42"
  }
}
```

読み込みは **`setdefault`** にする。手元で環境変数を立てればそちらが優先され、
A/B 実験のときに提出物を作り直さずに済む。

```python
def build_submission_runtime(config_path=None):
    config = load_runtime_config(config_path)
    for key, value in (config.get("env") or {}).items():
        os.environ.setdefault(str(key), str(value))
    kind = str(os.environ.get("IVLA_RUNTIME") or config.get("runtime", "standard")).lower()
    ...
```

**`policy_server.py` と `verify_inference.py` の両方をこの 1 関数に通すこと。**
片方だけ別経路にすると、自己チェックが本番と違うものを測る。

### B9. `policy_server.py` は触る範囲を最小にする

テンプレートの `BasePolicy` / シリアライゼーション / FastAPI エンドポイントは変更しない。
`MyPolicy` の中身だけを差し替える。加えて、起動方法（`python policy_server.py` /
`uvicorn policy_server:app` / 別 cwd）に依らず import できるよう自分のディレクトリを
`sys.path` へ入れておく。

```python
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
```

---

## 5. 手順 C — 検証を 3 段階に分ける

**学習用 conda env で通っても意味がない。** 余計なパッケージが入っているので、
requirements.txt の漏れが検出できない。

### C1. 3.10 で構文が通るか（最速・数秒）

```bash
python3.10 -m compileall -q submission/vendor/lerobot
```

`tomllib`・`StrEnum`・`itertools.batched`・`asyncio.TaskGroup`・`datetime.UTC`・`except*` などの
**3.11 以降専用**の機能が混ざっていないかを機械的に確認する（`match` 文は 3.10 で使えるので対象外）。
本キットの vendor（107 ファイル）を下の grep にかけると該当 0 件で、無改変のまま
C3 の python3.10 venv で実際に動いている。

```bash
grep -rnE "tomllib|StrEnum|itertools\.batched|asyncio\.TaskGroup|datetime\.UTC|except\*" \
     submission/vendor/lerobot --include="*.py"
```

### C2. クリーン venv での自己チェック（requirements の漏れを潰す）

```bash
python3.10 -m venv /tmp/subm_venv
/tmp/subm_venv/bin/pip install -r submission/requirements.txt
/tmp/subm_venv/bin/python submission/verify_inference.py --benchmark
```

`verify_inference.py` はダミー観測（ハーネスが `/act` に送るのと同じ shape）を作って
以下を確認する。**実機データもシミュレータも要らない**ので、移植先でも真っ先に用意する。

- vendor 版 lerobot と transformers パッチが実際に有効になっているか（`__file__` を印字）
- モデルのロードが 120 秒以内か
- action が float32 shape (7,) で NaN/Inf を含まないか
- 推論ありステップ / キャッシュ返却ステップそれぞれのレイテンシ（10 秒制限）
- グリッパ次元が `{-1, +1}` に二値化されているか
- エピソード跨ぎの `reset()` が効くか

### C3. 採点環境と同じ Docker で二重 venv を再現

本番では `evaluate.py` が提出物用の venv を別に作る。ここを再現しないと
「ハーネスの CPU torch」と「推論の CUDA torch」が同居して壊れる。

```dockerfile
ARG BASE_IMAGE=parc2026:latest
FROM ${BASE_IMAGE}
# /workspace/venv  … 評価ハーネス（CPU torch）※ベースイメージ側
# /opt/ivla-venv   … ポリシーサーバー（CUDA torch + transformers 5.2.0）
COPY inference/requirements.txt /tmp/ivla-requirements.txt
RUN python3.10 -m venv /opt/ivla-venv \
    && /opt/ivla-venv/bin/pip install --no-cache-dir -r /tmp/ivla-requirements.txt
ENV IVLA_SERVER_PYTHON=/opt/ivla-venv/bin/python
```

→ `eval/Dockerfile.eval`。あわせて次の 2 つを踏むので先に対処しておく。

- コンテナを**呼び出し元 UID** で動かすと、ビルド時に root が `/tmp` へ作ったファイル
  （robosuite の `/tmp/robosuite.log` 等）が原因で `PermissionError` になる。
  `RUN rm -rf /tmp/* /tmp/.[!.]*` しておく。
- LIBERO は `~/.libero/config.yaml` を読む。root の HOME にしか無いので、
  `/etc/libero` へ写して `LIBERO_CONFIG_PATH` で指す。

### C4. ハーネス側の検証

```bash
python validate_submission.py submission            # requirements の外部ソース検査など
python submission/policy_server.py --port 8000 &
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2
```

---

## 6. サイレント失敗カタログ（ここだけは暗記する）

移植で時間を溶かすのは全部この型である。**エラーが出ないのに結果だけ悪くなる。**

| 症状 | 原因 | 検出方法 |
|---|---|---|
| 学習は進むが精度が出ない | torchcodec の ABI 不一致で動画デコードが例外 → ゼロ埋め画像 | デコード結果を 1 枚保存して目視 |
| 学習では効くのに評価だけ精度が出ない | 推論バックエンドの `ResizeImagesWithPadFn` が hydrate されておらず `mapping` が空で no-op。学習 224 に対し推論が生解像度のまま Qwen processor に入り `image_grid_thw` が食い違う | 学習経路と推論経路で `image_grid_thw` を突き合わせる |
| 行動が微妙におかしい | `from_pretrained(strict=False)` で一部がランダム初期化のまま | `assert_checkpoint_covers_model()` |
| 精度が出ない | 別の `lerobot` が先に import されている | `print(Path(lerobot.__file__).parent)` |
| 起動が 120 秒に間に合わない | `resize_token_embeddings` の `mean_resizing=True` | ロード時間を計測して印字 |
| 推論が数倍遅い | `flash-linear-attention` 不在で Gated DeltaNet が純 torch 実装 | `modeling_qwen3_5.is_fast_path_available` |
| ローカルでは動くが本番で落ちる | 環境変数に依存した設定 | 設定を `runtime_config.json` に焼く |
| ローカルでは動くが本番で落ちる | 学習 conda env の余剰パッケージに依存 | クリーン venv で C2 |
| 提出物が無駄に 7GB 大きい | VLM 重みと tie 重複テンソルの同梱 | B4 / B5 |

---

## 7. 移植チェックリスト

移植先で上から順に潰す。

- [ ] 上流の `requires-python` が採点環境の Python 以下であることを確認した
- [ ] 学習（3.11 conda）と推論（3.10 venv）を分けた
- [ ] 上流が緩く固定している C 拡張依存（torchcodec 等）を torch に合わせてピンした
- [ ] `src/<pkg>` を `vendor/` へコピーし、`sys.path` 先頭へ入れた
- [ ] 「別の同名パッケージが import 済みなら落とす」ガードを入れた
- [ ] pip で入らない設定ファイル（dataset schema 等）も vendor へコピーした
- [ ] transformers 差し替えを「コピー → 失敗したら sys.modules 注入」で実装した
- [ ] 差し替え対象を import する**前**にパッチを当てる順序になっている
- [ ] `requirements.txt` は PyPI のみ・`transformers` は完全固定
- [ ] VLM 重みを外し、`index.json` も消した
- [ ] tie された重複テンソルを削り、ストリーミングで書き出した
- [ ] `assert_checkpoint_covers_model()` を起動パスに入れた
- [ ] `mean_resizing=False` で起動時間を短縮した
- [ ] 設定を `runtime_config.json` に焼き、`setdefault` で読んだ
- [ ] `policy_server.py` と `verify_inference.py` が同じビルド関数を通っている
- [ ] `python3.10 -m compileall` が通った
- [ ] クリーン venv で `verify_inference.py --benchmark` が PASS した
- [ ] 採点環境と同じ Docker（二重 venv）で end-to-end が通った
- [ ] `validate_submission.py` が ERROR 0 件

---

## 8. 参照ファイル

| 目的 | ファイル |
|---|---|
| 学習環境の構築 | `setup/01_setup.sh`, `setup/_find_conda.sh`, `env.example.sh` |
| 提出パッケージの生成 | `export/05_export_submission.py` |
| 3.10 ブートストラップ + 推論 | `inference/internvla_runtime.py` |
| 提出必須のサーバー | `inference/policy_server.py` |
| 自己チェック | `inference/verify_inference.py` |
| 二重 venv の Docker | `eval/Dockerfile.eval` |
| 採点側の制約 | `validate_submission.py`, `evaluate.py`, `Dockerfile` |
