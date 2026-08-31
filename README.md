# PARC 2026 — 配布環境

PARC 2026 のための配布環境である。
本リポジトリを用いることで、参加者が実装したポリシーを example タスク上で
実行し、提出前にローカル環境で採点および動作確認を行うことができる。

本環境で実施できる作業は次のとおりである。
- 自身のポリシーを HTTP サーバーとして起動し、Track 1〜3 の example タスクで評価する
- 提出物（zip）を、本番と同一の手順でエンドツーエンドに検証する
- 提出物の妥当性および動作を、提出前に自動チェックする
  （必須ファイルの有無、サーバーが起動して所定の応答を返すこと、
  各リクエストが制限時間内に完了することを確認する）

最初に参照すべきファイルは次のとおりである。
- 提出物の作成方法・動作する最小実装: [submission_template/](submission_template/)
  （`policy_server.py` の `MyPolicy` は編集前でもそのまま動作する）
- 提出前チェック: [validate_submission.py](validate_submission.py)
- 学習の参考例: [examples/](examples/)（提出には必須ではない）

評価パイプラインおよび提出物チェックスクリプトは、本番採点と同じ評価処理・制約を
再現する。ただし、本番評価とは以下の点で異なる。

- 同梱されているのは公開されている example タスクのみである。本番の採点は、
  **公開されていないタスクを含む別のタスクセット**で実施される
- 出力されるのは成功率および軌道メトリクスの生値である。リーダーボードの順位を決定する
  スコア算出設定は含まれない
- 推論タイムアウト（[下記](#タイムアウト仕様)）および成功判定（[下記](#成功判定)）は
  本番と同一である

## 動作要件

本配布環境は、本番の採点環境と同一の **GPU 構成**である。CPU のみの環境はサポートしない。

- NVIDIA GPU（本番採点は NVIDIA L4）
- NVIDIA ドライバ **R580 系以降**（CUDA 13 対応）
- Docker + nvidia-container-toolkit（Docker で使う場合）
- Python 3.10、git、unzip（setup.sh でホストに直接構築する場合）

## 1. セットアップ

本配布環境は **Docker で使うことを基本とする**。リポジトリ同梱の
[Dockerfile](Dockerfile) は本番の採点イメージと同一の依存構成の定義であり、
このイメージ上での動作確認が、採点環境での動作に最も近い確認になる。
イメージが取得・インストールする第三者製ソフトウェアとそのライセンスは
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照すること。

### 演習環境（配布 GPU 環境）で使う

ビルド済みイメージが AMI に配置されているため、`docker build` は不要である。
イメージ名・タグは演習環境の案内を参照すること（以下では `parc2026` とする）。

```bash
# 対話シェルを起動する（~/data を /data として共有する）
docker run -it --rm --gpus all -v ~/data:/data parc2026
```

- コンテナ内は `/workspace` に本リポジトリ一式が展開済み・venv 有効の状態であり、
  [§2](#2-評価を回す) のコマンドがそのまま使える
- 提出 zip・評価結果など、ホストと受け渡すファイルは `/data`（= ホストの `~/data`）に置く
- コンテナは root で動作するため、`/data` に書いたファイルはホスト側で root 所有になる。
  必要なら `sudo chown -R $USER:$USER ~/data/<対象>` で戻すこと

#### 学習データセットの展開

使用するデータセットの tar を
[scripts/extract_dataset.sh](scripts/extract_dataset.sh)（ホスト側で実行）で
`~/dataset/` から `~/data/` 以下に展開してから学習に使用すること。

```bash
bash scripts/extract_dataset.sh                                   # 展開できる tar の一覧を表示
bash scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar  # 指定した tar を ~/data/ に展開
```

ホームディレクトリの容量には限りがあるため、必要なデータセットのみ展開し、
不要になったデータは削除すること。

### 自身の GPU マシンで使う

```bash
docker build -t parc2026 .    # 初回のみ（アセット取得を含む）
docker run -it --rm --gpus all -v <作業ディレクトリ>:/data parc2026
```

### 代替: ホストに直接構築する（setup.sh）

Docker を使わず、ホストの venv に同じ構成を作ることもできる。
[setup.sh](setup.sh) は Dockerfile がビルド時に実行しているものと同一のスクリプトで、
venv、ピン止めした依存、CUDA 13 版 torch、LIBERO-plus の取得とパッチ、
アセットのダウンロードと配線を一括して実行する。

```bash
bash setup.sh     # 初回のみ（アセット取得を含めて 10〜20 分）
source env.sh     # 評価を実行するシェルで毎回実行する
```

> setup.sh は `~/.libero/config.yaml` を上書きする（既存の設定は `.bak` に退避される）。
> 既に LIBERO を使用しており元の設定に戻す場合は、`~/.libero/config.yaml.bak`
> を書き戻すこと。

> この方法では venv の中身は Docker と同一になるが、OS パッケージ層はホスト依存となり、
> 提出 zip のエンドツーエンド検証（[§2](#2-評価を回す) の `evaluate.py`）は
> `/workspace` 前提のため使えない。**提出前の最終確認は Docker で行うこと。**

## 2. 評価を回す

コンテナで 2 つ目のシェルが要る場合は `docker exec -it <コンテナ名> bash` で入る
（`docker ps` でコンテナ名を確認できる）。

```bash
# 1) 自身のポリシーサーバーを起動する（別ターミナル。テンプレートは編集前でも
#    ランダム action を返すので、まずそのまま起動して疎通確認できる）
python submission_template/policy_server.py --port 8000

# 2) 評価を実行する（--track で対象トラックを選ぶ。複数指定可）
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 300
python -m pipeline --server-url http://localhost:8000 --track track1 track2 track3 --n-episodes 2

# タスクを指定して評価する（example タスク名を指定する。存在しない名前は候補一覧つきでエラーとなる）
python -m pipeline --server-url http://localhost:8000 --track track1 --tasks <task_id>

# 提出 zip をエンドツーエンドで検証する（zip 展開 → 依存インストール → 評価まで自動実行）
# ※ Docker 内で実行すること（/workspace 前提。setup.sh のホスト構築では使えない）
python evaluate.py /data/my_submission.zip --n-episodes 2
```

トラックは次の 3 つである。本番の採点はトラックごとに独立して行われる。

| トラック | スイート | 内容 |
|---|---|---|
| track1 | `libero_t1` | 同一タスク同一ドメイン |
| track2 | `libero_t2` | 同一タスク未知ドメイン |
| track3 | `libero_t3` | 既知タスク組み合わせ未知ドメイン |

結果は `results/<submission_id>.json` に出力される。成功率、ステップ数、軌道メトリクス
（経路長、jerk、SPARC 等）の詳細が含まれる。

## 3. 提出前のチェック

提出物の妥当性（必須ファイル、zip 構造、エンドポイント）と、実際に起動して
動作すること（/health→/reset→/act が正常に応答し、応答が制限時間内であること）を検査する。

```bash
python validate_submission.py my_submission.zip            # 静的検査 + 起動スモークテスト
python validate_submission.py my_submission.zip --static   # 静的検査のみ（起動しない）
```

---

## 提出フォーマット

提出物は **HTTP ポリシーサーバー一式の zip** である。サーバーは次の 3 エンドポイントを
実装する。

| エンドポイント | 役割 |
|---|---|
| `GET /health` | 起動確認（200 を返すまで評価側がポーリングする） |
| `POST /reset` | エピソード開始（`instruction`, `seed` を JSON で受け取る） |
| `POST /act` | 観測（msgpack）→ action を返す。**float32 shape (7,)** `[dx, dy, dz, droll, dpitch, dyaw, gripper]` |

## 採点環境

本番の採点は **GPU コンテナ**上で実施される。構成は次のとおりであり、
**本配布環境はこれと同一の依存構成で構築される**。

| 項目 | 値 |
|---|---|
| ベースイメージ | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` |
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA L4 |
| Python | 3.10.12（システム Python、`/usr/bin/python3.10`） |
| CUDA Toolkit | **13.0**（`nvcc` V13.0.88。devel ベースのため `nvcc` によるソースビルドが可能） |
| cuDNN | 9.14.0 |
| NCCL | 2.28.3+cuda13.0 |
| PyTorch | **2.11.0+cu130**（`torch.version.cuda` は `13.0`） |
| Triton | 3.6.0 |
| NVIDIA ドライバ | R580 系 |
| レンダリング | `MUJOCO_GL=EGL`（GPU レンダリング） |

評価パイプライン側の主要な依存は次の版で固定されている。

```
numpy==1.26.4        mujoco==3.7.0       robosuite==1.4.0    gym==0.25.2      bddl==3.6.0
fastapi==0.140.7     uvicorn==0.51.0     msgpack==1.2.1      requests==2.34.2
huggingface_hub==1.25.1   opencv-python-headless==4.11.0.86  scipy==1.15.3   h5py==3.16.0
Pillow==12.3.0       matplotlib==3.10.9  einops==0.8.2       hydra-core==1.3.2
```

評価環境（LIBERO-plus / LIBERO）は次のコミットに固定されている。setup.sh はこのコミットを
取得し、本番の採点イメージも同じコミットを参照する。

| リポジトリ | コミット |
|---|---|
| [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | `4976dc30028e805ff8094b55501d532c48fec182` |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| アセット（`Sylvest/LIBERO-plus` の `assets.zip`） | `dd2bd61b7d9a6fef1abc52d606e983b41886a149` |

プリインストールされているパッケージの全量は[付録](#付録-採点環境のプリインストール一覧)に示す。

### requirements.txt の書き方

提出物の依存は、**`--system-site-packages` 付きで作成された提出物専用の venv** に対して
`pip install -r requirements.txt` される。したがって次のようになる。

- **`requirements.txt` に書かなかったライブラリは、採点イメージにプリインストール
  されている版がそのまま使われる。** `torch` を書かなければ上表の `2.11.0+cu130` が使われ、
  イメージ側の CUDA 13 のライブラリと整合した状態で GPU が利用できる。
  `numpy` / `fastapi` / `uvicorn` / `msgpack` / `huggingface_hub` 等も同様に省略できる。
- **書いた版は venv 側が優先される。** その版は提出物のサーバーにのみ効き、
  評価パイプライン側には影響しない。

### CUDA 12 系の torch を使用する場合の注意

採点イメージは CUDA 13.0 ベースであり、システム側には **`libnvJitLink.so.13`** しか
存在しない（`.so.12` の代替にはならない）。CUDA 12 ビルドの torch を使用する場合、それが必要とする `nvidia-*-cu12` の
wheel が **venv 側に一式そろっている必要がある**。特に `nvidia-nvjitlink-cu12` が
入らないと、サーバー起動時に次のエラーで失敗する。

```
ImportError: libnvJitLink.so.12: cannot open shared object file: No such file or directory
```

`torch` を単に pin しただけであれば依存として自動的に導入されるが、`requirements.txt` 内に
バージョン衝突がある場合、pip の解決結果が変わって導入されないことがある。
**`pip install` 時に `... requires X, but you have Y` という警告を残さないこと。**
CUDA 12 の wheel 一式が venv にそろっていれば、イメージ側が CUDA 13 でも動作する。

**CUDA 12 系の wheel を入れる場合は、torch も CUDA 12 版に揃えること。** `nvidia-*-cu12` と
`nvidia-*-cu13` の wheel は cuDNN や NCCL を同じパスへインストールするため、両方が入ると
後からインストールされた側で上書きされる。片方だけを入れた状態（例: torch はイメージの
`2.11.0+cu130` のまま、`jax[cuda12]` など CUDA 12 系のパッケージを追加する）にすると、
torch が CUDA 12 用の cuDNN を読み、GPU 推論が失敗することがある。この不整合は
`pip check` では検出できない。

### lerobot を使用する場合

lerobot は **0.4.4** を使用すること。0.5.0 以降は Python 3.12 以上を要求し、
採点環境の Python 3.10 には導入できない。

- **学習に使う**: [examples/pi05_libero_finetune/](examples/pi05_libero_finetune/) の
  `scripts/setup_train.sh` が学習専用の venv を作り、lerobot 0.4.4 を導入する。
  **評価用の venv（ルートの setup.sh）とは分けること。** 評価用は採点環境の再現が
  目的であり、学習の依存で構成を変えてはならない。

- **提出物で使う**: `requirements.txt` に `lerobot==0.4.4` と記載する。
  依存が数 GB あるため、初回のインストールには時間がかかる。

**提出物の venv 側の構成が変わる点に注意すること。** lerobot 0.4.4 は
`torch<2.11.0` を要求し、依存する `opencv-python-headless` が `numpy>=2.0` を
要求するため、`requirements.txt` に `lerobot` を書くと提出物の venv には
**CUDA 12 版の torch 2.10.0 と numpy 2.x** が導入される。採点環境側の
`torch 2.11.0+cu130` / `numpy 1.26.4` は使用されない。

この構成でも GPU は使用できる（`nvidia-*-cu12` の wheel 一式が venv 側にそろうため）。
ただし前節の `libnvJitLink.so.12` の注意が該当するため、提出前に必ず次を確認すること。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 例: 2.10.0+cu128 True
```

`transformers` は lerobot 側の制約と組み合わせが変わりやすいため、
バージョンを明示して固定することを推奨する。

なお、lerobot を導入すると `pip install` の最後に次の警告が出るが、これは既知であり
問題ない。`gcsfs` と lerobot が導入する `fsspec` の版がかみ合わないというもので、
GCS を使わない提出物には影響しない。

```
gcsfs ... requires fsspec>=..., but you have fsspec ... which is incompatible.
```

これ以外の `... requires X, but you have Y` が出た場合は、前節のとおり解消すること。

### 提出前の確認

ローカルと本番で torch の構成が異なることに起因する失敗は、次の 1 行で検出できる。
本配布環境で venv を作成し、`pip install -r requirements.txt` を実行したあとに
確認することを推奨する。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 成功判定

本番の採点と同一の基準である。エピソードが成功と扱われるのは、**タスクのゴール条件を
満たし、かつ衝突が発生していない**場合のみである。

衝突は「操作対象以外の物体を動かしたか」で判定する。タスクが操作対象とする物体
（BDDL の `:obj_of_interest`）を除く全物体について、初期位置からの変位（xyz 各軸の
絶対値の和）を各ステップで監視し、その最大値が **1 mm** を超えた物体が 1 つでもあれば、
そのエピソードは失敗となる。

- 対象物体を掴んで動かすことは当然に許容される。判定対象は「それ以外の物体」である
- 変位は環境が落ち着いた時点（エピソード開始直前）の位置を基準とする
- 動かしてしまった物体を元の位置へ戻しても、変位の最大値で判定するため失敗のままである

## タイムアウト仕様

本番の採点と同一の制約である。

**`/act`・`/reset` の 1 リクエストが 10 秒を超えた場合、そのトラックは失敗（error 扱い）
となり 0 点となる。** これは平均でも累積でもなく、1 回でも超過するとそのトラック全体が
失敗となる制約である。モデルの推論が 10 秒以内に収まることを必ず確認すること。

| 対象 | 上限 | 超えると |
|---|---|---|
| 推論: `/act`（および `/reset`）1 リクエスト | **10 秒** | そのトラックは error 扱いの 0 点 |
| サーバー起動（モデルロードを含む） | 本番採点は **1200 秒**（本配布環境の既定は 120 秒。`SERVER_TIMEOUT` で変更可） | 評価不能として終了 |

- タイムアウトは **HTTP リクエスト単位**である。平均・累積・エピソード単位の制限はない。
- アクションチャンクをサーバー内にキャッシュするモデルの場合、推論が実行される「重い」
  リクエストのみが上限の対象となる（実質的な制約は「チャンク 1 回分の推論 ≤ 10 秒」である）。
- [validate_submission.py](validate_submission.py) のスモークテストは、同一の 10 秒基準で
  レイテンシを警告する。提出前に必ず一度実行することを推奨する。

## ディレクトリ構成

| パス | 役割 |
|---|---|
| [pipeline/](pipeline/) | 評価パイプライン（Track 1〜3 共通） |
| [compe/t1/](compe/t1/) | Track 1 の example タスク定義 |
| [compe/t2/](compe/t2/) | Track 2 の example タスク定義 |
| [compe/t3/](compe/t3/) | Track 3 の example タスク定義 |
| [submission_template/](submission_template/) | 提出テンプレート（`policy_server.py` の `MyPolicy` のみ編集。編集前でも動作する） |
| [evaluate.py](evaluate.py) | 提出 zip の一括評価 |
| [validate_submission.py](validate_submission.py) | 提出物チェックスクリプト |
| [examples/](examples/) | 学習の参考例（SmolVLA の Colab ノートブック / pi0.5 の追加学習レシピ）。提出には必須ではない |
| [scripts/](scripts/) | 補助スクリプト（演習環境での学習データセット展開） |
| [tests/](tests/) | ハーネスの単体テスト |
| [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) | setup.sh が取得する第三者製ソフトのライセンス表記 |

## 付録: 採点環境のプリインストール一覧

本番採点イメージのシステム Python 3.10 における `pip list` である。
`requirements.txt` に書かなかったライブラリは、ここに記載の版が使用される。

**本配布環境と本番の採点環境は同一の構成である。**

lerobot はこの一覧には含まれない。使用する場合は
[lerobot を使用する場合](#lerobot-を使用する場合)を参照すること。

ただしイメージ更新に伴って版が変わりうるため、**自分のモデルに必要な依存は
`requirements.txt` に明示すること**（イメージ側に入っていることを前提にしない）。

<details>
<summary>pip list（全量）</summary>

```
ImageIO==2.37.4                      httpcore==1.0.9                      nvidia-nvshmem-cu13==3.4.5
Jinja2==3.1.6                        httpx==0.28.1                        nvidia-nvtx==13.0.85
MarkupSafe==3.0.3                    huggingface_hub==1.28.0              omegaconf==2.3.1
PyOpenGL==3.1.10                     hydra-core==1.3.2                    opencv-python-headless==4.11.0.86
PyYAML==6.0.3                        idna==3.19                           opencv-python==4.11.0.86
Pygments==2.21.0                     importlib_resources==7.1.0           packaging==26.3
Wand==0.7.2                          iniconfig==2.3.0                     pillow==12.3.0
absl-py==2.5.0                       joblib==1.5.3                        pip==26.2.1
annotated-doc==0.0.5                 jsonschema-specifications==2025.9.1  platformdirs==4.11.3
annotated-types==0.8.0               jsonschema==4.26.0                   pluggy==1.6.0
antlr4-python3-runtime==4.9.3        jupyter_core==5.9.1                  pydantic==2.13.4
anyio==4.14.2                        jupytext==1.19.5                     pydantic_core==2.46.4
attrs==26.1.0                        kiwisolver==1.5.0                    pyparsing==3.3.2
bddl==3.6.0                          lazy-loader==0.5                     pytest==9.1.1
certifi==2026.7.22                   llvmlite==0.49.0                     python-dateutil==2.9.0.post0
charset-normalizer==3.5.1            markdown-it-py==4.2.0                referencing==0.37.0
click==8.4.2                         matplotlib==3.10.9                   regex==2026.7.19
cloudpickle==3.1.2                   mdit-py-plugins==0.6.1               requests==2.34.2
contourpy==1.3.2                     mdurl==0.1.2                         robosuite==1.4.0
cuda-bindings==13.3.1                mpmath==1.3.0                        rpds-py==0.30.0
cuda-pathfinder==1.6.1               msgpack==1.2.1                       scikit-image==0.25.2
cuda-toolkit==13.0.2                 mujoco==3.7.0                        scipy==1.15.3
cycler==0.12.1                       nbformat==5.11.1                     setuptools==81.0.0
defusedxml==0.7.1                    networkx==3.4.2                      six==1.17.0
easydict==1.13                       nltk==3.10.3                         starlette==1.6.0
einops==0.8.2                        numba==0.67.0                        sympy==1.14.0
etils==1.13.0                        numpy==1.26.4                        termcolor==3.3.0
exceptiongroup==1.3.1                nvidia-cublas==13.1.0.3              tifffile==2025.5.10
fastapi==0.141.1                     nvidia-cuda-cupti==13.0.85           tomli==2.4.1
fastjsonschema==2.22.2               nvidia-cuda-nvrtc==13.0.88           torch==2.11.0+cu130
filelock==3.32.4                     nvidia-cuda-runtime==13.0.96         tqdm==4.70.0
fonttools==4.63.0                    nvidia-cudnn-cu13==9.19.0.56         traitlets==5.16.1
fsspec==2026.7.0                     nvidia-cufft==12.0.0.61              triton==3.6.0
future==1.0.0                        nvidia-cufile==1.15.1.6              typing-inspection==0.4.4
glfw==2.10.2                         nvidia-curand==10.4.0.35             typing_extensions==4.16.0
gym-notices==0.1.0                   nvidia-cusolver==12.0.4.66           urllib3==2.7.0
gym==0.25.2                          nvidia-cusparse==12.6.3.3            uvicorn==0.52.4
h11==0.16.0                          nvidia-cusparselt-cu13==0.8.0        wheel==0.48.0
h5py==3.16.0                         nvidia-nccl-cu13==2.28.9             zipp==4.1.0
hf-xet==1.6.0                        nvidia-nvjitlink==13.0.88
```

</details>

> 上記の `nvidia-*` は **CUDA 13 系**（`-cu13` または版番号 13.x）である。CUDA 12 ビルドの
> torch を使用する場合は、必要な `nvidia-*-cu12` を `requirements.txt` に自分で含めること
> （[CUDA 12 系の torch を使用する場合の注意](#cuda-12-系の-torch-を使用する場合の注意)）。
