# InternVLA-A1.5 — 学習と評価の手順

対象環境: クラウド GPU 1 枚（NVIDIA RTX PRO 6000 Blackwell Server Edition / 96GB / driver 595 系 = CUDA 13.2）

この文書は「クローンした直後の状態から、学習を回して、提出物を評価するまで」を通しで書いたものである。
背景・設計判断は次を参照:

- `docs/plans/internvla-a15-finetune-plan.md` — 実装計画（F 番号の事実表、リスク表）
- `docs/plans/handoff-p2.md` — 引き継ぎ書
- `docs/internvla-py310-porting.md` — Python 3.10 採点環境への移植手順
- `README.md`（このディレクトリ） — レシピの説明

---

## 0. 全体像

環境は **2 つに分かれている**。混ぜてはいけない（移植手順書 §2）。

| 用途 | 場所 | Python | 作り方 |
|---|---|---|---|
| **学習** | conda env `internvla_a1_5` | 3.11 | `examples/internvla_a15_libero_finetune/scripts/setup_train.sh` |
| **評価・提出前チェック** | venv `PARC2026_final/venv` | 3.10 | リポジトリルートの `setup.sh` |

学習環境で提出物を検証してはいけない。採点環境は Python 3.10 で外部通信が無く、
学習 env には「たまたま入っている」依存が大量にあるため、検証にならない。

ディスクの割り当て（この機体の実測）:

| パス | 用途 | 備考 |
|---|---|---|
| `~/dataset` | 配布 tar（読み取り専用マウント） | 永続。常にここから展開し直せる |
| `~/data` | 展開済みデータセット + `hf/`（重み） | **再起動のたびに空になる。**（`/opt/dlami/nvme/vol/data.img` が作り直される。実測で 2 回消えた） |
| `~/ivla-a15-outputs` | チェックポイント | `/home/ryokondo` 上。再起動をまたいで**残る** |
| `~/ivla-a15-logs` | 学習ログ + VRAM CSV | 同上 |

**この配置は意図的である。** 消えて困るもの（チェックポイント・ログ）を永続側の
`/home/ryokondo` に、消えても数分で入れ直せるもの（データセット・重み）をエフェメラル側に
置いている。`/home` は 98GB あり、データセットと重みを追い出すことでチェックポイント用に
90GB を空けられる。

### 再起動したら最初にこれを実行する

```bash
cd ~/PARC2026_final/examples/internvla_a15_libero_finetune
source env_train.sh
bash scripts/provision_data.sh     # データセット再展開 + robot_type 書き換え + 重み再取得 + 検証
```

冪等なので、消えていなければ何もせず検証だけして終わる。所要時間は実測でデータセット約 2 分
＋重み約 5 分。学習中に `~/data` が消えると学習は落ちるが、`checkpoints/last` は永続側に
残っているので、上記を流してから `bash scripts/tmux_train.sh` で再開できる。

---

## 1. 環境構築

```bash
cd ~/PARC2026_final

# 1-1. 学習環境を作る（conda 3.11 / torch / 上流 / torchcodec / transformers 差し替え）
cd examples/internvla_a15_libero_finetune
bash scripts/setup_train.sh

# 1-2. 以後、学習を回すシェルでは毎回これを source する
source env_train.sh

# 1-3. データセット展開 + robot_type 書き換え + 重み取得 + 検証（冪等）
bash scripts/provision_data.sh
```

`provision_data.sh` は次を順にやる。個別に実行することもできる。

```bash
bash ~/PARC2026_final/scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar  # 約 20GB
python scripts/prepare_dataset.py --root ~/data/libero_combined_20hz               # robot_type
bash scripts/fetch_weights.sh                                                      # 約 31GB
python scripts/verify_train_env.py                                                 # 検証
```

`setup_train.sh` は最後に `scripts/verify_train_env.py` を実行する。**全項目が PASS
（WARN は可）でなければ先へ進まないこと。** 期待される出力:

```
[  OK  ] C1 Python 3.11 — 3.11.16
[  OK  ] C2 torch / CUDA — torch=2.10.0+cu130 ... bf16_matmul=OK bf16_linear_bias(cuBLASLt)=OK arch_list_has_sm_120=True
[  OK  ] C3 依存の版 — numpy=2.2.6 torchvision=0.25.0+cu130 transformers=5.2.0 nvidia-cublas=13.6.1.10 nvidia_wheels=cu13
[  OK  ] C4 transformers 差し替え — 1/1 ファイルがバイト一致
[  OK  ] C5 上流 lerobot — provenance 7/7 一致
[  OK  ] C6 overlay 登録 — robot_types=['libero_combined']
[  OK  ] C7 torchcodec 実デコード — shape=(3, 256, 256) mean=118.0 nonzero=196575
[ WARN ] C8 オプショナル依存 — fla=0.5.0 causal_conv1d=なし flash_attn=2.8.3
```

> C8 は `flash_attn` だけ特別扱いで、動画ヘッド前提（`IVLA_ACTION_LOSS_ONLY` が `true` でない）
> のときに flash-attn が無ければ **FAIL** にする。`fla` / `causal_conv1d` は無くても
> 純 torch にフォールバックするので WARN のままでよい。

### 1.1 この環境で踏んだ落とし穴（同じ罠に戻らないための記録）

| 症状 | 原因 | 対処（スクリプトに反映済み） |
|---|---|---|
| `conda create` が ToS で止まる | Miniconda の既定チャンネル `repo.anaconda.com/pkgs/main` が ToS 同意を要求。商用ライセンス条件も付く | **Miniforge**（conda-forge のみ）を入れる。`conda create` にも `--override-channels -c conda-forge` |
| `torch.cuda.is_available()` は True・fp32 も通るのに **bf16/fp16 の matmul だけ** `CUBLAS_STATUS_INVALID_VALUE` | 上流手順どおりの `torch 2.10.0+cu128` は、この GPU（sm_120 / RTX PRO 6000 Blackwell **Server Edition**）の tensor core パスを cuBLAS 12.8 が持たない。学習は `--policy.dtype=bfloat16` なので致命的 | **torch の版は上流検証どおり 2.10.0 のまま、CUDA だけ cu130 ビルドにする**。`verify_train_env.py` の C2 が毎回 bf16 の行列積を実際に回して検証する |
| 素の matmul は通るのに、**bias 付き `Linear` だけ** `CUBLAS_STATUS_NOT_INITIALIZED`（`cublasLtMatmulAlgoGetHeuristic`）。VRAM は 94GiB 空いていて OOM ではない | torch 2.10.0+cu130 が引く `nvidia-cublas 13.1.0.3` は、この GPU で **cuBLASLt の経路だけ**が壊れている。bias 付き Linear（VLM の qkv 等）はモデル中に無数にあるので最初の forward で落ちる | `nvidia-cublas>=13.6.1.10` に上げる。**ただし flash-attn より後に。** flash-attn は torch を依存に持ち、torch は `nvidia-cublas==13.1.0.3` をピンしているので、先に上げても `pip install flash-attn` が壊れた版に巻き戻す（実測）。`setup_train.sh` は 7.5/9（flash-attn の後）で入れている。`DISABLE_ADDMM_CUDA_LT=1` でも回避できるが epilogue fusion を捨てることになるのでライブラリを直す。`verify_train_env.py` の C2 が bias 付き Linear を実際に回して毎回検証する |
| `~/data` が空になっている | 再起動のたびにエフェメラル領域が作り直される | `bash scripts/provision_data.sh` |
| CUDA 12 系と 13 系の nvidia wheel が同居 | 両方が `site-packages/nvidia/*/lib` に展開され、cuDNN/cuBLAS が別系統で上書きされる | 入れ替え前に古い系統を uninstall。`verify_train_env.py` の C3 が同居を FAIL にする |
| `setup_train.sh` が無言で途中終了 | `pip freeze \| grep ...` がヒットしないと exit 1 → `pipefail` + `set -e` で死ぬ | 該当行に `\|\| true` |
| 学習が `AssertionError`（`wan/modules/attention.py:112` の `assert FLASH_ATTN_2_AVAILABLE`）で落ちる | **動画ヘッド（`action_loss_only=false`）には flash-attn が必須。** WAN の DiT は `wan/modules/model.py:249` が `flash_attention()` を直接呼ぶ。同じファイルの `attention()` には sdpa フォールバックがあるが、`model.py` はそちらを通らない | `FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=$(nproc) pip install --no-build-isolation flash-attn==2.8.3`。`FLASH_ATTN_CUDA_ARCHS` の既定は `80;90;100;120` で 4 アーキテクチャ分コンパイルするので、この機体の `120` だけに絞ると約 1/4 の時間で終わる（別 GPU に持ち回るなら既定に戻す）。`setup_train.sh` は `IVLA_ACTION_LOSS_ONLY=true` でない限り既定でビルドする。`verify_train_env.py` の C8 は動画ヘッド前提で flash-attn が無ければ **FAIL** にする |
| flash-attn のビルドが何時間も終わらない | **`ninja` が入っていない**と torch の `CUDAExtension` が直列コンパイルにフォールバックする（実測で並列度 1、load average 1.2） | `pip install ninja` を先に入れる。`setup_train.sh` は 4/9 で入れる。`MAX_JOBS=$(nproc)` と併用する |
| `causal-conv1d` のビルドが終わらない | 既製 wheel が無くソースからビルドする | 既定でスキップ。無くても純 torch にフォールバックする。必要なら `IVLA_BUILD_CUDA_KERNELS=1` |
| 学習は進むのにカメラが黒画像 | torchcodec の ABI が torch と不一致。**import は通る**ので起動時に気づけない | `torchcodec==0.10.*` を `pip install -e` の**後**に明示ピン。`verify_train_env.py` の C7 が実データの mp4 を 1 フレームデコードして PNG 保存 + 非ゼロ assert |

---

## 2. データの事実（P2 で確定済み）

`artifacts/facts/facts.json` に記録。`python scripts/inspect_dataset.py --root ~/data/libero_combined_20hz --out artifacts/facts/facts.json` で再生成できる。

| 項目 | 値 |
|---|---|
| fps | 20 |
| 総エピソード / 総フレーム | 19,533 / 3,028,708 |
| 画像 | `observation.images.front` / `observation.images.wrist`、ともに **256×256×3**（video） |
| `observation.state` | 8 次元（EE 位置 3 + 姿勢 3 + グリッパ指 2） |
| `action` | 7 次元 |
| **グリッパ規約** | **`minus_one_one`**。dim6 は `{-1, 0, +1}` の三値 |
| **グリッパの向き** | **`+1` = 閉、`-1` = 開**（データセットと LIBERO env は同一規約 = 恒等写像） |
| `stats.json` | フラット 1 セット（スイート別ではない）。`q01`/`q99` を持つ |
| `robot_type` | `panda` → **`libero_combined`** に書き換え済み |
| 画像の向き | **未確定（TBD）**。§6 参照。**学習には影響しない**（推論のみ） |

### グリッパの向きの根拠

「開閉が反転すると全タスク 0%」（計画 U1）なので、推測ではなく実データで確かめた。
同一エピソード内で `action[6]` を出した**次**ステップの指の開き幅 `state[6] - state[7]` の変化:

| `action[6]` | n | 平均 Δ開き幅 | 平均 開き幅 | 意味 |
|---:|---:|---:|---:|:--|
| `+1` | 1,486,312 | **−0.00046** | 0.0443 | 閉じる |
| `−1` | 1,145,047 | **+0.00060** | 0.0736 | 開く |

これを `inference/runtime_config.json` に焼いてある:

```json
"gripper": {
  "dataset_convention": "minus_one_one",
  "threshold": null,          // minus_one_one では 0.0 が使われる
  "below_threshold": "open",  // 0 未満なら env_open(-1) = 恒等写像
  "env_close": 1.0, "env_open": -1.0
}
```

### スキーマの `image_mapping` を上流から変えている

上流 `libero.yaml` の源キーは `observation.images.image` / `.wrist_image` だが、
配布データセットの feature 名は `observation.images.front` / `.wrist` である。
`lerobot_policy_parc/schemas/libero_combined.yaml` はこちらに合わせてある。
不一致だと `RemapImageKeyTransformFn`（`transforms/core.py:238` の `data.pop(old_key)`）が
`KeyError` になる（黙って落ちるのではなく即例外になるのは救い）。

**行き先キー（`image0` / `image1`）は上流と同じでなければならない**（モデルが読むキーなので）。

---

## 3. バックボーン VLM の学習率を 0.1 倍する仕組み

**指示**: バックボーン VLM だけ学習率を 0.1 倍（他は `5e-5`、VLM は `5e-6`）。

### なぜ CLI 引数だけでは届かないか

上流 `lerobot/optim/factory.py:37` は

```python
params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
optimizer = cfg.optimizer.build(params)
```

で、**どちらの分岐でもパラメータ名が失われる**（`InternVLAA15Policy.get_optim_params()` は
`self.parameters()` を返すだけ）。名前が無いと「どれが VLM か」を判定できない。
さらに `configs/train.py:115-119` は `use_policy_training_preset=True` のとき
`cfg.optimizer = policy.get_optimizer_preset()` で **CLI の `--optimizer.*` を上書きする**。

### 上流の `xvla-adamw` を使わない理由

上流にも同趣旨の `XVLAAdamWConfig`（`optim/optimizers.py:107`）があるが、振り分けが
`if "vlm" in name.lower():` である。**InternVLA-A1.5 のパラメータ名に `vlm` という語は
1 つも出てこない。** 実際の階層は:

```
InternVLAA15Policy.model                  -> InternVLAA15                  (modeling:539)
  .qwen3_5_with_expert                    -> InternVLAA15WithExpertModel   (modeling:550)
    .qwen3_5                              -> Qwen3_5ForConditionalGeneration  ← VLM 本体
    .action_expert                        -> Qwen3_5TextModel                 ← action expert
```

つまり VLM のパラメータ名は `model.qwen3_5_with_expert.qwen3_5.*`。`xvla-adamw` を
そのまま使うと VLM グループが**空になり、例外も警告も出ないまま全パラメータがフル lr で
回る**（「0.1 倍したつもり」になる）。

### 実装

overlay の 2 ファイル。上流は 1 バイトも編集していない。

| ファイル | 役割 |
|---|---|
| `lerobot_policy_parc/optim_vlm_lr.py` | `ParcAdamWVlmScaledConfig`（choice 名 `parc_adamw_vlm_scaled`）。名前の**前方一致**で `vlm` / `other` の 2 グループに分け、**どちらかが空なら `RuntimeError`** |
| `lerobot_policy_parc/_monkeypatch.py` | 計画 §3 の例外規定に従う唯一の monkeypatch ファイル。`get_optim_params` → `dict(named_parameters())`、`get_optimizer_preset` → 上記 config。適用時に `logging.warning` で名前を出す |

`scripts/train_entry.py` が `overlay.assert_training_patches()` を呼び、当たっていなければ
**学習を始める前に例外**にする。

- 倍率は `IVLA_VLM_LR_SCALE`（既定 `0.1`）。値は `train_config.json` に残るので、
  後から「その run が本当に 0.1 だったか」を checkpoint から確認できる。
- スケジューラは触っていない。`LambdaLR` は各グループの `initial_lr` に同じ係数を掛けるので、
  **0.1 の比は warmup / cosine decay を通して保たれる**。
- `param_groups[0]` は `other` にしてある。`lerobot_train.py:130` がログ用の lr を
  `param_groups[0]["lr"]` から取るため、`vlm` を先頭に置くとログと W&B の `lr` が
  0.1 倍の値になって設定を読み違える。

### 効いていることの確認

学習ログに次が出る:

```
lerobot_policy_parc._monkeypatch applied: InternVLAA15Policy.get_optim_params -> dict(named_parameters())
lerobot_policy_parc._monkeypatch applied: InternVLAA15Config.get_optimizer_preset -> ParcAdamWVlmScaledConfig(vlm_lr_scale=0.1)
parc_adamw_vlm_scaled: vlm group N tensors / X.XM params @ lr=5e-06 (= 5e-05 * 0.1) | other group M tensors / Y.YM params @ lr=5e-05
```

`vlm group` の行が無い、または params 数が 0 なら効いていない。

---

## 4. 学習

### 4.1 確定ハイパーパラメータ

| 項目 | 値 | 根拠 |
|---|---|---|
| `steps` | 30,000 | ユーザー指定 |
| `scheduler_warmup_steps` | 600 | 上流比率 2000/100000 = 2% を 30k に**再スケール**（計画 F12）。上流の 2000 をそのまま使うと warmup が 6.7% を食う |
| `scheduler_decay_steps` | 30,000 | `steps` と一致。上流の 100000 のままだと cosine が 3/10 しか進まない |
| `optimizer_lr` | 5e-5 | 上流どおり |
| `scheduler_decay_lr` | 5e-6 | 上流どおり |
| **`IVLA_VLM_LR_SCALE`** | **0.1** | ユーザー指定。VLM の実効 lr は 5e-6 → 5e-7 |
| `gradient_checkpointing` | false | |
| `action_loss_only` | false（動画ヘッド on） | WAN の重みが必要 |
| `freeze_learnable_tokens` | false | foresight token を学習対象に |
| `enable_vqa_loss` | true | |
| `save_freq` | 5,000（= steps/6） | 6 個残る。ディスクは §4.5 の自動整理で担保する |
| `batch_size` | **8**（probe で確定。`artifacts/probe/probe_report.md`） | peak 73GB/96GB・0.81 s/step・約 6.8h。BS=16 は OOM。BS=4 より速い（9.88 対 7.13 サンプル/秒） |
| ダウンサンプル確率 | nearest **0.10（固定）** / box 0.45 / triangle 0.30 / cubic 0.15 | ユーザー指定 |

> **gradient accumulation は使えない。** 上流 `lerobot_train.py` の `update_policy()` は
> 毎バッチ `optimizer.step()` する（`accelerator.accumulate()` を使っていない）。
> よって **実効バッチ = `batch_size`**。ここを大きく取れるかが probe の主目的になる。

### 4.2 手順

```bash
source env_train.sh

# 4-2-1. smoke（20 step 通し）。**これが PASS するまで本走を始めない**
bash scripts/smoke_ivla.sh

# 4-2-2. batch size probe（VRAM と sec/step の実測）
bash scripts/probe_ivla_bs.sh
cat artifacts/probe/probe_report.md

# 4-2-3. probe の結果から batch size を決めて本走（tmux）
IVLA_BS=8 bash scripts/tmux_train.sh
```

現在の本走の設定（2026-09-08 開始、`ivla_a15_libero_combined`）:

| 項目 | 値 |
|---|---|
| batch_size | 8（実効バッチ 8。gradient accumulation は無い） |
| VRAM | 73,071 / 97,887 MiB（`expandable_segments:True` 込み） |
| スループット | 1.11 iters/s（0.81 s/step） |
| ETA | 約 7 時間 30 分 |
| lr | other 5e-5 / **vlm 5e-6**（warmup 600 → cosine で 5e-6 / **5e-7** へ） |
| 動画ヘッド | on（`action_loss_only=false`、WAN 重みロード確認済み） |

`smoke_ivla.sh` が見ているもの:

- overlay の monkeypatch が当たったか / optimizer が 2 グループに分かれたか
- **WAN の重みが本当に読めたか**。`wan_model.py:165-168` は読み込み失敗を
  `logger.warning("Using random initialization instead")` で握り潰し、
  **ランダム初期化のまま学習を続ける**。smoke はこの文字列があれば FAIL にする
- loss に nan/inf が無いか
- チェックポイントが保存できるか（1 個のサイズ = ディスク見積り。計画 R8）

### 4.3 tmux で回す（ssh が切れても止めない）

ssh が切れると、そのシェルの子プロセスは SIGHUP で死ぬ。tmux のサーバーは ssh
セッションから切り離されているので、接続が切れても中は動き続ける。

```bash
bash scripts/tmux_train.sh          # 起動
bash scripts/tmux_train.sh status   # 生存確認 + GPU + ログ末尾
bash scripts/tmux_train.sh attach   # 画面に接続（抜けるのは Ctrl-b d。exit しないこと）
bash scripts/tmux_train.sh logs     # ログ追尾（Ctrl-C で追尾だけ止まる）
bash scripts/tmux_train.sh stop     # 停止
```

**再接続後の手順**:

```bash
ssh <host>
cd ~/PARC2026_final/examples/internvla_a15_libero_finetune
bash scripts/tmux_train.sh status
```

**中断からの再開**: `scripts/train_ivla_a15.sh` の `_maybe_resume` が
`$IVLA_OUTPUT_DIR/$RUN_NAME/checkpoints/last/pretrained_model` を見て自動判定する。
落ちたら同じコマンドをもう一度実行するだけでよい（`--resume=true` が自動で付く）。
**逆に、最初からやり直したいときは先に出力ディレクトリを消すこと**（既存があると
resume 扱いになる）。

### 4.4 監視するもの

```bash
tail -f ~/ivla-a15-logs/ivla_a15_libero_combined.log        # 学習ログ
tail -f ~/ivla-a15-logs/ivla_a15_libero_combined.vram.csv   # 5 秒ごとの VRAM
nvidia-smi -l 5
```

ログで確認すること:

- `vlm group ... @ lr=5e-06` が出ている（§3）
- `Using random initialization instead` が**出ていない**（WAN がランダム初期化されていない）
- `loss` が下がる / `nan` が出ない
- `lr` が warmup 600 step で 5e-5 に達し、その後 cosine で 5e-6 へ落ちる

### 4.5 ディスク — チェックポイントの自動整理

1 チェックポイント = **16GB**（実測）で、内訳は:

| 中身 | サイズ | 用途 |
|---|---:|---|
| `pretrained_model/` | 5.1GB | **評価・提出に使う**（`model.safetensors` / `config.json` / `stats.json` / `train_config.json`） |
| `training_state/` | 11GB | **resume にしか使わない**（optimizer / scheduler / RNG） |

`steps=30000` / `save_freq=5000` なら 6 個で **96GB** になり、`/home` の空き（約 85GB）に
入らない。終盤で「保存できずに落ちる」のが一番損失が大きい（計画 R8）。

そこで `train_ivla_a15.sh` は `_start_checkpoint_janitor` を起動し、**5 分ごとに
「最新以外の `training_state/` を削除」**する。`last` は最新への symlink なので、
その実体だけは必ず残る。結果として 6 個で `5×5.1 + 16 ≈ 42GB` に収まる。

```
[janitor] 005000/training_state を削除（resume には最新のみ必要）
```

この行は janitor のサブシェルの標準出力なので **tmux のペインに出る**（`tee` を通らないので
`$LOG_FILE` には残らない）。`bash scripts/tmux_train.sh attach` で確認できる。
効いているかはディレクトリのサイズを見るのが早い:

```
$ du -sh ~/ivla-a15-outputs/ivla_a15_libero_combined/checkpoints/*/
5.1G    .../checkpoints/005000/     ← training_state を削除済み
16G     .../checkpoints/010000/     ← last。resume 用に残す
```

resume は常に `checkpoints/last` から行うので、この整理で再開性は落ちない。
古いチェックポイントは `pretrained_model/` が残るので、D7 のチェックポイント選択と
エクスポートはそのまま行える。

### 4.6 OOM が出たときの順序（計画 §8.1）

1. `IVLA_BS` を下げる（実効バッチがそのまま減る）
2. `IVLA_GRADIENT_CHECKPOINTING=true`（スループット −25〜35%）
3. `IVLA_ACTION_LOSS_ONLY=true` で **WAN 分岐ごと切る**。`modeling_internvla_a1_5.py:576` により
   WAN DiT + VAE が構築されなくなる。**ただし動画教師と foresight token の学習を失う**ので
   最後の手段。採用したら `artifacts/probe/probe_report.md` に理由を書く
4. `IVLA_NUM_WORKERS` を下げる（VRAM ではなく host RAM 対策）

---

## 5. 評価

学習環境（conda 3.11）ではなく、**採点環境と同じ Python 3.10 venv** で行う。

```bash
cd ~/PARC2026_final

# 5-1. 評価環境を作る（LIBERO-plus / アセット / venv。初回のみ）
bash setup.sh

# 5-2. 評価を回すシェルで毎回
source env.sh
```

### 5.1 提出物を作る

```bash
cd examples/internvla_a15_libero_finetune
source env_train.sh     # export は学習 env の依存を使う
python export/export_submission.py \
    --ckpt-dir     ~/ivla-a15-outputs/ivla_a15_libero_combined/checkpoints/<step>/pretrained_model \
    --vlm-src      "$(python -c 'import transformers,pathlib;from huggingface_hub import snapshot_download;print(snapshot_download("Qwen/Qwen3.5-2B"))')" \
    --upstream-src ~/InternVLA-A-series/src \
    --out-dir      ~/submission
```

`--vlm-src` は Qwen3.5-2B のスナップショットディレクトリ。**重みは同梱されない**
（`copy_vlm_config_only` が config / tokenizer / processor だけを拾う。移植手順書 B4）。
`--upstream-src` は `src/` を指すこと（`vendor/lerobot` にそのままコピーされる）。

### 5.2 提出物の自己チェック（クリーン venv で）

**学習 conda env では絶対に検証しない**（移植手順書 手順 C2）。学習 env には
たまたま入っている依存が大量にあり、検証にならない。

```bash
python3.10 -m venv /tmp/subm-venv
/tmp/subm-venv/bin/pip install -r ~/submission/requirements.txt
/tmp/subm-venv/bin/python ~/submission/verify_inference.py --benchmark \
    --upstream-src ~/InternVLA-A-series/src
```

見るもの: モデルのロード時間（**120 秒**制限）、1 回の推論レイテンシ（**10 秒**制限）、
`action` が `(7,) float32` で NaN/Inf 無し、`action[6] ∈ {-1, +1}`。

### 5.3 静的検査

```bash
cd ~/PARC2026_final
python validate_submission.py ~/submission          # 静的検査 + 起動スモークテスト
python validate_submission.py ~/submission --static # 静的検査のみ
```

`requirements.txt` に外部ソース指定（`git+` / `--index-url` / `-f` / `-e` / `-r` / `-c`）が
あると ERROR になる。**ERROR 0 でなければ提出できない。**

### 5.4 実際にロールアウトする

```bash
source env.sh

# 端末 1: ポリシーサーバー
cd ~/submission && python policy_server.py --port 8000

# 端末 2: 評価パイプライン
cd ~/PARC2026_final
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 300
python -m pipeline --server-url http://localhost:8000 --track track1 track2 track3 --n-episodes 2
```

zip を一括で評価する場合:

```bash
python evaluate.py ~/submission.zip --n-episodes 2
```

| track | suite | 内容 |
|---|---|---|
| track1 | `libero_t1` | 同一タスク同一ドメイン |
| track2 | `libero_t2` | — |
| track3 | `libero_t3` | reverse タスク |

### 5.5 チェックポイントの選び方（計画 D7）

保存した 5〜6 個を `pipeline` で 2 エピソードずつ回し、上位 2 個で本評価する。

### 5.6 A/B（計画 P7-37）

`inference/runtime_config.json` の以下を振って、成功率と jerk / SPARC / path length を
同一シードで比較する。

| キー | 候補 |
|---|---|
| `chunking.mode` | `none` / `rtc` / `ensemble` |
| `chunking.replan_steps` | 10 / 16 / 25 / 50 |
| `chunking.w_max` | 0.5 / 0.8 |
| `num_inference_steps` | 10 / 5（レイテンシが厳しい場合） |

`replan_steps` は**実測レイテンシを測ってから**決める。予算式（計画 §8.3）:
`(300 / replan_steps) × latency_sec + env_step_time < 120`。

---

## 6. 残っている作業

| # | 項目 | 影響範囲 | やり方 |
|---|---|---|---|
| 1 | **画像の向きの確定**（`runtime_config.json` の `image_orientation` が `TBD`） | **推論のみ。学習には影響しない** | ルートの `setup.sh` で評価 venv を作り、`tools/dump_env_frames.py` で採点 env の生フレームを、`tools/dump_dataset_frames.py` でデータセット側を PNG 保存 → `tools/orientation_match.py` で照合 → **必ず目視でも確認**（agentview と wrist を別々に決める） |
| 2 | `RenderDownsampleFn` の実データ目視（計画 P3-22） | 学習の質 | `p_nearest=1.0` に固定して 1 サンプル保存 → 明らかにエイリアスしていることを確認 → 既定確率に戻して再保存。2 枚を `artifacts/` に残す |
| 3 | `tools/parity_check.py` の実データ実行（計画 P3-21 / P5-31） | F6 の回帰 | 学習経路と推論経路で `image_grid_thw` が一致することを assert |
| 4 | probe → batch size の確定 | 学習速度 | §4.2 |
| 5 | エクスポート → 3 段階検証（計画 P6/P7） | 提出 | §5 |

> `TBD` が残っている状態で推論を起動すると `internvla_runtime.py` が明示的に例外を投げる。
> 黙って既定値で動くことはないので、埋め忘れは起動時に必ず発覚する。

---

## 7. よく使うコマンド早見表

```bash
# 学習環境に入る
cd ~/PARC2026_final/examples/internvla_a15_libero_finetune && source env_train.sh

# 環境が壊れていないか
python scripts/verify_train_env.py

# 学習の状態
bash scripts/tmux_train.sh status

# 手元テスト（overlay のロジック。GPU 不要）
python -m pytest tests/ -q

# データの事実を取り直す
python scripts/inspect_dataset.py --root ~/data/libero_combined_20hz --out artifacts/facts/facts.json

# 評価環境に入る
cd ~/PARC2026_final && source env.sh
```

## 8. 絶対に守ること（引き継ぎ書 §3 より）

1. **上流 `InternVLA-A-series` を編集しない。** 拡張は `lerobot_policy_parc` overlay 経由。
   monkeypatch が要る場合は `_monkeypatch.py` の 1 ファイルに集約し、適用を `logging.warning` で出す。
2. **`submission_template/` を編集しない。** `tests/test_policy_server_parity.py` が差分を検査している。
3. **設定をコードに直書きしない。** 画像の向き・グリッパ規約・chunking は `runtime_config.json` 経由。
4. **256→128 のダウンサンプルに `resize_with_pad` を使わない**（bilinear 固定でカーネルの多様性が消える。計画 F15）。
5. **推論経路には必ず `resize_with_pad(224,224)` を通す**（上流の推論バックエンドはこれを省略している。計画 F6）。
6. **`chunking["mode"]` を直接書き換えない。** `InternVLARuntime.set_chunking_mode()` を通す。
7. `q01_q99` の逆正規化を独自に書き直さない。

---

## 9. VQA あり/なし比較

ベースライン（VQA なし）を無改変で残したまま、RoboInter-VQA サブセットを混ぜた
「VQA あり」run を **同一シード・同一 step・同一 chunking 設定**で回して比較する。
実装計画は `docs/plans/internvla-a15-vqa-finetune-plan.md`。

### 9.1 データ出所と代替理由

- 公式 "example VQA data" は上流 README の TODO（`- [ ] Release the example VQA data ...`）で
  **未公開**。そのため同組織のロボット領域 VQA データセット
  **`InternRobotics/RoboInter-VQA`**（HF dataset）で代替する。
- 由来は DROID / RH20T 等の実ロボット画像 + VQA アノテーション。全体は約 150GB あるので
  **必ずサブセットのみ取得**する（`scripts/fetch_vqa_data.sh` がサイズガード付きで取得）。

### 9.2 サブセット選定

| 項目 | 値 |
|---|---|
| カテゴリ | `Understanding` + `Task_planning` の 2 つ（`Generation` は使わない） |
| メタ形式 | `llava_format` のみ（`smart_resize_format` / `origin_format` は取得しない） |
| サンプル数 | カテゴリ合計 `--max-samples 40000`（既定は均等割り 20000 / 20000） |
| 画像前処理 | `prepare_vqa_data.py --pad-square`（既定 ON。短辺パディング → 256×256） |
| 連結 | 2 カテゴリを `--merge-into all.jsonl` で 1 本に連結（`MultiVQADataset` は per-dataset
  weight を持たないため）。`source` はカテゴリ別（`robointer_vqa/Understanding` 等）で残す |
| 展開後ディスク | `IVLA_VQA_MAX_GIB=15` 未満（`fetch_vqa_data.sh` が超過時は取得せず失敗）。実測は（PV4 で記入） |

再取得（`$HOME/data` はエフェメラル）:

```bash
source env_train.sh && source env.vqa.sh      # env.vqa.example.sh を元に作る
IVLA_VQA_ENABLE=1 bash scripts/provision_data.sh
```

`fetch_vqa_data.sh` は最初にリポジトリのファイル一覧 + サイズ表を stdout に出し、
取得予定パターンの合計が `IVLA_VQA_MAX_GIB` を超えるなら 1 バイトも取得せず `exit 1`。
image zip は `unzip -o` で in-place 展開し、展開後に削除する（`IVLA_VQA_KEEP_ZIP=1` で残す）。
`raw/.fetch_done` マーカで冪等。

### 9.3 weight と混合

- `--vqa_dataset.weight=0.10`（**本走で使う唯一の値。weight の A/B はしない**）。
  `env.vqa.sh` で `IVLA_VQA_WEIGHT` を上書きすれば別値でも回せるが計画上は回さない。
- 混合は上流 `MixedMultimodalDataset([robot_ds, vqa_ds], weights=[1-w, w])` +
  `MultiMixedWeightedSampler`。`--policy.enable_vqa_loss=true`（ベースラインと同一）で
  mixed collate 経路に入る。
- サンプル算（`batch_size=8` / `steps=30000` の実測ベース）:

| `weight` | VQA サンプル / 1000 step | 30k step 合計 | サブセット 40k に対する周回 |
|---|---|---|---|
| 0.05 | ~400 | ~12,000 | ~0.30 epoch |
| **0.10（本走）** | ~800 | ~24,000 | ~0.60 epoch |
| 0.15 | ~1,200 | ~36,000 | ~0.90 epoch |

- robot 側の実効 draw 数は `w` の分だけ減る（w=0.10 で 240k → 216k robot draws / 30k step）。
  **比較は step 数を固定**して行い、robot draw 数の差はこの節に記録する（PV5 で記入）。

### 9.4 解像度処理

- VQA 画像も overlay の `RenderDownsampleFn` で **256→128 の 4 カーネル振り分け**
  （box 0.45 / triangle 0.30 / cubic 0.15 / nearest 0.10、robot と同一確率）を通す。
  `--vqa_dataset.type=internvla_a1_5_parc` + `--vqa_dataset.render_target=128`。
- 上流 `_make_vqa_dataset` は VQA transform を hydrate しないので、`RenderDownsampleFn` は
  `keys` 空 + `auto_detect_keys=True` の経路で `observation.images.image0` 等を拾う。
  ログに `RenderDownsampleFn was not hydrated; falling back to key auto-detection [...]`
  が 1 回出るのは**想定挙動**（VQA 経路が生きている signal）。
- 狙い = 採点環境の native 128 解像度での入力分布に合わせ、言語 grounding を底上げする。
- **追加拡張（JPEG 劣化・blur 等）は未実施。** 将来の A/B 候補。

### 9.5 比較評価

親計画 `docs/plans/internvla-a15-finetune-plan.md` §5 の評価 pipeline を流用する。

- VQA あり run（`ivla_a15_libero_combined_vqa`）と VQA なし run（`ivla_a15_libero_combined`）を
  **同一シード（42）・同一 step（30000）・同一 chunking 設定**で track1/2/3 各 N エピソード評価。
- 成功率 + jerk / SPARC / path length を並べる。チェックポイント選択手順（親計画 D7）も両 run で同一。

| track | 指標 | VQA なし | VQA あり (w=0.10) |
|---|---|---|---|
| track1 | 成功率 | （PV5 で記入） | （PV5 で記入） |
| track1 | jerk / SPARC / path length | （PV5 で記入） | （PV5 で記入） |
| track2 | 成功率 | （PV5 で記入） | （PV5 で記入） |
| track2 | jerk / SPARC / path length | （PV5 で記入） | （PV5 で記入） |
| track3 | 成功率 | （PV5 で記入） | （PV5 で記入） |
| track3 | jerk / SPARC / path length | （PV5 で記入） | （PV5 で記入） |

提出候補の判断（どちらを出すか）はこの表から説明する（PV5 で記入）。

### 9.6 ライセンス

- `InternRobotics/RoboInter-VQA` の利用条件は HF ページの license 記載を確認する（PV5 で記入）。
  由来データセット（DROID = CC-BY / RH20T = 各自条件）にも触れる。
- **非商用・研究利用限定の可能性があるため、提出前に条項を必ず確認する**（PV5 で確認）。
  VQA データは学習にのみ使用し、提出物には同梱しない。
- `THIRD_PARTY_LICENSES.md` に RoboInter-VQA を 1 行追記する（PV5 で記入）。
