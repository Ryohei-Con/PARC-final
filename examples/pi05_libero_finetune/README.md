# pi0.5 の追加学習（LoRA）

lerobot の pi0.5 を LIBERO 系データで LoRA 追加学習し、提出用のチェックポイントを
作るまでのレシピ。提出物（ポリシーサーバー）の作り方はこのレシピでは扱わない。
採点環境で実際に動作したサンプル提出物 `pi05_step005000_submission_py310.zip` を
参照すること。

学習は GPU 必須で、下表の条件では 1 GPU で数十時間かかる。手元の GPU に
合わせて `PI05_BS` / `PI05_GA` / `PI05_STEPS` を調整すること。

## ファイル

| パス | 内容 |
|---|---|
| [scripts/setup_train.sh](scripts/setup_train.sh) | 学習用 venv の構築（lerobot の clone + パッチ適用） |
| [scripts/full_pi05_lora.sh](scripts/full_pi05_lora.sh) | LoRA 学習のランチャー |
| [scripts/smoke_pi05.sh](scripts/smoke_pi05.sh) | 短い学習 → チェックポイント → マージまでを一度通す動作確認 |
| [scripts/probe_pi05_bs.sh](scripts/probe_pi05_bs.sh) | 少数ステップだけ回して VRAM とスループットを測る |
| [scripts/_train_common.sh](scripts/_train_common.sh) | ランチャー共通のヘルパー（resume、VRAM サンプラ等） |
| [scripts/merge_lora.py](scripts/merge_lora.py) | LoRA をベース重みへマージして提出用チェックポイントを作る |
| [patches/](patches/) | lerobot に当てる 2 本のパッチ（[下記](#パッチ)） |

## 1. 環境構築

pi0.5 は gated repo の `google/paligemma-3b-pt-224` を取得する。先に
[配布ページ](https://huggingface.co/google/paligemma-3b-pt-224)で利用条件に同意し、
その権限を持つトークンを用意すること。

```bash
cd examples/pi05_libero_finetune
bash scripts/setup_train.sh
source env_train.sh
export HF_TOKEN=hf_...        # gated repo の取得に必須
```

**学習用 venv は評価用 venv（ルートの `setup.sh`）とは別に作る。** 評価用 venv は
採点環境の再現が目的なので、学習の依存で汚さないこと。評価と提出前チェックは
ルートの `setup.sh` + `env.sh` 側で行う。

lerobot は **学習側で v0.4.4** に固定している。0.5.0 以降は Python 3.12 以上を要求し、
採点イメージの Python 3.10 にはそのままでは入らない。提出側でどの版を使うかは別問題
であり、このレシピでは決めない（0.4.4 で学習したチェックポイントを別版で配信すること
自体はできる。サンプル提出物がその構成である）。

## 2. データセット

LIBERO と LIBERO-plus を 20Hz で統合したデータセットを使う。演習環境では
`~/dataset/` に tar で配布しているので、先にルートの
[scripts/extract_dataset.sh](../../scripts/extract_dataset.sh) で `~/data/` に展開する。

```bash
bash ../../scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar
```

場所は env で指定する。

```bash
export PI05_DATASET_REPO_ID=/path/to/hf/libero_combined_20hz
export PI05_DATASET_ROOT=~/data/libero_combined_20hz
```

## 3. 学習

```bash
bash scripts/smoke_pi05.sh                      # レシピが通ることを確認する
BS=16 STEPS=30 bash scripts/probe_pi05_bs.sh    # 載る batch size を決める
bash scripts/full_pi05_lora.sh
```

`smoke_pi05.sh` は 20 ステップだけ学習してチェックポイントを保存し、
`merge_lora.py` でマージできるところまでを確認する。

なお、スモークが見る「保存されたステップ番号」は勾配累積のパッチの有無で
変わらないため、それだけでは累積が効いているかは判定できない。

既定の学習条件:

| 項目 | 値 |
|---|---|
| ベース重み | `lerobot/pi05_libero_base` |
| 手法 | LoRA r=16（gemma_expert の q/v + action projection、lerobot の PI0.5 既定） |
| batch × grad accum | 16 × 8 =（実効）128 |
| steps | 20,000 |
| lr | 5e-5（decay 5e-6、warmup 1,000） |
| dtype | bfloat16 |

`PI05_BS` / `PI05_GA` / `PI05_STEPS` / `PI05_SAVE_STEPS` / `PI05_LORA_R` /
`PI05_LR` / `RUN_NAME` を env で上書きできる。`WANDB_API_KEY` を設定した場合のみ
W&B に記録する（プロジェクト名は `WANDB_PROJECT`）。

`$OUT_DIR/checkpoints/last` があれば自動で resume する。新規に回したい場合は
`RUN_NAME` を変えるか出力先を消すこと。

## 4. 提出物にする

学習が出力するのは LoRA アダプタであり、そのままでは提出できない。
ベース重みへマージして 1 つのチェックポイントにする。

```bash
python scripts/merge_lora.py \
    --adapter ~/pi05-ft-outputs/<RUN_NAME>/checkpoints/020000/pretrained_model \
    --out     ~/pi05-ft-outputs/<RUN_NAME>/merged
```

採点環境は外部通信を遮断するため、重みもトークナイザも zip に同梱する必要がある。
`merge_lora.py` はトークナイザ（`google/paligemma-3b-pt-224`）を `<out>/tokenizer/`
に出力する。gated repo であり、実行時には取得できないためである。

ここから先――`policy_server.py` の実装、`requirements.txt`、zip の作り方――は
**このレシピでは扱わない**。pi0.5 を採点環境で動かすには lerobot と transformers の
版の組み合わせに固有の注意が要り、レシピ側に固定の手順を置くと実態とずれるためである。
採点環境で実際に動作した構成がサンプル提出物として配布されているので、そちらを
参照すること。

    pi05_step005000_submission_py310.zip

zip の構成要件（`policy_server.py` / `requirements.txt` / `model_weights/` の 3 点）と
提出前チェックの手順はテンプレート側にまとまっている
（[submission_template/README.md](../../submission_template/README.md)）。

## 採点条件との対応

学習の設定は採点側に合わせてある。提出側に関わる欄はサンプル提出物の構成である。
変更する場合はここがずれないよう注意する。

| 項目 | 採点環境 | 本例 |
|---|---|---|
| 制御周波数 | 20Hz | 20Hz のデータで学習 |
| 観測解像度 | 128×128 | 256×256 で学習し、サーバー側で拡大 |
| 1 エピソードの上限 | 300 ステップ | — |
| 推論タイムアウト | 1 リクエスト 10 秒 | action chunk 10 ステップ分を 1 回で推論 |
| サーバー起動 | 120 秒以内 | — |
| Python | 3.10 | 学習は lerobot 0.4.4（3.10 対応の最終版）。提出側の版はサンプル提出物を参照 |

## パッチ

`patches/` の 2 本を lerobot v0.4.4 に当てる（`setup_train.sh` が自動で行う）。

| パッチ | 内容 |
|---|---|
| `pi05-config-defaults.patch` | `PI05Config` に `drop_n_last_frames=49` を追加する。lerobot は policy config にこの属性があるときだけ `EpisodeAwareSampler` を使うため（`lerobot_train.py`）、無いと各エピソード末尾の 49 フレームが、エピソード外に出た action をパディング（最終 action の繰り返し）した状態で学習に入る。pi0.5 の損失は `action_is_pad` を見ない |
| `grad-accum-env-var.patch` | `LEROBOT_GRAD_ACCUM` で勾配累積を有効にする（lerobot の CLI が公開していないため）。LR スケジューラも実効ステップごとに進むよう直す |

`merge_lora.py` はこのキーを出力から取り除く。パッチを当てていない lerobot では
このキーがあるとロードに失敗するため、マージ済みチェックポイントには残さない。

## ライセンス

pi0.5 のベース重み `lerobot/pi05_libero_base` は **Gemma Terms of Use** で提供される。利用条件は
[../../THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) と配布元の表記を
確認すること。
