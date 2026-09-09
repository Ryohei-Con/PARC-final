# InternVLA-A1.5 の追加学習（フルファインチューニング）

[InternVLA-A1.5](https://github.com/InternRobotics/InternVLA-A-series) を
`libero_combined_20hz` でフルファインチューニングし、PARC 2026 の 3 トラックで
動く単一ポリシーを作るためのレシピ。

pi0.5 のレシピ（[../pi05_libero_finetune/](../pi05_libero_finetune/)）と違い、
**学習から提出物の生成・検証までを 1 本で扱う**。採点環境（Python 3.10 / 通信遮断 /
native 128 レンダ）と学習環境（Python 3.11 conda / 256 データ）の経路差を明示的に
潰すことが主眼である。詳しい背景は
[../../docs/internvla-py310-porting.md](../../docs/internvla-py310-porting.md) を参照。

> **現在の進捗: Phase P1（手元実装）まで完了。**
> 学習・エクスポート・提出はまだ実行していない。未確定事項は
> [下記](#未確定事項tbd)のとおりデータを見るまで `TBD` のままにしてある。

## 設計方針 — 上流は 1 バイトも書き換えない（overlay）

上流に third-party plugin 機構があり、`sys.path` 上の `lerobot_policy_*` を自動
import する（`src/lerobot/utils/import_utils.py:133-155`）。加えて transform は
`DataTransformFn.register_subclass`、dataset config は
`DatasetConfig.register_subclass`、robot schema は `load_schemas_from_path()` と、
必要な拡張点がすべて公開 API で埋まっている。したがって **patch は当てず、
`lerobot_policy_parc/` を overlay として注入する。**

パッチを当てないので、提出物に同梱する `vendor/lerobot` を上流とバイト一致に
保てる。「どちらの lerobot が動いているか」の問題が構造的に消え、エクスポート時に
`diff` で機械検証できる。

ただし plugin の自動 import は **失敗を握り潰す**（`except Exception:
logging.exception(...)`）。overlay が入っていないまま学習が始まる事故を防ぐため、
主経路は `scripts/train_entry.py` で明示 import し、`assert_installed()` が
レジストリを実照会してから上流の `lerobot_train.main()` を呼ぶ。

## ファイル

| パス | 内容 |
|---|---|
| [lerobot_policy_parc/](lerobot_policy_parc/) | 上流に注入する overlay パッケージ |
| `lerobot_policy_parc/transforms_render.py` | `RenderDownsampleFn`（256→128 のカーネルランダム化） |
| `lerobot_policy_parc/dataset_config.py` | `RenderDownsampleFn` を `ResizeImagesWithPadFn` の直前に挿入する DatasetConfig |
| `lerobot_policy_parc/schemas/libero_combined.yaml` | `libero_combined` の robot schema |
| `lerobot_policy_parc/_provenance.py` | 上流ファイルの SHA256 記録・照合 |
| [scripts/train_entry.py](scripts/train_entry.py) | overlay を明示 import してから上流の train を呼ぶ入口 |
| [scripts/inspect_dataset.py](scripts/inspect_dataset.py) | 事実収集（`facts.json` を作る） |
| [inference/chunk_blending.py](inference/chunk_blending.py) | RTC guidance / temporal ensembling の純ロジック |
| [inference/internvla_runtime.py](inference/internvla_runtime.py) | 3.10 ブートストラップ + 推論本体 |
| [inference/policy_server.py](inference/policy_server.py) | 提出サーバー（テンプレートの `MyPolicy` だけ差し替え） |
| [inference/verify_inference.py](inference/verify_inference.py) | 提出前の自己チェック |
| [inference/runtime_config.json](inference/runtime_config.json) | 推論設定の焼き込み先 |
| [tools/](tools/) | 向きの照合・学習/推論の parity チェック |
| [export/export_submission.py](export/export_submission.py) | 提出パッケージの生成 |
| [tests/](tests/) | 手元（GPU 不要）で回る単体テスト |

学習ランチャー（`setup_train.sh` / `_train_common.sh` / `train_ivla_a15.sh` /
`probe_ivla_bs.sh` / `smoke_ivla.sh`）と `scripts/prepare_dataset.py`（`info.json` の
`robot_type` を `libero_combined` へ冪等に書き換える。計画 §5.2 / P3-19）は
**まだ作っていない**。前者は probe（P4）でバッチ系を確定してから、後者は実データを
展開する P3 で置く。

## 潰しているサイレント失敗

いずれも「エラーが出ないのに結果だけ悪くなる」型で、気付くのに時間を溶かす。

| # | 症状 | 原因 | 対策 |
|---|---|---|---|
| F6 | 学習では効くのに評価だけ精度が出ない | 上流の推論バックエンドは `ResizeImagesWithPadFn` を hydrate せずに構築するため `mapping` が空で no-op。学習 224 に対し推論が生解像度のまま Qwen processor に入り `image_grid_thw` が食い違う | `_obs_to_sample` が `resize_with_pad(224, 224)` を**無条件で**通す。学習側と同一関数を呼ぶ |
| 解像度 | 精度が出ない | 学習は 256→224 の縮小、採点は 128→224 の拡大 | 学習側に `RenderDownsampleFn`（256→128）を挟み、カーネルを振って頑健化 |
| B1 | 精度が出ない | 別の `lerobot` が先に import されている | `install_vendored_lerobot` が検出して落とす |
| B6 | 行動が微妙におかしい | `from_pretrained(strict=False)` で一部がランダム初期化のまま | `assert_checkpoint_covers_model()` を起動パスに入れる |
| B7 | 起動が 120 秒に間に合わない | `resize_token_embeddings` の `mean_resizing=True` | `fast_token_embedding_resize()` で `False` に固定 |
| B8 | ローカルでは動くが本番で落ちる | 環境変数に依存した設定 | `runtime_config.json` に焼き、`os.environ.setdefault` で読む |

## 1. 256 → 128 のダウンサンプル

採点環境は LIBERO を native 128×128 でレンダリングする（`pipeline/config.py:52-54`）。
学習データは 256×256 なので、そのまま `ResizeImagesWithPadFn(224, 224)` に通すと
「256→224 の縮小」になり、推論時の「128→224 の拡大」と別物の画像統計になる。

`RenderDownsampleFn` が `ResizeImagesWithPadFn` の**直前**で 256→128 を行う。
採点側がどのカーネルで 128 を作っているかは確定できないので、カーネルを振って頑健にする。

| カーネル | 確率 | 実装 |
|---|---|---|
| box | 0.45 | `F.avg_pool2d(x, 2)`（2 倍ちょうどなので厳密。`cv2.INTER_AREA` と一致） |
| triangle | 0.30 | `F.interpolate(bilinear, antialias=True)` |
| cubic | 0.15 | `F.interpolate(bicubic, antialias=True)` + `clamp(0, 1)` |
| nearest | **0.10（固定）** | `x[..., oy::2, ox::2]`、`oy`/`ox` を独立に `{0,1}` から抽選 |

決めごと:

- **カーネルはサンプル単位で 1 回だけ引く。** 入力は `[T=5, C, H, W]`
  （`image_delta_indices = [0, 12, 25, 37, 50]`）で来るので、フレームごとに引くと
  WAN の教師動画がちらついて video loss に偽の時間変化が入る。2 カメラも同一カーネル。
- **256→128 に `resize_with_pad` を使ってはならない。** `resize_with_pad` は
  bilinear 固定なので、2 倍縮小では box 平均に潰れてカーネルのバリエーションが消える。
- 128→224 の側は既存の `ResizeImagesWithPadFn` に任せる。正方形入力では padding が
  1 px も入らず素の bilinear と等価なので、**学習側は変更不要**である。

## 2. チャンク境界の平滑化

チャンク境界の急変は jerk / SPARC / path length に直接効き、1mm 衝突ルール下では
失敗にもつながる。`runtime_config.json` の `chunking.mode` 1 キーで切り替える。

| mode | 内容 |
|---|---|
| `none` | 平滑化なし（ベースライン） |
| `rtc` | 前チャンクの重なり区間をソフト目標として denoise の速度場に混ぜる |
| `ensemble` | 重複チャンクの指数加重平均 |

- **ハードマスクは採らない。** `w_max` は 1.0 未満（既定 0.8）に制約する。
  1.0 は重なり区間を前チャンクで上書きするのと等価で、新しい観測の情報を捨てる。
- グリッパ（index 6）と padding（7 以降）は guidance から除外する。グリッパは
  二値の離散量なので、平滑化すると開閉が遅れて掴み損ねる。
- 最終値（`mode` / `replan_steps` / `w_max`）は P7 の A/B で決める。
- **実行時にモードを切り替える入口は `InternVLARuntime.set_chunking_mode(mode)` だけ。**
  `runtime.chunking["mode"]` を直接書き換えると ensembler の構築・破棄とチャンク状態の
  リセットが伴わず、「`ensemble` を名乗ったまま実体は `none` 経路」という状態になる。
  `verify_inference.py --chunking ...` もこの入口を通る（B8: 自己チェックと本番を
  同じ経路に通す）。生 dict の書き換えは `get_action()` が `RuntimeError` で弾く。
- 推論はチャンクを引いた step だけで走る。`verify_inference.py` は
  **推論回数が `ceil(steps / replan_steps)` であること**を検査する。ここが毎ステップに
  退化すると計画 §8.3 の予算式 `(300 / replan) × latency < 120s` が崩れる。

## 3. 学習条件（確定値）

`steps` はユーザー指定。**`warmup` と `decay_steps` は実 step 数に再スケール済み**
である。上流 launch script の `warmup=2000` / `decay_steps=100000` は 100k step
前提の値で、そのままコピーすると warmup が全体の 6.7% を食い、cosine は最初の
3/10 しか進まずに終わる。

| 項目 | 値 | 根拠 |
|---|---|---|
| `steps` | **30,000** | ユーザー指定。実効バッチ 32 で 960k サンプル |
| `scheduler_warmup_steps` | **600** | 上流の比率（2000/100000 = 2%）を 30k step に再スケール |
| `scheduler_decay_steps` | **30,000** | `steps` と一致させる |
| `optimizer_lr` | **5e-5** | 上流どおり |
| `scheduler_decay_lr` | **5e-6** | 上流どおり |
| `save_freq` | 5,000（`steps // 6`） | 5〜6 個のチェックポイントが残る |
| `log_freq` | 50 | 単 GPU なので上流の 200 より細かく |
| dtype | bfloat16 | 上流どおり |
| 実効バッチ | 32〜64 | **probe（P4）後に確定** |
| `batch_size` / `grad accum` | 未定 | **probe 後に確定** |

値は [env.example.sh](env.example.sh) にも記録してある。

96GB でも `action_loss_only=false`（WAN の DiT + VAE を載せる）は BS=1 でも
載らない可能性がある。probe の第 1 の目的はここの可否判定である。

## 未確定事項（TBD）

**データを見るまで推測で埋めない。** 下記は `runtime_config.json` と `facts.json` に
`"TBD"` として置いてあり、`TBD` のまま推論を起動しようとすると
`internvla_runtime.py` が明示的に例外を投げる（黙って既定値で動かさない）。

| 項目 | 決まらないと何が壊れるか | 確認方法 | 確定先 |
|---|---|---|---|
| **画像の向き**（agentview / wrist） | 静かに精度が落ちる。上下逆でも「動くが掴めない」 | `tools/dump_dataset_frames.py` + `tools/dump_env_frames.py` + `tools/orientation_match.py`、**加えて必ず目視** | `runtime_config.json` の `image_orientation` |
| **グリッパ規約**（`[0,1]` か `[-1,1]` か） | 開閉が反転し全タスク 0% になる | `stats.json` の action dim6 の min/max/mean + 実サンプルのヒストグラム | `runtime_config.json` の `gripper.dataset_convention` |
| **`observation.state` のレイアウト**（8 次元 EE か joint か） | state が別物になり、`tokenize_state` のプロンプトも壊れる | `info.json` の features + `stats.json` の次元 | `artifacts/facts/facts.json` |
| 画像解像度（本当に 256 か） | 128 なら downsample が恒等になり対策全体が空振り | `info.json` の features shape | 同上 |
| `stats.json` のキー構造 | スイート別だと stats の引き当てが失敗する | キー一覧のダンプ | 同上 |
| fps（20 か） | `image_delta_indices` の時間スパンがずれる | `info.json` | 同上 |
| 正規化モード（mean_std か min_max か） | 逆正規化が壊れ、行動のスケールが桁で違う | `train_config.json` の `normalize` transform の `mode` | 学習開始後に確定 |

## 手元テストの回し方

GPU は要らない。**上流リポジトリは `sys.path` に足して読むだけ**で、pip install も
書き換えもしない。

```bash
cd examples/internvla_a15_libero_finetune

# Windows（このリポジトリの作業環境）
"C:/Users/ryohe/miniconda3/envs/internvla_a1_5/python.exe" -m pytest tests/ -q

# 上流の場所が既定と違う場合
IVLA_REPO=/path/to/InternVLA-A-series \
  "C:/Users/ryohe/miniconda3/envs/internvla_a1_5/python.exe" -m pytest tests/ -q
```

必要なパッケージ: `torch` / `torchvision` / `numpy` / `transformers` / `safetensors` /
`pyyaml` / `pillow` に加えて `pytest` / `draccus` / `scipy`。

`cv2` は**任意**である。あれば box カーネルと `cv2.INTER_AREA` の同値性も確認するが、
無ければその 1 アサーションだけ skip する（`box == interpolate(antialias=False)` の
同値性は cv2 なしでも必ず検証される）。

`tests/conftest.py` は、上流 `lerobot` の import チェーンが引く重い依存
（`pandas` / `pyarrow` / `datasets` / `av` / `imageio` / `accelerate` / `einops` /
`diffusers`）を `sys.modules` のダミーで置き換える。いずれも
「import されるだけで、テスト対象のロジックには使われない」ものに限っている。
何をスタブしたかは同ファイルの docstring に列挙してある。

| テスト | 内容 |
|---|---|
| `test_render_downsample.py` | カーネル同値性 / 単一抽選 / `[T,C,H,W]` 対応 / 頻度 / F15 |
| `test_dataset_config.py` | 挿入位置・冪等性・`action_mode`・draccus 往復 |
| `test_schema.py` | `get_schema("libero_combined")` の内容 |
| `test_chunk_blending.py` | RTC guidance と TemporalEnsembler の数値性質 |
| `test_state_utils.py` | `quat2axisangle` vs scipy、`runtime_config.json` の検証 |
| `test_runtime_guards.py` | サイレント失敗ガードがコード上に存在するか |
| `test_policy_server_parity.py` | テンプレートとの差分が `MyPolicy` + import 行だけか |
| `test_export_submission.py` | `copy_safetensors_without` ほか |
| `test_orientation_match.py` | 既知の変換を当てられるか |

## クラウドで行う作業（未実施）

| Phase | 内容 |
|---|---|
| P2 | 学習環境構築、`inspect_dataset.py` で `facts.json`、向きの確定 |
| P3 | `info.json` の `robot_type` 書き換え、`parity_check.py` |
| P4 | probe（BS ラダー）→ 本学習 |
| P5 | 推論ランタイムの実測（レイテンシ / `image_grid_thw` の一致） |
| P6 | `export_submission.py` で提出ツリー生成 |
| P7 | 3 段階検証（3.10 compileall / クリーン venv / Docker 二重 venv）→ A/B → 提出 |

## ライセンス

InternVLA-A1.5 のベース重みは **CC BY-NC-SA 4.0**（非商用）で提供される。
Qwen3.5 と WAN2.2 も含め、利用条件は
[../../THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) と配布元の表記を
確認すること。
