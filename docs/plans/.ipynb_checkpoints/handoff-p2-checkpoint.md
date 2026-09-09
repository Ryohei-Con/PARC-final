# 引き継ぎ: Phase P2（クラウド GPU 環境での学習環境構築・事実確定）

作成日: 2026-09-08 / 引き継ぎ元: 手元 Windows（GPU 無し）セッション
対象読者: クラウド GPU（RTX PRO 6000 96GB）で作業する次のエージェント / セッション

---

## 0. まず読むもの（この順で）

1. **`docs/plans/internvla-a15-finetune-plan.md`** — 実装計画の正典。**§0（F1〜F15 の事実表）、§6 の P2/P3、§8（学習ハイパラ・probe）、§9 の SC-4/SC-5、§10（リスクと未確定事項）を必ず読むこと。** 特に §0 の F 番号はコード中のコメントから頻繁に参照されているので、読まずに進めるとコードの意図が分からなくなる。
2. **`docs/internvla-py310-porting.md`** — Python 3.10 採点環境への移植手順書。§3（学習環境構築の落とし穴）、§6（サイレント失敗カタログ）は P2 の作業そのもの。
3. **`examples/internvla_a15_libero_finetune/README.md`** — このレシピの説明。確定ハイパラ・未確定事項（TBD）の一覧が書いてある。
4. **`PARC-final/README.md`**（リポジトリ直下） — 採点環境の制約（Python 3.10、外部通信遮断、`/act` 10 秒、1mm 衝突ルールなど）。

**上流実装 `InternVLA-A-series`（クローン先はローカルにしか無いので、クラウド側では git clone し直す必要がある）** も随時参照する。HEAD は `e6fc904` を前提にコードを書いている。もし clone し直したバージョンの HEAD が違う場合は、`lerobot_policy_parc/upstream_provenance.json` のハッシュ照合で警告が出るはずなので、その警告を無視せず差分を確認すること。

---

## 1. 現状（Phase P1 完了）

手元 Windows（GPU 無し）で、計画の Phase P1（手元で書けて手元で検証できる部分）を実装済み。3 回のレビュー往復（generator ⇄ evaluator）を経て収束した。

- 場所: `PARC-final/examples/internvla_a15_libero_finetune/`
- `pytest tests/ -q` → **245 passed / 7 skipped**（skip はいずれも正当: cv2 未導入 1 件、`w<0` の別カバー 5 件、`IVLA_VLM_DIR` 未設定によるオンライン parity テスト 1 件）
- 上流 `InternVLA-A-series` は **1 バイトも変更していない**（overlay 方式。`lerobot_policy_parc` パッケージが `sys.path` 経由で schema・transform・dataset config を登録する）
- `submission_template/` は無変更
- git commit はしていない（作業ツリーに置いてあるだけ）

### 主な成果物

| パス | 内容 |
|---|---|
| `lerobot_policy_parc/` | overlay 本体。`RenderDownsampleFn`（256→128 カーネルランダム化）、`InternVLAA15ParcDatasetConfig`、`libero_combined` schema |
| `inference/internvla_runtime.py` | 3.10 推論ランタイム本体。RTC / temporal ensembling / F6 対策済み resize / 各種起動時バリデーション |
| `inference/chunk_blending.py` | RTC guidance と ensembler の純ロジック（numpy+torch のみ、lerobot 非依存） |
| `inference/policy_server.py` | `submission_template/policy_server.py` のコピー（`MyPolicy` と先頭 import だけ差し替え） |
| `inference/verify_inference.py` | 提出前自己チェックスクリプト |
| `inference/runtime_config.json` | 実行時設定。**`image_orientation` と `gripper.dataset_convention` が `"TBD"`** — これを P2 で確定する |
| `scripts/inspect_dataset.py` | データセットの事実収集（`facts.json` を吐く） |
| `tools/orientation_match.py` / `dump_dataset_frames.py` / `dump_env_frames.py` | 画像の向き確定用ツール |
| `tools/parity_check.py` | 学習/推論の `image_grid_thw` 突き合わせ（`--mode mini` で軽量版） |
| `export/export_submission.py` | 提出パッケージ生成 |
| `tests/` | 12 本、245 テスト |

**未作成**（今回のスコープ外。P2/P3 で必要になったら作る）:
- `scripts/setup_train.sh`（学習 conda env 構築）
- `scripts/train_ivla_a15.sh`（本学習ランチャー）
- `scripts/probe_ivla_bs.sh`（batch size probe）
- `scripts/smoke_ivla.sh`（20 step smoke test）
- `scripts/_train_common.sh`（共通ヘルパー）
- `scripts/prepare_dataset.py`（`info.json` の `robot_type` 書き換え）

これらは計画 §5.2 に責務と関数シグネチャが書いてある。`PARC-final/examples/pi05_libero_finetune/scripts/` に類似のパターンがあるので、流用できる部分は流用すること（ただし pi05 は lerobot 0.4.4 前提で環境が違うので、そのままコピーはできない）。

---

## 2. 次にやること（Phase P2: §6 の項目 13〜18）

### 2.1 学習環境構築

計画 §0 の F 番号、手順書 §3 を参照しながら、上記の未作成スクリプトを書いて実行する。

- Python **3.11** conda env（学習は 3.11、推論は 3.10 と環境を分ける — 手順書 §2 の設計原則）
- `torch` / `torchvision` はクラウド GPU の CUDA バージョンに合わせる
- `pip install -e ${IVLA_REPO}`（InternVLA-A-series を editable install）
- **torchcodec のバージョンを torch に厳密に合わせてピン**（手順書 §3.1）。`import` が通るだけでは不十分 — サイレント失敗の代表例（学習は進むが動画デコードが失敗してゼロ埋め画像になる）。**必ずデコードした 1 フレームを PNG に保存して目視/非ゼロ assert すること。**
- Qwen3.5 用の transformers 差し替え（`transformers_replace/models/` を site-packages にコピー）
- オプショナル依存（`flash-linear-attention` 等）は失敗を致命的にしない（`set +e` で囲む）

### 2.2 データセット展開

```bash
bash scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar
```
（`PARC-final/scripts/extract_dataset.sh` を使う。`~/dataset/` から `~/data/` へ）

### 2.3 事実収集（**最優先・最重要**）

`scripts/inspect_dataset.py` を実行して `facts.json` を作る。**ここで確定する値が `runtime_config.json` の `"TBD"` を埋める。**

計画 §10.1（U1〜U7）に未確定事項の一覧がある。特に致命的な 3 つ:

| 未確定 | 誤ると何が起きるか | 確認方法 |
|---|---|---|
| **グリッパ規約**（`[0,1]` か `[-1,1]` か） | 開閉が反転し全タスク 0% | `stats.json` の action dim6 の min/max/mean + 実サンプルのヒストグラム |
| **画像の向き**（4 通りの組合せ） | 静かに精度が落ちる（「動くが掴めない」） | `tools/dump_dataset_frames.py` + `tools/dump_env_frames.py` + `tools/orientation_match.py`。**必ず目視でも確認すること**。agentview と wrist で別々に決める |
| **`observation.state` の中身**（8 次元 EE か joint か） | state が別物になりプロンプトも壊れる | `info.json` の features + `stats.json` の次元 |

他に fps・画像解像度（本当に 256 か）・`stats.json` のキー構造（統合済み 1 キーか、スイート別か）・正規化モード（`mean_std`/`min_max`/`q01_q99`）も確認必須。

**確定したら `runtime_config.json` の `"TBD"` を書き換えること。コードに直書きしない。** `internvla_runtime.py` は `"TBD"` のまま起動しようとすると明示的に例外を投げる設計になっている（黙って既定値で動かないので、埋め忘れは起動時に必ず発覚する）。

### 2.4 データ準備 & schema 確定（§6 の P3、項目 19〜22）

- `scripts/prepare_dataset.py`（要作成）で `info.json` の `robot_type` を `libero_combined` に書き換える。**単一 robot_type にする**のがポイント（上流の libero launch script はスイート別に robot_type を振るが、推論時にスイートを判別できないので、ここは意図的に反転させる）。
- `RenderDownsampleFn` が実データで効いているか目視確認（`p_nearest=1.0` に固定して 1 サンプル保存 → 明らかにエイリアスしていることを確認 → 既定確率に戻して再度保存）。
- `tools/parity_check.py` を実データで実行し、学習と推論の前処理が一致するか確認（F6 の回帰）。

---

## 3. 絶対に守ること

1. **上流 `InternVLA-A-series` を編集しない。** 拡張はすべて `lerobot_policy_parc` overlay 経由（`register_third_party_plugins()` による自動 import、または `scripts/train_entry.py` による明示 import）。理由は計画 §3 参照 — vendor する `src/lerobot` を上流とバイト一致に保つため。
2. **`submission_template/` を編集しない。** テンプレートとの差分が `MyPolicy` + 先頭 import 行だけであることを `tests/test_policy_server_parity.py` が検査している。
3. **設定はコードに直書きしない。** 画像の向き・グリッパ規約・chunking パラメータはすべて `runtime_config.json` 経由。
4. **256→128 のダウンサンプルに `resize_with_pad` を使わない。** `resize_with_pad` は bilinear 固定で、2 倍縮小では box 平均に潰れてカーネルの多様性が消える（計画 F15）。128→224 の側は逆に `resize_with_pad` をそのまま使ってよい（学習側は既にそうなっている。変更不要）。
5. **推論経路には必ず `resize_with_pad(224,224)` を無条件で通す。** 上流の推論バックエンドはこれを省略しており（F6）、学習は 224・推論は生解像度という食い違いが起きる。`internvla_runtime.py` は既にこれをやっているので、**書き換えるときは消さないこと**。
6. **`chunking["mode"]` を直接書き換えない。** `InternVLARuntime.set_chunking_mode()` を通すこと。生 dict の書き換えは `get_action()` 時点で `RuntimeError` になる設計（レビューで見つかった実バグの再発防止）。
7. `q01_q99` の逆正規化を独自に書き直さない。上流 `NormalizeTransformFn`（`transforms/core.py:296-313`）の厳密な逆になっていることをテストで確認済み。統計キーが無ければ黙って別のキーで代用せず `KeyError` にする設計。

---

## 4. ハイパラ確定値（ユーザー指定・変更しないこと）

| 項目 | 値 |
|---|---|
| `steps` | 30,000 |
| `scheduler_warmup_steps` | 600 |
| `scheduler_decay_steps` | 30,000 |
| `optimizer_lr` | 5e-5 |
| `scheduler_decay_lr` | 5e-6 |
| `gradient_checkpointing` | false |
| `action_loss_only` | false（動画ヘッド on） |
| `freeze_learnable_tokens` | false（foresight token 学習対象） |
| `enable_vqa_loss` | true |

`batch_size` / `grad_accum` は probe（`scripts/probe_ivla_bs.sh`、要作成）の実測後に確定する。計画 §8.1 のメモリ削減ラダー（batch↓→grad_accum↑ → gradient_checkpointing=true → 最終手段として action_loss_only=true）を、OOM が出た場合の判断順序として使うこと。**WAN 分岐が 96GB に載らない可能性は現実にあるので、probe で最初に判定すること。**

ダウンサンプルのカーネル確率（ユーザー指定・固定）:

| kernel | 確率 |
|---|---|
| nearest | **0.10（固定・変更禁止）** |
| box | 0.45 |
| triangle | 0.30 |
| cubic | 0.15 |

---

## 5. レビューで見つかった過去の欠陥（再発させないための参考）

実装は 3 回のレビュー往復を経ている。今後コードを触る際、同じ型の欠陥を作らないよう記録しておく。

- **ensemble モードが env 1 ステップごとに毎回モデル推論していた。** 「新チャンクを引くか」の判定にチャンクの中身（`None` かどうか）を使うと、ensemble のように毎ステップ中身を書き換えるモードで判定が壊れる。判定はカーソルと専用フラグ（`_has_chunk`）だけで行うこと。
- **`verify_inference.py` が生 dict でモードを書き換えて、意図と違う経路を検証していた。** 状態を持つオブジェクト（ensembler の有無など）と紐づく設定は、専用のセッター（`set_chunking_mode()`）経由でしか変更できないようにすること。
- **逆正規化で `q01_q99` モードなのに min/max を使っていた。** 正規化モードごとに使う統計キーが違うので、モード分岐と統計キーの対応を必ず上流実装と突き合わせること。

---

## 6. 質問・不明点があれば

- 計画ファイル（`docs/plans/internvla-a15-finetune-plan.md`）の §10（リスクと未確定事項）に大半の判断根拠が書いてある。
- ユーザーは PARC 2026 コンペの参加者。学習はクラウド RTX PRO 6000（VRAM 96GB）1 枚。手元は GPU 無し Windows。
