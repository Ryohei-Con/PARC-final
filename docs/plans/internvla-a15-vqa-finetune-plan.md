# 実装計画: InternVLA-A1.5 レシピに「VQA ありファインチューニング」を追加する

作成日: 2026-09-09 / 対象: `examples/internvla_a15_libero_finetune/`
ステータス: **ユーザー承認待ち**
親計画: `docs/plans/internvla-a15-finetune-plan.md`（F1〜F15 の事実表・様式はこれを踏襲）
関連: `docs/plans/handoff-p2.md`, `examples/internvla_a15_libero_finetune/TRAINING.md`

### 確定事項の更新（2026-09-09 ユーザー確認）

| # | 決定 | 影響する節 |
|---|---|---|
| U1 | 計画は推奨デフォルトで**承認**。generator に渡して実装開始。 | 全体 |
| U2 | VQA サブセットは **`Understanding` + `Task_planning` の 2 カテゴリ**（`Generation` は使わない）。 | §5.1, §5.3, §5.6, §6 PV1 |
| U3 | `prepare_vqa_data.py --pad-square` は **既定 ON**（正方形パディング → 256×256）。 | §5.3, §5.5, R3 |
| U4 | **VQA weight の A/B はやらない。** 本走は `IVLA_VQA_WEIGHT=0.10` の **1 本のみ**。VQA あり/なしの比較に集中する。0.05 / 0.15 は env で上書き可能にはしておくが計画上は回さない。 | §5.7, §6 PV4, §8 SC-V5, §10.2 |

---

## 0. 前提の再確認（ユーザー承認済み。変更しない）

| # | 確定事項 |
|---|---|
| A1 | VQA データセットは **`InternRobotics/RoboInter-VQA`**（HF dataset）。公式 "example VQA data" は上流 README の TODO（`- [ ] Release the example VQA data ...`）で未公開のため、同組織のロボット領域 VQA で代替する。 |
| A2 | 配置先は **`~/data`（揮発領域・再起動で消える）**。`scripts/provision_data.sh` に再取得ステップを追加して入れ直せる運用にする。RoboInter-VQA は全体 150GB なので **必ずサブセットのみ取得**（`hf download --include` / カテゴリ選定 / prep 側の `--max-samples` / 必要 zip だけ展開）。 |
| A3 | コードは **別スクリプト新設**。既存 `scripts/train_ivla_a15.sh` / `scripts/_train_common.sh` / `env_train.sh` / `env.example.sh` は **無改変**（VQA なしベースライン）。新スクリプト `scripts/train_ivla_a15_vqa.sh` と新 env テンプレート `env.vqa.example.sh` を足す。`_train_common.sh` のヘルパーは **source して再利用**（無改変）。 |
| A4 | 解像度ロバスト化は **既存 `RenderDownsampleFn` を流用**。overlay の `InternVLAA15ParcVQADatasetConfig`（`@VQADatasetConfig.register_subclass("internvla_a1_5_parc")`）が `RenderDownsampleFn` を `ResizeVQAImagesWithPadFn` の直前に挿入する実装を既に持つ（現状 dead path）。既存の 4 カーネル振り分け（box 0.45 / triangle 0.30 / cubic 0.15 / nearest 0.10、p_nearest は固定 0.10）を VQA 画像にも適用する。**JPEG 劣化・blur などの追加拡張は今回スコープ外**（A/B 項目としてドキュメント化に留める）。 |

### 触ってはいけないもの（親計画 §3・§4 の再掲）

- **上流 `~/InternVLA-A-series` は 1 バイトも編集しない。** overlay（`lerobot_policy_parc`）経由でのみ拡張する。
- **`submission_template/` は編集しない。**
- **`scripts/train_ivla_a15.sh` / `scripts/_train_common.sh` / `env_train.sh` / `env.example.sh` は編集しない**（A3）。
- **設定をコードに直書きしない**（データパス・weight・カーネル確率は env / CLI 経由）。
- **256→128 のダウンサンプルに `resize_with_pad` を使わない**（F15。`RenderDownsampleFn` が自前でカーネルを振る）。

**この計画で既存ファイルに手を入れるのは 3 箇所だけ:**
1. `scripts/provision_data.sh` — 末尾に「新設スクリプトを呼ぶ数行」を追記（A2 で明示的に許容された箇所）。
2. `lerobot_policy_parc/dataset_config.py` — overlay。`InternVLAA15ParcVQADatasetConfig` の docstring から "dead path" 記述を更新し、必要なら最小修正（§5.5 で判定）。
3. `examples/internvla_a15_libero_finetune/TRAINING.md` — 「VQA あり/なし比較」節を追記（§5.10）。

---

## 1. 調査で確定した事実（V 番号）

`[検証済]` は該当ファイルを直接読んで裏取りした。

| # | 事実 | 根拠 | 計画への影響 |
|---|---|---|---|
| V1 | **`_make_vqa_dataset` は `VQADataset` に `max_samples` を渡さない。** `VQADataset(root=root, repo_id=rid, seed=seed)` のみ。`VQADataset.__init__` の `max_samples` 引数は存在するが config からは到達不能 `[検証済]` | `datasets/factory.py:452-458`, `datasets/vqa_dataset.py:45-66` | **サブセット制限は prep 時に「小さい `.jsonl` を書く」ことで行う。** 実行時オプションでは絞れない |
| V2 | **`_make_vqa_dataset` は VQA transform を `hydrate()` しない。** `vqa_cfg.data_transforms.inputs` をそのまま `TransformedVQADataset.from_base(..., transforms=...)` に渡す `[検証済]` | `datasets/factory.py:448-458` | VQA 側の `RenderDownsampleFn` は **未 hydrate で動く** → `auto_detect_keys=True` 経路で `observation.images.image0/1` を検出し、worker ごとに 1 回だけ「not hydrated; falling back to key auto-detection」の warning を出す（`transforms_render.py:210-219`）。これは **エラーではなく想定挙動**であり、smoke ではこの行が出ることを「VQA 経路が生きている」signal に使える |
| V3 | `VQADataset.__getitem__` の出力キー: `observation.images.image{i}`, `mask{i}`, `observation.state`(zeros[32]), `action`(zeros[chunk,32]), `conversation`(openai 形式), `source`, `robot_type="vqa"` `[検証済]` | `datasets/vqa_dataset.py:174-188` | jsonl は **1 行 1 オブジェクト**、キー `image`（str または list）、任意 `data_path` / `source` / `conversations`(LLaVA human/gpt) |
| V4 | `VQADataset._resolve_jsonl_path`: `repo_id` は「`.jsonl` ファイルパス」または「`.jsonl` をちょうど 1 個含むディレクトリ」。複数 `.jsonl` を 1 ディレクトリに置くと `ValueError`。非絶対パスは `root` 必須。複数指定は空白区切り文字列 / YAML リスト → `parse_repo_ids` → `MultiVQADataset`（重みなし・uniform） `[検証済]` | `datasets/vqa_dataset.py:68-94`, `datasets/factory.py:99-114, 433-464` | カテゴリを複数使うなら **jsonl を 1 本に連結する**か、**3 本の絶対パスを空白区切り**で渡す |
| V5 | `VQADataset._resolve_image_path` 探索順: 絶対パスかつ存在 → `jsonl_path.parent/image` → (`data_path` があれば) `root/data_path/image`・`data_path/image` → `root/image` `[検証済]` | `datasets/vqa_dataset.py:114-138` | prep は **`image` を「jsonl と同じディレクトリからの相対パス」に正規化**するのが最も堅い。`root` は不要になる |
| V6 | `InternVLAA15VQADatasetConfig.data_transforms.inputs` に `ResizeVQAImagesWithPadFn` は **ちょうど 1 個**。`ResizeVQAImagesWithPadFn` は `height:int` / `width:int` を必須引数に持つ（既定値なし） `[検証済]` | `configuration_internvla_a1_5.py:210-232`, `transforms/core.py:578-591` | overlay の `_insert_before(..., ResizeVQAImagesWithPadFn, ...)` の `len(anchors)==1` assert は現状満たされる |
| V7 | overlay `InternVLAA15ParcVQADatasetConfig` は既に `render_downsample` / `render_target` / `render_p_*` フィールドを持ち、`__post_init__` で `RenderDownsampleFn` を `ResizeVQAImagesWithPadFn` の直前に冪等挿入する。`@VQADatasetConfig.register_subclass("internvla_a1_5_parc")` `[検証済]` | `lerobot_policy_parc/dataset_config.py:128-161` | CLI: `--vqa_dataset.type=internvla_a1_5_parc` で選択、`--vqa_dataset.render_target=128` 等が draccus のフィールド経路でそのまま届く（要 smoke 確認 = V-TBD1） |
| V8 | mixed collate は `cfg.policy.type=="internvla_a1_5" and enable_vqa_loss=True` のとき有効。ベースラインは既に `--policy.enable_vqa_loss=true` `[検証済]` | `datasets/factory.py:586-591`, `scripts/train_ivla_a15.sh:105` | **collate 側の追加変更は不要。**VQA データを与えるだけで mixed 経路に入る |
| V9 | VQA データがあると `MixedMultimodalDataset([robot_ds, vqa_ds], weights=[1-w, w])`（`w = cfg.vqa_dataset.weight`、既定 0.1）。sampler は `MultiMixedWeightedSampler` `[検証済]` | `datasets/factory.py:558-567, 595-601` | weight で robot:VQA の混合比が決まる。§5.7 に A/B 候補とサンプル算 |
| V10 | CLI 形は上流 `launch/internvla_a15_pretrain.sh` のコメント: `--vqa_dataset.type` / `.root` / `.repo_id` / `.weight`。`launch/internvla_a15_finetune_libero.sh` は VQA 未使用（`--vqa_dataset.*` なし、`--policy.enable_vqa_loss=true` のみ） `[検証済]` | `launch/internvla_a15_pretrain.sh:90-95, 156-160` | ベースラインが VQA を使っていないのは「データを与えていない」だけで、受け皿は完備 |
| V11 | RoboInter-VQA の構造（HF ページより。実ファイル一覧は取得時に確認 = V-TBD2）: `robotinter/{Generation,Understanding,Task_planning}/{image,meta}/...`。`meta/` は `.json`（JSON 配列）で 3 フォーマット（`origin_format` / `llava_format` / `smart_resize_format`）。`llava_format` が `conversations`+画像 の LLaVA スキーマ。レコードキー: `id`, `task`, `conversations`(from/value), `images`(パス), `gt`, `h`, `w`, `new_h`, `new_w`。画像は `.zip`（`unzip -o` で in-place 展開）。規模: Generation ~1.3M / Understanding ~85K / Task_planning ~956K、全体 150GB | HF dataset ページ | prep で `.json` 配列 → `.jsonl`、`images` → `image` リネーム、`source` 付与、画像パス正規化が必須 |
| V12 | `vqa_dataset.py` の import は軽い（`torch` / `PIL` / `torchvision.transforms.ToTensor` / `lerobot.transforms.*`）。`tests/conftest.py` の既存スタブで足りる見込み `[検証済]` | `datasets/vqa_dataset.py:1-33` | 手元テストで `VQADataset` を直接叩ける（合成 jsonl + ダミー画像） |
| V13 | VQA 画像は最終的に `ResizeVQAImagesWithPadFn` → `tensor_to_pil_image` → `Qwen3VLProcessor`（smart_resize）へ渡る。`RenderDownsampleFn` は `[C,H,W]` float[0,1] に対して働く `[検証済]` | `transform_internvla_a1_5.py:265-299` | `RenderDownsampleFn` は「128 のボトルネックを通す」目的で機能する。ただし非正方形画像を強制的に 128×128 にすると **アスペクト比が歪む**（R3 / TBD） |

---

## 2. 目的

- ベースライン（VQA なし）の学習・スクリプト・env を無改変で残したまま、**RoboInter-VQA サブセットを混ぜた VQA あり run** を同一 step 数・同一シードで回せるようにする。
- VQA 画像にも既存 `RenderDownsampleFn`（256→128 カーネル振り分け）を通し、採点環境の 128 解像度に合わせた入力分布での **理解能力（言語 grounding）を底上げ**する狙いを検証可能にする。
- `~/data` が消える運用でも `provision_data.sh` 一発でサブセットを入れ直せるようにする。

---

## 3. 変更対象ファイル

### 3.1 新規

| パス | 種別 | 責務 |
|---|---|---|
| `scripts/fetch_vqa_data.sh` | 新規 | RoboInter-VQA の **サブセットだけ** `hf download --include` で `~/data/robointer_vqa` に取得。画像 zip を `unzip -o`。冪等（既にあれば検証のみ）。 |
| `scripts/prepare_vqa_data.py` | 新規 | `.json` 配列 → `.jsonl`、`images`→`image`、`source` 付与、画像パス正規化、`--max-samples` でサブセット。出力は `~/data/robointer_vqa/lerobot_vqa/<name>.jsonl`。冪等。 |
| `env.vqa.example.sh` | 新規 | `IVLA_VQA_*` の既定値テンプレート。`env_train.sh` の後に追加 source する運用（`env.example.sh` は無改変）。 |
| `scripts/train_ivla_a15_vqa.sh` | 新規 | `_train_common.sh` を source し、`train_ivla_a15.sh` と同じ骨格に `--vqa_dataset.*` 群を足した VQA あり run ランチャー。`RUN_NAME` と `OUT_DIR` を別に。 |
| `scripts/smoke_ivla_vqa.sh` | 新規 | 小サブセット + 20 step で mixed 経路が回ることを確認。 |
| `tests/test_vqa_dataset_config.py` | 新規 | overlay の VQA transform チェーン + `prepare_vqa_data.py` の変換 + 合成 jsonl での `VQADataset` 読み込み + 128 へ落ちる、を GPU 不要で検査。 |

ドキュメント用の `docs/` は新設せず、`TRAINING.md` に追記（§5.10）。

### 3.2 既存への追記（3 箇所のみ）

| パス | 操作 | 内容 |
|---|---|---|
| `scripts/provision_data.sh` | 末尾に追記 | 既存の robot データ / 重み処理の **後**に「`IVLA_VQA_ENABLE=1` のときだけ `fetch_vqa_data.sh` → `prepare_vqa_data.py` を呼ぶ」数行。未取得でも既存フローは失敗しない。 |
| `lerobot_policy_parc/dataset_config.py` | 編集 | `InternVLAA15ParcVQADatasetConfig` の docstring の "dead path" 記述を更新。§5.5 の判定で必要になった最小修正のみ（アスペクト比対応 or `keys` 明示など。**上流は触らない**）。 |
| `examples/internvla_a15_libero_finetune/TRAINING.md` | 追記 | 「§9 VQA あり/なし比較」節。データ出所・サブセット・weight・解像度処理・評価方法・ライセンス。 |

### 3.3 上流リポジトリ / submission_template

**編集なし。** 読み取り専用。

---

## 4. フェーズ構成

| Phase | 名称 | 実行場所 | 成果物 | 依存 |
|---|---|---|---|---|
| PV1 | データ取得・変換（手元実装 + クラウド実行） | prep ロジックは手元で書けてテスト可 / 取得はクラウド | `fetch_vqa_data.sh`, `prepare_vqa_data.py`, 変換済み `.jsonl` + 画像ツリー | 親計画 P2 完了（学習 env・robot データ） |
| PV2 | overlay 判定・最小修正 | 手元 | `dataset_config.py` の docstring 更新（＋必要なら最小修正）、`test_vqa_dataset_config.py` | 上流 VQA transform チェーンの読解 |
| PV3 | 学習スクリプト・env | 手元で書く / クラウドで smoke | `env.vqa.example.sh`, `train_ivla_a15_vqa.sh`, `smoke_ivla_vqa.sh`, `provision_data.sh` 追記 | PV1, PV2 |
| PV4 | VQA あり本走 + weight A/B | クラウド | `ivla_a15_libero_combined_vqa` checkpoints、A/B 記録 | 親計画 P4 のベースライン run と同一シード |
| PV5 | ドキュメント・比較評価 | クラウド | `TRAINING.md §9`、track1/2/3 の成功率 + jerk/SPARC の VQA あり/なし比較表 | 親計画 P5〜P7 の評価 pipeline を流用 |

---

## 5. 各コンポーネント詳細

### 5.1 サブセット選定とサイズ見積り

**制約:** `~/data` は既に libero データ + 重みで約 50GB 使用済み。RoboInter-VQA 全体は 150GB。**サブセットは展開後合計 15GB 未満に収める。**

**取得の基本方針（`hf download --include`）:**
- **`meta/` は `llava_format` の `.json` のみ**取得（`smart_resize_format` / `origin_format` は不要）。
- **`image/` は使うカテゴリの `.zip` だけ**取得。zip が分割されている / カテゴリ内でさらに分かれている可能性があるので、取得前に **ファイル一覧をダンプして実サイズを確認する**（V-TBD2。`fetch_vqa_data.sh` の最初のステップ）。

**デフォルトのサブセット（U2 で確定）:**

| 項目 | 選定 | 概算 |
|---|---|---|
| カテゴリ | **`Understanding` + `Task_planning`**（`Understanding` ~85K、`Task_planning` ~956K。空間関係・物体属性・アフォーダンス + 手順分解・長期タスクの言語理解。採点タスクに近い「理解」系） | meta llava_format `.json` 数十〜数百 MB |
| 画像 zip | `Understanding/image/*.zip` + `Task_planning/image/*.zip`（必要分のみ。V-TBD2 で構成確認後に include パターン確定） | 展開後 **要実測（V-TBD2 / V-TBD3）** |
| prep 後サンプル数 | カテゴリ**合計**で `--max-samples 40000`（`seed` 固定で決定的サンプリング。カテゴリ別上限は `--max-samples-per-cat` で調整可、既定は均等割り 20000 / 20000） | `.jsonl` ~数十 MB |
| 合計ディスク | zip + 展開 + jsonl | **展開後 15GB 未満に収める**（`fetch_vqa_data.sh` の `IVLA_VQA_MAX_GIB` ガード。超えたら取得せず失敗 → `--max-samples` と zip を削る） |

`Task_planning` は総数が大きいので、**画像 zip は `--max-samples` に対応する分だけ取れば十分**（zip 全部は展開しない設計を優先。`fetch_vqa_data.sh` の一覧ステップで zip 粒度を見て、必要な zip だけ include する）。

**使わないカテゴリ:**
- `Generation`（1.3M と巨大。今回のサブセットには含めない）。

**`fetch_vqa_data.sh` は `IVLA_VQA_CATEGORIES`（既定 `Understanding`）と `IVLA_VQA_IMAGE_INCLUDE`（zip の include パターン）を env で差し替え可能にする。**

### 5.2 `scripts/fetch_vqa_data.sh`

責務: HF から **サブセットだけ**取得して展開。`fetch_weights.sh` の `dl()` パターン（`snapshot_download` + `allow_patterns`）を踏襲。

疑似仕様:

```bash
set -euo pipefail
source _train_common.sh            # _activate_conda / パス既定
: "${IVLA_VQA_ROOT:=$HOME/data/robointer_vqa}"
: "${IVLA_VQA_HF_REPO:=InternRobotics/RoboInter-VQA}"
: "${IVLA_VQA_CATEGORIES:=Understanding}"          # 空白区切りで複数可
: "${IVLA_VQA_IMAGE_INCLUDE:=}"                    # 未指定なら "<cat>/image/*.zip"

_activate_conda

# 0. 冪等ガード: 展開済みマーカ（.fetch_done）があれば検証だけして exit 0
# 1. ファイル一覧ダンプ（HfApi().list_repo_files, repo_type=dataset）→ サイズ表を stdout。
#    ここで「取得予定パターンにマッチするファイルの合計サイズ」を出し、
#    IVLA_VQA_MAX_GIB（既定 15）を超えるなら**取得せずに exit 1**（安全側）。
# 2. meta: 各カテゴリの llava_format .json を snapshot_download(allow_patterns=[
#      f"robotinter/{cat}/meta/*llava*",  ...])  → $IVLA_VQA_ROOT/raw
# 3. image: snapshot_download(allow_patterns=[ $IVLA_VQA_IMAGE_INCLUDE or f"robotinter/{cat}/image/*.zip" ])
# 4. unzip -o 各 zip を「その zip があるディレクトリ」へ in-place 展開（HF ページの "Extract them in place"）。
#    展開済み zip は削除して容量を戻す（IVLA_VQA_KEEP_ZIP=1 で残す）。
# 5. .fetch_done マーカを書く（取得したカテゴリ・ファイル数・合計バイトを記録）。
# 6. 検証: 各カテゴリの llava_format .json が読め、先頭レコードの images パスが
#    展開後ツリーに存在することを 1 件だけ確認。
```

- **`--repo-type dataset`** を必ず付ける（`fetch_weights.sh` は model なので流用時に注意）。
- オフライン再現性: 取得後は `HF_HUB_OFFLINE=1` でも動く（prep も本走もローカルファイルしか触らない）。

### 5.3 `scripts/prepare_vqa_data.py`

責務: RoboInter-VQA の `llava_format` `.json`（配列）を `VQADataset` が読める `.jsonl` へ変換。冪等。

入力 → 出力:

| 変換 | 内容 |
|---|---|
| フォーマット | `.json`（JSON 配列） → `.jsonl`（1 行 1 オブジェクト） |
| キー | `images`（list） → `image`（V3。list のまま可。単一なら str に落としてもよい） |
| `conversations` | LLaVA 形式（`from`:`human`/`gpt`, `value`）はそのまま保持（`VQADataset` が `llava_to_openai` する） |
| 画像パス | `images` の各要素を **出力 `.jsonl` と同じディレクトリからの相対パス**に正規化（V5 の第 2 候補 `jsonl_path.parent/image` で解決させる）。展開画像ツリーへの相対 or シンボリックリンクを張る |
| `source` | `f"robointer_vqa/{category}"` を付与（ログ・比較用。V3） |
| サブセット | `--max-samples N` + `--seed`（既定 42）で決定的サンプリング（`VQADataset` の `max_samples` が使えないため V1、ここで絞る） |
| 破損レコード | `image` が解決できない / `conversations` が空 のレコードは **スキップしてカウントを stderr に出す**（黙って混ぜない） |
| アスペクト比（R3 対策・§5.5 の判定次第） | オプション `--pad-square`: 画像を短辺基準で正方形にゼロパディングし 256×256 にリサイズして保存（`RenderDownsampleFn` の整数ステップ 2x 経路を有効化し、robot データと同じ前処理にそろえる）。**デフォルト ON を推奨** |

出力レイアウト（V4 / V5 準拠。U2 = 2 カテゴリ）:

```
~/data/robointer_vqa/
  raw/robotinter/Understanding/{meta/*llava*.json, image/...}    # fetch が置く
  raw/robotinter/Task_planning/{meta/*llava*.json, image/...}    # fetch が置く
  lerobot_vqa/
    all.jsonl                     # ← --vqa_dataset.repo_id にこの絶対パスを渡す（2 カテゴリ連結）
    images/...                    # jsonl からの相対。--pad-square 時はここに再生成
```

- **U2: 2 カテゴリを `--merge-into all.jsonl` で 1 本に連結する。** `MultiVQADataset` は per-dataset weight を持たない（V4 / R6）ので、複数 jsonl を空白区切りで渡すとカテゴリ比が制御できない。連結すればカテゴリ比は `--max-samples-per-cat`（既定は `--max-samples` の均等割り = 20000 / 20000）で決まる。
- `source` は連結後もカテゴリ別（`robointer_vqa/Understanding` / `robointer_vqa/Task_planning`）に残す。
- CLI: `--src ~/data/robointer_vqa/raw --out ~/data/robointer_vqa/lerobot_vqa --categories "Understanding Task_planning" --max-samples 40000 --merge-into all.jsonl --pad-square`。
- 冪等: 出力 `.jsonl` が存在し行数が期待どおりなら「検証のみ」で exit 0（`--force` で再生成）。

### 5.4 `scripts/provision_data.sh` への追記

既存の `1/3 データセット` → `2/3 robot_type` → `3/3 重み` → `検証` の **後**に、以下を末尾追加:

```bash
# 4/4 VQA データ（任意。IVLA_VQA_ENABLE=1 のときだけ）
if [ "${IVLA_VQA_ENABLE:-0}" = "1" ]; then
    say "4/4 VQA サブセット（RoboInter-VQA）"
    bash "$IVLA_DIR/scripts/fetch_vqa_data.sh"
    python "$IVLA_DIR/scripts/prepare_vqa_data.py" \
        --src  "${IVLA_VQA_ROOT:-$HOME/data/robointer_vqa}/raw" \
        --out  "${IVLA_VQA_ROOT:-$HOME/data/robointer_vqa}/lerobot_vqa" \
        --categories "${IVLA_VQA_CATEGORIES:-Understanding Task_planning}" \
        --max-samples "${IVLA_VQA_MAX_SAMPLES:-40000}" \
        --merge-into all.jsonl \
        --pad-square
else
    say "4/4 VQA データはスキップ（IVLA_VQA_ENABLE=1 で有効化）"
fi
```

- 既存挙動は完全に不変（`IVLA_VQA_ENABLE` 未設定なら 1 行 echo のみ）。
- 失敗しても robot 側の provision は既に終わっているので、ベースライン run はいつでも開始できる。

### 5.5 overlay 変更が必要かの判定（PV2）

**読むべき上流箇所:**
- `configuration_internvla_a1_5.py:197-247` — `InternVLAA15VQADatasetConfig` の `data_transforms` 定義と `__post_init__`。`ResizeVQAImagesWithPadFn` が 1 個であること（V6）。
- `transforms/core.py:578-591` — `ResizeVQAImagesWithPadFn.__call__`（キー判定 `k.startswith(OBS_IMAGES) or k == OBS_IMAGE or "image" in k`）。
- `transform_internvla_a1_5.py:265-330` — `InternVLAA15VQAProcessorTransformFn`（`_collect_images` → `tensor_to_pil_image` → Qwen processor。V13）。
- `datasets/factory.py:448-464` — VQA transform が hydrate されないこと（V2）。

**判定チェックリスト（Generator が PV2 でコードを読んで確認する）:**

| 確認項目 | 期待 | 外れたときの最小修正（overlay のみ） |
|---|---|---|
| `InternVLAA15ParcVQADatasetConfig()` の `data_transforms.inputs` に `RenderDownsampleFn` が **1 個**、`ResizeVQAImagesWithPadFn` の直前 | 満たす（現状） | `_insert_before` のロジックを触らず、`__post_init__` の順序を見直す |
| `ResizeVQAImagesWithPadFn` が実チェーンに **ちょうど 1 個**（`_insert_before` の `len(anchors)!=1` で落ちない） | 満たす（V6） | 上流が将来 2 個にしたら overlay 側で「最初の 1 個」を対象にするよう緩和 |
| CLI `--vqa_dataset.type=internvla_a1_5_parc --vqa_dataset.render_target=128 ...` が届く | draccus のフィールド経路で届くはず（V7） | 届かなければ overlay に無し。届かない理由が draccus のネスト解決なら env で prep 済み jsonl を作る側に寄せる（本質的には overlay 修正不要） |
| 未 hydrate でも `RenderDownsampleFn` が VQA 画像キーを拾う | `auto_detect_keys=True` で拾う（V2） | 拾えないなら overlay で `keys=["observation.images.image0", "observation.images.image1", "observation.images.image2"]` を明示セット（`__post_init__` 内） |
| 非正方形 VQA 画像でアスペクト比が歪む（V13 / R3） | `--pad-square` で prep 時に正方形化すれば回避 | prep 側で対応（overlay 修正不要）。prep で対応しない方針なら overlay に `RenderDownsampleFn` の「アスペクト保持モード」を足す検討（**スコープ拡大なので TBD として保留**） |

**確定している overlay 変更:** `InternVLAA15ParcVQADatasetConfig` の docstring の「**現行レシピでは dead path**（計画 §5.1 注記 / D5）」を削り、「`train_ivla_a15_vqa.sh` から `--vqa_dataset.type=internvla_a1_5_parc` で使う。ベースライン `train_ivla_a15.sh` は依然 VQA データを与えないので、そちらでは引き続き未使用」に更新する。`tests/test_dataset_config.py` の `test_vqa_*` の docstring も同様に更新（テストの assert は不変）。

### 5.6 `env.vqa.example.sh`

`env.example.sh` は無改変。VQA 専用の変数だけを別テンプレートに置き、`env_train.sh` の後で追加 source する運用にする（`cp env.vqa.example.sh env.vqa.sh` して編集 → `source env_train.sh && source env.vqa.sh`）。

```bash
# --- VQA あり run 用の追加 env（env_train.sh の後に source する）---
export IVLA_VQA_ENABLE=1
export IVLA_VQA_ROOT="${HOME}/data/robointer_vqa"
export IVLA_VQA_HF_REPO="InternRobotics/RoboInter-VQA"
export IVLA_VQA_CATEGORIES="Understanding Task_planning"   # U2。空白区切り
export IVLA_VQA_MAX_SAMPLES=40000                  # カテゴリ合計。既定は均等割り
export IVLA_VQA_MAX_GIB=15                         # fetch がこの合計サイズを超えたら中止
# prep 出力（train スクリプトが --vqa_dataset.repo_id に渡す。絶対パス or 空白区切り複数）
# U2: 2 カテゴリを 1 本に連結する（--merge-into all.jsonl）。MultiVQADataset の
# uniform 制約（V4 / R6）を避けられる。
export IVLA_VQA_REPO_ID="${IVLA_VQA_ROOT}/lerobot_vqa/all.jsonl"
export IVLA_VQA_WEIGHT=0.10                        # U4: 本走はこの 1 値のみ。A/B しない
export IVLA_VQA_RENDER_TARGET=128                  # RenderDownsampleFn(VQA) の目標解像度
export IVLA_VQA_RUN_NAME="ivla_a15_libero_combined_vqa"
```

- `IVLA_VQA_RENDER_P_*` は **ベースラインと同じ既定**（box 0.45 / triangle 0.30 / cubic 0.15 / nearest 0.10）を `train_ivla_a15_vqa.sh` 側の `:=` で持たせる。A/B するなら env で上書き。
- 既存 `env.example.sh` / `env_train.sh` には 1 文字も追加しない。

### 5.7 `scripts/train_ivla_a15_vqa.sh`

`train_ivla_a15.sh` を土台に、**差分を最小化**して書く（コピペ改変ではあるが、共通ヘルパーは `_train_common.sh` の source で共有）。

**ベースラインと同じにする部分**（`_train_common.sh` 経由で再利用）:
- `_activate_conda` / `_resolve_paths` / `_assert_env` / `_maybe_resume` / `_wandb_args`
- `_start_vram_sampler` / `_start_checkpoint_janitor` / `_prune_checkpoint_states` / `_summarize_run`
- `accelerate launch scripts/train_entry.py` 入口、`--policy.*` 群（`enable_vqa_loss=true` を含め **全部同一**）
- `--dataset.*` 群（robot 側の `internvla_a1_5_parc` + `render_*` も **同一**）
- ハイパラ（`IVLA_STEPS=30000` / `warmup=600` / `decay=30000` / `lr` / `save_freq` / `IVLA_VLM_LR_SCALE=0.1` など）は env 由来でベースラインと共有

**ベースラインと変える部分:**

| 項目 | ベースライン | VQA あり |
|---|---|---|
| `RUN_NAME` | `ivla_a15_libero_combined` | `${IVLA_VQA_RUN_NAME:-ivla_a15_libero_combined_vqa}`（→ `OUT_DIR` / `LOG_FILE` が自動で別になる） |
| `--vqa_dataset.*` | なし | 下記を追加 |
| resume | 自分の `checkpoints/last` から | **同じ**（別 `OUT_DIR` なので baseline を汚さない） |

**追加する CLI 引数:**

```
--vqa_dataset.type=internvla_a1_5_parc
--vqa_dataset.repo_id="$IVLA_VQA_REPO_ID"          # 絶対 .jsonl パス（複数なら空白区切り）
--vqa_dataset.root=""                              # image を jsonl 相対に正規化済みなので空でよい（V5）
--vqa_dataset.weight="${IVLA_VQA_WEIGHT:-0.10}"
--vqa_dataset.seed="$IVLA_SEED"
--vqa_dataset.render_downsample=true
--vqa_dataset.render_target="${IVLA_VQA_RENDER_TARGET:-128}"
--vqa_dataset.render_p_box="${IVLA_VQA_RENDER_P_BOX:-0.45}"
--vqa_dataset.render_p_triangle="${IVLA_VQA_RENDER_P_TRIANGLE:-0.30}"
--vqa_dataset.render_p_cubic="${IVLA_VQA_RENDER_P_CUBIC:-0.15}"
--vqa_dataset.render_p_nearest="${IVLA_VQA_RENDER_P_NEAREST:-0.10}"
```

**新スクリプト内だけの追加ガード（`_train_common.sh` は無改変なので、この関数は新スクリプトにローカル定義）:**

```bash
_assert_vqa_data() {
    local first; first="$(echo "$IVLA_VQA_REPO_ID" | awk '{print $1}')"
    if [ ! -e "$first" ]; then
        echo "VQA jsonl が無い: $first" >&2
        echo "  IVLA_VQA_ENABLE=1 bash scripts/provision_data.sh を先に実行すること" >&2
        exit 1
    fi
}
```

**weight とサンプル算（`batch_size=8`, `steps=30000` の実測ベース）:**

**U4: weight A/B はやらない。本走は `IVLA_VQA_WEIGHT=0.10` の 1 本のみ。** 下表は参考。

| `IVLA_VQA_WEIGHT` | VQA サンプル / 1000 step（`1000 * 8 * w`） | 30k step 合計 | サブセット 40k に対する周回 |
|---|---|---|---|
| 0.05 | ~400 | ~12,000 | ~0.30 epoch |
| **0.10（本走で使う唯一の値）** | ~800 | ~24,000 | ~0.60 epoch |
| 0.15 | ~1,200 | ~36,000 | ~0.90 epoch |

- robot 側の実効サンプル数は `w` の分だけ減る（w=0.10 で 240k → 216k robot draws / 30k step）。**VQA あり/なしの比較は step 数を固定して行う**（robot draw 数の差は `TRAINING.md §9` に記録する）。
- 既定 **0.10** は上流 pretrain のコメント既定 0.15 よりやや保守的（理解タスクの過学習と robot 性能低下のバランス）。`env.vqa.sh` で `IVLA_VQA_WEIGHT` を上書きすれば別値でも回せるが、計画上は回さない。

### 5.8 `scripts/smoke_ivla_vqa.sh`

`smoke_ivla.sh` と同じ枠組み（20 step、専用 `RUN_NAME`、ログを本走と分離）で、**`train_ivla_a15_vqa.sh` を呼ぶ**。事前に極小サブセットを用意:

```bash
# smoke 用に 200 サンプルだけの jsonl を作る（既にあれば再利用）
python scripts/prepare_vqa_data.py --src ... --out ~/data/robointer_vqa/lerobot_vqa_smoke \
    --categories Understanding --max-samples 200 --pad-square
IVLA_VQA_REPO_ID=~/data/robointer_vqa/lerobot_vqa_smoke/understanding.jsonl \
IVLA_VQA_WEIGHT=0.5 \
RUN_NAME=smoke_ivla_vqa IVLA_BS=1 IVLA_STEPS=20 IVLA_SAVE_FREQ=20 IVLA_LOG_FREQ=1 \
  bash scripts/train_ivla_a15_vqa.sh
```

**ログで確認するポイント（`chk_cmd` で grep）:**
- `Mixed dataset created` または `MixedMultimodalDataset`（`factory.py:566`）が出ている
- `[make_vqa_dataset] all_repo_ids=` に prep 済み jsonl パスが出ている
- VQA weight のログ（`dataset_weights=[0.5, 0.5]` 相当 / `MixedMultimodalDataset` の `__repr__`）
- `RenderDownsampleFn was not hydrated; falling back to key auto-detection [...image0...]`（V2。VQA 経路で `RenderDownsampleFn` が実際に画像に触れている証拠）
- ベースライン smoke の既存チェック（monkeypatch / optimizer 2 グループ / WAN 重みロード / loss 有限 / checkpoint 保存）は **すべて維持**
- **`loss` に nan/inf が無い**（VQA バッチと robot バッチの両方が流れて有限）

### 5.9 テスト（`tests/test_vqa_dataset_config.py`）

GPU 不要・`tests/conftest.py` のスタブ前提（V12）。`test_dataset_config.py` の隣に置き、既存の `overlay` / `offline_processors` フィクスチャを再利用。

| テスト | 内容 |
|---|---|
| `test_vqa_render_downsample_is_before_vqa_resize` | 既存（`test_dataset_config.py` にある）。docstring だけ更新、assert は不変 |
| `test_vqa_render_downsample_count_is_one` | `InternVLAA15ParcVQADatasetConfig()` の `data_transforms.inputs` に `RenderDownsampleFn` が **ちょうど 1 個**、`__post_init__` 二重呼びでも 1 個（冪等） |
| `test_vqa_render_probabilities_propagated` | `render_p_*` が `RenderDownsampleFn` に伝わる（`probabilities() == {box:0.45, ...}`、`p_nearest == 0.10`） |
| `test_vqa_draccus_roundtrip` | `draccus.encode` → `decode(VQADatasetConfig, payload)` で `render_downsample` discriminator が残り、チェーン順が保存される（`train_config.json` 往復。F3） |
| `test_prepare_vqa_json_to_jsonl` | 合成 `.json` 配列（`images`/`conversations`/`h`/`w` 付き）を `prepare_vqa_data.py` の変換関数に通し、`.jsonl` が 1 行 1 オブジェクト・`image` キーになる・`source` が付く・破損レコードがスキップされる |
| `test_prepare_vqa_image_path_resolves` | 変換後 `.jsonl` を合成ダミー画像（PIL で 320×180 の PNG を生成）と一緒に置き、`VQADataset(root=None, repo_id=<abs jsonl>)` が読める（`_resolve_image_path` が `jsonl_path.parent/image` で解決。V5） |
| `test_prepare_vqa_pad_square` | `--pad-square` 相当の関数が 320×180 → 256×256（正方形・パディング）を出す |
| `test_vqa_sample_downsampled_to_target` | `VQADataset` の 1 サンプル（256×256 ダミー画像）を overlay チェーン（`RenderDownsampleFn` 部分のみ）に通すと `observation.images.image0` が 128×128 になる。`p_nearest=1.0` 固定で位相ジッタも確認 |
| `test_baseline_vqa_config_still_dead_path_in_train_ivla_a15` | `scripts/train_ivla_a15.sh` に `--vqa_dataset` が **含まれない**ことを grep で確認（ベースライン不変の回帰） |

**conftest 変更が要る場合:** `prepare_vqa_data.py` が `datasets` ライブラリを使わず標準 `json` だけで書けるようにする（V12。スタブ済み `datasets` に依存しない）。もし `VQADataset` import で未スタブの重い依存が出たら `_STUB_SPECS` に 1 行足す（`conftest.py` は recipe テスト基盤なので追記可。上流・4 ファイルには当たらない）。

### 5.10 ドキュメント（`TRAINING.md §9`）

`§8` の後に「§9 VQA あり/なし比較」を追記。含める内容:

1. **データ出所と代替理由:** 公式 example VQA data は上流 README の TODO で未公開。同組織の `InternRobotics/RoboInter-VQA` で代替。DROID / RH20T 由来の実ロボット画像 + VQA アノテーション。
2. **サブセット選定:** `Understanding` + `Task_planning` カテゴリ、`llava_format` メタ、合計 `--max-samples 40000`、`--pad-square`、2 カテゴリを `all.jsonl` に連結。概算ディスク（実測値を記入）。再取得は `IVLA_VQA_ENABLE=1 bash scripts/provision_data.sh`。
3. **weight と混合:** `--vqa_dataset.weight=0.10`（本走で使う唯一の値。A/B はしない）。`MixedMultimodalDataset` の仕組み（V9）とサンプル算（§5.7 の表）。robot draw 数がベースライン比で ~10% 減ることを明記。
4. **解像度処理:** VQA 画像も `RenderDownsampleFn` で 256→128 の 4 カーネル振り分けを通す（robot と同一確率）。狙い = 採点環境 128 解像度での理解能力向上。**追加拡張（JPEG 劣化・blur）は未実施**、将来の A/B 候補。
5. **比較評価:** 親計画 §5 の pipeline を流用。**VQA あり run（`ivla_a15_libero_combined_vqa`）と VQA なし run（`ivla_a15_libero_combined`）を同一シード・同一 step・同一 chunking 設定**で track1/2/3 各 N エピソード評価し、成功率 + jerk / SPARC / path length を並べる。チェックポイント選択（D7）も両 run で同じ手順。
6. **ライセンス:** RoboInter-VQA の利用条件（HF ページの license 記載を転記）、由来データセット（DROID = CC-BY / RH20T = 各自条件）に触れる。`THIRD_PARTY_LICENSES.md` にも 1 行追記。**非商用条件の可能性があるので提出前に要確認**（TBD）。

---

## 6. 実装手順

各ステップは独立してテスト可能な粒度。

### PV1 — データ取得・変換

1. **`scripts/prepare_vqa_data.py` を書く。** 変換ロジック（`.json`→`.jsonl`、`images`→`image`、`source`、パス正規化、`--max-samples`、`--pad-square`、破損スキップ）を**純関数**に分離し、`if __name__ == "__main__"` は薄く。標準 `json` のみ使用（`datasets` に依存しない）。
2. **`tests/test_vqa_dataset_config.py` の prep 系テストを書く**（`test_prepare_vqa_*`）。合成 `.json` + PIL ダミー画像で手元 CPU で緑にする。
3. **`scripts/fetch_vqa_data.sh` を書く。** まず「ファイル一覧 + サイズ表」ステップだけ実装してクラウドで実行し、**`Understanding` の実サイズ・zip 構成を確認**（V-TBD2 を解消）。結果を `env.vqa.example.sh` の `IVLA_VQA_IMAGE_INCLUDE` 既定と §5.1 の概算表に反映。
4. **`fetch_vqa_data.sh` の取得・展開・冪等ガード・検証を仕上げる。** クラウドで実行し、`~/data/robointer_vqa/raw` にサブセットが落ちることを確認。
5. **`prepare_vqa_data.py` をクラウドで実行**し、`~/data/robointer_vqa/lerobot_vqa/understanding.jsonl` を生成。`python -c "from lerobot.datasets.vqa_dataset import VQADataset; d=VQADataset(root=None, repo_id='<abs>'); print(len(d)); print(d[0].keys())"` で読めることを確認。

### PV2 — overlay 判定

6. **上流 VQA transform チェーンを読む**（§5.5 の「読むべき箇所」）。§5.5 のチェックリストを埋め、必要な overlay 最小修正を確定。
7. **`lerobot_policy_parc/dataset_config.py` の docstring を更新**（"dead path" → "train_ivla_a15_vqa.sh から使用"）。§5.5 で必要と判定した最小修正があれば同時に入れる（`keys` 明示 or アスペクト対応は prep 側優先）。
8. **`tests/test_vqa_dataset_config.py` の overlay 系テストを書く**（`test_vqa_render_*` / `test_vqa_draccus_roundtrip` / `test_vqa_sample_downsampled_to_target` / `test_baseline_vqa_config_still_dead_path_in_train_ivla_a15`）。`pytest tests/ -q` が全緑（既存テストが 1 件も壊れないこと）。

### PV3 — 学習スクリプト・env

9. **`env.vqa.example.sh` を追加**（§5.6）。`env.example.sh` / `env_train.sh` は無改変であることを `git diff` で確認。
10. **`scripts/train_ivla_a15_vqa.sh` を書く。** `train_ivla_a15.sh` との `diff` が「`RUN_NAME` の既定・`_assert_vqa_data`・`--vqa_dataset.*` 群の追加」に限定されることを確認。`_train_common.sh` は source のみ。
11. **`scripts/provision_data.sh` に `4/4` ブロックを追記**（§5.4）。`IVLA_VQA_ENABLE` 未設定で従来と完全に同じ出力になることを確認（既存の `verify_train_env.py` 実行位置は変えない）。
12. **`scripts/smoke_ivla_vqa.sh` を書く**（§5.8）。

### PV4 — 本走 + A/B

13. **クラウドで `IVLA_VQA_ENABLE=1 bash scripts/provision_data.sh`** を実行し、VQA サブセットまで含めて provision が緑。
14. **`bash scripts/smoke_ivla_vqa.sh`** を実行。§5.8 のログチェックが全 PASS。**これが通るまで本走しない。**
15. **`RenderDownsampleFn`(VQA) の実データ目視**（親計画 P3-22 の VQA 版）。`IVLA_VQA_RENDER_P_NEAREST=1.0` で 1 サンプル取り、224 リサイズ後の VQA 画像を PNG 保存 → 明らかにエイリアスしていること。既定確率に戻して再保存。2 枚を `artifacts/` に残す。
16. **VQA あり本走**（`RUN_NAME=ivla_a15_libero_combined_vqa`、`IVLA_VQA_WEIGHT=0.10`）。ベースライン run と **同一シード（42）・同一 step（30000）**。`_maybe_resume` / janitor / vram sampler は自動で効く。
17. **（U4 で削除）weight A/B はやらない。** 本走はステップ 16 の 1 本のみ。

### PV5 — 比較評価・ドキュメント

18. **親計画 §5 の pipeline で VQA あり/なし を比較評価。** 同一シード・同一 chunking 設定で track1/2/3。成功率 + jerk / SPARC / path length の表を作る。
19. **`TRAINING.md §9` を追記**（§5.10）。実測ディスク・実測サンプル数・比較表・ライセンス。`THIRD_PARTY_LICENSES.md` に RoboInter-VQA を 1 行追記。
20. **`README.md` の「クラウドで行う作業」表に PV1〜PV5 を 1 行追記**（任意・軽微）。

---

## 7. 検証方法

### 7.1 Generator 自己チェック（手元 / GPU 不要）

- [ ] `pytest examples/internvla_a15_libero_finetune/tests/ -q` が全緑（**既存テストの pass 数が減らない**）
- [ ] `git diff --stat` の対象が **`provision_data.sh` / `dataset_config.py` / `TRAINING.md` / `README.md`（追記のみ）＋新規ファイル**に限定される
- [ ] `git diff scripts/train_ivla_a15.sh scripts/_train_common.sh env_train.sh env.example.sh` が **空**
- [ ] `git -C ~/InternVLA-A-series status` が clean（上流無改変）
- [ ] `git diff submission_template/` が空
- [ ] `prepare_vqa_data.py` が標準 `json` のみで `.json`配列→`.jsonl`、`images`→`image`、`source` 付与、破損スキップを行う（単体テスト）
- [ ] 合成 `.jsonl` + ダミー画像で `VQADataset` が読め、`observation.images.image0` が取れる
- [ ] `InternVLAA15ParcVQADatasetConfig` の transform チェーンに `RenderDownsampleFn` がちょうど 1 個、`ResizeVQAImagesWithPadFn` の直前、冪等
- [ ] `render_p_*` が CLI フィールド経由で `RenderDownsampleFn` に伝わる（draccus 往復テスト）
- [ ] VQA サンプル画像が overlay チェーンで 128×128 に落ちる
- [ ] `train_ivla_a15_vqa.sh` と `train_ivla_a15.sh` の `diff` が `--vqa_dataset.*` / `RUN_NAME` / `_assert_vqa_data` に限定
- [ ] `provision_data.sh` は `IVLA_VQA_ENABLE` 未設定で従来と同一の出力

### 7.2 クラウドでのみ確認できること（Evaluator）

- [ ] `fetch_vqa_data.sh` のサイズガードが効き、サブセット展開後の `~/data/robointer_vqa` が **15GB 未満**
- [ ] `IVLA_VQA_ENABLE=1 bash scripts/provision_data.sh` が冪等に完了（2 回目は「検証のみ」）
- [ ] `smoke_ivla_vqa.sh` が exit 0 で、ログに `MixedMultimodalDataset` / VQA repo_id / `RenderDownsampleFn ... auto-detection` が出る
- [ ] smoke で `loss` に nan/inf 無し、checkpoint 保存成功、ベースライン smoke の既存チェックも全 PASS
- [ ] `RenderDownsampleFn`(VQA) の実画像 2 枚（nearest 固定 / 既定確率）が `artifacts/` にある
- [ ] VQA あり本走が 30000 step 到達、`train_config.json` に `vqa_dataset` セクション（`type=internvla_a1_5_parc` / `weight` / `render_*`）が記録されている
- [ ] VQA あり run と VQA なし run が同一シード・同一 step で、track1/2/3 の成功率 + jerk/SPARC/path length の比較表がある
- [ ] `TRAINING.md §9` に実測ディスク・実測サンプル数・ライセンスが埋まっている（`TBD` が残っていない）

---

## 8. sprint contract（完了条件）

各 SC は Generator（機械的に実行して exit code / grep で判定）と Evaluator（成果物の性質を確認）の両方が解釈できる形で書く。

### SC-V1 — データ取得・変換（PV1）

- **Generator:** `pytest tests/test_vqa_dataset_config.py -q` が exit 0。`prepare_vqa_data.py --help` が exit 0。
- **Evaluator:**
  - `prepare_vqa_data.py` が `.json` 配列 → 1 行 1 オブジェクト `.jsonl`、`images`→`image`、`source` 付与、`--max-samples` の決定的サンプリング、破損レコードのスキップ + カウント表示を行う
  - `fetch_vqa_data.sh` が `--repo-type dataset` かつ `allow_patterns` で **カテゴリ / llava_format / 画像 zip を絞って**取得し、`IVLA_VQA_MAX_GIB` 超過時は取得せず失敗する
  - `fetch_vqa_data.sh` / `prepare_vqa_data.py` が冪等（既存を検出して「検証のみ」）
  - 画像パスが `VQADataset._resolve_image_path` の `jsonl_path.parent/image` で解決できる形（`root` 不要）

### SC-V2 — overlay（PV2）

- **Generator:** `pytest tests/ -q` が exit 0 で、**pass 数が実装前以上**。
- **Evaluator:**
  - 上流 `InternVLA-A-series` が無改変（`git status` clean）
  - `InternVLAA15ParcVQADatasetConfig` の transform チェーンに `RenderDownsampleFn` が **ちょうど 1 個**、`ResizeVQAImagesWithPadFn` の**直前**、`__post_init__` 二重呼びで冪等
  - `render_p_nearest` の既定が **0.10**、4 カーネル確率の合計が 1.0
  - docstring から "dead path" 記述が消え、`train_ivla_a15_vqa.sh` から使う旨に更新されている
  - overlay 以外（上流・submission_template・4 つの無改変ファイル）に変更が無い

### SC-V3 — 学習スクリプト・env（PV3）

- **Generator:** `bash -n scripts/train_ivla_a15_vqa.sh scripts/fetch_vqa_data.sh scripts/smoke_ivla_vqa.sh` が exit 0。`git diff scripts/train_ivla_a15.sh scripts/_train_common.sh env_train.sh env.example.sh` が空。
- **Evaluator:**
  - `train_ivla_a15_vqa.sh` が `_train_common.sh` を source し、`_activate_conda` / `_maybe_resume` / janitor / vram sampler を再利用
  - `--vqa_dataset.type=internvla_a1_5_parc` / `.repo_id` / `.weight` / `.render_*` を渡し、`--policy.enable_vqa_loss=true` を維持
  - `RUN_NAME` / `OUT_DIR` / `LOG_FILE` がベースラインと別（baseline の checkpoint を汚さない）
  - `provision_data.sh` の追記が **末尾のみ**、`IVLA_VQA_ENABLE` でガードされ、未設定時の既存挙動が不変
  - `env.example.sh` が無改変で、VQA 変数は `env.vqa.example.sh` に分離

### SC-V4 — smoke（PV4 前段）

- **Generator:** `bash scripts/smoke_ivla_vqa.sh` が exit 0。
- **Evaluator:** smoke ログに以下が出る
  - `MixedMultimodalDataset` 構築（`factory.py:566`）と VQA repo_id
  - `RenderDownsampleFn` が VQA 画像キー（`observation.images.image0` 等）に作用（auto-detection の warning 行）
  - `loss` が有限、checkpoint 保存成功
  - ベースライン smoke の既存チェック（monkeypatch / optimizer 2 グループ / WAN 重み / nan-inf）が全 PASS

### SC-V5 — 本走（PV4）

- **Generator:** VQA あり run が指定 step（30000）到達、checkpoint が 5〜6 個。
- **Evaluator:**
  - `train_config.json` に `vqa_dataset`（`type` / `weight` / `render_downsample` / `render_p_*`）が記録
  - シード・step・chunking 設定がベースライン run と一致（比較可能性）
  - `RenderDownsampleFn`(VQA) の実画像 2 枚が `artifacts/` にある

### SC-V6 — 比較評価・ドキュメント（PV5）

- **Generator:** `TRAINING.md` に `## 9` 節が存在し、`TBD` 文字列を含まない。
- **Evaluator:**
  - track1/2/3 の成功率 + jerk / SPARC / path length が **VQA あり / なしで同一シード**で並んだ表がある
  - 最終的にどちらを提出候補にするか（および weight）が表から説明できる
  - データ出所（RoboInter-VQA 代替理由 = 公式未公開）・サブセット・ライセンス（非商用条件の有無）が明記され、`THIRD_PARTY_LICENSES.md` に反映

---

## 9. リスクと副作用

| # | リスク | 影響 | 緩和 |
|---|---|---|---|
| R1 | **RoboInter-VQA のサブセットでも `~/data` を圧迫**（既に 50GB 使用済み・揮発領域） | provision が容量不足で失敗、または展開途中で中断 | `fetch_vqa_data.sh` が取得前に合計サイズを見積もり `IVLA_VQA_MAX_GIB`（既定 15）超過で中止。zip は展開後に削除。`--max-samples` で jsonl を絞る |
| R2 | **RoboInter-VQA の実ファイル構成が HF ページの記述と違う**（zip 分割・カテゴリ内サブフォルダ・`images` パスのプレフィックス） | prep が画像を解決できず全レコードスキップ | PV1 手順 3 で**まずファイル一覧をダンプ**してから include パターンと prep のパス正規化を確定（V-TBD2） |
| R3 | **非正方形 VQA 画像を `RenderDownsampleFn` が 128×128 に強制 → アスペクト比が歪む**（V13）。上流 `ResizeVQAImagesWithPadFn` は本来 letterbox で保持 | VQA 画像の物体形状が歪み、grounding 学習にノイズ | `prepare_vqa_data.py --pad-square` を **既定 ON**（短辺パディング → 256×256）。robot データと同じ「正方形 256 → 128 整数ステップ」経路に載る。overlay は触らない |
| R4 | **VQA を混ぜると robot サンプル数が `w` 分減る**（step 固定なので） | ベースライン比で robot 学習量が 5〜15% 減り、単純比較で不利に見える | A/B は step 固定で行い、robot draw 数の差を記録。weight 0.05〜0.15 に留める。必要なら VQA あり run だけ step を微増する案を TBD に |
| R5 | **`--vqa_dataset.render_*` が draccus のネスト解決で CLI から届かない** | `RenderDownsampleFn` が既定値で動く（128/0.45/0.30/0.15/0.10） → 実害は小さいが A/B できない | smoke ログで `RenderDownsampleFn` の実パラメータを確認（V-TBD1）。届かなければ overlay の既定値を正とし、A/B は overlay 側 or 別 config で |
| R6 | **`MultiVQADataset` は per-dataset weight を持たない**（V4。uniform） | 複数カテゴリを使うとき比率を制御できない | 既定は単一カテゴリ（`Understanding`）。複数使うなら prep で `--merge-into` して 1 jsonl にし、カテゴリ比は `--max-samples` をカテゴリ別に指定して調整 |
| R7 | **RoboInter-VQA が非商用ライセンス / 由来データ（RH20T 等）に再配布制限** | 提出物への影響（ただし VQA データは提出物に同梱しない。学習にのみ使用） | `TRAINING.md §9` と `THIRD_PARTY_LICENSES.md` に明記。提出前にライセンス条項を確認（TBD-Q3） |
| R8 | **`tests/conftest.py` のスタブが `vqa_dataset.py` の import 経路をカバーしきれない** | 手元テストが collection error | V12 で import は軽いと確認済み。足りなければ `_STUB_SPECS` に追記（recipe テスト基盤なので可） |
| R9 | **VQA バッチと robot バッチの collate 差**（`_multimodal_collate`）で稀に shape 不整合 | 本走が数百 step 後に落ちる | smoke（20 step）で両種のバッチが流れることを確認。`enable_vqa_loss=true` は上流既定経路なので新規リスクは小さい（V8） |
| R10 | **`provision_data.sh` 追記が既存フローを壊す** | ベースライン運用に影響 | 追記は末尾のみ・`IVLA_VQA_ENABLE` ガード・`verify_train_env.py` の実行位置は変えない。SC-V3 で回帰確認 |

---

## 10. 未確定事項（TBD）とユーザーへの質問

推測で埋めず、以下は PV1 のクラウド実行 or ユーザー確認で解消する。

### 10.1 調査で解消する（PV1）

| # | 未確定 | 解消方法 |
|---|---|---|
| V-TBD1 | `--vqa_dataset.render_target` 等が draccus のネストフィールド経路で CLI から `InternVLAA15ParcVQADatasetConfig` に届くか | PV3 の smoke ログで `RenderDownsampleFn` の実パラメータを確認。届かなければ R5 の緩和へ |
| V-TBD2 | RoboInter-VQA の `Understanding` カテゴリの実ファイル構成（zip 数・分割・サイズ、`llava_format` の `.json` ファイル名、`images` パスのプレフィックス） | `fetch_vqa_data.sh` の「ファイル一覧 + サイズ」ステップをクラウドで先行実行 |
| V-TBD3 | `Understanding` サブセット（`--max-samples 40000` / `--pad-square`）の展開後実ディスク | PV1 手順 4〜5 の実測 |
| V-TBD4 | `llava_format` レコードの `images` が単数か複数か、`conversations` の画像トークン（`<image>`）の有無と位置 | 先頭 100 レコードを prep のドライランでダンプ |

### 10.2 ユーザーに確認したいこと

- ~~**TBD-Q1（サブセットの範囲）**~~ → **解決（U2）**: `Understanding` + `Task_planning` の 2 カテゴリ、合計 40k サンプル、展開後 15GB 以内。`Generation` は使わない。
- ~~**TBD-Q2（VQA weight のデフォルト）**~~ → **解決（U4）**: 本走は `0.10` の 1 本のみ。A/B しない。
- **TBD-Q3（ライセンス）:** RoboInter-VQA / 由来データ（DROID / RH20T）の利用条件を確認済みか。非商用・研究利用限定だった場合、コンペ提出（学習にのみ使用、データ非同梱）で問題ないという理解で良いか。**→ PV5 のドキュメント化時にユーザーへ再確認。実装はブロックしない。**
- ~~**TBD-Q4（`--pad-square` の是非）**~~ → **解決（U3）**: prep 時に正方形パディング（256×256）。既定 ON。overlay にアスペクト保持モードは足さない。
- ~~**TBD-Q5（A/B の計算予算）**~~ → **解決（U4）**: A/B しない。
- **TBD-Q6（比較のベースライン run）:** 比較対象は `ivla_a15_libero_combined`（seed 42）で確定という前提で進める。VQA あり run もこの seed に合わせる。**PV5 で相違があればユーザーに確認。**

---

## 11. 参考: このタスクで読むべき最小集合

| 目的 | ファイル:行 |
|---|---|
| VQA データセットの jsonl / 画像解決 / 出力キー | `~/InternVLA-A-series/src/lerobot/datasets/vqa_dataset.py:45-94, 114-138, 174-188` |
| VQA を robot と混ぜる経路・hydrate されない点 | `src/lerobot/datasets/factory.py:425-464, 558-601` |
| VQA DatasetConfig と transform チェーン | `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py:197-247` |
| VQA Resize / 画像キー判定 | `src/lerobot/transforms/core.py:578-591` |
| VQA processor（PIL 変換 → Qwen） | `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py:265-330` |
| VQADatasetConfig 基底（repo_id/root/weight/seed） | `src/lerobot/configs/default.py:56-68` |
| CLI 形の参照 | `~/InternVLA-A-series/launch/internvla_a15_pretrain.sh:90-95, 156-160` |
| overlay の VQA config（既存） | `examples/internvla_a15_libero_finetune/lerobot_policy_parc/dataset_config.py:128-161` |
| RenderDownsampleFn 本体 | `examples/internvla_a15_libero_finetune/lerobot_policy_parc/transforms_render.py` |
| ベースライン学習スクリプト | `examples/internvla_a15_libero_finetune/scripts/train_ivla_a15.sh` |
| 共通ヘルパー（source して再利用） | `examples/internvla_a15_libero_finetune/scripts/_train_common.sh` |
| provision（唯一の既存編集先） | `examples/internvla_a15_libero_finetune/scripts/provision_data.sh` |
| 取得スクリプトのパターン | `examples/internvla_a15_libero_finetune/scripts/fetch_weights.sh` |
| 親計画（様式・F 番号） | `docs/plans/internvla-a15-finetune-plan.md` |
