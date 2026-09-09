# InternVLA-A1.5 モデルアーキテクチャまとめ

作成日: 2026-09-08
出典: `InternVLA-A-series/src/lerobot/policies/internvla_a1_5/`（HEAD `e6fc904`）、`InternVLA.pdf`（arXiv:2607.04988）、および本セッションでのコード読解。

このドキュメントは PARC 2026 で InternVLA-A1.5-base をファインチューニングするにあたって、**どこに何があるか・どこを触るとどこに効くか**を把握するためのリファレンス。数式の再導出ではなく、実装ファイルとの対応を優先する。

---

## 1. 全体像

InternVLA-A1.5 は 1 つのモデルで 3 つのことを同時にやる。

1. **理解（Understanding）** — VLM（Qwen3.5-2B）で画像・言語を処理する
2. **予見（Foresight）** — 学習可能な foresight token が、共有の多モーダル文脈から未来の動画を予見するよう、WAN2.2-5B（凍結）に教師される
3. **行動（Action）** — 軽量な action expert が、VLM と共有の full-attention 層を通じて連続的な行動チャンクを flow matching で生成する

推論時は動画分岐を捨てて行動だけを出す（`inference_backend="optimized"`, `action_loss_only=True`）ため、**学習時と推論時でネットワークの実効的な形が変わる**のが最大の特徴。この非対称性が PARC 側の実装（`action_loss_only` の扱い、エクスポート時に何を落とすか）に直結する。

```
                    ┌─────────────────────────────────────────┐
  画像 (2枚)         │        Qwen3.5-2B VLM backbone           │
  言語トークン  ───▶ │  (Gated DeltaNet 層 + Full-Attention層)  │──┐
  (state をテキスト化)│                                          │  │ 共有 full-attention 層
                    └─────────────────────────────────────────┘  │ (KV を suffix と共有)
                                       │                          │
                          foresight tokens (50個, 学習可能)        │
                                       │                          ▼
                    ┌──────────────────▼──────┐      ┌──────────────────────────┐
                    │  WAN2.2-5B (凍結・教師)   │      │   Action Expert           │
                    │  未来フレームを予測させる  │      │  (VLM と同じ層構造・別重み) │
                    │  → video_loss           │      │  Flow matching で行動生成   │
                    └──────────────────────────┘      └──────────┬───────────────┘
                    ★ 推論時は完全に切り離す                        ▼
                    (action_loss_only=True)              action chunk (50, 32)
```

---

## 2. VLM backbone — Qwen3.5-2B + Gated DeltaNet

### 2.1 レイヤー構成

Qwen3.5 は **Gated DeltaNet 層（線形アテンション）と Full-Attention 層が交互に混在**するハイブリッド構成。どの層がどちらかは `config.layer_types`（`"linear_attention"` / `"full_attention"` のリスト）で決まる。

- 実装: `InternVLA-A-series/src/lerobot/policies/internvla_a1_5/transformers_replace/models/qwen3_5/modeling_qwen3_5.py`
- これは**上流 transformers を差し替える版**で、通常の HF `Qwen3_5` には無い改造が入っている（Gated DeltaNet の実装と、action expert とのフック）。この差し替えを当てずに `transformers` を pip install しただけでは動かない（`docs/internvla-py310-porting.md` §1 の壁 3）。

### 2.2 Gated DeltaNet 層の役割

- 線形時間・線形メモリの再帰的アテンションで、長い文脈（多数の画像トークン・長い履歴）を安く処理する。
- **重要**: Gated DeltaNet 層は「再帰状態を共有できない」ため、**VLM と action expert を完全に独立に処理する**（`modeling_internvla_a1_5.py:137-179`、`if layer_type == "linear_attention": ... models = [qwen3_5, action_expert]` を別々にループ）。
- 対して full-attention 層は、prefix（VLM 側）と suffix（action expert 側）の Q/K/V を結合して 1 つの attention として計算する（後述）。

### 2.3 Vision 入力

- Qwen の標準的な画像トークン化（`AutoProcessor` の smart_resize 相当）を使う。画像は `do_rescale=False` で CHW float[0,1] のまま渡す（`transform_internvla_a1_5.py:169-173`）。
- 学習時の解像度は 224×224（`InternVLAA15DatasetConfig.height/width`）。**この解像度と一致しない画像を渡すと `image_grid_thw`（視覚トークン数）が学習時と変わり、精度が静かに落ちる**（PARC 側で "F6" と呼んでいる問題。詳細は `docs/plans/internvla-a15-finetune-plan.md` の F6/F7/F15）。

---

## 3. Action Expert — 軽量な第二の Transformer

### 3.1 構成の作り方

`InternVLAA15WithExpertModel.__init__`（`modeling_internvla_a1_5.py:360-411`）で、**VLM のテキスト設定をコピーして action expert を作る**:

```python
action_expert_config_hf.layer_types              = vlm_text_config.layer_types            # 同じ層構成
action_expert_config_hf.num_hidden_layers         = vlm_text_config.num_hidden_layers
action_expert_config_hf.rope_parameters           = vlm_text_config.rope_parameters
action_expert_config_hf.linear_conv_kernel_dim    = vlm_text_config.linear_conv_kernel_dim  # Gated DeltaNet 関連
...
self.action_expert = Qwen3_5TextModel(config=action_expert_config_hf)
self.action_expert.embed_tokens = None   # トークン埋め込みは使わない（連続値の action/state を直接埋め込むため）
```

- **hidden_size / intermediate_size / head_dim は縮小可能**（PARC の config では `action_expert_hidden_size=1024`, `action_expert_intermediate_size=3072` — VLM 本体の 2048/6144 相当より小さい）。つまり action expert は VLM より**軽量な同型モデル**。
- **num_attention_heads / num_key_value_heads は VLM と同じ値を強制**（`:389-390`）。これは full-attention 層で prefix と suffix の Q/K/V を結合するために、ヘッド数を揃える必要があるため。
- 重みは VLM と**完全に別**（層構造だけ真似た別インスタンス）。

### 3.2 full-attention 層での結合（"shared full-attention layers"）

`modeling_internvla_a1_5.py:183-` の `elif layer_type == "full_attention":` ブロックで、VLM 側（prefix）と action expert 側（suffix）それぞれで Q/K/V/gate を別々に計算した後、**KV をトークン軸で連結して 1 つの attention として計算する**。これにより:

- action expert 側のクエリ（suffix）が VLM 側のキー/バリュー（prefix = 画像・言語の文脈）を直接参照できる（cross-modal attention に相当）。
- `knowledge_insulation=True` の場合、suffix → prefix への attention の勾配を `detach` し、action 側の勾配が VLM 側に逆伝播しないようにできる（PARC 設定では `knowledge_insulation=false` = 逆伝播を許可、公式 LIBERO レシピに合わせている）。
- linear_attention 層では上記の共有が起きない（再帰状態を持つため独立処理）ので、KI は linear 層では自動的に成立する。

### 3.3 推論時の KV キャッシュ

`sample_actions`（`modeling_internvla_a1_5.py:761-`）は、まず `embed_prefix` で画像+言語をエンコードして **prefix の KV キャッシュを 1 回だけ計算**し（`use_cache=True`）、その後の denoise ステップ（後述）では**KV キャッシュを使い回して suffix だけを毎回計算**する。VLM 本体の forward は 1 チャンクにつき 1 回で済む。

---

## 4. Foresight token と WAN2.2 による動画予見

### 4.1 学習可能トークン

- `self.learnable_tokens = nn.Parameter(torch.zeros(num_learnable_tokens=50, action_expert_hidden_size))`（`:568`）— PARC では 50 個、初期化は `trunc_normal_(std=0.02)`。
- これらは action expert の suffix シーケンスに混ぜられ、full-attention 層を通じて VLM の文脈（画像・言語）を参照する。
- `freeze_learnable_tokens` フラグでこれらを凍結できるが、**PARC の方針は学習対象のまま**（`freeze_learnable_tokens=false`）。

### 4.2 WAN2.2-5B への接続

- `action_loss_only=False` のときだけ `WanVideoModel.from_pretrained(...)` が構築される（`:576-584`）。WAN 本体（DiT）は `freeze_wan_dit=True` で凍結、VAE も凍結。
- `learnable_to_wan_proj: Linear(action_expert_hidden_size, wan_dim)` が foresight token の出力を WAN の潜在次元に射影する。
- 学習時、foresight token の出力から生成した動画潜在と、実際の未来フレーム（データセットの `observation.video_frames`）を WAN の flow matching スケジューラで比較し `video_loss` を計算する（`_compute_video_loss`, `:1309-`）。
- **教師データの時間軸**: `image_delta_indices = [0,12,25,37,50]`（`num_video_frames=4` → 5 フレーム、chunk_size=50 に対して均等割り）。20Hz データなら 2.5 秒スパンの未来を予見させる計算になる。

### 4.3 推論時は丸ごと切り離す

- `action_loss_only=True` にすると `WanVideoModel` も `learnable_to_wan_proj` も**構築されない**（`:576` の `if not config.action_loss_only:` 節ごと skip）。
- ただし `learnable_tokens` / `learnable_tokens_in_proj` は action expert の suffix に混ざる普通のパラメータなので、`action_loss_only=True` でも**存在し続け、attention に参加し続ける**。つまり「foresight を学習したことで action の質が上がる」という効果自体は推論時にも残るが、動画そのものを生成する経路は無い。
- `state_dict()` は `_checkpoint_excluded_prefixes = ("model.wan_video_model.",)` で WAN 本体の重みを保存時から除外済み（チェックポイントを軽くするため）。PARC のエクスポート手順では、これに加えて `learnable_to_wan_proj.*` と buffer `_wan_grid_sizes`（推論時に未構築のキー）を落とす。`learnable_tokens` 系は**絶対に落とさない**。

---

## 5. Flow Matching による行動生成

### 5.1 学習時の定式化

上流の実装（`modeling_internvla_a1_5.py:1125-1126` 付近）:

```python
x_t  = time * noise + (1 - time) * actions     # time: 1（純ノイズ）→ 0（正解行動）の補間
u_t  = noise - actions                          # 回帰目標（速度場）
```

モデルは `denoise_step` でこの `u_t` を予測するよう `loss_action`（MSE 相当）で学習される。

### 5.2 推論時のサンプリング（Euler 積分）

`sample_actions`（`:807-833`）:

```python
dt = -1.0 / num_inference_steps      # num_inference_steps=10 が既定
x_t = ノイズ（time=1 から開始）
while time >= -dt/2:
    v_t = denoise_step(state, x_t, time, ...)   # KV キャッシュ済み prefix を参照
    x_t = x_t + dt * v_t
    time += dt
return x_t   # time=0 相当、これが予測行動チャンク
```

- 1 チャンクの生成に **VLM の forward 1 回（prefix embed）+ denoise_step 10 回**（action expert 側のみ、KV キャッシュ再利用）がかかる。
- `denoise_step` は `chunk_size=50` 分の行動を一度に出す（PARC では `n_action_steps=50` = チャンク全体を使う）。

### 5.3 PARC で追加した RTC（Real-Time Chunking 風の平滑化）

上記の euler ループに guidance 項を差し込み、前チャンクとの連続性を上げる（`examples/internvla_a15_libero_finetune/inference/chunk_blending.py`）:

```
v = (1 - W) * v_model + W * (x_t - target) / t_safe     # W: 重複区間で減衰するソフト重み
```

- 符号は `x_t - target`（`target - x_t` ではない）。§5.1 の `x_t = time*noise + (1-time)*actions` から `x_t - actions = time*(noise-actions) = time*u_t` が導けるため、`v = (x_t - y)/t` が正しい（この点は実装過程で符号ミスを訂正済み）。
- グリッパ次元（index 6）は guidance から除外（二値量なので blend すると開閉が遅れる）。
- `w_max < 1.0`（既定 0.8）でハードマスクは採らない。
- 詳細と設計判断の根拠は `docs/plans/internvla-a15-finetune-plan.md` §5.3、実装は上記ファイル参照。**採点ハーネスは推論中に環境を進めない完全同期なので、RTC の目的は「実時間短縮」ではなく「jerk/SPARC などの軌道メトリクス改善」であることに注意**（同計画の異議 2 参照）。

---

## 6. 行動出力の構造

- `output_features["action"].shape = (max_action_dim=32,)`。実際に使うのは先頭 7 次元 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`（LIBERO の EE デルタ + 二値グリッパ）、残りはゼロパディング。
- `action_delta_indices = list(range(chunk_size))` — チャンク全体（50 ステップ分）を密に予測する。
- 正規化は `normalization_mapping` がすべて `IDENTITY`（ネットワーク内部では正規化しない）。実際の正規化/逆正規化は **データ変換層**（`NormalizeTransformFn`）が `dataset.meta.stats` を使って行う。モデル本体とは独立なので、**チェックポイントの正規化統計とデータセットの正規化統計は別物**（このためスイート別 stats を持つ `InternVLA-A1.5-Libero` の統計をそのまま使う必要はない、という判断につながった）。

---

## 7. FAST action tokens と VQA 損失

- `use_fast_action_tokens=True` の場合、行動を離散トークン化（FAST tokenizer）して**言語トークン列の一部として**埋め込み、VLM の言語モデリング損失（`loss_per_token`）でも学習する（`action_token_min/max` の範囲のトークン ID）。
- `enable_vqa_loss=True` は、この FAST action-token 損失（と、VQA データセットがあれば言語 QA 損失）を合算する設定。**現行の PARC 想定データ（`libero_combined_20hz`）には VQA データセットが与えられていないため、実質は「robot サンプル上の FAST action-token / 言語損失」を指す**（`docs/plans/internvla-a15-finetune-plan.md` の D5 参照。VQA 用の transform チェーン自体は存在するが dead path）。
- `block_action_attend_fast_tokens=True`（既定）で、denoise 時に suffix（連続行動）が FAST トークン位置を直接参照しないようマスクする（`_compute_fast_token_mask`）。

---

## 8. 学習損失の内訳

`InternVLAA15Policy.forward` が返す合計損失（`:1592-1665` 付近）:

```
total_loss = loss_action                              # flow matching の MSE
           + lambda_vqa * loss_vqa                      # FAST action-token / 言語損失
           + video_loss_weight * video_loss              # foresight → WAN 教師損失（action_loss_only=False のときのみ非ゼロ）
```

PARC の設定（公式 LIBERO レシピ踏襲）: `enable_vqa_loss=true`, `lambda_vqa=1.0`, `video_loss_only=false`, `video_loss_weight=1`, `action_loss_only=false`。

---

## 9. PARC 固有の変更点（このモデル自体には手を入れていない）

上流のモデル定義・重みは一切変更していない。PARC 側の変更はすべて**データ変換層と推論ランタイム**に閉じている（`examples/internvla_a15_libero_finetune/`）。

| 変更 | 層 | 目的 |
|---|---|---|
| `RenderDownsampleFn`（256→128 カーネルランダム化） | データ変換（学習のみ） | 採点環境の native 128 レンダとの解像度ギャップを埋め、視覚ロバスト性を上げる |
| `libero_combined` schema（単一 robot_type） | データ変換（学習のみ） | 推論時にスイートを判別できないため、単一の正規化統計に統一する |
| `resize_with_pad(224,224)` の明示呼び出し | 推論ランタイムのみ | 上流の推論バックエンドはこの resize を省略しており、学習（224）と推論（生解像度）の食い違いが起きる（F6） |
| RTC guidance / temporal ensembling | 推論の euler ループのみ（モデル本体は不変） | チャンク境界の不連続を平滑化し、jerk/SPARC/衝突ルールに対処 |

これらの設計判断の詳細と根拠は `docs/plans/internvla-a15-finetune-plan.md` を参照。

---

## 10. ファイル早見表

| 知りたいこと | 見るファイル |
|---|---|
| VLM/action expert のレイヤー結合ロジック全体 | `InternVLA-A-series/src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py`（`InternVLAA15WithExpertModel`, `:360-`） |
| flow matching の euler ループ本体 | 同ファイル `sample_actions`（`:761-833`）, `denoise_step`（`:834-880`） |
| WAN 動画分岐・foresight token | 同ファイル `:568-594`（構築条件）, `_compute_video_loss`（`:1309-`） |
| 全ハイパラの定義とデフォルト値 | `configuration_internvla_a1_5.py`（`InternVLAA15Config`） |
| データ変換チェーンの並び順 | 同ファイル `InternVLAA15DatasetConfig.__post_init__`（`:36-102`） |
| Qwen3.5 の Gated DeltaNet 差し替え実装 | `transformers_replace/models/qwen3_5/modeling_qwen3_5.py` |
| 公式 LIBERO ファインチューニングの設定値 | `InternVLA-A-series/launch/internvla_a15_finetune_libero.sh` |
| PARC 側の推論ランタイム（RTC 含む） | `PARC-final/examples/internvla_a15_libero_finetune/inference/` |
| PARC の設計判断の根拠一覧 | `PARC-final/docs/plans/internvla-a15-finetune-plan.md` |
