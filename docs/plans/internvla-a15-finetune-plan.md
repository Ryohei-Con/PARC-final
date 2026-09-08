# 実装計画: PARC 2026 — InternVLA-A1.5 ファインチューニング & 提出環境構築

作成日: 2026-09-08 / 対象リポジトリ: `PARC-final`
ステータス: **ユーザー承認待ち**

---

## 0. 調査で確定した事実と、前提の訂正

`[検証済]` が付いた行は、計画作成後に該当ファイルを直接読んで裏取りした。

| # | 事実 | 根拠 | 計画への影響 |
|---|---|---|---|
| F1 | **上流に third-party plugin 機構がある。** `lerobot_train.py` の `main()` が `register_third_party_plugins()` を呼び、`sys.path` 上の `lerobot_policy_*` を自動 import する `[検証済]` | `src/lerobot/utils/import_utils.py:133-155`, `scripts/lerobot_train.py:407-409` | 上流を一切編集せず overlay を注入できる。**patch ではなく overlay を採る根拠の中心**（§3） |
| F2 | ただし plugin の import 失敗は `except Exception: logging.exception(...)` で**握り潰される** `[検証済]` | 同 `:151-153` | plugin 名だけに頼らず、明示 import する wrapper entrypoint を主経路にする |
| F3 | `DataTransformFn` は `draccus.ChoiceRegistry` | `src/lerobot/transforms/core.py:29` | 新規 transform を `register_subclass` で追加でき、`train_config.json` にも往復シリアライズされる |
| F4 | schema は `register_schema()` / `load_schemas_from_path()` が公開されている `[検証済]` | `src/lerobot/dataset_schemas/registry.py:175-189` | 上流の `configs/*.yaml` を触らずに `libero_combined` を足せる |
| F5 | **`state_dict()` は既に `model.wan_video_model.*` を除外している** `[検証済]` | `modeling_internvla_a1_5.py:1373`（`_checkpoint_excluded_prefixes`）, `:1426-1437` | 「WAN で +10GB」という当初の前提は**誤り**。残る WAN 由来キーは `model.learnable_to_wan_proj.*` と buffer `model._wan_grid_sizes` のみ。エクスポートはこの 2 つを落とす（数十 MB） |
| F6 | **上流の推論バックエンドは画像をリサイズしていない。** `ResizeImagesWithPadFn` を hydrate せずに構築しており `mapping` が空 → `__call__` のループが 0 回。さらに仮に hydrate しても mapping のキー（`observation.images.image` / `observation.images.wrist_image`）とサンプルのキー（`observation.images.image0/1/2`）が食い違うので二重に no-op。推論経路の resize はこの 1 箇所だけで、`_prepare_single` → `state_normalizer` → `processor` の残りに resize は無い `[公式 HEAD e6fc904 の clean checkout で再検証済]` | `policy_backend_internvla_a1_5.py:128,154-164`, `canonical_preprocess.py:47`, `transforms/core.py:142-145`, `dataset_schemas/configs/libero.yaml` | **最優先のサイレント失敗。** 学習は 224、推論は生解像度 → Qwen の smart_resize が別の `image_grid_thw`（視覚トークン数）を出す。採点環境は 128 なので学習の 224 と大きくずれる。自前ランタイムでは 224 へ明示的に resize すること |
| F15 | **`resize_with_pad(x, 224, 224)` は正方形入力では「単なる bilinear 補間」と完全に等価。** `scale = min(224/H, 224/W)` が H=W で一致するため `new_h = new_w = 224` となり **padding は 1 px も入らない**。実体は `F.interpolate(size=(224,224), mode="bilinear", align_corners=False)` + `clamp(0,1)`。`antialias` は渡されないので既定 False だが、128→224 は**拡大**なので antialias は元々無関係 `[検証済]` | `src/lerobot/transforms/utils.py:26-80` | **学習側は既に `ResizeImagesWithPadFn(224,224)` を通しているので、追加の変更は不要**（LIBERO の画像は正方形）。やるべきことは「推論側で同じ関数を通す」ことだけ。ただし我々が足す 256→128 は**別物**で、カーネルを振るのが目的なので `resize_with_pad` を使ってはならない |
| F7 | 画像は `do_rescale=False` で CHW float[0,1] のまま Qwen processor に渡る | `transform_internvla_a1_5.py:169-173` | F6 の影響が確実に出る経路。parity harness で `image_grid_thw` を突き合わせる |
| F8 | `image_delta_indices = [0,12,25,37,50]`（chunk 50 / `num_video_frames`=4） | `configuration_internvla_a1_5.py:424-426` | 生データは `[T=5,C,H,W]`。20Hz で 2.5 秒スパン。カーネルをフレーム毎に引くと WAN 教師がちらつく |
| F9 | libero schema は `action_reorder` / `state_reorder` を持たない | `dataset_schemas/configs/libero.yaml` | グリッパは index 6 のまま。RTC の除外次元 = 6 で正しい |
| F10 | `/act` タイムアウトは **10 秒**（`RemotePolicyClient(timeout_sec=10.0)`）。`episode_timeout_sec=120` / `gpu_time_limit_sec=3600` は `pipeline/config.py` に**宣言のみで配布ハーネスでは未使用** | `pipeline/remote_policy.py:42`, `pipeline/config.py:65-66` | 本番で有効化される可能性を残し、「1 エピソード 300 step の総推論時間」を予算として扱う（§8.3） |
| F11 | 採点環境の観測は `LIBERO_EVAL_CAMERA` 既定 **128**、制御 20Hz、`max_steps_per_episode=300` | `pipeline/config.py:52-54`, `pipeline/environment.py:118` | 解像度整合（256→128→224）と 20Hz データの整合が正しい |
| F12 | 上流レシピの `warmup=2000` / `decay_steps=100000` は **100k step 前提** | `launch/internvla_a15_finetune_libero.sh:157-158` | 1 GPU で 20k〜30k step に落とすなら**再スケールが必須**。素直にコピーすると warmup が全体の 10% を食い、decay がほぼ効かない |
| F13 | 推論時 `action_loss_only=True` では `learnable_to_wan_proj` が構築されない | `modeling_internvla_a1_5.py:576-584` | チェックポイントに残っていても **unexpected key**（missing ではない）。`assert_checkpoint_covers_model()` は通る |
| F14 | 提出物の zip 上限 20GB / 展開 40GB。`requirements.txt` の禁止オプションは `-i/--index-url/-f/-e/-r/-c` とスキーム `git:/https:/file:` 等 | `validate_submission.py:16-33` | 余裕は大きい。B4/B5 の削減は「起動時間とデバッグ容易性」のためであってサイズ制約のためではない |

**この調査で否定/修正された前提**

- 「WAN 重みが state_dict に載って +10GB」→ **既に除外済み**（F5）。実測で確認し、実際に載っていなければエクスポートの該当ステップは「検証のみ」に格下げする。
- 追加で発見した最重要項目 → **F6（推論時にリサイズが走らない）**。これを踏むと画像が壊れないのに精度だけ落ちる。手順書 §6 のカタログに 1 行追加する。

---

## 1. 目的

- InternVLA-A1.5-base を `libero_combined_20hz` でフルファインチューニングし、PARC 2026 の 3 トラックで動く単一ポリシーを作る。
- 採点環境（Py3.10 / 通信遮断 / native 128 レンダ）と学習環境（Py3.11 conda / 256 データ）の**経路差をすべて明示的に潰す**。
- チャンク境界の急変を平滑化し、Total Score の jerk / SPARC / path length と 1mm 衝突ルール下の失敗を減らす。

---

## 2. フェーズ構成と成果物

手元は GPU 無し Windows。**「手元で書けて手元で検証できるもの」を P1 に全部前倒しする。** クラウドは「事実収集」と「学習」と「最終検証」にだけ使う。

| Phase | 名称 | 実行場所 | 主な成果物 |
|---|---|---|---|
| P1 | 手元実装（純ロジック + テスト + 調査スクリプト） | 手元 (Win, CPU) | overlay 一式、chunk blending、事実収集スクリプト、単体テスト |
| P2 | 学習環境構築 + 事実確定 | クラウド | `env_train.sh`、`facts.json`（データ規約・向き・解像度） |
| P3 | データ準備 & schema 確定 | クラウド | `libero_combined` schema、patch 済み `info.json`、parity レポート |
| P4 | probe → 本学習 | クラウド | `probe_report.md`、checkpoints |
| P5 | 推論ランタイム実装 | 手元で実装 → クラウドで実測 | `internvla_runtime.py`、`policy_server.py`、`runtime_config.json` |
| P6 | エクスポート | クラウド | `submission/` ツリー、`MANIFEST.json` |
| P7 | 3 段階検証 + 提出 | クラウド | `verify_inference.py` PASS、Docker E2E、`validate_submission.py` ERROR 0 |

ルートディレクトリ（以降 `IVLA_DIR` と表記）:

```
PARC-final/examples/internvla_a15_libero_finetune/
```

`examples/pi05_libero_finetune/` の構成（`README.md` / `scripts/_train_common.sh` / `patches/`）に合わせる。

---

## 3. 上流への介入方針 — **overlay を採る**（patch は採らない）

**結論: PARC-final 側に `lerobot_policy_parc` パッケージを置き、`sys.path` 経由で登録する。上流 `InternVLA-A-series` は 1 バイトも書き換えない。**

根拠:

1. **必要な拡張点がすべて公開 API で埋まっている。** transform は `DataTransformFn.register_subclass`（F3）、dataset config は `DatasetConfig.register_subclass`、robot schema は `load_schemas_from_path`（F4）。patch を当てないと届かない箇所が現時点で 1 つも無い。
2. **patch は上流の commit が動くたび壊れる。** `examples/pi05_libero_finetune/patches/` は lerobot v0.4.4 にピンして初めて成立している。InternVLA-A-series は版が固定されていないクローンなので、同じやり方は再現性を落とす。
3. **vendor/lerobot を上流とバイト一致に保てる。** 提出物には `src/lerobot` を丸ごと同梱する（手順書 B1）。ここが上流と一致していれば「どちらの lerobot が動いているか」の問題が構造的に消え、エクスポート時に `diff` で機械検証できる。patch を当てると学習用と提出用で 2 系統のツリーができ、片方だけ patch 忘れという事故が入る。
4. RTC guidance は**推論時のみ**の変更なので、そもそも上流に触る必要が無い（§5.3 で euler ループを自前再実装する）。

例外規定:

- vendor へ**追加**する唯一のファイルは `vendor/lerobot/dataset_schemas/configs/libero_combined.yaml`（新規追加であって既存ファイルの改変ではない）。`MANIFEST.json` に「上流に無い追加ファイル」として明記する。
- どうしても monkeypatch が要ることが後で判明した場合は、**`lerobot_policy_parc/_monkeypatch.py` の 1 ファイルに集約**し、適用時に `logging.warning` で必ず名前を出す。上流ファイルは編集しない。

主経路は **wrapper entrypoint（`scripts/train_entry.py`）で overlay を明示 import** する。plugin 自動 import（F1）は失敗が握り潰される（F2）ので、保険としてパッケージ名を `lerobot_policy_parc` に合わせるだけに留める。

---

## 4. 変更対象

### 4.1 既存ファイルの編集

| ファイル | 操作 | 内容 |
|---|---|---|
| `PARC-final/examples/README.md` | 追記 | 新レシピへのリンク 1 行 |
| `PARC-final/docs/internvla-py310-porting.md` | 追記 | §6 サイレント失敗カタログに F6（推論時 resize が無効）を 1 行追加 |
| `PARC-final/THIRD_PARTY_LICENSES.md` | 追記 | InternVLA-A1.5 / Qwen3.5 / WAN のライセンス表記 |

**`submission_template/` は絶対に編集しない。** 配布物であり、かつ「テンプレートとの差分が MyPolicy だけ」であることを回帰テストの根拠に使う。

### 4.2 上流リポジトリ

**編集なし。** 読み取り専用の参照元として扱う。`IVLA_DIR/lerobot_policy_parc/_provenance.py` が起動時に上流の主要ファイル（`configuration_internvla_a1_5.py` / `transforms/core.py` / `modeling_internvla_a1_5.py`）の SHA256 を記録し、想定と違えば警告する。

---

## 5. 新規ファイル一覧と責務

`IVLA_DIR = PARC-final/examples/internvla_a15_libero_finetune`

### 5.1 overlay（学習側の上流拡張）

| パス | 責務 |
|---|---|
| `IVLA_DIR/lerobot_policy_parc/__init__.py` | schema / transform / dataset config を副作用で登録。`assert_installed()` を公開 |
| `IVLA_DIR/lerobot_policy_parc/transforms_render.py` | `RenderDownsampleFn`（256→128 のカーネルランダム化） |
| `IVLA_DIR/lerobot_policy_parc/dataset_config.py` | `InternVLAA15ParcDatasetConfig` / `InternVLAA15ParcVQADatasetConfig` |
| `IVLA_DIR/lerobot_policy_parc/schema_bootstrap.py` | `install_parc_schemas()` |
| `IVLA_DIR/lerobot_policy_parc/schemas/libero_combined.yaml` | `libero_combined` robot_type 定義 |
| `IVLA_DIR/lerobot_policy_parc/_provenance.py` | 上流ファイルのハッシュ記録・照合 |

```python
# transforms_render.py
@DataTransformFn.register_subclass("render_downsample")
@dataclass
class RenderDownsampleFn(DataTransformFn):
    target_h: int = 128
    target_w: int = 128
    p_box: float = 0.45          # 2x2 box 平均 (= INTER_AREA = antialias=False bilinear)
    p_triangle: float = 0.30     # 4-tap 三角 (antialias=True bilinear = PIL BILINEAR)
    p_cubic: float = 0.15        # bicubic antialias=True, clamp(0,1)
    p_nearest: float = 0.10      # 位相ジッタ付き点サンプル（ユーザー指定で固定）
    nearest_phase_jitter: bool = True
    same_kernel_across_views: bool = True
    enabled: bool = True
    keys: list[str] = field(default_factory=list)  # hydrate で schema.image_mapping.keys()

    def hydrate(self, dataset) -> "RenderDownsampleFn": ...
    def __call__(self, data: DataDict) -> DataDict: ...
    # 内部
    def _worker_rng(self) -> random.Random          # worker ごとに 1 個生成しキャッシュ
    def _draw(self, rng) -> tuple[str, tuple[int, int]]   # (kernel_name, (oy, ox))
    def _apply(self, x, kernel, phase)              # x: [C,H,W] or [T,C,H,W]
```

実装上の決めごと（Generator への指示）:

- **カーネルはサンプル単位で 1 回だけ引く**。`__call__` の先頭で `_draw()` を 1 回呼び、`same_kernel_across_views=True` なら全カメラ・全 T フレームでそれを使い回す（F8 の理由）。
- `box` は `F.avg_pool2d(x, 2)` で実装する（2 倍ちょうどなので数学的に厳密。`cv2` 依存を持ち込まない）。倍率が 2 でない場合のみ `F.interpolate(mode="bilinear", antialias=False)` にフォールバックする。
- `nearest` は `x[..., oy::2, ox::2]` のスライス。`oy, ox` は独立に `{0,1}`。
- `cubic` は overshoot するので `.clamp_(0, 1)` を必ず入れる。
- `H <= target_h` のときは**何もせずに返す**（データが最初から 128 だった場合に壊れない）。
- **256→128 に `resize_with_pad` を使ってはならない。** `resize_with_pad` は `mode="bilinear"` / `antialias` 指定なしで固定されており（F15）、2 倍縮小では box 平均に潰れてカーネルのバリエーションが消える。この transform は自前で `avg_pool2d` / `interpolate` / スライスを使い分ける。128→224 の側は既存の `ResizeImagesWithPadFn`（= `resize_with_pad`）にそのまま任せる。
- 確率の配分根拠: 採点側は native 128 のポイントサンプルなのでエイリアシング側に寄っている。しかし nearest 単独は破壊的すぎるので**ユーザー指定どおり 0.10 に固定**。残り 0.90 は「最も素直で、256 データを作った側のパイプラインである可能性が最も高い」box を厚めに 0.45、軽いボケ側の triangle 0.30、リンギング/シャープ側（box とは逆向きのアーティファクト）の cubic 0.15。**この配分は 1 本の config で差し替えられるようにし、A/B の対象とする。**
- RNG は `random.Random(torch.initial_seed() ^ worker_id)` 相当を **worker ごとに 1 回だけ**生成してインスタンスにキャッシュする。全 worker が同じ列を引く事故を防ぐ。

```python
# dataset_config.py
@DatasetConfig.register_subclass("internvla_a1_5_parc")
@dataclass
class InternVLAA15ParcDatasetConfig(InternVLAA15DatasetConfig):
    render_downsample: bool = True
    render_target: int = 128
    render_p_box: float = 0.45
    render_p_triangle: float = 0.30
    render_p_cubic: float = 0.15
    render_p_nearest: float = 0.10

    def __post_init__(self):
        super().__post_init__()
        # 1. 既存の RenderDownsampleFn を全部除去（冪等性）
        # 2. 最初の ResizeImagesWithPadFn の直前に 1 個だけ挿入
        # 3. ResizeImagesWithPadFn がちょうど 1 個であることを assert


@VQADatasetConfig.register_subclass("internvla_a1_5_parc")
@dataclass
class InternVLAA15ParcVQADatasetConfig(InternVLAA15VQADatasetConfig):
    render_downsample: bool = True   # ResizeVQAImagesWithPadFn の直前に挿入
```

> **注記**: 現行の libero レシピは VQA データセットを与えていない（`launch/internvla_a15_finetune_libero.sh` に `--vqa_dataset*` が無い。`--policy.enable_vqa_loss=true` は robot サンプル上の FAST action-token / 言語損失を指す）。したがって VQA チェーンは**現状 dead path** である。実装はするが、動作確認は合成データでの単体テストまでとし、「本走で効いていない」ことを Evaluator に明示する。

### 5.2 スクリプト（学習運用）

| パス | 責務 |
|---|---|
| `IVLA_DIR/README.md` | レシピ本体のドキュメント |
| `IVLA_DIR/env.example.sh` | `IVLA_REPO` / `IVLA_CONDA_ENV` / `IVLA_DATASET_ROOT` / `HF_HOME` 等 |
| `IVLA_DIR/scripts/setup_train.sh` | conda 3.11 env 構築（手順書 §3）。torch / torchcodec の **ABI ピン**、transformers 差し替え、conda 探索 |
| `IVLA_DIR/scripts/_train_common.sh` | pi05 版の関数群を流用（`_resolve_paths` / `_activate_conda` / `_maybe_resume` / `_start_vram_sampler` / `_summarize_run`） |
| `IVLA_DIR/scripts/train_entry.py` | **overlay を明示 import してから `lerobot_train.main()` を呼ぶ**。登録確認に失敗したら即 `SystemExit` |
| `IVLA_DIR/scripts/train_ivla_a15.sh` | 本学習ランチャー（上流 launch script の ARGS を基に、単 GPU / overlay / 単一 robot_type 用に組み替え） |
| `IVLA_DIR/scripts/probe_ivla_bs.sh` | BS ラダー probe（§8） |
| `IVLA_DIR/scripts/smoke_ivla.sh` | 20 step 学習 → 保存 → エクスポート → `verify_inference.py` まで一気通貫 |
| `IVLA_DIR/scripts/prepare_dataset.py` | `info.json` の `robot_type` を `libero_combined` へ冪等に書き換え。書き換え前後を stdout に出す |
| `IVLA_DIR/scripts/inspect_dataset.py` | **事実収集**（P2）。`facts.json` と PNG を吐く |

```python
# train_entry.py
def main() -> None:
    _ensure_overlay_on_syspath()                 # IVLA_DIR を sys.path[0] へ
    import lerobot_policy_parc as overlay        # ← 失敗したら例外がそのまま上がる（F2 対策）
    overlay.assert_installed()                   # schema と choice registry を実照会
    logging.info("overlay=%s upstream=%s", overlay.__file__, lerobot.__file__)
    from lerobot.scripts.lerobot_train import main as lerobot_main
    lerobot_main()
```

### 5.3 推論・提出

| パス | 責務 |
|---|---|
| `IVLA_DIR/inference/policy_server.py` | `submission_template/policy_server.py` のコピー。`MyPolicy` と先頭の `sys.path` 注入だけを差し替える |
| `IVLA_DIR/inference/internvla_runtime.py` | 3.10 ブートストラップ + 推論本体 |
| `IVLA_DIR/inference/chunk_blending.py` | **RTC guidance と temporal ensembling の純ロジック**（手元でテスト可能） |
| `IVLA_DIR/inference/verify_inference.py` | 提出前自己チェック |
| `IVLA_DIR/inference/runtime_config.json` | 設定の焼き込み（手順書 B8） |
| `IVLA_DIR/inference/requirements.txt` | PyPI ピンのみ |

```python
# internvla_runtime.py
def bootstrap(runtime_dir) -> dict                # vendor lerobot 注入 + transformers パッチ
def load_runtime_config(path=None) -> dict
def build_submission_runtime(config_path=None)    # policy_server と verify の共通入口

def quat2axisangle(quat)                          # model2libero_interface.py:11-32 の移植
def orient_image(arr, mode)                       # "none"|"flip_ud"|"flip_lr"|"rot180"
def assert_checkpoint_covers_model(policy, checkpoint_file) -> int

class InternVLARuntime:
    def __init__(self, ckpt_dir, vlm_dir, cfg): ...
    def reset(self, instruction=""): ...
    def get_action(self, obs) -> np.ndarray       # (7,) float32
    # 内部
    def _obs_to_sample(self, obs)
        # 向き補正 → HWC uint8 → CHW float[0,1] → resize_with_pad(224,224)  ← F6 対策で必ず通す
        # → mask 付与 → NormalizeTransformFn(state) → InternVLAA15ChatProcessorTransformFn(mode="eval")
    def _predict_chunk_normalized(self, sample, guidance)   # [chunk, D] 正規化空間
    def _denormalize(self, chunk_norm)
    def _to_env_action(self, a7)                            # clip + gripper 二値化
```

```python
# chunk_blending.py   ← 「手元で完全に単体テストできる」中核
@dataclass
class GuidanceSpec:
    weights: np.ndarray    # [chunk]  0..1、重複区間で減衰するソフト重み
    target:  np.ndarray    # [chunk, D]  前チャンクを時間軸で揃えたもの（正規化空間）
    dim_mask: np.ndarray   # [D] bool  index 6（gripper）と 7..D-1（padding）は False

def build_guidance(prev_chunk_norm, replan_steps, chunk_size, action_dim,
                   w_max=0.8, schedule="linear", exclude_dims=(6,)) -> GuidanceSpec | None
    # target[j] = prev_chunk_norm[replan_steps + j]  (j < chunk_size - replan_steps)
    # weights[j] = w_max * (1 - j / L),  L = chunk_size - replan_steps,  j >= L で 0
    # w_max < 1.0 に保つ = ハードマスク無しのソフトのみ

def euler_integrate(denoise_fn, x0, num_steps, guidance=None)
    # 上流 modeling_internvla_a1_5.py:807-833 の euler ループを再実装
    #   dt = -1/num_steps ;  while time >= -dt/2:
    #       v_model = denoise_fn(x_t, time)
    #       t_safe  = clamp(time, min=|dt|)                 # 最終ステップの 0 除算を防ぐ
    #       v = where(dim_mask & (W>0),  (1-W)*v_model + W*(x_t - y)/t_safe,  v_model)
    #       x_t = x_t + dt * v ; time += dt
    # denoise_fn を差し替え可能にしてあるので、スタブで CPU 単体テストできる
    #
    # ★符号に注意（初版の計画は逆で、適用すると x が y から遠ざかっていた）。
    #   上流の flow matching 規約は modeling_internvla_a1_5.py:1125-1126:
    #       x_t = time * noise + (1 - time) * actions
    #       u_t = noise - actions          # ← v の回帰目標
    #   ここから x_t - actions = time * (noise - actions) = time * u_t、
    #   すなわち v = (x_t - actions) / time。v は「clean から noise へ向かう」向きで、
    #   サンプリングは dt = -1/N で x += dt*v と積分する。よって y へ収束させる
    #   guidance 速度は (x_t - y)/t が正しい。w→1 で最終ステップに x == y と
    #   なることをテストで担保する。

@torch.no_grad()
def sample_actions_guided(model, *, pixel_values, image_grid_thw, lang_tokens, lang_masks,
                          state, fast_token_mask=None, num_steps=None, noise=None, guidance=None)
    # embed_prefix → KV cache 構築（上流 :787-817 と同一手順）
    # → euler_integrate(denoise_fn=lambda x,t: model.denoise_step(...), ...)

class TemporalEnsembler:
    """比較対象。重複チャンクの指数加重平均。"""
    def __init__(self, chunk_size, action_dim, m=0.1, newest_dims=(6,)): ...
    def reset(self): ...
    def push(self, chunk_norm): ...
    def pop(self) -> np.ndarray            # [D] 正規化空間
    # 重み w_i = exp(-m * i) （i = チャンクの古さ）
    # newest_dims は平均せず最新チャンクの値を採る（グリッパの遷移遅れを防ぐ）
```

`runtime_config.json` の骨子:

```json
{
  "runtime": "standard",
  "env": { "IVLA_ROBOT_TYPE": "libero_combined" },
  "image_orientation": { "agentview": "TBD", "wrist": "TBD" },
  "state": { "source": "eef_pos+axisangle+gripper_qpos", "dim": 8 },
  "gripper": { "dataset_convention": "TBD", "env_close": 1.0, "env_open": -1.0 },
  "chunking": { "mode": "rtc", "replan_steps": 16, "w_max": 0.8,
                "schedule": "linear", "exclude_dims": [6], "ensemble_m": 0.1 },
  "resize": { "height": 224, "width": 224 },
  "num_inference_steps": 10
}
```

読み込みは **`os.environ.setdefault`**（手順書 B8）。手元で env を立てれば A/B のたびに zip を作り直さずに済む。

### 5.4 ツール・テスト

| パス | 責務 |
|---|---|
| `IVLA_DIR/tools/dump_env_frames.py` | 採点 env を dry-run して `agentview_image` / `robot0_eye_in_hand_image` の**生**フレームを PNG 保存 |
| `IVLA_DIR/tools/dump_dataset_frames.py` | データセットから同カメラのフレームを PNG 保存 |
| `IVLA_DIR/tools/orientation_match.py` | 4 変換 {none, flip_ud, flip_lr, rot180} の**行/列平均輝度プロファイル相関**を算出し argmax を出す |
| `IVLA_DIR/tools/parity_check.py` | 学習 dataloader と推論前処理の `pixel_values` / `image_grid_thw` / `input_ids` を突き合わせる |
| `IVLA_DIR/export/export_submission.py` | 提出パッケージ生成 |
| `IVLA_DIR/tests/test_render_downsample.py` | カーネル同値性・単一抽選・[T,C,H,W] 対応 |
| `IVLA_DIR/tests/test_chunk_blending.py` | RTC / ensembler の数値性質 |
| `IVLA_DIR/tests/test_dataset_config.py` | transform 挿入位置と冪等性、draccus 往復 |
| `IVLA_DIR/tests/test_schema.py` | `get_schema("libero_combined")` の内容 |
| `IVLA_DIR/tests/test_state_utils.py` | `quat2axisangle` vs `scipy.Rotation.as_rotvec` |
| `IVLA_DIR/tests/test_policy_server_parity.py` | `submission_template/policy_server.py` との差分が `MyPolicy` + import 行だけであることを検査 |

```python
# export_submission.py
def export(ckpt_dir, vlm_src, upstream_src, out_dir, *, drop_wan_keys=True, dedupe_tied=True) -> dict
def copy_vendor_lerobot(upstream_src, dst) -> dict
    # src/lerobot を丸ごとコピー → 各ファイルの SHA256 を MANIFEST へ
    # libero_combined.yaml を dataset_schemas/configs/ へ「追加」し、追加ファイルとして記録
def copy_vlm_config_only(src, dst) -> list[str]
    # *.json / *.txt / *.model / *.jinja のみ。model.safetensors.index.json は必ず除外
def list_droppable_keys(safetensors_path) -> list[str]
    # model.wan_video_model.* / model.learnable_to_wan_proj.* / model._wan_grid_sizes
    # ※ learnable_tokens / learnable_tokens_in_proj は絶対に落とさない
def find_tied_duplicates(safetensors_path) -> list[str]
def copy_safetensors_without(src, dst, drop_keys)   # ストリーミング
def write_runtime_config(dst, cfg)
```

---

## 6. 実装手順

### P1. 手元実装（GPU 不要・Windows で完結）

1. **`IVLA_DIR` の骨組みを作る。** `README.md` / `env.example.sh` / ディレクトリのみ。`examples/README.md` に 1 行追記。
2. **`schemas/libero_combined.yaml` を書く。** 上流 `configs/libero.yaml` の 1 エントリを `robot_type: libero_combined` に変えたもの。`feature_mapping` / `image_mapping` / `action_mask_spec: [6,-1]` / `action_mode: end_effector` は**そのまま**。`action_reorder` / `state_reorder` は書かない（F9）。
3. **`schema_bootstrap.py` + `__init__.py`。** `install_parc_schemas()` は `load_schemas_from_path()` を呼び、登録された robot_type 名のリストを返す。`assert_installed()` は `get_schema("libero_combined")` と choice registry の実照会で検証する。
4. **`transforms_render.py` を実装。** §5.1 の仕様どおり。
5. **`tests/test_render_downsample.py`。** 手元 CPU で以下を証明する:
   - `avg_pool2d(x,2)` ≡ `F.interpolate(bilinear, antialias=False)` ≡（cv2 があれば）`INTER_AREA`（`atol=1e-6`）。**「振っても無意味」の前提そのものを回帰テストにする。**
   - `nearest` / `triangle` / `cubic` は box と**有意に異なる**（`MSE > 1e-4`）。
   - `[T=5,C,256,256]` を渡すと 5 フレームすべてが同一カーネルで処理される（`p_nearest=1.0` に固定し、位相が T 全体で一定であることを確認）。
   - 2 カメラが同一カーネルになる。
   - 入力が 128 のときは恒等。`enabled=False` は恒等。
   - 1000 サンプル引いてカーネル頻度が設定確率に一致（相対誤差 5%）。**特に nearest が 0.10 前後**。
6. **`dataset_config.py` を実装 + `tests/test_dataset_config.py`。** 挿入位置（`ResizeImagesWithPadFn` の直前）、2 回 `__post_init__` しても 1 個しか入らないこと、`action_mode="abs"` で `DeltaActionTransformFn` が消えること、draccus で dict 化 → 復元して等価になること。
   - **上流 import が要る**ので、手元に `InternVLA-A-series/src` を `sys.path` に足した状態で pytest する。`torch` / `draccus` / `torchvision` は CPU 版で入る。`torchcodec` / `flash-attn` は不要（必要なら `conftest.py` でスタブ）。
7. **`chunk_blending.py` を実装 + `tests/test_chunk_blending.py`。**
   - `guidance=None` と `weights=0` が**ビット一致**する（ベースライン非破壊）。
   - `weights[j]=1.0` に強制すると `x_final[j] ≈ target[j]`（`atol=1e-3`）。ソフト重み `0.8` では近づくが一致はしない。
   - `dim_mask[6]=False` の次元が `guidance=None` の結果と**ビット一致**する（グリッパ非干渉）。
   - `num_steps=10` で NaN/Inf が出ない（`t_safe` のクランプが効いている）。
   - `build_guidance`: `replan_steps=chunk_size` で `None`、`replan_steps < chunk_size` で `L = chunk - replan`、`weights` が単調非増加で `[0,1]` に収まる。
   - `TemporalEnsembler`: 単一チャンクなら素通し、同一チャンクを繰り返し push しても値が変わらない、`newest_dims` が平均されない、`reset()` で状態が消える。
8. **`internvla_runtime.py` の「純粋部分」を先に書く。** `quat2axisangle` / `orient_image` / `_to_env_action` / `load_runtime_config` / `assert_checkpoint_covers_model`。`tests/test_state_utils.py` で `scipy.spatial.transform.Rotation.as_rotvec` と突き合わせる（`w<0` の符号反転や `w≈±1` の縮退を含むケースを入れる）。
9. **`policy_server.py` を作り、`tests/test_policy_server_parity.py` を書く。** テンプレートを AST で読み、`MyPolicy` クラスと先頭の import/`sys.path` ブロック以外が**トークン列として一致**することを検査する。
10. **`inspect_dataset.py` / `dump_env_frames.py` / `dump_dataset_frames.py` / `orientation_match.py` / `parity_check.py` を書く。** 実行はクラウドだが、コードは手元で書き切る。合成データでの smoke（`orientation_match.py` に既知の変換を掛けた画像を食わせて正解が返ること）は手元で通す。
11. **`export_submission.py` を書く。** 実行はクラウド。`copy_safetensors_without` は小さな合成 safetensors で手元テストできる。
12. **`requirements.txt` の初版。** 手順書 B3 の一覧をベースに、**torch は `2.11.0`**（採点環境が torch 2.11.0+cu130）。手順書の `2.10.0` は別環境向けの値なのでそのまま使わない。`transformers` の版は P7-C2 で確定する（差し替え `modeling_qwen3_5.py` が内部 API を叩くので**完全固定**）。

### P2. 学習環境構築 + 事実確定（クラウド）

13. `scripts/setup_train.sh` を実行。conda 3.11 / ffmpeg+svt-av1 / torch + torchvision（クラウド GPU の CUDA に合わせる）/ `pip install -e ${IVLA_REPO}` / **torchcodec を torch に合わせて明示ピン**（手順書 §3.1）/ transformers 差し替えコピー / オプショナル依存は `set +e`。
14. **torchcodec の ABI を実証する。** `import` が通るだけでは不足。データセットの動画を 1 フレームデコードして PNG 保存し、非ゼロであることを assert するチェックを `setup_train.sh` の最後に入れる。
15. `bash ../../scripts/extract_dataset.sh lerobot/libero_combined_20hz.tar` でデータ展開。
16. **`inspect_dataset.py` を実行して `facts.json` を作る。** 出力必須項目: `fps` / `robot_type`（現状値）/ 画像 feature の shape / `observation.state` の次元と統計 / `action` の次元と dim6 のヒストグラム / `stats.json` のキー数 / エピソード数・総フレーム数。
17. **`dump_dataset_frames.py` + `dump_env_frames.py` + `orientation_match.py` を実行して向きを確定する。**
    - dataset 側: `agentview` / `wrist` を各 8 枚 PNG 保存。
    - env 側: `pipeline` を dry-run し、`_capture_frame` が掛ける `[::-1]` の**手前**で生 obs を PNG 保存。
    - `orientation_match.py` が 4 変換それぞれについて相関を出す。argmax を候補とする。
    - **必ず目視でも確認する。** テーブル面・ロボットアーム・グリッパの位置関係が一致するか。カメラごとに独立に決める。
    - 結果を `runtime_config.json` の `image_orientation` に焼く。**コードに直書きしない。**
18. `facts.json` と PNG 一式を `IVLA_DIR/artifacts/facts/` に置き、README から参照する。

### P3. データ準備 & schema 確定（クラウド）

19. `scripts/prepare_dataset.py` で `~/data/libero_combined_20hz/meta/info.json` の `robot_type` を `libero_combined` に書き換える（冪等・書き換え前後をログ）。**上流 launch script のスイート別 robot_type パッチを意図的に反転させる**方針をコメントで明記する。
20. `train_entry.py` の起動確認で overlay 登録が通ることを確認。`assert_installed()` が緑。
21. **`parity_check.py`（学習側だけ）を実行。** `make_dataset` で 1 サンプル作り、`observation.pixel_values.shape` / `image_grid_thw` / `input_ids` の長さ / `observation.video_frames.shape`（`[5,3,224,224]` を期待）/ `action.shape`（`[50,32]`）を記録する。
22. **`RenderDownsampleFn` が実際に効いていることを実データで目視確認する。** `p_nearest=1.0` に固定して 1 サンプル取り、224 リサイズ後の画像を PNG 保存 → 明らかにエイリアスした画像になっていること。既定確率に戻して再度保存。2 枚を `artifacts/` に残す。

### P4. probe → 本学習（クラウド）

23. `probe_ivla_bs.sh` を実行（§8）。`probe_report.md` に peak VRAM / sec/step / OOM 有無を表で残す。
24. probe 結果から batch / grad accum / steps / warmup / decay を確定し、`train_ivla_a15.sh` の既定値を更新する。**`warmup` と `decay_steps` は必ず実 step 数に合わせて再スケールする（F12）。**
25. `smoke_ivla.sh`（20 step → 保存 → エクスポート → `verify_inference.py`）を通す。**ここで一気通貫が通るまで本走を始めない。**
26. 本学習。`_start_vram_sampler` / `_maybe_resume` を有効にする。`save_freq` は 5〜6 個のチェックポイントが残る値。W&B は offline。

### P5. 推論ランタイム（実装は手元、実測はクラウド）

27. `internvla_runtime.py` を完成させる。**F6 対策として、`_obs_to_sample` は `resize_with_pad(224,224)` を必ず通す**（上流バックエンドをそのまま真似しない）。前処理順序は学習の transform 順序と 1:1 対応させる:

    | 学習 | 推論 |
    |---|---|
    | (dataset) `[T,C,256,256]` | env obs `(128,128,3) uint8` |
    | `RenderDownsampleFn` → 128 | （既に 128。恒等） |
    | `ResizeImagesWithPadFn(224)` | `resize_with_pad(224,224)` ← **必須**。学習側と**同一関数**（`lerobot.transforms.utils.resize_with_pad`）を呼ぶこと。正方形入力では padding が入らず素の bilinear と等価なので、学習側は現状のままで変更不要（F15） |
    | `RemapImageKeyTransformFn` | `image0/image1` に直接詰める + `*_mask` |
    | `NormalizeTransformFn` | `NormalizeTransformFn(selected_keys=[OBS_STATE])` |
    | `InternVLAA15ChatProcessorTransformFn(mode="train")` | 同 `(mode="eval")`、`use_fast_action_tokens=False`、`action_mode` は schema から |

28. `_predict_chunk_normalized` に `sample_actions_guided` を接続。**正規化空間の前チャンクをキャッシュ**し、`build_guidance` に渡す。
29. `_to_env_action`: 上流と同じく `np.clip(action, low, high)` を掛けたうえで、`facts.json` で確定した規約に従い `action[6]` を二値化する。既定は `action[6] = 1.0 if a < 0.5 else -1.0`（データが `[0,1]` 規約の場合）。規約が `[-1,1]` だった場合の分岐も `runtime_config.json` で切り替えられるようにする。
30. `verify_inference.py` を完成させる。手順書 C2 の項目に加えて:
    - vendor lerobot の `__file__` と、上流 `src/lerobot` との**ハッシュ一致件数**を印字
    - モデルロード時間（120 秒制限）／推論レイテンシ（10 秒制限）
    - `action` が `(7,) float32`・NaN/Inf なし・`action[6] ∈ {-1,+1}`
    - `reset()` でチャンクキャッシュと ensembler が消えること
    - **`--chunking rtc|ensemble|none` を切り替えて、replan 境界の `||a_t − a_{t−1}||` を印字**（平滑化が効いているかの直接指標）
31. `parity_check.py` の推論側を接続し、**同一の生フレームから学習経路と推論経路で `image_grid_thw` が一致すること**を assert する。ここが F6 の回帰テスト。

### P6. エクスポート（クラウド）

32. `export_submission.py` を実行。順に:
    1. `list_droppable_keys()` で `wan_video_model.*` が**実際にチェックポイントに存在するか**を確認し、結果を MANIFEST に記録（F5 の実測）。
    2. `learnable_to_wan_proj.*` と `_wan_grid_sizes` を落とす。**`learnable_tokens` / `learnable_tokens_in_proj` は残す。**
    3. tie 重複テンソルを SHA256 比較で検出し、あればストリーミングで除去。無ければスキップしてログに残す。
    4. VLM は config/tokenizer/processor だけコピー（`model.safetensors.index.json` を削除）。`snapshot_download` のキャッシュに重みが入っている場合があるので、**コピー時にも同じ patterns で絞る**。
    5. `src/lerobot` を vendor へコピー + `libero_combined.yaml` を追加。
    6. `runtime_config.json` を書く。
    7. `MANIFEST.json`（元 checkpoint パス / step / 上流 commit / 落としたキー / 追加ファイル / 各ファイルの SHA256）。

### P7. 検証・提出

33. **C1**: `python3.10 -m compileall -q submission/vendor/lerobot` + 3.11 専用構文の grep（`tomllib|StrEnum|itertools\.batched|asyncio\.TaskGroup|datetime\.UTC|except\*`）。
34. **C2**: クリーン `python3.10 -m venv` に `submission/requirements.txt` だけを入れて `verify_inference.py --benchmark`。落ちた import を requirements に足すループ。**学習 conda env では絶対に検証しない。**
35. **C3**: 採点と同じ Docker で二重 venv（ハーネス CPU torch / 推論 CUDA torch）を再現。`/tmp` の掃除と `LIBERO_CONFIG_PATH` の手当を先に入れる。
36. **C4**: `python validate_submission.py submission` が ERROR 0、`python -m pipeline --server-url ... --track track1 --n-episodes 2` が完走。
37. **A/B**: `runtime_config.json` の `chunking.mode` を `none` / `rtc` / `ensemble` で振り、`replan_steps` を `{10, 16, 25, 50}` で振って、成功率と jerk / SPARC / path length を比較。**最終値はここで決める。**

---

## 7. 検証方法

### 7.1 手元（GPU 無し Windows）で確認できること

- [ ] `pytest examples/internvla_a15_libero_finetune/tests/` が全部緑
- [ ] `box ≡ interpolate(antialias=False) ≡ INTER_AREA` の同値性（`atol=1e-6`）
- [ ] `nearest/triangle/cubic` が box と有意差あり
- [ ] カーネルがサンプル単位で 1 回だけ引かれる（T=5 / 2 カメラで一定）
- [ ] `p_nearest` の実測頻度が 0.10 ± 5%
- [ ] `RenderDownsampleFn` が `ResizeImagesWithPadFn` の直前にちょうど 1 個入る / 冪等 / draccus 往復
- [ ] `get_schema("libero_combined")` が期待の `image_mapping` / `action_mode` を返す
- [ ] RTC guidance の 4 性質（非破壊 / 収束 / グリッパ非干渉 / NaN なし）
- [ ] TemporalEnsembler の 4 性質
- [ ] `quat2axisangle` が scipy と一致
- [ ] `policy_server.py` がテンプレートと `MyPolicy` 以外一致
- [ ] `copy_safetensors_without` が合成 safetensors で正しく動く
- [ ] `orientation_match.py` が既知の変換を当てられる
- [ ] `python3.10 -m compileall` と 3.11 構文 grep（Windows に py3.10 を入れれば可）
- [ ] `validate_submission.py` をダミー提出ツリーに掛けて ERROR 0
- [ ] msgpack で obs/action が往復する
- [ ] `parity_check.py` の**縮小版**: Qwen の processor ファイル（22MB、重み不要）だけ落とし、合成 256×256 画像を学習チェーン相当と推論チェーン相当に流して `image_grid_thw` 一致を確認 ← **F6 の回帰テストを手元に持ってこられる。優先度高**

### 7.2 クラウドでしか確認できないこと

- [ ] torchcodec がデコードできる（PNG 目視 + 非ゼロ assert）
- [ ] `facts.json`（state 次元 / グリッパ規約 / 画像解像度 / fps / stats キー数）
- [ ] 画像の向き（dataset PNG vs env PNG の照合）
- [ ] `make_dataset` が通り、1 サンプルの shape が想定どおり
- [ ] `RenderDownsampleFn` を通した実画像の目視
- [ ] probe（peak VRAM / sec/step / OOM）／本学習
- [ ] エクスポート後のサイズ・キー欠損
- [ ] モデルロード 120 秒 / 推論レイテンシ 10 秒
- [ ] クリーン venv での `verify_inference.py`
- [ ] Docker 二重 venv の E2E
- [ ] RTC / ensemble / replan の A/B（成功率と軌道メトリクス）

---

## 8. 学習時間の見積もりと step 数の落とし所

### 8.1 probe の回し方

`probe_ivla_bs.sh`: `BS ∈ {1, 2, 4, 8} × 30 steps × {gradient_checkpointing=false, true}`

- `_start_vram_sampler`（5 秒間隔の `nvidia-smi`）で peak VRAM を取る。
- **sec/step は step 10〜30 の中央値**（最初の数 step は cudnn ベンチマークとコンパイルで遅い）。
- OOM は失敗として記録し、次の設定へ進む（`set +e` で囲む）。
- 出力は `probe_report.md` の表（BS / GC / peak VRAM MiB / sec/step / OOM）。

**メモリ削減ラダー**（OOM 時に上から順に適用）:

1. `batch_size` を下げ、`grad accum` で実効バッチを維持する
2. `--policy.gradient_checkpointing=true`（スループット −25〜35% 想定）
3. `--policy.action_loss_only=true` に落として **WAN 分岐ごと切る**。`modeling_internvla_a1_5.py:576` により WAN DiT + VAE が構築されなくなり、メモリ・時間とも大幅に減る。**ただし動画教師と foresight token の学習を失う**ので最後の手段。判断は probe の数字を見て明示的に行い、`probe_report.md` に理由を書く
4. `--num_workers` を下げる（VRAM ではなく host RAM 対策）

96GB でも `action_loss_only=false` は WAN の DiT + VAE をメモリに載せるので、**BS=1 でも載らない可能性は現実にある**。probe の第 1 の目的はここの可否判定である。

### 8.2 step 数の決め方

```
sec_per_step  = probe の実測（grad accum 込みの 1 optimizer step あたり）
wall_budget_h = 学習に割ける実時間（先に決める。例: 24h）
steps         = floor(0.8 * wall_budget_h * 3600 / sec_per_step)   # 0.8 は保存・中断・評価のマージン
```

**確定値**（steps はユーザー指定で 30,000。バッチ系のみ probe 後に確定）:

| 項目 | 値 | 根拠 |
|---|---|---|
| 実効バッチ | 32〜64（`batch_size × grad_accum`） | 上流は 16 × N GPU。単 GPU で総サンプル数は追えないので、バッチを厚めにして step あたりの情報量を稼ぐ。**probe 後に確定** |
| steps | **30,000（確定）** | ユーザー指定。実効 32 で 960k サンプル。`facts.json` の総フレーム数で epoch 数を検算する |
| `scheduler_warmup_steps` | **600（確定）** | 上流の比率（2000/100000 = 2%）を 30k step に再スケールした値。**上流の 2000 をそのまま使わない**（F12）。そのままだと warmup が全体の 6.7% を食う |
| `scheduler_decay_steps` | **30,000（確定）** | `steps` と一致させる。上流の 100000 のままだと cosine が最初の 3/10 しか進まず、実質 lr がほぼ一定のまま終わる |
| `scheduler_decay_lr` | 5e-6 | 上流どおり |
| `optimizer_lr` | 5e-5 | 上流どおり |
| `save_freq` | `steps // 6` | 5〜6 個。1 ckpt のサイズを smoke で実測してディスクを確認 |
| `log_freq` | 50 | 単 GPU なので上流の 200 より細かく |

**中断・再開前提で組む。** `_maybe_resume` により `checkpoints/last` から再開できるようにし、時間上限のあるクラウドセッションを跨げるようにする。

### 8.3 推論レイテンシと replan の決め方

- 1 チャンク推論 = prefix forward 1 回（VLM 2B）+ denoise 10 回（action expert のみ、KV cache 再利用）。
- 制約は 2 つ。**(a) `/act` 1 リクエスト 10 秒**（F10、確実）、**(b) 1 エピソードの総推論時間**（`episode_timeout_sec=120` は配布版では未使用だが本番で有効化される可能性）。
- 予算式: `(300 / replan_steps) × latency_sec + env_step_time < 120`。
- 例: `latency = 0.6s` なら `replan=10` で 18s、`replan=16` で 11s。余裕あり。`latency = 2.5s` なら `replan=10` で 75s となり危険。
- **したがって replan は「実測 latency を測ってから決める」。** 暫定既定は `replan_steps=16`、A/B で `{10, 16, 25, 50}` を振る。
- RTC の重複区間 `L = chunk_size − replan_steps` は `replan` が小さいほど長くなり、平滑化の効きは強くなるが推論回数が増える。このトレードオフを A/B の表に載せる。

---

## 9. sprint contract（完了条件）

### SC-1 — overlay とデータ変換（P1 の 1〜6）

- Generator: `pytest .../tests/test_render_downsample.py test_dataset_config.py test_schema.py -q` が exit 0
- Evaluator:
  - 上流リポジトリに改変が無い（`git -C <上流> status` が clean）
  - `RenderDownsampleFn` が `ResizeImagesWithPadFn` の**直前**に入るテストが存在する
  - box/interpolate/INTER_AREA 同値性テストが存在する（前提の自己検証になっている）
  - カーネルがサンプル単位で 1 回だけ引かれるテストが `[T,C,H,W]` で書かれている
  - `p_nearest` の既定が **0.10**

### SC-2 — チャンク平滑化（P1 の 7）

- Generator: `pytest .../test_chunk_blending.py -q` が exit 0
- Evaluator:
  - `euler_integrate` が `denoise_fn` を引数に取り、モデル無しでテストできる構造
  - `guidance=None` と `weights=0` のビット一致テストがある
  - `dim_mask[6]=False` の非干渉テストがある
  - RTC と temporal ensembling の**両方**が実装され、`runtime_config.json` の 1 キーで切り替わる
  - `w_max < 1.0`（ハードマスク不採用）が既定値とテストで担保されている

### SC-3 — サイレント失敗ガード（P1 の 8〜9, P5 の 27, 30）

- Generator: `pytest .../test_state_utils.py test_policy_server_parity.py -q` が exit 0
- Evaluator: 以下がコード上に存在する
  - 推論経路に `resize_with_pad(224,224)` が**無条件で**入っている（F6）
  - `install_vendored_lerobot` に「別 lerobot が import 済みなら落とす」ガード
  - `assert_checkpoint_covers_model()` が起動パスに入っている
  - `mean_resizing=False` のコンテキストマネージャ
  - 画像の向き・グリッパ規約が `runtime_config.json` 由来で、コードに直書きされていない
  - `policy_server.py` と `verify_inference.py` が同じ `build_submission_runtime()` を通る

### SC-4 — 事実確定（P2〜P3）

- Generator: `artifacts/facts/facts.json` と PNG 一式が存在し、`prepare_dataset.py` が冪等に完了
- Evaluator: `facts.json` に `fps` / `image_hw` / `state_dim` / `state_layout` / `action_dim` / `gripper_convention` / `stats_keys` / `robot_type_before` / `robot_type_after` がすべて埋まり `TBD` が残っていない。`image_orientation.agentview` / `.wrist` が確定し**根拠 PNG が添付**されている

### SC-5 — 学習（P4）

- Generator: `smoke_ivla.sh` が exit 0、`probe_report.md` が生成、本学習が指定 step 到達
- Evaluator: probe 表と OOM 記録がある / `warmup`・`decay_steps` が実 step 数に再スケール済み（F12）/ `action_loss_only=true` に落とした場合は根拠が文書に残る / `train_config.json` に `RenderDownsampleFn` が確率つきで記録されている

### SC-6 — 提出物（P6〜P7）

- Generator: C1〜C4 がすべて PASS
- Evaluator: `MANIFEST.json` に上流 commit / 落としたキー / 追加ファイル / SHA256 がある / `vendor/lerobot` が上流と追加 1 ファイルを除いてバイト一致 / `learnable_tokens`・`learnable_tokens_in_proj` が**残っている** / `model.safetensors.index.json` が VLM ディレクトリに**無い** / ロード時間と推論レイテンシがログに数値で出る / `submission_template/` に変更が無い

### SC-7 — A/B（P7 の 37）

- Generator: `chunking.mode × replan_steps` の格子で評価を回し結果表を出力
- Evaluator: 成功率と jerk / SPARC / path length が同一シードで比較され、最終 `runtime_config.json` の値が表から説明できる

---

## 10. リスクと未確定事項

### 10.1 データを見ないと決まらないもの（P2 で潰す）

| # | 未確定事項 | 決まらないと何が壊れるか | 確認方法 |
|---|---|---|---|
| U1 | **グリッパ規約**（`[0,1]` か `[-1,1]` か） | 開閉が反転し、全タスクが 0% になる | `stats.json` の action dim6 min/max/mean + 実サンプルのヒストグラム |
| U2 | **画像の向き**（dataset と env で 4 通りの組合せ） | 静かに精度が落ちる。上下逆でも「動くが掴めない」 | PNG 目視 + 輝度プロファイル相関 |
| U3 | **`observation.state` の中身**（8 次元 EE か joint か） | state が全く別物になり、tokenize_state のプロンプトも壊れる | `info.json` の features + `stats.json` の次元 |
| U4 | **画像解像度**（本当に 256 か） | 128 なら downsample が恒等になり対策全体が空振り | `info.json` の features shape |
| U5 | **`stats.json` のキー構造**（統合済み 1 キーか、スイート別か） | スイート別だと `_pick_stats_key` が `ValueError` を投げる | キー一覧のダンプ |
| U6 | **fps**（20 か） | `image_delta_indices` の時間スパンがずれ WAN 教師の時間解像度が変わる | `info.json` |
| U7 | **正規化モード**（mean_std か min_max か） | 逆正規化が壊れ、行動のスケールが桁で違う | `train_config.json` の `normalize` transform の `mode` |

### 10.2 環境・リソースのリスク

| # | リスク | 影響 | 緩和 |
|---|---|---|---|
| R1 | **WAN 分岐が 96GB に載らない** | `video_loss` を諦めることになる | probe で早期判定。ラダー §8.1。`action_loss_only=true` の代替レシピを最初から用意 |
| R2 | **torch 2.11.0 の PyPI 既定 wheel が cu130 でない / GPU で動かない** | 提出物が採点環境で CUDA を掴めない | クリーン venv に requirements を入れて `torch.version.cuda` と `torch.cuda.is_available()` を印字（C2 の必須項目） |
| R3 | **`transformers==5.2.0` が torch 2.11 と合わない** | 起動時 ImportError | C2 で版を総当りし、動いた版に**完全固定** |
| R4 | torchcodec の ABI 不一致（手順書 §3.1） | 学習が進むのにカメラが黒画像 | デコード PNG の非ゼロ assert を setup に組み込む |
| R5 | `Qwen/Qwen3.5-2B` / `physical-intelligence/fast` の取得失敗 | 学習が起動しない | 事前に `HF_HOME` へキャッシュし `HF_HUB_OFFLINE=1` で再現確認 |
| R6 | 推論レイテンシが想定超（replan を上げられない） | 平滑化の効きが弱まる／エピソード予算超過 | `num_inference_steps` を 10 → 5 に落とす選択肢を A/B に含める。`flash-linear-attention` の有無で Gated DeltaNet の速度が数倍違うので推論環境でその有無を印字 |
| R7 | クラウドセッションの時間上限で学習が切れる | 進捗喪失 | `_maybe_resume` と `save_freq` を短めに |
| R8 | ディスク容量（データ + 6 チェックポイント） | 保存失敗で学習が落ちる | smoke で 1 ckpt サイズを実測し `save_freq` と保持数を決める |

### 10.3 設計上の未確定事項

| # | 事項 | 決め方 |
|---|---|---|
| D1 | ダウンサンプル確率の配分（box 0.45 / triangle 0.30 / cubic 0.15 / **nearest 0.10 固定**） | 提案値で本走。余力があれば box 0.60 / triangle 0.20 / cubic 0.10 の変種と A/B |
| D2 | `w_max`（RTC のソフト重み上限）0.8 | A/B（`{0.5, 0.8}`）。1.0 は実質ハードマスクなので採らない |
| D3 | `replan_steps` | レイテンシ実測後に §8.3 の予算式で決め A/B で確定 |
| D4 | RTC か temporal ensembling か | A/B。同等なら**実装が単純な ensembling を採る**（提出物のバグ面積が小さい） |
| D5 | VQA チェーンへの downsample 適用 | 現行レシピでは dead path。フラグを残し「効いていない」ことを文書化 |
| D6 | `enable_vqa_loss` / `video_loss_weight` を上流どおり据え置くか | R1 の probe 結果次第。落とす場合は理由を記録 |
| D7 | どのチェックポイント（step）を提出するか | 保存した 5〜6 個を `pipeline` で 2 エピソードずつ評価し、上位 2 個を本評価 |

---

## 11. 参考: 上流の該当箇所（Generator が読むべき最小集合）

| 目的 | ファイル:行 |
|---|---|
| transform の登録と挿入位置 | `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py:36-102`, `:210-232` |
| Resize / VQA Resize の実装 | `src/lerobot/transforms/core.py:128-146`, `:578-591` |
| 動画フレーム抽出（Resize の後に走る） | `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py:626-654` |
| `image_delta_indices` | `configuration_internvla_a1_5.py:411-427` |
| euler ループ（RTC を差し込む対象） | `modeling_internvla_a1_5.py:761-833` |
| WAN 構築条件と state_dict 除外 | `modeling_internvla_a1_5.py:576-594`, `:1373`, `:1426-1437` |
| schema 登録 API | `src/lerobot/dataset_schemas/registry.py:88-128`, `:175-189` |
| plugin 自動 import | `src/lerobot/utils/import_utils.py:133-155` |
| 推論バックエンドの参照実装（**リサイズが無効な点に注意**） | `evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py`, `canonical_preprocess.py:28-55` |
| state 生成と画像回転の参照 | `evaluation/LIBERO/model2libero_interface.py:11-32`, `:97-115` |
| 逆正規化と後処理 | `evaluation/LIBERO/policy_server/backends/base_backend.py:130-180` |
| 採点ハーネス側の観測・向き | `PARC-final/pipeline/remote_policy.py:11-19`, `pipeline/rollout.py:254-266` |
| 採点環境の解像度・step 上限 | `PARC-final/pipeline/config.py:52-66` |
| 提出物の制約 | `PARC-final/validate_submission.py:16-34` |
