# サードパーティ ライセンス表記

本配布環境（PARC 2026 配布環境）は、`setup.sh` の実行時に第三者が
権利を有するソフトウェア・データセットを取得・インストールして利用する。これらは
本リポジトリには同梱しておらず、それぞれの提供元から取得される。各ソフトウェアの
著作権および利用条件は、それぞれのライセンスに従う。

以下は `setup.sh` が取得・インストールする主要な第三者製ソフトウェアとその
ライセンスの一覧である。正確・最新の条文は各配布元を参照すること。

---

## リポジトリ（`git clone` で取得）

| ソフトウェア | 取得元 | ライセンス |
|---|---|---|
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT License（Copyright (c) 2023 Lifelong Robot Learning） |
| LIBERO-plus | https://github.com/sylvestf/LIBERO-plus | **ライセンス未記載**（下記の注意を参照） |

### LIBERO-plus のライセンスに関する注意

`sylvestf/LIBERO-plus` のリポジトリには本表記作成時点で LICENSE ファイルおよび
README 上のライセンス記載が確認できない。明示的なライセンスが付与されていない
著作物は、既定では著作権者に全権利が留保される点に留意すること。利用条件が
不明確なため、再配布や商用利用の可否等は、必要に応じて原著作者に確認すること。
なお LIBERO-plus は LIBERO（MIT）を基盤とする派生物であり、LIBERO 由来部分には
LIBERO の MIT License が及ぶ。

---

## データセット（Hugging Face Hub から取得）

| データセット | 取得元 | ライセンス |
|---|---|---|
| LIBERO-plus assets（`assets.zip`） | https://huggingface.co/datasets/Sylvest/LIBERO-plus | LIBERO-plus に準ずる（上記の注意を参照） |

---

## Python パッケージ（`pip install` で取得）

`setup.sh` がピン止めしてインストールする主要パッケージとそのライセンス。
バージョンは配布時点のもの。推移的依存（各パッケージがさらに引く依存）は
含めていない。

| パッケージ | バージョン | ライセンス |
|---|---|---|
| torch | 2.11.0+cpu | BSD-3-Clause |
| mujoco | 3.7.0 | Apache-2.0 |
| robosuite | 1.4.0 | MIT License |
| numpy | 1.26.4 | BSD-3-Clause |
| gym | 0.25.2 | MIT License |
| bddl | 3.6.0 | MIT License |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| easydict | 1.13 | LGPL-3.0 |
| hydra-core | 1.3.2 | MIT License |
| einops | 0.8.2 | MIT License |
| opencv-python-headless | 4.11.0.86 | Apache-2.0 |
| scipy | 1.15.3 | BSD-3-Clause |
| pyyaml | 6.0.3 | MIT License |
| h5py | 3.16.0 | BSD-3-Clause |
| Pillow | 12.3.0 | MIT-CMU (HPND) |
| termcolor | 3.3.0 | MIT License |
| tqdm | 4.70.0 | MPL-2.0 AND MIT |
| matplotlib | 3.10.9 | PSF-based (matplotlib license) |
| requests | 2.34.2 | Apache-2.0 |
| msgpack | 1.2.1 | Apache-2.0 |
| fastapi | 0.140.13 | MIT License |
| uvicorn | 0.51.0 | BSD-3-Clause |
| huggingface_hub | 1.25.1 | Apache-2.0 |
| wand | 0.7.2 | MIT License |
| scikit-image | 0.25.2 | BSD-3-Clause |
| pytest | 9.1.1 | MIT License |

> 注: `easydict` は LGPL-3.0、`tqdm` は MPL-2.0 AND MIT。これらは他パッケージの
> 寛容型ライセンス（MIT / BSD / Apache-2.0）とは条件が異なるため、配布形態を
> 変える場合は各条項を確認すること。

---

## 学習の参考例（`examples/`）が取得するもの

以下は `setup.sh` ではなく、[examples/](examples/) 配下の手順で取得される。
提出には必須ではない。

### pi0.5 追加学習レシピ（[examples/pi05_libero_finetune/](examples/pi05_libero_finetune/)）

| ソフトウェア／重み | 取得元 | ライセンス |
|---|---|---|
| lerobot（v0.4.4） | https://github.com/huggingface/lerobot | Apache-2.0 |
| pi0.5 ベース重み | https://huggingface.co/lerobot/pi05_libero_base | **Gemma Terms of Use**（下記の注意を参照） |
| transformers | PyPI | Apache-2.0 |
| peft | PyPI | Apache-2.0 |

`patches/` に含まれる差分は lerobot（Apache-2.0）に対する変更であり、同ライセンスに従う。

### pi0.5 ベース重みのライセンスに関する注意

`lerobot/pi05_libero_base` は Apache-2.0 ではなく **Gemma Terms of Use** で提供される
（pi0.5 が PaliGemma / Gemma を基盤とするため）。Gemma のライセンスには利用禁止事項
（Gemma Prohibited Use Policy）および再配布時の条件が定められている。

### InternVLA-A1.5 追加学習レシピ（[examples/internvla_a15_libero_finetune/](examples/internvla_a15_libero_finetune/)）

| ソフトウェア／重み | 取得元 | ライセンス |
|---|---|---|
| InternVLA-A-series（コード） | https://github.com/InternRobotics/InternVLA-A-series | Apache-2.0 |
| InternVLA-A1.5 ベース重み | https://huggingface.co/InternRobotics/InternVLA-A1.5 | **CC BY-NC-SA 4.0**（非商用・下記の注意を参照） |
| Qwen3.5-2B（VLM バックボーン） | https://huggingface.co/Qwen | Apache-2.0（配布元の表記を確認すること） |
| Wan2.2（動画補助タスクの DiT / VAE） | https://github.com/Wan-Video/Wan2.2 | Apache-2.0 |
| FAST action tokenizer | https://huggingface.co/physical-intelligence/fast | Apache-2.0 |
| transformers / accelerate / diffusers | PyPI | Apache-2.0 |
| safetensors | PyPI | Apache-2.0 |
| draccus | PyPI | MIT License |

このレシピは上流 `InternVLA-A-series` を**書き換えない**。`lerobot_policy_parc/` は
上流の公開 API（`register_subclass` / `load_schemas_from_path`）だけを使う独立した
追加コードであり、上流の派生物ではない。ただし提出物には上流の `src/lerobot` を
そのまま同梱するため、その部分には上流の Apache-2.0 が及ぶ。

### InternVLA-A1.5 ベース重みのライセンスに関する注意

`InternRobotics/InternVLA-A1.5` の重みは **CC BY-NC-SA 4.0** で提供される。
コード（Apache-2.0）とはライセンスが異なる点に注意すること。CC BY-NC-SA 4.0 は

- **非商用利用に限る**（NonCommercial）
- 表示（Attribution）が必要
- 改変物を頒布する場合は**同一ライセンスで継承**する（ShareAlike）

ことを求める。追加学習した重みは原重みの派生物にあたるため、再配布する場合は
同じ条件が及ぶ。コンペへの提出を超えて利用・公開する場合は、配布元の最新の表記と
条文を必ず確認すること。

Wan2.2 の重みはコードと別に条件が付く場合がある。動画補助タスク（`action_loss_only=false`）
を使う場合は配布元の表記を確認すること。学習後のチェックポイントには WAN の重みは
含まれない（上流の `state_dict()` が `model.wan_video_model.*` を除外するため）。

---

各ライセンスの全文は、`setup.sh` 実行後に取得された各パッケージ／リポジトリの
配布物内（例: 各 `git clone` 先の `LICENSE`、`venv/lib/.../site-packages/<pkg>/`
配下のライセンスファイルや `*.dist-info/`）に含まれる。
