# 参考例（examples）

| ファイル | 内容 |
|---|---|
| [smolvla_libero_spatial_lora.ipynb](smolvla_libero_spatial_lora.ipynb) | SmolVLA を LIBERO-plus Spatial で LoRA 追加学習する Google Colab ノートブック |
| [pi05_libero_finetune/](pi05_libero_finetune/) | pi0.5 を LIBERO 系データで LoRA 追加学習し、提出物にするまでのレシピ |
| [internvla_a15_libero_finetune/](internvla_a15_libero_finetune/) | InternVLA-A1.5 を LIBERO 系データでフルファインチューニングし、Python 3.10 の採点環境向けに提出物を組むまでのレシピ |

規模が異なるので、目的に応じて選ぶこと。

## smolvla_libero_spatial_lora.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で比較する。

### 使い方

1. Google Colab で開き、ランタイムのタイプを GPU（T4 で足りる）に変更する
2. 上から順に実行する。所要時間は T4 で数時間程度である
3. マージ済みモデル一式（zip）と、追加学習前後の成功率の比較（CSV）が出力される

学習条件は 10 タスク × 各 5 エピソード（計 50 エピソード）、3,000 steps、
バッチサイズ 1 で、Colab で完走することを優先した最小構成である。
性能を伸ばす場合はここを出発点に、自身の環境で条件を組み直すとよい。

### 提出物にするまでの作業

出力されるのは LeRobot 形式のモデル重みであり、これ単体では提出できない。
[submission_template/](../submission_template/) の `MyPolicy` にモデルを組み込み、
ポリシーサーバーの形にする。観測と action の仕様は
[submission_template/policy_server.py](../submission_template/policy_server.py)
の docstring にある。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。

### ノートブック内の評価と、本番の採点の違い

ノートブック内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | ノートブック | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval` | `python -m pipeline` + 提出したポリシーサーバー |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

### 実行環境

ノートブックの環境構築は Colab 向けで、[setup.sh](../setup.sh) とは独立している。
依存パッケージのバージョンが一致しない箇所があるため、評価と提出前チェックは
リポジトリ側の環境（`setup.sh` + `env.sh`）で行うこと。

ノートブックが利用する第三者製ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。

## pi05_libero_finetune/

lerobot の pi0.5 を LIBERO 系データで LoRA 追加学習し、提出用のポリシー
サーバーに組み込むまで一式。学習用 venv の構築スクリプト、学習ランチャー、
LoRA のマージ、提出サーバーを含む。

手順とハイパーパラメータは [pi05_libero_finetune/README.md](pi05_libero_finetune/README.md)
を参照。要点は次のとおり。

- **学習用 venv は評価用（ルートの `setup.sh`）とは別に作る。** 評価用 venv は
  採点環境の再現が目的であり、学習の依存で汚さないこと。
- lerobot は **v0.4.4** に固定する。0.5.0 以降は Python 3.12 以上を要求し、
  採点イメージの Python 3.10 に入らないため、学習と推論で版を揃えている。
- 学習が出力するのは LoRA アダプタなので、`scripts/merge_lora.py` でベース重みへ
  マージしてから提出物に同梱する（採点環境は外部通信を遮断するため）。
- ベース重みは Gemma Terms of Use で提供される。[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) を参照。

## internvla_a15_libero_finetune/

[InternVLA-A1.5](https://github.com/InternRobotics/InternVLA-A-series) を
`libero_combined_20hz` でフルファインチューニングし、Python 3.10 の採点環境で
動く提出物を組むまで一式。overlay パッケージ、チャンク境界の平滑化、提出物の
生成・検証スクリプト、手元（GPU 不要）で回る単体テストを含む。

手順とハイパーパラメータは
[internvla_a15_libero_finetune/README.md](internvla_a15_libero_finetune/README.md)
を参照。要点は次のとおり。

- **上流リポジトリを 1 バイトも書き換えない（overlay 方式）。** 上流には
  `lerobot_policy_*` を自動 import する plugin 機構があり、transform・dataset
  config・robot schema の登録がすべて公開 API で足りる。パッチを当てないので、
  提出物に同梱する `vendor/lerobot` を上流とバイト一致に保てる。
- **学習 256 / 採点 128 の解像度差を学習側で潰す。** 256→128 のダウンサンプルを
  挟み、カーネル（box / triangle / cubic / nearest）をランダムに振って頑健にする。
- **推論経路のリサイズを明示的に入れる。** 上流の推論バックエンドはリサイズが
  no-op になっており、学習 224 に対し推論が生解像度のまま VLM に入る。エラーは
  出ないまま精度だけ落ちるので、自前ランタイムでは学習と同一の関数を無条件で通す。
- ベース重みは CC BY-NC-SA 4.0（非商用）で提供される。[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) を参照。
