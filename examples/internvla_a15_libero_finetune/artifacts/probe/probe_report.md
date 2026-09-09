# probe_report — InternVLA-A1.5 batch size probe

- 日時: 2026-09-08（クラウド RTX PRO 6000 Blackwell Server Edition / 97,887 MiB）
- 条件: 30 step。sec/step は **step 10 以降の中央値**（最初の数 step は cudnn ベンチマークと
  コンパイルで遅いので除く）
- `action_loss_only=false`（**WAN 動画ヘッド on**）/ `gradient_checkpointing=false`
- `vlm_lr_scale=0.1`（バックボーン VLM のみ lr を 0.1 倍）
- 上流 `lerobot_train.py` に gradient accumulation は無いので **実効バッチ = batch_size**

## 結果

| batch_size | GC | peak VRAM (MiB) | 使用率 | sec/step | サンプル/秒 | 30k step の所要 | 結果 |
|---:|:---|---:|---:|---:|---:|---:|:---|
| 2 | false | 39,651 | 41% | 0.481 | 4.16 | 4.0 h | OK |
| 4 | false | 53,495 | 56% | 0.561 | 7.13 | 4.7 h | OK |
| 8 | false | **73,071** | **76%** | **0.810** | **9.88** | **6.8 h** | **OK（採用）** |
| 16 | false | 96,949 | 99% | n/a | — | — | **OOM** |

- BS=2 / BS=8 は `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 込みの実測。
- BS=4 / BS=16 はそれ無しの実測（BS=8 は無しだと 80,581 MiB = 84%）。
  **`expandable_segments` で BS=8 のピークが 80.6GB → 73.1GB に下がり、スループットは
  0.812 → 0.810 s/step とほぼ変わらなかった**ので、本走では有効にしている。

## 判断

**`batch_size=8` を採用する。**

- OK の中で最大。かつ **BS=4 よりスループットも高い**（9.88 対 7.13 サンプル/秒）ので、
  実効バッチとサンプル効率の両方で勝っている。
- 96GB に **WAN 分岐（`action_loss_only=false`）が載ることを確認できた**（計画 R1 の判定）。
  よってメモリ削減ラダー（計画 §8.1）の 2〜4 は不要:
  - `gradient_checkpointing=true` にしない（スループット −25〜35% を払う理由が無い）
  - `action_loss_only=true` にしない（動画教師と foresight token の学習を維持できる）
- 30,000 step の所要見込みは **約 6.8 時間**。

## 補足: BS=16 の OOM について

peak 96,949 MiB / 97,887 MiB。`gradient_checkpointing=true` にすれば載る可能性はあるが、
BS=8 が余裕を持って回りスループットでも有利なので追試していない。

## 補足: 最初の実行で BS=2 が exit=127 になった件

1 回目の probe 中に `scripts/train_ivla_a15.sh` と `scripts/_train_common.sh` を編集して
しまい、**実行中のシェルスクリプトのバイトオフセットがずれて**壊れた（bash はスクリプトを
逐次読みするため）。OOM でも環境の問題でもない。上の表の BS=2 の行は編集後に取り直した
クリーンな実測値である。**学習・probe の実行中に該当スクリプトを編集しないこと。**
