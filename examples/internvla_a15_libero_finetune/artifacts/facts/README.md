# artifacts/facts/

P2（クラウド）で収集する**事実**の置き場。P1 の時点では空である。

ここが埋まるまで、`inference/runtime_config.json` の `image_orientation` と
`gripper.dataset_convention` は `"TBD"` のままにしておくこと。`"TBD"` で推論を
起動しようとすると `internvla_runtime.py` が例外を投げる（黙って既定値で動かさない）。

## 置くもの

| ファイル | 生成元 | 内容 |
|---|---|---|
| `facts.json` | `scripts/inspect_dataset.py` | fps / robot_type / 画像 shape / state 次元と統計 / action 次元と dim6 ヒストグラム / stats のキー / エピソード数・総フレーム数 |
| `dataset_agentview_*.png` `dataset_wrist_*.png` | `tools/dump_dataset_frames.py` | データセット側の生フレーム（各 8 枚） |
| `dataset_frames.json` | 同上 | 上記のメタデータ |
| `env_agentview_image_*.png` `env_robot0_eye_in_hand_image_*.png` | `tools/dump_env_frames.py` | 採点 env 側の生フレーム。`pipeline/rollout.py` の `[::-1]` の**手前**で保存する |
| `env_frames.json` | 同上 | 上記のメタデータ |
| `orientation_*.json` | `tools/orientation_match.py` | 4 変換 {none, flip_ud, flip_lr, rot180} の相関と argmax |
| `render_downsample_nearest.png` `render_downsample_default.png` | P3-22 の目視確認 | `p_nearest=1.0` に固定した実画像と既定確率の実画像 |

## 完了条件（計画 SC-4）

`facts.json` に次がすべて埋まり、`TBD` が 1 つも残っていないこと。

- `fps` / `image_hw` / `state_dim` / `state_layout` / `action_dim`
- `gripper_convention` / `stats_keys`
- `robot_type_before`（`inspect_dataset.py`）/ `robot_type_after`（`prepare_dataset.py`）

加えて `image_orientation.agentview` / `.wrist` が確定し、**根拠 PNG が添付**されていること。
相関の argmax だけで決めないこと。テーブル面・アーム・グリッパの位置関係を目視で確認する。
