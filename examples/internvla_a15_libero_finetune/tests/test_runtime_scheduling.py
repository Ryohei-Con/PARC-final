"""``InternVLARuntime`` のスケジューリングと逆正規化の単体テスト（計画 P1-8 / P5-29）。

モデルもチェックポイントも要らない。``_predict_chunk_normalized`` をスタブに差し替えて
``get_action`` を**本番と同じ経路で**駆動する。ここに以下の回帰テストを置く:

- **推論回数が ``ceil(steps / replan_steps)`` になる**（3 モードすべて）。
  ``need_new_chunk`` の条件に ``_current_chunk_env is None`` が入っていると、env 空間の
  チャンクを持たない ensemble モードで毎ステップ推論になり ``replan_steps`` が死ぬ。
  計画 §8.3 の予算式 ``(300 / replan) * latency < 120s`` の前提が崩れ、SC-7 の
  rtc <-> ensemble A/B も比較不能になる。
- **``set_chunking_mode()`` が ensembler を構築 / 破棄する**。``chunking["mode"]`` の
  生 dict 書き換えは、ensembler が無いまま ensemble を名乗る状態を作る（= 実体は
  none 経路なのに ensemble として計測する）。
- ``reset()`` がチャンク状態と ensembler の内部状態を消す。
- ``_denormalize`` が 3 モード（``mean_std`` / ``min_max`` / ``q01_q99``）で
  正規化の**厳密な逆**になっている。``q01_q99`` で ``min`` / ``max`` を使うと
  ``[q01, q99] ⊂ [min, max]`` の比だけ行動が過大になる（計画 U7）。
- ``_to_env_action`` の clip とグリッパ二値化。
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from internvla_runtime import InternVLARuntime  # noqa: E402

from conftest import requires_lerobot  # noqa: E402

ACTION_DIM = 32  # モデルの出力次元（padding 込み）
REAL_DIM = 7  # stats.json 側の実次元
CHUNK = 50

BASE_CFG = {
    "image_orientation": {"agentview": "none", "wrist": "none"},
    "gripper": {"dataset_convention": "zero_one"},
    "chunking": {
        "mode": "rtc",
        "replan_steps": 16,
        "w_max": 0.8,
        "schedule": "linear",
        "exclude_dims": [6],
        "ensemble_m": 0.1,
    },
    "resize": {"height": 224, "width": 224},
    "num_inference_steps": 10,
}


def _stats(dim: int = REAL_DIM) -> dict[str, np.ndarray]:
    """``[q01, q99]`` が ``[min, max]`` の**内側**にある統計。

    内側にあるからこそ「q01_q99 なのに min/max で逆正規化する」バグが
    数値のズレとして現れる。
    """
    return {
        "min": np.full(dim, -2.0, dtype=np.float32),
        "max": np.full(dim, 2.0, dtype=np.float32),
        "q01": np.full(dim, -0.4, dtype=np.float32),
        "q99": np.full(dim, 0.6, dtype=np.float32),
        "mean": np.full(dim, 0.1, dtype=np.float32),
        "std": np.full(dim, 0.25, dtype=np.float32),
    }


class _StubbedRuntime:
    """``get_action`` を駆動できる最小構成のランタイム + 推論回数カウンタ。"""

    def __init__(self, mode: str = "rtc", replan: int = 16, norm_mode: str = "min_max") -> None:
        cfg = copy.deepcopy(BASE_CFG)
        cfg["chunking"]["mode"] = mode
        cfg["chunking"]["replan_steps"] = replan

        self.runtime = InternVLARuntime(ckpt_dir=Path("."), vlm_dir=Path("."), cfg=cfg)
        self.runtime.chunk_size = CHUNK
        self.runtime.action_dim = ACTION_DIM
        self.runtime.action_stats = _stats()
        self.runtime._norm_mode_cache = norm_mode
        # load() を短絡させる。モデルもチェックポイントも触らない。
        self.runtime._loaded = True
        self.runtime._obs_to_sample = lambda obs: obs
        self.runtime._predict_chunk_normalized = self._predict
        # load() の末尾と同じ手順で ensembler を用意する。
        self.runtime._sync_ensembler()

        self.calls: list[object] = []

    def _predict(self, sample, guidance):
        self.calls.append(guidance)
        index = len(self.calls)
        chunk = np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)
        steps = np.linspace(-0.5, 0.5, CHUNK, dtype=np.float32)
        for dim in range(REAL_DIM):
            chunk[:, dim] = steps * (0.1 * (dim + 1)) + 0.01 * index
        return chunk

    @property
    def count(self) -> int:
        return len(self.calls)


def _dummy_obs(step: int) -> dict[str, np.ndarray]:
    return {"step": np.asarray([step], dtype=np.float32)}


# --------------------------------------------------------------------------- #
# 1. HIGH-1 回帰: 推論は replan_steps ごとにしか走らない
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["none", "rtc", "ensemble"])
@pytest.mark.parametrize("replan", [10, 16, 25, 50])
def test_inference_count_follows_replan_steps(mode, replan):
    """300 step で ``ceil(300 / replan)`` 回しかモデルを呼ばない（全モード）。

    ensemble だけ毎ステップ推論する、という退化を直接検出する。計画 §8.3 の
    予算式が成立する前提であり、SC-7 の A/B が比較可能である前提でもある。
    """
    harness = _StubbedRuntime(mode=mode, replan=replan)
    harness.runtime.reset(instruction="pick up the black bowl")

    steps = 300
    for step in range(steps):
        action = harness.runtime.get_action(_dummy_obs(step))
        assert action.shape == (7,)
        assert action.dtype == np.float32
        assert np.all(np.isfinite(action))

    expected = math.ceil(steps / replan)
    assert harness.count == expected, (
        f"mode={mode} replan={replan}: ran {harness.count} inferences, expected {expected}"
    )


def test_ensemble_matches_the_other_modes_in_inference_count():
    """レビューの実測（none=19 / rtc=19 / ensemble=300）を固定する。"""
    counts = {}
    for mode in ("none", "rtc", "ensemble"):
        harness = _StubbedRuntime(mode=mode, replan=16)
        harness.runtime.reset()
        for step in range(300):
            harness.runtime.get_action(_dummy_obs(step))
        counts[mode] = harness.count
    assert counts == {"none": 19, "rtc": 19, "ensemble": 19}


def test_chunk_cursor_wraps_at_chunk_size_even_if_replan_is_larger():
    """``replan_steps > chunk_size`` でもチャンク長で必ず引き直す。"""
    harness = _StubbedRuntime(mode="none", replan=80)
    harness.runtime.reset()
    for step in range(150):
        harness.runtime.get_action(_dummy_obs(step))
    assert harness.count == math.ceil(150 / CHUNK)


def test_rtc_passes_guidance_from_the_second_chunk_on():
    """rtc は初回のみ guidance なし、以降は前チャンク由来の guidance を渡す。"""
    harness = _StubbedRuntime(mode="rtc", replan=16)
    harness.runtime.reset()
    for step in range(40):
        harness.runtime.get_action(_dummy_obs(step))

    assert harness.calls[0] is None
    assert all(call is not None for call in harness.calls[1:])
    spec = harness.calls[1]
    assert spec.overlap == CHUNK - 16
    assert not bool(spec.dim_mask[6]), "gripper must stay outside the guidance"


def test_none_and_ensemble_never_pass_guidance():
    for mode in ("none", "ensemble"):
        harness = _StubbedRuntime(mode=mode, replan=16)
        harness.runtime.reset()
        for step in range(40):
            harness.runtime.get_action(_dummy_obs(step))
        assert all(call is None for call in harness.calls), mode


# --------------------------------------------------------------------------- #
# 2. HIGH-2 回帰: set_chunking_mode が唯一の入口
# --------------------------------------------------------------------------- #
def test_set_chunking_mode_builds_and_tears_down_the_ensembler():
    harness = _StubbedRuntime(mode="rtc")
    runtime = harness.runtime
    assert runtime._ensembler is None

    runtime.set_chunking_mode("ensemble")
    assert runtime.chunking["mode"] == "ensemble"
    assert runtime._ensembler is not None
    assert runtime._ensembler.chunk_size == CHUNK
    assert runtime._ensembler.action_dim == ACTION_DIM

    runtime.set_chunking_mode("rtc")
    assert runtime._ensembler is None, "the ensembler must be torn down when leaving ensemble mode"

    runtime.set_chunking_mode("none")
    assert runtime._ensembler is None


def test_set_chunking_mode_actually_changes_the_executed_path():
    """ensemble に切り替えたら ensemble の経路が走る（none 経路に落ちない）。

    ensemble は重複チャンクの平均を返すので、``_current_chunk_env`` を持たない。
    ここが None のまま action が返るのが ensemble 経路の指紋。
    """
    harness = _StubbedRuntime(mode="none", replan=16)
    runtime = harness.runtime

    runtime.reset()
    runtime.get_action(_dummy_obs(0))
    assert runtime._current_chunk_env is not None  # none 経路の指紋

    runtime.set_chunking_mode("ensemble")
    runtime.get_action(_dummy_obs(0))
    assert runtime._current_chunk_env is None
    assert len(runtime._ensembler) == 1


def test_raw_dict_write_of_the_mode_is_rejected():
    """``chunking["mode"]`` の直接書き換えは黙って none 経路に落とさず例外にする。

    これを許すと ``verify_inference.py`` が別モードの latency と jerk を
    ``[chunking=ensemble]`` として印字してしまう（計画 B8 違反）。
    """
    harness = _StubbedRuntime(mode="rtc")
    harness.runtime.chunking["mode"] = "ensemble"  # 生 dict の書き換え（禁止経路）
    with pytest.raises(RuntimeError, match="set_chunking_mode"):
        harness.runtime.get_action(_dummy_obs(0))


def test_ensembler_is_rebuilt_when_the_model_dims_change():
    """``load()`` 前に ensemble へ切り替えても、確定した chunk_size で作り直される。

    ``set_chunking_mode()`` はロード前にも呼べる（``verify_inference.py`` の
    ``--chunking`` は ``load()`` 後だが、``policy_server`` 側の順序に依存しない）。
    その場合の ensembler は暫定次元なので、``load()`` 末尾の ``_sync_ensembler()``
    で作り直さないと ``push()`` が shape 不一致で落ちる。
    """
    harness = _StubbedRuntime(mode="ensemble")
    runtime = harness.runtime
    first = runtime._ensembler
    assert first is not None

    runtime.chunk_size = 25  # load() が config から確定させた想定
    runtime._sync_ensembler()
    assert runtime._ensembler is not first
    assert runtime._ensembler.chunk_size == 25

    # 同じ次元でもう一度呼んでも作り直さず、中身だけ空にする。
    same = runtime._ensembler
    runtime._sync_ensembler()
    assert runtime._ensembler is same
    assert len(runtime._ensembler) == 0


def test_set_chunking_mode_rejects_unknown_modes():
    harness = _StubbedRuntime()
    with pytest.raises(ValueError, match="unknown chunking mode"):
        harness.runtime.set_chunking_mode("smooth")


def test_set_chunking_mode_clears_the_chunk_state():
    harness = _StubbedRuntime(mode="rtc", replan=16)
    runtime = harness.runtime
    runtime.reset()
    for step in range(5):
        runtime.get_action(_dummy_obs(step))
    assert runtime._chunk_cursor == 5

    runtime.set_chunking_mode("none")
    assert runtime._chunk_cursor == 0
    assert runtime._prev_chunk_norm is None
    assert runtime._current_chunk_env is None
    assert runtime._has_chunk is False


def test_verify_inference_does_not_write_the_chunking_dict():
    """``verify_inference.py`` に ``chunking["mode"] = ...`` が残っていないこと。

    生 dict の書き換えが残る限り、同じズレ（別モードの数値を印字）が再発する。
    """
    import ast

    verify_py = Path(__file__).resolve().parent.parent / "inference" / "verify_inference.py"
    tree = ast.parse(verify_py.read_text(encoding="utf-8"), filename=str(verify_py))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                if target.value.attr == "chunking":
                    offenders.append(ast.dump(target))
    assert not offenders, (
        "verify_inference.py writes runtime.chunking[...] directly; "
        f"go through set_chunking_mode() instead: {offenders}"
    )


# --------------------------------------------------------------------------- #
# 3. reset()
# --------------------------------------------------------------------------- #
def test_reset_clears_chunk_state_and_ensembler():
    harness = _StubbedRuntime(mode="ensemble", replan=16)
    runtime = harness.runtime
    runtime.reset()
    for step in range(20):
        runtime.get_action(_dummy_obs(step))

    assert len(runtime._ensembler) > 0
    assert runtime._prev_chunk_norm is not None
    assert runtime._has_chunk is True

    runtime.reset(instruction="second episode")
    assert len(runtime._ensembler) == 0
    assert runtime._prev_chunk_norm is None
    assert runtime._chunk_cursor == 0
    assert runtime._current_chunk_env is None
    assert runtime._has_chunk is False
    assert runtime.instruction == "second episode"

    # reset 直後の 1 step 目は必ず推論する。
    before = harness.count
    runtime.get_action(_dummy_obs(0))
    assert harness.count == before + 1


# --------------------------------------------------------------------------- #
# 4. _denormalize: 正規化 -> 逆正規化の往復（計画 U7）
# --------------------------------------------------------------------------- #
def _normalize_reference(x: np.ndarray, stats: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """上流 ``NormalizeTransformFn``（transforms/core.py:296-313）と同じ式。"""
    eps = np.float32(1e-6)
    if mode == "mean_std":
        return (x - stats["mean"]) / (stats["std"] + eps)
    if mode == "min_max":
        low, high = stats["min"], stats["max"]
    elif mode == "q01_q99":
        low, high = stats["q01"], stats["q99"]
    else:
        raise ValueError(mode)
    return 2.0 * (x - low) / (high - low + eps) - 1.0


@pytest.mark.parametrize("mode", ["mean_std", "min_max", "q01_q99"])
def test_denormalize_is_the_exact_inverse_of_normalize(mode):
    harness = _StubbedRuntime(norm_mode=mode)
    stats = harness.runtime.action_stats

    rng = np.random.default_rng(20260908)
    raw = rng.uniform(-0.35, 0.55, size=(CHUNK, REAL_DIM)).astype(np.float32)
    normalized = _normalize_reference(raw, stats, mode).astype(np.float32)

    # モデルは padding 込みの 32 次元を返す。余った次元は無視される。
    padded = np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)
    padded[:, :REAL_DIM] = normalized

    recovered = harness.runtime._denormalize(padded)
    assert recovered.shape == (CHUNK, REAL_DIM)
    np.testing.assert_allclose(recovered, raw, atol=1e-5, rtol=0)


@requires_lerobot
@pytest.mark.parametrize("mode", ["mean_std", "min_max", "q01_q99"])
def test_denormalize_round_trips_against_the_upstream_transform(mode):
    """参照実装を自作の式ではなく**上流の transform 本体**にした往復テスト。"""
    import torch
    from lerobot.transforms.core import NormalizeTransformFn
    from lerobot.utils.constants import ACTION

    harness = _StubbedRuntime(norm_mode=mode)
    stats = harness.runtime.action_stats

    rng = np.random.default_rng(7)
    raw = rng.uniform(-0.3, 0.5, size=(CHUNK, REAL_DIM)).astype(np.float32)

    normalizer = NormalizeTransformFn(
        selected_keys=[ACTION], norm_stats={ACTION: dict(stats)}, mode=mode
    )
    normalized = normalizer({ACTION: torch.from_numpy(raw.copy())})[ACTION].numpy()

    padded = np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)
    padded[:, :REAL_DIM] = normalized

    recovered = harness.runtime._denormalize(padded)
    np.testing.assert_allclose(recovered, raw, atol=1e-5, rtol=0)


def test_q01_q99_does_not_fall_back_to_min_max():
    """``q01_q99`` を ``min`` / ``max`` で逆変換すると値が過大になることを固定する。

    ``[q01, q99] = [-0.4, 0.6]`` に対し ``[min, max] = [-2, 2]`` なので、
    取り違えるとスケールが 4 倍になる。
    """
    harness = _StubbedRuntime(norm_mode="q01_q99")
    stats = harness.runtime.action_stats

    normalized = np.zeros((1, ACTION_DIM), dtype=np.float32)  # 正規化空間の中央
    recovered = harness.runtime._denormalize(normalized)[0]

    q_mid = (stats["q01"] + stats["q99"]) / 2.0
    minmax_mid = (stats["min"] + stats["max"]) / 2.0
    np.testing.assert_allclose(recovered, q_mid, atol=1e-5, rtol=0)
    assert not np.allclose(recovered, minmax_mid)


@pytest.mark.parametrize(
    ("mode", "missing"),
    [("q01_q99", "q01"), ("q01_q99", "q99"), ("mean_std", "std"), ("min_max", "max")],
)
def test_denormalize_raises_when_a_required_stat_is_missing(mode, missing):
    """必要な統計が無いとき、黙って別のキーで代用せず例外にする。"""
    harness = _StubbedRuntime(norm_mode=mode)
    harness.runtime.action_stats = {
        key: value for key, value in harness.runtime.action_stats.items() if key != missing
    }
    with pytest.raises(KeyError, match=missing):
        harness.runtime._denormalize(np.zeros((1, ACTION_DIM), dtype=np.float32))


def test_denormalize_rejects_unknown_modes():
    harness = _StubbedRuntime(norm_mode="whatever")
    with pytest.raises(ValueError, match="unknown normalization mode"):
        harness.runtime._denormalize(np.zeros((1, ACTION_DIM), dtype=np.float32))


# --------------------------------------------------------------------------- #
# 5. MEDIUM-4: 正規化モードはホットパスで読み直さない
# --------------------------------------------------------------------------- #
def test_norm_mode_is_resolved_once(monkeypatch):
    """``train_config.json`` の読み直しが 1 回で済むこと（推論ホットパス対策）。"""
    harness = _StubbedRuntime(norm_mode="min_max")
    runtime = harness.runtime
    runtime._norm_mode_cache = None  # キャッシュを空に戻す

    calls = {"n": 0}

    def counting_norm_mode():
        calls["n"] += 1
        return "min_max"

    runtime._norm_mode = counting_norm_mode

    for _ in range(5):
        runtime._denormalize(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert runtime._resolved_norm_mode() == "min_max"
    assert calls["n"] == 1


def test_get_action_does_not_reparse_train_config():
    """ensemble モードでも ``_norm_mode`` は 1 回しか呼ばれない。"""
    harness = _StubbedRuntime(mode="ensemble", replan=16)
    runtime = harness.runtime
    runtime._norm_mode_cache = None

    calls = {"n": 0}

    def counting_norm_mode():
        calls["n"] += 1
        return "min_max"

    runtime._norm_mode = counting_norm_mode
    runtime.reset()
    for step in range(100):
        runtime.get_action(_dummy_obs(step))
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# 6. _to_env_action
# --------------------------------------------------------------------------- #
def test_to_env_action_clips_to_the_action_stats_range():
    harness = _StubbedRuntime()
    runtime = harness.runtime

    raw = np.array([10.0, -10.0, 0.5, -0.5, 3.0, -3.0, 0.9], dtype=np.float32)
    out = runtime._to_env_action(raw)

    assert out.shape == (7,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:6], [2.0, -2.0, 0.5, -0.5, 2.0, -2.0], atol=0, rtol=0)


def test_to_env_action_binarizes_the_gripper():
    harness = _StubbedRuntime()
    runtime = harness.runtime
    close = runtime.gripper["env_close"]
    open_ = runtime.gripper["env_open"]

    below = runtime._to_env_action(np.array([0, 0, 0, 0, 0, 0, 0.1], dtype=np.float32))
    above = runtime._to_env_action(np.array([0, 0, 0, 0, 0, 0, 0.9], dtype=np.float32))
    assert float(below[6]) == close
    assert float(above[6]) == open_
    assert float(below[6]) != float(above[6])


def test_to_env_action_drops_padding_dims():
    harness = _StubbedRuntime()
    padded = np.zeros(ACTION_DIM, dtype=np.float32)
    padded[:7] = 0.25
    harness.runtime.action_stats = _stats(ACTION_DIM)
    out = harness.runtime._to_env_action(padded)
    assert out.shape == (7,)


def test_to_env_action_rejects_short_vectors():
    harness = _StubbedRuntime()
    with pytest.raises(ValueError, match="at least 7"):
        harness.runtime._to_env_action(np.zeros(6, dtype=np.float32))


def test_to_env_action_rejects_non_finite_values():
    harness = _StubbedRuntime()
    harness.runtime.action_stats = dict(
        _stats(), min=np.full(REAL_DIM, -np.inf, dtype=np.float32),
        max=np.full(REAL_DIM, np.inf, dtype=np.float32),
    )
    bad = np.zeros(7, dtype=np.float32)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        harness.runtime._to_env_action(bad)


def test_gripper_is_always_binary_over_a_full_episode():
    """全ステップで ``action[6]`` が ``{env_close, env_open}`` に入る（C2 の項目）。"""
    harness = _StubbedRuntime(mode="ensemble", replan=16)
    runtime = harness.runtime
    allowed = {runtime.gripper["env_close"], runtime.gripper["env_open"]}
    runtime.reset()
    for step in range(120):
        action = runtime.get_action(_dummy_obs(step))
        assert float(action[6]) in allowed


# --------------------------------------------------------------------------- #
# 7. MEDIUM-2 回帰: verify_inference の期待推論回数が実装と一致する
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("replan", [1, 7, 10, 16, 25, 50, 60, 80, 300])
def test_expected_inference_calls_matches_the_real_runtime(replan):
    """``verify_inference.expected_inference_calls`` が**実測回数**と一致すること。

    自己チェックの期待値を ``ceil(steps / replan)`` にしていると、
    ``replan > chunk_size`` で actual 6 vs expected 5 のようにずれ、
    **正しい実装を "replan_steps is not being honoured" と誤報する**
    （チャンクは自分の長さを超えて引き延ばせないので実装の方が正しい）。
    """
    from verify_inference import expected_inference_calls

    steps = 300
    harness = _StubbedRuntime(mode="none", replan=replan)
    harness.runtime.reset()
    for step in range(steps):
        harness.runtime.get_action(_dummy_obs(step))

    expected = expected_inference_calls(steps, replan, harness.runtime.chunk_size)
    assert harness.count == expected, (
        f"replan={replan}: runtime ran {harness.count} inferences but "
        f"expected_inference_calls() predicted {expected}"
    )
    assert expected == math.ceil(steps / min(replan, CHUNK))


def test_expected_inference_calls_is_not_plain_ceil_over_replan():
    """``replan > chunk_size`` で ``ceil(steps/replan)`` と**異なる**こと。

    この差が無いと上のテストは ``ceil(steps/replan)`` のままでも通ってしまう。
    """
    from verify_inference import expected_inference_calls

    assert expected_inference_calls(300, 60, CHUNK) == 6
    assert math.ceil(300 / 60) == 5  # 誤報していた側の値
    assert expected_inference_calls(0, 16, CHUNK) == 0


def test_replan_interval_is_the_single_definition_of_the_interval():
    """実装側の ``replan_interval()`` と自己チェック側の式が一致すること。"""
    from verify_inference import replan_interval as verify_interval

    for replan in (1, 16, 50, 60, 999):
        harness = _StubbedRuntime(mode="rtc", replan=replan)
        runtime = harness.runtime
        assert runtime.replan_interval() == verify_interval(replan, runtime.chunk_size)
        assert runtime.replan_interval() == min(replan, CHUNK)


def test_load_warns_when_replan_exceeds_the_chunk_size(caplog):
    """``replan_steps > chunk_size`` は **warning**（エラーではない）。"""
    import logging

    harness = _StubbedRuntime(mode="none", replan=60)
    runtime = harness.runtime

    with caplog.at_level(logging.WARNING, logger="internvla_runtime"):
        warned = runtime._warn_if_replan_exceeds_chunk()
    assert warned is True
    assert any("replan_steps=60" in record.getMessage() for record in caplog.records)
    assert any("chunk_size=50" in record.getMessage() for record in caplog.records)

    # 起動は止めない（設定として安全側に倒れるだけ）。
    runtime.reset()
    action = runtime.get_action(_dummy_obs(0))
    assert action.shape == (7,)


def test_load_does_not_warn_for_a_sane_replan(caplog):
    import logging

    harness = _StubbedRuntime(mode="none", replan=16)
    with caplog.at_level(logging.WARNING, logger="internvla_runtime"):
        warned = harness.runtime._warn_if_replan_exceeds_chunk()
    assert warned is False
    assert not [r for r in caplog.records if "replan_steps" in r.getMessage()]


def test_load_calls_the_replan_warning():
    """``load()`` の末尾で警告チェックが呼ばれていること（AST）。"""
    import ast

    runtime_py = Path(__file__).resolve().parent.parent / "inference" / "internvla_runtime.py"
    tree = ast.parse(runtime_py.read_text(encoding="utf-8"), filename=str(runtime_py))
    load_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load"
    )
    assert "_warn_if_replan_exceeds_chunk()" in ast.unparse(load_fn)


# --------------------------------------------------------------------------- #
# 8. LOW 回帰: _CallCounter は外せる
# --------------------------------------------------------------------------- #
def test_call_counter_can_be_uninstalled_from_a_plain_runtime():
    """元がクラス側のメソッドなら、外したあとインスタンス属性を残さないこと。"""
    from verify_inference import _CallCounter

    harness = _StubbedRuntime(mode="none", replan=16)
    runtime = harness.runtime
    # 素の状態（インスタンス属性なし）を作る。
    del runtime._predict_chunk_normalized
    assert "_predict_chunk_normalized" not in vars(runtime)

    with _CallCounter(runtime) as counter:
        assert "_predict_chunk_normalized" in vars(runtime)
        assert counter.count == 0

    assert "_predict_chunk_normalized" not in vars(runtime), (
        "uninstall() must restore the class-level method, not leave a bound method behind"
    )


def test_call_counter_restores_a_stubbed_method_and_stops_counting():
    from verify_inference import _CallCounter

    harness = _StubbedRuntime(mode="none", replan=16)
    runtime = harness.runtime
    original = runtime._predict_chunk_normalized

    with _CallCounter(runtime) as counter:
        runtime.reset()
        for step in range(60):
            runtime.get_action(_dummy_obs(step))
        counted = counter.count
        assert counted == math.ceil(60 / 16)

    assert runtime._predict_chunk_normalized is original
    # 外したあとの呼び出しは数えられない。
    runtime.reset()
    for step in range(60):
        runtime.get_action(_dummy_obs(step))
    assert counter.count == counted

    counter.uninstall()  # 二重呼び出しは無害
    assert runtime._predict_chunk_normalized is original


# --------------------------------------------------------------------------- #
# 9. LOW 回帰: _action_stat のメッセージが要求元を正しく言う
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", ["min", "max"])
def test_to_env_action_requires_min_max_in_every_normalization_mode(missing):
    """``_to_env_action`` の clip は正規化モードと無関係に min/max を要求する。

    その事実を「正規化モードが要求している」と説明すると、``mean_std`` の
    checkpoint で min/max が欠けたときに**嘘の説明**の例外になる。
    """
    harness = _StubbedRuntime(norm_mode="mean_std")
    runtime = harness.runtime
    runtime.action_stats = {
        key: value for key, value in runtime.action_stats.items() if key != missing
    }

    # 逆正規化そのものは mean/std だけで通る（min/max に依存しない）。
    denormalized = runtime._denormalize(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert denormalized.shape == (1, REAL_DIM)

    with pytest.raises(KeyError) as excinfo:
        runtime._to_env_action(np.zeros(7, dtype=np.float32))

    message = str(excinfo.value)
    assert missing in message
    assert "clipping" in message, message
    assert "mean_std" not in message, (
        "the message must not blame the normalization mode for a stat the clip requires: "
        + message
    )


def test_denormalize_error_still_names_the_normalization_mode():
    """逆正規化側は「どのモードが要求しているか」を言う（要求元が違う）。"""
    harness = _StubbedRuntime(norm_mode="q01_q99")
    runtime = harness.runtime
    runtime.action_stats = {
        key: value for key, value in runtime.action_stats.items() if key != "q01"
    }
    with pytest.raises(KeyError) as excinfo:
        runtime._denormalize(np.zeros((1, ACTION_DIM), dtype=np.float32))
    message = str(excinfo.value)
    assert "q01" in message
    assert "denormalization" in message and "q01_q99" in message
    assert "clipping" not in message
