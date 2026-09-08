"""RTC guidance と temporal ensembling の単体テスト（計画 P1-7 / §7.1 / SC-2）。

**このテストは lerobot もモデルも要らない。** ``chunk_blending`` は numpy と torch
だけに依存し、``euler_integrate`` は ``denoise_fn`` を引数で受け取るので、スタブ関数で
CPU 単体テストが完結する。

- [x] ``guidance=None`` と ``weights=0`` がビット一致（ベースライン非破壊）
- [x] ``weights[j]=w`` を上げると ``x_final[j]`` が ``target[j]`` に近づく
- [x] ``dim_mask[6]=False`` の次元が ``guidance=None`` とビット一致（グリッパ非干渉）
- [x] ``num_steps`` を振っても NaN/Inf が出ない
- [x] ``t_safe`` のクランプは**0 除算の防止ではない**（``time`` は 0 に到達しない）。
      「クランプの有無で結果が丸め誤差以上には変わらない」ことを、クランプ無しの
      参照実装との比較で明示的に固定する
- [x] ``build_guidance`` の境界条件と重みの単調性
- [x] ``TemporalEnsembler`` の 4 性質
- [x] ``w_max < 1.0``（ハードマスク不採用）が既定値とテストの両方で担保されている
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from chunk_blending import (  # noqa: E402
    DEFAULT_W_MAX,
    GuidanceSpec,
    TemporalEnsembler,
    build_guidance,
    euler_integrate,
)

CHUNK = 50
DIM = 32
REPLAN = 16


# --------------------------------------------------------------------------- #
# スタブ denoise_fn
# --------------------------------------------------------------------------- #
def make_linear_denoise_fn(anchor: torch.Tensor):
    """次元ごとに独立な速度場 ``v = (anchor - x) / max(t, eps)``。

    flow-matching の「x を anchor へ運ぶ」理想速度そのもの。**次元間で結合しない**ので、
    「guidance を掛けていない次元がビット一致する」という主張が意味を持つ。
    実モデルは次元間で結合するため、その主張は「guidance が直接触らない」という意味に
    限定される（結合経由の間接的な影響までは保証しない）。
    """

    def denoise_fn(x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.view(-1, 1, 1).clamp(min=1e-3)
        return (anchor - x_t) / t

    return denoise_fn


def make_constant_denoise_fn(value: float = 0.25):
    def denoise_fn(x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x_t, value)

    return denoise_fn


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(20260908)


@pytest.fixture()
def prev_chunk(rng) -> np.ndarray:
    chunk = rng.normal(size=(CHUNK, DIM)).astype(np.float32)
    chunk[:, 7:] = 0.0  # padding 次元
    return chunk


# --------------------------------------------------------------------------- #
# build_guidance
# --------------------------------------------------------------------------- #
def test_build_guidance_returns_none_without_previous_chunk():
    assert build_guidance(None, REPLAN, CHUNK, DIM) is None


def test_build_guidance_returns_none_when_replan_equals_chunk(prev_chunk):
    assert build_guidance(prev_chunk, CHUNK, CHUNK, DIM) is None


def test_build_guidance_overlap_length(prev_chunk):
    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)
    assert guidance is not None
    assert guidance.overlap == CHUNK - REPLAN


def test_build_guidance_target_is_time_aligned(prev_chunk):
    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)
    overlap = CHUNK - REPLAN
    np.testing.assert_array_equal(guidance.target[:overlap], prev_chunk[REPLAN:])
    np.testing.assert_array_equal(guidance.target[overlap:], 0.0)


def test_build_guidance_weights_are_monotone_and_bounded(prev_chunk):
    for schedule in ("linear", "cosine"):
        guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM, schedule=schedule)
        weights = guidance.weights
        assert weights.shape == (CHUNK,)
        assert float(weights.min()) >= 0.0
        assert float(weights.max()) < 1.0
        assert np.all(np.diff(weights) <= 1e-7), f"{schedule} weights are not monotone"
        assert float(weights[0]) == pytest.approx(DEFAULT_W_MAX, abs=1e-6)
        assert float(weights[CHUNK - REPLAN]) == 0.0


def test_build_guidance_dim_mask_excludes_gripper_and_padding(prev_chunk):
    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)
    mask = guidance.dim_mask
    assert mask.shape == (DIM,)
    assert bool(mask[:6].all())
    assert not bool(mask[6]), "gripper (index 6) must be excluded from guidance"
    assert not bool(mask[7:].any()), "padding dims (7..) must be excluded from guidance"


def test_default_w_max_is_soft():
    """ハードマスクは採らない（計画 D2 / SC-2）。"""
    assert DEFAULT_W_MAX == 0.8
    assert 0.0 < DEFAULT_W_MAX < 1.0


def test_build_guidance_rejects_hard_mask(prev_chunk):
    with pytest.raises(ValueError, match="hard mask"):
        build_guidance(prev_chunk, REPLAN, CHUNK, DIM, w_max=1.0)
    with pytest.raises(ValueError, match="soft mask only"):
        build_guidance(prev_chunk, REPLAN, CHUNK, DIM, w_max=1.5)


def test_guidance_spec_rejects_weights_at_one():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        GuidanceSpec(
            weights=np.ones(CHUNK, dtype=np.float32),
            target=np.zeros((CHUNK, DIM), dtype=np.float32),
            dim_mask=np.ones(DIM, dtype=bool),
        )


def test_build_guidance_rejects_bad_shapes(prev_chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        build_guidance(prev_chunk[:10], REPLAN, CHUNK, DIM)
    with pytest.raises(ValueError, match="action_dim"):
        build_guidance(prev_chunk[:, :8], REPLAN, CHUNK, DIM)
    with pytest.raises(ValueError, match="replan_steps"):
        build_guidance(prev_chunk, 0, CHUNK, DIM)


# --------------------------------------------------------------------------- #
# euler_integrate — ベースライン非破壊
# --------------------------------------------------------------------------- #
def test_zero_weights_are_bit_identical_to_no_guidance(prev_chunk):
    """``weights=0`` の guidance は ``guidance=None`` と**ビット一致**する。

    ``torch.where`` が ``W == 0`` の位置で ``v_model`` をそのまま選ぶため。
    ベースラインを壊さないことの保証。
    """
    torch.manual_seed(0)
    x0 = torch.randn(1, CHUNK, DIM)
    anchor = torch.randn(1, CHUNK, DIM)
    denoise_fn = make_linear_denoise_fn(anchor)

    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM, w_max=0.0)
    assert guidance is not None
    assert float(guidance.weights.max()) == 0.0

    without = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=None)
    with_zero = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=guidance)

    assert torch.equal(without, with_zero)


def test_non_overlapping_tail_is_bit_identical(prev_chunk):
    """重なり区間の外（j >= L）は guidance が触らない。"""
    torch.manual_seed(1)
    x0 = torch.randn(1, CHUNK, DIM)
    denoise_fn = make_linear_denoise_fn(torch.randn(1, CHUNK, DIM))
    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)

    baseline = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=None)
    guided = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=guidance)

    overlap = CHUNK - REPLAN
    assert torch.equal(baseline[:, overlap:, :], guided[:, overlap:, :])


def test_gripper_dimension_is_bit_identical(prev_chunk):
    """``dim_mask[6]=False`` の次元は ``guidance=None`` と**ビット一致**する。

    スタブ ``denoise_fn`` は次元ごとに独立なので、この主張は「guidance が index 6 に
    一切触らない」ことを直接示す。
    """
    torch.manual_seed(2)
    x0 = torch.randn(1, CHUNK, DIM)
    denoise_fn = make_linear_denoise_fn(torch.randn(1, CHUNK, DIM))
    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)

    baseline = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=None)
    guided = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=guidance)

    assert torch.equal(baseline[:, :, 6], guided[:, :, 6])
    assert torch.equal(baseline[:, :, 7:], guided[:, :, 7:])
    # 逆に、guidance の対象次元は必ず変化していること（テストが空振りしていない証明）
    assert not torch.equal(baseline[:, :5, :6], guided[:, :5, :6])


# --------------------------------------------------------------------------- #
# euler_integrate — 収束
# --------------------------------------------------------------------------- #
def test_weights_one_would_pull_to_target(prev_chunk):
    """``weights -> 1`` に近づけると ``x_final[j]`` が ``target[j]`` に収束する。

    ``GuidanceSpec`` は ``w = 1`` そのものを禁止している（ハードマスク不採用）ので、
    上限に十分近い 0.999 で確認する。
    """
    torch.manual_seed(3)
    x0 = torch.randn(1, CHUNK, DIM)
    denoise_fn = make_constant_denoise_fn(0.5)

    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)
    overlap = guidance.overlap
    guidance.weights[:overlap] = 0.999

    result = euler_integrate(denoise_fn, x0.clone(), num_steps=50, guidance=guidance)
    target = torch.as_tensor(guidance.target, dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(guidance.dim_mask, dtype=torch.bool)

    error = (result[0, :overlap, mask] - target[0, :overlap, mask]).abs().max()
    assert float(error) < 1e-3, f"did not converge to the target: max|err|={float(error):.4e}"


def test_soft_weight_moves_toward_but_not_onto_target(prev_chunk):
    """既定のソフト重み 0.8 では target に近づくが一致はしない。"""
    torch.manual_seed(4)
    x0 = torch.randn(1, CHUNK, DIM)
    denoise_fn = make_constant_denoise_fn(0.5)

    guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM, w_max=DEFAULT_W_MAX)
    baseline = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=None)
    guided = euler_integrate(denoise_fn, x0.clone(), num_steps=10, guidance=guidance)

    target = torch.as_tensor(guidance.target, dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(guidance.dim_mask, dtype=torch.bool)

    head = slice(0, 5)  # 重みが最大に近い先頭
    baseline_error = (baseline[0, head][:, mask] - target[0, head][:, mask]).abs().mean()
    guided_error = (guided[0, head][:, mask] - target[0, head][:, mask]).abs().mean()

    assert float(guided_error) < float(baseline_error)
    assert float(guided_error) > 1e-3, "soft guidance must not collapse onto the target"


def test_guided_integration_stays_finite_for_any_num_steps(prev_chunk):
    """``num_steps`` を振っても出力が有限であること。

    .. note:: **これは ``t_safe`` のクランプを検査していない。** 下のループには
       ``num_steps=10``（クランプが実際に binding する N。
       :func:`test_the_time_clamp_binds_at_the_default_num_steps` を参照）が
       含まれるが、binding してもずれは丸め誤差の範囲なので、クランプを外しても
       このテストは通る。ここが見ているのは「guidance を掛けた積分が発散しない」
       ことだけである。クランプの有無による差は
       :func:`test_the_time_clamp_never_changes_the_result_beyond_float_rounding`
       が見る。
    """
    torch.manual_seed(5)
    x0 = torch.randn(1, CHUNK, DIM)
    denoise_fn = make_constant_denoise_fn(1.0)

    for num_steps in (1, 2, 5, 10, 50):
        guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)
        result = euler_integrate(denoise_fn, x0.clone(), num_steps=num_steps, guidance=guidance)
        assert torch.isfinite(result).all(), f"non-finite output at num_steps={num_steps}"


def _euler_without_time_clamp(denoise_fn, x0, num_steps, guidance):
    """``euler_integrate`` から ``t_safe`` のクランプ**だけ**を外した参照実装。

    それ以外（ループ条件・積分・where の合成）は本体と同じ式にしてある。

    .. note:: 本体とは**ビット一致しない**。``num_steps`` が 10 / 20 / 25 のときは
       float32 の累積誤差で ``time`` が ``|dt|`` をわずかに下回り、クランプが実際に
       binding するため（実測 N=10 で max abs diff ~4.8e-07）。だから
       :func:`test_the_time_clamp_never_changes_the_result_beyond_float_rounding`
       は ``assert_close`` であって ``assert_equal`` ではない。
    """
    device = x0.device
    batch = x0.shape[0]
    weights_t = torch.as_tensor(guidance.weights, dtype=torch.float32, device=device).view(1, -1, 1)
    target_t = torch.as_tensor(guidance.target, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(guidance.dim_mask, dtype=torch.bool, device=device).view(1, 1, -1)
    active_t = mask_t & (weights_t > 0.0)

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)

    x_t = x0
    while time >= -dt / 2:
        v_model = denoise_fn(x_t, time.expand(batch))
        v_guide = (x_t - target_t.to(v_model.dtype)) / time.to(v_model.dtype)  # クランプ無し
        w = weights_t.to(v_model.dtype)
        v_t = torch.where(active_t, (1.0 - w) * v_model + w * v_guide, v_model)
        x_t = x_t + dt.to(x_t.dtype) * v_t
        time = time + dt
    return x_t


def test_the_time_clamp_never_changes_the_result_beyond_float_rounding(prev_chunk):
    """``t_safe`` のクランプが load-bearing でないことを固定する。

    **このテストはクランプを消しても通る。それが主張の内容である。**
    ``chunk_blending`` のコメントは「0 除算を防ぐ」と読める書き方をしていたが、
    実際には ``time`` は 0 に到達しない（最小値は ``1/N``）ので、クランプが無くても
    ``(x_t - target) / time`` は発散しない。クランプは時刻スケジュールを変えたときの
    **防御**として置いてある。

    クランプは float32 の累積誤差により ``num_steps`` が 10 / 20 / 25 のときは実際に
    binding する（:func:`test_the_time_clamp_binds_at_the_default_num_steps`）。
    ここで検証するのは「binding しても結果は丸め誤差以上には変わらない」ことであり、
    そのため許容誤差付きの ``assert_close`` を使う。もし将来スケジュールを変えて
    クランプの効果が丸め誤差を超えたら、このテストが落ちて「防御コードが
    load-bearing になった」ことに気づける。
    """
    torch.manual_seed(11)
    x0 = torch.randn(1, CHUNK, DIM)
    denoise_fn = make_constant_denoise_fn(1.0)

    for num_steps in (1, 2, 5, 10, 50):
        guidance = build_guidance(prev_chunk, REPLAN, CHUNK, DIM)
        clamped = euler_integrate(denoise_fn, x0.clone(), num_steps=num_steps, guidance=guidance)
        unclamped = _euler_without_time_clamp(denoise_fn, x0.clone(), num_steps, guidance)
        assert torch.isfinite(unclamped).all(), (
            f"the unclamped reference diverged at num_steps={num_steps}; "
            "the clamp would then be load-bearing and this test must be revisited"
        )
        torch.testing.assert_close(clamped, unclamped, rtol=1e-4, atol=1e-5)


def test_timesteps_never_reach_zero_so_the_clamp_is_not_a_zero_division_guard(prev_chunk):
    """``denoise_fn`` に渡る ``time`` の最小値が ``1/N`` で、0 にならないこと。

    これがクランプを「0 除算の防止」と呼べない理由そのもの。``time`` は float32 で
    ``time += dt`` と累積するので最終値は ``1/N`` から数 ULP ずれる（``|dt|`` を
    わずかに下回ることもあり、既定の ``num_steps=10`` はそれに当たる）。ずれの向きは
    プラットフォーム依存なので、ここでは相対誤差だけを見る。
    """
    seen: list[float] = []

    def recording_denoise_fn(x_t, timestep):
        seen.append(float(timestep[0]))
        return torch.zeros_like(x_t)

    x0 = torch.zeros(1, CHUNK, DIM)
    for num_steps in (1, 2, 5, 10, 25, 50):
        seen.clear()
        euler_integrate(recording_denoise_fn, x0, num_steps=num_steps, guidance=None)
        assert len(seen) == num_steps
        assert min(seen) == pytest.approx(1.0 / num_steps, rel=1e-4)
        assert min(seen) > 0.0, "time must never reach 0; the clamp is not what prevents that"


def test_the_time_clamp_binds_at_the_default_num_steps():
    """既定の ``num_inference_steps=10`` でクランプが実際に binding すること。

    ``chunk_blending`` の docstring と、上の 2 本のテストの主張が依拠している実測
    事実そのものを固定する。厳密な実数演算なら ``time`` の最小値は ``1/N`` ちょうど
    でクランプは恒等だが、``time`` は float32 で ``time += dt`` と累積するため最終値が
    数 ULP ずれ、N が 10 / 20 / 25 のときは ``|dt|`` をわずかに下回る。float32 の
    加算は IEEE 754 で一意に決まるのでこの結果は再現する。

    ここが落ちたら、それは「クランプは binding しない」側に事実が変わったという
    ことなので、``chunk_blending.py`` の note と上の 2 本の docstring を更新すること。
    """
    def binds(num_steps: int) -> bool:
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32)
        time = torch.tensor(1.0, dtype=torch.float32)
        last = time
        while time >= -dt / 2:
            last = time
            time = time + dt
        # euler_integrate は t_safe = clamp(time, min=|dt|) を使う。
        # 最終 time が |dt| を下回っていれば、そこでクランプが効いている。
        return bool(last < dt.abs())

    assert binds(10), "既定の num_inference_steps=10 で binding しなくなった"
    assert {n for n in (2, 5, 10, 20, 25, 50, 100) if binds(n)} == {10, 20, 25}

    # binding していても、ずれは相対 1e-5 未満（数値的な意味は無い）。
    dt10 = torch.tensor(-1.0 / 10, dtype=torch.float32)
    time = torch.tensor(1.0, dtype=torch.float32)
    last = time
    while time >= -dt10 / 2:
        last = time
        time = time + dt10
    assert abs(float(last) - 0.1) / 0.1 < 1e-5


# --------------------------------------------------------------------------- #
# TemporalEnsembler
# --------------------------------------------------------------------------- #
def test_ensembler_single_chunk_is_passthrough(rng):
    chunk = rng.normal(size=(CHUNK, DIM)).astype(np.float32)
    ensembler = TemporalEnsembler(CHUNK, DIM)
    ensembler.push(chunk)

    for step in range(5):
        np.testing.assert_allclose(ensembler.pop(), chunk[step], rtol=0, atol=1e-6)


def test_ensembler_preserves_constant_chunks():
    """同じ（時間的に一定な）チャンクを繰り返し push しても値が変わらない。

    ``pop()`` は重み和 1 の凸結合なので、候補が全部同じ値なら結果もその値になる。
    """
    value = np.arange(DIM, dtype=np.float32)
    chunk = np.tile(value, (CHUNK, 1))

    ensembler = TemporalEnsembler(CHUNK, DIM)
    for _ in range(10):
        ensembler.push(chunk)
        np.testing.assert_allclose(ensembler.pop(), value, rtol=0, atol=1e-5)


def test_ensembler_weights_sum_to_one_and_decay(rng):
    chunk = rng.normal(size=(CHUNK, DIM)).astype(np.float32)
    ensembler = TemporalEnsembler(CHUNK, DIM, m=0.1)
    for _ in range(4):
        ensembler.push(chunk)

    weights = ensembler.weights()
    assert weights.shape == (4,)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(weights) < 0.0), "older chunks must get smaller weights"


def test_ensembler_newest_dims_are_not_averaged(rng):
    """``newest_dims``（グリッパ）は平均せず最新チャンクの値を採る。"""
    old = np.zeros((CHUNK, DIM), dtype=np.float32)
    new = np.ones((CHUNK, DIM), dtype=np.float32)

    ensembler = TemporalEnsembler(CHUNK, DIM, m=0.1, newest_dims=(6,))
    ensembler.push(old)
    ensembler.push(new)
    action = ensembler.pop()

    assert action[6] == pytest.approx(1.0), "gripper must follow the newest chunk"
    assert 0.0 < action[0] < 1.0, "other dims must be averaged"


def test_ensembler_reset_clears_state(rng):
    chunk = rng.normal(size=(CHUNK, DIM)).astype(np.float32)
    ensembler = TemporalEnsembler(CHUNK, DIM)
    ensembler.push(chunk)
    ensembler.pop()
    assert len(ensembler) == 1

    ensembler.reset()
    assert len(ensembler) == 0
    with pytest.raises(RuntimeError, match="no chunks"):
        ensembler.pop()


def test_ensembler_drops_exhausted_chunks(rng):
    chunk = rng.normal(size=(4, DIM)).astype(np.float32)
    ensembler = TemporalEnsembler(4, DIM)
    ensembler.push(chunk)
    for _ in range(4):
        ensembler.pop()
    assert len(ensembler) == 0


def test_ensembler_rejects_bad_chunks():
    ensembler = TemporalEnsembler(CHUNK, DIM)
    with pytest.raises(ValueError, match="chunk must be"):
        ensembler.push(np.zeros((CHUNK, 8), dtype=np.float32))
    bad = np.zeros((CHUNK, DIM), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ensembler.push(bad)


# --------------------------------------------------------------------------- #
# runtime_config.json の 1 キーで切り替わること（SC-2 の Evaluator 項目）
# --------------------------------------------------------------------------- #
def test_runtime_config_switches_between_rtc_and_ensemble():
    import json

    from internvla_runtime import resolve_chunking

    config_path = Path(__file__).resolve().parent.parent / "inference" / "runtime_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["chunking"]["mode"] in ("none", "rtc", "ensemble")
    assert config["chunking"]["w_max"] < 1.0
    assert config["chunking"]["exclude_dims"] == [6]

    for mode in ("none", "rtc", "ensemble"):
        config["chunking"]["mode"] = mode
        assert resolve_chunking(config)["mode"] == mode

    config["chunking"]["mode"] = "nonsense"
    with pytest.raises(ValueError, match="chunking.mode"):
        resolve_chunking(config)

    config["chunking"]["mode"] = "rtc"
    config["chunking"]["w_max"] = 1.0
    with pytest.raises(ValueError, match="hard masking"):
        resolve_chunking(config)
