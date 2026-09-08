"""RenderDownsampleFn の単体テスト（計画 P1-5 / §7.1 / SC-1）。

対応するチェックリスト項目:

- [x] ``box == interpolate(antialias=False) == INTER_AREA`` の同値性（atol=1e-6）
- [x] ``nearest`` / ``triangle`` / ``cubic`` が box と有意差あり（MSE > 1e-4）
- [x] カーネルがサンプル単位で 1 回だけ引かれる（``[T=5, C, H, W]`` / 2 カメラ）
- [x] ``p_nearest`` の実測頻度が 0.10 +-5%
- [x] 入力が 128 のときは恒等 / ``enabled=False`` は恒等
- [x] ``resize_with_pad`` は正方形入力では素の bilinear（F15）
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import requires_cv2, requires_lerobot

pytestmark = requires_lerobot


@pytest.fixture()
def render_fn():
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    return RenderDownsampleFn


def _synthetic(size: int = 256, channels: int = 3, seed: int = 0) -> torch.Tensor:
    """高周波成分のある合成画像 ``[C, H, W]``（float [0, 1]）。

    勾配 + Nyquist 近傍の縞 + ブロブ + 粒状ノイズ。カーネルの差はほぼ全部
    高周波側に出るので、ノイズ項（テクスチャ相当）を十分に入れる。実測の
    box との MSE は triangle 5.4e-4 / cubic 2.0e-4 / nearest 6.3e-3。
    **cubic と box の差が最も小さい**（2 倍ちょうどの縮小では antialias 付き
    bicubic は box 平均にかなり近い）。
    """
    generator = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, size), torch.linspace(0.0, 1.0, size), indexing="ij"
    )
    base = 0.3 * yy + 0.2 * xx
    stripes = 0.25 * torch.sin(2.0 * torch.pi * 24.0 * xx)  # Nyquist 近傍
    blob = 0.35 * torch.exp(-(((yy - 0.3) ** 2 + (xx - 0.7) ** 2) / 0.01))
    noise = 0.25 * torch.rand((size, size), generator=generator)
    gray = (base + stripes + blob + noise).clamp(0.0, 1.0)
    return torch.stack([gray * (0.6 + 0.2 * c) for c in range(channels)], dim=0).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# 1. box == interpolate(antialias=False) == INTER_AREA
# --------------------------------------------------------------------------- #
def test_box_equals_interpolate_antialias_false(render_fn):
    """「カーネルを振っても無意味」という前提そのものを回帰テストにする。

    2 倍ちょうどの縮小では avg_pool2d と bilinear(antialias=False) は同じ値になる。
    だからこそ box 以外のカーネルを混ぜる意味がある。
    """
    x = _synthetic(256)
    fn = render_fn(keys=["img"])

    via_avg_pool = fn._apply(x, "box", (0, 0))
    via_interpolate = F.interpolate(
        x.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False, antialias=False
    ).squeeze(0)

    assert via_avg_pool.shape == (3, 128, 128)
    torch.testing.assert_close(via_avg_pool, via_interpolate, atol=1e-6, rtol=0)


@requires_cv2
def test_box_equals_cv2_inter_area(render_fn):
    """cv2 があれば INTER_AREA とも一致することを確かめる（任意依存）。"""
    import cv2

    x = _synthetic(256)
    fn = render_fn(keys=["img"])
    via_avg_pool = fn._apply(x, "box", (0, 0)).numpy()

    hwc = x.permute(1, 2, 0).numpy()
    via_cv2 = cv2.resize(hwc, (128, 128), interpolation=cv2.INTER_AREA)
    via_cv2 = np.transpose(via_cv2, (2, 0, 1))

    np.testing.assert_allclose(via_avg_pool, via_cv2, atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# 2. 他のカーネルは box と有意に違う
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", ["triangle", "cubic", "nearest"])
def test_other_kernels_differ_from_box(render_fn, kernel):
    x = _synthetic(256)
    fn = render_fn(keys=["img"])

    box = fn._apply(x, "box", (0, 0))
    other = fn._apply(x, kernel, (1, 0))

    mse = float(((box - other) ** 2).mean())
    assert mse > 1e-4, f"kernel={kernel} is indistinguishable from box (MSE={mse:.3e})"


def test_cubic_stays_in_range(render_fn):
    """bicubic は overshoot するので clamp(0, 1) が効いていること。"""
    x = _synthetic(256)
    fn = render_fn(keys=["img"])
    out = fn._apply(x, "cubic", (0, 0))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_nearest_phase_shifts_output(render_fn):
    """nearest の位相ジッタが実際に別の画素を拾っていること。"""
    x = _synthetic(256)
    fn = render_fn(keys=["img"])
    a = fn._apply(x, "nearest", (0, 0))
    b = fn._apply(x, "nearest", (1, 1))
    assert not torch.allclose(a, b)
    torch.testing.assert_close(a, x[:, 0::2, 0::2], atol=0, rtol=0)
    torch.testing.assert_close(b, x[:, 1::2, 1::2], atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# 3. カーネルはサンプル単位で 1 回だけ引かれる
# --------------------------------------------------------------------------- #
def _nearest_phase_of(frame_in: torch.Tensor, frame_out: torch.Tensor) -> tuple[int, int]:
    """nearest 出力から位相 (oy, ox) を逆算する。一致が無ければ失敗させる。"""
    for oy in (0, 1):
        for ox in (0, 1):
            if torch.equal(frame_out, frame_in[..., oy::2, ox::2]):
                return oy, ox
    raise AssertionError("output does not match any nearest phase")


def test_single_draw_across_time_dimension(render_fn):
    """``[T=5, C, 256, 256]`` の全フレームが同一カーネル・同一位相で処理される。

    フレームごとに引くと WAN の教師動画がちらついて video loss に偽の時間変化が
    入る（計画 F8: ``image_delta_indices = [0, 12, 25, 37, 50]``）。
    """
    frames = torch.stack([_synthetic(256, seed=t) for t in range(5)], dim=0)
    fn = render_fn(keys=["img"], p_box=0.0, p_triangle=0.0, p_cubic=0.0, p_nearest=1.0)

    out = fn({"img": frames.clone()})["img"]
    assert out.shape == (5, 3, 128, 128)

    phases = {_nearest_phase_of(frames[t], out[t]) for t in range(5)}
    assert len(phases) == 1, f"phase varies across the time dimension: {phases}"


class _ScriptedRandom:
    """``_draw`` が引く値を台本どおりに返す RNG（決定的に分岐を突く）。

    ``_draw`` は ``random()`` を 1 回、nearest のときだけ ``randint(0, 1)`` を 2 回
    引く。台本を使い切ったら例外にして、「引かれた回数」自体も検査対象にする。
    """

    def __init__(self, randoms: list[float], randints: list[int]) -> None:
        self.randoms = list(randoms)
        self.randints = list(randints)

    def random(self) -> float:
        if not self.randoms:
            raise AssertionError("_draw() consumed more random() values than scripted")
        return self.randoms.pop(0)

    def randint(self, a: int, b: int) -> int:
        if not self.randints:
            raise AssertionError("_draw() consumed more randint() values than scripted")
        return self.randints.pop(0)


def _nearest_only(render_fn, **kwargs):
    """常に nearest を引く設定（位相だけで抽選回数を観測できる）。"""
    return render_fn(p_box=0.0, p_triangle=0.0, p_cubic=0.0, p_nearest=1.0, **kwargs)


def _scripted(fn, randoms: list[float], randints: list[int]) -> _ScriptedRandom:
    rng = _ScriptedRandom(randoms, randints)
    fn._rng = rng
    fn._rng_worker = 0  # _worker_rng() がこの台本 RNG をそのまま返すようにする
    return rng


def test_single_draw_across_views(render_fn):
    """2 カメラが同一カーネル・同一位相になる（same_kernel_across_views=True）。

    ``p_nearest=1.0`` の 1 サンプルだけだと、独立に 2 回引いても偶然一致する確率が
    1/4 ある（位相 4 通り）。ここでは 2 通りで担保する:

    1. **決定的**: 台本 RNG の 2 回目の抽選が「使われないこと」を検査する。
       ``__call__`` を通すので ``same_kernel_across_views`` の分岐が実際に走る。
    2. **統計的**: 多数の worker seed で全部一致することを確認する。
    """
    agent = _synthetic(256, seed=1)
    wrist = _synthetic(256, seed=2)

    # --- 1. 決定的: 2 回目の抽選は消費されない ---
    fn = _nearest_only(render_fn, keys=["cam0", "cam1"])
    # 1 回目 -> phase (0, 0)、2 回目（引かれたら） -> phase (1, 1)
    rng = _scripted(fn, randoms=[0.5, 0.5], randints=[0, 0, 1, 1])
    out = fn({"cam0": agent.clone(), "cam1": wrist.clone()})
    assert _nearest_phase_of(agent, out["cam0"]) == (0, 0)
    assert _nearest_phase_of(wrist, out["cam1"]) == (0, 0)
    assert rng.randoms == [0.5], "a second kernel was drawn despite same_kernel_across_views=True"
    assert rng.randints == [1, 1]

    # --- 2. 統計的: どの seed でも 2 カメラが一致する ---
    for seed in range(24):
        fn = _nearest_only(render_fn, keys=["cam0", "cam1"])
        fn._rng = random.Random(seed)
        fn._rng_worker = 0
        out = fn({"cam0": agent.clone(), "cam1": wrist.clone()})
        phase0 = _nearest_phase_of(agent, out["cam0"])
        phase1 = _nearest_phase_of(wrist, out["cam1"])
        assert phase0 == phase1, f"seed={seed}: {phase0} != {phase1}"


def test_different_kernel_per_view_when_disabled(render_fn):
    """``same_kernel_across_views=False`` なら view ごとに引き直す。

    ``_draw`` を直接叩くのではなく ``__call__`` を通す。直接叩くと
    ``same_kernel_across_views`` の分岐（``transforms_render.py`` の
    ``if index > 0 and not self.same_kernel_across_views``）が 1 度も実行されない。
    """
    agent = _synthetic(256, seed=1)
    wrist = _synthetic(256, seed=2)

    fn = _nearest_only(render_fn, keys=["cam0", "cam1"], same_kernel_across_views=False)
    rng = _scripted(fn, randoms=[0.5, 0.5], randints=[0, 0, 1, 1])

    out = fn({"cam0": agent.clone(), "cam1": wrist.clone()})
    assert _nearest_phase_of(agent, out["cam0"]) == (0, 0)
    assert _nearest_phase_of(wrist, out["cam1"]) == (1, 1)
    assert rng.randoms == [] and rng.randints == [], "the per-view draw did not happen"


def test_time_dimension_is_never_redrawn_even_per_view(render_fn):
    """view ごとに引き直しても、``[T, C, H, W]`` の T 方向では引き直さない（F8）。"""
    frames = torch.stack([_synthetic(256, seed=t) for t in range(5)], dim=0)
    fn = _nearest_only(render_fn, keys=["cam0"], same_kernel_across_views=False)
    _scripted(fn, randoms=[0.5], randints=[1, 0])

    out = fn({"cam0": frames.clone()})["cam0"]
    phases = {_nearest_phase_of(frames[t], out[t]) for t in range(5)}
    assert phases == {(1, 0)}


# --------------------------------------------------------------------------- #
# 4. カーネル頻度
# --------------------------------------------------------------------------- #
def test_p_nearest_default_is_010(render_fn):
    """p_nearest はユーザー指定で 0.10 固定（SC-1 の Evaluator 項目）。"""
    fn = render_fn()
    assert fn.p_nearest == 0.10
    assert fn.p_box == 0.45
    assert fn.p_triangle == 0.30
    assert fn.p_cubic == 0.15


def test_kernel_frequencies_match_configuration(render_fn):
    """抽選頻度が設定確率に一致する（相対誤差 5% 以内）。

    n は seed 固定でも 5% を安定して満たせる大きさにする（nearest の 1 sigma は
    n=1e5 で 0.95%、5% は約 5 sigma に相当する）。
    """
    fn = render_fn()
    rng = random.Random(20260908)
    n = 100_000

    counts = {name: 0 for name in fn.probabilities()}
    for _ in range(n):
        kernel, _phase = fn._draw(rng)
        counts[kernel] += 1

    for name, expected in fn.probabilities().items():
        observed = counts[name] / n
        relative_error = abs(observed - expected) / expected
        assert relative_error < 0.05, (
            f"kernel={name}: observed={observed:.4f} expected={expected:.4f} "
            f"relative error={relative_error:.3%}"
        )
    assert abs(counts["nearest"] / n - 0.10) / 0.10 < 0.05


def test_probabilities_must_sum_to_one(render_fn):
    with pytest.raises(ValueError, match="sum to 1.0"):
        render_fn(p_box=0.5, p_triangle=0.3, p_cubic=0.15, p_nearest=0.10)


# --------------------------------------------------------------------------- #
# 5. 恒等になるケース
# --------------------------------------------------------------------------- #
def test_identity_when_input_is_already_target(render_fn):
    x = _synthetic(128)
    fn = render_fn(keys=["img"], p_box=0.0, p_triangle=0.0, p_cubic=0.0, p_nearest=1.0)
    out = fn({"img": x.clone()})["img"]
    torch.testing.assert_close(out, x, atol=0, rtol=0)


def test_identity_when_disabled(render_fn):
    x = _synthetic(256)
    fn = render_fn(keys=["img"], enabled=False)
    out = fn({"img": x.clone()})["img"]
    torch.testing.assert_close(out, x, atol=0, rtol=0)


def test_not_hydrated_still_downsamples(render_fn):
    """未 hydrate でも黙って no-op にならないこと（F6 型のサイレント失敗の予防）。"""
    x = _synthetic(256)
    fn = render_fn()  # keys 未設定
    out = fn({"observation.images.image": x.clone(), "observation.state": torch.zeros(8)})
    assert out["observation.images.image"].shape == (3, 128, 128)
    assert out["observation.state"].shape == (8,)


def test_hydrated_keys_that_match_nothing_raise(render_fn):
    """hydrate 済みなのに 1 つも当たらないなら**例外**（黙って no-op にしない）。

    これは計画 F6 の失敗形そのもの（schema の ``image_mapping`` のキーとサンプルの
    キーの食い違い）。``auto_detect_keys`` は ``keys`` が空のときしか作動しないので、
    この経路を覆えない。
    """
    x = _synthetic(256)
    fn = render_fn(keys=["observation.images.image", "observation.images.wrist_image"])

    with pytest.raises(KeyError, match="none of the configured keys"):
        fn({"observation.images.image0": x.clone(), "observation.state": torch.zeros(8)})


def test_partial_key_match_still_downsamples(render_fn):
    """一部だけ一致した場合は、一致した分をダウンサンプルする（例外にしない）。"""
    x = _synthetic(256)
    fn = render_fn(keys=["cam0", "cam_missing"])
    out = fn({"cam0": x.clone()})
    assert out["cam0"].shape == (3, 128, 128)


def test_rejects_non_float_images(render_fn):
    fn = render_fn(keys=["img"])
    with pytest.raises(TypeError, match="float images"):
        fn({"img": torch.zeros(3, 256, 256, dtype=torch.uint8)})


# --------------------------------------------------------------------------- #
# 6. F15: resize_with_pad は正方形入力では素の bilinear
# --------------------------------------------------------------------------- #
def test_resize_with_pad_is_plain_bilinear_for_square_inputs():
    """正方形入力では padding が 1 px も入らず、素の bilinear と一致する（F15）。

    この事実があるので、**学習側の ``ResizeImagesWithPadFn(224, 224)`` は変更不要**で、
    やるべきことは「推論側で同じ関数を通す」ことだけになる（F6 対策）。
    """
    from lerobot.transforms.utils import resize_with_pad

    for size in (128, 256):
        x = _synthetic(size)
        padded = resize_with_pad(x, 224, 224)
        plain = F.interpolate(
            x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0).clamp(0.0, 1.0)

        assert padded.shape == (3, 224, 224)
        torch.testing.assert_close(padded, plain, atol=0, rtol=0)

        # padding が入っていれば必ず縁に 0 の帯ができる。1 px も無いことを確認する。
        assert float(padded[:, 0, :].abs().max()) > 0.0
        assert float(padded[:, -1, :].abs().max()) > 0.0
        assert float(padded[:, :, 0].abs().max()) > 0.0
        assert float(padded[:, :, -1].abs().max()) > 0.0


def test_render_downsample_does_not_use_resize_with_pad():
    """256 -> 128 に resize_with_pad を使っていないこと（計画 F15 の禁止事項）。"""
    import inspect

    from lerobot_policy_parc import transforms_render

    source = inspect.getsource(transforms_render.RenderDownsampleFn)
    assert "resize_with_pad" not in source
