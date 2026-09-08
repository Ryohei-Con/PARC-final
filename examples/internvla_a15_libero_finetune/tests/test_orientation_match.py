"""``orientation_match.py`` の合成データ smoke（計画 P1-10 / §7.1）。

実行そのものはクラウド（P2-17）だが、「既知の変換を掛けた画像を食わせて正解が
返ること」は手元で確認できる。ここが通らないと、向きの判定を道具に頼れない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from orientation_match import (  # noqa: E402
    TRANSFORMS,
    correlation,
    match_orientation,
    profiles,
    score_transform,
    to_gray,
)


def _structured_scene(size: int = 128) -> np.ndarray:
    """上下も左右も非対称な合成シーン（HWC uint8）。

    ロボットのシーンを模して「明るいテーブル面が下側」「明るい物体が右上」という
    ような非対称性を入れる。対称な画像だと 4 変換が区別できず、テストが空振りする。
    """
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    table = 0.55 * (yy > 0.65)  # 下側にテーブル面
    gradient = 0.25 * xx  # 左右の勾配
    blob = 0.7 * np.exp(-(((yy - 0.22) ** 2 + (xx - 0.78) ** 2) / 0.004))  # 右上に物体
    gray = np.clip(0.1 + table + gradient + blob, 0.0, 1.0)
    rgb = np.stack([gray, gray * 0.85, gray * 0.7], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


@pytest.fixture()
def scene() -> np.ndarray:
    return _structured_scene()


# --------------------------------------------------------------------------- #
# 4 変換の識別
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("truth", sorted(TRANSFORMS))
def test_recovers_known_transform(scene, truth):
    """``env`` 画像に ``truth`` の逆変換が掛かっている状況を作り、``truth`` を当てる。

    ``TRANSFORMS`` はいずれも自己逆変換（involution）なので、
    ``env = TRANSFORMS[truth](dataset)`` としておけば
    ``TRANSFORMS[truth](env) == dataset`` になり、正解は ``truth`` である。
    """
    dataset_image = scene
    env_image = TRANSFORMS[truth](dataset_image)

    result = match_orientation(dataset_image, env_image)
    assert result["best"] == truth, result["scores"]
    assert result["margin"] > 0.0


@pytest.mark.parametrize("truth", sorted(TRANSFORMS))
def test_recovers_known_transform_across_resolutions(scene, truth):
    """dataset 256 / env 128 のように解像度が違っても当てられること。

    プロファイルを線形補間して長さを揃えるので cv2 も PIL も要らない。
    """
    dataset_image = _structured_scene(256)
    env_image = TRANSFORMS[truth](_structured_scene(128))

    result = match_orientation(dataset_image, env_image)
    assert result["best"] == truth, result["scores"]


def test_correct_transform_scores_close_to_one(scene):
    assert score_transform(scene, scene, "none") == pytest.approx(1.0, abs=1e-6)


def test_transforms_are_involutions(scene):
    for mode, fn in TRANSFORMS.items():
        np.testing.assert_array_equal(fn(fn(scene)), scene, err_msg=mode)


def test_unknown_transform_is_rejected(scene):
    with pytest.raises(ValueError, match="unknown transform"):
        score_transform(scene, scene, "rot90")


# --------------------------------------------------------------------------- #
# 部品
# --------------------------------------------------------------------------- #
def test_to_gray_normalizes_uint8(scene):
    gray = to_gray(scene)
    assert gray.dtype == np.float32
    assert gray.shape == scene.shape[:2]
    assert 0.0 <= float(gray.min()) and float(gray.max()) <= 1.0


def test_profiles_shapes(scene):
    rows, cols = profiles(scene)
    assert rows.shape == (scene.shape[0],)
    assert cols.shape == (scene.shape[1],)


def test_correlation_edge_cases():
    assert correlation(np.ones(10), np.ones(10)) == 0.0  # 分散 0 -> 0 を返す
    x = np.linspace(0.0, 1.0, 32)
    assert correlation(x, x) == pytest.approx(1.0, abs=1e-9)
    assert correlation(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_symmetric_image_produces_a_small_margin():
    """完全対称な画像では判定が曖昧になり、margin が小さくなること。

    この場合 CLI は「目視で決めろ」と警告する。テストはその前提（margin が
    実際に小さくなる）を固定する。
    """
    size = 64
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    radial = np.clip(1.0 - np.sqrt(yy**2 + xx**2), 0.0, 1.0)
    image = (np.stack([radial] * 3, axis=-1) * 255.0).astype(np.uint8)

    result = match_orientation(image, image)
    assert result["margin"] < 0.05
