"""``internvla_runtime`` の純粋部分の単体テスト（計画 P1-8 / §7.1 / SC-3）。

- [x] ``quat2axisangle`` が ``scipy.spatial.transform.Rotation.as_rotvec`` と一致
      （``w < 0`` の符号反転、``w ~= +-1`` の縮退を含む）
- [x] ``orient_image`` の 4 変換と、``"TBD"`` / 未知値の拒否
- [x] ``binarize_gripper`` が ``runtime_config.json`` 由来のしきい値で動く
- [x] ``load_runtime_config`` / ``apply_config_env``（``setdefault`` であること）
- [x] ``resolve_image_orientation`` / ``resolve_gripper`` が ``"TBD"`` で例外を投げる
- [x] ``extract_state`` が 8 次元（eef_pos + axisangle + gripper_qpos）を作る
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from internvla_runtime import (  # noqa: E402
    ORIENTATIONS,
    apply_config_env,
    binarize_gripper,
    extract_state,
    load_runtime_config,
    orient_image,
    quat2axisangle,
    resolve_chunking,
    resolve_gripper,
    resolve_image_orientation,
)

RUNTIME_CONFIG = Path(__file__).resolve().parent.parent / "inference" / "runtime_config.json"

scipy = pytest.importorskip(
    "scipy.spatial.transform",
    reason="scipy is needed to cross-check quat2axisangle against Rotation.as_rotvec",
)
Rotation = scipy.Rotation


# --------------------------------------------------------------------------- #
# quat2axisangle vs scipy
# --------------------------------------------------------------------------- #
def _normalized(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    return (quat / np.linalg.norm(quat)).astype(np.float32)


#: robosuite / LIBERO 規約は (x, y, z, w)。scipy の from_quat も既定で同じ順序。
EXPLICIT_QUATS = [
    ("identity_w_plus_1", [0.0, 0.0, 0.0, 1.0]),
    ("identity_w_minus_1", [0.0, 0.0, 0.0, -1.0]),
    ("x_90", [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]),
    ("y_90", [0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]),
    ("z_90", [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
    ("z_minus_90", [0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4)]),
    # --- w < 0（同じ回転の「裏側」表現。符号反転して短い方を採るかが問題になる）---
    ("z_90_negated", [0.0, 0.0, -np.sin(np.pi / 4), -np.cos(np.pi / 4)]),
    ("x_170", [np.sin(np.deg2rad(85)), 0.0, 0.0, np.cos(np.deg2rad(85))]),
    ("x_190_w_negative", [np.sin(np.deg2rad(95)), 0.0, 0.0, np.cos(np.deg2rad(95))]),
    ("mixed_w_negative", [0.3, -0.4, 0.5, -0.7071]),
    # --- w ~= +-1 の縮退（den = sqrt(1 - w^2) -> 0）---
    ("near_identity", [1e-7, -1e-7, 1e-7, 1.0]),
    ("near_identity_negative", [1e-7, -1e-7, 1e-7, -1.0]),
]


@pytest.mark.parametrize("name,quat", EXPLICIT_QUATS, ids=[n for n, _ in EXPLICIT_QUATS])
def test_quat2axisangle_represents_the_same_rotation_as_scipy(name, quat):
    """``quat2axisangle`` の結果が scipy と**同じ回転**を表すこと（全ケース）。

    回転ベクトルの表現は一意ではない（``theta`` と ``theta - 2*pi`` は同じ回転）ので、
    回転行列に直して比べるのが正しい同値判定である。
    """
    q = _normalized(quat)

    ours = quat2axisangle(q)
    assert ours.shape == (3,)
    assert ours.dtype == np.float32

    ours_matrix = Rotation.from_rotvec(ours.astype(np.float64)).as_matrix()
    scipy_matrix = Rotation.from_quat(q.astype(np.float64)).as_matrix()
    np.testing.assert_allclose(ours_matrix, scipy_matrix, atol=2e-5, rtol=0, err_msg=f"case={name}")


@pytest.mark.parametrize("name,quat", EXPLICIT_QUATS, ids=[n for n, _ in EXPLICIT_QUATS])
def test_quat2axisangle_matches_as_rotvec_when_w_is_non_negative(name, quat):
    """``w >= 0`` のときは ``as_rotvec()`` と**値そのもの**が一致する。

    上流 ``model2libero_interface.py:11-32`` は ``2 * acos(w) / sqrt(1 - w^2)`` を
    ``q[:3]`` に掛ける。``w >= 0`` では回転角が ``[0, pi]`` に収まるので、scipy が返す
    正準形と同じになる。``w < 0`` の挙動は
    ``test_quat2axisangle_does_not_canonicalize_negative_w`` を参照。
    """
    q = _normalized(quat)
    if float(q[3]) < 0.0:
        pytest.skip("w < 0 is covered by test_quat2axisangle_does_not_canonicalize_negative_w")

    ours = quat2axisangle(q)
    theirs = Rotation.from_quat(q.astype(np.float64)).as_rotvec()
    np.testing.assert_allclose(ours, theirs, atol=2e-5, rtol=0, err_msg=f"case={name}")


def test_quat2axisangle_does_not_canonicalize_negative_w():
    """``w < 0`` では回転角が ``pi`` を超え、scipy の正準形とは ``2*pi`` ずれる。

    scipy の ``as_rotvec()`` は先にクォータニオンの符号を揃えて角度を ``[0, pi]`` に
    畳むが、**上流の実装は畳まない**。学習データの ``observation.state`` は上流の
    実装で作られているので、推論側も畳んではいけない。畳むと 8 次元 state の
    3:6 成分が学習と食い違い、静かに精度だけ落ちる。
    """
    q = _normalized([0.0, 0.0, -np.sin(np.pi / 4), -np.cos(np.pi / 4)])  # w = -0.707
    ours = quat2axisangle(q)
    canonical = Rotation.from_quat(q.astype(np.float64)).as_rotvec()

    angle_ours = float(np.linalg.norm(ours))
    angle_canonical = float(np.linalg.norm(canonical))

    assert angle_ours > np.pi, f"expected an un-folded angle, got {angle_ours}"
    assert angle_canonical <= np.pi
    assert angle_ours + angle_canonical == pytest.approx(2.0 * np.pi, abs=1e-4)

    # 同じ回転であることは行列で確認できる。
    np.testing.assert_allclose(
        Rotation.from_rotvec(ours.astype(np.float64)).as_matrix(),
        Rotation.from_quat(q.astype(np.float64)).as_matrix(),
        atol=2e-5,
        rtol=0,
    )


def test_quat2axisangle_matches_scipy_on_random_quaternions():
    """ランダム 500 本。回転としての同値（行列）と、``w >= 0`` での値一致。"""
    rng = np.random.default_rng(20260908)
    checked_positive = 0
    for _ in range(500):
        q = _normalized(rng.normal(size=4))
        ours = quat2axisangle(q)

        np.testing.assert_allclose(
            Rotation.from_rotvec(ours.astype(np.float64)).as_matrix(),
            Rotation.from_quat(q.astype(np.float64)).as_matrix(),
            atol=2e-4,
            rtol=0,
            err_msg=f"quat={q}",
        )
        if float(q[3]) >= 0.0:
            checked_positive += 1
            np.testing.assert_allclose(
                ours,
                Rotation.from_quat(q.astype(np.float64)).as_rotvec(),
                atol=2e-4,
                rtol=0,
                err_msg=f"quat={q}",
            )
    assert checked_positive > 100, "the random sample did not cover enough w >= 0 cases"


def test_quat2axisangle_is_zero_at_degenerate_w():
    """``w = +-1`` の縮退では ``[0, 0, 0]`` を返す（0 除算しない）。"""
    for w in (1.0, -1.0):
        out = quat2axisangle([0.0, 0.0, 0.0, w])
        np.testing.assert_array_equal(out, np.zeros(3, dtype=np.float32))
        assert np.all(np.isfinite(out))


def test_quat2axisangle_clamps_out_of_range_w():
    """数値誤差で ``|w| > 1`` になっても acos が NaN にならないこと。"""
    for w in (1.0000001, -1.0000001):
        out = quat2axisangle([0.0, 0.0, 0.0, w])
        assert np.all(np.isfinite(out))


def test_quat2axisangle_rejects_bad_length():
    with pytest.raises(ValueError, match="length 4"):
        quat2axisangle([0.0, 0.0, 1.0])


def test_quat2axisangle_negation_flips_the_rotation_vector():
    """``q`` と ``-q`` は同じ姿勢だが、この実装では別の回転ベクトルになる。

    上流がそういう実装なので学習データもそうなっている。**勝手に短い方へ畳まない**
    ことを固定しておく（畳むと学習時の state と食い違う）。
    """
    q = _normalized([0.0, 0.0, np.sin(np.deg2rad(80)), np.cos(np.deg2rad(80))])
    positive = quat2axisangle(q)
    negative = quat2axisangle(-q)

    assert not np.allclose(positive, negative)
    # ただし表す回転は同じ。
    np.testing.assert_allclose(
        Rotation.from_rotvec(positive.astype(np.float64)).as_matrix(),
        Rotation.from_rotvec(negative.astype(np.float64)).as_matrix(),
        atol=2e-5,
        rtol=0,
    )


# --------------------------------------------------------------------------- #
# extract_state
# --------------------------------------------------------------------------- #
def test_extract_state_layout():
    obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": _normalized([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        "robot0_gripper_qpos": np.array([0.03, -0.03], dtype=np.float32),
    }
    state = extract_state(obs)

    assert state.shape == (8,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(state[3:6], quat2axisangle(obs["robot0_eef_quat"]), atol=1e-6)
    np.testing.assert_allclose(state[6:], [0.03, -0.03], atol=1e-6)


def test_extract_state_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="unexpected LIBERO state shapes"):
        extract_state(
            {
                "robot0_eef_pos": np.zeros(2, dtype=np.float32),
                "robot0_eef_quat": np.zeros(4, dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
        )


# --------------------------------------------------------------------------- #
# orient_image
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sample_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (8, 6, 3), dtype=np.uint8)


def test_orient_image_transforms(sample_image):
    np.testing.assert_array_equal(orient_image(sample_image, "none"), sample_image)
    np.testing.assert_array_equal(orient_image(sample_image, "flip_ud"), sample_image[::-1])
    np.testing.assert_array_equal(orient_image(sample_image, "flip_lr"), sample_image[:, ::-1])
    np.testing.assert_array_equal(
        orient_image(sample_image, "rot180"), sample_image[::-1, ::-1]
    )


def test_orient_image_returns_contiguous(sample_image):
    for mode in ORIENTATIONS:
        assert orient_image(sample_image, mode).flags["C_CONTIGUOUS"]


def test_orient_image_rejects_tbd(sample_image):
    """``"TBD"`` のまま推論しようとしたら**黙って none に落とさず**例外を投げる。"""
    with pytest.raises(ValueError, match="TBD"):
        orient_image(sample_image, "TBD")


def test_orient_image_rejects_unknown_mode(sample_image):
    with pytest.raises(ValueError, match="unknown image orientation"):
        orient_image(sample_image, "rot90")


def test_orient_image_rejects_non_hwc():
    with pytest.raises(ValueError, match="HWC"):
        orient_image(np.zeros((8, 8), dtype=np.uint8), "none")


# --------------------------------------------------------------------------- #
# runtime_config.json
# --------------------------------------------------------------------------- #
def test_shipped_runtime_config_is_loadable():
    config = load_runtime_config(RUNTIME_CONFIG)
    assert config["runtime"] == "standard"
    assert config["resize"]["height"] == 224
    assert config["resize"]["width"] == 224
    assert config["num_inference_steps"] == 10


def test_shipped_runtime_config_still_has_tbd_fields():
    """P2 でデータを見るまで向きとグリッパ規約は TBD のままであること。

    埋まっていたら「誰かが推測で埋めた」ということなので、ここで気付けるようにする。
    確定したらこのテストを SC-4 の完了と同時に更新する。
    """
    config = load_runtime_config(RUNTIME_CONFIG)
    assert config["image_orientation"]["agentview"] == "TBD"
    assert config["image_orientation"]["wrist"] == "TBD"
    assert config["gripper"]["dataset_convention"] == "TBD"


def test_resolve_image_orientation_rejects_tbd():
    config = load_runtime_config(RUNTIME_CONFIG)
    with pytest.raises(ValueError, match="TBD"):
        resolve_image_orientation(config)


def test_resolve_gripper_rejects_tbd():
    config = load_runtime_config(RUNTIME_CONFIG)
    with pytest.raises(ValueError, match="TBD"):
        resolve_gripper(config)


def test_resolve_image_orientation_accepts_resolved_values():
    config = load_runtime_config(RUNTIME_CONFIG)
    config["image_orientation"] = {"agentview": "rot180", "wrist": "none"}
    assert resolve_image_orientation(config) == {"agentview": "rot180", "wrist": "none"}

    config["image_orientation"]["wrist"] = "rot90"
    with pytest.raises(ValueError, match="must be one of"):
        resolve_image_orientation(config)


def test_resolve_gripper_defaults_by_convention():
    config = load_runtime_config(RUNTIME_CONFIG)

    config["gripper"]["dataset_convention"] = "zero_one"
    resolved = resolve_gripper(config)
    assert resolved["threshold"] == 0.5
    assert resolved["env_close"] == 1.0
    assert resolved["env_open"] == -1.0

    config["gripper"]["dataset_convention"] = "minus_one_one"
    assert resolve_gripper(config)["threshold"] == 0.0

    config["gripper"]["dataset_convention"] = "something_else"
    with pytest.raises(ValueError, match="dataset_convention"):
        resolve_gripper(config)


def test_binarize_gripper_follows_the_config():
    """既定規則: ``action[6] = +1 if a < 0.5 else -1``（データが ``[0, 1]`` 規約のとき）。"""
    zero_one = {
        "binarize": True,
        "threshold": 0.5,
        "below_threshold": "close",
        "env_close": 1.0,
        "env_open": -1.0,
    }
    assert binarize_gripper(0.1, zero_one) == 1.0
    assert binarize_gripper(0.9, zero_one) == -1.0

    inverted = dict(zero_one, below_threshold="open")
    assert binarize_gripper(0.1, inverted) == -1.0
    assert binarize_gripper(0.9, inverted) == 1.0

    passthrough = dict(zero_one, binarize=False)
    assert binarize_gripper(0.42, passthrough) == pytest.approx(0.42)


def test_apply_config_env_uses_setdefault(monkeypatch):
    """環境変数が既にあればそちらが優先される（移植手順書 B8 / A/B のため）。"""
    monkeypatch.delenv("IVLA_ROBOT_TYPE", raising=False)
    config = load_runtime_config(RUNTIME_CONFIG)

    applied = apply_config_env(config)
    assert applied["IVLA_ROBOT_TYPE"] == "libero_combined"

    monkeypatch.setenv("IVLA_ROBOT_TYPE", "libero_spatial")
    applied = apply_config_env(config)
    assert applied["IVLA_ROBOT_TYPE"] == "libero_spatial"


def test_apply_config_env_skips_comment_keys(monkeypatch, tmp_path):
    payload = {"env": {"_comment": "ignore me", "IVLA_TEST_KEY": "value"}}
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.delenv("IVLA_TEST_KEY", raising=False)
    applied = apply_config_env(load_runtime_config(path))
    assert applied == {"IVLA_TEST_KEY": "value"}


def test_load_runtime_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_runtime_config(tmp_path / "nope.json")


def test_resolve_chunking_defaults():
    config = load_runtime_config(RUNTIME_CONFIG)
    chunking = resolve_chunking(config)
    assert chunking["mode"] == "rtc"
    assert chunking["replan_steps"] == 16
    assert chunking["w_max"] == 0.8
    assert chunking["exclude_dims"] == (6,)
