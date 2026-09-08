"""DatasetConfig overlay の単体テスト（計画 P1-6 / §7.1 / SC-1）。

- [x] ``RenderDownsampleFn`` が ``ResizeImagesWithPadFn`` の**直前**にちょうど 1 個入る
- [x] ``__post_init__`` を 2 回走らせても 1 個のまま（冪等）
- [x] ``action_mode="abs"`` で ``DeltaActionTransformFn`` が消える
- [x] draccus で dict 化 -> 復元して等価
- [x] VQA チェーンでは ``ResizeVQAImagesWithPadFn`` の直前に入る（dead path だが実装はする）
"""

from __future__ import annotations

import pytest

from conftest import requires_lerobot

pytestmark = requires_lerobot


@pytest.fixture(scope="module")
def overlay():
    import lerobot_policy_parc as module

    module.assert_installed()
    return module


@pytest.fixture()
def config_cls(overlay, offline_processors):
    """`offline_processors` の理由は conftest.py の同名フィクスチャの docstring を参照。

    要約: 上流の DatasetConfig は __post_init__ で Qwen / FAST のトークナイザを
    Hub から取りに行く。ここで検証したいのは **transform チェーンの並び順** なので、
    ダウンロードだけを止める。
    """
    from lerobot_policy_parc.dataset_config import InternVLAA15ParcDatasetConfig

    return InternVLAA15ParcDatasetConfig


@pytest.fixture()
def vqa_config_cls(overlay, offline_processors):
    from lerobot_policy_parc.dataset_config import InternVLAA15ParcVQADatasetConfig

    return InternVLAA15ParcVQADatasetConfig


def _chain(config) -> list[str]:
    return [type(t).__name__ for t in config.data_transforms.inputs]


# --------------------------------------------------------------------------- #
# 挿入位置
# --------------------------------------------------------------------------- #
def test_render_downsample_is_immediately_before_resize(config_cls):
    from lerobot.transforms.core import ResizeImagesWithPadFn
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = config_cls(repo_id="dummy")
    inputs = config.data_transforms.inputs

    render_indices = [i for i, t in enumerate(inputs) if isinstance(t, RenderDownsampleFn)]
    resize_indices = [i for i, t in enumerate(inputs) if isinstance(t, ResizeImagesWithPadFn)]

    assert len(render_indices) == 1, _chain(config)
    assert len(resize_indices) == 1, _chain(config)
    assert render_indices[0] + 1 == resize_indices[0], _chain(config)


def test_render_downsample_probabilities_are_propagated(config_cls):
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = config_cls(repo_id="dummy")
    render = next(t for t in config.data_transforms.inputs if isinstance(t, RenderDownsampleFn))
    assert render.target_h == 128 and render.target_w == 128
    assert render.p_nearest == 0.10
    assert render.probabilities() == {
        "box": 0.45,
        "triangle": 0.30,
        "cubic": 0.15,
        "nearest": 0.10,
    }


def test_custom_probabilities_are_propagated(config_cls):
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = config_cls(
        repo_id="dummy",
        render_p_box=0.60,
        render_p_triangle=0.20,
        render_p_cubic=0.10,
        render_p_nearest=0.10,
    )
    render = next(t for t in config.data_transforms.inputs if isinstance(t, RenderDownsampleFn))
    assert render.p_box == 0.60
    assert render.p_nearest == 0.10


def test_disabled_removes_the_transform(config_cls):
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = config_cls(repo_id="dummy", render_downsample=False)
    assert not any(isinstance(t, RenderDownsampleFn) for t in config.data_transforms.inputs)


# --------------------------------------------------------------------------- #
# 冪等性
# --------------------------------------------------------------------------- #
def test_post_init_is_idempotent(config_cls):
    from lerobot.transforms.core import ResizeImagesWithPadFn
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = config_cls(repo_id="dummy")
    before = _chain(config)

    config.__post_init__()
    config.__post_init__()

    after = _chain(config)
    assert before == after
    assert sum(isinstance(t, RenderDownsampleFn) for t in config.data_transforms.inputs) == 1
    assert sum(isinstance(t, ResizeImagesWithPadFn) for t in config.data_transforms.inputs) == 1


def test_two_instances_do_not_share_transform_lists(config_cls):
    """dataclass の default_factory が共有されていないこと（挿入が累積しない）。"""
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    first = config_cls(repo_id="a")
    second = config_cls(repo_id="b")
    for config in (first, second):
        assert sum(
            isinstance(t, RenderDownsampleFn) for t in config.data_transforms.inputs
        ) == 1


# --------------------------------------------------------------------------- #
# action_mode
# --------------------------------------------------------------------------- #
def test_abs_action_mode_removes_delta_transform(config_cls):
    from lerobot.transforms.core import DeltaActionTransformFn

    config = config_cls(repo_id="dummy", action_mode="abs")
    assert not any(isinstance(t, DeltaActionTransformFn) for t in config.data_transforms.inputs)


def test_delta_action_mode_keeps_delta_transform(config_cls):
    from lerobot.transforms.core import DeltaActionTransformFn

    config = config_cls(repo_id="dummy", action_mode="delta")
    assert any(isinstance(t, DeltaActionTransformFn) for t in config.data_transforms.inputs)


def test_bad_action_mode_is_rejected(config_cls):
    with pytest.raises(AssertionError):
        config_cls(repo_id="dummy", action_mode="nonsense")


# --------------------------------------------------------------------------- #
# draccus 往復
# --------------------------------------------------------------------------- #
def test_draccus_roundtrip(config_cls):
    """``train_config.json`` に記録され、そこから復元できること（F3）。

    ``draccus.encode`` は「具象クラスを直接 encode したとき」には discriminator
    （``CHOICE_TYPE_KEY``）を付けない。付くのは**親の field 型が基底クラスのとき**で、
    ``train_config.json`` はまさにその形（``TrainPipelineConfig.dataset: DatasetConfig``）
    になる。テストではその 1 段を手で足して往復させる。
    """
    import draccus

    from lerobot.configs.default import DatasetConfig

    config = config_cls(repo_id="dummy", render_p_nearest=0.10)
    encoded = draccus.encode(config)

    assert isinstance(encoded, dict)
    assert encoded["render_p_nearest"] == 0.10
    assert DatasetConfig.get_choice_name(config_cls) == "internvla_a1_5_parc"

    # ネストした transform 側には discriminator が入る（= train_config.json に残る）
    transform_types = [t.get(draccus.CHOICE_TYPE_KEY) for t in encoded["data_transforms"]["inputs"]]
    assert "render_downsample" in transform_types
    assert transform_types.index("render_downsample") + 1 == transform_types.index(
        "resize_with_pad"
    )

    payload = dict(encoded)
    payload[draccus.CHOICE_TYPE_KEY] = DatasetConfig.get_choice_name(config_cls)

    restored = draccus.decode(DatasetConfig, payload)
    assert type(restored) is config_cls
    assert _chain(restored) == _chain(config)
    assert draccus.encode(restored) == encoded


def test_render_downsample_probabilities_land_in_train_config(config_cls):
    """SC-5 の Evaluator 項目: 確率つきで記録されていること。"""
    import draccus

    config = config_cls(repo_id="dummy")
    encoded = draccus.encode(config)
    render = next(
        t for t in encoded["data_transforms"]["inputs"] if t.get("type") == "render_downsample"
    )
    assert render["p_box"] == 0.45
    assert render["p_triangle"] == 0.30
    assert render["p_cubic"] == 0.15
    assert render["p_nearest"] == 0.10


# --------------------------------------------------------------------------- #
# VQA（dead path）
# --------------------------------------------------------------------------- #
def test_vqa_render_downsample_is_before_vqa_resize(vqa_config_cls):
    """現行レシピでは VQA データセットを与えないので **dead path**（計画 D5）。

    それでも実装はしておき、挿入位置だけは単体テストで固定する。
    """
    from lerobot.transforms.core import ResizeVQAImagesWithPadFn
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = vqa_config_cls()
    inputs = config.data_transforms.inputs

    render_indices = [i for i, t in enumerate(inputs) if isinstance(t, RenderDownsampleFn)]
    resize_indices = [i for i, t in enumerate(inputs) if isinstance(t, ResizeVQAImagesWithPadFn)]

    assert len(render_indices) == 1
    assert len(resize_indices) == 1
    assert render_indices[0] + 1 == resize_indices[0]


def test_vqa_post_init_is_idempotent(vqa_config_cls):
    config = vqa_config_cls()
    before = _chain(config)
    config.__post_init__()
    assert _chain(config) == before


# --------------------------------------------------------------------------- #
# 上流不変
# --------------------------------------------------------------------------- #
def test_upstream_config_is_unchanged(config_cls, offline_processors):
    """上流 ``InternVLAA15DatasetConfig`` には RenderDownsampleFn が入らないこと。"""
    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (
        InternVLAA15DatasetConfig,
    )
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    upstream = InternVLAA15DatasetConfig(repo_id="dummy")
    assert not any(isinstance(t, RenderDownsampleFn) for t in upstream.data_transforms.inputs)
