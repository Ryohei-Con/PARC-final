"""``libero_combined`` schema の登録内容（計画 P1-2/3 / §7.1 / SC-1）。

- [x] ``get_schema("libero_combined")`` が期待の ``image_mapping`` / ``action_mode`` を返す
- [x] 上流 ``libero_10`` と feature/image mapping が一致する（1 エントリの複製である）
- [x] ``action_reorder`` / ``state_reorder`` を持たない（F9: グリッパは index 6 のまま）
"""

from __future__ import annotations

import pytest

from conftest import requires_lerobot

pytestmark = requires_lerobot

EXPECTED_IMAGE_MAPPING = {
    "observation.images.image": "observation.images.image0",
    "observation.images.wrist_image": "observation.images.image1",
}


@pytest.fixture(scope="module")
def overlay():
    import lerobot_policy_parc as module

    module.assert_installed()
    return module


def test_schema_is_registered(overlay):
    from lerobot.dataset_schemas import get_registry

    assert "libero_combined" in get_registry().list_available()


def test_schema_contents(overlay):
    from lerobot.dataset_schemas import get_schema

    schema = get_schema("libero_combined")
    assert schema.robot_type == "libero_combined"
    assert schema.image_mapping == EXPECTED_IMAGE_MAPPING
    assert schema.action_mode == "end_effector"
    assert schema.action_mask_spec == [6, -1]
    assert schema.feature_mapping == {
        "observation.state": ["observation.state"],
        "action": ["action"],
    }


def test_schema_has_no_reorder(overlay):
    """F9: libero schema は reorder を持たない -> グリッパは index 6 のまま。

    RTC guidance の ``exclude_dims=[6]`` はこの事実に依存している。
    """
    from lerobot.dataset_schemas import get_schema

    schema = get_schema("libero_combined")
    assert schema.action_reorder is None
    assert schema.state_reorder is None


def test_action_mask_marks_gripper_absolute(overlay):
    """``action_mask_spec=[6, -1]`` -> 先頭 6 次元が delta、index 6 が絶対値。"""
    from lerobot.dataset_schemas import get_schema

    mask = get_schema("libero_combined").action_mask
    assert mask.shape[0] == 7
    assert bool(mask[:6].all())
    assert not bool(mask[6])


def test_matches_upstream_libero_entry(overlay):
    """上流 ``libero_10`` の 1 エントリを robot_type だけ変えたもの（計画 P1-2）。"""
    from lerobot.dataset_schemas import get_schema

    ours = get_schema("libero_combined")
    upstream = get_schema("libero_10")

    assert ours.feature_mapping == upstream.feature_mapping
    assert ours.image_mapping == upstream.image_mapping
    assert ours.action_mask_spec == upstream.action_mask_spec
    assert ours.action_mode == upstream.action_mode


def test_assert_installed_reports_registrations(overlay):
    summary = overlay.assert_installed()
    assert "libero_combined" in summary["robot_types"]
    assert "render_downsample" in summary["transform_choices"]
    assert "internvla_a1_5_parc" in summary["dataset_choices"]
    assert "internvla_a1_5_parc" in summary["vqa_dataset_choices"]


def test_upstream_schema_yaml_is_untouched(upstream_repo):
    """上流の ``configs/`` に ``libero_combined.yaml`` を置いていないこと（§3 overlay 方針）。"""
    configs = upstream_repo / "src" / "lerobot" / "dataset_schemas" / "configs"
    assert not (configs / "libero_combined.yaml").exists()
