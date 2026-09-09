"""``libero_combined`` schema の登録内容（計画 P1-2/3 / §7.1 / SC-1）。

- [x] ``get_schema("libero_combined")`` が期待の ``image_mapping`` / ``action_mode`` を返す
- [x] 上流 ``libero_10`` と feature mapping / action 系が一致する
- [x] ``image_mapping`` の **源キーだけ**は上流と違う（実データの feature 名に合わせる）
- [x] ``action_reorder`` / ``state_reorder`` を持たない（F9: グリッパは index 6 のまま）
"""

from __future__ import annotations

import pytest

from conftest import requires_lerobot

pytestmark = requires_lerobot

#: 源キーは配布データセット libero_combined_20hz の meta/info.json の feature 名。
#: 上流 libero サブセット（observation.images.image / .wrist_image）とは異なる。
EXPECTED_IMAGE_MAPPING = {
    "observation.images.front": "observation.images.image0",
    "observation.images.wrist": "observation.images.image1",
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
    """上流 ``libero_10`` の 1 エントリを元にしている（計画 P1-2）。

    ``image_mapping`` の **源キーだけ**は意図的に違う。上流 libero サブセットの
    feature 名（``observation.images.image`` / ``.wrist_image``）ではなく、配布
    データセット ``libero_combined_20hz`` の実際の feature 名（``.front`` / ``.wrist``）
    に合わせる必要があるため。**行き先キー（image0/image1）は上流と同じ**でなければ
    ならない（モデル側が読むキーなので）。
    """
    from lerobot.dataset_schemas import get_schema

    ours = get_schema("libero_combined")
    upstream = get_schema("libero_10")

    assert ours.feature_mapping == upstream.feature_mapping
    assert ours.action_mask_spec == upstream.action_mask_spec
    assert ours.action_mode == upstream.action_mode
    # 行き先は一致、源キーは実データ由来で異なる
    assert sorted(ours.image_mapping.values()) == sorted(upstream.image_mapping.values())
    assert set(ours.image_mapping) != set(upstream.image_mapping)


def test_assert_installed_reports_registrations(overlay):
    summary = overlay.assert_installed()
    assert "libero_combined" in summary["robot_types"]
    assert "render_downsample" in summary["transform_choices"]
    assert "internvla_a1_5_parc" in summary["dataset_choices"]
    assert "internvla_a1_5_parc" in summary["vqa_dataset_choices"]
    assert "parc_adamw_vlm_scaled" in summary["optimizer_choices"]


def test_upstream_schema_yaml_is_untouched(upstream_repo):
    """上流の ``configs/`` に ``libero_combined.yaml`` を置いていないこと（§3 overlay 方針）。"""
    configs = upstream_repo / "src" / "lerobot" / "dataset_schemas" / "configs"
    assert not (configs / "libero_combined.yaml").exists()
