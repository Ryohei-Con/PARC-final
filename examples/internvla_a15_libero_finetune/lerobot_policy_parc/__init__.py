"""PARC 2026 用 lerobot overlay パッケージ。

上流 ``InternVLA-A-series`` を 1 バイトも編集せず、公開されている拡張点だけで
必要な追加を行う（計画 §3）:

- robot schema ``libero_combined`` を ``load_schemas_from_path()`` で登録（F4）
- ``RenderDownsampleFn`` を ``DataTransformFn.register_subclass`` で登録（F3）
- ``InternVLAA15ParcDatasetConfig`` / ``InternVLAA15ParcVQADatasetConfig`` を
  ``DatasetConfig`` / ``VQADatasetConfig`` の choice registry へ登録

パッケージ名を ``lerobot_policy_`` 始まりにしてあるので、上流の
``register_third_party_plugins()``（``utils/import_utils.py:133-155``、F1）でも
自動 import される。ただし plugin 自動 import は失敗を握り潰す（F2）ので、
**主経路は ``scripts/train_entry.py`` からの明示 import** である。

使い方::

    import lerobot_policy_parc as overlay
    overlay.assert_installed()      # 登録されていなければ即例外
"""

from __future__ import annotations

import logging

from . import _provenance
from .schema_bootstrap import PARC_ROBOT_TYPES, install_parc_schemas

logger = logging.getLogger(__name__)

__all__ = [
    "PARC_ROBOT_TYPES",
    "RenderDownsampleFn",
    "InternVLAA15ParcDatasetConfig",
    "InternVLAA15ParcVQADatasetConfig",
    "assert_installed",
    "install",
    "installed_summary",
]

#: choice registry へ登録する名前
TRANSFORM_CHOICE = "render_downsample"
DATASET_CHOICE = "internvla_a1_5_parc"

_INSTALLED = False


def install() -> None:
    """schema / transform / dataset config を登録する（冪等）。"""
    global _INSTALLED
    if _INSTALLED:
        return

    install_parc_schemas()

    # import 副作用で choice registry へ登録される。
    from .transforms_render import RenderDownsampleFn as _RenderDownsampleFn  # noqa: F401
    from .dataset_config import (  # noqa: F401
        InternVLAA15ParcDatasetConfig as _Robot,
        InternVLAA15ParcVQADatasetConfig as _VQA,
    )

    _provenance.check_provenance()
    _INSTALLED = True


def installed_summary() -> dict:
    """登録内容のサマリ（ログ・デバッグ用）。"""
    from lerobot.configs.default import DatasetConfig, VQADatasetConfig
    from lerobot.dataset_schemas import get_registry
    from lerobot.transforms.core import DataTransformFn

    return {
        "robot_types": [
            rt for rt in PARC_ROBOT_TYPES if rt in get_registry().list_available()
        ],
        "transform_choices": sorted(DataTransformFn.get_known_choices()),
        "dataset_choices": sorted(DatasetConfig.get_known_choices()),
        "vqa_dataset_choices": sorted(VQADatasetConfig.get_known_choices()),
    }


def assert_installed() -> dict:
    """登録が本当に効いているかを**実照会**で検証する。

    plugin 自動 import は例外を握り潰す（F2）ので、名前が登録されていることを
    レジストリに問い合わせて確認する。失敗したら例外を投げる。
    """
    install()

    from lerobot.configs.default import DatasetConfig, VQADatasetConfig
    from lerobot.dataset_schemas import get_registry, get_schema
    from lerobot.transforms.core import DataTransformFn

    problems: list[str] = []

    available = get_registry().list_available()
    for robot_type in PARC_ROBOT_TYPES:
        if robot_type not in available:
            problems.append(f"robot_type '{robot_type}' not registered")
            continue
        schema = get_schema(robot_type)
        if schema.robot_type != robot_type:
            problems.append(
                f"get_schema('{robot_type}') returned robot_type={schema.robot_type!r}"
            )
        if not schema.image_mapping:
            problems.append(f"schema '{robot_type}' has an empty image_mapping")

    if TRANSFORM_CHOICE not in DataTransformFn.get_known_choices():
        problems.append(f"DataTransformFn choice '{TRANSFORM_CHOICE}' not registered")
    if DATASET_CHOICE not in DatasetConfig.get_known_choices():
        problems.append(f"DatasetConfig choice '{DATASET_CHOICE}' not registered")
    if DATASET_CHOICE not in VQADatasetConfig.get_known_choices():
        problems.append(f"VQADatasetConfig choice '{DATASET_CHOICE}' not registered")

    if problems:
        raise RuntimeError("lerobot_policy_parc overlay is not installed: " + "; ".join(problems))

    summary = installed_summary()
    logger.info("lerobot_policy_parc overlay OK: %s", summary["robot_types"])
    return summary


# --- import 副作用での登録（plugin 自動 import 経路） ------------------------- #
# 失敗しても import 自体は通す。理由をログに残し、assert_installed() で落とす。
try:
    install()
except Exception:  # pragma: no cover - lerobot が無い環境など
    logger.exception("lerobot_policy_parc: deferred install failed; call assert_installed()")


def __getattr__(name: str):
    """遅延属性（``lerobot_policy_parc.RenderDownsampleFn`` 等）。"""
    if name == "RenderDownsampleFn":
        from .transforms_render import RenderDownsampleFn

        return RenderDownsampleFn
    if name in ("InternVLAA15ParcDatasetConfig", "InternVLAA15ParcVQADatasetConfig"):
        from . import dataset_config

        return getattr(dataset_config, name)
    raise AttributeError(name)
