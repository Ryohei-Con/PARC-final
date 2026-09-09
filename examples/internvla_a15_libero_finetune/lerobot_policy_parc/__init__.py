"""PARC 2026 用 lerobot overlay パッケージ。

上流 ``InternVLA-A-series`` を 1 バイトも編集せず、公開されている拡張点だけで
必要な追加を行う（計画 §3）:

- robot schema ``libero_combined`` を ``load_schemas_from_path()`` で登録（F4）
- ``RenderDownsampleFn`` を ``DataTransformFn.register_subclass`` で登録（F3）
- ``InternVLAA15ParcDatasetConfig`` / ``InternVLAA15ParcVQADatasetConfig`` を
  ``DatasetConfig`` / ``VQADatasetConfig`` の choice registry へ登録
- ``ParcAdamWVlmScaledConfig``（バックボーン VLM だけ lr を 0.1 倍）を
  ``OptimizerConfig`` の choice registry へ登録

学習時のみ、上流に届かない 2 点を :mod:`._monkeypatch` で差し替える。これは
``install()`` ではなく **``install_training_patches()``** が行う（上流モデルの import が
必要で重いため、登録だけしたい経路と分けてある）。主経路 ``scripts/train_entry.py`` は
必ずこれを呼ぶ。

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

from . import _monkeypatch, _provenance
from .schema_bootstrap import PARC_ROBOT_TYPES, install_parc_schemas

logger = logging.getLogger(__name__)

__all__ = [
    "PARC_ROBOT_TYPES",
    "ParcAdamWVlmScaledConfig",
    "assert_training_patches",
    "install_training_patches",
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
OPTIMIZER_CHOICE = "parc_adamw_vlm_scaled"

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
    from .optim_vlm_lr import ParcAdamWVlmScaledConfig as _Optim  # noqa: F401

    _provenance.check_provenance()
    _INSTALLED = True


def installed_summary() -> dict:
    """登録内容のサマリ（ログ・デバッグ用）。"""
    from lerobot.configs.default import DatasetConfig, VQADatasetConfig
    from lerobot.dataset_schemas import get_registry
    from lerobot.transforms.core import DataTransformFn

    from lerobot.optim.optimizers import OptimizerConfig

    return {
        "optimizer_choices": sorted(OptimizerConfig.get_known_choices()),
        "training_patches": _monkeypatch.applied(),
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
    from lerobot.optim.optimizers import OptimizerConfig
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
    if OPTIMIZER_CHOICE not in OptimizerConfig.get_known_choices():
        problems.append(f"OptimizerConfig choice '{OPTIMIZER_CHOICE}' not registered")

    if problems:
        raise RuntimeError("lerobot_policy_parc overlay is not installed: " + "; ".join(problems))

    summary = installed_summary()
    logger.info("lerobot_policy_parc overlay OK: %s", summary["robot_types"])
    return summary


def install_training_patches() -> list[str]:
    """学習経路でだけ必要な monkeypatch を当てる（計画 §3 の例外規定）。

    上流モデル（``modeling_internvla_a1_5``）の import が要るので、schema/transform の
    登録だけをしたい経路（テスト等）と分けてある。:func:`install` には含めない。
    """
    install()
    return _monkeypatch.apply()


def assert_training_patches() -> dict:
    """monkeypatch が本当に効いているかを**実照会**で検証する。

    「VLM の学習率を 0.1 倍したつもり」で全パラメータが同じ lr のまま回る事故を、
    学習を始める前にここで落とす。
    """
    applied = install_training_patches()

    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config

    from .optim_vlm_lr import ParcAdamWVlmScaledConfig

    problems: list[str] = []
    if not applied:
        problems.append("no monkeypatch applied")

    # preset が実際に差し替わっているか（CLI の --optimizer.* は preset に上書きされる）
    preset = InternVLAA15Config().get_optimizer_preset()
    if not isinstance(preset, ParcAdamWVlmScaledConfig):
        problems.append(
            f"get_optimizer_preset() returned {type(preset).__name__}"
            f"（{ParcAdamWVlmScaledConfig.__name__} のはず）"
        )
    elif preset.vlm_lr_scale >= 1.0:
        logger.warning(
            "vlm_lr_scale=%s（1.0 以上）。VLM の学習率は下がらない。%s を確認すること。",
            preset.vlm_lr_scale, _monkeypatch.ENV_VLM_LR_SCALE,
        )

    if problems:
        raise RuntimeError("lerobot_policy_parc training patches not applied: " + "; ".join(problems))

    summary = {"patches": applied, "vlm_lr_scale": preset.vlm_lr_scale}
    logger.info("lerobot_policy_parc training patches OK: %s", summary)
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
    if name == "ParcAdamWVlmScaledConfig":
        from .optim_vlm_lr import ParcAdamWVlmScaledConfig

        return ParcAdamWVlmScaledConfig
    raise AttributeError(name)
