"""`libero_combined` robot schema の登録。

上流 `lerobot.dataset_schemas.registry` の公開 API `load_schemas_from_path()`
だけを使う（計画 F4）。上流の `configs/*.yaml` には一切触らない。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

#: このパッケージが登録する robot_type。`assert_installed()` の照会対象。
PARC_ROBOT_TYPES: tuple[str, ...] = ("libero_combined",)


def schema_files() -> list[Path]:
    """overlay が持つ schema YAML の一覧を返す。"""
    if not SCHEMA_DIR.is_dir():
        raise FileNotFoundError(f"schema directory not found: {SCHEMA_DIR}")
    files = sorted(SCHEMA_DIR.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no *.yaml under {SCHEMA_DIR}")
    return files


def install_parc_schemas() -> list[str]:
    """overlay の schema をグローバルレジストリへ登録し、登録後の robot_type 名を返す。

    Returns:
        登録済み robot_type の一覧（レジストリ全体。overlay 分だけではない）。

    Raises:
        FileNotFoundError: schema ディレクトリ／ファイルが無い。
        RuntimeError: 読み込み後も overlay の robot_type が引けない。
    """
    from lerobot.dataset_schemas import get_registry, load_schemas_from_path

    for path in schema_files():
        load_schemas_from_path(path)

    registry = get_registry()
    available = registry.list_available()

    missing = [rt for rt in PARC_ROBOT_TYPES if rt not in available]
    if missing:
        raise RuntimeError(
            f"schema registration failed for {missing}. "
            f"loaded files={[str(p) for p in schema_files()]}, available={available}"
        )

    logger.info("registered PARC schemas: %s", list(PARC_ROBOT_TYPES))
    return available
