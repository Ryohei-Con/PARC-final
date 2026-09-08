"""上流 ``InternVLA-A-series`` の主要ファイルのハッシュを記録・照合する。

overlay 方針（計画 §3 / §4.2）では上流を 1 バイトも編集しない。代わりに、
overlay が前提にしている上流ファイルが変わっていないかを起動時に照合し、
違っていれば **warning を出す**（落とさない）。

「静かに壊れる」ことを避けるのが目的なので、照合結果は必ずログに出す。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
#: 期待ハッシュの保存先。`record_provenance()` で更新する。
PROVENANCE_FILE = _HERE / "upstream_provenance.json"

#: overlay が挙動を前提にしている上流ファイル（`src/lerobot` からの相対パス）。
TRACKED_FILES: tuple[str, ...] = (
    "lerobot/transforms/core.py",
    "lerobot/transforms/utils.py",
    "lerobot/dataset_schemas/registry.py",
    "lerobot/dataset_schemas/schema.py",
    "lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py",
    "lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py",
    "lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_src_dir() -> Path | None:
    """上流 ``src`` ディレクトリを解決する。

    優先順:
      1. 環境変数 ``IVLA_REPO`` (= リポジトリルート) の ``src``
      2. import 済み ``lerobot`` の親ディレクトリ
    """
    repo = os.environ.get("IVLA_REPO")
    if repo:
        candidate = Path(repo).expanduser().resolve() / "src"
        if (candidate / "lerobot" / "__init__.py").is_file():
            return candidate

    try:
        import lerobot
    except Exception:  # pragma: no cover - lerobot が無い環境
        return None

    lerobot_file = getattr(lerobot, "__file__", None)
    if not lerobot_file:
        return None
    return Path(lerobot_file).resolve().parent.parent


def compute_hashes(src_dir: Path | None = None) -> dict[str, str]:
    """追跡対象ファイルの SHA256 を計算する。見つからないものは ``"<missing>"``。"""
    src_dir = src_dir or upstream_src_dir()
    if src_dir is None:
        raise RuntimeError(
            "cannot locate upstream src/; set IVLA_REPO or make lerobot importable"
        )

    result: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = src_dir / relative
        result[relative] = sha256_file(path) if path.is_file() else "<missing>"
    return result


def record_provenance(src_dir: Path | None = None, out_file: Path | None = None) -> dict[str, str]:
    """現在の上流ハッシュを ``upstream_provenance.json`` に書き出す。"""
    hashes = compute_hashes(src_dir)
    out_file = out_file or PROVENANCE_FILE
    out_file.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote upstream provenance to %s", out_file)
    return hashes


def load_expected(out_file: Path | None = None) -> dict[str, str]:
    out_file = out_file or PROVENANCE_FILE
    if not out_file.is_file():
        return {}
    return json.loads(out_file.read_text(encoding="utf-8"))


def check_provenance(src_dir: Path | None = None, out_file: Path | None = None) -> list[str]:
    """記録済みハッシュと照合し、食い違ったファイルの一覧を返す（warning も出す）。

    記録が無い場合（初回）は空リストを返し、その旨を info ログに出す。
    """
    expected = load_expected(out_file)
    if not expected:
        logger.info(
            "no upstream provenance recorded yet; run _provenance.record_provenance() once"
        )
        return []

    try:
        actual = compute_hashes(src_dir)
    except RuntimeError as exc:
        logger.warning("upstream provenance check skipped: %s", exc)
        return []

    drifted = [name for name, digest in expected.items() if actual.get(name) != digest]
    if drifted:
        logger.warning(
            "upstream files changed since provenance was recorded: %s. "
            "The PARC overlay assumes upstream behaviour at commit e6fc904; re-verify §0 facts.",
            drifted,
        )
    else:
        logger.info("upstream provenance OK (%d files)", len(expected))
    return drifted


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(record_provenance(), indent=2, sort_keys=True))
