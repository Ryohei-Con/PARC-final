"""InternVLA-A1.5 の Python 3.10 ブートストラップ + 推論ランタイム。

提出物（zip）の中で ``policy_server.py`` と ``verify_inference.py`` の**両方**が
この 1 ファイルの :func:`build_submission_runtime` を通る（移植手順書 B8）。
片方だけ別経路にすると、自己チェックが本番と違うものを測ることになる。

このファイルが潰しているサイレント失敗（移植手順書 §6 / 計画 §0）:

F6  上流の推論バックエンドは画像をリサイズしていない。
    ``ResizeImagesWithPadFn`` を hydrate せずに構築しているため ``mapping`` が空で、
    ``__call__`` のループが 0 回になる。さらに mapping のキー
    （``observation.images.image`` / ``...wrist_image``）とサンプルのキー
    （``observation.images.image0/1/2``）が食い違うので二重に no-op。
    → :meth:`InternVLARuntime._obs_to_sample` は
      ``lerobot.transforms.utils.resize_with_pad(x, 224, 224)`` を**無条件で**通す。
      学習側と同一関数を呼ぶ。

B1  別の ``lerobot`` が先に import されていると、どちらが動いているか分からない。
    → :func:`install_vendored_lerobot` が検出して落とす。

B6  ``from_pretrained(strict=False)`` で一部がランダム初期化のまま静かに動く。
    → :func:`assert_checkpoint_covers_model` を起動パスに入れる。

B7  ``resize_token_embeddings`` の ``mean_resizing=True`` で起動が数十秒延びる。
    → :func:`fast_token_embedding_resize` で ``False`` に固定する。

B8  環境変数に依存した設定は本番で渡ってこない。
    → 画像の向き・グリッパ規約・chunking はすべて ``runtime_config.json`` 由来。
      **コードに直書きしない。** ``"TBD"`` のまま起動しようとしたら例外を投げる。
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent

#: 画像の向き補正の選択肢。runtime_config.json の image_orientation に書く値。
ORIENTATIONS: tuple[str, ...] = ("none", "flip_ud", "flip_lr", "rot180")

#: 未確定を示す番人。この値のまま起動しようとすると例外になる。
TBD = "TBD"

#: 採点環境が送ってくる観測キー（submission_template/policy_server.py の docstring）。
OBS_AGENTVIEW = "agentview_image"
OBS_WRIST = "robot0_eye_in_hand_image"


# =========================================================================== #
# 1. ブートストラップ（vendor lerobot / transformers パッチ）
# =========================================================================== #
def install_vendored_lerobot(vendor_dir: str | Path) -> Path:
    """``vendor/lerobot`` を ``sys.path`` の先頭に入れる（移植手順書 B1）。

    **既に別の lerobot が import 済みなら黙って先勝ちさせず例外を投げる。**
    どちらの lerobot が動いているか分からないまま精度だけ落ちるのを防ぐ。
    """
    vendor_dir = Path(vendor_dir).resolve()
    if not (vendor_dir / "lerobot" / "__init__.py").is_file():
        raise FileNotFoundError(f"vendored lerobot not found under {vendor_dir}")

    if "lerobot" in sys.modules:
        loaded_file = getattr(sys.modules["lerobot"], "__file__", "") or ""
        loaded = Path(loaded_file).resolve() if loaded_file else None
        if loaded is None or vendor_dir not in loaded.parents:
            raise RuntimeError(
                f"a different 'lerobot' is already imported: {loaded}. "
                f"Expected it under {vendor_dir}. Import internvla_runtime first."
            )
        return vendor_dir

    path_str = str(vendor_dir)
    while path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
    importlib.invalidate_caches()
    return vendor_dir


def _inject_qwen35_module(patch_file: Path) -> None:
    package_name = "transformers.models.qwen3_5"
    module_name = f"{package_name}.modeling_qwen3_5"
    package = importlib.import_module(package_name)
    spec = importlib.util.spec_from_file_location(module_name, patch_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {patch_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)  # 半端な状態を残さない
        raise
    setattr(package, "modeling_qwen3_5", module)


def patch_transformers_qwen3_5(vendor_dir: str | Path) -> str:
    """差し替え版 ``modeling_qwen3_5.py`` を有効にする（移植手順書 B2）。

    Returns:
        ``"already-patched"`` / ``"copied"`` / ``"injected"`` / ``"absent"``。
    """
    patch_file = Path(vendor_dir) / "transformers_patch" / "models" / "qwen3_5" / "modeling_qwen3_5.py"
    if not patch_file.is_file():
        logger.warning("transformers patch not found at %s; using stock transformers", patch_file)
        return "absent"

    import transformers

    target = Path(transformers.__file__).parent / "models" / "qwen3_5" / "modeling_qwen3_5.py"
    if target.is_file() and target.read_bytes() == patch_file.read_bytes():
        return "already-patched"

    if "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        raise RuntimeError(
            "transformers.models.qwen3_5.modeling_qwen3_5 is already imported; "
            "call patch_transformers_qwen3_5() before importing the model"
        )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(patch_file, target)
        importlib.invalidate_caches()
        return "copied"
    except OSError:
        _inject_qwen35_module(patch_file)
        return "injected"


def bootstrap(runtime_dir: str | Path | None = None) -> dict[str, Any]:
    """vendor lerobot の注入 + transformers パッチ。``MyPolicy.__init__`` の先頭で呼ぶ。"""
    runtime_dir = Path(runtime_dir or _THIS_DIR).resolve()

    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))

    result: dict[str, Any] = {"runtime_dir": str(runtime_dir)}

    vendor_dir = runtime_dir / "vendor"
    if (vendor_dir / "lerobot" / "__init__.py").is_file():
        result["vendor_lerobot"] = str(install_vendored_lerobot(vendor_dir))
        result["transformers_patch"] = patch_transformers_qwen3_5(vendor_dir)
    else:
        # 開発中（提出前）はレポジトリの上流 src を使ってよい。
        logger.warning("no vendor/lerobot under %s; relying on the ambient lerobot", runtime_dir)
        result["vendor_lerobot"] = None
        result["transformers_patch"] = "absent"

    import lerobot

    result["lerobot_file"] = getattr(lerobot, "__file__", None)
    logger.info("bootstrap: %s", result)
    return result


@contextlib.contextmanager
def fast_token_embedding_resize():
    """``resize_token_embeddings`` を ``mean_resizing=False`` に固定する（B7）。

    既定 ``True`` は ``[250368, 2048]`` の embedding の平均・共分散を計算するので
    **これだけで数十秒**かかる。しかも結果は直後の ``_init_new_rows()`` と
    チェックポイントの strict ロードで完全に上書きされるので、最終的な重みは
    1 ビットも変わらない。
    """
    from transformers.modeling_utils import PreTrainedModel

    original = PreTrainedModel.resize_token_embeddings

    def _patched(self, *args, **kwargs):
        kwargs.setdefault("mean_resizing", False)
        return original(self, *args, **kwargs)

    PreTrainedModel.resize_token_embeddings = _patched
    try:
        yield
    finally:
        PreTrainedModel.resize_token_embeddings = original


@contextlib.contextmanager
def vlm_built_from_config_only(vlm_dir: str | Path):
    """VLM を config だけから構築する（B4）。重みは直後に checkpoint が全部上書きする。"""
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    original = Qwen3_5ForConditionalGeneration.from_pretrained

    def _from_config(pretrained_model_name_or_path=None, *args, **kwargs):
        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(
            pretrained_model_name_or_path or str(vlm_dir)
        )
        with no_init_weights():
            return Qwen3_5ForConditionalGeneration(config)

    Qwen3_5ForConditionalGeneration.from_pretrained = staticmethod(_from_config)
    try:
        yield
    finally:
        Qwen3_5ForConditionalGeneration.from_pretrained = original


# =========================================================================== #
# 2. 設定
# =========================================================================== #
def load_runtime_config(path: str | Path | None = None) -> dict[str, Any]:
    """``runtime_config.json`` を読む。

    探索順:
      1. 引数 ``path``
      2. 環境変数 ``IVLA_RUNTIME_CONFIG``
      3. このファイルと同じディレクトリの ``runtime_config.json``
    """
    if path is None:
        path = os.environ.get("IVLA_RUNTIME_CONFIG") or (_THIS_DIR / "runtime_config.json")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"runtime config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"runtime config must be a JSON object, got {type(config).__name__}")
    config["_path"] = str(path.resolve())
    return config


def apply_config_env(config: dict[str, Any]) -> dict[str, str]:
    """``config["env"]`` を ``os.environ.setdefault`` で流し込む（B8）。

    setdefault なので、手元で環境変数を立てればそちらが優先される。
    """
    applied: dict[str, str] = {}
    for key, value in (config.get("env") or {}).items():
        if str(key).startswith("_"):
            continue
        os.environ.setdefault(str(key), str(value))
        applied[str(key)] = os.environ[str(key)]
    return applied


def _chunk_blending():
    """``chunk_blending`` モジュールを返す（設定検証から参照するため）。

    ``SCHEDULES`` / ``REAL_ACTION_DIM`` / ``DEFAULT_EXCLUDE_DIMS`` の**唯一の定義元**は
    ``chunk_blending`` 側なので、値をこちらに複製しない。複製すると片方だけ増えたときに
    「config では通るが実行時に落ちる」という本来潰したかった不整合が復活する。

    :func:`bootstrap` を通らずに ``internvla_runtime`` だけを import した経路
    （単体テストなど）でも解決できるように、``sys.path`` を補ってから import する。
    """
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    return importlib.import_module("chunk_blending")


def _reject_tbd(value: Any, what: str) -> Any:
    if isinstance(value, str) and value.strip().upper() == TBD:
        raise ValueError(
            f"{what} is still \"TBD\" in runtime_config.json. "
            "It must be determined from the data (plan P2) before running inference. "
            "Refusing to fall back to a default silently."
        )
    return value


def resolve_image_orientation(config: dict[str, Any]) -> dict[str, str]:
    """``image_orientation`` を検証して返す。``TBD`` なら例外。"""
    section = config.get("image_orientation") or {}
    result: dict[str, str] = {}
    for camera in ("agentview", "wrist"):
        if camera not in section:
            raise KeyError(f"runtime_config.json: image_orientation.{camera} is missing")
        mode = _reject_tbd(section[camera], f"image_orientation.{camera}")
        mode = str(mode)
        if mode not in ORIENTATIONS:
            raise ValueError(
                f"runtime_config.json: image_orientation.{camera}={mode!r} "
                f"must be one of {ORIENTATIONS}"
            )
        result[camera] = mode
    return result


def resolve_gripper(config: dict[str, Any]) -> dict[str, Any]:
    """``gripper`` を検証して返す。``dataset_convention`` が ``TBD`` なら例外。"""
    section = config.get("gripper") or {}
    convention = _reject_tbd(
        section.get("dataset_convention", TBD), "gripper.dataset_convention"
    )
    convention = str(convention)
    if convention not in ("zero_one", "minus_one_one"):
        raise ValueError(
            f"runtime_config.json: gripper.dataset_convention={convention!r} "
            "must be 'zero_one' or 'minus_one_one'"
        )

    threshold = section.get("threshold")
    if threshold is None:
        threshold = 0.5 if convention == "zero_one" else 0.0
    below = str(section.get("below_threshold", "close"))
    if below not in ("close", "open"):
        raise ValueError(
            f"runtime_config.json: gripper.below_threshold={below!r} must be 'close' or 'open'"
        )

    return {
        "dataset_convention": convention,
        "binarize": bool(section.get("binarize", True)),
        "threshold": float(threshold),
        "below_threshold": below,
        "env_close": float(section.get("env_close", 1.0)),
        "env_open": float(section.get("env_open", -1.0)),
    }


def resolve_chunking(config: dict[str, Any]) -> dict[str, Any]:
    """``chunking`` を検証して返す。``w_max >= 1``（ハードマスク）は拒否する。

    **``schedule`` と ``exclude_dims`` もここで検証する。** どちらも素通しにすると
    ``InternVLARuntime.__init__`` を通過してモデルまでロードしてしまい、
    ``schedule`` は最初の replan 境界（= env 十数ステップ目）で ``chunk_blending``
    の ``_weight_schedule`` が ``/act`` の中から ``ValueError`` を投げ、
    ``exclude_dims`` の範囲外値は**黙って捨てられる**（``build_guidance`` /
    ``TemporalEnsembler`` が範囲でフィルタしている）。前者は
    ``__init__`` の「設定の検証はモデルをロードする前に済ませる」という不変条件を
    破り、後者はサイレント失敗そのものなので、両方ここで落とす。
    """
    blending = _chunk_blending()
    section = config.get("chunking") or {}
    mode = str(section.get("mode", "none")).lower()
    if mode not in ("none", "rtc", "ensemble"):
        raise ValueError(
            f"runtime_config.json: chunking.mode={mode!r} must be 'none', 'rtc' or 'ensemble'"
        )

    replan_steps = int(section.get("replan_steps", 16))
    if replan_steps < 1:
        raise ValueError(f"runtime_config.json: chunking.replan_steps={replan_steps} must be >= 1")

    w_max = float(section.get("w_max", 0.8))
    if not (0.0 <= w_max < 1.0):
        raise ValueError(
            f"runtime_config.json: chunking.w_max={w_max} must satisfy 0 <= w_max < 1 "
            "(hard masking is deliberately not supported)"
        )

    schedule = str(section.get("schedule", "linear"))
    if schedule not in blending.SCHEDULES:
        raise ValueError(
            f"runtime_config.json: chunking.schedule={schedule!r} "
            f"must be one of {blending.SCHEDULES}"
        )

    exclude_dims = _resolve_exclude_dims(
        section.get("exclude_dims", blending.DEFAULT_EXCLUDE_DIMS),
        real_action_dim=int(blending.REAL_ACTION_DIM),
    )

    ensemble_m = float(section.get("ensemble_m", 0.1))
    if ensemble_m < 0.0:
        raise ValueError(
            f"runtime_config.json: chunking.ensemble_m={ensemble_m} must be >= 0 "
            "(TemporalEnsembler rejects negative decay rates)"
        )

    return {
        "mode": mode,
        "replan_steps": replan_steps,
        "w_max": w_max,
        "schedule": schedule,
        "exclude_dims": exclude_dims,
        "ensemble_m": ensemble_m,
    }


def _resolve_exclude_dims(raw: Any, *, real_action_dim: int) -> tuple[int, ...]:
    """``chunking.exclude_dims`` を検証して ``tuple[int, ...]`` にする。

    範囲外の値は ``build_guidance`` / ``TemporalEnsembler`` が黙って捨てるので、
    ここで落とさないと「除外したつもりの次元が除外されていない」状態のまま走る。
    padding 次元（``>= real_action_dim``）は元から guidance の対象外なので、
    そこを指定するのは設定ミスである。
    """
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError(
            f"runtime_config.json: chunking.exclude_dims={raw!r} must be a list of integers"
        )

    dims: list[int] = []
    for entry in raw:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ValueError(
                f"runtime_config.json: chunking.exclude_dims contains a non-integer entry "
                f"{entry!r}"
            )
        index = int(entry)
        if index != entry:
            raise ValueError(
                f"runtime_config.json: chunking.exclude_dims contains a non-integer entry "
                f"{entry!r}"
            )
        if not (0 <= index < real_action_dim):
            raise ValueError(
                f"runtime_config.json: chunking.exclude_dims contains {index}, which is outside "
                f"the real action dims [0, {real_action_dim}). Padding dims are already excluded "
                "from the guidance, and out-of-range entries would be dropped silently."
            )
        if index in dims:
            raise ValueError(
                f"runtime_config.json: chunking.exclude_dims contains {index} more than once"
            )
        dims.append(index)
    return tuple(dims)


# =========================================================================== #
# 3. 純粋関数（手元で単体テストできる部分）
# =========================================================================== #
def quat2axisangle(quat) -> np.ndarray:
    """クォータニオン ``(x, y, z, w)`` を axis-angle（回転ベクトル）に変換する。

    上流 ``evaluation/LIBERO/model2libero_interface.py:11-32`` の移植。
    ``scipy.spatial.transform.Rotation.as_rotvec()`` と一致する（符号を含む）。

    Args:
        quat: 長さ 4 の配列。**robosuite 規約で末尾が w**。

    Returns:
        ``(3,) float32`` の回転ベクトル。恒等回転では ``[0, 0, 0]``。
    """
    q = np.asarray(quat, dtype=np.float32).reshape(-1).copy()
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion of length 4, got {q.shape}")

    # acos の定義域を外れないようにクランプする（数値誤差で |w| > 1 になりうる）。
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0

    den = float(np.sqrt(1.0 - float(q[3]) * float(q[3])))
    if np.isclose(den, 0.0):
        # w = +-1 の縮退（回転角 0 または 2pi）。
        return np.zeros(3, dtype=np.float32)

    return ((q[:3] * 2.0 * float(np.arccos(float(q[3])))) / den).astype(np.float32)


def orient_image(arr, mode: str) -> np.ndarray:
    """画像の向きを補正する。``mode`` は ``runtime_config.json`` 由来。

    ``"TBD"`` や未知の値は**例外**にする。黙って ``"none"`` に落とさない。
    """
    mode = str(mode)
    _reject_tbd(mode, "image orientation")
    if mode not in ORIENTATIONS:
        raise ValueError(f"unknown image orientation {mode!r}; expected one of {ORIENTATIONS}")

    image = np.asarray(arr)
    if image.ndim != 3:
        raise ValueError(f"orient_image expects HWC image, got shape {image.shape}")

    if mode == "none":
        out = image
    elif mode == "flip_ud":
        out = image[::-1, :, :]
    elif mode == "flip_lr":
        out = image[:, ::-1, :]
    else:  # rot180
        out = image[::-1, ::-1, :]
    return np.ascontiguousarray(out)


def binarize_gripper(value: float, gripper_cfg: dict[str, Any]) -> float:
    """モデル出力のグリッパ次元を env 規約の ``{env_close, env_open}`` に落とす。

    しきい値と向きは ``runtime_config.json`` 由来。コードに直書きしない。
    """
    if not gripper_cfg.get("binarize", True):
        return float(value)
    below = gripper_cfg["below_threshold"] == "close"
    is_below = float(value) < float(gripper_cfg["threshold"])
    close = is_below if below else not is_below
    return float(gripper_cfg["env_close"] if close else gripper_cfg["env_open"])


def assert_checkpoint_covers_model(policy, checkpoint_file: str | Path) -> int:
    """checkpoint がモデルの全パラメータを覆っているかを検証する（移植手順書 B6）。

    ``PreTrainedPolicy.from_pretrained`` の ``strict`` は既定 False で、キーが足りなくても
    例外を投げずログを出すだけ。ランダム初期化のまま静かに動いてしまうので、起動時に
    明示的に落とす。tie された重みは保存時に片方が落ちるので、同じストレージを指す
    キーが checkpoint にあれば欠損とみなさない。

    Returns:
        モデル側の state_dict のキー数。

    Raises:
        RuntimeError: checkpoint に足りないキーがある。
    """
    from safetensors import safe_open

    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")

    with safe_open(str(checkpoint_file), framework="pt") as handle:
        ckpt_keys = set(handle.keys())

    state = policy.state_dict()

    by_storage: dict[int, list[str]] = {}
    for name, tensor in state.items():
        by_storage.setdefault(tensor.data_ptr(), []).append(name)

    missing: list[str] = []
    for name in sorted(set(state) - ckpt_keys):
        twins = by_storage.get(state[name].data_ptr(), [])
        if any(twin in ckpt_keys for twin in twins):
            continue  # tie された重み
        missing.append(name)

    if missing:
        raise RuntimeError(
            f"checkpoint {checkpoint_file} is missing {len(missing)} model parameters: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    unexpected = sorted(ckpt_keys - set(state))
    if unexpected:
        # F13: action_loss_only=True では learnable_to_wan_proj が構築されないので
        # unexpected 側に出る。これは欠損ではないので警告に留める。
        logger.info(
            "checkpoint has %d keys not present in the model (expected for WAN-free "
            "inference builds): %s",
            len(unexpected),
            unexpected[:10],
        )
    return len(state)


def extract_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    """LIBERO の観測から 8 次元 state を作る。

    ``eef_pos[3] + quat2axisangle(eef_quat)[3] + gripper_qpos[2]``。
    上流 ``model2libero_interface.py:97-115`` と同じ構成。
    """
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] != 3 or eef_quat.shape[0] != 4 or gripper_qpos.shape[0] != 2:
        raise ValueError(
            "unexpected LIBERO state shapes: "
            f"eef_pos={eef_pos.shape}, eef_quat={eef_quat.shape}, "
            f"gripper_qpos={gripper_qpos.shape}"
        )
    axisangle = quat2axisangle(eef_quat)
    return np.concatenate([eef_pos, axisangle, gripper_qpos], axis=0).astype(np.float32)


# =========================================================================== #
# 4. 推論ランタイム
# =========================================================================== #
class InternVLARuntime:
    """提出物の推論本体。

    前処理は**学習の transform 順序と 1:1 対応**させる（計画 §P5-27）:

    ===================================  ==========================================
    学習                                 推論
    ===================================  ==========================================
    (dataset) ``[T, C, 256, 256]``       env obs ``(128, 128, 3) uint8``
    ``RenderDownsampleFn`` -> 128        （既に 128。恒等）
    ``ResizeImagesWithPadFn(224)``       ``resize_with_pad(224, 224)`` <- **必須**
    ``RemapImageKeyTransformFn``         ``image0`` / ``image1`` に直接詰める + mask
    ``NormalizeTransformFn``             ``NormalizeTransformFn([OBS_STATE])``
    ``...ChatProcessor(mode="train")``   同 ``(mode="eval")``
    ===================================  ==========================================
    """

    def __init__(self, ckpt_dir: str | Path, vlm_dir: str | Path, cfg: dict[str, Any]) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.vlm_dir = Path(vlm_dir)
        self.cfg = dict(cfg)

        # --- 設定の検証は「モデルをロードする前」に済ませる（起動直後に落とす）---
        self.orientation = resolve_image_orientation(self.cfg)
        self.gripper = resolve_gripper(self.cfg)
        self.chunking = resolve_chunking(self.cfg)
        resize_cfg = self.cfg.get("resize") or {}
        self.resize_h = int(resize_cfg.get("height", 224))
        self.resize_w = int(resize_cfg.get("width", 224))
        self.num_inference_steps = int(self.cfg.get("num_inference_steps", 10))

        self.instruction = ""
        self._prev_chunk_norm: np.ndarray | None = None
        self._chunk_cursor = 0
        self._current_chunk_env: np.ndarray | None = None
        #: 現エピソードで 1 度でもチャンクを引いたか。**「新しいチャンクを引くか」の
        #: 判定はこのフラグとカーソルだけで決める。** ``_current_chunk_env`` の有無で
        #: 判定すると、env 空間のチャンクを持たない ensemble モードで毎ステップ推論に
        #: なり ``replan_steps`` が死ぬ。
        self._has_chunk = False
        self._ensembler = None
        #: ``train_config.json`` から解決した正規化モードのキャッシュ（ホットパスで再読しない）。
        self._norm_mode_cache: str | None = None

        self._load_time_sec = 0.0
        self._loaded = False
        self.policy = None
        self.processor = None
        self.state_normalizer = None
        self.action_stats: dict[str, np.ndarray] = {}
        self.chunk_size = 50
        self.action_dim = 32

    # ------------------------------------------------------------------ #
    # ロード
    # ------------------------------------------------------------------ #
    def load(self) -> float:
        """モデルと前処理を構築する。所要秒数を返す（120 秒制限の監視用）。"""
        if self._loaded:
            return self._load_time_sec

        started = time.time()

        import torch
        from lerobot.dataset_schemas import get_schema
        from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (
            InternVLAA15Config,
        )
        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
            InternVLAA15ChatProcessorTransformFn,
        )
        from lerobot.transforms.core import NormalizeTransformFn
        from lerobot.utils.constants import ACTION, OBS_STATE

        config = InternVLAA15Config.from_pretrained(str(self.ckpt_dir))
        config.vlm_model_name_or_path = str(self.vlm_dir)

        with fast_token_embedding_resize(), vlm_built_from_config_only(self.vlm_dir):
            self.policy = InternVLAA15Policy.from_pretrained(
                config=config, pretrained_name_or_path=str(self.ckpt_dir)
            )

        checkpoint_file = self.ckpt_dir / "model.safetensors"
        assert_checkpoint_covers_model(self.policy, checkpoint_file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(device)
        self.policy.eval()
        self.device = device

        self.chunk_size = int(config.chunk_size)
        self.action_dim = int(config.max_action_dim)

        robot_type = os.environ.get("IVLA_ROBOT_TYPE", "libero_combined")
        schema = get_schema(robot_type)
        stats = self._load_stats(robot_type)

        self.state_normalizer = NormalizeTransformFn(
            selected_keys=[OBS_STATE],
            norm_stats={OBS_STATE: stats[OBS_STATE]},
            mode=self._resolved_norm_mode(),
        )
        self.action_stats = stats[ACTION]
        self.processor = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=str(self.vlm_dir),
            max_length=int(config.max_prompt_length),
            tokenize_state=bool(config.tokenize_state),
            max_state_dim=int(config.max_state_dim),
            use_fast_action_tokens=False,
            mode="eval",
            action_mode=getattr(schema, "action_mode", "joint"),
        )

        # chunk_size / action_dim が確定してから ensembler を作り直す。
        self._sync_ensembler()
        # chunk_size が確定して初めて replan_steps との大小が判定できる。
        self._warn_if_replan_exceeds_chunk()

        self._load_time_sec = time.time() - started
        self._loaded = True
        logger.info("InternVLARuntime loaded in %.1f s on %s", self._load_time_sec, device)
        return self._load_time_sec

    def replan_interval(self) -> int:
        """実際にチャンクを引き直す間隔 ``min(replan_steps, chunk_size)``。

        チャンクは自分の長さを超えて引き延ばせないので、``replan_steps`` が
        ``chunk_size`` より大きくても引き直しは ``chunk_size`` ごとになる
        （:meth:`get_action` の ``need_new_chunk`` と同じ式）。推論回数の期待値や
        エピソード予算はこの値で計算しないと実装と食い違う
        （``verify_inference.expected_inference_calls`` が参照する）。
        """
        return min(int(self.chunking["replan_steps"]), int(self.chunk_size))

    def _warn_if_replan_exceeds_chunk(self) -> bool:
        """``replan_steps > chunk_size`` を **warning** で知らせる（エラーにはしない）。

        実行としては安全側（``chunk_size`` ごとに引き直す）に倒れるので起動は止めない。
        ただし「50 step のチャンクに replan=60 を設定した」は設定ミスの可能性が高く、
        A/B の表で replan=60 と replan=50 が同じ数字になる理由がここにあるので、
        黙って丸めずログに残す。

        Returns:
            警告を出したかどうか。
        """
        replan = int(self.chunking["replan_steps"])
        chunk_size = int(self.chunk_size)
        if replan <= chunk_size:
            return False
        logger.warning(
            "chunking.replan_steps=%d exceeds the model chunk_size=%d; "
            "the runtime will re-plan every %d steps instead. "
            "Set replan_steps <= chunk_size to make the configuration say what it does.",
            replan,
            chunk_size,
            chunk_size,
        )
        return True

    def _resolved_norm_mode(self) -> str:
        """正規化モードを**1 回だけ**解決してキャッシュする。

        ``_denormalize`` は ensemble モードでは 1 action ごとに呼ばれる。そこから
        毎回 :meth:`_norm_mode` を呼ぶと ``train_config.json`` を推論ホットパスで
        パースし続けることになるので、解決済みの値を使い回す。
        """
        if self._norm_mode_cache is None:
            self._norm_mode_cache = self._norm_mode()
        return self._norm_mode_cache

    def _norm_mode(self) -> str:
        """学習時の正規化モード。``train_config.json`` から読む（計画 U7）。

        **ホットパスから直接呼ばないこと。** :meth:`_resolved_norm_mode` を通す。
        """
        train_config = self.ckpt_dir / "train_config.json"
        if not train_config.is_file():
            train_config = self.ckpt_dir.parent / "train_config.json"
        if train_config.is_file():
            payload = json.loads(train_config.read_text(encoding="utf-8"))
            for transform in (
                payload.get("dataset", {}).get("data_transforms", {}).get("inputs", [])
            ):
                if isinstance(transform, dict) and transform.get("type") == "normalize":
                    return str(transform.get("mode", "mean_std"))
        logger.warning(
            "could not read the normalization mode from train_config.json; falling back to mean_std"
        )
        return "mean_std"

    def _load_stats(self, robot_type: str) -> dict[str, dict[str, np.ndarray]]:
        from lerobot.utils.constants import ACTION, OBS_STATE

        stats_file = self.ckpt_dir / "stats.json"
        if not stats_file.is_file():
            raise FileNotFoundError(f"stats.json not found next to the checkpoint: {stats_file}")
        raw = json.loads(stats_file.read_text(encoding="utf-8"))

        if robot_type in raw:
            selected = raw[robot_type]
        elif len(raw) == 1:
            selected = next(iter(raw.values()))
        elif OBS_STATE in raw:
            selected = raw
        else:
            raise KeyError(
                f"cannot pick stats for robot_type={robot_type!r}; available keys={list(raw)[:20]}"
            )

        out: dict[str, dict[str, np.ndarray]] = {}
        for key in (OBS_STATE, ACTION):
            if key not in selected:
                raise KeyError(f"stats.json has no entry for {key!r} (keys={list(selected)[:20]})")
            out[key] = {
                name: np.asarray(value, dtype=np.float32)
                for name, value in selected[key].items()
                if isinstance(value, (list, tuple))
            }
        return out

    # ------------------------------------------------------------------ #
    # chunking モードの切替（**唯一の入口**）
    # ------------------------------------------------------------------ #
    def _sync_ensembler(self) -> None:
        """``chunking["mode"]`` に合わせて ensembler を構築 / 破棄する。

        ``mode`` を書き換える経路は必ずここを通す。生の dict を書き換えるだけだと
        ``_ensembler`` が None のまま ensemble モードを名乗ることになり、
        **実体は none 経路なのに ensemble として計測する**というズレが起きる。
        """
        if self.chunking["mode"] != "ensemble":
            self._ensembler = None
            return

        from chunk_blending import TemporalEnsembler

        current = self._ensembler
        # ``newest_dims`` も比較する。現状 ``exclude_dims`` は起動後に変わらないので
        # 到達不能だが、A/B で exclude_dims を振れるようにしたときに「古い除外次元の
        # ensembler を使い回す」事故を構造的に防ぐ（差分は再構築で吸収される）。
        if (
            current is not None
            and current.chunk_size == self.chunk_size
            and current.action_dim == self.action_dim
            and current.m == float(self.chunking["ensemble_m"])
            and tuple(current.newest_dims) == tuple(self.chunking["exclude_dims"])
        ):
            current.reset()
            return

        self._ensembler = TemporalEnsembler(
            chunk_size=self.chunk_size,
            action_dim=self.action_dim,
            m=self.chunking["ensemble_m"],
            newest_dims=self.chunking["exclude_dims"],
        )

    def set_chunking_mode(self, mode: str) -> str:
        """chunking モードを切り替える。**``chunking["mode"]`` を書く唯一の入口。**

        ensembler の構築 / 破棄とチャンク状態のリセットをまとめて行う。
        ``verify_inference.py`` の ``--chunking`` もここを通る（手順書 B8:
        自己チェックと本番を同じ経路に通す）。生の dict を直接書き換えると、
        ``mode`` と ``_ensembler`` がずれて別モードの数値を測ることになる。

        Args:
            mode: ``"none"`` / ``"rtc"`` / ``"ensemble"``。

        Returns:
            設定後のモード。

        Raises:
            ValueError: 未知のモード。
        """
        mode = str(mode).lower()
        if mode not in ("none", "rtc", "ensemble"):
            raise ValueError(
                f"unknown chunking mode {mode!r}; expected 'none', 'rtc' or 'ensemble'"
            )
        self.chunking["mode"] = mode
        self._sync_ensembler()
        self.reset(self.instruction)
        return mode

    # ------------------------------------------------------------------ #
    # 推論
    # ------------------------------------------------------------------ #
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時。チャンクキャッシュと ensembler を消す。"""
        self.instruction = str(instruction or "")
        self._prev_chunk_norm = None
        self._chunk_cursor = 0
        self._current_chunk_env = None
        self._has_chunk = False
        if self._ensembler is not None:
            self._ensembler.reset()

    def _obs_to_sample(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        """観測を学習と同じ形のサンプルに変換する。

        **F6 対策**: ``resize_with_pad(224, 224)`` を無条件で通す。上流の推論
        バックエンドはここを no-op にしてしまっているので真似しない。学習側と
        同一の関数（``lerobot.transforms.utils.resize_with_pad``）を呼ぶこと。
        """
        import torch
        from lerobot.transforms.utils import resize_with_pad
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        for key in (OBS_AGENTVIEW, OBS_WRIST):
            if key not in obs:
                raise KeyError(f"observation is missing {key!r}; got {sorted(obs)}")

        images = {
            "image0": orient_image(obs[OBS_AGENTVIEW], self.orientation["agentview"]),
            "image1": orient_image(obs[OBS_WRIST], self.orientation["wrist"]),
        }

        sample: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(extract_state(obs)),
            "task": self.instruction,
        }

        for name, array in images.items():
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            else:
                tensor = tensor.float()
                if float(tensor.max()) > 1.0:
                    tensor = tensor / 255.0
            tensor = tensor.permute(2, 0, 1).contiguous()  # HWC -> CHW
            # ---- F6 対策: 無条件のリサイズ ----
            tensor = resize_with_pad(tensor, self.resize_h, self.resize_w)
            sample[f"{OBS_IMAGES}.{name}"] = tensor
            sample[f"{OBS_IMAGES}.{name}_mask"] = True

        # 3 枚目のカメラは存在しないので、上流 RemapImageKeyTransformFn と同じく
        # 白画像 + mask False で埋める。
        sample[f"{OBS_IMAGES}.image2"] = torch.ones_like(sample[f"{OBS_IMAGES}.image0"])
        sample[f"{OBS_IMAGES}.image2_mask"] = False

        sample = self.state_normalizer(sample)
        sample = self.processor(sample)
        return sample

    def _sample_to_inputs(self, sample: dict[str, Any]) -> dict[str, Any]:
        import torch

        inputs: dict[str, Any] = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
                continue
            if isinstance(value, bool) or not isinstance(value, torch.Tensor):
                continue
            inputs[key] = value.unsqueeze(0).to(self.device)
        return inputs

    def _predict_chunk_normalized(self, sample: dict[str, Any], guidance) -> np.ndarray:
        """正規化空間のチャンク ``[chunk, D]`` を返す。"""
        import torch

        from chunk_blending import sample_actions_guided

        inputs = self._sample_to_inputs(sample)
        model = self.policy.model

        with torch.no_grad():
            if guidance is None:
                chunk = self.policy.predict_action_chunk(inputs)
            else:
                chunk = sample_actions_guided(
                    model,
                    pixel_values=inputs["observation.pixel_values"],
                    image_grid_thw=inputs["observation.image_grid_thw"],
                    lang_tokens=inputs["observation.input_ids"],
                    lang_masks=inputs["observation.attention_mask"],
                    state=inputs["observation.state"],
                    fast_token_mask=inputs.get("observation.fast_token_mask"),
                    num_steps=self.num_inference_steps,
                    guidance=guidance,
                )
        return chunk.detach().float().cpu().numpy()[0]

    def _action_stat(self, name: str, *, required_by: str) -> np.ndarray:
        """action の統計を取り出す。**無ければ別のキーで代用せず例外にする。**

        ここで黙って ``min`` / ``max`` に落ちると、``q01_q99`` の逆正規化が
        ``[q01, q99] ⊂ [min, max]`` の比だけ系統的に過大スケールになる
        （計画 U7 の「行動のスケールが桁で違う」）。

        Args:
            name: ``stats.json`` の統計キー。
            required_by: **要求元**を説明する文字列。``min`` / ``max`` は
                :meth:`_to_env_action` の clip が正規化モードと**無関係に**要求する
                ので、ここを一律「正規化モードが要求している」と書くと、
                ``mean_std`` の checkpoint で min/max が欠けたときに
                「mean_std が min を要求している」という**嘘の説明**になる。
        """
        stats = self.action_stats
        if name not in stats:
            raise KeyError(
                f"stats.json has no {name!r} entry for the action key "
                f"(available: {sorted(stats)}). It is required by {required_by}; "
                "refusing to substitute another key."
            )
        return np.asarray(stats[name], dtype=np.float32).reshape(-1)

    def _denormalize(self, chunk_norm: np.ndarray) -> np.ndarray:
        """正規化空間 -> env の action 空間。

        上流 ``NormalizeTransformFn``（``transforms/core.py:296-313``）の**厳密な逆**
        を取る。eps も上流と同じ ``1e-6``::

            mean_std : x * (std + eps) + mean
            min_max  : (x + 1) / 2 * (max - min + eps) + min
            q01_q99  : (x + 1) / 2 * (q99 - q01 + eps) + q01

        .. note:: ``q01_q99`` は **q01 / q99 を使う**。``min`` / ``max`` で代用すると
           ``[q01, q99] ⊂ [min, max]`` の比だけ行動が過大になる（サイレント失敗）。

        .. note:: 上流 ``base_backend.denormalize_actions`` の ``min_max`` 分岐は
           ``clip(x, 0, 1) * (high - low) + low`` で、正規化側が ``[-1, 1]`` へ
           写しているのと整合しない。ここでは**正規化側の厳密な逆**を採る。
        """
        arr = np.asarray(chunk_norm, dtype=np.float32)
        eps = np.float32(1e-6)

        mode = self._resolved_norm_mode()
        required_by = f"denormalization in mode {mode!r}"
        if mode == "mean_std":
            offset = self._action_stat("mean", required_by=required_by)
            scale = self._action_stat("std", required_by=required_by) + eps
        elif mode == "min_max":
            offset = self._action_stat("min", required_by=required_by)
            scale = self._action_stat("max", required_by=required_by) - offset + eps
        elif mode == "q01_q99":
            offset = self._action_stat("q01", required_by=required_by)
            scale = self._action_stat("q99", required_by=required_by) - offset + eps
        else:
            raise ValueError(f"unknown normalization mode {mode!r}")

        dim = int(offset.shape[0])
        if arr.shape[-1] < dim:
            raise ValueError(f"model action dim {arr.shape[-1]} < stats action dim {dim}")
        normalized = arr[..., :dim]

        if mode == "mean_std":
            denorm = normalized * scale + offset
        else:
            denorm = (normalized + 1.0) / 2.0 * scale + offset
        return denorm.astype(np.float32)

    def _to_env_action(self, a7: np.ndarray) -> np.ndarray:
        """クリップ + グリッパの二値化。``(7,) float32`` を返す。

        クリップ範囲は上流 ``base_backend.postprocess_actions`` と同じく生の action
        空間の ``min`` / ``max``（正規化モードとは独立）。**したがって ``mean_std`` や
        ``q01_q99`` の checkpoint でも ``stats.json`` に ``min`` / ``max`` が要る。**
        欠けている場合の例外メッセージは「clip が要求している」と言うこと
        （「正規化モードが要求している」は嘘になる）。
        """
        arr = np.asarray(a7, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 7:
            raise ValueError(f"action must have at least 7 dims, got {arr.shape[0]}")
        clip_reason = (
            "clipping the action to the raw action range "
            "(required regardless of the normalization mode)"
        )
        low = self._action_stat("min", required_by=clip_reason)[: arr.shape[0]]
        high = self._action_stat("max", required_by=clip_reason)[: arr.shape[0]]
        arr = np.clip(arr, low, high).astype(np.float32)

        out = arr[:7].copy()
        out[6] = binarize_gripper(float(out[6]), self.gripper)
        if not np.all(np.isfinite(out)):
            raise ValueError(f"non-finite action produced: {out}")
        return out.astype(np.float32)

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測 1 個から action ``(7,) float32`` を返す。"""
        from chunk_blending import build_guidance

        self.load()
        mode = self.chunking["mode"]

        if mode == "ensemble" and self._ensembler is None:
            # 生の dict で mode だけ書き換えると ensembler が無いまま ensemble を
            # 名乗ることになる。黙って none 経路に落ちるくらいなら落とす。
            raise RuntimeError(
                "chunking.mode='ensemble' but no TemporalEnsembler was built. "
                "Switch modes through InternVLARuntime.set_chunking_mode() "
                "instead of assigning to runtime.chunking['mode']."
            )

        # **``_current_chunk_env`` の有無を条件に入れないこと。** ensemble モードは
        # env 空間のチャンクを保持しないので、入れると毎ステップ推論になって
        # ``replan_steps`` が完全に死ぬ。判定はカーソルとフラグだけで決める。
        need_new_chunk = not self._has_chunk or self._chunk_cursor >= self.replan_interval()

        if need_new_chunk:
            sample = self._obs_to_sample(obs)
            guidance = None
            if mode == "rtc":
                guidance = build_guidance(
                    self._prev_chunk_norm,
                    # 境界判定と同じ ``replan_interval()`` を渡す。生の ``replan`` を
                    # 渡すと「実際に経過した env ステップ数」と「target をずらす量」が
                    # 食い違う（``replan > chunk_size`` のときだけ差が出て、そこは
                    # 双方 None になるので現状は無害だが、同じ量の定義を 2 つ持たない）。
                    replan_steps=self.replan_interval(),
                    chunk_size=self.chunk_size,
                    action_dim=self.action_dim,
                    w_max=self.chunking["w_max"],
                    schedule=self.chunking["schedule"],
                    exclude_dims=self.chunking["exclude_dims"],
                )
            chunk_norm = self._predict_chunk_normalized(sample, guidance)
            self._prev_chunk_norm = chunk_norm
            self._chunk_cursor = 0
            self._has_chunk = True

            if mode == "ensemble":
                self._ensembler.push(chunk_norm)
                self._current_chunk_env = None
            else:
                self._current_chunk_env = self._denormalize(chunk_norm)

        if mode == "ensemble":
            action_norm = self._ensembler.pop()
            action = self._denormalize(action_norm[None, :])[0]
        else:
            if self._current_chunk_env is None:
                raise RuntimeError("no action chunk is cached; this is a scheduling bug")
            action = self._current_chunk_env[self._chunk_cursor]

        self._chunk_cursor += 1
        return self._to_env_action(action)


# =========================================================================== #
# 5. 共通入口
# =========================================================================== #
def build_submission_runtime(config_path: str | Path | None = None) -> InternVLARuntime:
    """``policy_server.py`` と ``verify_inference.py`` が**共通で**通る入口（B8）。

    片方だけ別経路にすると、自己チェックが本番と違うものを測ることになる。
    """
    config = load_runtime_config(config_path)
    apply_config_env(config)

    runtime_dir = Path(config["_path"]).parent
    bootstrap(runtime_dir)

    kind = str(os.environ.get("IVLA_RUNTIME") or config.get("runtime", "standard")).lower()
    if kind != "standard":
        raise ValueError(f"unknown runtime {kind!r}; only 'standard' is implemented")

    ckpt_dir = Path(
        os.environ.get("IVLA_CKPT_DIR") or (runtime_dir / "model_weights" / "checkpoint")
    )
    vlm_dir = Path(os.environ.get("IVLA_VLM_DIR") or (runtime_dir / "model_weights" / "vlm"))

    return InternVLARuntime(ckpt_dir=ckpt_dir, vlm_dir=vlm_dir, cfg=config)
