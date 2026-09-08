"""手元（GPU 無し Windows）で上流 lerobot を import するための最小セットアップ。

上流 ``InternVLA-A-series`` は **pip install しない**。``src`` を ``sys.path`` に足して
読むだけにする（上流を 1 バイトも変更しないため / 計画 §3・§4.2）。

その代わり、上流 ``lerobot`` の import チェーンが引く重い依存が手元に無いので、
**必要最小限だけ** ``sys.modules`` にスタブを入れる。スタブで誤魔化している箇所は
``STUBBED_MODULES`` に列挙し、``--stub-report`` 相当の情報を
``test_environment.py::test_stub_inventory_is_declared`` が印字する。

スタブ対象（いずれも「import されるだけで、テスト対象のロジックには一切使われない」もの）:

======================  ====================================================
モジュール              なぜ import されるか / なぜスタブで安全か
======================  ====================================================
pandas / pyarrow        ``lerobot.datasets.lerobot_dataset`` の parquet I/O。
                        本テストはデータセットを読まない。
datasets                同上（HF datasets）。
av                      ``lerobot.datasets.video_utils`` の動画デコード。
                        本テストは動画を読まない。
imageio                 同上。
accelerate              ``lerobot.utils.utils``。分散学習用。手元では未使用。
einops / diffusers      ``...internvla_a1_5.transform_internvla_a1_5`` 経由の
                        WAN 分岐。モデルを構築しないので未使用。
======================  ====================================================

``transformers`` は**実物**を使う（スタブしない）。ただし transformers は import 時に
``accelerate`` の *パッケージメタデータ* を要求するので、**スタブを入れる前に**
先に import しておく必要がある。順序を間違えると
``PackageNotFoundError: accelerate`` になる。
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# パス解決
# --------------------------------------------------------------------------- #
TESTS_DIR = Path(__file__).resolve().parent
IVLA_DIR = TESTS_DIR.parent
PARC_ROOT = IVLA_DIR.parent.parent

#: 上流リポジトリ（読み取り専用）。環境変数 IVLA_REPO で上書きできる。
_DEFAULT_UPSTREAM = PARC_ROOT.parent / "InternVLA" / "InternVLA-A-series"


def _upstream_repo() -> Path | None:
    import os

    env = os.environ.get("IVLA_REPO")
    candidates = [Path(env).expanduser() for env in ([env] if env else [])]
    candidates.append(_DEFAULT_UPSTREAM)
    for candidate in candidates:
        if (candidate / "src" / "lerobot" / "__init__.py").is_file():
            return candidate.resolve()
    return None


UPSTREAM_REPO = _upstream_repo()
UPSTREAM_SRC = (UPSTREAM_REPO / "src") if UPSTREAM_REPO else None

#: 実際にスタブしたモジュール名（テストから参照して報告する）
STUBBED_MODULES: list[str] = []

#: import を通すために必要なスタブの一覧（上のテーブルと対応）
_STUB_SPECS: tuple[str, ...] = (
    "pandas",
    "pyarrow",
    "pyarrow.parquet",
    "pyarrow.dataset",
    "datasets",
    "datasets.features",
    "datasets.features.features",
    "datasets.table",
    "datasets.utils",
    "datasets.utils.logging",
    "av",
    "imageio",
    "accelerate",
    "einops",
    "diffusers",
    "diffusers.configuration_utils",
    "diffusers.models",
    "diffusers.models.modeling_utils",
)


class _LazyMeta(type):
    """属性アクセスと呼び出しを無限に受け流すダミー型。"""

    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _LazyMeta(f"{cls.__name__}.{name}", (object,), {})
        setattr(cls, name, sub)
        return sub

    def __call__(cls, *args, **kwargs):
        return _LazyMeta(f"{cls.__name__}()", (object,), {})


class _LazyStubModule(types.ModuleType):
    """未インストールのパッケージを import だけ通すためのダミーモジュール。"""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        obj = _LazyMeta(name, (object,), {"__module__": self.__name__})
        setattr(self, name, obj)
        return obj


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _install_stub(dotted: str) -> None:
    parts = dotted.split(".")
    for index in range(len(parts)):
        name = ".".join(parts[: index + 1])
        if name in sys.modules:
            continue
        module = _LazyStubModule(name)
        module.__path__ = []  # パッケージとして振る舞わせる
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        module.__file__ = f"<parc-test-stub {name}>"
        module.__version__ = "0.0.0+parc-test-stub"
        sys.modules[name] = module
        if index > 0:
            setattr(sys.modules[".".join(parts[:index])], parts[index], module)
        STUBBED_MODULES.append(name)


def _bootstrap_upstream() -> None:
    """上流 src を sys.path に足し、足りない依存をスタブする。"""
    if UPSTREAM_SRC is None:
        return

    # transformers は「実物」を先に import する。スタブより後だと
    # transformers の依存チェックが偽の accelerate を見て落ちる。
    if _module_available("transformers"):
        import transformers  # noqa: F401

    for dotted in _STUB_SPECS:
        top = dotted.split(".")[0]
        if _module_available(top) and top not in STUBBED_MODULES:
            continue  # 実物があるならそれを使う
        _install_stub(dotted)

    src = str(UPSTREAM_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


# overlay パッケージ（`lerobot_policy_parc`）を import できるようにする
if str(IVLA_DIR) not in sys.path:
    sys.path.insert(0, str(IVLA_DIR))

_bootstrap_upstream()


# --------------------------------------------------------------------------- #
# skip ヘルパ
# --------------------------------------------------------------------------- #
def _lerobot_import_error() -> str | None:
    if UPSTREAM_SRC is None:
        return (
            "upstream InternVLA-A-series not found "
            f"(looked for {_DEFAULT_UPSTREAM}; set IVLA_REPO)"
        )
    try:
        import lerobot.transforms.core  # noqa: F401
    except Exception as exc:  # pragma: no cover - 環境依存
        return f"upstream lerobot import failed: {type(exc).__name__}: {exc}"
    return None


LEROBOT_SKIP_REASON = _lerobot_import_error()

requires_lerobot = pytest.mark.skipif(
    LEROBOT_SKIP_REASON is not None,
    reason=f"needs upstream lerobot on sys.path -- {LEROBOT_SKIP_REASON}",
)


def _cv2_import_error() -> str | None:
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


CV2_SKIP_REASON = _cv2_import_error()

# cv2 は「あれば INTER_AREA との同値も確かめる」任意依存（計画 §7.1）。
# 無い環境では該当アサーションだけを skip する。box == interpolate(antialias=False)
# の同値性は cv2 なしでも必ず検証される。
requires_cv2 = pytest.mark.skipif(
    CV2_SKIP_REASON is not None,
    reason=f"cv2 is an optional dependency for the INTER_AREA cross-check -- {CV2_SKIP_REASON}",
)


@pytest.fixture()
def offline_processors(monkeypatch):
    """**テスト専用スタブ**: HF Hub からのダウンロードを伴う ``__post_init__`` を無効化する。

    上流の 3 つの transform は構築するだけで Hub にアクセスする:

    - ``InternVLAA15ChatProcessorTransformFn.__post_init__``
      -> ``Qwen3VLProcessor.from_pretrained("Qwen/Qwen3.5-2B")``
      （``transform_internvla_a1_5.py:88-93``）
    - ``InternVLAA15VQAProcessorTransformFn.__post_init__`` -> 同上（``:276``）
    - ``FASTInternVLAA15ActionTokenizerTransformFn.__post_init__``
      -> ``AutoProcessor.from_pretrained("physical-intelligence/fast")`` ほか（``:385``）

    ``InternVLAA15DatasetConfig.__post_init__`` はこれらを**必ず**構築するので、
    ネットワークとモデルキャッシュが無い手元では ``DatasetConfig`` を作れない。

    このフィクスチャが潰すのは「重みとトークナイザの取得」だけである。テスト対象は
    **transform チェーンの並び順と draccus 往復**なので、processor の中身は結果に
    影響しない。``__call__`` を通す種類のテストではこのフィクスチャを使わない
    （使うとトークナイズが動かず、無意味に緑になるため）。
    """
    from lerobot.policies.internvla_a1_5 import transform_internvla_a1_5 as tfm

    def _skip_download(self):
        self.processor = None
        self.action_tokenizer = None
        self.qwen35_tokenizer = None
        self.vision_start_token_id = -1
        self.vision_end_token_id = -1
        self.image_token_id = -1
        # 上流 __post_init__ の「ダウンロードと無関係な副作用」だけは再現する。
        # assistant_end_tokens は `list[int] = None` と宣言されており、
        # __post_init__ が None を実値で埋める（transform_internvla_a1_5.py:400-401）。
        # 埋めないと draccus.encode が "Couldn't encode None" で落ちる。
        if getattr(self, "assistant_end_tokens", "missing") is None:
            self.assistant_end_tokens = [248045, 74455, 198, 248068, 271, 248069, 271]

    for cls_name in (
        "InternVLAA15ChatProcessorTransformFn",
        "InternVLAA15VQAProcessorTransformFn",
        "FASTInternVLAA15ActionTokenizerTransformFn",
    ):
        cls = getattr(tfm, cls_name)
        monkeypatch.setattr(cls, "__post_init__", _skip_download, raising=True)

    # ここまで来ても Hub を触ろうとしたら、待たずに即失敗させる。
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    yield


@pytest.fixture(scope="session")
def ivla_dir() -> Path:
    return IVLA_DIR


@pytest.fixture(scope="session")
def parc_root() -> Path:
    return PARC_ROOT


@pytest.fixture(scope="session")
def upstream_repo() -> Path:
    if UPSTREAM_REPO is None:
        pytest.skip("upstream InternVLA-A-series not found; set IVLA_REPO")
    return UPSTREAM_REPO
