"""サイレント失敗ガードがコード上に存在することの機械検証（SC-3 の Evaluator 項目）。

推論経路そのものは GPU とチェックポイントが要るので手元では動かせない。しかし
「ガードが**入っていること**」は AST とソースの検査で確認できる。ここが抜けると
エラーは出ないまま精度だけ落ちる（移植手順書 §6）ので、回帰テストにしておく。

SC-3 の Evaluator が求める項目:

- [x] 推論経路に ``resize_with_pad(224, 224)`` が**無条件で**入っている（F6）
- [x] ``install_vendored_lerobot`` に「別 lerobot が import 済みなら落とす」ガード
- [x] ``assert_checkpoint_covers_model()`` が起動パスに入っている
- [x] ``mean_resizing=False`` のコンテキストマネージャ
- [x] 画像の向き・グリッパ規約が ``runtime_config.json`` 由来で、コードに直書きされていない
- [x] ``policy_server.py`` と ``verify_inference.py`` が同じ ``build_submission_runtime()`` を通る
- [x] ``TBD`` の番人（``_reject_tbd``）自体が働いている（列挙チェックに肩代わりさせない）
- [x] ``chunking`` の設定（``schedule`` / ``exclude_dims`` を含む）が**モデルロード前**に検証される
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

INFERENCE_DIR = Path(__file__).resolve().parent.parent / "inference"
RUNTIME_PY = INFERENCE_DIR / "internvla_runtime.py"
SERVER_PY = INFERENCE_DIR / "policy_server.py"
VERIFY_PY = INFERENCE_DIR / "verify_inference.py"

sys.path.insert(0, str(INFERENCE_DIR))


@pytest.fixture(scope="module")
def runtime_tree() -> ast.Module:
    return ast.parse(RUNTIME_PY.read_text(encoding="utf-8"), filename=str(RUNTIME_PY))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    found = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            found.append(child)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            found.append(child)
    return found


def _enclosing_conditions(tree: ast.AST, target: ast.AST) -> list[ast.AST]:
    """``target`` を囲む ``if`` / ``try`` の一覧（無条件かどうかの判定用）。"""
    stack: list[ast.AST] = []
    result: list[ast.AST] = []

    def visit(node: ast.AST) -> bool:
        if node is target:
            result.extend(stack)
            return True
        pushed = isinstance(node, (ast.If, ast.Try, ast.While))
        if pushed:
            stack.append(node)
        for child in ast.iter_child_nodes(node):
            if visit(child):
                return True
        if pushed:
            stack.pop()
        return False

    visit(tree)
    return result


# --------------------------------------------------------------------------- #
# F6: 推論経路の resize_with_pad が無条件であること
# --------------------------------------------------------------------------- #
def test_obs_to_sample_calls_resize_with_pad(runtime_tree):
    """``_obs_to_sample`` が ``resize_with_pad`` を呼んでいること（F6 対策の本体）。"""
    fn = _find_function(runtime_tree, "_obs_to_sample")
    calls = _calls_named(fn, "resize_with_pad")
    assert calls, (
        "_obs_to_sample must call resize_with_pad. The upstream inference backend "
        "silently skips the resize (F6): its ResizeImagesWithPadFn is never hydrated, "
        "so mapping is empty and __call__ loops zero times."
    )


def test_resize_with_pad_call_is_unconditional(runtime_tree):
    """``resize_with_pad`` の呼び出しが ``if`` / ``try`` に囲まれていないこと。

    条件付きにすると「条件が偽のときだけ静かに壊れる」状態に戻る。
    """
    fn = _find_function(runtime_tree, "_obs_to_sample")
    for call in _calls_named(fn, "resize_with_pad"):
        conditions = _enclosing_conditions(fn, call)
        assert not conditions, (
            "the resize_with_pad call must be unconditional, but it is nested inside "
            f"{[type(c).__name__ for c in conditions]}"
        )


def test_resize_with_pad_comes_from_upstream_lerobot():
    """学習側と**同一関数**を呼ぶこと（自前で書き直さない）。"""
    source = RUNTIME_PY.read_text(encoding="utf-8")
    assert "from lerobot.transforms.utils import resize_with_pad" in source


def test_resize_target_comes_from_the_config():
    """224 をコードに直書きせず ``runtime_config.json`` の ``resize`` から取ること。"""
    source = RUNTIME_PY.read_text(encoding="utf-8")
    assert "self.resize_h" in source and "self.resize_w" in source
    assert 'resize_cfg.get("height", 224)' in source
    assert 'resize_cfg.get("width", 224)' in source


# --------------------------------------------------------------------------- #
# B1: 別 lerobot が import 済みなら落とす
# --------------------------------------------------------------------------- #
def test_install_vendored_lerobot_has_already_imported_guard(runtime_tree):
    fn = _find_function(runtime_tree, "install_vendored_lerobot")
    source = ast.unparse(fn)
    assert "'lerobot' in sys.modules" in source or '"lerobot" in sys.modules' in source
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert raises, "install_vendored_lerobot must raise when a different lerobot is loaded"
    assert "already imported" in source


def test_install_vendored_lerobot_rejects_a_foreign_lerobot(monkeypatch, tmp_path):
    """実際に別の lerobot を装って呼び、例外になることを確かめる。"""
    import types

    from internvla_runtime import install_vendored_lerobot

    vendor = tmp_path / "vendor"
    (vendor / "lerobot").mkdir(parents=True)
    (vendor / "lerobot" / "__init__.py").write_text("", encoding="utf-8")

    foreign = tmp_path / "elsewhere" / "lerobot"
    foreign.mkdir(parents=True)
    foreign_init = foreign / "__init__.py"
    foreign_init.write_text("", encoding="utf-8")

    fake_module = types.ModuleType("lerobot")
    fake_module.__file__ = str(foreign_init)
    monkeypatch.setitem(sys.modules, "lerobot", fake_module)

    with pytest.raises(RuntimeError, match="already imported"):
        install_vendored_lerobot(vendor)


def test_install_vendored_lerobot_requires_the_package(tmp_path):
    from internvla_runtime import install_vendored_lerobot

    with pytest.raises(FileNotFoundError):
        install_vendored_lerobot(tmp_path)


# --------------------------------------------------------------------------- #
# B6: assert_checkpoint_covers_model が起動パスに入っている
# --------------------------------------------------------------------------- #
def test_assert_checkpoint_covers_model_is_called_on_load(runtime_tree):
    fn = _find_function(runtime_tree, "load")
    calls = _calls_named(fn, "assert_checkpoint_covers_model")
    assert calls, "InternVLARuntime.load() must call assert_checkpoint_covers_model()"

    conditions = _enclosing_conditions(fn, calls[0])
    assert not conditions, (
        "assert_checkpoint_covers_model() must run unconditionally on the startup path, "
        f"but it is nested inside {[type(c).__name__ for c in conditions]}"
    )


def test_assert_checkpoint_covers_model_raises_on_missing_keys(tmp_path):
    """欠損キーを検出して落ちること（``strict=False`` のサイレント失敗の検出）。"""
    import torch
    from safetensors.torch import save_file

    from internvla_runtime import assert_checkpoint_covers_model

    checkpoint = tmp_path / "model.safetensors"
    save_file({"a": torch.ones(2, 2)}, str(checkpoint))

    class _Policy:
        def __init__(self, state):
            self._state = state

        def state_dict(self):
            return self._state

    complete = _Policy({"a": torch.ones(2, 2)})
    assert assert_checkpoint_covers_model(complete, checkpoint) == 1

    incomplete = _Policy({"a": torch.ones(2, 2), "b": torch.zeros(3)})
    with pytest.raises(RuntimeError, match="missing 1 model parameters"):
        assert_checkpoint_covers_model(incomplete, checkpoint)


def test_assert_checkpoint_covers_model_tolerates_tied_weights(tmp_path):
    """tie された重みは保存時に片方が落ちる。欠損とみなさないこと。"""
    import torch
    from safetensors.torch import save_file

    from internvla_runtime import assert_checkpoint_covers_model

    checkpoint = tmp_path / "model.safetensors"
    save_file({"embed.weight": torch.ones(2, 2)}, str(checkpoint))

    shared = torch.ones(2, 2)

    class _Policy:
        def state_dict(self):
            return {"embed.weight": shared, "lm_head.weight": shared}

    assert assert_checkpoint_covers_model(_Policy(), checkpoint) == 2


# --------------------------------------------------------------------------- #
# B7: mean_resizing=False
# --------------------------------------------------------------------------- #
def test_fast_token_embedding_resize_sets_mean_resizing_false(runtime_tree):
    fn = _find_function(runtime_tree, "fast_token_embedding_resize")
    source = ast.unparse(fn)
    assert 'kwargs.setdefault(\'mean_resizing\', False)' in source or (
        'kwargs.setdefault("mean_resizing", False)' in source
    )
    assert "finally" in RUNTIME_PY.read_text(encoding="utf-8")


def test_fast_token_embedding_resize_restores_the_original():
    """コンテキストを抜けたら元に戻すこと（後続の挙動を汚さない）。"""
    from transformers.modeling_utils import PreTrainedModel

    from internvla_runtime import fast_token_embedding_resize

    original = PreTrainedModel.resize_token_embeddings
    with fast_token_embedding_resize():
        assert PreTrainedModel.resize_token_embeddings is not original
    assert PreTrainedModel.resize_token_embeddings is original


def test_load_uses_both_context_managers(runtime_tree):
    fn = _find_function(runtime_tree, "load")
    source = ast.unparse(fn)
    assert "fast_token_embedding_resize()" in source
    assert "vlm_built_from_config_only" in source


# --------------------------------------------------------------------------- #
# B8: 設定はコードに直書きしない
# --------------------------------------------------------------------------- #
def test_orientation_and_gripper_come_from_the_config(runtime_tree):
    """向きとグリッパ規約が ``runtime_config.json`` 由来であること。"""
    init = _find_function(runtime_tree, "__init__")
    source = ast.unparse(init)
    assert "resolve_image_orientation(self.cfg)" in source
    assert "resolve_gripper(self.cfg)" in source
    assert "resolve_chunking(self.cfg)" in source


def test_no_hardcoded_orientation_at_call_sites(runtime_tree):
    """``orient_image`` の呼び出しが**設定値**を渡していること。

    向きを文字列リテラルで渡している箇所があれば、それは事実上のコード直書きで
    ``runtime_config.json`` を無視することになる。AST で検出する。
    """
    calls = _calls_named(runtime_tree, "orient_image")
    assert calls, "the runtime must apply orient_image() to the incoming frames"

    for call in calls:
        for argument in call.args:
            assert not (
                isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            ), f"orient_image() must not receive a hardcoded orientation: {ast.unparse(call)}"

    unparsed = {ast.unparse(call) for call in calls}
    assert "orient_image(obs[OBS_AGENTVIEW], self.orientation['agentview'])" in unparsed
    assert "orient_image(obs[OBS_WRIST], self.orientation['wrist'])" in unparsed


#: ``_reject_tbd`` **だけ**が出す文言。列挙チェック
#: （``image_orientation.agentview='TBD' must be one of (...)``）にも "TBD" は
#: 含まれるので、``match="TBD"`` では番人を消しても素通りしてしまう（変異が生き残る）。
TBD_GUARD_MESSAGE = r'still "TBD"'


def test_tbd_configuration_blocks_startup(tmp_path):
    """``TBD`` のままの設定で ``InternVLARuntime`` を作ろうとしたら例外になること。

    「黙って既定値で動かさない」の実行時保証。**モデルは読まない**ので手元で確認できる
    （設定の検証を ``__init__`` の先頭に置いてあるため）。

    ``match`` は ``_reject_tbd`` **固有**の文言に寄せる。単に ``"TBD"`` を探すと、
    番人を無効化しても後段の列挙チェックのメッセージ
    （``image_orientation.agentview='TBD' must be one of (...)``）に "TBD" が
    含まれるせいでテストが通ってしまい、SC-3 のガードが実質無検査になる。
    """
    import json

    from internvla_runtime import InternVLARuntime, load_runtime_config

    config = load_runtime_config(INFERENCE_DIR / "runtime_config.json")

    with pytest.raises(ValueError, match=TBD_GUARD_MESSAGE):
        InternVLARuntime(tmp_path / "ckpt", tmp_path / "vlm", config)

    resolved = json.loads(json.dumps(config))
    resolved["image_orientation"] = {"agentview": "rot180", "wrist": "rot180"}
    resolved["gripper"]["dataset_convention"] = "zero_one"

    runtime = InternVLARuntime(tmp_path / "ckpt", tmp_path / "vlm", resolved)
    assert runtime.orientation == {"agentview": "rot180", "wrist": "rot180"}
    assert runtime.gripper["threshold"] == 0.5
    assert runtime.resize_h == 224 and runtime.resize_w == 224


def test_the_tbd_sentinel_is_what_rejects_tbd_not_the_enum_check():
    """``_reject_tbd`` 自体が働いていること（列挙チェックに肩代わりさせない）。

    ``_reject_tbd`` を無効化しても列挙チェックが起動を止めるので**機能欠陥ではない**が、
    それを許すと SC-3 のガードがテストで守られていない状態になる。番人が出す文言を
    直接固定して、番人を消したら落ちるようにする。

    ``gripper.dataset_convention`` は既定値そのものが ``TBD`` なので、
    「キーが無い」ケースでも番人が働く（列挙チェックのメッセージには依存しない）。
    """
    from internvla_runtime import TBD, _reject_tbd, resolve_gripper

    # 1. 番人の単体。ここが素通しになったら即落ちる。
    with pytest.raises(ValueError, match=TBD_GUARD_MESSAGE):
        _reject_tbd(TBD, "some.key")
    with pytest.raises(ValueError, match=r"some\.key"):
        _reject_tbd(" tbd ", "some.key")  # 空白と小文字も TBD として扱う
    assert _reject_tbd("none", "some.key") == "none"

    # 2. gripper 経路。既定値が TBD なので、キーごと無い設定でも番人が止める。
    with pytest.raises(ValueError, match=TBD_GUARD_MESSAGE):
        resolve_gripper({})

    # 3. orient_image も番人を通る。
    from internvla_runtime import orient_image

    with pytest.raises(ValueError, match=TBD_GUARD_MESSAGE):
        orient_image(np.zeros((4, 4, 3), dtype=np.uint8), TBD)


# --------------------------------------------------------------------------- #
# MEDIUM-1: chunking の設定は「モデルをロードする前」に全部検証する
# --------------------------------------------------------------------------- #
def _chunking_cfg(**overrides):
    section = {
        "mode": "rtc",
        "replan_steps": 16,
        "w_max": 0.8,
        "schedule": "linear",
        "exclude_dims": [6],
        "ensemble_m": 0.1,
    }
    section.update(overrides)
    return {"chunking": section}


def test_resolve_chunking_rejects_unknown_schedules():
    """``schedule`` は起動時に落とす。

    素通しにすると ``__init__`` もモデルロードも通過し、最初の replan 境界
    （env 十数ステップ目）で ``chunk_blending._weight_schedule`` が ``/act`` の
    **中から** ``ValueError`` を投げる。``__init__`` の
    「設定の検証はモデルをロードする前に済ませる」という不変条件が破れる。
    """
    from chunk_blending import SCHEDULES
    from internvla_runtime import resolve_chunking

    for schedule in SCHEDULES:
        assert resolve_chunking(_chunking_cfg(schedule=schedule))["schedule"] == schedule

    for bad in ("cosinus", "COSINE", "", "quadratic"):
        with pytest.raises(ValueError, match="chunking.schedule"):
            resolve_chunking(_chunking_cfg(schedule=bad))


def test_bad_schedule_fails_before_the_model_is_loaded(tmp_path):
    """``InternVLARuntime.__init__`` の時点で落ちること（ロード 120 秒を捨てない）。"""
    import json

    from internvla_runtime import InternVLARuntime, load_runtime_config

    config = json.loads(json.dumps(load_runtime_config(INFERENCE_DIR / "runtime_config.json")))
    config["image_orientation"] = {"agentview": "none", "wrist": "none"}
    config["gripper"]["dataset_convention"] = "zero_one"
    config["chunking"]["schedule"] = "cosinus"

    with pytest.raises(ValueError, match="chunking.schedule"):
        InternVLARuntime(tmp_path / "ckpt", tmp_path / "vlm", config)


def test_resolve_chunking_rejects_out_of_range_exclude_dims():
    """範囲外の ``exclude_dims`` は**黙って捨てられる**ので起動時に落とす。

    ``build_guidance`` も ``TemporalEnsembler`` も範囲でフィルタするため、
    ``exclude_dims=[7]`` は「除外したつもりで除外されていない」状態になる。
    """
    from chunk_blending import REAL_ACTION_DIM
    from internvla_runtime import resolve_chunking

    assert resolve_chunking(_chunking_cfg(exclude_dims=[6]))["exclude_dims"] == (6,)
    assert resolve_chunking(_chunking_cfg(exclude_dims=[]))["exclude_dims"] == ()
    assert resolve_chunking(_chunking_cfg(exclude_dims=[0, 6]))["exclude_dims"] == (0, 6)

    for bad in ([REAL_ACTION_DIM], [-1], [32], [6, 7]):
        with pytest.raises(ValueError, match="exclude_dims"):
            resolve_chunking(_chunking_cfg(exclude_dims=bad))

    for malformed in (6, "6", None, ["6"], [6.5], [6, 6]):
        with pytest.raises(ValueError, match="exclude_dims"):
            resolve_chunking(_chunking_cfg(exclude_dims=malformed))


def test_resolve_chunking_rejects_negative_ensemble_m():
    from internvla_runtime import resolve_chunking

    with pytest.raises(ValueError, match="ensemble_m"):
        resolve_chunking(_chunking_cfg(ensemble_m=-0.1))


def test_shipped_runtime_config_passes_every_chunking_check():
    """出荷する ``runtime_config.json`` の chunking が検証を通ること。"""
    from chunk_blending import SCHEDULES
    from internvla_runtime import load_runtime_config, resolve_chunking

    config = load_runtime_config(INFERENCE_DIR / "runtime_config.json")
    resolved = resolve_chunking(config)
    assert resolved["schedule"] in SCHEDULES
    assert resolved["exclude_dims"] == (6,)
    assert 0.0 <= resolved["w_max"] < 1.0


# --------------------------------------------------------------------------- #
# B8: policy_server と verify_inference が同じ入口を通る
# --------------------------------------------------------------------------- #
def test_both_entry_points_use_build_submission_runtime():
    """片方だけ別経路にすると自己チェックが本番と違うものを測る（移植手順書 B8）。"""
    for path in (SERVER_PY, VERIFY_PY):
        source = path.read_text(encoding="utf-8")
        assert "build_submission_runtime" in source, f"{path.name} must go through the shared entry"

    server_tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
    mypolicy = next(
        node
        for node in ast.walk(server_tree)
        if isinstance(node, ast.ClassDef) and node.name == "MyPolicy"
    )
    assert _calls_named(mypolicy, "build_submission_runtime")


def test_build_submission_runtime_applies_config_env(runtime_tree):
    fn = _find_function(runtime_tree, "build_submission_runtime")
    source = ast.unparse(fn)
    assert "apply_config_env(config)" in source
    assert "bootstrap(runtime_dir)" in source


def test_apply_config_env_uses_setdefault_not_assignment(runtime_tree):
    """``os.environ[...] = ...`` ではなく ``setdefault`` であること（A/B のため）。"""
    fn = _find_function(runtime_tree, "apply_config_env")
    source = ast.unparse(fn)
    assert "os.environ.setdefault" in source
    assignments = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Attribute)
            and t.value.attr == "environ"
            for t in node.targets
        )
    ]
    assert not assignments, "apply_config_env must not overwrite existing environment variables"


# --------------------------------------------------------------------------- #
# requirements.txt が validate_submission.py の禁止事項に触れないこと
# --------------------------------------------------------------------------- #
def test_requirements_have_no_forbidden_options():
    """F14: ``-i/--index-url/-f/-e/-r/-c`` とスキームは使えない。"""
    forbidden_options = ("-i ", "--index-url", "-f ", "--find-links", "-e ", "-r ", "-c ")
    forbidden_schemes = ("git:", "git+", "https:", "http:", "file:")

    for line in (INFERENCE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for option in forbidden_options:
            assert not stripped.startswith(option.strip() + " "), f"forbidden option in: {line}"
        for scheme in forbidden_schemes:
            assert scheme not in stripped, f"forbidden scheme in: {line}"


def test_requirements_pin_transformers_exactly():
    """差し替え版 ``modeling_qwen3_5.py`` が内部 API を叩くので完全固定が必須。"""
    text = (INFERENCE_DIR / "requirements.txt").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("transformers")]
    assert lines, "transformers must be listed"
    assert lines[0].startswith("transformers=="), f"transformers must be pinned exactly: {lines[0]}"


def test_requirements_pin_torch_to_the_grading_environment():
    """採点環境は torch 2.11.0+cu130。移植手順書の 2.10.0 をそのまま使わない（P1-12）。"""
    text = (INFERENCE_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "torch==2.11.0" in text
