"""``policy_server.py`` がテンプレートと ``MyPolicy`` 以外一致すること（P1-9 / §7.1 / SC-3）。

提出テンプレートは「``MyPolicy`` の中身だけを差し替える。それ以外（``BasePolicy``、
シリアライゼーション、FastAPI エンドポイント）は変更不可」と定めている
（``submission_template/policy_server.py`` の冒頭、移植手順書 B9）。

**テンプレートを 1 行も編集していないことの回帰テスト**でもある。テンプレート側が
変わったらこのテストが落ちるので、追従漏れに気付ける。

比較は AST で行う（空白・改行・コメントの差は無視し、**構文木として**一致するかを見る）。
許す差分は次の 2 つだけ:

1. ``MyPolicy`` クラス定義の中身
2. 先頭（``class BasePolicy`` より前）に追加された import / ``sys.path`` 注入ブロック

モジュール docstring は「テンプレートである」旨の説明なので差し替えを許す。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[3] / "submission_template" / "policy_server.py"
OURS = Path(__file__).resolve().parent.parent / "inference" / "policy_server.py"

#: 先頭に足してよい文の種類（import と sys.path 注入）
ALLOWED_PROLOGUE_NODES = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.If)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _dump(node: ast.AST) -> str:
    """位置情報を落とした構文木のダンプ（行番号のずれを無視するため）。"""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _named(body: list[ast.stmt], name: str) -> ast.stmt | None:
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


@pytest.fixture(scope="module")
def template_body() -> list[ast.stmt]:
    assert TEMPLATE.is_file(), f"submission template not found: {TEMPLATE}"
    return _strip_docstring(_parse(TEMPLATE).body)


@pytest.fixture(scope="module")
def ours_body() -> list[ast.stmt]:
    assert OURS.is_file(), f"our policy_server.py not found: {OURS}"
    return _strip_docstring(_parse(OURS).body)


def test_both_files_parse():
    _parse(TEMPLATE)
    _parse(OURS)


def test_only_mypolicy_and_prologue_differ(template_body, ours_body):
    """テンプレートの全ノードが、順序を保ったまま我々の側にも存在すること。

    ``MyPolicy`` を除いた残りが**構文木として完全一致**し、我々の側の余剰ノードが
    prologue（``class BasePolicy`` より前の import / ``sys.path`` 注入）だけであることを
    確かめる。
    """
    template_rest = [n for n in template_body if not _is_mypolicy(n)]
    ours_rest = [n for n in ours_body if not _is_mypolicy(n)]

    template_dumps = [_dump(n) for n in template_rest]
    ours_dumps = [_dump(n) for n in ours_rest]

    # テンプレート側のノードは、順序を保って全部我々の側にあること。
    cursor = 0
    extra_indices: list[int] = []
    for index, dump in enumerate(ours_dumps):
        if cursor < len(template_dumps) and dump == template_dumps[cursor]:
            cursor += 1
        else:
            extra_indices.append(index)

    missing = template_dumps[cursor:]
    assert not missing, (
        f"{len(missing)} statement(s) from the template are missing or modified in {OURS.name}. "
        f"First missing: {missing[0][:200]}"
    )

    # 余剰ノードは prologue（BasePolicy より前）に限られること。
    base_policy_index = next(
        i for i, node in enumerate(ours_rest) if isinstance(node, ast.ClassDef)
        and node.name == "BasePolicy"
    )
    for index in extra_indices:
        node = ours_rest[index]
        assert index < base_policy_index, (
            f"extra statement outside the prologue at index {index}: {_dump(node)[:200]}"
        )
        assert isinstance(node, ALLOWED_PROLOGUE_NODES), (
            f"prologue statement of type {type(node).__name__} is not allowed: {_dump(node)[:200]}"
        )


def _is_mypolicy(node: ast.stmt) -> bool:
    return isinstance(node, ast.ClassDef) and node.name == "MyPolicy"


def test_mypolicy_exists_in_both(template_body, ours_body):
    assert _named(template_body, "MyPolicy") is not None
    assert _named(ours_body, "MyPolicy") is not None


def test_mypolicy_was_actually_replaced(template_body, ours_body):
    """``MyPolicy`` が **TODO のまま**でないこと（テストが空振りしていない証明）。"""
    template_mypolicy = _named(template_body, "MyPolicy")
    ours_mypolicy = _named(ours_body, "MyPolicy")
    assert _dump(template_mypolicy) != _dump(ours_mypolicy)

    source = ast.unparse(ours_mypolicy)
    assert "np.random.uniform" not in source, "MyPolicy is still the random placeholder policy"
    assert "build_submission_runtime" in source, (
        "MyPolicy must go through internvla_runtime.build_submission_runtime() "
        "so that policy_server.py and verify_inference.py share one path (B8)"
    )


def test_mypolicy_still_implements_the_interface(ours_body):
    ours_mypolicy = _named(ours_body, "MyPolicy")
    assert isinstance(ours_mypolicy, ast.ClassDef)
    assert [b.id for b in ours_mypolicy.bases if isinstance(b, ast.Name)] == ["BasePolicy"]

    methods = {
        node.name
        for node in ours_mypolicy.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"__init__", "get_action", "reset"} <= methods


def test_immutable_sections_are_byte_identical(template_body, ours_body):
    """``BasePolicy`` / シリアライゼーション / FastAPI 部分が構文木として同一。"""
    for name in ("BasePolicy", "deserialize_obs", "serialize_action", "set_policy",
                 "health", "reset_policy", "act"):
        template_node = _named(template_body, name)
        ours_node = _named(ours_body, name)
        assert template_node is not None, f"{name} is missing from the template"
        assert ours_node is not None, f"{name} is missing from {OURS.name}"
        assert _dump(template_node) == _dump(ours_node), f"{name} was modified"


def test_prologue_injects_sys_path():
    """起動方法に依らず ``internvla_runtime`` を import できること（移植手順書 B9）。"""
    source = OURS.read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(_THIS_DIR))" in source
    assert "Path(__file__).resolve().parent" in source


def test_template_is_unmodified():
    """``submission_template/`` を編集していないこと（SC-6 の Evaluator 項目）。

    テンプレート側に MyPolicy のプレースホルダ（ランダムポリシー）が残っていれば、
    我々が触っていない証拠になる。
    """
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "np.random.uniform(-1, 1, size=7)" in source
    assert "TODO: モデルのロード" in source
    assert "internvla" not in source.lower()
