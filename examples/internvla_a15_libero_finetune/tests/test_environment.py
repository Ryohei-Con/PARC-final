"""テスト環境そのものの申告（``conftest.py`` のスタブ在庫を可視化する）。

``conftest.py`` は上流 lerobot を import するために一部の重い依存を ``sys.modules``
のダミーで置き換えている。**どれを誤魔化しているかが分からないままだと、
「テストは緑だが実環境では動かない」に気づけない。** ここでその一覧を印字し、
スタブが想定の範囲に収まっていることを機械検証する。

``pytest tests/test_environment.py -s`` で一覧が読める。
"""

from __future__ import annotations

import sys

import conftest

#: スタブしてよいトップレベルパッケージ。ここに無いものを勝手にスタブしない。
#: いずれも「import されるだけで、テスト対象のロジックには使われない」もの。
ALLOWED_STUB_ROOTS = {
    "pandas",
    "pyarrow",
    "datasets",
    "av",
    "imageio",
    "accelerate",
    "einops",
    "diffusers",
}

#: 絶対にスタブしてはならないもの（数値検証がこれらの実装に依存している）。
FORBIDDEN_STUBS = {"torch", "numpy", "transformers", "lerobot", "lerobot_policy_parc", "scipy"}


def test_stub_inventory_is_declared(capsys):
    """``conftest.STUBBED_MODULES`` の中身を印字し、想定の範囲内であることを確かめる。"""
    stubs = list(conftest.STUBBED_MODULES)

    with capsys.disabled():
        print()
        print("=" * 60)
        print(f"conftest stub inventory ({len(stubs)} modules)")
        print(f"  upstream repo: {conftest.UPSTREAM_REPO}")
        print(f"  lerobot skip : {conftest.LEROBOT_SKIP_REASON or 'none (real upstream in use)'}")
        print(f"  cv2 skip     : {conftest.CV2_SKIP_REASON or 'none (real cv2 in use)'}")
        for name in stubs:
            print(f"  - {name}")
        if not stubs:
            print("  (nothing stubbed: every dependency is installed for real)")
        print("=" * 60)

    for name in stubs:
        root = name.split(".")[0]
        assert root in ALLOWED_STUB_ROOTS, f"unexpected stub: {name}"


def test_critical_modules_are_never_stubbed():
    """数値検証にかかるモジュールは必ず実物であること。"""
    for name in FORBIDDEN_STUBS:
        assert name not in conftest.STUBBED_MODULES
        module = sys.modules.get(name)
        if module is None:
            continue
        assert not isinstance(module, conftest._LazyStubModule), f"{name} is a test stub"


def test_stub_modules_are_declared_in_the_conftest_table():
    """実際にスタブしたものが ``_STUB_SPECS`` に宣言されていること。"""
    declared = set(conftest._STUB_SPECS)
    for name in conftest.STUBBED_MODULES:
        assert name in declared, f"{name} was stubbed but is not declared in _STUB_SPECS"
