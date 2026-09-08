"""F6 の手元回帰: 学習チェーンと推論チェーンの ``image_grid_thw`` が一致すること。

計画 §7.1 の最終項目（``parity_check.py --mode mini``）を pytest に載せたもの。
Qwen の **processor ファイル（22MB、重み不要）** が要るので、``IVLA_VLM_DIR`` が
指すディレクトリが実在するときだけ走る。クラウドで 1 回通せば回帰として残る::

    IVLA_VLM_DIR=~/hf/Qwen3.5-2B pytest tests/test_parity_mini.py -q

**なぜこれが要るか（F6）**: 上流の推論バックエンドは ``ResizeImagesWithPadFn`` を
hydrate せずに構築しているため画像をリサイズしない。学習は 224、推論は生解像度のまま
Qwen の smart_resize に入るので視覚トークン数が別物になり、**エラーは出ないまま精度
だけ落ちる**。このテストは同一の合成フレームを両チェーンに流して一致を確認する。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

IVLA_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = IVLA_DIR / "tools"

from conftest import requires_lerobot  # noqa: E402


def _vlm_dir() -> Path | None:
    raw = os.environ.get("IVLA_VLM_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


requires_vlm_processor = pytest.mark.skipif(
    _vlm_dir() is None,
    reason=(
        "needs the Qwen processor files (~22MB, no weights); "
        "set IVLA_VLM_DIR to a directory containing preprocessor_config.json"
    ),
)


def _load_parity_check():
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    import parity_check

    return parity_check


@requires_lerobot
@requires_vlm_processor
def test_train_and_inference_chains_agree_on_image_grid_thw(capsys):
    """``parity_check.py --mode mini`` 相当を pytest から実行する（F6 の回帰）。"""
    import argparse

    parity_check = _load_parity_check()

    args = argparse.Namespace(
        mode="mini",
        vlm_dir=str(_vlm_dir()),
        train_config=None,
        use_image_token=False,
    )
    code = parity_check.run_mini(args)
    captured = capsys.readouterr()
    assert code == 0, f"parity_check --mode mini failed:\n{captured.out}\n{captured.err}"
    assert '"image_grid_thw_match": true' in captured.out.lower()


@requires_lerobot
def test_parity_check_mini_is_importable_and_wired():
    """``--mode mini`` の骨格が壊れていないこと（processor が無くても確認できる）。

    ここまでは環境変数なしで常に走る。``run_mini`` が
    「RenderDownsample -> resize_with_pad(224)」と「env 128 -> resize_with_pad(224)」の
    2 チェーンを組み立てていること、合成フレームが期待どおりの shape であることを見る。
    """
    import inspect

    import numpy as np

    parity_check = _load_parity_check()

    frame = parity_check._synthetic_frame(256, seed=0)
    assert frame.shape == (256, 256, 3)
    assert frame.dtype == np.uint8

    chw = parity_check._to_chw_float01(frame)
    assert tuple(chw.shape) == (3, 256, 256)
    assert float(chw.min()) >= 0.0 and float(chw.max()) <= 1.0

    source = inspect.getsource(parity_check.run_mini)
    assert "RenderDownsampleFn" in source
    assert source.count("resize_with_pad") >= 3  # import + 学習側 + 推論側
    assert "image_grid_thw" in source
