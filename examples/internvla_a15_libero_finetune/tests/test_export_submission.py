"""``export_submission.py`` の手元テスト（計画 P1-11 / §7.1）。

エクスポート本体の実行はクラウドだが、safetensors を触る部分は**小さな合成
checkpoint**で手元テストできる。

- [x] ``copy_safetensors_without`` が指定キーを落とし、残りをビット一致で保つ
- [x] **``learnable_tokens`` / ``learnable_tokens_in_proj`` を落とさない**（保護）
- [x] ``list_droppable_keys`` が WAN 由来キーだけを拾う（F5 の実測もここで表現）
- [x] ``find_tied_duplicates`` が tie された重複を検出する（B5）
- [x] ``copy_vlm_config_only`` が重みと ``model.safetensors.index.json`` を持って行かない（B4）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))

from export_submission import (  # noqa: E402
    PROTECTED_SUBSTRINGS,
    copy_safetensors_without,
    copy_vlm_config_only,
    find_tied_duplicates,
    list_droppable_keys,
)


def _make_checkpoint(path: Path, *, with_wan: bool = True, tie: bool = True) -> dict:
    """本物の checkpoint のキー構成を真似た合成 safetensors。"""
    shared = torch.arange(24, dtype=torch.float32).reshape(6, 4) * 0.5

    tensors: dict[str, torch.Tensor] = {
        "model.qwen3_5_with_expert.qwen3_5.model.embed_tokens.weight": shared.clone(),
        "model.action_out_proj.weight": torch.randn(4, 8),
        "model.action_out_proj.bias": torch.randn(4),
        # ---- 絶対に落としてはいけないもの ----
        "model.learnable_tokens": torch.randn(50, 8),
        "model.learnable_tokens_in_proj.weight": torch.randn(8, 8),
        "model.learnable_tokens_in_proj.bias": torch.randn(8),
        # ---- bfloat16 と整数 buffer も混ぜる（dtype 保存の確認）----
        "model.some_bf16_weight": torch.randn(3, 5).to(torch.bfloat16),
    }
    if tie:
        tensors["model.qwen3_5_with_expert.qwen3_5.lm_head.weight"] = shared.clone()
    if with_wan:
        tensors["model.learnable_to_wan_proj.weight"] = torch.randn(16, 8)
        tensors["model.learnable_to_wan_proj.bias"] = torch.randn(16)
        tensors["model._wan_grid_sizes"] = torch.tensor([2, 7, 7], dtype=torch.int64)
        tensors["model.wan_video_model.blocks.0.weight"] = torch.randn(4, 4)

    save_file(tensors, str(path), metadata={"format": "pt", "step": "30000"})
    return tensors


@pytest.fixture()
def checkpoint(tmp_path) -> tuple[Path, dict]:
    path = tmp_path / "model.safetensors"
    tensors = _make_checkpoint(path)
    return path, tensors


def _read_all(path: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            out[key] = handle.get_tensor(key)
    return out


# --------------------------------------------------------------------------- #
# list_droppable_keys
# --------------------------------------------------------------------------- #
def test_list_droppable_keys_picks_only_wan_keys(checkpoint):
    path, _tensors = checkpoint
    droppable = list_droppable_keys(path)

    assert set(droppable) == {
        "model.learnable_to_wan_proj.weight",
        "model.learnable_to_wan_proj.bias",
        "model._wan_grid_sizes",
        "model.wan_video_model.blocks.0.weight",
    }


def test_list_droppable_keys_never_includes_learnable_tokens(checkpoint):
    """``learnable_tokens`` / ``learnable_tokens_in_proj`` は foresight token の本体。

    ``learnable_to_wan_proj`` と名前が似ているので、接頭辞の書き間違いで巻き込むと
    静かに壊れる。明示的に守る。
    """
    path, _tensors = checkpoint
    droppable = list_droppable_keys(path)

    for key in droppable:
        for protected in PROTECTED_SUBSTRINGS:
            assert protected not in key

    assert "model.learnable_tokens" not in droppable
    assert "model.learnable_tokens_in_proj.weight" not in droppable
    assert "model.learnable_tokens_in_proj.bias" not in droppable


def test_list_droppable_keys_is_empty_when_wan_already_excluded(tmp_path):
    """F5 の実測: 上流 ``state_dict()`` は ``model.wan_video_model.*`` を既に除外している。

    その場合ここは 0 件になる。エクスポート側はそれを前提にせず、実際に数えて
    MANIFEST に記録する。
    """
    path = tmp_path / "model.safetensors"
    _make_checkpoint(path, with_wan=False)
    assert list_droppable_keys(path) == []


# --------------------------------------------------------------------------- #
# copy_safetensors_without
# --------------------------------------------------------------------------- #
def test_copy_safetensors_without_drops_requested_keys(checkpoint, tmp_path):
    path, tensors = checkpoint
    out = tmp_path / "out.safetensors"

    drop = list_droppable_keys(path)
    report = copy_safetensors_without(path, out, drop)

    kept = _read_all(out)
    assert set(kept) == set(tensors) - set(drop)
    assert set(report["dropped"]) == set(drop)
    assert report["bytes_after"] < report["bytes_before"]


def test_copy_safetensors_without_is_bit_identical(checkpoint, tmp_path):
    """落とさなかったテンソルは**ビット一致**で残ること（dtype / shape / 値）。"""
    path, tensors = checkpoint
    out = tmp_path / "out.safetensors"

    drop = list_droppable_keys(path)
    copy_safetensors_without(path, out, drop)

    kept = _read_all(out)
    for key, tensor in kept.items():
        original = tensors[key]
        assert tensor.dtype == original.dtype, key
        assert tensor.shape == original.shape, key
        assert torch.equal(tensor, original), key


def test_copy_safetensors_without_preserves_metadata(checkpoint, tmp_path):
    path, _tensors = checkpoint
    out = tmp_path / "out.safetensors"

    copy_safetensors_without(path, out, [], metadata={"exported_by": "parc"})

    with safe_open(str(out), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["format"] == "pt"
    assert metadata["step"] == "30000"
    assert metadata["exported_by"] == "parc"


def test_copy_safetensors_without_nothing_is_a_faithful_copy(checkpoint, tmp_path):
    path, tensors = checkpoint
    out = tmp_path / "out.safetensors"

    copy_safetensors_without(path, out, [])
    kept = _read_all(out)

    assert set(kept) == set(tensors)
    for key, tensor in kept.items():
        assert torch.equal(tensor, tensors[key]), key


def test_copy_safetensors_without_refuses_protected_keys(checkpoint, tmp_path):
    """保護キーを名指しで落とそうとしたら例外（誤爆の番人）。"""
    path, _tensors = checkpoint
    out = tmp_path / "out.safetensors"

    with pytest.raises(RuntimeError, match="protected key"):
        copy_safetensors_without(path, out, ["model.learnable_tokens"])
    with pytest.raises(RuntimeError, match="protected key"):
        copy_safetensors_without(path, out, ["model.learnable_tokens_in_proj.weight"])


def test_copy_safetensors_without_rejects_unknown_keys(checkpoint, tmp_path):
    path, _tensors = checkpoint
    out = tmp_path / "out.safetensors"
    with pytest.raises(KeyError, match="not in"):
        copy_safetensors_without(path, out, ["model.does_not_exist"])


def test_copy_safetensors_without_refuses_to_write_empty(tmp_path):
    path = tmp_path / "tiny.safetensors"
    save_file({"only": torch.ones(2)}, str(path))
    with pytest.raises(ValueError, match="empty safetensors"):
        copy_safetensors_without(path, tmp_path / "out.safetensors", ["only"])


# --------------------------------------------------------------------------- #
# find_tied_duplicates
# --------------------------------------------------------------------------- #
def test_find_tied_duplicates_detects_shared_weights(checkpoint):
    """Qwen3.5 は embed_tokens と lm_head で重みを共有する（B5）。"""
    path, _tensors = checkpoint
    duplicates = find_tied_duplicates(path)

    assert duplicates == ["model.qwen3_5_with_expert.qwen3_5.lm_head.weight"], duplicates


def test_find_tied_duplicates_keeps_one_copy(checkpoint, tmp_path):
    path, tensors = checkpoint
    out = tmp_path / "out.safetensors"

    duplicates = find_tied_duplicates(path)
    copy_safetensors_without(path, out, duplicates)

    kept = _read_all(out)
    assert "model.qwen3_5_with_expert.qwen3_5.embed_tokens.weight" not in duplicates
    embed_key = "model.qwen3_5_with_expert.qwen3_5.model.embed_tokens.weight"
    assert embed_key in kept
    assert torch.equal(kept[embed_key], tensors[embed_key])


def test_find_tied_duplicates_is_empty_without_ties(tmp_path):
    path = tmp_path / "model.safetensors"
    _make_checkpoint(path, tie=False)
    assert find_tied_duplicates(path) == []


# --------------------------------------------------------------------------- #
# copy_vlm_config_only
# --------------------------------------------------------------------------- #
def test_copy_vlm_config_only_excludes_weights_and_index(tmp_path):
    src = tmp_path / "vlm"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    (src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (src / "vocab.txt").write_text("hello\n", encoding="utf-8")
    (src / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (src / "tokenizer.model").write_bytes(b"\x00\x01")
    # ---- 持って行ってはいけないもの ----
    (src / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (src / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 64)
    (src / "pytorch_model.bin").write_bytes(b"\x00" * 64)

    dst = tmp_path / "out"
    copied = copy_vlm_config_only(src, dst)

    assert "config.json" in copied
    assert "tokenizer_config.json" in copied
    assert "vocab.txt" in copied
    assert "chat_template.jinja" in copied
    assert "tokenizer.model" in copied

    assert "model.safetensors.index.json" not in copied
    assert not (dst / "model.safetensors.index.json").exists()
    assert list(dst.rglob("*.safetensors")) == []
    assert list(dst.rglob("*.bin")) == []


def test_copy_vlm_config_only_fails_loudly_if_weights_slip_through(tmp_path, monkeypatch):
    """パターンを緩めた場合に気付けること（`snapshot_download` のキャッシュ対策）。"""
    import export_submission

    src = tmp_path / "vlm"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors").write_bytes(b"\x00" * 16)

    monkeypatch.setattr(export_submission, "VLM_PATTERNS", ("*.json", "*.safetensors"))
    with pytest.raises(RuntimeError, match="still contains weights"):
        copy_vlm_config_only(src, tmp_path / "out")
