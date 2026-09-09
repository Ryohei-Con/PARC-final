"""VQA ありファインチューニングの単体テスト（計画 internvla-a15-vqa-finetune-plan §5.9）。

GPU 不要。``tests/conftest.py`` のスタブ前提。2 系統:

1. ``prepare_vqa_data.py`` の変換ロジック（純関数 + 合成 .json / PIL ダミー画像）
2. overlay ``InternVLAA15ParcVQADatasetConfig`` の transform チェーン
   （``RenderDownsampleFn`` が ``ResizeVQAImagesWithPadFn`` の直前に 1 個・冪等・
   ``render_p_*`` 伝播・draccus 往復・128 へ落ちる）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conftest import requires_lerobot

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_vqa_data as pvd  # noqa: E402


# --------------------------------------------------------------------------- #
# 合成データ
# --------------------------------------------------------------------------- #
def _good_record(idx: int, image_rel) -> dict:
    images = list(image_rel) if isinstance(image_rel, (list, tuple)) else [image_rel]
    return {
        "id": f"rec-{idx}",
        "task": "spatial_relation",
        "conversations": [
            {"from": "human", "value": "<image>\nWhere is the bowl?"},
            {"from": "gpt", "value": "The bowl is on the left of the plate."},
        ],
        "images": images,
        "gt": "left",
        "h": 180,
        "w": 320,
    }


def _write_png(path: Path, size=(320, 180)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (123, 45, 200)).save(path)


# --------------------------------------------------------------------------- #
# 1. prepare_vqa_data.py — 純関数
# --------------------------------------------------------------------------- #
def test_prepare_vqa_json_to_jsonl(tmp_path):
    """`.json` 配列 -> 変換 -> `.jsonl`。images->image、source 付与、破損スキップ。"""
    raw = [
        _good_record(0, "Understanding/image/a/0.jpg"),
        {"id": "bad-empty-conv", "conversations": [], "images": ["x.jpg"]},
        {"id": "bad-no-image", "conversations": [{"from": "human", "value": "hi"}], "images": []},
        {"id": "bad-missing-image", "conversations": [{"from": "human", "value": "hi"}]},
        _good_record(1, ["Understanding/image/a/1.jpg", "Understanding/image/a/2.jpg"]),
    ]
    src_json = tmp_path / "meta_llava_format.json"
    src_json.write_text(json.dumps(raw), encoding="utf-8")

    loaded = pvd.load_json_array(src_json)
    assert len(loaded) == 5

    converted = []
    skipped = 0
    for rec in loaded:
        out = pvd.llava_record_to_vqa(rec, category="Understanding")
        if out is None:
            skipped += 1
        else:
            converted.append(out)

    assert skipped == 3
    assert len(converted) == 2

    # images -> image、単一なら str、複数なら list
    assert converted[0]["image"] == "Understanding/image/a/0.jpg"
    assert converted[1]["image"] == ["Understanding/image/a/1.jpg", "Understanding/image/a/2.jpg"]
    # source はカテゴリ別
    assert converted[0]["source"] == "robointer_vqa/Understanding"
    # conversations は保持
    assert converted[0]["conversations"][0]["from"] == "human"

    # jsonl は 1 行 1 オブジェクト
    out_jsonl = tmp_path / "all.jsonl"
    n = pvd._write_jsonl(out_jsonl, converted)
    assert n == 2
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "image" in obj and "conversations" in obj and "source" in obj


def test_prepare_vqa_subsample_is_deterministic():
    records = list(range(1000))
    a = pvd.subsample(records, 50, seed=42)
    b = pvd.subsample(records, 50, seed=42)
    c = pvd.subsample(records, 50, seed=7)
    assert a == b
    assert a != c
    assert len(a) == 50
    assert a == sorted(a)  # 元の順序を保つ
    # max_samples が None / 全件以上なら素通し
    assert pvd.subsample(records, None, seed=42) is records
    assert pvd.subsample(records, 5000, seed=42) is records


def test_prepare_vqa_pad_square():
    """320x180 -> 256x256（正方形・短辺パディング）。"""
    from PIL import Image

    img = Image.new("RGB", (320, 180), (10, 20, 30))
    out = pvd.pad_to_square(img, size=256)
    assert out.size == (256, 256)
    assert out.mode == "RGB"
    # パディングは黒（上下端に黒帯が入る）
    assert out.getpixel((128, 2)) == (0, 0, 0)


def test_prepare_vqa_rel_after_marker():
    assert str(pvd._rel_after_marker("robotinter/Understanding/image/a/b/0.jpg")) == "a/b/0.jpg"
    assert str(pvd._rel_after_marker("images/0.jpg")) == "0.jpg"
    assert str(pvd._rel_after_marker("0.jpg")) == "0.jpg"


def test_prepare_vqa_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        pvd.main(["--help"])
    assert exc.value.code == 0


# --------------------------------------------------------------------------- #
# 2. prepare_vqa_data.py — 画像解決まで含めた e2e（合成ツリー）
# --------------------------------------------------------------------------- #
@requires_lerobot
def test_prepare_vqa_image_path_resolves(tmp_path):
    """変換後 jsonl + 合成画像を `VQADataset` が読める（`jsonl_path.parent/image` で解決。V5）。"""
    from lerobot.datasets.vqa_dataset import VQADataset

    src = tmp_path / "raw"
    meta_dir = src / "robotinter" / "Understanding" / "meta"
    img_dir = src / "robotinter" / "Understanding" / "image" / "a"
    meta_dir.mkdir(parents=True)
    _write_png(img_dir / "0.png")
    _write_png(img_dir / "1.png")

    raw = [
        _good_record(0, "Understanding/image/a/0.png"),
        _good_record(1, "Understanding/image/a/1.png"),
    ]
    (meta_dir / "data_llava_format.json").write_text(json.dumps(raw), encoding="utf-8")

    out = tmp_path / "lerobot_vqa"
    summary = pvd.build_vqa_dataset(
        src=src,
        out=out,
        categories=["Understanding"],
        merge_into="all.jsonl",
        pad_square=True,
        seed=42,
    )
    jsonl = out / "all.jsonl"
    assert jsonl.is_file()
    assert summary["targets"][str(jsonl)] == 2

    ds = VQADataset(root=None, repo_id=str(jsonl))
    assert len(ds) == 2
    sample = ds[0]
    assert "observation.images.image0" in sample
    # --pad-square で 256x256 に再生成されている
    assert tuple(sample["observation.images.image0"].shape[-2:]) == (256, 256)

    # 冪等: 2 回目は "検証のみ"
    again = pvd.build_vqa_dataset(
        src=src, out=out, categories=["Understanding"], merge_into="all.jsonl", pad_square=True
    )
    assert again["status"] == "verified"


@requires_lerobot
def test_prepare_vqa_skips_unresolvable_images(tmp_path):
    """画像が展開ツリーに無いレコードはスキップされ、件数が summary に出る。"""
    src = tmp_path / "raw"
    meta_dir = src / "robotinter" / "Understanding" / "meta"
    img_dir = src / "robotinter" / "Understanding" / "image" / "a"
    meta_dir.mkdir(parents=True)
    _write_png(img_dir / "0.png")

    raw = [
        _good_record(0, "Understanding/image/a/0.png"),
        _good_record(1, "Understanding/image/a/missing.png"),
    ]
    (meta_dir / "data_llava_format.json").write_text(json.dumps(raw), encoding="utf-8")

    out = tmp_path / "lerobot_vqa"
    summary = pvd.build_vqa_dataset(
        src=src, out=out, categories=["Understanding"], merge_into="all.jsonl", pad_square=False
    )
    assert summary["targets"][str(out / "all.jsonl")] == 1
    assert summary["skipped"]["Understanding"]["image"] == 1


# --------------------------------------------------------------------------- #
# 3. overlay: InternVLAA15ParcVQADatasetConfig の transform チェーン
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def overlay():
    import lerobot_policy_parc as module

    module.assert_installed()
    return module


@pytest.fixture()
def vqa_config_cls(overlay, offline_processors):
    from lerobot_policy_parc.dataset_config import InternVLAA15ParcVQADatasetConfig

    return InternVLAA15ParcVQADatasetConfig


def _chain(config) -> list[str]:
    return [type(t).__name__ for t in config.data_transforms.inputs]


# 挿入位置（``ResizeVQAImagesWithPadFn`` の直前に 1 個）と ``__post_init__`` 冪等は
# ``test_dataset_config.py::test_vqa_render_downsample_is_before_vqa_resize`` /
# ``::test_vqa_post_init_is_idempotent`` が既にカバーしている。ここでは重複させず、
# 新しい観点（prep 変換 / VQADataset 読み取り / draccus 往復 / render_p_* 伝播 /
# 128 へ落ちる / subsample 決定性）だけを検査する。


@requires_lerobot
def test_vqa_render_downsample_disabled_removes_transform(vqa_config_cls):
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = vqa_config_cls(render_downsample=False)
    assert not any(isinstance(t, RenderDownsampleFn) for t in config.data_transforms.inputs)


@requires_lerobot
def test_vqa_render_probabilities_propagated(vqa_config_cls):
    """``render_p_*`` が ``RenderDownsampleFn`` に伝わる（既定 / カスタム）。"""
    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    config = vqa_config_cls()
    render = next(t for t in config.data_transforms.inputs if isinstance(t, RenderDownsampleFn))
    assert render.target_h == 128 and render.target_w == 128
    assert render.p_nearest == 0.10
    assert render.probabilities() == {"box": 0.45, "triangle": 0.30, "cubic": 0.15, "nearest": 0.10}

    custom = vqa_config_cls(
        render_target=96,
        render_p_box=0.50,
        render_p_triangle=0.25,
        render_p_cubic=0.15,
        render_p_nearest=0.10,
    )
    r2 = next(t for t in custom.data_transforms.inputs if isinstance(t, RenderDownsampleFn))
    assert r2.target_h == 96
    assert r2.p_box == 0.50


@requires_lerobot
def test_vqa_draccus_roundtrip(vqa_config_cls):
    """``train_config.json`` 往復で discriminator とチェーン順が保存される（F3）。"""
    import draccus

    from lerobot.configs.default import VQADatasetConfig

    config = vqa_config_cls(render_p_nearest=0.10)
    encoded = draccus.encode(config)
    assert isinstance(encoded, dict)
    assert encoded["render_p_nearest"] == 0.10
    assert VQADatasetConfig.get_choice_name(vqa_config_cls) == "internvla_a1_5_parc"

    transform_types = [t.get(draccus.CHOICE_TYPE_KEY) for t in encoded["data_transforms"]["inputs"]]
    assert "render_downsample" in transform_types
    assert transform_types.index("render_downsample") + 1 == transform_types.index("vqa_resize_with_pad")

    payload = dict(encoded)
    payload[draccus.CHOICE_TYPE_KEY] = VQADatasetConfig.get_choice_name(vqa_config_cls)
    restored = draccus.decode(VQADatasetConfig, payload)
    assert type(restored) is vqa_config_cls
    assert _chain(restored) == _chain(config)
    assert draccus.encode(restored) == encoded


@requires_lerobot
def test_vqa_sample_downsampled_to_target(overlay):
    """256x256 の VQA 画像を overlay の ``RenderDownsampleFn`` に通すと 128x128 になる。

    上流 ``_make_vqa_dataset`` は transform を hydrate しない（V2）ので、``keys`` 空 +
    ``auto_detect_keys=True`` の経路で ``observation.images.image0`` を拾う。
    """
    import torch

    from lerobot_policy_parc.transforms_render import RenderDownsampleFn

    fn = RenderDownsampleFn(
        target_h=128, target_w=128, p_box=0.0, p_triangle=0.0, p_cubic=0.0, p_nearest=1.0
    )
    sample = {
        "observation.images.image0": torch.rand(3, 256, 256),
        "observation.images.image1": torch.rand(3, 256, 256),
        "mask0": torch.tensor(True),
        "observation.state": torch.zeros(32),
    }
    out = fn(sample)
    assert tuple(out["observation.images.image0"].shape) == (3, 128, 128)
    assert tuple(out["observation.images.image1"].shape) == (3, 128, 128)
    # 画像でないキーは触らない
    assert out["observation.state"].shape == (32,)
    # nearest 固定なので位相ジッタの記録が残る
    assert fn._last_draw is not None
    assert fn._last_draw[0] == "nearest"
    # nearest の位相ジッタは {0,1}^2 から引く（半ピクセルオフセット）
    oy, ox = fn._last_draw[1]
    assert oy in (0, 1) and ox in (0, 1)


# --------------------------------------------------------------------------- #
# 4. ベースライン不変の回帰
# --------------------------------------------------------------------------- #
def test_baseline_vqa_config_still_dead_path_in_train_ivla_a15():
    """``scripts/train_ivla_a15.sh``（ベースライン）に ``--vqa_dataset`` が無いこと。"""
    script = Path(__file__).resolve().parent.parent / "scripts" / "train_ivla_a15.sh"
    text = script.read_text(encoding="utf-8")
    assert "--vqa_dataset" not in text


def test_vqa_train_script_wires_vqa_dataset_args():
    """``scripts/train_ivla_a15_vqa.sh`` は ``--vqa_dataset.*`` を渡し ``enable_vqa_loss`` を維持。"""
    script = Path(__file__).resolve().parent.parent / "scripts" / "train_ivla_a15_vqa.sh"
    text = script.read_text(encoding="utf-8")
    assert "--vqa_dataset.type=internvla_a1_5_parc" in text
    assert "--vqa_dataset.repo_id=" in text
    assert "--vqa_dataset.weight=" in text
    assert "--vqa_dataset.render_target=" in text
    assert "--policy.enable_vqa_loss=true" in text
