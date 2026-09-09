"""学習環境の検証（計画 P2-13/14、移植手順書 §3.1 / §6）。

**「import が通る」で満足しない**のがこのスクリプトの存在理由である。移植手順書 §6 の
サイレント失敗カタログにあるとおり、この構成の壊れ方は「起動はする・ログ上は学習が
進む・精度だけ落ちる」形をとる。よってここでは実際に動かして値を見る:

  C1 Python / OS
  C2 torch — **GPU で行列積を実行する**。sm_120（Blackwell）のカーネルを持たない
     wheel は ``torch.cuda.is_available()`` が True でも実行時に落ちる
  C3 numpy / その他の版
  C4 transformers 差し替え — 「ファイルがある」ではなく**上流ソースとバイト一致**を見る
  C5 lerobot（上流）の import と上流ファイルのハッシュ照合（overlay の前提）
  C6 overlay ``lerobot_policy_parc`` の登録（実照会）
  C7 **torchcodec が実データの mp4 を非ゼロにデコードする**（手順書 §3.1 / 計画 R4）
     ABI 不一致は import では発覚せず、学習中ずっと黒画像を食わせ続ける
  C8 オプショナル依存の有無（致命的ではないが速度に効くので記録する）

失敗（FAIL）が 1 つでもあれば exit 1。WARN は先へ進むが記録に残す。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path

IVLA_DIR = Path(os.environ.get("IVLA_DIR", Path(__file__).resolve().parent.parent))
IVLA_REPO = Path(os.environ.get("IVLA_REPO", Path.home() / "InternVLA-A-series"))
DATASET_ROOT = Path(os.environ.get("IVLA_DATASET_ROOT", Path.home() / "data/libero_combined_20hz"))
ARTIFACTS = IVLA_DIR / "artifacts" / "facts"

_RESULTS: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _RESULTS.append((status, name, detail))
    mark = {"PASS": "  OK  ", "WARN": " WARN ", "FAIL": " FAIL "}[status]
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def check(name: str):
    """例外を FAIL に落とすデコレータ。1 項目の失敗で残りを飛ばさない。"""

    def wrapper(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — 検証スクリプトなので握って続行する
            record("FAIL", name, f"{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)

    return wrapper


# ---------------------------------------------------------------------------
# C1 / C3
# ---------------------------------------------------------------------------
print("=" * 78)
print("InternVLA-A1.5 学習環境の検証")
print("=" * 78)

record(
    "PASS" if sys.version_info[:2] == (3, 11) else "WARN",
    "C1 Python 3.11",
    f"{platform.python_version()} @ {sys.prefix}",
)


# ---------------------------------------------------------------------------
# C2 torch — GPU で実際に計算する
# ---------------------------------------------------------------------------
@check("C2 torch / CUDA")
def _c2() -> None:
    import torch

    detail = f"torch={torch.__version__} cuda_build={torch.version.cuda}"
    if not torch.cuda.is_available():
        record("FAIL", "C2 torch / CUDA", detail + " — GPU が見えない")
        return

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / 2**30

    # ★ ここが本番 (1)。`is_available()` は wheel が sm_120 のカーネルを持っていなくても
    #    True を返す。実際に行列積 + bfloat16 を回して初めて分かる。
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = (a @ b).float()
    torch.cuda.synchronize()
    assert torch.isfinite(c).all(), "GPU 行列積の結果に NaN/Inf がある"

    # ★ ここが本番 (2)。**bias 付き Linear は cuBLASLt という別経路を通る。**
    #    素の matmul が通っても Lt だけ壊れていることが実際にあった（nvidia-cublas
    #    13.1.0.3 + sm_120 で CUBLAS_STATUS_NOT_INITIALIZED）。モデルは bias 付き
    #    Linear だらけなので、これを見ないと「環境 OK」と言った直後に学習が落ちる。
    x = torch.randn(1024, 1152, device="cuda", dtype=torch.bfloat16)
    lin = torch.nn.Linear(1152, 3456, bias=True).cuda().to(torch.bfloat16)
    y = lin(x)
    torch.cuda.synchronize()
    assert torch.isfinite(y.float()).all(), "bias 付き Linear の結果に NaN/Inf がある"

    arch_list = torch.cuda.get_arch_list()
    sm = f"sm_{major}{minor}"
    arch_ok = sm in arch_list
    record(
        "PASS" if arch_ok else "WARN",
        "C2 torch / CUDA",
        f"{detail} device={name} cap={major}.{minor} vram={total_gb:.0f}GiB "
        f"bf16_matmul=OK bf16_linear_bias(cuBLASLt)=OK arch_list_has_{sm}={arch_ok}",
    )


@check("C3 依存の版")
def _c3() -> None:
    import importlib.metadata as md

    import numpy
    import torchvision
    import transformers

    try:
        cublas = md.version("nvidia-cublas")
    except Exception:  # noqa: BLE001 — conda 経由等で入っていない構成もある
        cublas = "不明"
    detail = (
        f"numpy={numpy.__version__} torchvision={torchvision.__version__} "
        f"transformers={transformers.__version__} nvidia-cublas={cublas}"
    )

    # cu12 系と cu13 系の nvidia wheel は同じ site-packages/nvidia/*/lib へ展開される。
    # 同居すると torch が読む cuDNN / cuBLAS が別 CUDA 系のもので上書きされ、GPU 計算が
    # 静かに壊れる（配布環境の setup.sh も同じ検査をしている）。
    names = {d.metadata["Name"].lower() for d in md.distributions() if d.metadata["Name"]}
    cu12 = sorted(n for n in names if n.startswith("nvidia-") and n.endswith("-cu12"))
    cu13 = sorted(n for n in names if n.startswith("nvidia-") and n.endswith("-cu13"))
    if cu12 and cu13:
        record(
            "FAIL",
            "C3 依存の版",
            f"{detail} — CUDA 12 系と 13 系の nvidia wheel が同居している "
            f"(cu12: {cu12[:3]}... / cu13: {cu13[:3]}...)",
        )
        return
    record("PASS", "C3 依存の版", f"{detail} nvidia_wheels={'cu13' if cu13 else 'cu12' if cu12 else 'なし'}")


# ---------------------------------------------------------------------------
# C4 transformers 差し替え — バイト一致で見る
# ---------------------------------------------------------------------------
@check("C4 transformers 差し替え")
def _c4() -> None:
    import transformers

    tdir = Path(transformers.__file__).parent
    src_root = IVLA_REPO / "src/lerobot/policies/internvla_a1_5/transformers_replace/models"
    if not src_root.is_dir():
        record("FAIL", "C4 transformers 差し替え", f"上流に差し替え元が無い: {src_root}")
        return

    total = matched = 0
    mismatched: list[str] = []
    for src in src_root.rglob("*.py"):
        rel = src.relative_to(src_root)
        dst = tdir / "models" / rel
        total += 1
        if dst.is_file() and dst.read_bytes() == src.read_bytes():
            matched += 1
        else:
            mismatched.append(str(rel))

    # Qwen3.5 の差し替えが入っていなければ学習は起動しない（Gated DeltaNet / action expert のフック）。
    qwen = tdir / "models/qwen3_5/modeling_qwen3_5.py"
    if matched == total and qwen.is_file():
        record("PASS", "C4 transformers 差し替え", f"{matched}/{total} ファイルがバイト一致")
    else:
        record(
            "FAIL",
            "C4 transformers 差し替え",
            f"{matched}/{total} 一致 / qwen3_5={qwen.is_file()} 不一致={mismatched[:5]}",
        )


# ---------------------------------------------------------------------------
# C5 上流 lerobot + provenance
# ---------------------------------------------------------------------------
@check("C5 上流 lerobot")
def _c5() -> None:
    import lerobot
    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (  # noqa: F401
        InternVLAA15Config,
    )

    loaded = Path(lerobot.__file__).resolve().parent
    expected = (IVLA_REPO / "src/lerobot").resolve()
    if loaded != expected:
        record("FAIL", "C5 上流 lerobot", f"別の lerobot が import された: {loaded}（想定 {expected}）")
        return

    prov = json.loads((IVLA_DIR / "lerobot_policy_parc/upstream_provenance.json").read_text())
    bad = []
    for rel, want in prov.items():
        path = IVLA_REPO / "src" / rel
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if got != want:
            bad.append(rel)
    if bad:
        record("WARN", "C5 上流 lerobot", f"{loaded} — provenance 不一致 {len(bad)}/{len(prov)}: {bad}")
    else:
        record("PASS", "C5 上流 lerobot", f"{loaded} — provenance {len(prov)}/{len(prov)} 一致")


# ---------------------------------------------------------------------------
# C6 overlay
# ---------------------------------------------------------------------------
@check("C6 overlay 登録")
def _c6() -> None:
    sys.path.insert(0, str(IVLA_DIR))
    overlay = importlib.import_module("lerobot_policy_parc")
    summary = overlay.assert_installed()
    record("PASS", "C6 overlay 登録", f"robot_types={summary.get('robot_types')}")


# ---------------------------------------------------------------------------
# C7 torchcodec — 実データを本当にデコードする
# ---------------------------------------------------------------------------
@check("C7 torchcodec 実デコード")
def _c7() -> None:
    import torch
    import torchcodec
    from torchcodec.decoders import VideoDecoder

    videos = sorted(DATASET_ROOT.glob("videos/*/chunk-*/*.mp4"))
    if not videos:
        record(
            "WARN",
            "C7 torchcodec 実デコード",
            f"torchcodec={torchcodec.__version__} — データセットの mp4 が無い（{DATASET_ROOT}）。"
            "展開後に再実行すること",
        )
        return

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    details = []
    for path in videos[:1] + [v for v in videos if v.parent.parent != videos[0].parent.parent][:1]:
        dec = VideoDecoder(str(path))
        frame = dec[0]  # [C,H,W] uint8
        arr = frame.to(torch.uint8).cpu()
        nonzero = int((arr != 0).sum())
        # ★ ABI 不一致は「例外 → ゼロ埋め」で表に出ないので、非ゼロを assert する。
        assert nonzero > 0, f"デコード結果が全ゼロ: {path}"
        assert arr.shape[0] == 3, f"想定外の shape: {tuple(arr.shape)}"
        stem = path.parent.parent.name.replace(".", "_")
        out = ARTIFACTS / f"torchcodec_decode_{stem}.png"
        try:
            from torchvision.utils import save_image

            save_image(arr.float() / 255.0, out)
        except Exception:  # noqa: BLE001 — PNG が書けなくても検証自体は成立する
            out = None
        details.append(
            f"{path.parent.parent.name}: shape={tuple(arr.shape)} "
            f"mean={arr.float().mean():.1f} nonzero={nonzero}"
            + (f" -> {out.name}" if out else "")
        )
    record("PASS", "C7 torchcodec 実デコード", f"torchcodec={torchcodec.__version__} " + " | ".join(details))


# ---------------------------------------------------------------------------
# C8 オプショナル依存
# ---------------------------------------------------------------------------
@check("C8 オプショナル依存")
def _c8() -> None:
    found = []
    for mod in ("fla", "causal_conv1d", "flash_attn"):
        try:
            m = importlib.import_module(mod)
            found.append(f"{mod}={getattr(m, '__version__', 'ok')}")
        except Exception:  # noqa: BLE001 — 無いのが正常なケースもある
            found.append(f"{mod}=なし")
    # fla / causal_conv1d は無くても純 torch にフォールバックする（手順書 §3.3）。
    # ただし Gated DeltaNet の速度が数倍違うので、推論レイテンシ（計画 R6）の
    # 判断材料として必ず記録する。
    #
    # **flash_attn だけは別扱い。** 動画ヘッド（action_loss_only=false）では必須で、
    # 無いと最初の video_loss 計算で AssertionError になる
    # （wan/modules/model.py:249 -> attention.py:112 の `assert FLASH_ATTN_2_AVAILABLE`）。
    video_head = os.environ.get("IVLA_ACTION_LOSS_ONLY", "false").lower() != "true"
    has_flash_attn = "flash_attn=なし" not in found
    if video_head and not has_flash_attn:
        record(
            "FAIL",
            "C8 オプショナル依存",
            " ".join(found)
            + " — 動画ヘッド（action_loss_only=false）には flash-attn が必須。"
            "`FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=$(nproc) pip install --no-build-isolation flash-attn==2.8.3`"
            "（30〜60 分）か、IVLA_ACTION_LOSS_ONLY=true で WAN 分岐を切ること",
        )
        return
    record("WARN" if "なし" in " ".join(found) else "PASS", "C8 オプショナル依存", " ".join(found))


# ---------------------------------------------------------------------------
print("=" * 78)
fails = [r for r in _RESULTS if r[0] == "FAIL"]
warns = [r for r in _RESULTS if r[0] == "WARN"]
print(f"PASS={len(_RESULTS) - len(fails) - len(warns)}  WARN={len(warns)}  FAIL={len(fails)}")
if fails:
    print("\n失敗した項目:")
    for _, name, detail in fails:
        print(f"  - {name}: {detail}")
print("=" * 78)
sys.exit(1 if fails else 0)
