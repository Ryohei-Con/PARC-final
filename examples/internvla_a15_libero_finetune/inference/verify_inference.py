"""提出前の自己チェック（移植手順書 C2）。

**実機データもシミュレータも要らない。** ダミー観測（ハーネスが ``/act`` に送るのと
同じ shape）を作って、提出物が単体で成立しているかを確認する。

``policy_server.py`` と**同じ** :func:`internvla_runtime.build_submission_runtime` を
通ること（移植手順書 B8）。片方だけ別経路にすると自己チェックが本番と違うものを測る。

検査項目:

- vendor 版 lerobot と transformers パッチが実際に有効か（``__file__`` を印字）
- vendor lerobot と上流 ``src/lerobot`` のハッシュ一致件数（``--upstream-src`` 指定時）
- モデルのロード時間（120 秒制限）
- 推論レイテンシ（10 秒制限）。推論するステップとキャッシュを返すステップの両方
- action が ``(7,) float32`` / NaN Inf なし / ``action[6]`` が ``{-1, +1}``
- ``reset()`` でチャンクキャッシュと ensembler が消えること
- ``--chunking rtc|ensemble|none`` を切り替えて replan 境界の
  ``||a_t - a_{t-1}||`` を印字（平滑化が効いているかの直接指標）
- **モデル推論の回数が ``ceil(steps / min(replan_steps, chunk_size))`` であること**
  （計画 §8.3 の予算式の前提。毎ステップ推論に退化していても latency だけ見ると
  気づけない）。``ceil(steps / replan_steps)`` **ではない**: チャンクは自分の長さを
  超えて引き延ばせないので、``replan_steps > chunk_size`` では実装の方が正しい
  （:func:`expected_inference_calls` を参照）

モード切替は ``runtime.set_chunking_mode()`` を通す。``runtime.chunking["mode"]`` を
直接書き換えると ensembler の構築が伴わず、**別モードの数値を ensemble として印字**
することになる（手順書 B8: 自己チェックと本番を同じ経路に通す）。

使い方::

    python verify_inference.py --benchmark
    python verify_inference.py --chunking none --chunking rtc --chunking ensemble
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

#: 起動 120 秒 / 推論 10 秒（PARC-final: pipeline/remote_policy.py:42 と README）
LOAD_LIMIT_SEC = 120.0
ACT_LIMIT_SEC = 10.0


def make_dummy_obs(rng: np.random.Generator, camera: int = 128) -> dict[str, np.ndarray]:
    """ハーネスが ``/act`` に送るのと同じ 6 キーのダミー観測。"""
    return {
        "agentview_image": rng.integers(0, 256, (camera, camera, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (camera, camera, 3), dtype=np.uint8),
        "robot0_joint_pos": rng.normal(size=7).astype(np.float32),
        "robot0_eef_pos": rng.normal(size=3).astype(np.float32),
        "robot0_eef_quat": _random_unit_quat(rng),
        "robot0_gripper_qpos": rng.normal(size=2).astype(np.float32),
    }


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4).astype(np.float32)
    return (q / np.linalg.norm(q)).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_vendor_to_upstream(vendor_lerobot: Path, upstream_src: Path) -> dict:
    """vendor/lerobot と上流 src/lerobot のバイト一致件数を数える。"""
    upstream_root = Path(upstream_src) / "lerobot"
    same = 0
    differs: list[str] = []
    extra: list[str] = []
    for path in sorted(Path(vendor_lerobot).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(vendor_lerobot)
        counterpart = upstream_root / relative
        if not counterpart.is_file():
            extra.append(str(relative))
            continue
        if _sha256(path) == _sha256(counterpart):
            same += 1
        else:
            differs.append(str(relative))
    return {"identical": same, "different": differs, "only_in_vendor": extra}


def check_action(action: np.ndarray, gripper_values: tuple[float, float]) -> list[str]:
    problems: list[str] = []
    if not isinstance(action, np.ndarray):
        problems.append(f"action is {type(action).__name__}, expected np.ndarray")
        return problems
    if action.shape != (7,):
        problems.append(f"action.shape={action.shape}, expected (7,)")
    if action.dtype != np.float32:
        problems.append(f"action.dtype={action.dtype}, expected float32")
    if not np.all(np.isfinite(action)):
        problems.append(f"action contains NaN/Inf: {action}")
    if action.shape == (7,) and float(action[6]) not in gripper_values:
        problems.append(f"action[6]={action[6]} not in {gripper_values}")
    return problems


def expected_inference_calls(steps: int, replan_steps: int, chunk_size: int) -> int:
    """``steps`` ステップの間に走るべきモデル推論の回数。

    **``ceil(steps / replan_steps)`` ではない。** チャンクは自分の長さを超えて
    引き延ばせないので、実装（``InternVLARuntime.get_action`` の ``need_new_chunk``
    / ``InternVLARuntime.replan_interval``）は ``min(replan_steps, chunk_size)``
    ごとに引き直す。``replan_steps > chunk_size`` のときに ``ceil(steps/replan)``
    を期待すると、**正しい実装を「replan_steps is not being honoured」と誤報する**
    （replan=60 / chunk=50 / 300 step で actual 6 vs expected 5）。

    Args:
        steps: 流すステップ数。
        replan_steps: ``chunking.replan_steps``。
        chunk_size: モデルの action chunk 長。

    Returns:
        期待される推論回数。
    """
    steps = int(steps)
    if steps <= 0:
        return 0
    interval = replan_interval(replan_steps, chunk_size)
    return -(-steps // interval)  # ceil


def replan_interval(replan_steps: int, chunk_size: int) -> int:
    """実際の引き直し間隔 ``min(replan_steps, chunk_size)``（実装と同じ式）。"""
    interval = min(int(replan_steps), int(chunk_size))
    if interval < 1:
        raise ValueError(
            f"replan interval must be >= 1, got min({replan_steps}, {chunk_size}) = {interval}"
        )
    return interval


class _CallCounter:
    """``_predict_chunk_normalized`` の呼び出し回数を数える薄いラッパ。

    本番の経路（``get_action``）をそのまま使い、回数だけを外から観測する。

    **必ず外せること。** インスタンス属性で恒久的に上書きすると、``run()`` を
    programmatic に呼んだ場合や同じ runtime を使い回した場合にラッパが残り、
    以後の呼び出しを黙って数え続ける。``with`` で使うか :meth:`uninstall` を呼ぶ::

        with _CallCounter(runtime) as counter:
            ...
    """

    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._original = runtime._predict_chunk_normalized
        # 元がクラス側のメソッドだったのか、既にインスタンス属性だったのかを覚える。
        # 前者で無条件に代入して戻すと、外したはずのラッパの代わりに bound method が
        # インスタンス属性として残る（見た目は同じでも状態は元に戻っていない）。
        self._had_instance_attr = "_predict_chunk_normalized" in vars(runtime)
        self._installed = True
        self.count = 0

        def counting(sample, guidance):
            self.count += 1
            return self._original(sample, guidance)

        runtime._predict_chunk_normalized = counting

    def reset(self) -> None:
        """カウンタだけを 0 に戻す（ラッパは付けたまま）。"""
        self.count = 0

    def uninstall(self) -> None:
        """ラッパを外して元の状態に戻す。二重呼び出しは無害。"""
        if not self._installed:
            return
        if self._had_instance_attr:
            self._runtime._predict_chunk_normalized = self._original
        else:
            del self._runtime._predict_chunk_normalized
        self._installed = False

    def __enter__(self) -> "_CallCounter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.uninstall()
        return False


def run(args: argparse.Namespace) -> int:
    from internvla_runtime import build_submission_runtime

    rng = np.random.default_rng(args.seed)
    problems: list[str] = []

    print("=" * 60)
    print("verify_inference.py")
    print("=" * 60)

    started = time.time()
    runtime = build_submission_runtime(args.config)
    print(f"[config] {runtime.cfg.get('_path')}")
    print(f"[config] orientation={runtime.orientation} gripper={runtime.gripper}")
    print(f"[config] chunking={runtime.chunking}")

    import lerobot
    import transformers

    print(f"[import] lerobot      = {lerobot.__file__}")
    print(f"[import] transformers = {transformers.__file__} ({transformers.__version__})")
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5

        print(
            f"[import] modeling_qwen3_5 = {modeling_qwen3_5.__file__} "
            f"fast_path={getattr(modeling_qwen3_5, 'is_fast_path_available', 'n/a')}"
        )
    except Exception as exc:  # pragma: no cover - transformers 版に依存
        print(f"[import] modeling_qwen3_5 unavailable: {exc}")

    if args.upstream_src:
        vendor = _THIS_DIR / "vendor" / "lerobot"
        if vendor.is_dir():
            report = compare_vendor_to_upstream(vendor, Path(args.upstream_src))
            print(
                f"[vendor] identical={report['identical']} "
                f"different={len(report['different'])} only_in_vendor={report['only_in_vendor']}"
            )
            if report["different"]:
                problems.append(f"vendor/lerobot differs from upstream: {report['different'][:5]}")
        else:
            print(f"[vendor] no vendor/lerobot under {_THIS_DIR}")

    load_sec = runtime.load()
    print(f"[load] {load_sec:.1f}s (limit {LOAD_LIMIT_SEC:.0f}s)")
    if load_sec > LOAD_LIMIT_SEC:
        problems.append(f"model load took {load_sec:.1f}s > {LOAD_LIMIT_SEC:.0f}s")
    print(f"[load] total wall clock incl. bootstrap: {time.time() - started:.1f}s")

    gripper_values = (runtime.gripper["env_close"], runtime.gripper["env_open"])

    # モデル推論の**回数**を数える。計画 §8.3 の予算式
    # ``(300 / interval) * latency < 120s`` はここが interval =
    # ``min(replan_steps, chunk_size)`` ごとであることが前提。毎ステップ推論に
    # 退化していても latency の中央値だけ見ていると気づけない。
    # ラッパは with を抜けるときに必ず外す（runtime に残さない）。
    with _CallCounter(runtime) as inference_calls:
        problems.extend(_run_chunking_modes(runtime, args, rng, gripper_values, inference_calls))

    return _finish(runtime, rng, problems)


def _run_chunking_modes(runtime, args, rng, gripper_values, inference_calls) -> list[str]:
    """``--chunking`` の各モードを回して問題のリストを返す。"""
    problems: list[str] = []

    modes = args.chunking or [runtime.chunking["mode"]]
    for mode in modes:
        # **生の dict を書き換えないこと。** set_chunking_mode() が ensembler の
        # 構築 / 破棄とチャンク状態のリセットまで面倒を見る唯一の入口。dict を直接
        # 書き換えると ensemble を名乗ったまま実体は none 経路が走り、別モードの
        # 数値を ensemble として印字することになる（手順書 B8）。
        runtime.set_chunking_mode(mode)
        assert runtime.chunking["mode"] == mode
        if mode == "ensemble" and runtime._ensembler is None:
            problems.append("[ensemble] set_chunking_mode() did not build a TemporalEnsembler")
        if mode != "ensemble" and runtime._ensembler is not None:
            problems.append(f"[{mode}] the temporal ensembler was not torn down")
        runtime.reset(instruction="pick up the black bowl and place it on the plate")
        inference_calls.reset()
        latencies: list[float] = []
        actions: list[np.ndarray] = []

        for step in range(args.steps):
            obs = make_dummy_obs(rng)
            tick = time.time()
            action = runtime.get_action(obs)
            latencies.append(time.time() - tick)
            actions.append(np.asarray(action, dtype=np.float32))
            problems.extend(f"[{mode}] step{step}: {p}" for p in check_action(action, gripper_values))

        replan = int(runtime.chunking["replan_steps"])
        # **境界の判定も期待回数も ``min(replan, chunk_size)`` で数える。** 実装は
        # チャンク長を超えて引き延ばせないので、replan だけで数えると
        # replan > chunk_size のときに境界を取り違え、正しい実装を誤報する。
        interval = replan_interval(replan, runtime.chunk_size)
        jumps = [
            float(np.linalg.norm(actions[i][:6] - actions[i - 1][:6]))
            for i in range(1, len(actions))
        ]
        boundary = [jumps[i - 1] for i in range(1, len(actions)) if i % interval == 0]
        interior = [jumps[i - 1] for i in range(1, len(actions)) if i % interval != 0]

        print(
            f"[chunking={mode}] latency: first={latencies[0]:.2f}s "
            f"max={max(latencies):.2f}s median={float(np.median(latencies)):.3f}s"
        )
        print(
            f"[chunking={mode}] ||a_t - a_(t-1)||: "
            f"boundary_mean={float(np.mean(boundary)) if boundary else float('nan'):.4f} "
            f"interior_mean={float(np.mean(interior)) if interior else float('nan'):.4f}"
        )
        expected_calls = expected_inference_calls(args.steps, replan, runtime.chunk_size)
        episode_budget = (300.0 / interval) * float(np.median(latencies))
        print(
            f"[chunking={mode}] model inferences: {inference_calls.count} "
            f"(expected {expected_calls} for {args.steps} steps @ replan={replan}, "
            f"chunk_size={runtime.chunk_size} -> interval={interval}); "
            f"300-step episode budget ~= {episode_budget:.1f}s"
        )
        if interval != replan:
            print(
                f"[chunking={mode}] note: replan_steps={replan} > chunk_size="
                f"{runtime.chunk_size}, so the runtime re-plans every {interval} steps"
            )
        if inference_calls.count != expected_calls:
            problems.append(
                f"[{mode}] ran {inference_calls.count} model inferences over {args.steps} steps "
                f"but replan interval min(replan_steps={replan}, chunk_size="
                f"{runtime.chunk_size})={interval} implies {expected_calls}. "
                "replan_steps is not being honoured."
            )
        if max(latencies) > ACT_LIMIT_SEC:
            problems.append(f"[{mode}] /act latency {max(latencies):.2f}s > {ACT_LIMIT_SEC:.0f}s")

    return problems


def _finish(runtime, rng, problems: list[str]) -> int:
    """``reset()`` の検査をしてから合否を印字する。"""
    # reset がキャッシュを消すか。ensembler の検査を空振りさせないため、
    # **ensemble モードで実際にチャンクを溜めてから** reset する。
    runtime.set_chunking_mode("ensemble")
    if runtime._ensembler is None:
        problems.append("set_chunking_mode('ensemble') did not build a TemporalEnsembler")
    else:
        runtime.reset(instruction="ensembler reset probe")
        for _ in range(3):
            runtime.get_action(make_dummy_obs(rng))
        if len(runtime._ensembler) == 0:
            problems.append("the temporal ensembler stayed empty while stepping")
        runtime.reset(instruction="second episode")
        if len(runtime._ensembler) != 0:
            problems.append("reset() did not clear the temporal ensembler")
    if runtime._prev_chunk_norm is not None or runtime._chunk_cursor != 0:
        problems.append("reset() did not clear the chunk cache")
    if runtime._current_chunk_env is not None or runtime._has_chunk:
        problems.append("reset() did not clear the cached action chunk")

    print("=" * 60)
    if problems:
        print(f"FAIL ({len(problems)} problems)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="runtime_config.json のパス")
    parser.add_argument("--steps", type=int, default=40, help="ダミー観測を流すステップ数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--chunking",
        action="append",
        choices=["none", "rtc", "ensemble"],
        help="切り替えて比較する（複数指定可）",
    )
    parser.add_argument("--upstream-src", default=None, help="上流 src/ のパス（ハッシュ照合用）")
    parser.add_argument("--benchmark", action="store_true", help="レイテンシ表示を厚くする")
    args = parser.parse_args()
    if args.benchmark:
        args.steps = max(args.steps, 60)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
