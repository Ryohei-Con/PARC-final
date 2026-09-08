"""チャンク境界の平滑化 — RTC guidance と temporal ensembling の純ロジック。

**このモジュールは numpy と torch だけに依存する。** lerobot もモデルも import しない
ので、GPU 無しの手元で完全に単体テストできる（計画 §5.3 / SC-2）。

2 つの方式を実装する。``runtime_config.json`` の ``chunking.mode`` 1 キーで切り替わる:

``"rtc"``
    Real-Time Chunking 風の guidance。前チャンクの重なり区間を「軟らかい目標」として
    denoise の速度場に混ぜる。重み ``W`` は先頭ほど強く、重なりの終端で 0 になる。
``"ensemble"``
    Temporal ensembling。重複するチャンク同士を指数加重平均する。実装が単純で、
    提出物としてのバグ面積が小さい（計画 D4）。
``"none"``
    平滑化なし。ベースライン。

**ハードマスクは採らない**（計画 D2）。``w_max`` は 1.0 未満に制約する。``w_max = 1.0``
は「重なり区間を前チャンクで上書きする」のと等価で、モデルが新しい観測から得た
情報を捨ててしまう。既定は 0.8。

グリッパ次元（index 6）と padding 次元（7 以降）は ``dim_mask=False`` にして guidance
から外す（計画 F9: libero schema は ``action_reorder`` を持たないのでグリッパは
index 6 のまま）。グリッパは二値の離散量なので、連続的な平滑化を掛けると開閉が
遅れて掴み損ねる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

__all__ = [
    "GuidanceSpec",
    "build_guidance",
    "euler_integrate",
    "sample_actions_guided",
    "TemporalEnsembler",
    "DEFAULT_W_MAX",
    "SCHEDULES",
]

#: RTC のソフト重み上限の既定値。**1.0 未満に保つこと**（ハードマスク不採用）。
DEFAULT_W_MAX: float = 0.8

#: 重みスケジュール。いずれも j=0 で最大、j=L で 0 の単調非増加。
SCHEDULES: tuple[str, ...] = ("linear", "cosine")

#: LIBERO の実次元数（xyz + rpy + gripper）。7 以降は padding。
REAL_ACTION_DIM: int = 7

#: 既定の guidance 除外次元（gripper）。
DEFAULT_EXCLUDE_DIMS: tuple[int, ...] = (6,)


@dataclass
class GuidanceSpec:
    """RTC guidance の 3 点セット（すべて正規化空間）。

    Attributes:
        weights: ``[chunk]`` float32、``[0, 1)``。重なり区間で減衰するソフト重み。
        target: ``[chunk, D]`` float32。前チャンクを時間軸で揃えたもの。
        dim_mask: ``[D]`` bool。``False`` の次元は guidance の対象外。
    """

    weights: np.ndarray
    target: np.ndarray
    dim_mask: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        self.target = np.asarray(self.target, dtype=np.float32)
        self.dim_mask = np.asarray(self.dim_mask, dtype=bool).reshape(-1)

        if self.target.ndim != 2:
            raise ValueError(f"GuidanceSpec.target must be [chunk, D], got {self.target.shape}")
        chunk, dim = self.target.shape
        if self.weights.shape[0] != chunk:
            raise ValueError(
                f"GuidanceSpec.weights has length {self.weights.shape[0]}, expected {chunk}"
            )
        if self.dim_mask.shape[0] != dim:
            raise ValueError(
                f"GuidanceSpec.dim_mask has length {self.dim_mask.shape[0]}, expected {dim}"
            )
        if np.any(self.weights < 0.0) or np.any(self.weights >= 1.0):
            raise ValueError(
                "GuidanceSpec.weights must lie in [0, 1) -- hard masking (w=1) is not supported"
            )
        if not np.all(np.isfinite(self.target)):
            raise ValueError("GuidanceSpec.target contains non-finite values")

    @property
    def chunk_size(self) -> int:
        return int(self.target.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.target.shape[1])

    @property
    def overlap(self) -> int:
        """重みが正の区間長 L。"""
        return int(np.count_nonzero(self.weights > 0.0))


def _weight_schedule(length: int, chunk_size: int, w_max: float, schedule: str) -> np.ndarray:
    weights = np.zeros(chunk_size, dtype=np.float32)
    if length <= 0:
        return weights
    j = np.arange(length, dtype=np.float32)
    if schedule == "linear":
        weights[:length] = w_max * (1.0 - j / float(length))
    elif schedule == "cosine":
        weights[:length] = w_max * 0.5 * (1.0 + np.cos(math.pi * j / float(length)))
    else:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    return weights.astype(np.float32)


def build_guidance(
    prev_chunk_norm: np.ndarray | None,
    replan_steps: int,
    chunk_size: int,
    action_dim: int,
    *,
    w_max: float = DEFAULT_W_MAX,
    schedule: str = "linear",
    exclude_dims: tuple[int, ...] = DEFAULT_EXCLUDE_DIMS,
    real_action_dim: int = REAL_ACTION_DIM,
) -> GuidanceSpec | None:
    """前チャンクから guidance を作る。重なりが無ければ ``None``。

    Args:
        prev_chunk_norm: 正規化空間の前チャンク ``[chunk_size, D]``。``None`` なら
            ``None`` を返す（エピソード先頭）。
        replan_steps: 前回の推論から進んだ env ステップ数。
        chunk_size: モデルが返すチャンク長。
        action_dim: モデルの action 次元（padding 込み。通常 32）。
        w_max: ソフト重みの上限。``0 <= w_max < 1``。
        schedule: ``"linear"`` または ``"cosine"``。
        exclude_dims: guidance を掛けない次元（既定は gripper の 6）。
        real_action_dim: padding でない次元数。これ以降は自動的に除外する。

    Returns:
        ``GuidanceSpec`` または ``None``。

    Raises:
        ValueError: 引数が不正（``w_max >= 1`` を含む）。
    """
    if prev_chunk_norm is None:
        return None

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {replan_steps}")
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if not (0.0 <= w_max < 1.0):
        raise ValueError(
            f"w_max must satisfy 0 <= w_max < 1 (soft mask only), got {w_max}. "
            "w_max = 1.0 is a hard mask and is deliberately not supported."
        )

    prev = np.asarray(prev_chunk_norm, dtype=np.float32)
    if prev.ndim != 2:
        raise ValueError(f"prev_chunk_norm must be [chunk, D], got {prev.shape}")
    if prev.shape[0] != chunk_size:
        raise ValueError(
            f"prev_chunk_norm has {prev.shape[0]} steps but chunk_size={chunk_size}"
        )
    if prev.shape[1] != action_dim:
        raise ValueError(
            f"prev_chunk_norm has action_dim={prev.shape[1]} but expected {action_dim}"
        )

    overlap = chunk_size - replan_steps
    if overlap <= 0:
        # 重なりが無いので平滑化のしようがない。
        return None

    target = np.zeros((chunk_size, action_dim), dtype=np.float32)
    target[:overlap] = prev[replan_steps:replan_steps + overlap]

    weights = _weight_schedule(overlap, chunk_size, float(w_max), schedule)

    dim_mask = np.zeros(action_dim, dtype=bool)
    dim_mask[: min(real_action_dim, action_dim)] = True
    for dim in exclude_dims:
        index = int(dim)
        if 0 <= index < action_dim:
            dim_mask[index] = False

    return GuidanceSpec(weights=weights, target=target, dim_mask=dim_mask)


def euler_integrate(
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    num_steps: int,
    guidance: GuidanceSpec | None = None,
) -> torch.Tensor:
    """上流 ``modeling_internvla_a1_5.py:807-833`` の euler ループの再実装。

    上流と同じ離散化::

        dt = -1 / num_steps
        time = 1.0
        while time >= -dt / 2:
            v = denoise_fn(x_t, time)
            x_t = x_t + dt * v
            time += dt

    ``guidance`` を与えると、対象次元だけ速度場を差し替える::

        t_safe = clamp(time, min=|dt|)                 # 防御。0 除算は起きない（下の note）
        v_guide = (x_t - target) / t_safe              # t=0 で x_t を target に落とす速度
        v = where(dim_mask & (W > 0), (1 - W) * v_model + W * v_guide, v_model)

    .. note:: **``t_safe`` のクランプは「0 除算を防ぐ」ものではない（防御コードである）。**

       上のループの ``time`` は ``1.0`` から ``|dt| = 1/N`` ずつ下がり、最後に
       ``denoise_fn`` を呼ぶときの値は ``1/N`` である（``time`` が ``0`` になるのは
       最終更新の**後**で、そのときループ条件 ``time >= -dt/2`` が偽になる）。
       つまり ``time`` は 0 に到達しないので、``(x_t - target) / time`` は
       クランプが無くても発散しない。

       厳密な実数演算ならクランプは恒等（binding しない）。ただし ``time`` は
       ``float32`` で ``time += dt`` と累積するため、最終値は ``1/N`` から
       数 ULP ずれる。**実測では ``num_steps`` が 10 / 20 / 25 のとき ``|dt|`` を
       わずかに下回り、クランプが実際に binding する**（相対 ~1e-6。既定の
       ``num_inference_steps=10`` はこれに該当する）。効果は丸め誤差の範囲で、
       数値的な意味は無い。この列挙自体は
       ``test_the_time_clamp_binds_at_the_default_num_steps`` が固定している。

       したがってこのクランプの実際の役割は、``num_steps`` や時刻スケジュールを
       変えた場合（``time`` を厳密に 0 まで下げる離散化、非一様スケジュール等）に
       備えた**防御**である。この性質—「``time`` は 0 に到達しない」「クランプの
       有無で結果は丸め誤差以上には変わらない」—は
       ``tests/test_chunk_blending.py`` の
       ``test_timesteps_never_reach_zero_so_the_clamp_is_not_a_zero_division_guard``
       と ``test_the_time_clamp_never_changes_the_result_beyond_float_rounding``
       が固定している。**後者はクランプを外しても通る**（クランプが load-bearing で
       ないことがまさに主張の内容だから）。

    .. note:: **計画 §5.3 の擬似コードとの相違（符号）**

       計画には ``v = ... W*(y - x_t)/t_safe`` と書かれているが、これは符号が逆で、
       適用すると ``x`` が ``y`` から**遠ざかる**。上流の flow matching の規約は
       ``modeling_internvla_a1_5.py:1125-1126``::

           x_t = time * noise + (1 - time) * actions
           u_t = noise - actions          # <- これが v の回帰目標

       つまり ``v`` は「clean な actions から noise へ向かう向き」で、
       ``(x_t - actions) / time`` に等しい。サンプリングは ``dt = -1/N`` で
       ``x += dt * v`` と積分するので、``v_guide = (x_t - y) / t`` にして初めて
       ``x`` が ``y`` へ収束する。``w -> 1`` のとき最終ステップで厳密に
       ``x = y`` になることを ``tests/test_chunk_blending.py`` で確認している。

    ``W == 0`` の位置は ``torch.where`` が ``v_model`` をそのまま選ぶので、
    ``guidance=None`` と**ビット一致**する（テストで担保）。

    Args:
        denoise_fn: ``(x_t, timestep) -> v``。``timestep`` は ``[B]``。
            モデル無しのスタブに差し替えられる（これが単体テスト可能性の要）。
        x0: 初期ノイズ ``[B, chunk, D]``。
        num_steps: euler ステップ数。
        guidance: ``GuidanceSpec`` または ``None``。

    Returns:
        ``x_final``（``x0`` と同じ shape / dtype / device）。
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if x0.ndim != 3:
        raise ValueError(f"x0 must be [B, chunk, D], got {tuple(x0.shape)}")

    device = x0.device
    batch = x0.shape[0]

    weights_t: torch.Tensor | None = None
    target_t: torch.Tensor | None = None
    active_t: torch.Tensor | None = None
    if guidance is not None:
        if (guidance.chunk_size, guidance.action_dim) != tuple(x0.shape[1:]):
            raise ValueError(
                f"guidance shape {(guidance.chunk_size, guidance.action_dim)} "
                f"does not match x0 {tuple(x0.shape[1:])}"
            )
        weights_t = torch.as_tensor(guidance.weights, dtype=torch.float32, device=device)
        weights_t = weights_t.view(1, -1, 1)
        target_t = torch.as_tensor(guidance.target, dtype=torch.float32, device=device)
        target_t = target_t.unsqueeze(0)
        mask_t = torch.as_tensor(guidance.dim_mask, dtype=torch.bool, device=device)
        mask_t = mask_t.view(1, 1, -1)
        active_t = mask_t & (weights_t > 0.0)

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    abs_dt = torch.abs(dt)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)

    x_t = x0
    while time >= -dt / 2:
        expanded_time = time.expand(batch)
        v_model = denoise_fn(x_t, expanded_time)

        if active_t is not None:
            # **防御的クランプ。0 除算を防いでいるわけではない。**
            # このループが denoise_fn を呼ぶときの time の最小値は 1/N (> 0) なので、
            # クランプが無くても (x - y)/time は発散しない。float32 の累積誤差で
            # 最終ステップの time が |dt| を数 ULP 下回ることがあり（N=10, 25 で実測）
            # その分だけ binding するが、効果は丸め誤差の範囲。時刻スケジュールを
            # 変えて time が 0 に達するようにした場合の保険として残す。
            # 詳細は docstring の note と tests/test_chunk_blending.py を参照。
            t_safe = torch.clamp(time, min=abs_dt)
            v_guide = (x_t - target_t.to(v_model.dtype)) / t_safe.to(v_model.dtype)
            w = weights_t.to(v_model.dtype)
            blended = (1.0 - w) * v_model + w * v_guide
            v_t = torch.where(active_t, blended, v_model)
        else:
            v_t = v_model

        x_t = x_t + dt.to(x_t.dtype) * v_t
        time = time + dt

    return x_t


@torch.no_grad()
def sample_actions_guided(
    model,
    *,
    pixel_values,
    image_grid_thw,
    lang_tokens,
    lang_masks,
    state,
    fast_token_mask=None,
    num_steps: int | None = None,
    noise=None,
    guidance: GuidanceSpec | None = None,
) -> torch.Tensor:
    """上流 ``InternVLAA15.sample_actions`` の guidance 付き版。

    上流 ``modeling_internvla_a1_5.py:760-833`` と**同じ手順**で prefix を埋め込み、
    KV cache を作ってから、euler ループだけを :func:`euler_integrate` に差し替える。
    ここで上流のコードをコピーしているのは overlay 方針（上流を編集しない）による。

    Note:
        この関数はモデルを必要とするので手元では実行できない。数値的な性質は
        :func:`euler_integrate` 側で単体テストしてある。
    """
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import make_att_2d_masks

    if num_steps is None:
        num_steps = model.config.num_inference_steps

    bsize = state.shape[0]
    device = state.device
    dtype = state.dtype

    if noise is None:
        actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
        noise = model.sample_noise(actions_shape, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        pixel_values, image_grid_thw, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids, _rope_deltas = model.get_position_ids(
        lang_tokens, image_grid_thw, prefix_pad_masks
    )

    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.qwen3_5_with_expert.qwen3_5.language_model.config._attn_implementation = "eager"

    _, past_key_values = model.qwen3_5_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        knowledge_insulation=model.config.knowledge_insulation,
    )

    max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

    if model.config.block_action_attend_fast_tokens:
        fast_mask = model._compute_fast_token_mask(lang_tokens, fast_token_mask)
    else:
        fast_mask = None

    def denoise_fn(x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return model.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            max_prefix_position_ids,
            x_t.to(dtype),
            timestep.to(dtype),
            fast_mask=fast_mask,
        )

    return euler_integrate(denoise_fn, noise, num_steps, guidance=guidance)


class TemporalEnsembler:
    """重複チャンクの指数加重平均（RTC の比較対象）。

    各チャンクは push された時刻からの経過ステップ数 ``offset`` を持つ。``pop()`` は
    生存しているチャンクの ``chunk[offset]`` を重み ``w_i = exp(-m * i)``（``i`` は
    チャンクの古さ、0 が最新）で凸結合して返す。

    ``newest_dims`` の次元は平均せず**最新チャンクの値をそのまま採る**。グリッパは
    二値の離散量なので、平均すると開閉の遷移が鈍って掴み損ねる。
    """

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        m: float = 0.1,
        newest_dims: tuple[int, ...] = DEFAULT_EXCLUDE_DIMS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if m < 0.0:
            raise ValueError(f"m must be >= 0, got {m}")

        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.m = float(m)
        self.newest_dims = tuple(
            int(d) for d in newest_dims if 0 <= int(d) < self.action_dim
        )
        self._chunks: list[np.ndarray] = []
        self._offsets: list[int] = []

    def reset(self) -> None:
        """エピソード境界で内部状態を消す。"""
        self._chunks.clear()
        self._offsets.clear()

    def __len__(self) -> int:
        return len(self._chunks)

    def push(self, chunk_norm: np.ndarray) -> None:
        """新しいチャンク ``[chunk_size, D]`` を登録する（最新として先頭に置く）。"""
        chunk = np.asarray(chunk_norm, dtype=np.float32)
        if chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"chunk must be {(self.chunk_size, self.action_dim)}, got {chunk.shape}"
            )
        if not np.all(np.isfinite(chunk)):
            raise ValueError("TemporalEnsembler.push got a non-finite chunk")

        self._chunks.insert(0, chunk.copy())
        self._offsets.insert(0, 0)
        self._drop_exhausted()

    def weights(self) -> np.ndarray:
        """現在の凸結合の重み（合計 1）。古さ順（先頭が最新）。"""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        raw = np.exp(-self.m * np.arange(len(self._chunks), dtype=np.float32))
        return (raw / raw.sum()).astype(np.float32)

    def pop(self) -> np.ndarray:
        """現在ステップの action ``[D]``（正規化空間）を返し、時間を 1 進める。"""
        if not self._chunks:
            raise RuntimeError("TemporalEnsembler.pop() called with no chunks; push() first")

        weights = self.weights()
        stacked = np.stack(
            [chunk[offset] for chunk, offset in zip(self._chunks, self._offsets)], axis=0
        )
        blended = (weights[:, None] * stacked).sum(axis=0).astype(np.float32)

        for dim in self.newest_dims:
            blended[dim] = stacked[0, dim]

        for index in range(len(self._offsets)):
            self._offsets[index] += 1
        self._drop_exhausted()

        return blended

    def _drop_exhausted(self) -> None:
        keep = [i for i, offset in enumerate(self._offsets) if offset < self.chunk_size]
        self._chunks = [self._chunks[i] for i in keep]
        self._offsets = [self._offsets[i] for i in keep]
