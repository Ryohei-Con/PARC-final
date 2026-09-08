"""ポリシーサーバー（InternVLA-A1.5 / PARC 2026 提出物）

submission_template/policy_server.py のコピー。**MyPolicy の中身と、先頭の
sys.path 注入ブロックだけ**を差し替えている（移植手順書 B9）。
BasePolicy / シリアライゼーション / FastAPI エンドポイントは 1 文字も変えない。

差分が本当にそれだけであることは
tests/test_policy_server_parity.py が AST で機械検証している。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

# 起動方法（python policy_server.py / uvicorn policy_server:app / 別 cwd）に
# 依らず internvla_runtime を import できるようにする（移植手順書 B9）。
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。渡されるのは以下の6キーのみ（画像2 + 固有受容4）。
                物体の絶対座標などは渡されない（画像から推定すること）:
                - "agentview_image": (128, 128, 3) uint8  — 正面カメラ
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8  — 手首カメラ
                - "robot0_joint_pos": (7,) float  — アーム関節角
                - "robot0_eef_pos": (3,) float  — エンドエフェクタ位置
                - "robot0_eef_quat": (4,) float  — エンドエフェクタ姿勢(quat)
                - "robot0_gripper_qpos": (2,) float  — グリッパ開度

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
                絶対座標ではなく **デルタ制御**（LIBERO の OSC_POSE コントローラ）。
                前 6 次元は EEF の相対移動・回転で通常 [-1, 1]、
                7 次元目は gripper 開閉指令。
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    """InternVLA-A1.5 のポリシー。

    実体は internvla_runtime.InternVLARuntime。ここは薄いアダプタに留める。
    runtime の構築は build_submission_runtime() を通す。verify_inference.py も
    同じ関数を通るので、自己チェックと本番が同じ経路になる（移植手順書 B8）。
    """

    def __init__(self, config_path: str | None = None):
        from internvla_runtime import build_submission_runtime

        self.runtime = build_submission_runtime(config_path)
        # 起動 120 秒制限の内訳を見るため、ロードはここで済ませて時間を出す。
        load_sec = self.runtime.load()
        print(f"[MyPolicy] runtime loaded in {load_sec:.1f}s", flush=True)

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        action = self.runtime.get_action(obs)
        return np.asarray(action, dtype=np.float32).reshape(7)

    def reset(self, instruction: str = "") -> None:
        self.runtime.reset(instruction=instruction)
        self.instruction = instruction


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
