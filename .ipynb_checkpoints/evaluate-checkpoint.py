import argparse
import atexit
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile


_LIBERO_PATH = "/workspace/LIBERO-plus"
_PIPELINE_PATH = "/workspace"
_COMPE_PATH = "/workspace/compe"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import validate_submission as _vs
except Exception as _e:
    _vs = None
    print(f"[evaluate] 警告: validate_submission を読み込めません: {_e}", file=sys.stderr)

try:
    import sandbox as _sb
except Exception as _e:
    _sb = None
    print(f"[evaluate] 警告: sandbox を読み込めません: {_e}", file=sys.stderr)

_EVAL_SECRET_PATHS = [
    "/workspace/LIBERO-plus",
    "/workspace/compe",
    "/workspace/pipeline",
    "/workspace/scoring_config.json",
    "/workspace/total_score_config.json",
    "/workspace/normalization_config.json",
]


def log(msg: str) -> None:
    print(f"[evaluate] {msg}", file=sys.stderr, flush=True)


class EvaluationInfraError(Exception):
    """評価インフラ側の障害（参加者のスコアとして記録してはいけない失敗）"""


def error_result(message: str) -> dict:
    return {
        "public_score": 0.0,
        "private_score": 0.0,
        "score_target": "public",
        "base_passed": False,
        "course_top_message": message,
    }


def find_submission_zip(args: list[str]) -> tuple[str, list[str]]:
    if args and os.path.isfile(args[0]):
        return args[0], args[1:]

    path = os.environ.get("USERSUBMISSION", "/home/app/assets/submission.zip")
    return path, args


def validate_and_unzip(zip_path: str, dest_dir: str) -> str:
    if _vs is None:
        log("提出物を展開（バリデータ無し・フォールバック）")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        return dest_dir

    log("提出物を静的検査")
    report = _vs.validate_zip(zip_path)
    for sev, code, msg in report.items:
        log(f"  [{sev}] {code}: {msg}")
    if not report.ok:
        reasons = "; ".join(f"{code}: {msg}"
                            for s, code, msg in report.items if s == _vs.ERROR)
        raise SubmissionRejected(f"提出物バリデーション失敗: {reasons}")

    log(f"提出物を安全に展開: {zip_path}")
    root = _vs.safe_extract(zip_path, dest_dir)
    return str(root)


class SubmissionRejected(Exception):
    """静的バリデーションで提出物が拒否された（参加者起因の即答エラー）"""


def install_requirements(submission_dir: str) -> None:
    req_path = os.path.join(submission_dir, "requirements.txt")
    if os.path.isfile(req_path):
        log("追加依存をインストール")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_path],
            check=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )


def start_policy_server(
    submission_dir: str, port: int, log_fp=None
) -> subprocess.Popen:
    log(f"ポリシーサーバーを起動 (port={port})")
    out = log_fp if log_fp is not None else sys.stderr
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "policy_server.py", "--port", str(port)],
        cwd=submission_dir,
        stdout=out,
        stderr=out,
        env=env,
    )
    return proc


def wait_for_server(
    port: int, proc: subprocess.Popen | None = None, timeout: int | None = None
) -> bool:
    if timeout is None:
        timeout = int(os.environ.get("SERVER_TIMEOUT", "120"))
    log(f"サーバーの起動を待機... (上限 {timeout}秒)")
    url = f"http://localhost:{port}/health"
    for elapsed in range(timeout):
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    log(f"サーバー起動確認 ({elapsed}秒)")
                    return True
        except Exception:
            pass
        if proc is not None and proc.poll() is not None:
            log(f"サーバープロセスが終了 (exit={proc.returncode}, {elapsed}秒)")
            return False
        time.sleep(1)
    return False


def read_log_tail(path: str, max_chars: int = 1500) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            data = f.read()
        return data[-max_chars:].strip()
    except OSError:
        return ""


def run_pipeline(
    server_url: str,
    tracks: list[str],
    output_dir: str,
    extra_args: list[str],
) -> int:
    log("評価パイプラインを実行")
    cmd = [
        sys.executable,
        "-m",
        "pipeline",
        "--server-url",
        server_url,
        "--track",
        *tracks,
        "--output-dir",
        output_dir,
        "--quiet",
    ]
    cmd += [*extra_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{_LIBERO_PATH}:{_PIPELINE_PATH}:{_COMPE_PATH}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    result = subprocess.run(
        cmd, stdout=sys.stderr, stderr=sys.stderr, env=env, cwd=_PIPELINE_PATH
    )
    return result.returncode


def build_omnicampus_result(output_dir: str, submission_id: str | None = None) -> dict:
    if submission_id:
        result_path = os.path.join(output_dir, f"{submission_id}.json")
        if not os.path.isfile(result_path):
            return error_result("評価エラー: 結果ファイルが見つかりません")
    else:
        files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
        if not files:
            return error_result("評価エラー: 結果ファイルが見つかりません")
        result_path = files[-1]

    with open(result_path) as f:
        result = json.load(f)

    tracks = result.get("tracks", [])
    if not tracks:
        return error_result("評価エラー: トラック結果がありません")

    errored = [
        t.get("track", "?")
        for t in tracks
        if t.get("overall_metrics", {}).get("error")
    ]
    if errored:
        raise EvaluationInfraError(
            f"トラック評価がインフラ障害で失敗しました: {', '.join(errored)} "
            f"（stderr のログを確認して再実行してください）"
        )

    scores = [t["overall_score"] for t in tracks]
    public_score = sum(scores) / len(scores)
    private_score = public_score

    base_passed = public_score > 0.0

    track_details = ", ".join(f"{t['track']}: {t['overall_score']:.3f}" for t in tracks)
    if base_passed:
        msg = f"Score: {public_score:.4f} ({track_details})"
    else:
        msg = f"Success rate: 0% ({track_details})"

    return {
        "public_score": round(public_score, 6),
        "private_score": round(private_score, 6),
        "score_target": "public",
        "base_passed": base_passed,
        "course_top_message": msg,
    }


def main() -> None:
    zip_path, extra_args = find_submission_zip(sys.argv[1:])

    if not os.path.isfile(zip_path):
        log(f"エラー: 提出物が見つかりません: {zip_path}")
        json.dump(error_result(f"提出物が見つかりません: {zip_path}"), sys.stdout)
        sys.exit(1)

    submission_dir = tempfile.mkdtemp(prefix="submission_")
    venv_dir = submission_dir + "_venv"
    server_log_path = submission_dir + "_server.log"
    server_proc = None
    tee_proc = None

    def cleanup():
        nonlocal server_proc
        if server_proc is not None and server_proc.poll() is None:
            log(f"ポリシーサーバーを停止 (PID={server_proc.pid})")
            try:
                import signal
                os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                server_proc.wait(timeout=10)
            except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    server_proc.kill()
            except Exception:
                server_proc.terminate()
        if tee_proc is not None:
            try:
                tee_proc.stdin.close()
                tee_proc.wait(timeout=5)
            except Exception:
                tee_proc.kill()
        shutil.rmtree(submission_dir, ignore_errors=True)
        shutil.rmtree(venv_dir, ignore_errors=True)
        try:
            os.remove(server_log_path)
        except OSError:
            pass

    atexit.register(cleanup)

    try:
        try:
            submission_root = validate_and_unzip(zip_path, submission_dir)
        except SubmissionRejected as e:
            log(f"提出物を拒否: {e}")
            json.dump(error_result(str(e)), sys.stdout)
            sys.exit(1)

        port = int(os.environ.get("SERVER_PORT", "8000"))
        output_dir = os.environ.get("EVAL_OUTPUT_DIR", "/workspace/results")

        server_python = sys.executable
        if _sb is not None:
            venv_py = _sb.create_submission_venv(submission_root, venv_dir, log=log)
            if venv_py:
                server_python = venv_py
            else:
                install_requirements(submission_root)
        else:
            install_requirements(submission_root)

        drop_to = None
        if _sb is not None:
            drop_to = _sb.resolve_unprivileged_ids()
            if _sb.is_root() and drop_to is not None:
                log(f"サンドボックス: 参加者サーバーを '{drop_to[2]}' で起動し FS を遮断")
                _sb.harden_filesystem(output_dir, submission_root, venv_dir,
                                      drop_to, _EVAL_SECRET_PATHS, log=log)
            else:
                os.makedirs(output_dir, exist_ok=True)
                if not _sb.is_root():
                    log("警告: 非 root 実行のため特権降格・FS 遮断をスキップ"
                        "（偽スコア注入対策が限定的）。venv 分離のみ有効")
                elif drop_to is None:
                    log("警告: 非特権ユーザー(nobody 等)が見つからず特権降格をスキップ")
        else:
            os.makedirs(output_dir, exist_ok=True)

        log(f"ポリシーサーバーを起動 (port={port}, python={server_python})")
        server_out = sys.stderr
        try:
            tee_proc = subprocess.Popen(
                ["tee", server_log_path],
                stdin=subprocess.PIPE,
                stdout=sys.stderr,
            )
            server_out = tee_proc.stdin
        except OSError:
            tee_proc = None
        if _sb is not None:
            server_proc = _sb.start_server_process(
                server_python, submission_root, port, drop_to, server_out)
        else:
            server_proc = start_policy_server(submission_root, port, server_out)

        if not wait_for_server(port, server_proc):
            if server_proc.poll() is not None:
                msg = ("ポリシーサーバーが起動に失敗しました"
                       f" (exit code {server_proc.returncode})")
            else:
                msg = "サーバーが制限時間内に起動しませんでした"
            log(f"エラー: {msg}")
            if tee_proc is not None:
                try:
                    tee_proc.stdin.close()
                    tee_proc.wait(timeout=5)
                except Exception:
                    pass
            tail = read_log_tail(server_log_path)
            if tail:
                msg += "\n--- サーバーログ末尾 ---\n" + tail
            json.dump(error_result(msg), sys.stdout)
            sys.exit(1)

        server_url = f"http://localhost:{port}"

        valid_tracks = ["track1", "track2", "track3"]
        tracks = os.environ.get("TRACK", "").replace(",", " ").split()
        invalid = [t for t in tracks if t not in valid_tracks]
        if invalid:
            log(f"エラー: TRACK の値が不正です: {invalid} (有効: {valid_tracks})")
            json.dump(error_result(f"TRACK の値が不正です: {invalid}"), sys.stdout)
            sys.exit(1)
        if not tracks:
            tracks = valid_tracks
        log(f"評価対象トラック: {tracks}")

        returncode = run_pipeline(server_url, tracks, output_dir, extra_args)
        if returncode != 0:
            log(f"エラー: パイプラインが終了コード {returncode} で失敗")
            json.dump(
                error_result(f"パイプラインエラー (exit={returncode})"), sys.stdout
            )
            sys.exit(1)

        omnicampus_result = build_omnicampus_result(output_dir, f"server_{port}")
        json.dump(omnicampus_result, sys.stdout)

        log("完了")

    except EvaluationInfraError as e:
        log(f"評価インフラ障害: {e}")
        json.dump(error_result(f"内部エラー: {e}"), sys.stdout)
        sys.exit(1)

    except Exception as e:
        log(f"予期しないエラー: {e}")
        json.dump(error_result(f"内部エラー: {e}"), sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
