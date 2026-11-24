#!/usr/bin/env python3
"""Run the control and lambda candidates sequentially on the isolated Matpool 5090."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


CURRENT = None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def handle_signal(signum, _frame):
    global CURRENT
    if CURRENT is not None and CURRENT.poll() is None:
        try:
            os.killpg(CURRENT.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


def run_logged(command, log_path, environment):
    global CURRENT
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] COMMAND {json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        CURRENT = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            text=True,
        )
        code = CURRENT.wait()
        CURRENT = None
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] EXIT {code}\n")
        log.flush()
        return code


def arm_for(value):
    return "base_post" if value == 0 else "cycle_post"


def training_command(args, value, preflight):
    command = [
        str(args.python),
        "-u",
        str(args.suite_dir / "fastvoicegrad_formal.py"),
        "--project-root",
        str(args.project_root),
        "--run-root",
        str(args.run_root),
        "--arm",
        arm_for(value),
        "--seed",
        "1234",
        "--lambda-cycle",
        format(value, ".8g"),
        "--parent",
        str(args.parent),
        "--screen-passes",
        str(args.screen_passes),
        "--metrics-endpoint",
        "--run-approved",
    ]
    if preflight:
        command.extend(("--session-update-limit", str(args.preflight_updates)))
    return command


def restart_vq(service):
    if not service:
        return {"skipped": True, "reason": "Matpool has no laboratory VQ service", "time_unix": time.time()}
    result = subprocess.run(
        ["systemctl", "--user", "start", service],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {"exit_code": result.returncode, "output": result.stdout[-2000:], "time_unix": time.time()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", default=(0.0, 0.5, 2.5, 10.0))
    parser.add_argument("--screen-passes", type=int, default=5)
    parser.add_argument("--preflight-updates", type=int, default=10)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--vq-service")
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "started_unix": time.time(),
        "lambdas": args.lambdas,
        "screen_passes": args.screen_passes,
        "preflight_updates": args.preflight_updates,
        "physical_gpu": args.gpu,
        "selection_uses_test": False,
        "vq_service_to_restart": args.vq_service,
        "candidates": {},
    }
    atomic_json(args.status, state)
    try:
        for value in args.lambdas:
            key = format(value, ".8g")
            candidate = {
                "arm": arm_for(value),
                "status": "preflight",
                "started_unix": time.time(),
                "attempts": [],
            }
            state["candidates"][key] = candidate
            state["current_lambda"] = value
            atomic_json(args.status, state)
            log_path = args.run_root / "queue_logs" / f"lambda_{key.replace('.', 'p')}.log"
            code = run_logged(training_command(args, value, True), log_path, environment)
            candidate["attempts"].append({"stage": "preflight", "exit_code": code, "time_unix": time.time()})
            if code != 0:
                candidate["status"] = "preflight_failed"
                candidate["finished_unix"] = time.time()
                atomic_json(args.status, state)
                continue
            candidate["status"] = "full_screen"
            atomic_json(args.status, state)
            code = run_logged(training_command(args, value, False), log_path, environment)
            candidate["attempts"].append({"stage": "full_screen", "exit_code": code, "time_unix": time.time()})
            if code != 0:
                time.sleep(30)
                code = run_logged(training_command(args, value, False), log_path, environment)
                candidate["attempts"].append(
                    {"stage": "automatic_resume_retry", "exit_code": code, "time_unix": time.time()}
                )
            candidate["status"] = "complete" if code == 0 else "failed_after_retry"
            candidate["finished_unix"] = time.time()
            atomic_json(args.status, state)
        state.pop("current_lambda", None)
        state["status"] = "ranking"
        atomic_json(args.status, state)
        ranking_output = args.run_root / "lambda_ranking.json"
        ranking_command = [
            sys.executable,
            str(args.suite_dir / "rank_lambda_screen.py"),
            "--run-root",
            str(args.run_root),
            "--lambdas",
            *[format(value, ".8g") for value in args.lambdas],
            "--start-pass",
            "100",
            "--end-pass",
            str(100 + args.screen_passes),
            "--output",
            str(ranking_output),
        ]
        ranking_code = run_logged(
            ranking_command, args.run_root / "queue_logs" / "ranking.log", environment
        )
        state["status"] = "complete" if ranking_code == 0 else "completed_candidates_but_ranking_failed"
        state["ranking_exit_code"] = ranking_code
        state["ranking_output"] = str(ranking_output)
        state["finished_unix"] = time.time()
        atomic_json(args.status, state)
        if ranking_code != 0:
            raise SystemExit(ranking_code)
    finally:
        state["vq_restart"] = restart_vq(args.vq_service)
        atomic_json(args.status, state)


if __name__ == "__main__":
    main()
