#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAT_RUN_JSON = ROOT / "data/metadata/chat_literature_last.json"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive terminal wrapper for ask_literature_langgraph.py. "
            "Unknown flags are forwarded to the underlying workflow."
        )
    )
    parser.add_argument(
        "--thread-id",
        default="chat",
        help="Thread id used for lightweight memory across turns.",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        help="Optional memory directory forwarded to the workflow.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=0,
        help="Maximum number of user turns. 0 means unlimited.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Only print the final answer from each workflow run.",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=float(os.environ.get("LITERATURE_CHAT_TURN_TIMEOUT", "900")),
        help="Hard timeout in seconds for each workflow turn. Use 0 to disable.",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Reset this thread memory before the first question.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable lightweight memory load/write.",
    )
    parser.add_argument(
        "--run-json",
        type=Path,
        default=DEFAULT_CHAT_RUN_JSON,
        help="Run record path used in compact mode.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the workflow.",
    )
    args, workflow_args = parser.parse_known_args()
    return args, workflow_args


def build_command(
    *,
    args: argparse.Namespace,
    workflow_args: list[str],
    question: str,
    reset_memory: bool,
) -> list[str]:
    command = [
        args.python,
        str(ROOT / "scripts/ask_literature_langgraph.py"),
        question,
        "--thread-id",
        args.thread_id,
    ]
    if args.memory_dir:
        command.extend(["--memory-dir", str(args.memory_dir)])
    if args.no_memory:
        command.append("--no-memory")
    if reset_memory:
        command.append("--reset-memory")
    if args.compact and "--run-json" not in workflow_args:
        command.extend(["--run-json", str(args.run_json)])
    command.extend(workflow_args)
    return command


def print_compact_answer(run_json: Path, completed: subprocess.CompletedProcess[str]) -> None:
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        print(f"[chat] workflow failed with exit code {completed.returncode}")
        return
    try:
        state = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(completed.stdout)


def _run_turn(command: list[str], *, timeout: float | None, compact: bool) -> subprocess.CompletedProcess[str] | None:
    try:
        if compact:
            return subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=None,
                check=False,
                timeout=timeout,
            )
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        limit = f"{timeout:.0f}s" if timeout else "configured timeout"
        print(f"[chat] workflow timed out after {limit}; killed the child process.")
        return None
        print(f"[chat] could not read compact run record: {exc}")
        return
    answer = str(state.get("final_answer") or "").strip()
    if answer:
        print(answer)
    else:
        print(completed.stdout)


def main() -> None:
    args, workflow_args = parse_args()
    print("Literature Chat REPL")
    print("输入问题开始；输入 :q / :quit / exit 退出；输入 :reset 让下一轮清空 thread memory。")
    print(f"thread_id={args.thread_id} compact={args.compact}")
    print()

    turn = 0
    reset_next = bool(args.reset_memory)
    while True:
        if args.turn_limit and turn >= args.turn_limit:
            print("[chat] turn limit reached.")
            return
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {":q", ":quit", "quit", "exit"}:
            return
        if question.lower() == ":reset":
            reset_next = True
            print("[chat] 下一轮会使用 --reset-memory。")
            continue
        if question.lower() == ":help":
            print("命令：:q 退出，:reset 下一轮清空 memory。其他输入都会作为问题运行。")
            continue

        turn += 1
        command = build_command(
            args=args,
            workflow_args=workflow_args,
            question=question,
            reset_memory=reset_next,
        )
        reset_next = False
        timeout = args.turn_timeout if args.turn_timeout and args.turn_timeout > 0 else None
        print(f"\n[chat] turn {turn} running...\n")
        if args.compact:
            completed = _run_turn(command, timeout=timeout, compact=True)
            if completed is not None:
                print_compact_answer(args.run_json, completed)
            print()
        else:
            _run_turn(command, timeout=timeout, compact=False)
            print()


if __name__ == "__main__":
    main()
