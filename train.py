"""Main BRIDGE-AD training workflow for the staged code release.

The internal bridge_ad package and complete training plans are not included
in this partial release. They will be released upon acceptance of the paper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path


TEXTURE_MODULE = "bridge_ad.runtime.train_texture"
BASE_MODULE = "bridge_ad.runtime.train_base"
BIPOLAR_MODULE = "bridge_ad.runtime.train_stage"

WORKFLOW = """BRIDGE-AD training workflow
  Normal-appearance branch preparation
  Base training: HP, then HN
  Bipolar training: paired HP/HN jobs for each configured round

Each bipolar pair uses the same fixed parent snapshot. The internal trainers
handle dataset preparation, model construction, losses, and checkpointing.
The internal package and complete training plans are not released yet.
"""


def argument_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a nonempty list of command-line argument strings")
    return value


def training_jobs(plan: dict) -> list[tuple[str, str, list[str]]]:
    """Resolve the stage order without loading any training dependencies."""
    jobs = [
        ("normal-appearance", TEXTURE_MODULE, argument_list(plan.get("texture"), "texture")),
        ("base-hp", BASE_MODULE, argument_list(plan.get("base_hp"), "base_hp")),
        ("base-hn", BASE_MODULE, argument_list(plan.get("base_hn"), "base_hn")),
    ]
    rounds = plan.get("bipolar_rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("bipolar_rounds must contain at least one paired HP/HN configuration")
    for index, pair in enumerate(rounds, start=1):
        if not isinstance(pair, dict):
            raise ValueError(f"Bipolar round {index} must contain hp and hn argument lists")
        for role in ("hp", "hn"):
            name = f"bipolar-{index}-{role}"
            jobs.append((name, BIPOLAR_MODULE, argument_list(pair.get(role), name)))
    return jobs


def check_dependencies(jobs: list[tuple[str, str, list[str]]]) -> None:
    for module in dict.fromkeys(module for _, module, _ in jobs):
        try:
            found = importlib.util.find_spec(module)
        except (ImportError, ValueError) as error:
            raise RuntimeError(
                "The internal BRIDGE-AD training package is unavailable. "
                "Its release is scheduled for after paper acceptance."
            ) from error
        if found is None:
            raise RuntimeError(f"Missing internal training module: {module}")


def run_training(jobs: list[tuple[str, str, list[str]]], dry_run: bool = False) -> None:
    if not dry_run:
        check_dependencies(jobs)
    for name, module, arguments in jobs:
        command = [sys.executable, "-m", module, *arguments]
        print(f"[{name}]", flush=True)
        # Commands are displayed for inspection and executed without a shell.
        display = subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)
        print(display, flush=True)
        if not dry_run:
            subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Complete internal training plan (JSON)")
    parser.add_argument("--dry-run", action="store_true", help="Print stage commands without training")
    parser.add_argument("--show-workflow", action="store_true", help="Show the stage order without dependencies")
    args = parser.parse_args()
    if args.show_workflow:
        print(WORKFLOW)
        return 0
    if args.config is None:
        parser.error("--config is required unless --show-workflow is used")
    try:
        plan = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError("The training plan must be a JSON object")
        run_training(training_jobs(plan), dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        print(f"Training stopped: a stage exited with code {error.returncode}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
